"""
benchmark_utils.py
==================

Shared utilities for the *aggregated* analysis of all the project's Knowledge
Tracing models (BKT, IRT, dynamic IRT, PFA, Optimized PFA, GKT, GIRT).

Guiding idea
------------
Each model uses a different split, different data and a different set of metrics.
To compare them on the same plot (ROC, PR, calibration) and in the same table,
everything is reduced to a single common artifact:

    each model's out-of-fold (OOF) PREDICTIONS.

One `oof_predictions.csv` file per model, with a single schema:

    fold    : int    index of the test fold (0..k-1) the prediction comes from
    y_true  : int    actual response (0/1)
    y_pred  : float  predicted probability P(correct) in [0, 1]
    skill   : (optional) competency identifier, for per-skill analyses

From this single file, this module recomputes ALL metrics the same way for every
model (hence comparable), and draws the overlaid curves.

Why predictions and not checkpoints?
- A .pth checkpoint alone is not enough: you still need the data + the model's
  own inference code to produce probabilities.
- A CSV of aggregated metrics (AUC/RMSE) cannot be used to draw a curve.
- Predictions (y_true, y_pred) are the smallest common denominator: every model
  produces them, and they are enough to recompute everything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    accuracy_score,
)

# --------------------------------------------------------------------------- #
#  File locations
# --------------------------------------------------------------------------- #

# benchmark_utils.py lives in <root>/analysis/. The project root is one level
# up. All "models/..." paths are relative to that root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

OOF_FILENAME = "oof_predictions.csv"
OOF_COLUMNS = ["fold", "y_true", "y_pred"]  # skill is optional


# --------------------------------------------------------------------------- #
#  Model registry
# --------------------------------------------------------------------------- #
#  display_name : name shown on plots and tables
#  oof_path     : path of the OOF predictions CSV (relative to MODELS_DIR)
#  color        : fixed color for consistency across figures
#  family       : pedagogical family (useful for grouping / commenting)
# --------------------------------------------------------------------------- #


@dataclass
class ModelSpec:
    key: str
    display_name: str
    oof_subpath: str
    color: str
    family: str

    @property
    def oof_path(self) -> Path:
        return MODELS_DIR / self.oof_subpath


# Order = display order. Colors chosen to stay legible when overlaid.
MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec("bkt",          "BKT",                 "bkt/oof_predictions.csv",                "#1f77b4", "probabilistic"),
    ModelSpec("irt",          "Static IRT",          "irt/oof_predictions.csv",                "#9467bd", "psychometric"),
    ModelSpec("dynamic_irt",  "Dynamic IRT",         "dynamic_irt/oof_predictions.csv",        "#2ca02c", "psychometric (dynamic)"),
    ModelSpec("pfa",          "PFA",                 "pfa/oof_predictions.csv",                "#ff7f0e", "logistic"),
    ModelSpec("optimized_pfa", "Optimized PFA",      "optimized_pfa/oof_predictions.csv",      "#17becf", "logistic (temporal)"),
    ModelSpec("gkt",          "GKT",                 "gkt/oof_predictions.csv",                "#d62728", "graph / neural"),
    ModelSpec("girt",         "GIRT (GKT+IRT)",      "girt/oof_predictions.csv",               "#e377c2", "graph + psychometric"),
]

REGISTRY_BY_KEY = {m.key: m for m in MODEL_REGISTRY}


# --------------------------------------------------------------------------- #
#  Writing / reading OOF predictions
# --------------------------------------------------------------------------- #


def save_oof(model_key: str, fold, y_true, y_pred, skill=None, append=False):
    """Write (or append) a model's OOF predictions in the standard schema.

    Parameters
    ----------
    model_key : key of the model in MODEL_REGISTRY (e.g. "bkt").
    fold      : int (same value for all rows) or array of same length as y_true.
    y_true    : 0/1 array.
    y_pred    : array of probabilities in [0, 1].
    skill     : optional array of competency identifiers.
    append    : if True, concatenate to an existing file (useful to write fold by fold).
    """
    spec = REGISTRY_BY_KEY[model_key]
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)

    if np.ndim(fold) == 0:
        fold = np.full(len(y_true), int(fold))
    fold = np.asarray(fold).astype(int)

    data = {"fold": fold, "y_true": y_true, "y_pred": y_pred}
    if skill is not None:
        data["skill"] = np.asarray(skill)

    df = pd.DataFrame(data)
    path = spec.oof_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def load_oof(model_key: str) -> Optional[pd.DataFrame]:
    """Load a model's OOF predictions, or None if the file is missing."""
    spec = REGISTRY_BY_KEY[model_key]
    if not spec.oof_path.exists():
        return None
    df = pd.read_csv(spec.oof_path)
    missing = [c for c in OOF_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{spec.oof_path}: missing columns {missing}")
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].clip(0.0, 1.0)
    return df


def load_all_oof(keys=None) -> dict[str, pd.DataFrame]:
    """Load the OOF of all available models. Skip (with a message) those whose
    file does not exist yet."""
    if keys is None:
        keys = [m.key for m in MODEL_REGISTRY]
    out = {}
    for k in keys:
        df = load_oof(k)
        if df is None:
            print(f"  [missing] {REGISTRY_BY_KEY[k].display_name:<22} -> "
                  f"{REGISTRY_BY_KEY[k].oof_subpath} (run the export first)")
        else:
            out[k] = df
    return out


# --------------------------------------------------------------------------- #
#  Unified metrics
# --------------------------------------------------------------------------- #


def expected_calibration_error(y_true, y_pred, n_bins=10):
    """Classic ECE (uniform binning over [0, 1])."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi if i < n_bins - 1 else y_pred <= hi)
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_pred[mask].mean()
            ece += abs(acc - conf) * mask.sum() / n
    return ece


def compute_metrics(y_true, y_pred):
    """Metric set common to ALL models.

    AUC      : discrimination power (0.5 = random).
    ACC      : accuracy at the 0.5 threshold.
    RMSE     : root mean squared error on the probabilities.
    Brier    : = RMSE^2 (Brier score), kept for the KT literature.
    LogLoss  : cross-entropy (penalizes confident wrong predictions).
    PR_AUC   : area under precision-recall (relevant for imbalanced classes).
    ECE      : expected calibration error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)

    single_class = len(np.unique(y_true)) < 2
    return {
        "AUC":     roc_auc_score(y_true, y_pred) if not single_class else np.nan,
        "ACC":     accuracy_score(y_true, (y_pred >= 0.5).astype(int)),
        "RMSE":    float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "Brier":   float(np.mean((y_true - y_pred) ** 2)),
        "LogLoss": log_loss(y_true, y_pred, labels=[0, 1]),
        "PR_AUC":  average_precision_score(y_true, y_pred) if not single_class else np.nan,
        "ECE":     expected_calibration_error(y_true, y_pred),
    }


def bootstrap_metric_ci(y_true, y_pred, metric="AUC", n_boot=1000,
                        alpha=0.05, seed=0):
    """Bootstrap confidence interval of a metric for ONE model
    (resampling its own predictions). Since each model has a different test set,
    we do not compare models with a paired test: instead we look at whether their
    CIs overlap.

    Returns (point_estimate, lo, hi).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = compute_metrics(y_true, y_pred)[metric]
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        stats.append(compute_metrics(yt, yp)[metric])
    if not stats:
        return point, np.nan, np.nan
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def per_skill_metric(oof: pd.DataFrame, metric="AUC", min_obs=30):
    """Metric computed per competency (column 'skill'), for one model.
    Skips competencies with fewer than min_obs observations or a single class.
    Returns a DataFrame [skill, n, <metric>]."""
    if "skill" not in oof.columns:
        return None
    rows = []
    for skill, g in oof.groupby("skill"):
        if len(g) < min_obs or g["y_true"].nunique() < 2:
            continue
        m = compute_metrics(g["y_true"].values, g["y_pred"].values)
        rows.append({"skill": skill, "n": len(g), metric: m[metric]})
    return pd.DataFrame(rows)


METRIC_ORDER = ["AUC", "PR_AUC", "ACC", "RMSE", "Brier", "LogLoss", "ECE"]
# True = "higher is better"
METRIC_HIGHER_IS_BETTER = {
    "AUC": True, "PR_AUC": True, "ACC": True,
    "RMSE": False, "Brier": False, "LogLoss": False, "ECE": False,
}


def per_fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    """Recompute the metrics fold by fold (one row per fold)."""
    rows = []
    for fold, g in oof.groupby("fold"):
        m = compute_metrics(g["y_true"].values, g["y_pred"].values)
        m["fold"] = int(fold)
        m["n"] = len(g)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def summarize_model(oof: pd.DataFrame) -> dict:
    """Summarize a model: mean +/- std over folds, plus the 'pooled' values
    (metrics computed on all concatenated predictions)."""
    pf = per_fold_metrics(oof)
    summary = {"n_folds": pf["fold"].nunique(), "n_obs": int(pf["n"].sum())}
    for metric in METRIC_ORDER:
        vals = pf[metric].values
        summary[f"{metric}_mean"] = float(np.nanmean(vals))
        summary[f"{metric}_std"] = float(np.nanstd(vals))
    pooled = compute_metrics(oof["y_true"].values, oof["y_pred"].values)
    for metric in METRIC_ORDER:
        summary[f"{metric}_pooled"] = pooled[metric]
    return summary


def build_comparison_table(oof_dict: dict[str, pd.DataFrame],
                           kind="mean_std") -> pd.DataFrame:
    """Comparison table of all models x all metrics.

    kind = "mean_std" : "0.78 +/- 0.01" (mean over folds).
    kind = "pooled"   : metric on concatenated predictions.
    kind = "raw"      : raw numeric columns (means), for sorting/plots.
    """
    rows = []
    for key, oof in oof_dict.items():
        spec = REGISTRY_BY_KEY[key]
        s = summarize_model(oof)
        row = {"model": spec.display_name, "family": spec.family,
               "n_folds": s["n_folds"], "n_obs": s["n_obs"]}
        for metric in METRIC_ORDER:
            if kind == "mean_std":
                row[metric] = f"{s[f'{metric}_mean']:.4f} ± {s[f'{metric}_std']:.4f}"
            elif kind == "pooled":
                row[metric] = round(s[f"{metric}_pooled"], 4)
            else:  # raw
                row[metric] = s[f"{metric}_mean"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    return table


# --------------------------------------------------------------------------- #
#  Curves (computation): ROC, PR, calibration
# --------------------------------------------------------------------------- #


def roc_points(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return fpr, tpr


def pr_points(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return recall, precision


def calibration_points(y_true, y_pred, n_bins=10):
    """Return (mean confidence, empirical frequency, weight) per bin."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    conf, freq, weight = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi if i < n_bins - 1 else y_pred <= hi)
        if mask.sum() > 0:
            conf.append(y_pred[mask].mean())
            freq.append(y_true[mask].mean())
            weight.append(mask.sum())
    return np.array(conf), np.array(freq), np.array(weight)


# --------------------------------------------------------------------------- #
#  Curves (overlaid plot)
# --------------------------------------------------------------------------- #


def _ensure_ax(ax, figsize=(7, 6)):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_roc_all(oof_dict, ax=None, pooled=True):
    """Overlaid ROC curves. pooled=True: one curve per model on concatenated
    predictions (legible). pooled=False: mean of the per-fold curves."""
    ax = _ensure_ax(ax)
    for key, oof in oof_dict.items():
        spec = REGISTRY_BY_KEY[key]
        auc = compute_metrics(oof["y_true"], oof["y_pred"])["AUC"]
        fpr, tpr = roc_points(oof["y_true"].values, oof["y_pred"].values)
        ax.plot(fpr, tpr, color=spec.color, lw=2,
                label=f"{spec.display_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Random (0.5)")
    ax.set_xlabel("False positive rate (FPR)")
    ax.set_ylabel("True positive rate (TPR)")
    ax.set_title("ROC curves — all models")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    return ax


def plot_pr_all(oof_dict, ax=None):
    """Overlaid precision-recall curves + baseline (positive rate)."""
    ax = _ensure_ax(ax)
    base = None
    for key, oof in oof_dict.items():
        spec = REGISTRY_BY_KEY[key]
        ap = compute_metrics(oof["y_true"], oof["y_pred"])["PR_AUC"]
        recall, precision = pr_points(oof["y_true"].values, oof["y_pred"].values)
        ax.plot(recall, precision, color=spec.color, lw=2,
                label=f"{spec.display_name} (PR-AUC={ap:.3f})")
        base = oof["y_true"].mean()
    if base is not None:
        ax.axhline(base, ls="--", color="grey", lw=1,
                   label=f"Base (positive rate={base:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves — all models")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    return ax


def plot_calibration_all(oof_dict, ax=None, n_bins=10):
    """Overlaid reliability diagrams."""
    ax = _ensure_ax(ax)
    for key, oof in oof_dict.items():
        spec = REGISTRY_BY_KEY[key]
        ece = compute_metrics(oof["y_true"], oof["y_pred"])["ECE"]
        conf, freq, _ = calibration_points(oof["y_true"].values,
                                           oof["y_pred"].values, n_bins)
        ax.plot(conf, freq, marker="o", ms=4, color=spec.color, lw=1.5,
                label=f"{spec.display_name} (ECE={ece:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Perfect calibration")
    ax.set_xlabel("Predicted probability (confidence)")
    ax.set_ylabel("Empirical success frequency")
    ax.set_title("Calibration — all models")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return ax


def plot_metric_bars(oof_dict, metric="AUC", ax=None):
    """Sorted bars of a metric with an error bar (per-fold std)."""
    ax = _ensure_ax(ax, figsize=(8, 5))
    rows = []
    for key, oof in oof_dict.items():
        pf = per_fold_metrics(oof)
        rows.append((REGISTRY_BY_KEY[key].display_name,
                     REGISTRY_BY_KEY[key].color,
                     np.nanmean(pf[metric].values),
                     np.nanstd(pf[metric].values)))
    higher = METRIC_HIGHER_IS_BETTER[metric]
    rows.sort(key=lambda r: r[2], reverse=higher)
    names = [r[0] for r in rows]
    means = [r[2] for r in rows]
    stds = [r[3] for r in rows]
    colors = [r[1] for r in rows]
    ax.bar(names, means, yerr=stds, color=colors, capsize=4, edgecolor="white")
    direction = "↑ better" if higher else "↓ better"
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per model ({direction}) — mean ± std over 5 folds")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    return ax


def plot_fold_dispersion(oof_dict, metric="AUC", ax=None):
    """Scatter per fold to visualize inter-fold variance."""
    ax = _ensure_ax(ax, figsize=(8, 5))
    for i, (key, oof) in enumerate(oof_dict.items()):
        spec = REGISTRY_BY_KEY[key]
        pf = per_fold_metrics(oof)
        x = np.full(len(pf), i) + np.random.uniform(-0.08, 0.08, len(pf))
        ax.scatter(x, pf[metric].values, color=spec.color, s=40, zorder=3)
        ax.scatter([i], [np.nanmean(pf[metric].values)], color="black",
                   marker="_", s=400, zorder=2)
    ax.set_xticks(range(len(oof_dict)))
    ax.set_xticklabels([REGISTRY_BY_KEY[k].display_name for k in oof_dict],
                       rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Dispersion of {metric} across folds (— = mean)")
    ax.grid(axis="y", alpha=0.3)
    return ax