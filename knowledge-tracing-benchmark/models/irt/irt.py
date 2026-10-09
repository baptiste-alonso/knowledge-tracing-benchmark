"""Static IRT baseline (1PL / 2PL) trained per competence with py-irt.

Run from the repository root:

    python -m models.irt.irt
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np

from py_irt.io import write_jsonlines
from py_irt.dataset import Dataset
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    root_mean_squared_error,
)


# =====================================================================
#  Models
# ---------------------------------------------------------------------
#  py-irt parametrises the item characteristic curves (ICC) as:
#    1PL : P = sigma(theta - b)              -> difficulty only (Rasch)
#    2PL : P = sigma(a * (theta - b))        -> + discrimination a
#  3PL is NOT implemented in py-irt (advertised but absent) and would be
#  identical to 2PL here, so we do not use it. 4PL exists with a different
#  parametrisation; the function is kept in case it is needed.
# =====================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def one_pl(theta, a, b, lambdas):
    return sigmoid(theta - b)


def two_pl(theta, a, b, lambdas):
    return sigmoid(a * (theta - b))


def four_pl(theta, a, b, lambdas):
    return (1 - lambdas) + lambdas * sigmoid(a * (theta - b))


MODELS = {1: one_pl, 2: two_pl, 4: four_pl}
MODEL_LABELS = {1: "1PL", 2: "2PL", 4: "4PL"}


# =====================================================================
#  Data preparation
# =====================================================================

# Project root: irt.py lives in models/irt/, so go up two levels.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datas_management"
SKILLS_DIR = PROJECT_ROOT / "models" / "irt" / "skills_json"


def load_data(csv_path=None):
    if csv_path is None:
        csv_path = DATA_DIR / "cleaned_events.csv"
    df = pd.read_csv(csv_path)
    df = df[[
        "id_eleve",
        "id_question",
        "id_competence",
        "bonne_reponse",
    ]].dropna()

    df["id_eleve"] = df["id_eleve"].astype(str)
    df["id_question"] = df["id_question"].astype(str)
    df["bonne_reponse"] = df["bonne_reponse"].astype(int)
    return df


def robust_split(df, test_size=0.2, seed=0):
    """Train/test split guaranteeing that every student and question in the
    test set appears at least once in the train set (otherwise cold-start:
    no estimated parameter -> impossible to evaluate). Orphan rows are moved
    back into the train set."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed
    )

    students = set(train_df["id_eleve"])
    questions = set(train_df["id_question"])

    keep_mask = test_df["id_eleve"].isin(students) & test_df["id_question"].isin(questions)

    add_train = test_df[~keep_mask]
    test_df = test_df[keep_mask]
    train_df = pd.concat([train_df, add_train], ignore_index=True)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_dataset(df, path, test_size=0.2, seed=0):
    """For each competence: write the train set as py-irt jsonlines and keep
    the test set (DataFrame). Returns {competence: test_df}."""
    os.makedirs(path, exist_ok=True)
    testing_dataset = {}

    for skill in df["id_competence"].unique():
        df_skill = df[df["id_competence"] == skill].drop(columns=["id_competence"])

        train_df, test_df = robust_split(df_skill, test_size=test_size, seed=seed)

        data = []
        for student, lines in train_df.groupby("id_eleve"):
            responses = dict(zip(lines["id_question"], lines["bonne_reponse"]))
            data.append({"subject_id": student, "responses": responses})

        write_jsonlines(f"{path}/{skill}.jsonlines", data)
        testing_dataset[skill] = test_df

    return testing_dataset


# =====================================================================
#  Training & prediction
# =====================================================================

def train_by_skill(skill, model=1, epochs=1000, dropout=0.0, path="skills_json"):
    config = IrtConfig(model_type=f"{model}pl", log_every=max(epochs // 10, 1), dropout=dropout)
    dataset = Dataset.from_jsonlines(f"{path}/{skill}.jsonlines")
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)
    trainer.train(epochs=epochs, device="cpu")
    return trainer


def predict_skill(trainer, test_df, model=1):
    """Return (y_true, y_pred) for one competence. Cleanly skips any
    student/question pair absent from the learned parameters."""
    params = trainer.last_params
    fn = MODELS[model]

    student_index = {v: k for k, v in params["subject_ids"].items()}
    question_index = {v: k for k, v in params["item_ids"].items()}

    y_true, y_pred = [], []
    for row in test_df.itertuples(index=False):
        si = student_index.get(row.id_eleve)
        qi = question_index.get(row.id_question)
        if si is None or qi is None:
            continue  # residual cold-start: skip

        theta = params["ability"][si]
        b = params["diff"][qi]
        a = params["disc"][qi] if model >= 2 else None
        lambdas = params["lambdas"][qi] if model >= 4 else None

        y_true.append(row.bonne_reponse)
        y_pred.append(float(fn(theta, a, b, lambdas)))

    return y_true, y_pred


# =====================================================================
#  Evaluation: per competence + aggregation
# =====================================================================

def safe_metrics(y_true, y_pred):
    """Compute log_loss / AUC / RMSE, handling degenerate cases
    (single class -> AUC undefined)."""
    yt = np.asarray(y_true)
    yp = np.clip(np.asarray(y_pred), 1e-7, 1 - 1e-7)

    out = {"n": len(yt)}
    out["log_loss"] = log_loss(yt, yp, labels=[0, 1]) if len(yt) else np.nan
    out["rmse"] = root_mean_squared_error(yt, yp) if len(yt) else np.nan
    out["auc"] = roc_auc_score(yt, yp) if len(np.unique(yt)) == 2 else np.nan
    return out


def evaluate(testing_dataset, models=(1, 2), epochs=1000, dropout=0.0,
             skill_set=None, path="skills_json"):
    """Return a results dict:
       results[model]["per_skill"][skill] = {log_loss, auc, rmse, n}
       results[model]["macro"]            = unweighted mean over competences
       results[model]["pooled"]           = metrics over all pairs at once
    """
    if skill_set is None:
        skill_set = list(testing_dataset.keys())

    results = {}
    for model in models:
        per_skill = {}
        pooled_true, pooled_pred = [], []

        for skill in skill_set:
            trainer = train_by_skill(skill, model=model, epochs=epochs,
                                     dropout=dropout, path=path)
            y_true, y_pred = predict_skill(trainer, testing_dataset[skill], model=model)

            if len(y_true) == 0:
                continue

            per_skill[skill] = safe_metrics(y_true, y_pred)
            pooled_true.extend(y_true)
            pooled_pred.extend(y_pred)

        macro = {
            m: np.nanmean([s[m] for s in per_skill.values()])
            for m in ("log_loss", "auc", "rmse")
        }
        results[model] = {
            "per_skill": per_skill,
            "macro": macro,
            "pooled": safe_metrics(pooled_true, pooled_pred),
        }

    return results


# =====================================================================
#  Console display
# ---------------------------------------------------------------------
#  One table per competence with 1PL and 2PL side by side, then the
#  aggregates (macro mean over competences + global pooled).
#  Conventions: higher AUC = better; lower log-loss / RMSE = better.
# =====================================================================

def _fmt(x):
    return "  --  " if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:7.4f}"


def print_results(results):
    models = sorted(results.keys())
    skills = sorted({s for r in results.values() for s in r["per_skill"]})
    metrics = ["log_loss", "auc", "rmse"]

    # Column widths
    skill_w = max([len("competence")] + [len(s) for s in skills]) + 2
    block = 3 * 8  # 3 metrics * 8 characters
    total_w = skill_w + 6 + len(models) * (block + 2)

    # Header: model names
    print("=" * total_w)
    print("  IRT RESULTS  (higher AUC = better ; lower log-loss / RMSE = better)")
    print("=" * total_w)
    head = f"{'competence':<{skill_w}}{'n':>6}"
    for m in models:
        head += "  " + f"{MODEL_LABELS[m]:^{block}}"
    print(head)
    sub = f"{'':<{skill_w}}{'':>6}"
    for _ in models:
        sub += "  " + f"{'log_loss':>8}{'auc':>8}{'rmse':>8}"
    print(sub)
    print("-" * total_w)

    # One row per competence
    for skill in skills:
        # n is identical across models (same test set), take the first available
        n = next((results[m]["per_skill"][skill]["n"]
                  for m in models if skill in results[m]["per_skill"]), 0)
        row = f"{skill:<{skill_w}}{n:>6}"
        for m in models:
            s = results[m]["per_skill"].get(skill)
            if s is None:
                row += "  " + f"{'--':>8}{'--':>8}{'--':>8}"
            else:
                row += "  " + f"{_fmt(s['log_loss']):>8}{_fmt(s['auc']):>8}{_fmt(s['rmse']):>8}"
        print(row)

    print("-" * total_w)

    # Aggregates: macro and pooled
    for agg_key, agg_name in [("macro", "MEAN (macro)"), ("pooled", "GLOBAL (pooled)")]:
        if agg_key == "pooled":
            n = next((results[m]["pooled"]["n"] for m in models), 0)
            row = f"{agg_name:<{skill_w}}{n:>6}"
        else:
            row = f"{agg_name:<{skill_w}}{'':>6}"
        for m in models:
            a = results[m][agg_key]
            row += "  " + f"{_fmt(a['log_loss']):>8}{_fmt(a['auc']):>8}{_fmt(a['rmse']):>8}"
        print(row)

    print("=" * total_w)


# =====================================================================
#  Pipeline
# =====================================================================

def main():
    EPOCHS = 1000
    DROPOUT = 0.2
    TEST_SIZE = 0.2

    df = load_data()
    testing_dataset = split_dataset(df, str(SKILLS_DIR), test_size=TEST_SIZE)
    skill_set = list(df["id_competence"].unique())

    results = evaluate(
        testing_dataset,
        models=(1, 2),
        epochs=EPOCHS,
        dropout=DROPOUT,
        skill_set=skill_set,
        path=str(SKILLS_DIR),
    )

    print_results(results)


if __name__ == "__main__":
    main()