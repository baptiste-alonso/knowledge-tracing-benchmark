"""Classic PFA (Performance Factors Analysis) baseline.

Logistic regression over one-hot question indicators plus the prior
success / failure counts accumulated per (student, macrocompetence).
Evaluated with a balanced student-wise 5-fold cross-validation.

Self-contained: no imports from any other project file.

Run from the repository root:

    python -m models.pfa.pfa
"""

import math
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, mean_squared_error, log_loss
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datas_management"
OUTPUT_DIR = PROJECT_ROOT / "models" / "pfa"


# ---------- DATA ----------
def load_data_for_pfa(path=None):
    """Load the cleaned events.

    Keeps id_macrocompetence in addition to the usual columns, since classic
    PFA groups prior_success / prior_failures by (id_eleve, id_macrocompetence).
    """
    if path is None:
        path = DATA_DIR / "cleaned_events.csv"
    df = pd.read_csv(path)
    needed_cols = ["id_eleve", "id_question", "id_competence",
                   "id_macrocompetence", "bonne_reponse", "temps", "timestamp"]
    df = df[needed_cols].dropna()

    df["student"] = LabelEncoder().fit_transform(df["id_eleve"])
    df["item"]    = LabelEncoder().fit_transform(df["id_question"])
    df["skill"]   = LabelEncoder().fit_transform(df["id_competence"])
    df = df.sort_values(["student", "skill", "timestamp"])

    df["temps"]    = df["temps"].clip(lower=1)
    df["log_time"] = np.log(df["temps"] / df["temps"].median())
    df["t_days"]   = (df["timestamp"] - df["timestamp"].min()) / 86400.0
    return df


def filter_rare(df, min_item_obs=10, min_skill_obs=20):
    """Drop items and skills with too few observations, then re-encode."""
    if min_item_obs > 0:
        keep_items = df["item"].value_counts()
        df = df[df["item"].isin(keep_items[keep_items >= min_item_obs].index)]
    if min_skill_obs > 0:
        keep_skills = df["skill"].value_counts()
        df = df[df["skill"].isin(keep_skills[keep_skills >= min_skill_obs].index)]
    df = df.copy()
    df["item"]  = LabelEncoder().fit_transform(df["item"])
    df["skill"] = LabelEncoder().fit_transform(df["skill"])
    return df.reset_index(drop=True)


def balanced_student_kfold_split(df, k=5, seed=42):
    """Balanced student-wise k-fold split (LPT bin-packing).

    Students are shuffled then assigned greedily to the least-loaded fold
    (longest-processing-time heuristic) so each fold holds roughly the same
    number of observations. Splitting by student keeps the evaluation
    cold-start: no test student is seen during training.
    """
    rng = np.random.default_rng(seed)

    counts   = df.groupby("student").size()
    students = counts.index.to_numpy()
    obs      = counts.to_numpy().astype(float)

    perm = rng.permutation(len(students))
    students, obs = students[perm], obs[perm]
    order = np.argsort(-obs, kind="stable")
    students, obs = students[order], obs[order]

    fold_loads = np.zeros(k)
    student_to_fold = {}
    for s, c in zip(students, obs):
        f = int(np.argmin(fold_loads))
        student_to_fold[s] = f
        fold_loads[f] += c

    members = [
        np.array([s for s, f in student_to_fold.items() if f == i])
        for i in range(k)
    ]

    splits = []
    for fi in range(k):
        test_students  = members[fi]
        train_students = np.concatenate([members[i] for i in range(k) if i != fi])

        tr = df[df["student"].isin(train_students)].copy()
        te = df[df["student"].isin(test_students)].copy()
        tr["student"] = tr["student"].map({s: i for i, s in enumerate(sorted(train_students))})
        te["student"] = te["student"].map({s: i for i, s in enumerate(sorted(test_students))})

        splits.append((tr.reset_index(drop=True), te.reset_index(drop=True)))
    return splits


# ---------- PFA FEATURES ----------
def count_success_failures(df):
    """Cumulative prior successes / failures per (student, macrocompetence).

    The shift(fill_value=0) ensures the count for trial t uses only trials
    0..t-1 (no leakage from the current trial).
    """
    df = df.sort_values(by="timestamp").copy()
    df["prior_success"]  = df.groupby(["id_eleve", "id_macrocompetence"])["bonne_reponse"].transform(
        lambda x: x.cumsum().shift(fill_value=0)
    )
    df["prior_failures"] = df.groupby(["id_eleve", "id_macrocompetence"])["bonne_reponse"].transform(
        lambda x: (x.cumsum() - x).shift(fill_value=0)
    )
    return df


def build_pfa_features(df):
    """PFA dataset = one-hot(id_question) + prior_success + prior_failures."""
    df = df.reset_index(drop=True)
    df = count_success_failures(df)
    df_f = df[["id_question", "prior_success", "prior_failures",
               "bonne_reponse", "timestamp"]]
    df_dummies = pd.get_dummies(df_f, columns=["id_question"])
    df_dummies = df_dummies.sort_values(by="timestamp").reset_index(drop=True)
    return df_dummies


def split_features_target(df):
    X = df.drop(columns=["bonne_reponse", "timestamp"])
    y = df["bonne_reponse"]
    return X, y


# ---------- METRICS ----------
def compute_metrics(model, X_test, y_test):
    """AUC / RMSE / BCE (LogLoss)."""
    y_prob = model.predict_proba(X_test)[:, 1]

    auc      = roc_auc_score(y_test, y_prob)
    rmse     = math.sqrt(mean_squared_error(y_test, y_prob))    # RMSE on probabilities
    logloss  = log_loss(y_test, np.clip(y_prob, 1e-7, 1 - 1e-7))
    return auc, rmse, logloss


# ---------- MAIN ----------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data_for_pfa()
    df = filter_rare(df, min_item_obs=10, min_skill_obs=20)
    print(f"After filtering: {len(df)} obs | "
          f"{df['item'].nunique()} items | "
          f"{df['skill'].nunique()} skills | "
          f"{df['id_macrocompetence'].nunique()} macrocompetences")

    folds = balanced_student_kfold_split(df, k=5, seed=42)

    results = []
    for fi, (train_df, test_df) in enumerate(folds):
        print(f"\n{'='*52}")
        print(f"  Fold {fi + 1} / 5  -  PFA classic")
        print(f"{'='*52}")

        train_pfa = build_pfa_features(train_df)
        test_pfa  = build_pfa_features(test_df)

        X_train, y_train = split_features_target(train_pfa)
        X_test,  y_test  = split_features_target(test_pfa)

        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        print(f"  train: {len(X_train)} obs | {X_train.shape[1]} features")
        print(f"  test : {len(X_test)} obs")

        model = LogisticRegression(max_iter=1000, solver="liblinear")
        model.fit(X_train, y_train)

        auc, rmse, logloss = compute_metrics(model, X_test, y_test)
        print(f"  TEST | AUC {auc:.4f} | RMSE {rmse:.4f} | BCE {logloss:.4f}")

        results.append({
            "fold":     fi + 1,
            "auc":      auc,
            "rmse":     rmse,
            "bce":      logloss,
        })

    res = pd.DataFrame(results)
    print(f"\n\n{'='*52}")
    print("CROSS-VALIDATION SUMMARY (5 FOLDS) - PFA classic")
    print(f"{'='*52}")
    print(f"  AUC      : {res['auc'].mean():.4f} ± {res['auc'].std():.4f}")
    print(f"  RMSE     : {res['rmse'].mean():.4f} ± {res['rmse'].std():.4f}")
    print(f"  BCE      : {res['bce'].mean():.4f} ± {res['bce'].std():.4f}")

    output_path = OUTPUT_DIR / "cv_results.csv"
    res.to_csv(output_path, index=False)
    print(f"\n  Saved -> {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()