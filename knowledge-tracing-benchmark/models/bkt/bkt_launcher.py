"""Run the BKT student-wise k-fold cross-validation.

Run from the repository root:

    python models/bkt/bkt_launcher.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Allow "from bkt_model import ..." regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bkt_model import BKTVectorized


def get_default_parameters(df, skill_id_col: str) -> dict:
    initial_parameters = {}

    # Classical default values
    p_0 = 0.3
    p_T = 0.15
    p_G = 0.1
    p_S = 0.05

    # Create parameters for each skill
    for skill in df[skill_id_col].unique():
        initial_parameters[skill] = np.array([p_0, p_T, p_G, p_S])

    return initial_parameters


def create_student_folds_per_skill(df, mapping, n_splits=5):
    """Create n_splits student-wise folds per skill for cross-validation."""
    skill_col = mapping["skill_id"]
    student_col = mapping["student_id"]

    # Store the fold index for each row
    fold_assignments = np.zeros(len(df), dtype=int)

    for skill in df[skill_col].unique():
        df_skill = df[df[skill_col] == skill]

        # Count the number of questions for each student
        student_counts = (
            df_skill.groupby(student_col).size().sort_values(ascending=False)
        )

        fold_loads = [0] * n_splits
        student_to_fold = {}

        for student, count in student_counts.items():
            # find the fold with the fewest observations
            fold_idx = np.argmin(fold_loads)

            student_to_fold[student] = fold_idx
            fold_loads[fold_idx] += count

        # assign folds to rows
        grouped = df_skill.groupby(student_col).groups
        for student, fold_idx in student_to_fold.items():
            idx = grouped[student]
            fold_assignments[idx] = fold_idx

    return fold_assignments


def cross_validate_bkt(df, mapping, n_splits=5):

    fold_assignments = create_student_folds_per_skill(df, mapping, n_splits)

    results = []

    for fold in range(n_splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*60}")

        test_idx = np.where(fold_assignments == fold)[0]
        train_idx = np.where(fold_assignments != fold)[0]

        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        print(f"Train: {len(df_train)} obs")
        print(f"Test:  {len(df_test)} obs")

        print("\n Training...")
        initial_parameters = get_default_parameters(df_train, mapping["skill_id"])
        bkt = BKTVectorized(df_train, df_test, mapping, initial_parameters)
        bkt.em_algorithm(iterations=100)

        # Evaluation
        print(" Evaluating...")
        metrics = bkt.evaluate()

        print(f"\n Fold {fold + 1} results:")
        print(f"   AUC:  {metrics['auc']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   LOG_V:  {metrics['log_v']:.4f}")

        results.append(
            {
                "fold": fold + 1,
                "auc": metrics["auc"],
                "rmse": metrics["rmse"],
                "log_v": metrics["log_v"],
                "n_train": len(df_train),
                "n_test": len(df_test),
            }
        )

    return results


def print_cv_summary(results):
    """Print the cross-validation summary."""

    results_df = pd.DataFrame(results)

    print(f"\n\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY (5 FOLDS)")
    print(f"{'='*60}")

    print(f"\n AUC:")
    print(f"   Mean:   {results_df['auc'].mean():.4f}")
    print(f"   Std:    {results_df['auc'].std():.4f}")
    print(f"   Min:    {results_df['auc'].min():.4f}")
    print(f"   Max:    {results_df['auc'].max():.4f}")

    print(f"\n RMSE:")
    print(f"   Mean:   {results_df['rmse'].mean():.4f}")
    print(f"   Std:    {results_df['rmse'].std():.4f}")
    print(f"   Min:    {results_df['rmse'].min():.4f}")
    print(f"   Max:    {results_df['rmse'].max():.4f}")

    print(f"\n Per-fold details:")
    print(f"{'='*60}\n")


def main():
    print(" Loading data...")
    df = pd.read_csv(PROJECT_ROOT / "datas_management" / "cleaned_events.csv")

    print(f"   Loaded: {len(df)} observations, {df['id_competence'].nunique()} skills")

    mapping = {
        "student_id": "id_eleve",
        "timestamp": "timestamp",
        "skill_id": "id_competence",
        "result": "bonne_reponse",
    }
    df = df.sort_values(
        by=[mapping["student_id"], mapping["skill_id"], mapping["timestamp"]]
    )
    df = df.reset_index(drop=True)

    print("\n Running cross-validation (5 folds)...\n")
    results = cross_validate_bkt(df, mapping, n_splits=5)

    print_cv_summary(results)

    output_path = PROJECT_ROOT / "models" / "bkt" / "cv_results2.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f" Results saved to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()