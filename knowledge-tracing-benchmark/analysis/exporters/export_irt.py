"""
Export out-of-fold predictions for static IRT (py-irt).

WARNING - evaluation regime different from sequential KT models:
py-irt learns an ability PER STUDENT. A student never seen at training has no
parameter -> impossible to predict. So student-wise CV (disjoint test students)
is NOT APPLICABLE to py-irt as is.

We therefore keep the model's native regime (robust_split per skill: random
split where every test student/question was seen in train = warm start), and
repeat it REPEATS times with different seeds to get a variance estimate. These
"folds" are thus repeated random splits, NOT a student-wise CV. Keep this in
mind when comparing.

We use the 2PL (better than the 1PL according to the analyses).
Also saves the py-irt parameters per fold/skill (pickle).

Run from the project ROOT:
    python -m analysis.exporters.export_irt
"""

import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "irt"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np

import benchmark_utils as bu
from irt import (
    load_data, split_dataset, train_by_skill, predict_skill,
)

REPEATS = 5          # number of repeated splits (set to 1 for a quick run)
MODEL = 2            # 2PL
EPOCHS = 1000
DROPOUT = 0.2
TEST_SIZE = 0.2
FOLDS_DIR = ROOT / "models" / "irt" / "folds"
TMP_DIR = ROOT / "models" / "irt" / "skills_json_cv"


def main():
    df = load_data()
    skills = list(df["id_competence"].unique())

    first = True
    for fold in range(REPEATS):
        seed = 42 + fold
        path = str(TMP_DIR / f"fold{fold}")
        testing_dataset = split_dataset(df, path, test_size=TEST_SIZE, seed=seed)

        y_true_all, y_pred_all, skill_all = [], [], []
        fold_params = {}
        for skill in skills:
            if skill not in testing_dataset:
                continue
            trainer = train_by_skill(skill, model=MODEL, epochs=EPOCHS,
                                     dropout=DROPOUT, path=path)
            yt, yp = predict_skill(trainer, testing_dataset[skill], model=MODEL)
            if len(yt) == 0:
                continue
            y_true_all.extend(yt)
            y_pred_all.extend(yp)
            skill_all.extend([skill] * len(yt))
            fold_params[skill] = trainer.last_params

        bu.save_oof("irt", fold, np.array(y_true_all), np.array(y_pred_all),
                    skill=np.array(skill_all), append=not first)
        first = False

        FOLDS_DIR.mkdir(parents=True, exist_ok=True)
        with open(FOLDS_DIR / f"irt_fold{fold}.pkl", "wb") as f:
            pickle.dump(fold_params, f)
        print(f"  fold {fold} (seed {seed}): {len(y_true_all)} predictions exported")

    print(f"OK -> {bu.REGISTRY_BY_KEY['irt'].oof_path}")


if __name__ == "__main__":
    main()