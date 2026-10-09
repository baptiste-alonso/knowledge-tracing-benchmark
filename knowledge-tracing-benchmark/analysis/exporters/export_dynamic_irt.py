"""
Export des predictions out-of-fold de l'IRT dynamique.

Ce modele SAUVEGARDE DEJA ses checkpoints par fold (models/dynamic_irt/folds/
dynamic_irt_fold{0..4}.pth). On les recharge, on reconstruit le set de test de
chaque fold, on predit, et on ecrit le CSV OOF standard.

A lancer depuis la RACINE du projet (les chemins sont relatifs a la racine) :
    python -m analysis.exporters.export_dynamic_irt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "dynamic_irt"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np
import torch

import benchmark_utils as bu
from dynamic_irt import (
    DynamicIRT, load_data, student_kfold_split, build_grouped_tensors, predict,
)

K = 5
SEED = 42
FOLDS_DIR = ROOT / "models" / "dynamic_irt" / "folds"


def main():
    df = load_data()  # lit datas_management/cleaned_events.csv
    folds = student_kfold_split(df, k=K, seed=SEED)
    n_items = int(df["item"].max() + 1)
    n_skills = int(df["skill"].max() + 1)

    first = True
    for fold_idx, (_, test_df) in enumerate(folds):
        ckpt_path = FOLDS_DIR / f"dynamic_irt_fold{fold_idx}.pth"
        if not ckpt_path.exists():
            print(f"  [skip] fold {fold_idx} : checkpoint absent ({ckpt_path.name})")
            continue

        test_tensors = build_grouped_tensors(test_df)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        n_students_test = int(test_df["student"].max() + 1)
        model = DynamicIRT(n_students_test, ckpt["n_skills"], ckpt["n_items"])
        state = ckpt["model_state_dict"]
        state.pop("theta0", None)  # theta0 depend des etudiants train -> cold start a 0
        model.load_state_dict(state, strict=False)
        model.eval()

        # predict() renvoie les probas triees par orig_idx ; on aligne y_true pareil
        p = predict(model, test_tensors).numpy()
        seq_student, seq_skill, seq_item, seq_y, seq_r, seq_mask, orig_idx = test_tensors
        valid_orig = orig_idx[seq_mask]
        order = torch.argsort(valid_orig)
        y_true = seq_y[seq_mask][order].numpy()
        skill = seq_skill.unsqueeze(1).expand_as(seq_item)[seq_mask][order].numpy()

        bu.save_oof("dynamic_irt", fold_idx, y_true, p, skill=skill,
                    append=not first)
        first = False
        print(f"  fold {fold_idx} : {len(y_true)} predictions exportees")

    print(f"OK -> {bu.REGISTRY_BY_KEY['dynamic_irt'].oof_path}")


if __name__ == "__main__":
    main()
