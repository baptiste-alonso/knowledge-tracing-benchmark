"""
Export out-of-fold predictions for Optimized PFA (LKT Model C: a PFA-style
logistic model with per-student forgetting, trained by gradient descent).

This model ALREADY SAVES its per-fold checkpoints (models/optimized_pfa/folds/
optimized_pfa_fold{0..4}.pth). We reload them, rebuild each fold's test set,
predict with the median per-student decay (cold start for unseen test students),
and write the standard OOF CSV.

Run from the project ROOT:
    python -m analysis.exporters.export_optimized_pfa
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "optimized_pfa"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np
import torch

import benchmark_utils as bu
from optimized_pfa_student_dependant import (
    LKT_ModelC, load_data, filter_rare, balanced_student_kfold_split,
    build_grouped_tensors, predict,
)

K = 5
SEED = 42
MIN_ITEM_OBS = 10
MIN_SKILL_OBS = 20
FOLDS_DIR = ROOT / "models" / "optimized_pfa" / "folds"


def main():
    df = load_data()
    df = filter_rare(df, min_item_obs=MIN_ITEM_OBS, min_skill_obs=MIN_SKILL_OBS)
    folds = balanced_student_kfold_split(df, k=K, seed=SEED)

    first = True
    for fold_idx, (_, test_df) in enumerate(folds):
        ckpt_path = FOLDS_DIR / f"optimized_pfa_fold{fold_idx}.pth"
        if not ckpt_path.exists():
            print(f"  [skip] fold {fold_idx}: checkpoint missing ({ckpt_path.name})")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = LKT_ModelC(
            ckpt["n_students"], ckpt["n_skills"], ckpt["n_items"],
            rasch=ckpt["config"]["rasch"],
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        test_tensors = build_grouped_tensors(test_df)
        # Test students are unseen -> median per-student decay (cold start).
        p = predict(model, test_tensors, use_median_d=True).numpy()

        (seq_student, seq_skill, seq_item, seq_y, seq_r, seq_t,
         seq_mask, orig_idx) = test_tensors
        valid_orig = orig_idx[seq_mask]
        order = torch.argsort(valid_orig)
        y_true = seq_y[seq_mask][order].numpy()
        skill = seq_skill.unsqueeze(1).expand_as(seq_item)[seq_mask][order].numpy()

        bu.save_oof("optimized_pfa", fold_idx, y_true, p, skill=skill,
                    append=not first)
        first = False
        print(f"  fold {fold_idx}: {len(y_true)} predictions exported")

    print(f"OK -> {bu.REGISTRY_BY_KEY['optimized_pfa'].oof_path}")


if __name__ == "__main__":
    main()
