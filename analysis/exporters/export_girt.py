"""
Export des predictions out-of-fold du modele GIRT (composite GKT+IRT).

GIRT SAUVEGARDE DEJA ses checkpoints par fold (models/girt/folds/
girt_fold{0..4}.pth). On reproduit exactement le bloc de prediction de
models/girt/analysis/analyze.py, puis on ecrit le CSV OOF standard.

A lancer depuis la RACINE du projet :
    python -m analysis.exporters.export_girt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "girt"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np
import torch

import benchmark_utils as bu
from girt import GIRT, _predict_one_batch          # noqa: E402
from load_data import (                            # noqa: E402
    load_data, student_kfold_split, build_student_sequences,
)

K = 5
SEED = 42
FOLDS_DIR = ROOT / "models" / "girt" / "folds"
BATCH = 64


def main():
    df = load_data()
    folds = student_kfold_split(df, k=K, seed=SEED)

    first = True
    for fold_idx, (_, test_df) in enumerate(folds):
        ckpt_path = FOLDS_DIR / f"girt_fold{fold_idx}.pth"
        if not ckpt_path.exists():
            print(f"  [skip] fold {fold_idx} : checkpoint absent ({ckpt_path.name})")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = GIRT(n_skills=ckpt["n_skills"], n_items=ckpt["n_items"], d=ckpt["d"])
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        test_tensors = build_student_sequences(test_df)
        seq_item, seq_skill, seq_y, _, seq_mask, orig_idx = test_tensors
        n_test_students = seq_item.shape[0]

        valid_orig = orig_idx[seq_mask]
        order = torch.argsort(valid_orig)
        y_true = seq_y[seq_mask][order].numpy()
        skill_flat = seq_skill[seq_mask][order].numpy()

        with torch.no_grad():
            A_cv = model.graph_construction()
        p_raw, p_orig = [], []
        for bs in range(0, n_test_students, BATCH):
            be = min(bs + BATCH, n_test_students)
            si = seq_item[bs:be]; sk = seq_skill[bs:be]
            sy = seq_y[bs:be]; sm = seq_mask[bs:be]; so = orig_idx[bs:be]
            T_b = int(sm.sum(dim=1).max().item())
            if T_b == 0:
                continue
            preds, origs = _predict_one_batch(
                model, si[:, :T_b], sk[:, :T_b], sy[:, :T_b],
                sm[:, :T_b], so[:, :T_b], A_cv,
            )
            p_raw.append(preds); p_orig.append(origs)
        p_all = torch.cat(p_raw)
        o_all = torch.cat(p_orig)
        p = p_all[torch.argsort(o_all)].cpu().detach().numpy()

        if len(p) != len(y_true):
            print(f"  [warn] fold {fold_idx} : len(p)={len(p)} != len(y_true)={len(y_true)} — saute")
            continue

        bu.save_oof("girt", fold_idx, y_true, p, skill=skill_flat, append=not first)
        first = False
        print(f"  fold {fold_idx} : {len(y_true)} predictions exportees")

    print(f"OK -> {bu.REGISTRY_BY_KEY['girt'].oof_path}")


if __name__ == "__main__":
    main()
