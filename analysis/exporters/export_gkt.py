"""
Export out-of-fold predictions for GKT (based on pykt-toolkit).

GKT currently only saves fold 0 (models/gkt/saved_models/
gkt_fold0_transition_hd64.ckpt). This script reproduces the logic of the
gkt_analysis.ipynb notebook for EACH fold: it builds the transition graph from
the fold's train set, loads the checkpoint if it exists, predicts on the test
set, and writes the standard OOF CSV.

Requirements:
  - pykt-toolkit importable (present in models/gkt/pykt-toolkit);
  - torch installed;
  - one checkpoint per fold in models/gkt/saved_models/.
Folds without a checkpoint are SKIPPED (train them first; the infrastructure
already exists: split_data.py + pykt training).

Run from the project ROOT:
    python -m analysis.exporters.export_gkt
"""

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GKT_DIR = ROOT / "models" / "gkt"
PYKT_DIR = GKT_DIR / "pykt-toolkit"
DATA_DIR = GKT_DIR / "data"
SAVED_DIR = GKT_DIR / "saved_models"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(PYKT_DIR))

import numpy as np
import pandas as pd
import torch

import benchmark_utils as bu

# Hyperparameters: must match the training (cf. notebook cell 2)
HIDDEN_DIM = 64
EMB_SIZE = 64
GRAPH_TYPE = "transition"
MAX_SEQ_LEN = 200
K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_pykt_module(name, relative_path):
    filepath = PYKT_DIR / "pykt" / relative_path
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_sequences(df, skill2idx, max_len=MAX_SEQ_LEN):
    """(skill, response) sequences padded with -1. Same as the notebook."""
    seqs_c, seqs_r = [], []
    for _, group in df.sort_values("timestamp").groupby("uid"):
        c = group["skill"].map(skill2idx).values
        r = group["response"].values
        if len(c) < 2:
            continue
        c, r = c[:max_len], r[:max_len]
        pad = max_len - len(c)
        c = np.concatenate([c, np.full(pad, -1, dtype=np.int64)])
        r = np.concatenate([r, np.full(pad, -1, dtype=np.int64)])
        seqs_c.append(c)
        seqs_r.append(r)
    return np.array(seqs_c, dtype=np.int64), np.array(seqs_r, dtype=np.int64)


def predict_on_sequences(model, c_arr, r_arr, batch_size=64):
    """Forward + collect (preds, targets, skills) on valid steps."""
    model.eval()
    preds_all, targ_all, skill_all = [], [], []
    n = len(c_arr)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            c_b = torch.tensor(c_arr[start:start + batch_size], dtype=torch.long).to(DEVICE)
            r_b = torch.tensor(r_arr[start:start + batch_size], dtype=torch.long).to(DEVICE)
            preds = model(c_b, r_b)                  # [B, L-1]
            shft_c = c_b[:, 1:]
            shft_r = r_b[:, 1:]
            mask = (shft_c != -1) & (shft_r != -1)
            preds_all.append(preds[mask].cpu().numpy())
            targ_all.append(shft_r[mask].float().cpu().numpy())
            skill_all.append(shft_c[mask].cpu().numpy())
    return (np.concatenate(preds_all),
            np.concatenate(targ_all),
            np.concatenate(skill_all))


def main():
    # Skill universe shared across all folds (consistent indices)
    all_skills = set()
    fold_files = {}
    for fold in range(K):
        paths = {s: DATA_DIR / f"{s}_fold_{fold}.csv" for s in ("train", "valid", "test")}
        if not all(p.exists() for p in paths.values()):
            continue
        fold_files[fold] = {s: pd.read_csv(p) for s, p in paths.items()}
        for s in ("train", "valid", "test"):
            all_skills.update(fold_files[fold][s]["skill"].unique())
    if not fold_files:
        print("No fold file found in models/gkt/data/. Run split_data.py.")
        return

    skill2idx = {s: i for i, s in enumerate(sorted(all_skills))}
    NUM_C = len(skill2idx)
    print(f"NUM_C (skills) = {NUM_C}")

    gkt_mod = _load_pykt_module("gkt", str(Path("models") / "gkt.py"))
    gkt_utils = _load_pykt_module("gkt_utils", str(Path("models") / "gkt_utils.py"))
    GKT = gkt_mod.GKT
    build_transition_graph = gkt_utils.build_transition_graph

    first = True
    for fold, dfs in fold_files.items():
        ckpt_path = SAVED_DIR / f"gkt_fold{fold}_{GRAPH_TYPE}_hd{HIDDEN_DIM}.ckpt"
        if not ckpt_path.exists():
            print(f"  [skip] fold {fold}: checkpoint missing ({ckpt_path.name}) — train this fold first")
            continue

        df_train, df_test = dfs["train"], dfs["test"]

        # Transition graph rebuilt from the fold's train set
        concept_seqs = (
            df_train.sort_values("timestamp")
            .assign(skill_idx=lambda d: d["skill"].map(skill2idx))
            .groupby("uid")["skill_idx"]
            .apply(lambda x: ",".join(x.astype(str).tolist()))
            .reset_index()
            .rename(columns={"skill_idx": "concepts"})
        )
        graph = build_transition_graph(concept_seqs, NUM_C)
        if not torch.is_tensor(graph):
            graph = torch.tensor(graph, dtype=torch.float32)
        graph = graph.float()

        model = GKT(num_c=NUM_C, hidden_dim=HIDDEN_DIM, emb_size=EMB_SIZE,
                    graph_type=GRAPH_TYPE, graph=graph, dropout=0.0,
                    emb_type="qid").to(DEVICE)

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)

        test_c, test_r = build_sequences(df_test, skill2idx)
        preds, targets, skills = predict_on_sequences(model, test_c, test_r)
        idx2skill = {i: s for s, i in skill2idx.items()}
        skill_labels = np.array([idx2skill.get(int(i), -1) for i in skills])

        bu.save_oof("gkt", fold, targets, preds, skill=skill_labels, append=not first)
        first = False
        print(f"  fold {fold}: {len(targets)} predictions exported")

    if first:
        print("No fold had a checkpoint — nothing exported.")
    else:
        print(f"OK -> {bu.REGISTRY_BY_KEY['gkt'].oof_path}")


if __name__ == "__main__":
    main()