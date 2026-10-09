"""Train the GIRT model (GKT + 2PL IRT) with student-wise k-fold CV.

Run from the repository root:

    python models/girt/train.py
"""

import os
import sys
from pathlib import Path

import torch
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Allow sibling imports regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from girt import GIRT
from load_data import load_data, student_kfold_split, build_student_sequences

# ---------- HYPERPARAMS ----------
d             = 64
batch_size    = 64
max_epochs    = 200
patience      = 30
learning_rate = 0.005
clip_grad     = 1.0


def main():
    df = load_data()
    folds    = student_kfold_split(df, k=5, seed=42)
    n_items  = int(df["item"].max()  + 1)
    n_skills = int(df["skill"].max() + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"n_items={n_items}, n_skills={n_skills}, d={d}")

    folds_dir = PROJECT_ROOT / "models" / "girt" / "folds"
    os.makedirs(folds_dir, exist_ok=True)

    for fold_idx, (train_df, test_df) in enumerate(folds):
        print(f"\n{'='*40}")
        print(f"  Fold {fold_idx + 1} / 5")
        print(f"{'='*40}")

        train_tensors = build_student_sequences(train_df)
        train_tensors = tuple(t.to(device) for t in train_tensors)
        n_students_train = train_tensors[0].shape[0]

        model     = GIRT(n_skills, n_items, d).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_loss  = float("inf")
        best_state = None
        no_improve = 0

        pbar = tqdm.tqdm(range(max_epochs), desc=f"Fold {fold_idx+1}")
        for epoch in pbar:
            model.train()
            perm           = torch.randperm(n_students_train, device=device)
            epoch_loss_sum = 0.0
            n_batches      = 0

            for batch_start in range(0, n_students_train, batch_size):
                batch_idx = perm[batch_start : batch_start + batch_size]
                batch     = tuple(t[batch_idx] for t in train_tensors)

                lens    = batch[4].sum(dim=1)
                T_batch = int(lens.max().item())
                if T_batch == 0:
                    continue
                batch = tuple(t[:, :T_batch] for t in batch)

                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

                epoch_loss_sum += loss.item()
                n_batches      += 1

            avg_loss = epoch_loss_sum / max(n_batches, 1)

            if avg_loss < best_loss - 1e-5:
                best_loss  = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    pbar.close()
                    print(f"  Early stop at epoch {epoch} | Best loss: {best_loss:.4f}")
                    break

            pbar.set_postfix(loss=f"{avg_loss:.4f}", best=f"{best_loss:.4f}")

        torch.save({
            "model_state_dict": best_state,
            "n_skills": n_skills,
            "n_items":  n_items,
            "d":        d,
            "fold":     fold_idx,
        }, folds_dir / f"girt_fold{fold_idx}.pth")
        print(f"  Saved fold {fold_idx} - best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()