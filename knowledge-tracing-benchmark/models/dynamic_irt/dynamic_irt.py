# The models in /folds are already trained and can be loaded with torch.load() for analysis
# Training the model (with lr=0.005, lam=0.1, bptt_steps=20, epochs=3000) takes around 15 minutes per fold.

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "datas_management" / "cleaned_events.csv"

# ---------- DATA ----------

def load_data(path=DEFAULT_DATA_PATH):
    df = pd.read_csv(path)
    df = df[["id_eleve","id_question","id_competence","bonne_reponse","temps","timestamp"]].dropna()
    student_enc = LabelEncoder()
    item_enc    = LabelEncoder()
    skill_enc   = LabelEncoder()
    df["student"] = student_enc.fit_transform(df["id_eleve"])
    df["item"]    = item_enc.fit_transform(df["id_question"])
    df["skill"]   = skill_enc.fit_transform(df["id_competence"])
    df = df.sort_values(["student", "skill", "timestamp"])
    df["temps"]    = df["temps"].clip(lower=1)
    median_time    = df["temps"].median()
    df["log_time"] = np.log(df["temps"] / median_time)
    return df


def student_kfold_split(df, k=5, seed=42):
    """
    Student-wise k-fold split.

    Shuffles the list of unique students, then partitions them into k folds.
    Each fold yields a (train_df, test_df) pair where test contains a disjoint
    1/k% of students and train contains the remaining (1-1/k)%.
    """
    rng = np.random.default_rng(seed)
    students = df["student"].unique()
    rng.shuffle(students)

    folds = np.array_split(students, k)
    splits = []

    for fold_idx in range(k):
        test_students  = folds[fold_idx]
        train_students = np.concatenate([folds[i] for i in range(k) if i != fold_idx])

        train_df = df[df["student"].isin(train_students)].copy()
        test_df  = df[df["student"].isin(test_students)].copy()

        train_enc = {s: i for i, s in enumerate(sorted(train_students))}
        test_enc  = {s: i for i, s in enumerate(sorted(test_students))}
        train_df["student"] = train_df["student"].map(train_enc)
        test_df["student"]  = test_df["student"].map(test_enc)

        splits.append((train_df.reset_index(drop=True),
                       test_df.reset_index(drop=True)))
    return splits


def build_grouped_tensors(df):
    """
    Groups observations by (student, skill) pair in order to speedup the process

    Returns
    -------
    seq_student : (n_pairs,)            student index for each pair
    seq_skill   : (n_pairs,)            skill index for each pair
    seq_item    : (n_pairs, max_len)    item index at each time step (0-padded)
    seq_y       : (n_pairs, max_len)    correct response              (0-padded)
    seq_r       : (n_pairs, max_len)    log response time             (0-padded)
    seq_mask    : (n_pairs, max_len)    True = real observation, False = padding
    orig_idx    : (n_pairs, max_len)    original row index in df (-1 = padding)
                                        used to reorder predictions to df order
    """
    df = df.sort_values(["student", "skill", "timestamp"])

    students_list, skills_list = [], []
    items_list, y_list, r_list, idx_list = [], [], [], []

    for (s, k), g in df.groupby(["student", "skill"], sort=True):
        students_list.append(s)
        skills_list.append(k)
        items_list.append(g["item"].values)
        y_list.append(g["bonne_reponse"].values)
        r_list.append(g["log_time"].values)
        idx_list.append(g.index.values)

    n_pairs = len(students_list)
    max_len = max(len(x) for x in items_list)

    seq_student = torch.tensor(students_list, dtype=torch.long)
    seq_skill   = torch.tensor(skills_list,   dtype=torch.long)
    seq_item    = torch.zeros(n_pairs, max_len, dtype=torch.long)
    seq_y       = torch.zeros(n_pairs, max_len, dtype=torch.float32)
    seq_r       = torch.zeros(n_pairs, max_len, dtype=torch.float32)
    seq_mask    = torch.zeros(n_pairs, max_len, dtype=torch.bool)
    orig_idx    = torch.full((n_pairs, max_len), -1, dtype=torch.long)

    for i in range(n_pairs):
        L = len(items_list[i])
        seq_item[i, :L] = torch.tensor(items_list[i], dtype=torch.long)
        seq_y[i,    :L] = torch.tensor(y_list[i],     dtype=torch.float32)
        seq_r[i,    :L] = torch.tensor(r_list[i],     dtype=torch.float32)
        seq_mask[i, :L] = True
        orig_idx[i, :L] = torch.tensor(idx_list[i],   dtype=torch.long)

    return seq_student, seq_skill, seq_item, seq_y, seq_r, seq_mask, orig_idx


# ---------- MODEL ----------

class DynamicIRT(nn.Module):
    """
    Dynamic Contextual IRT with per-skill Gaussian transition prior.

    Observation model:
        P(y=1 | θ_ikt, j) = σ( a_j * (θ_ikt - b_j) + β * log_time )

    Theta update:
        θ_ik,t+1 = θ_ikt + α_k * (y - p) + μ_k

    Parameters
    ----------
    n_students : int
    n_skills   : int
    n_items    : int
    lam        : float   weight of the dynamic regularisation term (default 0.1)
    bptt_steps : int     detach the computation graph every K steps to limit memory on long sequences (default 20).
                         Set to None to keep the full graph (exact gradients).
    """

    def __init__(self, n_students, n_skills, n_items, lam=0.1, bptt_steps=20):
        super().__init__()
        self.lam        = lam
        self.bptt_steps = bptt_steps

        self.register_buffer("theta0", torch.zeros(n_students, n_skills))

        # Item parameters
        self.a         = nn.Parameter(torch.ones(n_items)) # softplus → always > 0
        self.b         = nn.Parameter(torch.zeros(n_items)) # difficulty

        # Context weight (log response time)
        self.beta      = nn.Parameter(torch.tensor(0.0))

        # Per-skill dynamics
        self.alpha     = nn.Parameter(torch.full((n_skills,), 0.2)) # ELO like learning rate
        self.mu        = nn.Parameter(torch.zeros(n_skills)) # mean drift per step
        self.log_sigma = nn.Parameter(torch.zeros(n_skills)) # log σ_k

    def forward(self, grouped_tensors):
        """
        Grouped forward pass.

        Loops over time steps (max_len iterations), processing all
        (student, skill) pairs in parallel at each step.

        grouped_tensors is the output of build_grouped_tensors().
        """
        seq_student, seq_skill, seq_item, seq_y, seq_r, seq_mask, _ = grouped_tensors
        _, max_len = seq_item.shape

        a_pos = F.softplus(self.a) # (n_items,)  discrimination > 0
        sigma = torch.exp(self.log_sigma) # (n_skills,) σ_k > 0

        # Per-pair skill parameters
        alpha_k   = self.alpha[seq_skill]
        mu_k      = self.mu[seq_skill]
        sigma_k   = sigma[seq_skill]
        log_sig_k = self.log_sigma[seq_skill]

        # All students start from theta = 0 (cold start)
        theta_current = self.theta0[seq_student, seq_skill] + 0.0

        l_obs_steps = []
        l_dyn_steps = []

        for t in range(max_len):
            mask_t = seq_mask[:, t] # (n_pairs,)
            if not mask_t.any():
                break

            j   = seq_item[:, t] # (n_pairs,)
            y_t = seq_y[:,   t]
            r_t = seq_r[:,   t]

            # Observation model (vectorized over all pairs to speedup the process)
            logit = a_pos[j] * (theta_current - self.b[j]) + self.beta * r_t
            p_t   = torch.sigmoid(logit)

            # Observation loss
            l_obs_steps.append(F.binary_cross_entropy(p_t[mask_t], y_t[mask_t], reduction="mean"))

            expected_delta = alpha_k * (y_t - p_t) + mu_k # (n_pairs,)

            l_dyn_steps.append(((expected_delta[mask_t] ** 2) / (2 * sigma_k[mask_t] ** 2) + log_sig_k[mask_t] + 0.5 * log_sig_k[mask_t] ** 2).mean())

            # Theta update
            theta_current = theta_current + expected_delta * mask_t.float()

            if self.bptt_steps and ((t + 1) % self.bptt_steps == 0):
                theta_current = theta_current.detach()

        l_obs = torch.stack(l_obs_steps).mean()
        l_dyn = torch.stack(l_dyn_steps).mean()

        return l_obs + self.lam * l_dyn


# ---------- PREDICTION ----------

def predict(model, grouped_tensors):
    """
    Sequential prediction (calculates the predicted probability and the new value of theta at each step)
    """
    model.eval()
    seq_student, seq_skill, seq_item, seq_y, seq_r, seq_mask, orig_idx = grouped_tensors
    n_pairs, max_len = seq_item.shape

    a_pos   = F.softplus(model.a)
    alpha_k = model.alpha[seq_skill]
    mu_k    = model.mu[seq_skill]

    # All students start from 0
    theta_current = torch.zeros(n_pairs)

    n_obs     = int(seq_mask.sum().item())
    all_preds = torch.zeros(n_obs)
    all_orig  = torch.zeros(n_obs, dtype=torch.long)
    ptr = 0

    with torch.no_grad():
        for t in range(max_len):
            mask_t = seq_mask[:, t]
            if not mask_t.any():
                break

            j   = seq_item[:, t]
            y_t = seq_y[:,   t]
            r_t = seq_r[:,   t]

            logit = a_pos[j] * (theta_current - model.b[j]) + model.beta * r_t
            p_t   = torch.sigmoid(logit)

            n_valid = int(mask_t.sum().item())
            all_preds[ptr : ptr + n_valid] = p_t[mask_t]
            all_orig[ ptr : ptr + n_valid] = orig_idx[:, t][mask_t]
            ptr += n_valid

            delta         = (alpha_k * (y_t - p_t) + mu_k) * mask_t.float()
            theta_current = theta_current + delta

    # Reorder to match original df row order
    order = torch.argsort(all_orig)
    return all_preds[order].detach()


# ---------- MAIN ----------

def main():
    df = load_data()

    folds    = student_kfold_split(df, k=5, seed=42)
    n_items  = int(df["item"].max()  + 1)
    n_skills = int(df["skill"].max() + 1)

    for fold_idx, (train_df, test_df) in enumerate(folds):
        print(f"\n{'='*40}")
        print(f"  Fold {fold_idx + 1} / 5")
        print(f"{'='*40}")

        n_students_train = int(train_df["student"].max() + 1)

        train_tensors = build_grouped_tensors(train_df)
        test_tensors  = build_grouped_tensors(test_df)

        model     = DynamicIRT(n_students_train, n_skills, n_items, lam=0.1, bptt_steps=20)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        best_loss    = float("inf")
        patience     = 200          # stop if no improvement for 200 epochs
        no_improve   = 0
        max_epochs   = 3000

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = model(train_tensors)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            # Early stopping to speedup training process (just time saver)
            if loss_val < best_loss - 1e-5:
                best_loss  = loss_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop at epoch {epoch} | Best loss: {best_loss:.4f}")
                    break

            if epoch % 100 == 0:
                print(f"  Epoch {epoch:5d} | Loss: {loss_val:.4f}")

        torch.save({
            "model_state_dict": model.state_dict(),
            "n_students": n_students_train,
            "n_skills":   n_skills,
            "n_items":    n_items,
            "fold":       fold_idx,
        }, PROJECT_ROOT / "models" / "dynamic_irt" / "folds" / f"dynamic_irt_fold{fold_idx}.pth")


if __name__ == "__main__":
    main()