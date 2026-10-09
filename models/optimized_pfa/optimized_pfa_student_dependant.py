# Model C (LKT) — v3, ANTI-OVERFITTING, with PER-STUDENT decay d_i.
#
# Identical to Optimized_PFA_v3.py (Model A v3 harness) except for:
#
#   - log_d is a VECTOR of shape (n_students,) instead of a scalar.
#     d_i = exp(log_d[i]) gives a separate forgetting rate per student.
#
#   - Cold-start handling: per-student parameters cannot be transferred to new
#     students. At validation and test time, we substitute the MEDIAN of the
#     trained log_d values for every previously-unseen student. The forward()
#     pass during training uses the actual per-student log_d.
#
#   - Per-student log_d is added to the hierarchical (partial-pooling)
#     regularisation, so each student's decay is pulled toward the global mean
#     unless the data justifies divergence.

import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "datas_management" / "cleaned_events.csv"


# ---------- DATA ----------
def load_data(path=DEFAULT_DATA_PATH):
    df = pd.read_csv(path)
    df = df[["id_eleve", "id_question", "id_competence",
             "bonne_reponse", "temps", "timestamp"]].dropna()
    df["student"] = LabelEncoder().fit_transform(df["id_eleve"])
    df["item"]    = LabelEncoder().fit_transform(df["id_question"])
    df["skill"]   = LabelEncoder().fit_transform(df["id_competence"])
    df = df.sort_values(["student", "skill", "timestamp"])

    df["temps"]    = df["temps"].clip(lower=1)
    df["log_time"] = np.log(df["temps"] / df["temps"].median())
    df["t_days"]   = (df["timestamp"] - df["timestamp"].min()) / 86400.0
    return df


def filter_rare(df, min_item_obs=10, min_skill_obs=20):
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


def split_train_val(train_df, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    students = train_df["student"].unique().copy()
    rng.shuffle(students)

    n_val = max(1, int(round(len(students) * val_frac)))
    val_students = set(students[:n_val].tolist())

    val_df  = train_df[ train_df["student"].isin(val_students)].reset_index(drop=True)
    core_df = train_df[~train_df["student"].isin(val_students)].reset_index(drop=True)
    return core_df, val_df


def compute_item_difficulty_init(train_df, n_items, pseudo_counts=5.0):
    g = min(max(float(train_df["bonne_reponse"].mean()), 0.02), 0.98)
    b = np.full(n_items, -np.log(g / (1.0 - g)), dtype=np.float32)

    s = train_df.groupby("item")["bonne_reponse"].sum()
    c = train_df.groupby("item")["bonne_reponse"].count()
    for j in c.index:
        p = min(max(float((s[j] + g * pseudo_counts) / (c[j] + pseudo_counts)), 0.02), 0.98)
        b[int(j)] = -np.log(p / (1.0 - p))
    return torch.from_numpy(b)


def build_grouped_tensors(df):
    df = df.sort_values(["student", "skill", "timestamp"])

    students_list, skills_list = [], []
    items_list, y_list, r_list, t_list, idx_list = [], [], [], [], []
    for (s, k), gdf in df.groupby(["student", "skill"], sort=True):
        students_list.append(s)
        skills_list.append(k)
        items_list.append(gdf["item"].values)
        y_list.append(gdf["bonne_reponse"].values)
        r_list.append(gdf["log_time"].values)
        t_list.append(gdf["t_days"].values)
        idx_list.append(gdf.index.values)

    n_pairs = len(students_list)
    max_len = max(len(x) for x in items_list)

    seq_student = torch.tensor(students_list, dtype=torch.long)
    seq_skill   = torch.tensor(skills_list,   dtype=torch.long)
    seq_item    = torch.zeros(n_pairs, max_len, dtype=torch.long)
    seq_y       = torch.zeros(n_pairs, max_len, dtype=torch.float32)
    seq_r       = torch.zeros(n_pairs, max_len, dtype=torch.float32)
    seq_t       = torch.zeros(n_pairs, max_len, dtype=torch.float32)
    seq_mask    = torch.zeros(n_pairs, max_len, dtype=torch.bool)
    orig_idx    = torch.full((n_pairs, max_len), -1, dtype=torch.long)

    for i in range(n_pairs):
        L = len(items_list[i])
        seq_item[i, :L] = torch.tensor(items_list[i], dtype=torch.long)
        seq_y[i,    :L] = torch.tensor(y_list[i],     dtype=torch.float32)
        seq_r[i,    :L] = torch.tensor(r_list[i],     dtype=torch.float32)
        seq_t[i,    :L] = torch.tensor(t_list[i],     dtype=torch.float32)
        seq_mask[i, :L] = True
        orig_idx[i, :L] = torch.tensor(idx_list[i],   dtype=torch.long)

    return seq_student, seq_skill, seq_item, seq_y, seq_r, seq_t, seq_mask, orig_idx


# ---------- MODEL ----------
class LKT_ModelC(nn.Module):
    """
    Model C = Model A v3 with PER-STUDENT decay d_i.

    Logit:
        logit(P) = a_j * (theta_{i,k}(t) - b_j)
                 + beta * log_time
                 + gamma_k * log(1 + wsuc(i,k,t))
                 + rho_k   * log(1 + wfail(i,k,t))
                 + delta   * log(1 + mean_interval(i,k,t))

    where d_i = exp(log_d[i]) is the forgetting rate of student i.

    Per-student parameters cannot be reused on new students. Predictions on
    val/test go through predict(..., use_median_d=True), which uses the median
    of the trained log_d as a cold-start fallback.
    """
    def __init__(self, n_students, n_skills, n_items,
                 lam=0.1, tau=0.0, bptt_steps=20, b_init=None, rasch=False):
        super().__init__()
        self.lam        = lam
        self.tau        = tau
        self.bptt_steps = bptt_steps
        self.rasch      = rasch

        self.register_buffer("theta0", torch.zeros(n_students, n_skills))

        if rasch:
            self.register_buffer("a", torch.ones(n_items))   # frozen 1PL
        else:
            self.a = nn.Parameter(torch.ones(n_items))

        self.b = nn.Parameter(
            b_init.clone().float() if b_init is not None
            else torch.zeros(n_items)
        )

        self.beta      = nn.Parameter(torch.tensor(0.0))
        self.alpha     = nn.Parameter(torch.full((n_skills,), 0.2))
        self.mu        = nn.Parameter(torch.zeros(n_skills))
        self.log_sigma = nn.Parameter(torch.zeros(n_skills))
        self.gamma     = nn.Parameter(torch.zeros(n_skills))
        self.rho       = nn.Parameter(torch.zeros(n_skills))

        # *** Model C change: PER-STUDENT log_d. ***
        # Vector of shape (n_students,) — one forgetting rate per student.
        # Same initial value everywhere (log(0.05) = 0.05/day, half-life ~14d).
        self.log_d = nn.Parameter(
            torch.full((n_students,), np.log(0.05), dtype=torch.float32)
        )

        self.delta = nn.Parameter(torch.tensor(0.0))

    def _a_eff(self):
        return self.a if self.rasch else F.softplus(self.a)

    def _skill_reg(self):
        """
        Partial-pooling regularisation. Per-skill vectors are pulled toward
        their across-skill mean; per-student log_d is pulled toward its
        across-student mean.
        """
        def dev(p):
            return ((p - p.mean()) ** 2).mean()
        return (dev(self.alpha) + dev(self.mu) + dev(self.gamma)
                + dev(self.rho) + dev(self.log_sigma) + dev(self.log_d))

    def forward(self, gt):
        seq_student, seq_skill, seq_item, seq_y, seq_r, seq_t, seq_mask, _ = gt
        n_pairs, max_len = seq_item.shape

        a_pos = self._a_eff()
        sigma = torch.exp(self.log_sigma)

        alpha_k   = self.alpha[seq_skill]
        mu_k      = self.mu[seq_skill]
        sigma_k   = sigma[seq_skill]
        log_sig_k = self.log_sigma[seq_skill]
        gamma_k   = self.gamma[seq_skill]
        rho_k     = self.rho[seq_skill]

        # *** Per-STUDENT decay gathered per pair. ***
        # Only valid here because forward() is called on training data, where
        # seq_student indexes the model's own students.
        d_i = torch.exp(self.log_d[seq_student])   # shape (n_pairs,)

        theta    = self.theta0[seq_student, seq_skill] + 0.0
        wsuc     = torch.zeros(n_pairs)
        wfail    = torch.zeros(n_pairs)
        t_prev   = torch.zeros(n_pairs)
        has_prev = torch.zeros(n_pairs, dtype=torch.bool)
        sum_int  = torch.zeros(n_pairs)
        n_int    = torch.zeros(n_pairs)

        l_obs_sum  = 0.0
        n_obs      = 0
        l_dyn_steps = []

        for ts in range(max_len):
            mask_t = seq_mask[:, ts]
            if not mask_t.any():
                break

            j   = seq_item[:, ts]
            y_t = seq_y[:,   ts]
            r_t = seq_r[:,   ts]
            t_t = seq_t[:,   ts]

            dt    = t_t - t_prev
            decay = torch.exp(-d_i * dt)
            wsuc_now  = torch.where(has_prev, wsuc  * decay, wsuc)
            wfail_now = torch.where(has_prev, wfail * decay, wfail)
            mean_int  = sum_int / torch.clamp(n_int, min=1.0)

            logit = (
                a_pos[j] * (theta - self.b[j])
                + self.beta * r_t
                + gamma_k * torch.log1p(wsuc_now)
                + rho_k   * torch.log1p(wfail_now)
                + self.delta * torch.log1p(mean_int)
            )
            p_t = torch.sigmoid(logit)

            l_obs_sum += F.binary_cross_entropy(p_t[mask_t], y_t[mask_t], reduction="sum")
            n_obs     += mask_t.sum()

            exp_delta = alpha_k * (y_t - p_t) + mu_k
            l_dyn_steps.append((
                (exp_delta[mask_t] ** 2) / (2 * sigma_k[mask_t] ** 2)
                + log_sig_k[mask_t] + 0.5 * log_sig_k[mask_t] ** 2
            ).mean())

            theta = theta + exp_delta * mask_t.float()
            wsuc  = torch.where(mask_t, wsuc_now  + y_t,         wsuc)
            wfail = torch.where(mask_t, wfail_now + (1.0 - y_t), wfail)

            ci      = mask_t & has_prev
            sum_int = torch.where(ci, sum_int + dt,  sum_int)
            n_int   = torch.where(ci, n_int   + 1.0, n_int)
            t_prev  = torch.where(mask_t, t_t, t_prev)
            has_prev = has_prev | mask_t

            if self.bptt_steps and ((ts + 1) % self.bptt_steps == 0):
                theta = theta.detach()
                wsuc  = wsuc.detach()
                wfail = wfail.detach()

        l_obs = l_obs_sum / max(
            n_obs.item() if isinstance(n_obs, torch.Tensor) else n_obs, 1
        )
        l_dyn = torch.stack(l_dyn_steps).mean()
        return l_obs + self.lam * l_dyn + self.tau * self._skill_reg()


# ---------- PREDICTION / EVAL ----------
def predict(model, gt, use_median_d=True):
    """
    Sequential prediction.

    use_median_d=True (default): every pair uses the MEDIAN of the trained
    log_d. This is what we want for val and test students, who don't have a
    learned d_i.

    use_median_d=False: use model.log_d[seq_student] directly. Only valid
    when seq_student indexes the model's own training students.
    """
    model.eval()
    seq_student, seq_skill, seq_item, seq_y, seq_r, seq_t, seq_mask, orig_idx = gt
    n_pairs, max_len = seq_item.shape

    a_pos   = model._a_eff()
    alpha_k = model.alpha[seq_skill]
    mu_k    = model.mu[seq_skill]
    gamma_k = model.gamma[seq_skill]
    rho_k   = model.rho[seq_skill]

    # Pick per-pair decay.
    if use_median_d:
        median_d = torch.exp(model.log_d).median().item()
        d_i = torch.full((n_pairs,), median_d, dtype=torch.float32)
    else:
        d_i = torch.exp(model.log_d[seq_student])

    theta    = torch.zeros(n_pairs)
    wsuc     = torch.zeros(n_pairs)
    wfail    = torch.zeros(n_pairs)
    t_prev   = torch.zeros(n_pairs)
    has_prev = torch.zeros(n_pairs, dtype=torch.bool)
    sum_int  = torch.zeros(n_pairs)
    n_int    = torch.zeros(n_pairs)

    n_obs     = int(seq_mask.sum().item())
    all_preds = torch.zeros(n_obs)
    all_orig  = torch.zeros(n_obs, dtype=torch.long)
    ptr = 0

    with torch.no_grad():
        for ts in range(max_len):
            mask_t = seq_mask[:, ts]
            if not mask_t.any():
                break

            j   = seq_item[:, ts]
            y_t = seq_y[:,   ts]
            r_t = seq_r[:,   ts]
            t_t = seq_t[:,   ts]

            dt    = t_t - t_prev
            decay = torch.exp(-d_i * dt)
            wsuc_now  = torch.where(has_prev, wsuc  * decay, wsuc)
            wfail_now = torch.where(has_prev, wfail * decay, wfail)
            mean_int  = sum_int / torch.clamp(n_int, min=1.0)

            logit = (
                a_pos[j] * (theta - model.b[j])
                + model.beta * r_t
                + gamma_k * torch.log1p(wsuc_now)
                + rho_k   * torch.log1p(wfail_now)
                + model.delta * torch.log1p(mean_int)
            )
            p_t = torch.sigmoid(logit)

            nv = int(mask_t.sum().item())
            all_preds[ptr:ptr + nv] = p_t[mask_t]
            all_orig[ ptr:ptr + nv] = orig_idx[:, ts][mask_t]
            ptr += nv

            theta = theta + (alpha_k * (y_t - p_t) + mu_k) * mask_t.float()
            wsuc  = torch.where(mask_t, wsuc_now  + y_t,         wsuc)
            wfail = torch.where(mask_t, wfail_now + (1.0 - y_t), wfail)
            ci    = mask_t & has_prev
            sum_int = torch.where(ci, sum_int + dt,  sum_int)
            n_int   = torch.where(ci, n_int   + 1.0, n_int)
            t_prev  = torch.where(mask_t, t_t, t_prev)
            has_prev = has_prev | mask_t

    return all_preds[torch.argsort(all_orig)].detach()


def evaluate(model, gt, df, use_median_d=True):
    """
    Wrapper around predict + metrics. Default use_median_d=True is correct
    for val/test on new students. Pass use_median_d=False for training metrics.
    """
    preds = predict(model, gt, use_median_d=use_median_d).numpy()
    y = df["bonne_reponse"].to_numpy().astype(float)
    p = np.clip(preds, 1e-6, 1 - 1e-6)
    try:
        auc = roc_auc_score(y, preds)
    except ValueError:
        auc = float("nan")
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    bce  = float(log_loss(y, p, labels=[0, 1]))
    return auc, rmse, bce


# ---------- TRAINING ----------
def make_optimizer(model, lr, weight_decay):
    # 'b' stays OUT of decay (don't shrink the data-informed init toward 0).
    # 'log_d' stays OUT too (the per-student decay shouldn't be shrunk toward 0,
    # which would mean d -> 1 / very fast forgetting).
    decay_names = {"a", "beta", "gamma", "rho", "delta"}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if name in decay_names else no_decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def train_one_fold(train_df, n_skills, n_items, cfg, seed=0):
    core_df, val_df = split_train_val(train_df, val_frac=0.15, seed=seed)
    core_t = build_grouped_tensors(core_df)
    val_t  = build_grouped_tensors(val_df)

    n_students = int(core_df["student"].max() + 1)
    b_init = compute_item_difficulty_init(core_df, n_items)

    model = LKT_ModelC(
        n_students, n_skills, n_items,
        lam=cfg["lam"], tau=cfg["tau"], bptt_steps=20,
        b_init=b_init, rasch=cfg["rasch"],
    )

    opt = make_optimizer(model, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-4
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    check, es_pat = 10, cfg["es_patience_checks"]

    for epoch in range(cfg["max_epochs"]):
        model.train()
        opt.zero_grad()
        loss = model(core_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if epoch % check == 0:
            # Val students are NEW: use median d_i.
            _, _, val_bce = evaluate(model, val_t, val_df, use_median_d=True)
            sched.step(val_bce)

            if val_bce < best_val - 1e-4:
                best_val   = val_bce
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 100 == 0:
                # Train metrics use the actual per-student log_d.
                tr_auc, _, tr_bce = evaluate(model, core_t, core_df, use_median_d=False)
                va_auc, _, _      = evaluate(model, val_t,  val_df,  use_median_d=True)
                print(f"  ep {epoch:5d} | train_bce {tr_bce:.4f} | val_bce {val_bce:.4f} "
                      f"| gap {val_bce - tr_bce:+.4f} | train_auc {tr_auc:.4f} | val_auc {va_auc:.4f}")

            if no_improve >= es_pat:
                print(f"  Early stop @ {epoch} | best val_bce {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, n_students


# ---------- MAIN ----------
def main():
    CONFIG = {
        "rasch":              True,
        "tau":                0.01,
        "lam":                0.05,
        "weight_decay":       1e-3,
        "lr":                 0.001,
        "max_epochs":         3000,
        "es_patience_checks": 12,
        "min_item_obs":       10,
        "min_skill_obs":      20,
    }

    folds_dir = PROJECT_ROOT / "models" / "optimized_pfa" / "folds"
    os.makedirs(folds_dir, exist_ok=True)

    df = load_data()
    df = filter_rare(df, CONFIG["min_item_obs"], CONFIG["min_skill_obs"])

    folds    = balanced_student_kfold_split(df, k=5, seed=42)
    n_items  = int(df["item"].max()  + 1)
    n_skills = int(df["skill"].max() + 1)
    print(f"After filtering: {len(df)} obs | {n_items} items | {n_skills} skills")

    results = []
    for fi, (train_df, test_df) in enumerate(folds):
        print(f"\n{'='*52}\n  Fold {fi + 1} / 5  -  Model C (v3, per-student d_i)\n{'='*52}")

        model, n_students = train_one_fold(train_df, n_skills, n_items, CONFIG, seed=fi)

        # Test students are NEW: use median d_i.
        test_t = build_grouped_tensors(test_df)
        auc, rmse, bce = evaluate(model, test_t, test_df, use_median_d=True)
        print(f"  TEST | AUC {auc:.4f} | RMSE {rmse:.4f} | BCE {bce:.4f}")
        results.append({"fold": fi + 1, "auc": auc, "rmse": rmse, "bce": bce})

        # Inspect the distribution of learned d_i (one per trained student).
        d_per_student = torch.exp(model.log_d).detach().cpu().numpy()
        print(f"  d_i stats (trained students): "
              f"min={d_per_student.min():.4f}  "
              f"median={np.median(d_per_student):.4f}  "
              f"max={d_per_student.max():.4f}  (per day)")
        print(f"  d used for test students (median): "
              f"{np.median(d_per_student):.4f} per day")

        torch.save({
            "model_state_dict": model.state_dict(),
            "config":           CONFIG,
            "n_students":       n_students,
            "n_skills":         n_skills,
            "n_items":          n_items,
            "fold":             fi,
            "median_log_d":     float(np.log(np.median(d_per_student))),
        }, folds_dir / f"optimized_pfa_fold{fi}.pth")

    res = pd.DataFrame(results)
    print(f"\n\n{'='*52}\nCROSS-VALIDATION SUMMARY (5 FOLDS) - Model C v3\n{'='*52}")
    print(f"  AUC  : {res['auc'].mean():.4f} ± {res['auc'].std():.4f}")
    print(f"  RMSE : {res['rmse'].mean():.4f} ± {res['rmse'].std():.4f}")
    print(f"  BCE  : {res['bce'].mean():.4f} ± {res['bce'].std():.4f}")
    out_csv = PROJECT_ROOT / "models" / "optimized_pfa" / "cv_results.csv"
    res.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()