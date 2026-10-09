import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
    df["temps"]    = df["temps"].clip(lower=1)
    median_time    = df["temps"].median()
    df["log_time"] = np.log(df["temps"] / median_time)
    return df


def student_kfold_split(df, k=5, seed=42):
    """
    Student-wise k-fold split. Each test fold contains a disjoint 1/k% of students.
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


def build_student_sequences(df):
    """
    Build per-student temporal sequences (one row per student, all interactions in order).

    This is the proper KT structure: each row is one student's full interaction history,
    sorted by timestamp. Different rows = different students with independent knowledge states.

    Returns
    -------
    seq_item  : (n_students, max_len)   long, 0-padded
    seq_skill : (n_students, max_len)   long, 0-padded — the skill of each item
    seq_y     : (n_students, max_len)   float, 0-padded — response (0/1)
    seq_r     : (n_students, max_len)   float, 0-padded — log response time
    seq_mask  : (n_students, max_len)   bool, True = real observation, False = padding
    orig_idx  : (n_students, max_len)   long, -1 = padding — original df row index
                                         (used to reorder predictions to df order)
    """
    df = df.sort_values(["student", "timestamp"])

    items_list, skills_list, y_list, r_list, idx_list = [], [], [], [], []

    for s, g in df.groupby("student", sort=True):
        items_list.append(g["item"].values)
        skills_list.append(g["skill"].values)
        y_list.append(g["bonne_reponse"].values)
        r_list.append(g["log_time"].values)
        idx_list.append(g.index.values)

    n_students = len(items_list)
    max_len    = max(len(x) for x in items_list)

    seq_item  = torch.zeros(n_students, max_len, dtype=torch.long)
    seq_skill = torch.zeros(n_students, max_len, dtype=torch.long)
    seq_y     = torch.zeros(n_students, max_len, dtype=torch.float32)
    seq_r     = torch.zeros(n_students, max_len, dtype=torch.float32)
    seq_mask  = torch.zeros(n_students, max_len, dtype=torch.bool)
    orig_idx  = torch.full((n_students, max_len), -1, dtype=torch.long)

    for i in range(n_students):
        L = len(items_list[i])
        seq_item[i,  :L] = torch.from_numpy(np.array(items_list[i],  dtype=np.int64))
        seq_skill[i, :L] = torch.from_numpy(np.array(skills_list[i], dtype=np.int64))
        seq_y[i,     :L] = torch.from_numpy(np.array(y_list[i],      dtype=np.float32))
        seq_r[i,     :L] = torch.from_numpy(np.array(r_list[i],      dtype=np.float32))
        seq_mask[i,  :L] = True
        orig_idx[i,  :L] = torch.from_numpy(np.array(idx_list[i],    dtype=np.int64))

    return seq_item, seq_skill, seq_y, seq_r, seq_mask, orig_idx