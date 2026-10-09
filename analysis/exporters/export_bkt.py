"""
Export OOF + sauvegarde des modeles par fold pour BKT.

BKT ne sauvegardait jusqu'ici que des metriques agregees (cv_results*.csv).
Ce script relance la cross-validation (meme logique que bkt_launcher.py),
puis pour CHAQUE fold :
  - ecrit les predictions out-of-fold dans le CSV standard ;
  - sauvegarde les parametres appris [p0, pT, pG, pS] par skill dans
    models/bkt/folds/bkt_fold{i}.npz  (c'est tout l'etat d'un BKT).

A lancer depuis la RACINE du projet :
    python -m analysis.exporters.export_bkt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "bkt"))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np
import pandas as pd

import benchmark_utils as bu
from bkt_model import BKTVectorized
from bkt_launcher import get_default_parameters, create_student_folds_per_skill

K = 5
DATA = ROOT / "datas_management" / "cleaned_events.csv"
FOLDS_DIR = ROOT / "models" / "bkt" / "folds"

MAPPING = {
    "student_id": "id_eleve",
    "timestamp": "timestamp",
    "skill_id": "id_competence",
    "result": "bonne_reponse",
}


def save_bkt_params(params_dict, path):
    """Sauvegarde {skill: array[4]} dans un .npz (cles = skills)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{str(k): np.asarray(v) for k, v in params_dict.items()})


def main():
    df = pd.read_csv(DATA)
    df = df.sort_values([MAPPING["student_id"], MAPPING["skill_id"], MAPPING["timestamp"]])
    df = df.reset_index(drop=True)

    fold_assignments = create_student_folds_per_skill(df, MAPPING, n_splits=K)

    first = True
    for fold in range(K):
        test_idx = np.where(fold_assignments == fold)[0]
        train_idx = np.where(fold_assignments != fold)[0]
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        init = get_default_parameters(df_train, MAPPING["skill_id"])
        bkt = BKTVectorized(df_train, df_test, MAPPING, init)
        bkt.em_algorithm(iterations=100)

        # predict_with_results : DataFrame [student, skill, actual_result, predicted_result]
        preds = bkt.predict_with_results()
        y_true = preds["actual_result"].to_numpy()
        y_pred = preds["predicted_result"].to_numpy()
        skill = preds[MAPPING["skill_id"]].to_numpy()

        bu.save_oof("bkt", fold, y_true, y_pred, skill=skill, append=not first)
        first = False
        save_bkt_params(bkt.parameters_dict, FOLDS_DIR / f"bkt_fold{fold}.npz")
        print(f"  fold {fold} : {len(y_true)} predictions, params -> bkt_fold{fold}.npz")

    print(f"OK -> {bu.REGISTRY_BY_KEY['bkt'].oof_path}")


if __name__ == "__main__":
    main()