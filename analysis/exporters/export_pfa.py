"""
Export OOF + sauvegarde des modeles par fold pour PFA.

PFA ne sauvegardait que des metriques agregees. Ce script relance la
cross-validation (meme logique et meme split que models/pfa/pfa.py), puis pour
chaque fold :
  - ecrit les predictions out-of-fold dans le CSV standard ;
  - sauvegarde la regression logistique entrainee + la liste des colonnes de
    features dans models/pfa/folds/pfa_fold{i}.joblib (necessaire car les
    features one-hot dependent des questions vues a l'entrainement).

A lancer depuis la RACINE du projet :
    python -m analysis.exporters.export_pfa
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "pfa"))
sys.path.insert(0, str(ROOT / "analysis"))

import joblib
from sklearn.linear_model import LogisticRegression

import benchmark_utils as bu
from pfa import (
    load_data_for_pfa, filter_rare, balanced_student_kfold_split,
    build_pfa_features, split_features_target,
)

K = 5
SEED = 42
FOLDS_DIR = ROOT / "models" / "pfa" / "folds"


def main():
    df = load_data_for_pfa()
    df = filter_rare(df, min_item_obs=10, min_skill_obs=20)
    folds = balanced_student_kfold_split(df, k=K, seed=SEED)

    first = True
    for fi, (train_df, test_df) in enumerate(folds):
        train_pfa = build_pfa_features(train_df)
        test_pfa = build_pfa_features(test_df)

        X_train, y_train = split_features_target(train_pfa)
        X_test, y_test = split_features_target(test_pfa)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        model = LogisticRegression(max_iter=1000, solver="liblinear")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        bu.save_oof("pfa", fi, y_test.to_numpy(), y_prob, append=not first)
        first = False

        FOLDS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "columns": list(X_train.columns)},
                    FOLDS_DIR / f"pfa_fold{fi}.joblib")
        print(f"  fold {fi} : {len(y_test)} predictions, modele -> pfa_fold{fi}.joblib")

    print(f"OK -> {bu.REGISTRY_BY_KEY['pfa'].oof_path}")


if __name__ == "__main__":
    main()