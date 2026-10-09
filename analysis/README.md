# Analyse regroupee du benchmark KT

Comparaison de tous les modeles (BKT, IRT statique, IRT dynamique,
PFA, Optimized PFA, GKT, GIRT) **sur les memes graphiques et tables**.

## Idee

Plutot que de comparer des checkpoints heterogenes, on ramene chaque modele a un
seul artefact commun : ses **predictions out-of-fold** au schema

```
fold, y_true, y_pred [, skill]
```

ecrit dans `models/<modele>/oof_predictions.csv`. Le notebook recalcule ensuite
**toutes les metriques de maniere identique** pour tous les modeles.

## Workflow

1. **Generer les predictions** (une fois, avec les donnees `cleaned_events*.csv`
   et les checkpoints presents) :

   ```bash
   # depuis la racine du projet
   python -m analysis.exporters.run_all          # tout ce qui est disponible
   # ou individuellement :
   python -m analysis.exporters.export_dynamic_irt
   python -m analysis.exporters.export_girt
   python -m analysis.exporters.export_bkt
   python -m analysis.exporters.export_pfa
   python -m analysis.exporters.export_optimized_pfa
   python -m analysis.exporters.export_gkt
   python -m analysis.exporters.export_irt
   ```

2. **Ouvrir le notebook** `analysis/benchmark_analysis.ipynb` et tout executer.
   Sans aucun `oof_predictions.csv`, il bascule sur un jeu **synthetique de
   demonstration** (bandeau d'avertissement) pour montrer sa structure.

## Sauvegarde des modeles par fold

L'export persiste aussi, pour les modeles qui ne le faisaient pas :

| Modele       | Checkpoint par fold ecrit par l'export        |
|--------------|-----------------------------------------------|
| IRT dynamique| (deja present) `models/dynamic_irt/folds/`    |
| GIRT         | (deja present) `models/girt/folds/`           |
| BKT          | `models/bkt/folds/bkt_fold{i}.npz` (params)   |
| PFA          | `models/pfa/folds/pfa_fold{i}.joblib`         |
| Optimized PFA| `models/optimized_pfa/folds/optimized_pfa_fold{i}.pth` |
| IRT statique | `models/irt/folds/irt_fold{i}.pkl` (params py-irt) |
| GKT          | (deja present, fold 0) `models/gkt/saved_models/` — entrainer folds 1-4 |

## Avertissement methodologique

Deux regimes d'evaluation coexistent :

- **cold-start, student-wise** (eleves de test inconnus) : BKT, IRT dynamique,
  GIRT, GKT, PFA, Optimized PFA ;
- **warm-start, splits repetes** : IRT statique (py-irt).

Le second est plus facile : a garder en tete dans toute comparaison directe.
