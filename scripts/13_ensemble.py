"""Ensemble model by estimator pooling, and its evaluation (Sect. 4.6, 5.3).

1. Pooling: the fitted trees of the two final parent models (500 Unconstrained
   + 500 Slope Constrained) are merged into one 1000-tree Random Forest, whose
   predicted probability is the mean vote fraction of all trees.
2. Evaluation table (gls_common.mixed_table): all presences plus a random
   ENSEMBLE_ABSENCE_FRACTION of the pseudo-absences of each parent table,
   after thinning.
3. On this table:
   - equivalence of the pooled model with the arithmetic mean of the two
     parents' probabilities (maximum absolute difference);
   - block-wise evaluation: the records are grouped into spatial blocks
     (BLOCK_SIZE_KM) and the blocks into N_FOLDS folds, as for the parents;
     AUC, and precision, recall and F1 at the 0.5 threshold, are computed for
     the pooled model on each fold (the pooled model is not refitted);
     mean, variance, standard deviation and coefficient of variation across folds;
   - permutation importance (ROC-AUC scoring).

Input : $WORK_DIR/{slope_constrained,unconstrained}/model/rf_model.joblib
        $WORK_DIR/{slope_constrained,unconstrained}/training_table.txt
Output: $WORK_DIR/ensemble/model/{rf_model.joblib, evaluation_summary.txt,
        evaluation_per_fold.csv, permutation_importance.csv}
Usage : sbatch slurm/python_job.sh scripts/13_ensemble.py
"""
import copy
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from gls_common import (LABEL_COL, env, feature_columns, mixed_table, n_jobs,
                        spatial_blocks)

WORK_DIR = env("WORK_DIR")
OUT_DIR = os.path.join(WORK_DIR, "ensemble", "model")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = env("SEED", 42, int)
BLOCK_KM = env("BLOCK_SIZE_KM", cast=float)
N_FOLDS = env("N_FOLDS", cast=int)
SC, UN = "slope_constrained", "unconstrained"
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def pool(model_a, model_b):
    """Single Random Forest holding the trees of both models."""
    assert model_a.n_features_in_ == model_b.n_features_in_, "different number of features"
    assert np.array_equal(model_a.classes_, model_b.classes_), "different class order"
    if hasattr(model_a, "feature_names_in_") and hasattr(model_b, "feature_names_in_"):
        assert list(model_a.feature_names_in_) == list(model_b.feature_names_in_), \
            "different feature order"
    pooled = copy.deepcopy(model_a)
    pooled.estimators_ = list(model_a.estimators_) + list(model_b.estimators_)
    pooled.n_estimators = len(pooled.estimators_)
    return pooled


# --- 1. Pooling ---------------------------------------------------------------------------
parents = {m: joblib.load(os.path.join(WORK_DIR, m, "model", "rf_model.joblib")) for m in (UN, SC)}
ensemble = pool(parents[UN], parents[SC])
joblib.dump(ensemble, os.path.join(OUT_DIR, "rf_model.joblib"))
print(f"Pooled model: {ensemble.n_estimators} trees "
      f"({parents[UN].n_estimators} {UN} + {parents[SC].n_estimators} {SC})")

# --- 2. Evaluation table --------------------------------------------------------------------
table = mixed_table()
X, y = table[feature_columns(table)], table[LABEL_COL]
groups = spatial_blocks(table, BLOCK_KM)

# --- 3. Equivalence, block-wise metrics, permutation importance ------------------------------
proba = ensemble.predict_proba(X)
proba_mean = (parents[UN].predict_proba(X) + parents[SC].predict_proba(X)) / 2
max_diff = float(np.max(np.abs(proba - proba_mean)))
pred = ensemble.predict(X)

rows = []
for fold, (_, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, groups), start=1):
    y_te, p_te, c_te = y.iloc[te], proba[te, 1], pred[te]
    rows.append(dict(fold=fold, n=len(te), auc=roc_auc_score(y_te, p_te),
                     precision=precision_score(y_te, c_te), recall=recall_score(y_te, c_te),
                     f1=f1_score(y_te, c_te)))
    print(f"Fold {fold}: " + ", ".join(f"{k}={v:.4f}" for k, v in rows[-1].items()
                                       if isinstance(v, float)), flush=True)
folds = pd.DataFrame(rows)
folds.to_csv(os.path.join(OUT_DIR, "evaluation_per_fold.csv"), index=False)

counts = table["source"].value_counts()
lines = ["Ensemble (pooled model): block-wise evaluation",
         f"Records: {len(table):,} (presences {counts.get('presence', 0):,}; "
         f"pseudo-absences {SC} {counts.get(SC, 0):,}, {UN} {counts.get(UN, 0):,})",
         f"Spatial blocks: {groups.nunique()} | folds: {len(folds)}",
         f"Pooled vs mean of parent probabilities: max absolute difference = {max_diff:.2e}", ""]
for m in ("auc", "precision", "recall", "f1"):
    v = folds[m]
    lines.append(f"{m:<10} mean={v.mean():.4f}  var={v.var(ddof=0):.6f}  "
                 f"std={v.std(ddof=0):.4f}  CV%={100 * v.std(ddof=0) / v.mean():.2f}")
text = "\n".join(lines)
print("\n" + text, flush=True)
with open(os.path.join(OUT_DIR, "evaluation_summary.txt"), "w") as f:
    f.write(text + "\n")

print("\nComputing permutation importance...", flush=True)
perm = permutation_importance(
    ensemble, np.ascontiguousarray(X.values, dtype=np.float64),
    np.ascontiguousarray(y.values, dtype=np.int64),
    n_repeats=env("PERM_N_REPEATS", 10, int), random_state=SEED,
    n_jobs=n_jobs(), scoring="roc_auc")
(pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean,
               "importance_std": perm.importances_std})
   .sort_values("importance_mean", ascending=False)
   .to_csv(os.path.join(OUT_DIR, "permutation_importance.csv"), index=False))
print(f"Done. Outputs in {OUT_DIR}")
