"""Random Forest training with spatial block cross-validation (Sect. 4.4-4.5).

Steps:
  1. quality-based spatial thinning on a THIN_KM grid;
  2. spatial blocks of BLOCK_SIZE_KM grouped into N_FOLDS folds (GroupKFold);
     per-fold AUC, and precision/recall/F1 at the 0.5 threshold;
  3. final model fitted on the whole thinned dataset;
  4. permutation importance (ROC-AUC scoring) of the final model.

Input : $WORK_DIR/<model>/training_table.txt
Output: $WORK_DIR/<model>/model/{rf_model.joblib, cv_metrics_per_fold.csv,
        cv_metrics_summary.txt, permutation_importance.csv,
        best_fold_predictions.csv, best_fold_roc.png, best_fold_pr.png}
Usage : sbatch slurm/python_job.sh scripts/12_train_cv.py <slope_constrained|unconstrained>
"""
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold

from gls_common import (LABEL_COL, MODELS, add_km_coords, class_summary, env,
                        feature_columns, load_table, model_table, n_jobs, rf_params,
                        spatial_blocks, thin)

MODEL = sys.argv[1] if len(sys.argv) > 1 else ""
if MODEL not in MODELS:
    sys.exit("usage: 12_train_cv.py <slope_constrained|unconstrained>")

WORK_DIR = env("WORK_DIR")
TABLE = model_table(MODEL)
OUT_DIR = os.path.join(WORK_DIR, MODEL, "model")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = env("SEED", 42, int)
THIN_KM = env("THIN_KM", cast=float)
BLOCK_SIZE_KM = env("BLOCK_SIZE_KM", cast=float)
N_FOLDS = env("N_FOLDS", cast=int)
RF_PARAMS = rf_params()
print(f"Model: {MODEL}\nRF parameters: {RF_PARAMS}\n"
      f"Thinning: {THIN_KM} km | blocks: {BLOCK_SIZE_KM} km | folds: {N_FOLDS}", flush=True)

# --- Data and thinning ------------------------------------------------------------
df = add_km_coords(load_table(TABLE))
features = feature_columns(df)
print(f"Predictors: {len(features)}")
class_summary(df[LABEL_COL], "Before thinning: ")
df = thin(df, THIN_KM, SEED)
class_summary(df[LABEL_COL], "After thinning:  ")

X, Y = df[features], df[LABEL_COL]
groups = spatial_blocks(df, BLOCK_SIZE_KM)
print(f"Spatial blocks: {groups.nunique()}", flush=True)

# --- Spatial block cross-validation -----------------------------------------------
fold_rows, best, best_score = [], None, -np.inf
for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, Y, groups), start=1):
    rf = RandomForestClassifier(**RF_PARAMS).fit(X.iloc[tr], Y.iloc[tr])
    proba = rf.predict_proba(X.iloc[te])[:, 1]
    pred = rf.predict(X.iloc[te])                     # 0.5 threshold
    y_te = Y.iloc[te]
    row = dict(fold=fold, n_train=len(tr), n_test=len(te),
               auc=roc_auc_score(y_te, proba), precision=precision_score(y_te, pred),
               recall=recall_score(y_te, pred), f1=f1_score(y_te, pred))
    fold_rows.append(row)
    print(f"Fold {fold}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items()
                                       if isinstance(v, float)), flush=True)

    score = 0.5 * row["auc"] + 0.5 * row["recall"]   # fold kept for diagnostic plots
    if score > best_score:
        best_score = score
        best = dict(y=y_te.values, pred=pred, proba=proba,
                    coords=df.iloc[te][["lon", "lat", "x_km", "y_km"]].copy())

folds = pd.DataFrame(fold_rows)
folds.to_csv(os.path.join(OUT_DIR, "cv_metrics_per_fold.csv"), index=False)

lines = ["Spatial block cross-validation", f"Model: {MODEL}",
         f"Folds: {N_FOLDS} | blocks: {groups.nunique()} | records: {len(df)}", ""]
for m in ("auc", "precision", "recall", "f1"):
    v = folds[m]
    lines.append(f"{m:<10} mean={v.mean():.4f}  var={v.var(ddof=0):.6f}  "
                 f"std={v.std(ddof=0):.4f}  CV%={100 * v.std(ddof=0) / v.mean():.2f}")

tn, fp, fn, tp = confusion_matrix(best["y"], best["pred"]).ravel()
spec = tn / (tn + fp)
rec = recall_score(best["y"], best["pred"])
lines += ["", "Best fold (diagnostics)",
          f"TN={tn} FP={fp} FN={fn} TP={tp}",
          f"AUC={roc_auc_score(best['y'], best['proba']):.4f}  "
          f"accuracy={(tp + tn) / (tp + tn + fp + fn):.4f}  "
          f"balanced_accuracy={(rec + spec) / 2:.4f}  specificity={spec:.4f}"]
summary = "\n".join(lines)
print("\n" + summary, flush=True)
with open(os.path.join(OUT_DIR, "cv_metrics_summary.txt"), "w") as f:
    f.write(summary + "\n")

# --- Final model on the whole thinned dataset ----------------------------------------
print("\nFitting final model on the whole thinned dataset...", flush=True)
rf_final = RandomForestClassifier(**RF_PARAMS, oob_score=True).fit(X, Y)
joblib.dump(rf_final, os.path.join(OUT_DIR, "rf_model.joblib"))
print(f"OOB score: {rf_final.oob_score_:.4f}")

# --- Permutation importance -------------------------------------------------------------
print("Computing permutation importance...", flush=True)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
perm = permutation_importance(
    rf_final,
    np.ascontiguousarray(X.values, dtype=np.float64),
    np.ascontiguousarray(Y.values, dtype=np.int64),
    n_repeats=env("PERM_N_REPEATS", 10, int), random_state=SEED,
    n_jobs=n_jobs(), scoring="roc_auc",
)
(pd.DataFrame({"feature": features, "importance_mean": perm.importances_mean,
               "importance_std": perm.importances_std})
   .sort_values("importance_mean", ascending=False)
   .to_csv(os.path.join(OUT_DIR, "permutation_importance.csv"), index=False))

# --- Best-fold diagnostics ----------------------------------------------------------------
pred_df = best["coords"].assign(true_label=best["y"], pred_label=best["pred"],
                                pred_proba=best["proba"])
pred_df.to_csv(os.path.join(OUT_DIR, "best_fold_predictions.csv"), index=False)

prec_c, rec_c, _ = precision_recall_curve(best["y"], best["proba"])
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_c, prec_c)
ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision",
       title="Precision-recall curve (best fold)")
fig.savefig(os.path.join(OUT_DIR, "best_fold_pr.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

fpr, tpr, _ = roc_curve(best["y"], best["proba"])
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="steelblue", lw=2,
        label=f"ROC (AUC = {roc_auc_score(best['y'], best['proba']):.3f})")
ax.plot([0, 1], [0, 1], color="grey", lw=1.2, ls="--", label="Random classifier")
ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False positive rate",
       ylabel="True positive rate", title="ROC curve (best fold)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.savefig(os.path.join(OUT_DIR, "best_fold_roc.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Done. Outputs in {OUT_DIR}")
