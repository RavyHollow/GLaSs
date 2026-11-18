"""SHAP feature attribution of the Ensemble model (Sect. 6.3).

The explained sample is drawn from the ensemble evaluation table (all
presences plus a random half of the pseudo-absences of each parent table,
thinned; gls_common.mixed_table), balanced 50/50 between presences and
pseudo-absences and sampled to SHAP_SAMPLE_SIZE records (gls_common.mixed_sample).
SHAP values are computed with TreeExplainer (path-dependent perturbation), in batches.

Input : $SHAP_MODEL, $WORK_DIR/{slope_constrained,unconstrained}/training_table.txt
Output: $WORK_DIR/shap/shap_values_<tag>.npy        SHAP values (class 1), n x 52
        $WORK_DIR/shap/shap_features_<tag>.csv      explained feature values, n x 52
        $WORK_DIR/shap/shap_importance_<tag>.csv    mean |SHAP| per feature
        $WORK_DIR/shap/shap_sample_index_<tag>.csv  source of each record
Usage : sbatch slurm/python_job.sh scripts/14_shap.py
"""
import gc
import os

import joblib
import numpy as np
import pandas as pd
import shap

from gls_common import env, feature_columns, mixed_sample

DATA_MODE = env("SHAP_DATA_MODE", "THINNED")
REGION = env("SHAP_REGION", "ALL")
BALANCE = env("SHAP_BALANCE", "Stratified")
BATCH = env("SHAP_BATCH_SIZE", 100, int)
OUT_DIR = os.path.join(env("WORK_DIR"), "shap")
os.makedirs(OUT_DIR, exist_ok=True)
TAG = f"{DATA_MODE}_{REGION}_{BALANCE}"


# --- Data ---------------------------------------------------------------------------
df = mixed_sample()
features = feature_columns(df)
X = df[features]
print(f"Explained matrix: {X.shape}", flush=True)

# --- SHAP ---------------------------------------------------------------------------
model = joblib.load(env("SHAP_MODEL"))
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

parts = []
for i in range(0, len(X), BATCH):
    sv = explainer.shap_values(X.iloc[i:i + BATCH], check_additivity=False)
    if isinstance(sv, list):              # older SHAP: [class 0, class 1]
        sv = sv[1]
    elif sv.ndim == 3:                    # newer SHAP: (n, features, classes)
        sv = sv[:, :, 1]
    parts.append(sv)
    print(f"  rows {min(i + BATCH, len(X))} / {len(X)}", flush=True)
    gc.collect()
shap_values = np.concatenate(parts, axis=0)
assert shap_values.shape == X.shape

np.save(os.path.join(OUT_DIR, f"shap_values_{TAG}.npy"), shap_values)
X.to_csv(os.path.join(OUT_DIR, f"shap_features_{TAG}.csv"), index=False)
(pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
   .sort_values("mean_abs_shap", ascending=False)
   .to_csv(os.path.join(OUT_DIR, f"shap_importance_{TAG}.csv"), index=False))
(df[["source"]].rename_axis("thinned_row").reset_index()
   .to_csv(os.path.join(OUT_DIR, f"shap_sample_index_{TAG}.csv"), index=False))
print(f"Done. Outputs in {OUT_DIR}")
