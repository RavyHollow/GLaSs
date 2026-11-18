"""Spearman rank correlation among the 52 predictors (Sect. 6.3).

Computed on the training data of both parent models (each thinned as for
training; presences shared by the two tables counted once).

Input : $WORK_DIR/{slope_constrained,unconstrained}/training_table.txt
Output: $WORK_DIR/analysis/correlation_spearman.{csv,png}
Usage : sbatch slurm/python_job.sh scripts/10_feature_correlation.py [model ...]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gls_common import (CATEGORIES, LABEL_COL, MODELS, class_summary, env, feature_columns,
                        load_thinned)

SOURCES = sys.argv[1:] or list(MODELS)
OUT_DIR = os.path.join(env("WORK_DIR"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.concat([load_thinned(m) for m in SOURCES], ignore_index=True)
is_pres = df[LABEL_COL] == 1
df = pd.concat([df[is_pres].drop_duplicates(subset=["lon", "lat"]), df[~is_pres]])
class_summary(df[LABEL_COL], "Records: ")

features = feature_columns(df)
ordered = [f for feats in CATEGORIES.values() for f in feats if f in features]
corr = df[ordered].dropna().corr(method="spearman")
corr.to_csv(os.path.join(OUT_DIR, "correlation_spearman.csv"))

fig, ax = plt.subplots(figsize=(18, 15))
im = ax.imshow(corr.mask(np.triu(np.ones_like(corr, dtype=bool))), cmap="coolwarm",
               vmin=-1, vmax=1, interpolation="nearest")
ax.set_xticks(range(len(ordered)))
ax.set_xticklabels(ordered, rotation=90, fontsize=7)
ax.set_yticks(range(len(ordered)))
ax.set_yticklabels(ordered, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman's ρ")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "correlation_spearman.png"), dpi=300)
print(f"Done. Outputs in {OUT_DIR}")
