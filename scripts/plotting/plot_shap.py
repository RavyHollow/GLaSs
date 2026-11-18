"""SHAP summary figure (Sect. 6.3): mean |SHAP| by thematic category (left) and
beeswarm (right), sharing the feature axis; plus the table of contributions by thematic category.

The beeswarm colour encodes the feature value when shap_features_<tag>.csv is
available; otherwise each point is coloured by its own SHAP value within the
feature, and the colour bar is labelled accordingly.

Usage: python scripts/plotting/plot_shap.py [tag]   (default tag: THINNED_ALL_Stratified)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gls_common import CATEGORY_COLORS, FEATURE_CATEGORY, env  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "THINNED_ALL_Stratified"
SHAP_DIR = os.path.join(env("WORK_DIR"), "shap")
path = lambda name: os.path.join(SHAP_DIR, f"{name}_{TAG}")

shap_array = np.load(path("shap_values") + ".npy")
importance = pd.read_csv(path("shap_importance") + ".csv")
features_csv = path("shap_features") + ".csv"
have_values = os.path.exists(features_csv)

if have_values:
    X = pd.read_csv(features_csv)
    feature_names = list(X.columns)
    colour_data = X.values.astype(float)
else:
    # Column order of the SHAP array = feature order of the training table.
    feature_names = [l.split()[0] for l in open(env("PREDICTOR_LIST"))
                     if l.strip() and not l.startswith("#")]
    colour_data = shap_array.copy()

assert len(feature_names) == shap_array.shape[1]
check = pd.Series(np.abs(shap_array).mean(axis=0), index=feature_names)
assert np.allclose(check[importance["feature"]].values, importance["mean_abs_shap"].values,
                   rtol=1e-5), "feature names do not match the SHAP columns"

# --- Category table ---------------------------------------------------------------
cat = (pd.DataFrame({"feature": feature_names, "mean_abs_shap": check.values})
         .assign(category=lambda d: d["feature"].map(FEATURE_CATEGORY)))
table = cat.groupby("category").agg(total=("mean_abs_shap", "sum"), n=("feature", "size"))
table["total_pct"] = 100 * table["total"] / table["total"].sum()
table["mean_per_feature_pct"] = table["total_pct"] / table["n"]
table = table.sort_values("total_pct", ascending=False)[["total_pct", "n", "mean_per_feature_pct"]]
table.to_csv(os.path.join(SHAP_DIR, f"shap_category_table_{TAG}.csv"), float_format="%.3f")
print(table.round(3))

# --- Figure -------------------------------------------------------------------------
mean_abs = check.values
order_desc = np.argsort(mean_abs)[::-1]
feat_order = [feature_names[i] for i in order_desc][::-1]
vals_order = mean_abs[order_desc][::-1]
colors = [CATEGORY_COLORS[FEATURE_CATEGORY[f]] for f in feat_order]
n = len(feature_names)

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
fig, (ax_bar, ax_bees) = plt.subplots(1, 2, figsize=(16, 0.32 * n + 2), sharey=True,
                                      gridspec_kw={"width_ratios": [1, 2], "wspace": 0.05})
y = np.arange(n)
ax_bar.barh(y, vals_order, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
ax_bar.set_yticks(y)
ax_bar.set_yticklabels(feat_order, fontsize=13)
ax_bar.set_xlabel("mean(|SHAP value|)", fontsize=13)
ax_bar.set_xscale("log")
ax_bar.invert_xaxis()
ax_bar.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
ax_bar.set_axisbelow(True)
for yy, v in zip(y, vals_order):
    ax_bar.text(v * 1.03, yy, f" {v:.3f}", va="center", ha="left", fontsize=9, color="white")

plt.sca(ax_bees)
shap.plots.beeswarm(
    shap.Explanation(values=shap_array, data=colour_data, feature_names=feature_names),
    max_display=n, order=list(order_desc), ax=ax_bees, show=False, color_bar=True,
    group_remaining_features=False, plot_size=None,
)
if not have_values:
    cb = fig.axes[-1]
    cb.set_ylabel("SHAP value of the observation\n(within each feature)", fontsize=11)
    cb.set_yticklabels(["Lowest", "Highest"])
ax_bees.set_ylabel("")
ax_bees.tick_params(axis="y", labelleft=False, left=False)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()]
fig.legend(handles, CATEGORY_COLORS.keys(), loc="lower center", ncol=6, frameon=False,
           fontsize=13, bbox_to_anchor=(0.5, 0.02))
fig.suptitle(f"SHAP feature importance — Ensemble model\n(n={shap_array.shape[0]:,} samples)",
             fontsize=13, fontweight="bold", y=0.9)
fig.tight_layout(rect=[0, 0.02, 1, 1])
out = os.path.join(SHAP_DIR, f"shap_combined_{TAG}")
fig.savefig(out + ".png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
print(f"Saved: {out}.png/.pdf")
