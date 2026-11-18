"""Structured omnidirectional semivariograms and practical ranges (Sect. 4.3).

For every predictor and lag h, the semivariance is computed from the
anchor-offset pairs produced by 04_variogram_sampling.sh:

    gamma(h) = 0.5 * mean[(z(anchor) - z(anchor + h))^2]

and a spherical model is fitted (Webster and Oliver, 2007). Fits whose range
reaches the upper bound allowed by the lag window are flagged as saturated
(lower bounds) and excluded; the mean practical range of the remaining
(reliable) fits is adopted as the outer ring-buffer radius.

Input : $WORK_DIR/variogram/variogram_table.txt
Output: $WORK_DIR/variogram/{variogram_ranges.txt, variogram_ranges.csv,
                              variogram_summary_ranges.png, individual/*.png}
Usage : sbatch slurm/python_job.sh scripts/05_variogram_fit.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

from gls_common import CATEGORY_COLORS, FEATURE_CATEGORY, env

WORK_DIR = env("WORK_DIR")
OUT_DIR = os.path.join(WORK_DIR, "variogram")
IN_FILE = os.path.join(OUT_DIR, "variogram_table.txt")
LAGS_M = [int(v) for v in env("VARIO_LAGS_M").split() if int(v) > 0]
LAGS_KM = [d / 1000.0 for d in LAGS_M]
LAG_MAX_KM = LAGS_KM[-1]
SAT_THRESHOLD_KM = LAG_MAX_KM * 2.9      # fit upper bound is 3 x the largest lag


def spherical(h, nugget, sill, a):
    h = np.asarray(h, dtype=float)
    return np.where(h <= a, nugget + sill * (1.5 * (h / a) - 0.5 * (h / a) ** 3),
                    nugget + sill)


def color(name):
    return CATEGORY_COLORS.get(FEATURE_CATEGORY.get(name), "grey")


# --- Load ---------------------------------------------------------------------
df = pd.read_csv(IN_FILE, sep=r"\s+").replace(-9999, np.nan)
predictors = [c for c in df.columns if c not in ("ID", "lon", "lat", "dist")]
anchor = df[df["dist"] == 0].set_index("ID")
offsets = df[df["dist"] != 0]
print(f"Anchors: {len(anchor)} | offsets: {len(offsets)} | predictors: {len(predictors)}")

# --- Empirical semivariance and spherical fit ----------------------------------
results = {}
for pred in predictors:
    lags, gammas, npairs = [], [], []
    for lag_m, lag_km in zip(LAGS_M, LAGS_KM):
        pairs = (offsets[offsets["dist"] == lag_m].set_index("ID")[[pred]]
                 .join(anchor[[pred]], lsuffix="_off", rsuffix="_anc", how="inner")
                 .dropna())
        if len(pairs) < 10:
            continue
        gammas.append(0.5 * ((pairs[f"{pred}_off"] - pairs[f"{pred}_anc"]) ** 2).mean())
        lags.append(lag_km)
        npairs.append(len(pairs))

    res = {"ok": False, "range_km": np.nan, "lag_km": lags, "gamma": gammas}
    if len(lags) >= 3:
        lags_a, gamma_a = np.array(lags), np.array(gammas)
        scale = gamma_a.max() if gamma_a.max() > 0 else 1.0
        g = gamma_a / scale
        try:
            popt, _ = curve_fit(
                spherical, lags_a, g,
                p0=[g[0] * 0.05, g.max() * 0.9, lags_a[len(lags_a) // 2]],
                bounds=([0, 1e-6, lags_a[0] * 0.5], [g.max(), g.max() * 3, lags_a[-1] * 3]),
                maxfev=20000,
            )
            res.update(ok=True, nugget=popt[0] * scale, sill=popt[1] * scale,
                       range_km=popt[2], n_pairs=npairs)
        except Exception as exc:
            print(f"{pred}: fit failed ({exc})")
    results[pred] = res
    if res["ok"]:
        print(f"{pred:<24} range={res['range_km']:.3f} km  nugget={res['nugget']:.4g}  sill={res['sill']:.4g}")

# --- Summary --------------------------------------------------------------------
fitted = {p: r["range_km"] for p, r in results.items() if r["ok"] and np.isfinite(r["range_km"])}
saturated = [p for p, v in fitted.items() if v >= SAT_THRESHOLD_KM]
reliable = np.array([v for p, v in fitted.items() if p not in saturated])
all_ranges = np.array(list(fitted.values()))

summary = {
    "n_fitted": len(all_ranges),
    "n_reliable": len(reliable),
    "n_saturated": len(saturated),
    "median_all_km": float(np.median(all_ranges)),
    "median_reliable_km": float(np.median(reliable)),
    "mean_reliable_km": float(np.mean(reliable)),          # adopted outer radius
    "p25_reliable_km": float(np.percentile(reliable, 25)),
    "p75_reliable_km": float(np.percentile(reliable, 75)),
    "max_lag_km": LAG_MAX_KM,
}
outer_km = summary["mean_reliable_km"]

rows = []
for p in predictors:
    r = results[p]
    rows.append({"feature": p, "category": FEATURE_CATEGORY.get(p, ""),
                 "range_km": r["range_km"], "nugget": r.get("nugget", np.nan),
                 "sill": r.get("sill", np.nan),
                 "status": ("failed" if not r["ok"] else
                            "saturated" if p in saturated else "reliable")})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "variogram_ranges.csv"), index=False)

with open(os.path.join(OUT_DIR, "variogram_ranges.txt"), "w") as f:
    f.write("Structured omnidirectional semivariograms, spherical model\n")
    f.write(f"Saturated fits: range >= {SAT_THRESHOLD_KM:.1f} km (lag window {LAG_MAX_KM:.2f} km)\n\n")
    f.write(f"{'Predictor':<24} {'Range (km)':>11} {'Nugget':>12} {'Sill':>12}  Status\n")
    for r in rows:
        f.write(f"{r['feature']:<24} {r['range_km']:>11.3f} {r['nugget']:>12.5g} "
                f"{r['sill']:>12.5g}  {r['status']}\n")
    f.write("\n")
    f.write(f"Reliable / saturated fits          : {summary['n_reliable']} / {summary['n_saturated']}\n")
    f.write(f"Median practical range, all fits   : {summary['median_all_km']:.3f} km\n")
    f.write(f"Median practical range, reliable   : {summary['median_reliable_km']:.3f} km\n")
    f.write(f"Mean practical range, reliable     : {outer_km:.3f} km  (outer radius)\n")
    f.write(f"IQR, reliable                      : [{summary['p25_reliable_km']:.3f}, "
            f"{summary['p75_reliable_km']:.3f}] km\n")
print(open(os.path.join(OUT_DIR, "variogram_ranges.txt")).read())

# --- Figures --------------------------------------------------------------------
ind_dir = os.path.join(OUT_DIR, "individual")
os.makedirs(ind_dir, exist_ok=True)
for p in predictors:
    r, c, sat = results[p], color(p), p in saturated
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if r["lag_km"]:
        ax.scatter(r["lag_km"], r["gamma"], s=50, color=c, zorder=4, label="Empirical γ(h)")
    if r["ok"]:
        h = np.linspace(0, max(r["lag_km"]) * 1.05, 300)
        ax.plot(h, spherical(h, r["nugget"], r["sill"], r["range_km"]), color=c, lw=2,
                label=f"Spherical, a = {r['range_km']:.2f} km" + (" (saturated)" if sat else ""))
        ax.axvline(r["range_km"], color=c, lw=1.1, ls="--", alpha=0.5)
        ax.axvline(outer_km, color="darkorange", lw=1.5, ls=":",
                   label=f"Outer radius = {outer_km:.2f} km")
        ax.legend(fontsize=9, loc="lower right")
    ax.set_title(p, fontweight="bold", color=c)
    ax.set_xlabel("Lag distance (km)")
    ax.set_ylabel("Semivariance γ(h)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(ind_dir, f"variogram_{p}.png"), dpi=200)
    plt.close(fig)

names = list(fitted.keys())
fig, ax = plt.subplots(figsize=(10, 9))
for i, p in enumerate(names):
    ax.barh(i, fitted[p], color=color(p), alpha=0.4 if p in saturated else 0.9,
            edgecolor="white", height=0.65)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.axvline(outer_km, color="darkorange", lw=2.5, ls="--",
           label=f"Mean, reliable = {outer_km:.2f} km (outer radius)")
ax.axvline(summary["median_reliable_km"], color="grey", lw=1.5, ls=":",
           label=f"Median, reliable = {summary['median_reliable_km']:.2f} km")
handles = [Patch(facecolor=v, label=k) for k, v in CATEGORY_COLORS.items()]
handles += [Patch(facecolor="grey", alpha=0.4, label="Saturated fit (excluded)")]
leg1 = ax.legend(handles=handles, fontsize=8, loc="lower right")
ax.add_artist(leg1)
ax.legend(fontsize=8, loc="upper right")
ax.set_xlabel("Practical range a (km)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "variogram_summary_ranges.png"), dpi=200)
print("Done.")
