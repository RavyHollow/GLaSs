"""Histogram of UGLC polygon equivalent radii with the P99 inner radius.

Usage: python plot_inner_radius.py <polygon_radii_m.txt> <output.png>
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

radii_file, out_png = sys.argv[1], sys.argv[2]
GRID_M = 90

radii = np.loadtxt(radii_file)
radii = radii[radii > 0]
p50, p99 = np.percentile(radii, [50, 99])
inner = math.ceil(p99 / GRID_M) * GRID_M
print(f"N={len(radii):,}  P50={p50:.1f} m  P99={p99:.1f} m  inner radius={inner} m")

x_cap = p99 + 100
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(radii[radii <= x_cap], bins=80, color="#4575b4", edgecolor="white",
        linewidth=0.4, alpha=0.85, label="Polygon equivalent radius")
ax.axvline(p99, color="#d62728", lw=2.2, ls="--", label=f"P99 = {p99:.0f} m")
ax.axvline(p50, color="darkorange", lw=1.5, ls=":", label=f"Median = {p50:.0f} m")
ax.axvline(inner, color="#d62728", lw=1.0, alpha=0.35,
           label=f"P99 rounded up to the {GRID_M} m grid = {inner} m")
ax.set_xlabel("Equivalent circular radius (m)")
ax.set_ylabel("Count")
ax.set_title(f"UGLC polygon equivalent radius (N = {len(radii):,}, EPSG:6933)")
ax.set_xlim(0, x_cap)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3, lw=0.6)
fig.tight_layout()
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Saved: {out_png}")
