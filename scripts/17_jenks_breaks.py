"""Fisher-Jenks natural breaks of the global susceptibility index (Sect. 5.5).

The optimal partition into JENKS_CLASSES classes (minimum within-class sum of
squared deviations, Fisher 1958; Jenks 1967) is computed exactly on the global
histogram of the 0-100 index, i.e. on all valid pixels, each value weighted by
its pixel count. Breaks are the upper bounds (inclusive) of classes 1..k-1.
Pixel shares and goodness of variance fit (GVF) are reported for the computed
breaks and for the breaks used to classify the map (CLASS_BREAKS).

Input : $WORK_DIR/map_stats/<map>/global_histogram.txt (scripts/16_map_statistics.sh)
Output: $WORK_DIR/classes/jenks_breaks_<map>.txt
Usage : bash slurm/python_job.sh scripts/17_jenks_breaks.py [map]   (default: ensemble)
"""
import os
import sys

import numpy as np

from gls_common import env

MAP = sys.argv[1] if len(sys.argv) > 1 else "ensemble"
K = env("JENKS_CLASSES", 5, int)
USED = [int(v) for v in env("CLASS_BREAKS").split()]
NAMES = env("CLASS_NAMES").split()
HIST = os.path.join(env("WORK_DIR"), "map_stats", MAP, "global_histogram.txt")
OUT_DIR = os.path.join(env("WORK_DIR"), "classes")
os.makedirs(OUT_DIR, exist_ok=True)

hist = np.loadtxt(HIST)
values, counts = hist[:, 0], hist[:, 1]
keep = counts > 0
v, w = values[keep], counts[keep]
n = len(v)

# Weighted cumulative sums: SSD of values i..j (inclusive) in O(1)
cw = np.concatenate([[0], np.cumsum(w)])
cs = np.concatenate([[0], np.cumsum(w * v)])
cq = np.concatenate([[0], np.cumsum(w * v * v)])


def ssd(i, j):
    W, S, Q = cw[j + 1] - cw[i], cs[j + 1] - cs[i], cq[j + 1] - cq[i]
    return Q - S * S / W


# Fisher's dynamic programming: D[c, j] = min SSD of values 0..j in c classes
D = np.full((K + 1, n), np.inf)
first = np.zeros((K + 1, n), dtype=int)       # first value index of the last class
for j in range(n):
    D[1, j] = ssd(0, j)
for c in range(2, K + 1):
    for j in range(c - 1, n):
        cand = [D[c - 1, i - 1] + ssd(i, j) for i in range(c - 1, j + 1)]
        best = int(np.argmin(cand))
        D[c, j], first[c, j] = cand[best], best + c - 1

breaks, j = [], n - 1
for c in range(K, 1, -1):
    i = first[c, j]
    breaks.append(int(v[i - 1]))
    j = i - 1
breaks = sorted(breaks)

total_ssd = ssd(0, n - 1)


def describe(br):
    edges = [-np.inf] + br + [np.inf]
    shares, within = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v > lo) & (v <= hi)
        shares.append(100 * w[m].sum() / w.sum())
        idx = np.flatnonzero(m)
        within += ssd(idx[0], idx[-1]) if len(idx) else 0.0
    return shares, 1 - within / total_ssd


lines = [f"Fisher-Jenks natural breaks, {MAP} map, {K} classes",
         f"Valid pixels: {w.sum():,.0f} | values {int(v.min())}-{int(v.max())}", ""]
for label, br in (("computed", breaks), ("used (CLASS_BREAKS)", USED)):
    shares, gvf = describe(br)
    lines.append(f"Breaks {label}: {' '.join(map(str, br))}   GVF = {gvf:.6f}")
    lines.append("  " + "  ".join(f"{nm} {s:.1f}%" for nm, s in zip(NAMES, shares)))
text = "\n".join(lines)
print(text)
with open(os.path.join(OUT_DIR, f"jenks_breaks_{MAP}.txt"), "w") as f:
    f.write(text + "\n")
