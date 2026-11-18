"""Map values over UGLC polygon pixels (areal validation, Sect. 6.4).

1. Statistics of the map index over polygon pixels: all polygons, by landslide
   type and by physical factors (n, mean, median, std, min, max, p90, p95, p99).
2. Class distribution: share of map pixels and of polygon pixels in each
   susceptibility class, and their ratio (polygon share / map share).
   Classes come from CLASS_BREAKS, or from the map values themselves with
   --native (maps already stored as classes 1..5, e.g. NASA-GLS).

Input : $WORK_DIR/polygon_zonal/<map>/*_zonal.txt   (scripts/19_polygon_zonal.sh)
        $WORK_DIR/map_stats/<map>/global_histogram.txt (scripts/16_map_statistics.sh)
        $UGLC_DIR/{type,factor}_codes.csv             (scripts/02_uglc_polygons.py)
Output: $WORK_DIR/polygon_zonal/<map>_polygon_statistics.csv
        $WORK_DIR/polygon_zonal/<map>_class_distribution.csv
Usage : bash slurm/python_job.sh scripts/20_polygon_statistics.py [map] [--native]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from gls_common import env

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MAP = args[0] if args else "ensemble"
NATIVE = "--native" in sys.argv
WORK_DIR = env("WORK_DIR")
BASE = os.path.join(WORK_DIR, "polygon_zonal")
NAMES = env("CLASS_NAMES").split()
BREAKS = [int(v) for v in env("CLASS_BREAKS").split()]

files = sorted(glob.glob(os.path.join(BASE, MAP, "*_zonal.txt")))
if not files:
    sys.exit(f"No zonal files in {os.path.join(BASE, MAP)}")
pix = pd.concat((pd.read_csv(f, sep=" ", header=None, names=["value", "type_id", "factor_id"],
                             dtype="int32") for f in files if os.path.getsize(f) > 0),
                ignore_index=True)
print(f"{MAP}: {len(pix):,} polygon pixels from {len(files)} tiles")


# --- 1. Statistics by type and physical factors -----------------------------------
def stats(values):
    v = np.sort(values.to_numpy())
    n = len(v)
    return dict(n=n, mean=v.mean(), median=v[int(n * 0.50)], std=v.std(), min=v[0], max=v[-1],
                p90=v[int(n * 0.90)], p95=v[int(n * 0.95)], p99=v[int(n * 0.99)])


rows = [dict(category="ALL", label="all_polygons", **stats(pix["value"]))]
uglc_dir = env("UGLC_DIR")
for key, col in (("type", "type_id"), ("factor", "factor_id")):
    codes = pd.read_csv(os.path.join(uglc_dir, f"{key}_codes.csv"))
    labels = dict(zip(codes.iloc[:, 0], codes.iloc[:, 1]))
    for code, grp in pix.groupby(col):
        rows.append(dict(category=key.upper(), label=labels.get(code, code), **stats(grp["value"])))
table = pd.DataFrame(rows)
table.to_csv(os.path.join(BASE, f"{MAP}_polygon_statistics.csv"), index=False, float_format="%.3f")
print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


# --- 2. Class distribution: whole map vs polygon pixels ------------------------------
def to_class(values):
    values = np.asarray(values)
    return values if NATIVE else 1 + sum((values > b).astype(int) for b in BREAKS)


hist = np.loadtxt(os.path.join(WORK_DIR, "map_stats", MAP, "global_histogram.txt"))
map_counts = pd.Series(hist[:, 1], index=to_class(hist[:, 0])).groupby(level=0).sum()
poly_counts = pd.Series(to_class(pix["value"])).value_counts()

classes = range(1, len(NAMES) + 1)
dist = pd.DataFrame({
    "class": list(classes),
    "name": NAMES,
    "map_pct": [100 * map_counts.get(c, 0) / map_counts.sum() for c in classes],
    "polygon_pct": [100 * poly_counts.get(c, 0) / poly_counts.sum() for c in classes],
})
dist["ratio"] = dist["polygon_pct"] / dist["map_pct"]
dist.to_csv(os.path.join(BASE, f"{MAP}_class_distribution.csv"), index=False, float_format="%.3f")
print("\nClasses: " + ("map values" if NATIVE else f"breaks {' '.join(map(str, BREAKS))}"))
print(dist.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
