"""Landslide presences from the published UGLC point layer (Sect. 3).

Records with unreadable or out-of-range coordinates, and records falling on
no-data of the 90 m elevation layer (sea), are removed. The UGLC reliability
class is kept as RC (1 = most reliable) and used by the quality-based thinning.

Input : $UGLC_POINTS_CSV (UGLC point layer, WKT geometry), $ELEVATION_RASTER
Output: $UGLC_POINTS (ID lon lat P-A RC; P-A = 1)
Usage : bash slurm/python_job.sh scripts/01_uglc_points.py
"""
import os
import subprocess

import numpy as np
import pandas as pd

from gls_common import env

SRC = env("UGLC_POINTS_CSV")
OUT = env("UGLC_POINTS")
ID, GEOM, REL = env("UGLC_ID_FIELD"), env("UGLC_GEOM_FIELD"), env("UGLC_RELIABILITY_FIELD")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(SRC, sep=env("UGLC_SEP", "|"), usecols=[ID, GEOM, REL], dtype=str,
                 keep_default_na=False)
print(f"UGLC points: {len(df):,}")

xy = df[GEOM].str.extract(r"POINT\s*Z?\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)", expand=True)
df["lon"] = pd.to_numeric(xy[0], errors="coerce")
df["lat"] = pd.to_numeric(xy[1], errors="coerce")
df["RC"] = pd.to_numeric(df[REL], errors="coerce")
ok = df["lon"].between(-180, 180) & df["lat"].between(-90, 90) & df["RC"].notna()
print(f"Invalid coordinates or reliability: {(~ok).sum():,}")
df = df[ok].reset_index(drop=True)

elev = subprocess.run(
    ["gdallocationinfo", "-valonly", "-geoloc", env("ELEVATION_RASTER")],
    input="\n".join(f"{x:.8f} {y:.8f}" for x, y in zip(df["lon"], df["lat"])) + "\n",
    capture_output=True, text=True).stdout.split("\n")[:len(df)]
on_land = np.array([v.strip() not in ("", "-9999") for v in elev])
print(f"On no-data of the elevation layer: {(~on_land).sum():,}")
df = df[on_land]

out = pd.DataFrame({"ID": df[ID].values, "lon": df["lon"].map("{:.8f}".format).values,
                    "lat": df["lat"].map("{:.8f}".format).values, "P-A": 1,
                    "RC": df["RC"].astype(int).values})
out.to_csv(OUT, sep=" ", index=False)
print(f"Presences: {len(out):,} -> {OUT}")
