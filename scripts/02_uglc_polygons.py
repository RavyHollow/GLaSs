"""UGLC polygons for the inner ring radius and the areal validation (Sect. 4.3, 6.4).

Polygons with invalid or empty geometry, or whose centroid lies within
POLYGON_MIN_DIST_M of a presence point used for training ($UGLC_POINTS), are
excluded, so that the validation polygons are independent of the training
points. Each polygon receives a TYPE_ID (landslide type) and a FACTOR_ID (physical
factors) coded from the distinct values of the corresponding UGLC fields,
sorted alphabetically (code 1 = first value; empty values are coded as "ND").
The code tables are written next to the GeoPackage and used to label the
statistics of scripts/20_polygon_statistics.py.

Input : $UGLC_POLYGONS_CSV (UGLC polygon layer, WKT geometry), $UGLC_POINTS
Output: $UGLC_POLYGONS_GPKG (layer $UGLC_POLYGONS_LAYER: TYPE_ID, FACTOR_ID)
        $UGLC_DIR/type_codes.csv, $UGLC_DIR/factor_codes.csv
Usage : bash slurm/python_job.sh scripts/02_uglc_polygons.py
"""
import csv
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import shapely

from gls_common import env, nearest_distance_km

SRC = env("UGLC_POLYGONS_CSV")
SEP = env("UGLC_SEP", "|")
GEOM = env("UGLC_GEOM_FIELD", "WKT_GEOM")
FIELDS = {"TYPE": env("UGLC_TYPE_FIELD", "TYPE"),
          "FACTOR": env("UGLC_FACTOR_FIELD", "PHYSICAL FACTORS")}
OUT_DIR = env("UGLC_DIR")
GPKG = env("UGLC_POLYGONS_GPKG")
LAYER = env("UGLC_POLYGONS_LAYER", "UGLC_poly")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GPKG), exist_ok=True)

df = pd.read_csv(SRC, sep=SEP, usecols=[GEOM, *FIELDS.values()], dtype=str,
                 keep_default_na=False)
print(f"UGLC polygons: {len(df):,}")

geoms = shapely.from_wkt(df[GEOM].where(df[GEOM].str.strip() != "", None), on_invalid="ignore")
valid = ~shapely.is_missing(geoms) & shapely.is_valid(geoms) & ~shapely.is_empty(geoms)
print(f"Invalid or empty geometries: {(~valid).sum():,}")
df, geoms = df[valid].reset_index(drop=True), geoms[valid]

points = pd.read_csv(env("UGLC_POINTS"), sep=r"\s+", usecols=["lon", "lat"])
centroids = shapely.centroid(geoms)
cxy = np.column_stack([shapely.get_x(centroids), shapely.get_y(centroids)])
far = nearest_distance_km(cxy, points[["lon", "lat"]].to_numpy()) * 1000 >= env(
    "POLYGON_MIN_DIST_M", 90, float)
print(f"Centroid within {env('POLYGON_MIN_DIST_M')} m of a presence point: {(~far).sum():,}")
df = df[far].reset_index(drop=True)
print(f"Polygons kept: {len(df):,}")

out = pd.DataFrame({GEOM: df[GEOM]})
for key, field in FIELDS.items():
    labels = df[field].str.strip().replace("", "ND")
    codes = {lab: i for i, lab in enumerate(sorted(labels.unique()), start=1)}
    if len(codes) > 254:
        raise ValueError(f"{field}: {len(codes)} distinct values do not fit a Byte raster")
    out[f"{key}_ID"] = labels.map(codes)
    table = (pd.DataFrame({f"{key}_ID": list(codes.values()), key: list(codes.keys())})
               .assign(n_polygons=lambda t: t[key].map(labels.value_counts())))
    table.to_csv(os.path.join(OUT_DIR, f"{key.lower()}_codes.csv"), index=False)
    print(table.to_string(index=False))

with tempfile.TemporaryDirectory(dir=OUT_DIR) as tmp:
    tmp_csv = os.path.join(tmp, "uglc_polygons.csv")
    out.to_csv(tmp_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    subprocess.run([
        "ogr2ogr", "-overwrite", "-f", "GPKG", GPKG, tmp_csv,
        "-oo", f"GEOM_POSSIBLE_NAMES={GEOM}", "-oo", "KEEP_GEOM_COLUMNS=NO",
        "-oo", "AUTODETECT_TYPE=YES", "-a_srs", "EPSG:4326",
        "-nlt", "MULTIPOLYGON", "-nln", LAYER, "-lco", "GEOMETRY_NAME=geom",
    ], check=True)
print(f"Done: {GPKG}")
