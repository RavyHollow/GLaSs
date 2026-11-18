"""Predict the susceptibility of one tile with a fitted Random Forest (pyjeo).

The clipped predictors <tile_dir>/<name>.tif are stacked in the order of
<names_file> (the model column order); the class-1 probability is predicted
in batches of rows and written as Byte, int(probability x 100).

Usage: python predict_tile.py <tile_dir> <names_file> <model.joblib> <out.tif> [chunk_rows]
"""
import os
import sys

import joblib
import numpy as np
import pyjeo as pj

tile_dir, names_file, model_path, out_path = sys.argv[1:5]
chunk_rows = int(sys.argv[5]) if len(sys.argv) > 5 else 500

names = [l.strip() for l in open(names_file) if l.strip()]
paths = [os.path.join(tile_dir, f"{n}.tif") for n in names]
missing = [p for p in paths if not os.path.exists(p)]
if missing:
    sys.exit(f"Missing predictors: {missing}")

ref = pj.Jim(paths[0], otype="GDT_Float32")
ncol, nrow = ref.properties.nrOfCol(), ref.properties.nrOfRow()
print(f"Tile grid: {ncol} x {nrow}, {len(names)} predictors", flush=True)

stack = pj.Jim(ncol=ncol, nrow=nrow, otype="GDT_Float32")
stack.properties.copyGeoReference(ref)
for i, path in enumerate(paths):
    jim = pj.Jim(path, otype="GDT_Float32")
    if i == 0:
        stack.np()[:] = jim.np()[:]
    else:
        stack.geometry.stackPlane(jim)
assert stack.properties.nrOfPlane() == len(names), "stacking failed"

model = joblib.load(model_path)
assert model.n_features_in_ == len(names), "model and predictor list differ"

x = stack.np()                                   # (planes, rows, cols)
out = pj.Jim(ncol=ncol, nrow=nrow, otype="GDT_Byte")
out.properties.copyGeoReference(ref)
for r0 in range(0, nrow, chunk_rows):
    r1 = min(r0 + chunk_rows, nrow)
    block = x[:, r0:r1, :].reshape(len(names), -1).T
    proba = model.predict_proba(block)[:, 1].astype(np.float32)
    out.np()[r0:r1, :] = (proba * 100).astype(np.uint8).reshape(r1 - r0, ncol)
    print(f"  rows {r1} / {nrow}", flush=True)

out.io.write(out_path, co=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"])
print(f"Written: {out_path}")
