"""Block size for the spatial cross-validation (Sect. 4.5).

For each candidate block size, the thinned records are grouped into blocks and
the blocks into N_FOLDS folds (as in scripts/12_train_cv.py). For every fold,
the share of evaluation records that have a training record closer than the
outer ring radius (OUTER_RADIUS_M, the feature autocorrelation range) is
computed; these records can only lie near a block boundary. The block size is
increased until this proximity becomes negligible (<= BLOCK_PROXIMITY_MAX).

Input : $WORK_DIR/<model>/training_table.txt (default model: slope_constrained)
Output: $WORK_DIR/cv_design/block_size_selection_<model>.csv
Usage : sbatch slurm/python_job.sh scripts/11_block_size_selection.py [model]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from gls_common import MODELS, env, load_thinned, nearest_distance_km, spatial_blocks

MODEL = sys.argv[1] if len(sys.argv) > 1 else "slope_constrained"
if MODEL not in MODELS:
    sys.exit("usage: 11_block_size_selection.py [slope_constrained|unconstrained]")

N_FOLDS = env("N_FOLDS", cast=int)
RADIUS_KM = env("OUTER_RADIUS_M", cast=float) / 1000.0
MAX_SHARE = env("BLOCK_PROXIMITY_MAX", cast=float)
SIZES = [float(v) for v in env("BLOCK_SIZE_CANDIDATES_KM").split()]
OUT_DIR = os.path.join(env("WORK_DIR"), "cv_design")
os.makedirs(OUT_DIR, exist_ok=True)

df = load_thinned(MODEL)
lonlat = df[["lon", "lat"]].to_numpy()

rows = []
for size in SIZES:
    groups = spatial_blocks(df, size)
    n_blocks = groups.nunique()
    if n_blocks < N_FOLDS:
        print(f"{size:>7.0f} km: {n_blocks} blocks < {N_FOLDS} folds, skipped")
        continue
    near, test_share = [], []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(lonlat, groups=groups):
        d = nearest_distance_km(lonlat[te], lonlat[tr])
        near.append(np.mean(d < RADIUS_KM))
        test_share.append(len(te) / len(df))
    rows.append(dict(block_size_km=size, n_blocks=n_blocks,
                     proximity_share_mean=np.mean(near), proximity_share_max=np.max(near),
                     fold_size_min=np.min(test_share), fold_size_max=np.max(test_share)))
    r = rows[-1]
    print(f"{size:>7.0f} km: {n_blocks:4d} blocks | evaluation records within "
          f"{RADIUS_KM:.3f} km of training: mean {100 * r['proximity_share_mean']:.2f} %, "
          f"max {100 * r['proximity_share_max']:.2f} % | fold size "
          f"{100 * r['fold_size_min']:.1f}-{100 * r['fold_size_max']:.1f} %", flush=True)

res = pd.DataFrame(rows)
res["accepted"] = res["proximity_share_mean"] <= MAX_SHARE
res.to_csv(os.path.join(OUT_DIR, f"block_size_selection_{MODEL}.csv"), index=False)

accepted = res[res["accepted"]]
if len(accepted):
    print(f"\nSmallest block size with mean proximity <= {100 * MAX_SHARE:.1f} %: "
          f"{accepted['block_size_km'].min():.0f} km (adopted: {env('BLOCK_SIZE_KM')} km)")
else:
    print(f"\nNo candidate meets the {100 * MAX_SHARE:.1f} % proximity limit")
