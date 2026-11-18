#!/bin/bash
#SBATCH --job-name=gls_15_predict_map
#SBATCH -n 1 -c 16 -N 1
#SBATCH -t 24:00:00
#SBATCH --mem=400G
#SBATCH --array=1-116
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# -----------------------------------------------------------------------------
# Susceptibility map of one model, tile by tile (Sect. 5.4). The final map is
# "ensemble" (pooled 1000-tree model, scripts/13_ensemble.py); the parent
# models can be mapped in the same way.
#  1. clip the 52 predictors to the 20 x 20 degree tile on the 90 m grid;
#     a predictor whose size differs from the tile grid is resampled to it
#     (cubic spline), keeping square 90 m pixels;
#  2. stack them in model column order and predict the class-1 probability
#     (pyjeo; scripts/predict_tile.py);
#  3. store int(probability x 100) as Byte (0-100); pixels without elevation
#     are set to MAP_NODATA.
#
# Output: $MAP_DIR/<model>/susceptibility_<TILE>.tif
# Usage : sbatch scripts/15_predict_map.sh <ensemble|slope_constrained|unconstrained>
#         bash   scripts/15_predict_map.sh <model> <tile_index>
#         bash   scripts/15_predict_map.sh <model> vrt     (after all tiles: global VRT)
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

MODEL="${1:-}"
case "$MODEL" in slope_constrained|unconstrained|ensemble) ;; *) die "unknown model '$MODEL'" ;; esac
OUT_DIR="$MAP_DIR/$MODEL"
mkdir -p "$OUT_DIR" "$WORK_DIR/tmp"

if [ "${2:-}" = "vrt" ]; then
    gdalbuildvrt -q "$OUT_DIR/susceptibility_${MODEL}.vrt" "$OUT_DIR"/susceptibility_h??v??.tif
    log "Done: $OUT_DIR/susceptibility_${MODEL}.vrt"
    exit 0
fi

INDEX="${SLURM_ARRAY_TASK_ID:-${2:-}}"; [ -n "$INDEX" ] || die "tile index missing"
read -r TILE W S E N <<< "$(tile_record "$INDEX")"
RF_MODEL="$WORK_DIR/$MODEL/model/rf_model.joblib"
OUT="$OUT_DIR/susceptibility_${TILE}.tif"
[ -f "$RF_MODEL" ] || die "model not found: $RF_MODEL"

TMP=$(mktemp -d "$WORK_DIR/tmp/predict_${MODEL}_${TILE}_XXXX")
trap 'rm -rf "$TMP"' EXIT
export GDAL_CACHEMAX=2048 GDAL_NUM_THREADS=4

# Tile grid on the 90 m pixel
read -r NX NY <<< "$(awk -v w="$W" -v s="$S" -v e="$E" -v n="$N" -v r="$PIXEL_DEG" \
    'BEGIN {printf "%d %d\n", (e - w) / r + 0.5, (n - s) / r + 0.5}')"
log "Tile $TILE ($MODEL): W=$W S=$S E=$E N=$N, grid ${NX} x ${NY}"

# --- 1. Predictors clipped to the tile ----------------------------------------
clip_one() {
    local out="$TMP/$1.tif" size
    gdal_translate -q -ot Float32 -projwin "$W" "$N" "$E" "$S" \
        -co COMPRESS=DEFLATE -co ZLEVEL=9 -co NUM_THREADS=ALL_CPUS "$2" "$out"
    size=$(gdalinfo "$out" | awk '/^Size is/ {gsub(",", ""); print $3, $4}')
    if [ "$size" != "$NX $NY" ]; then
        echo "  $1: size $size, resampled to $NX $NY"
        gdalwarp -q -ot Float32 -te "$W" "$S" "$E" "$N" -ts "$NX" "$NY" -r cubicspline \
            -co COMPRESS=DEFLATE -co ZLEVEL=9 "$out" "$out.rs.tif"
        mv "$out.rs.tif" "$out"
    fi
}
export -f clip_one
export TMP W S E N NX NY
predictor_pairs | xargs -P "$N_JOBS" -n 2 bash -c 'clip_one "$0" "$1"'

predictor_names > "$TMP/predictor_names.txt"
N_CLIPPED=$(find "$TMP" -maxdepth 1 -name '*.tif' | wc -l)
[ "$N_CLIPPED" -eq "$(wc -l < "$TMP/predictor_names.txt")" ] || die "$N_CLIPPED predictors clipped"

# --- 2. Prediction ----------------------------------------------------------------------
gls_pyjeo_python "$GLS_ROOT/scripts/predict_tile.py" \
    "$TMP" "$TMP/predictor_names.txt" "$RF_MODEL" "$TMP/prediction.tif" "$PREDICT_CHUNK_ROWS"

# --- 3. Pixels without elevation -> no-data ---------------------------------------------------
gdal_calc.py -A "$TMP/elevation.tif" -B "$TMP/prediction.tif" \
    --calc="numpy.where(A == $ELEVATION_NODATA, $MAP_NODATA, B)" \
    --NoDataValue="$MAP_NODATA" --type=Byte --outfile="$OUT" \
    --co COMPRESS=LZW --co TILED=YES --co BIGTIFF=YES --overwrite --quiet

[ -f "$OUT" ] || die "prediction failed for $TILE"
log "Done: $OUT"
