#!/bin/bash
#SBATCH --job-name=gls_18_classify_map
#SBATCH -n 1 -c 1 -N 1
#SBATCH -t 02:00:00
#SBATCH --mem=16G
#SBATCH --array=1-116
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# -----------------------------------------------------------------------------
# Five-class susceptibility map (Sect. 5.5): 1 = Very Low ... 5 = Very High,
# from the 0-100 index and the upper class bounds in CLASS_BREAKS.
#
# Input : $MAP_DIR/<map>/susceptibility_<TILE>.tif
# Output: $MAP_DIR/<map>_classes/classes_<TILE>.tif (no-data = MAP_NODATA)
# Usage : sbatch scripts/18_classify_map.sh [map]          (default: ensemble)
#         bash   scripts/18_classify_map.sh <map> <tile_index>
#         bash   scripts/18_classify_map.sh <map> vrt
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

MAP="${1:-ensemble}"
OUT_DIR="$MAP_DIR/${MAP}_classes"
mkdir -p "$OUT_DIR"

if [ "${2:-}" = "vrt" ]; then
    gdalbuildvrt -q "$OUT_DIR/classes_${MAP}.vrt" "$OUT_DIR"/classes_h??v??.tif
    log "Done: $OUT_DIR/classes_${MAP}.vrt"
    exit 0
fi

INDEX="${SLURM_ARRAY_TASK_ID:-${2:-}}"; [ -n "$INDEX" ] || die "tile index missing"
read -r TILE _ <<< "$(tile_record "$INDEX")"
IN="$MAP_DIR/$MAP/susceptibility_${TILE}.tif"
OUT="$OUT_DIR/classes_${TILE}.tif"
[ -f "$IN" ] || die "missing $IN"

# 1 + number of breaks below the value: 1 + (A > b1) + (A > b2) + ...
EXPR="1"
for b in $CLASS_BREAKS; do EXPR="$EXPR + (A > $b)"; done

gdal_calc.py -A "$IN" --calc="numpy.where(A == $MAP_NODATA, $MAP_NODATA, $EXPR)" \
    --NoDataValue="$MAP_NODATA" --type=Byte --outfile="$OUT" \
    --co COMPRESS=LZW --co TILED=YES --overwrite --quiet

[ -f "$OUT" ] || die "classification failed for $TILE"
log "Done: $OUT (breaks: $CLASS_BREAKS)"
