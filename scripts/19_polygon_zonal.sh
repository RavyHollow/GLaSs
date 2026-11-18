#!/bin/bash
#SBATCH --job-name=gls_19_polygon_zonal
#SBATCH -n 1 -c 8 -N 1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH --array=1-116
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# -----------------------------------------------------------------------------
# Map values of the pixels covered by UGLC landslide polygons (areal
# validation, Sect. 6.4). The polygons are rasterised on the grid of each map
# tile (all-touched), with the landslide type (TYPE_ID) and the physical
# factors (FACTOR_ID) as pixel values. Any map stored in the same tile layout
# can be used, e.g. NASA-GLS resampled to the 90 m tile grid.
#
# Input : $UGLC_POLYGONS_GPKG (scripts/02_uglc_polygons.py)
#         $MAP_DIR/<map>/susceptibility_<TILE>.tif
# Output: $WORK_DIR/polygon_zonal/<map>/<TILE>_zonal.txt  (value type_id factor_id)
#         $WORK_DIR/polygon_zonal/{TYPE,FACTOR}/<TILE>_*.tif
# Usage : sbatch scripts/19_polygon_zonal.sh [map]          (default: ensemble)
#         bash   scripts/19_polygon_zonal.sh <map> <tile_index>
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

MAP_NAME="${1:-ensemble}"
INDEX="${SLURM_ARRAY_TASK_ID:-${2:-}}"; [ -n "$INDEX" ] || die "tile index missing"
read -r TILE _ <<< "$(tile_record "$INDEX")"

MAP="$MAP_DIR/$MAP_NAME/susceptibility_${TILE}.tif"
OUT_BASE="$WORK_DIR/polygon_zonal"
mkdir -p "$OUT_BASE/TYPE" "$OUT_BASE/FACTOR" "$OUT_BASE/$MAP_NAME" "$WORK_DIR/tmp"
OUT_TYPE="$OUT_BASE/TYPE/${TILE}_TYPE.tif"
OUT_FACTOR="$OUT_BASE/FACTOR/${TILE}_FACTOR.tif"
ZONAL="$OUT_BASE/$MAP_NAME/${TILE}_zonal.txt"
[ -f "$MAP" ] || die "missing $MAP"
[ -f "$ZONAL" ] && { log "$ZONAL exists, skipping"; exit 0; }

TMP=$(mktemp -d "$WORK_DIR/tmp/zonal_${MAP_NAME}_${TILE}_XXXX")
trap 'rm -rf "$TMP"' EXIT

read -r ULX ULY LRX LRY <<< "$(gdalinfo "$MAP" | awk '
    /Upper Left/  {gsub(/[(),]/, ""); ulx = $3; uly = $4}
    /Lower Right/ {gsub(/[(),]/, ""); lrx = $3; lry = $4}
    END {print ulx, uly, lrx, lry}')"
read -r XSIZE YSIZE <<< "$(gdalinfo "$MAP" | awk '/Size is/ {gsub(",", ""); print $3, $4; exit}')"
log "Tile $TILE: extent $ULX $LRY $LRX $ULY, size $XSIZE x $YSIZE"

rasterize() {   # <field> <output>
    [ -f "$2" ] && return 0
    gdal_rasterize -q -l "$UGLC_POLYGONS_LAYER" -a "$1" -ot Byte -a_nodata 0 -init 0 -at \
        -te "$ULX" "$LRY" "$LRX" "$ULY" -ts "$XSIZE" "$YSIZE" \
        -co COMPRESS=DEFLATE -co ZLEVEL=9 -co BIGTIFF=YES -co TILED=YES \
        "$UGLC_POLYGONS_GPKG" "$TMP/rasterized.tif"
    mv "$TMP/rasterized.tif" "$2"
}
rasterize TYPE_ID "$OUT_TYPE"
rasterize FACTOR_ID "$OUT_FACTOR"

# Map values restricted to polygon pixels
gdal_calc.py -A "$OUT_TYPE" -B "$MAP" --calc="numpy.where(A > 0, B, $MAP_NODATA)" \
    --NoDataValue="$MAP_NODATA" --outfile="$TMP/masked.tif" --type=Byte \
    --co="COMPRESS=DEFLATE" --overwrite --quiet

# Same grid for the three rasters: join the XYZ dumps line by line
gdal_translate -q -of XYZ "$TMP/masked.tif" "$TMP/susc.xyz" &
gdal_translate -q -of XYZ "$OUT_TYPE"      "$TMP/type.xyz" &
gdal_translate -q -of XYZ "$OUT_FACTOR"    "$TMP/fact.xyz" &
wait

paste -d " " <(awk '{print $3}' "$TMP/susc.xyz") \
             <(awk '{print $3}' "$TMP/type.xyz") \
             <(awk '{print $3}' "$TMP/fact.xyz") \
  | awk -v nd="$MAP_NODATA" '$1 != nd && $2 > 0 {print $1, $2, $3}' > "$ZONAL"

log "Polygon pixels: $(wc -l < "$ZONAL") -> $ZONAL"
