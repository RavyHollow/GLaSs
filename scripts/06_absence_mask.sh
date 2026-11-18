#!/bin/bash
#SBATCH --job-name=gls_06_absence_mask
#SBATCH -n 1 -c 1 -N 1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH --array=1-116
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# -----------------------------------------------------------------------------
# Per-tile mask of the area eligible for pseudo-absence sampling (Sect. 4.2).
# Eligible (1) = inside the ring buffer [INNER_RADIUS_M, OUTER_RADIUS_M] of at
# least one UGLC presence AND not permanent ice AND not permanent water,
# plus slope < SLOPE_MAX_DEG for the Slope Constrained scheme.
#
# Output: $WORK_DIR/<model>/absence_mask/absence_mask_<TILE>.tif
# Usage : sbatch scripts/06_absence_mask.sh <slope_constrained|unconstrained>
#         bash   scripts/06_absence_mask.sh <model> <tile_index>
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

MODEL="${1:-}"; check_model "$MODEL"
INDEX="${SLURM_ARRAY_TASK_ID:-${2:-}}"; [ -n "$INDEX" ] || die "tile index missing"
read -r TILE W S E N <<< "$(tile_record "$INDEX")"

OUT_DIR="$WORK_DIR/$MODEL/absence_mask"
mkdir -p "$OUT_DIR" "$WORK_DIR/tmp"
TMP=$(mktemp -d "$WORK_DIR/tmp/mask_${TILE}_XXXX")
trap 'rm -rf "$TMP"' EXIT
OUT="$OUT_DIR/absence_mask_${TILE}.tif"

log "Tile $TILE ($MODEL): W=$W S=$S E=$E N=$N"

# Extended extent: presences just outside the tile still shape the ring inside it
Wp=$(echo "$W - $TILE_MARGIN_DEG" | bc | awk '{printf "%.7f", $0}')
Sp=$(echo "$S - $TILE_MARGIN_DEG" | bc | awk '{printf "%.7f", $0}')
Ep=$(echo "$E + $TILE_MARGIN_DEG" | bc | awk '{printf "%.7f", $0}')
Np=$(echo "$N + $TILE_MARGIN_DEG" | bc | awk '{printf "%.7f", $0}')

crop() { gdal_translate -q -projwin "$W" "$N" "$E" "$S" -co COMPRESS=DEFLATE -co ZLEVEL=9 "$1" "$2"; }
crop "$ICE_MASK"     "$TMP/ice.tif"
crop "$WATER_MASK"   "$TMP/water.tif"
crop "$SLOPE_RASTER" "$TMP/slope.tif"

awk -v w="$Wp" -v s="$Sp" -v e="$Ep" -v n="$Np" \
    'NR > 1 && $2 >= w && $2 <= e && $3 >= s && $3 <= n {print $2, $3}' \
    "$UGLC_POINTS" > "$TMP/presence.txt"
N_PRES=$(wc -l < "$TMP/presence.txt")
log "Presence points in extended tile: $N_PRES"

if [ "$MODEL" = "slope_constrained" ]; then
    SLOPE_TERM="slope < $SLOPE_MAX_DEG && "
else
    SLOPE_TERM=""
fi

if [ "$N_PRES" -eq 0 ]; then
    GRASS_CMDS="
        r.external input=$TMP/ice.tif   output=ice   --overwrite
        r.external input=$TMP/water.tif output=water --overwrite
        r.external input=$TMP/slope.tif output=slope --overwrite
        g.region n=$N s=$S e=$E w=$W
        r.mapcalc 'mask = if(${SLOPE_TERM}ice == 1 && water == 1, 1, 0)' --overwrite
        r.out.gdal -f -c input=mask output=$OUT format=GTiff nodata=0 type=Byte \
            createopt='COMPRESS=DEFLATE,ZLEVEL=9' --overwrite"
else
    pkascii2ogr -f GPKG -a_srs EPSG:4326 -x 0 -y 1 -i "$TMP/presence.txt" -o "$TMP/presence.gpkg"
    GRASS_CMDS="
        r.external input=$TMP/ice.tif   output=ice   --overwrite
        r.external input=$TMP/water.tif output=water --overwrite
        r.external input=$TMP/slope.tif output=slope --overwrite
        g.region n=$Np s=$Sp e=$Ep w=$Wp
        v.in.ogr input=$TMP/presence.gpkg output=presence --overwrite
        v.to.rast input=presence output=presence_rast use=val value=1 --overwrite
        # zones: 1 = presence cell, 2 = (0, INNER], 3 = (INNER, OUTER], null = beyond OUTER
        r.buffer input=presence_rast output=buffer_zones distances=$INNER_RADIUS_M,$OUTER_RADIUS_M --overwrite
        g.region n=$N s=$S e=$E w=$W
        r.mapcalc 'ring = if(!isnull(buffer_zones) && buffer_zones == 3, 1, 0)' --overwrite
        r.mapcalc 'mask = if(${SLOPE_TERM}ice == 1 && water == 1 && ring == 1, 1, 0)' --overwrite
        r.out.gdal -f -c input=mask output=$OUT format=GTiff nodata=0 type=Byte \
            createopt='COMPRESS=DEFLATE,ZLEVEL=9' --overwrite"
fi

grass --tmp-location "$TMP/slope.tif" --exec bash -c "$GRASS_CMDS"

[ -f "$OUT" ] || die "mask not created for $TILE"
log "Done: $OUT"
