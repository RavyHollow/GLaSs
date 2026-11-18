#!/bin/bash
#SBATCH --job-name=gls_04_variogram_sampling
#SBATCH -n 1 -c 16 -N 1
#SBATCH -t 24:00:00
#SBATCH --mem=128G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# -----------------------------------------------------------------------------
# Sampling design for the structured omnidirectional semivariograms (Sect. 4.3).
#  1. Draw $VARIO_N_SEEDS on-land UGLC presence points (anchors).
#  2. For each anchor, place one offset point at each lag of $VARIO_LAGS_M
#     (exponentially increasing distances), in a random direction.
#  3. Re-draw offsets that fall on no-data (sea) at the same lag from the
#     anchor; drop those still invalid after VARIO_MAX_ITER attempts.
#  4. Extract the 52 predictors at anchors and offsets.
#
# Output: $WORK_DIR/variogram/variogram_table.txt (ID lon lat dist <predictors>)
# Usage : sbatch scripts/04_variogram_sampling.sh
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

DIR="$WORK_DIR/variogram"
TMP="$DIR/tmp"
mkdir -p "$DIR" "$TMP"
POINTS="$DIR/variogram_points.txt"
COORDS="$TMP/coords.txt"
OUT_FILE="$DIR/variogram_table.txt"

# --- 1. Anchors: on-land presence points ------------------------------------
log "STEP 1: sampling $VARIO_N_SEEDS on-land presence anchors"
SEEDS="$TMP/anchors.txt"
POOL="$TMP/pool.txt"
: > "$SEEDS"
awk 'NR > 1 && $4 == 1 {print $1, $2, $3}' "$UGLC_POINTS" \
    | shuf --random-source=<(seeded_random "$SEED") > "$POOL"

BATCH=6000
OFFSET=0
COLLECTED=0
while [ "$COLLECTED" -lt "$VARIO_N_SEEDS" ]; do
    NEED=$((VARIO_N_SEEDS - COLLECTED))
    awk -v s=$((OFFSET + 1)) -v n=$BATCH 'NR >= s && NR < s + n' "$POOL" > "$TMP/candidates.txt"
    CAND_N=$(wc -l < "$TMP/candidates.txt")
    [ "$CAND_N" -gt 0 ] || { log "WARNING: pool exhausted at $COLLECTED anchors"; break; }
    OFFSET=$((OFFSET + CAND_N))

    awk '{print $2, $3}' "$TMP/candidates.txt" \
        | { gdallocationinfo -valonly -geoloc "$ELEVATION_RASTER" 2>/dev/null || true; } > "$TMP/elev.txt"
    paste -d " " "$TMP/candidates.txt" "$TMP/elev.txt" \
        | awk -v need="$NEED" '$4 != "-9999" && $4 != "" && NR <= need {print $1, $2, $3}' >> "$SEEDS"
    COLLECTED=$(wc -l < "$SEEDS")
    log "  anchors: $COLLECTED / $VARIO_N_SEEDS"
done

# --- 2. Offsets ---------------------------------------------------------------
log "STEP 2: generating offset points"
echo "ID lon lat dist" > "$POINTS"
awk -v distances="$VARIO_LAGS_M" -v seed="$SEED" '
BEGIN { PI = 3.14159265358979323846; n_dist = split(distances, d_arr, " "); srand(seed) }
{
    for (d = 1; d <= n_dist; d++) {
        dist_m = d_arr[d]
        if (dist_m == 0) { printf "%s %.8f %.8f %d\n", $1, $2, $3, 0; continue }
        theta   = rand() * 2 * PI
        lat_new = $3 + (dist_m / 111320.0) * cos(theta)
        lon_new = $2 + (dist_m / 111320.0) / cos($3 * PI / 180.0) * sin(theta)
        if (lat_new >  90) lat_new =  90
        if (lat_new < -90) lat_new = -90
        if (lon_new >  180) lon_new -= 360
        if (lon_new < -180) lon_new += 360
        printf "%s %.8f %.8f %d\n", $1, lon_new, lat_new, dist_m
    }
}' "$SEEDS" >> "$POINTS"
log "  rows: $(($(wc -l < "$POINTS") - 1))"

# --- 3. Re-draw offsets on no-data -------------------------------------------
# An offset on no-data (sea, outside the DEM) is re-drawn at the same lag from
# its anchor, in a new random direction, up to VARIO_MAX_ITER times; offsets
# still invalid afterwards are dropped. Anchors are valid by construction.
log "STEP 3: validating points against the elevation layer"
check_points() {
    awk 'NR > 1 {print $2, $3}' "$POINTS" \
        | { gdallocationinfo -valonly -geoloc "$ELEVATION_RASTER" 2>/dev/null || true; } > "$TMP/elev.txt"
    # one line per point: <point fields> <elevation or empty>
    paste -d " " <(awk 'NR > 1' "$POINTS") "$TMP/elev.txt" > "$TMP/checked.txt"
    awk 'NF < 5 || $5 == "-9999" {c++} END {print c + 0}' "$TMP/checked.txt"
}
ITER=0
N_INVALID=$(check_points)
while [ "$N_INVALID" -gt 0 ] && [ "$ITER" -lt "$VARIO_MAX_ITER" ]; do
    ITER=$((ITER + 1))
    log "  iteration $ITER: $N_INVALID invalid offsets re-drawn"
    head -1 "$POINTS" > "$TMP/points_new.txt"
    awk -v iter="$ITER" -v seed="$SEED" '
        BEGIN { PI = 3.14159265358979323846; srand(seed + iter) }
        FNR == NR { alon[$1] = $2; alat[$1] = $3; next }        # anchors
        {
            if (NF >= 5 && $5 != "-9999") { printf "%s %.8f %.8f %d\n", $1, $2, $3, $4; next }
            d       = ($4 == 0) ? 90 : $4
            lon0    = alon[$1]; lat0 = alat[$1]
            theta   = rand() * 2 * PI
            lat_new = lat0 + (d / 111320.0) * cos(theta)
            lon_new = lon0 + (d / 111320.0) / cos(lat0 * PI / 180.0) * sin(theta)
            if (lat_new >  90) lat_new =  90
            if (lat_new < -90) lat_new = -90
            if (lon_new >  180) lon_new -= 360
            if (lon_new < -180) lon_new += 360
            printf "%s %.8f %.8f %d\n", $1, lon_new, lat_new, $4
        }' "$SEEDS" "$TMP/checked.txt" >> "$TMP/points_new.txt"
    mv "$TMP/points_new.txt" "$POINTS"
    N_INVALID=$(check_points)
done
if [ "$N_INVALID" -gt 0 ]; then
    log "  dropping $N_INVALID offsets still invalid after $ITER iterations"
    { head -1 "$POINTS"; awk 'NF >= 5 && $5 != "-9999" {print $1, $2, $3, $4}' "$TMP/checked.txt"; } \
        > "$TMP/points_new.txt"
    mv "$TMP/points_new.txt" "$POINTS"
fi

# --- 4. Predictor extraction --------------------------------------------------
log "STEP 4: extracting predictors"
awk 'NR > 1 {print $2, $3}' "$POINTS" > "$COORDS"

extract_one() {
    # points off the raster give an empty value (and a non-zero exit status)
    { echo "$1"; gdallocationinfo -valonly -geoloc "$2" < "$COORDS" 2>/dev/null || true; } > "$TMP/col_$1.txt"
    echo "  done: $1"
}
export -f extract_one
export COORDS TMP
predictor_pairs | xargs -P "$N_JOBS" -n 2 bash -c 'extract_one "$0" "$1"'

cols=()
while read -r name; do cols+=("$TMP/col_$name.txt"); done < <(predictor_names)
paste -d " " "$POINTS" "${cols[@]}" > "$OUT_FILE"

log "Final table: $(($(wc -l < "$OUT_FILE") - 1)) rows x $(head -1 "$OUT_FILE" | awk '{print NF}') columns"
rm -rf "$TMP"
log "Done: $OUT_FILE"
