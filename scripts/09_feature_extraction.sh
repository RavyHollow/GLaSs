#!/bin/bash
#SBATCH --job-name=gls_09_feature_extraction
#SBATCH -n 1 -c 8 -N 1
#SBATCH -t 24:00:00
#SBATCH --mem=256G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# -----------------------------------------------------------------------------
# Training table of one model (Sect. 4.1, 4.4):
#  1. merge UGLC presences and the model's pseudo-absences, with a unique ID;
#  2. extract the 52 predictors (config/predictors.tsv) at every record;
#  3. drop records with no-data in any predictor.
#
# Columns: ID lon lat P-A RC + 52 predictors.
# Output : $WORK_DIR/<model>/training_table.txt
# Usage  : sbatch scripts/09_feature_extraction.sh <slope_constrained|unconstrained>
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

MODEL="${1:-}"; check_model "$MODEL"
DIR="$WORK_DIR/$MODEL"
TMP="$DIR/extraction"
mkdir -p "$TMP"
ABSENCES="$DIR/absence_points.txt"
[ -s "$ABSENCES" ] || die "pseudo-absences not found: $ABSENCES"

RV="$TMP/IDu_x_y_pa_rv.txt"
COORDS="$TMP/coords.txt"
FULL="$TMP/training_table_raw.txt"
OUT="$DIR/training_table.txt"

HEADER="ID lon lat P-A RC"
N_META=5
N_PRED=$(predictor_names | wc -l)

# --- 1. Presence/pseudo-absence records with a unique ID ----------------------
log "Merging presences and pseudo-absences"
{
    # presences: columns selected by name from the header of $UGLC_POINTS
    awk 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
         { print $c["lon"], $c["lat"], $c["P-A"], $c["RC"] }' "$UGLC_POINTS"
    awk '{ print $2, $3, $4, $5 }' "$ABSENCES"
} | awk '{ print NR, $0 }' | sort | uniq > "$RV"
awk '{print $2, $3}' "$RV" > "$COORDS"
log "Records: $(wc -l < "$RV")"

# --- 2. Predictor extraction (one column file per predictor, resumable) -------
log "Extracting $N_PRED predictors"
extract_one() {
    local out="$TMP/col_$1.txt"
    [ -s "$out" ] && { echo "  exists: $1"; return 0; }
    # points off the raster give an empty value (and a non-zero exit status);
    # such records are removed in step 3
    { echo "$1"; gdallocationinfo -valonly -geoloc "$2" < "$COORDS" 2>/dev/null || true; } > "$out.part"
    mv "$out.part" "$out"
    echo "  done: $1"
}
export -f extract_one
export TMP COORDS
predictor_pairs | xargs -P "$N_JOBS" -n 2 bash -c 'extract_one "$0" "$1"'

cols=()
while read -r name; do cols+=("$TMP/col_$name.txt"); done < <(predictor_names)
paste -d " " <(echo "$HEADER"; cat "$RV") "${cols[@]}" | sed 's/  / /g' > "$FULL"

# --- 3. Remove records with no-data ------------------------------------------
log "Removing records with no-data"
GREP_ARGS=()
for p in $NODATA_PATTERNS; do GREP_ARGS+=(-e "$(sed 's/\./\\./g' <<< "$p")"); done

head -n 1 "$FULL" > "$OUT"
tail -n +2 "$FULL" \
    | { grep -v "${GREP_ARGS[@]}" || true; } \
    | sed 's/  */ /g; s/ *$//' \
    | awk -v nf=$((N_META + N_PRED)) 'NF == nf' >> "$OUT"

TOT=$(($(wc -l < "$FULL") - 1))
KEPT=$(($(wc -l < "$OUT") - 1))
log "Records: $TOT | kept: $KEPT | removed: $((TOT - KEPT))"
awk 'NR > 1 {n[$4]++} END {printf "Presence: %d | pseudo-absence: %d\n", n[1], n[0]}' "$OUT"
log "Done: $OUT"
