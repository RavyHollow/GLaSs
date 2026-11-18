#!/bin/bash
#SBATCH --job-name=gls_07_absence_sampling
#SBATCH -n 1 -c 1 -N 1
#SBATCH -t 24:00:00
#SBATCH --mem=128G
#SBATCH --array=1-116
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# -----------------------------------------------------------------------------
# Random pseudo-absence sampling within the eligibility mask of each tile
# (Sect. 4.2). The number of pseudo-absences per tile equals the number of
# presences in the tile, capped by the number of eligible cells; points are
# at least MIN_ABSENCE_SPACING_DEG apart and kept away from the tile edges by
# the same distance.
#
# Output: $WORK_DIR/<model>/absence_points/absence_points_<TILE>.txt
#         (ID lon lat P-A RC, with P-A = 0)
# Usage : sbatch scripts/07_absence_sampling.sh <slope_constrained|unconstrained>
#         bash   scripts/07_absence_sampling.sh <model> <tile_index>
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

MODEL="${1:-}"; check_model "$MODEL"
INDEX="${SLURM_ARRAY_TASK_ID:-${2:-}}"; [ -n "$INDEX" ] || die "tile index missing"
read -r TILE W S E N <<< "$(tile_record "$INDEX")"

MASK="$WORK_DIR/$MODEL/absence_mask/absence_mask_${TILE}.tif"
OUT_DIR="$WORK_DIR/$MODEL/absence_points"
mkdir -p "$OUT_DIR"
RAW="$OUT_DIR/random_points_${TILE}.txt"
OUT="$OUT_DIR/absence_points_${TILE}.txt"
rm -f "$RAW" "$OUT"
[ -f "$MASK" ] || die "mask not found: $MASK"

log "Tile $TILE ($MODEL)"

# Sampling extent shrunk by the minimum spacing to avoid edge effects
Nc=$(echo "$N - $MIN_ABSENCE_SPACING_DEG" | bc | awk '{printf "%.7f", $0}')
Sc=$(echo "$S + $MIN_ABSENCE_SPACING_DEG" | bc | awk '{printf "%.7f", $0}')
Ec=$(echo "$E - $MIN_ABSENCE_SPACING_DEG" | bc | awk '{printf "%.7f", $0}')
Wc=$(echo "$W + $MIN_ABSENCE_SPACING_DEG" | bc | awk '{printf "%.7f", $0}')

N_PRES=$(awk -v w="$W" -v e="$E" -v s="$S" -v n="$N" \
    'NR > 1 && $2 >= w && $2 <= e && $3 >= s && $3 <= n {c++} END {print c + 0}' "$UGLC_POINTS")

# Eligible cells = count of value 1 (value 0 is no-data)
N_VALID=$(gdalinfo -hist "$MASK" | awk '/buckets from/ {getline; print $2; exit}')
[[ "$N_VALID" =~ ^[0-9]+$ ]] || N_VALID=0

N_TARGET=$(( N_PRES < N_VALID ? N_PRES : N_VALID ))
log "Presences: $N_PRES | eligible cells: $N_VALID | target pseudo-absences: $N_TARGET"

if [ "$N_TARGET" -eq 0 ]; then
    log "Nothing to sample in $TILE"
    exit 0
fi

grass --tmp-location "$MASK" --exec bash -c "
    r.external input=$MASK output=mask_tile --overwrite
    g.region n=$Nc s=$Sc e=$Ec w=$Wc
    r.mask raster=mask_tile
    r.random.cells output=randcells distance=$MIN_ABSENCE_SPACING_DEG ncells=$N_TARGET seed=$SEED --overwrite
    r.to.vect input=randcells output=random_points type=point --overwrite
    v.out.ascii input=random_points output=$RAW format=point separator=space --overwrite
"
[ -f "$RAW" ] || die "GRASS sampling failed for $TILE"

# ID lon lat P-A RC (RC is not used for pseudo-absences)
awk '{printf "%d %s %s 0 1\n", NR, $1, $2}' "$RAW" > "$OUT"
rm -f "$RAW"

log "Pseudo-absences written: $(wc -l < "$OUT") -> $OUT"
