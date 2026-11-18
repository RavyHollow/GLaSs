#!/bin/bash
#SBATCH --job-name=gls_08_absence_merge
#SBATCH -n 1 -c 1 -N 1
#SBATCH -t 01:00:00
#SBATCH --mem=8G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# -----------------------------------------------------------------------------
# Merge the per-tile pseudo-absences into a single file and build a VRT of the
# eligibility masks.
#
# Output: $WORK_DIR/<model>/absence_points.txt
#         $WORK_DIR/<model>/absence_mask/absence_mask.vrt
# Usage : sbatch --dependency=afterok:<07 job id> scripts/08_absence_merge.sh <model>
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

MODEL="${1:-}"; check_model "$MODEL"
DIR="$WORK_DIR/$MODEL"

shopt -s nullglob
tiles=( "$DIR"/absence_points/absence_points_h??v??.txt )
[ ${#tiles[@]} -gt 0 ] || die "no per-tile pseudo-absence files in $DIR/absence_points"

cat "${tiles[@]}" > "$DIR/absence_points.txt"
gdalbuildvrt -q "$DIR/absence_mask/absence_mask.vrt" "$DIR"/absence_mask/absence_mask_h??v??.tif

log "Tiles merged: ${#tiles[@]}"
log "Pseudo-absences: $(wc -l < "$DIR/absence_points.txt") | presences: $(($(wc -l < "$UGLC_POINTS") - 1))"
log "Done: $DIR/absence_points.txt"
