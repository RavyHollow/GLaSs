#!/bin/bash
#SBATCH --job-name=gls_16_map_statistics
#SBATCH -n 1 -c 16 -N 1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# -----------------------------------------------------------------------------
# Histogram and summary statistics of a 0-100 susceptibility map, per tile and
# global (mean, median, std, p90, p95, min, max, pixel count).
#
# Output: $WORK_DIR/map_stats/<model>/{tile_hist/, all_tiles_stats.txt,
#                                       global_histogram.txt, global_stats.txt}
# Usage : sbatch scripts/16_map_statistics.sh <slope_constrained|unconstrained|ensemble>
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

MODEL="${1:-}"
case "$MODEL" in slope_constrained|unconstrained|ensemble) ;; *) die "unknown model '$MODEL'";; esac
VMIN="${VMIN:-0}"     # lowest index value included in the statistics
VMAX=100

IN_DIR="$MAP_DIR/$MODEL"
OUT_DIR="$WORK_DIR/map_stats/$MODEL"
mkdir -p "$OUT_DIR/tile_hist"

# Statistics from a "value count" histogram on stdin
hist_stats() {
    awk -v vmin="$VMIN" -v vmax="$VMAX" -v label="$1" '
    $1 >= vmin && $1 <= vmax && $2 > 0 { h[$1] += $2; n += $2; s += $1 * $2; s2 += $2 * $1 * $1
                               if (min == "" || $1 < min) min = $1; if ($1 > max) max = $1 }
    END {
        if (n == 0) exit
        mean = s / n; var = s2 / n - mean * mean; if (var < 0) var = 0
        for (i = vmin; i <= vmax; i++) {
            c += h[i]
            if (med == "" && c >= 0.50 * n) med = i
            if (p90 == "" && c >= 0.90 * n) p90 = i
            if (p95 == "" && c >= 0.95 * n) p95 = i
        }
        printf "%s %.6f %d %d %.6f %d %d %d %.0f\n", label, mean, med, max, sqrt(var), p90, p95, min, n
    }'
}
export -f hist_stats
export VMIN VMAX MAP_NODATA OUT_DIR

printf '%s\n' "$IN_DIR"/susceptibility_h??v??.tif | xargs -P "$N_JOBS" -n 1 bash -c '
    pkstat -i "$0" -hist -srcnodata "$MAP_NODATA" > "$OUT_DIR/tile_hist/$(basename "$0" .tif).hist"
'

HDR="tile mean median max std p90 p95 min pixel_count"
{ echo "$HDR"
  for f in "$OUT_DIR"/tile_hist/*.hist; do hist_stats "$(basename "$f" .hist)" < "$f"; done
} > "$OUT_DIR/all_tiles_stats.txt"

awk -v vmin="$VMIN" -v vmax="$VMAX" '{h[$1] += $2} END {for (i = vmin; i <= vmax; i++) print i, h[i] + 0}' \
    "$OUT_DIR"/tile_hist/*.hist > "$OUT_DIR/global_histogram.txt"
{ echo "$HDR"; hist_stats GLOBAL < "$OUT_DIR/global_histogram.txt"; } > "$OUT_DIR/global_stats.txt"

cat "$OUT_DIR/global_stats.txt"
log "Done: $OUT_DIR"
