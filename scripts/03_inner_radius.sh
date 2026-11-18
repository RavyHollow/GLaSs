#!/bin/bash
#SBATCH --job-name=gls_03_inner_radius
#SBATCH -n 1 -c 8 -N 1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# -----------------------------------------------------------------------------
# Inner radius of the pseudo-absence ring buffer (Sect. 4.3).
# For each UGLC polygon, the equivalent circular radius r = sqrt(A / pi) is
# computed from its area A in an equal-area projection (EPSG:6933); the 99th
# percentile of r, rounded up to the 90 m grid, gives the inner radius (360 m).
#
# Input : $UGLC_POLYGONS_GPKG (scripts/02_uglc_polygons.py)
# Output: $WORK_DIR/inner_radius/{polygon_radii_m.txt, inner_radius_summary.txt,
#                                 radius_histogram.png}
# Usage : sbatch scripts/03_inner_radius.sh
# -----------------------------------------------------------------------------

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

OUT="$WORK_DIR/inner_radius"
mkdir -p "$OUT"
GPKG="$OUT/uglc_polygons_6933.gpkg"
RADII="$OUT/polygon_radii_m.txt"
SUMMARY="$OUT/inner_radius_summary.txt"
[ -f "$UGLC_POLYGONS_GPKG" ] || die "run scripts/02_uglc_polygons.py first"

log "Reprojecting polygons to EPSG:6933"
ogr2ogr -overwrite -f GPKG "$GPKG" "$UGLC_POLYGONS_GPKG" "$UGLC_POLYGONS_LAYER" \
    -t_srs EPSG:6933 -nln UGLC_poly -lco GEOMETRY_NAME=geom

log "Computing equivalent radii (multipolygons: summed area)"
ogr2ogr -f CSV /vsistdout/ "$GPKG" -dialect SQLITE \
    -sql "SELECT ST_Area(geom) AS area_m2 FROM UGLC_poly" 2>/dev/null \
  | tail -n +2 \
  | awk -F',' 'BEGIN{pi=3.14159265358979323846} $1>0 {printf "%.4f\n", sqrt($1/pi)}' \
  > "$RADII"
log "Radii computed: $(wc -l < "$RADII")"

sort -n "$RADII" | awk '
{ val[n++] = $1; sum += $1 }
END {
    median = (n % 2 == 0) ? (val[n/2-1] + val[n/2]) / 2 : val[int(n/2)]
    p99 = val[int(n*0.99)]
    print  "UGLC polygon equivalent circular radius, r = sqrt(A/pi), A in EPSG:6933"
    print  "N polygons: " n
    printf "Min    : %10.2f m\n", val[0]
    printf "P5     : %10.2f m\n", val[int(n*0.05)]
    printf "P25    : %10.2f m\n", val[int(n*0.25)]
    printf "Mean   : %10.2f m\n", sum/n
    printf "Median : %10.2f m\n", median
    printf "P75    : %10.2f m\n", val[int(n*0.75)]
    printf "P95    : %10.2f m\n", val[int(n*0.95)]
    printf "P99    : %10.2f m\n", p99
    printf "Max    : %10.2f m\n", val[n-1]
    printf "Inner radius (P99 rounded up to the 90 m grid): %d m\n", (int(p99/90) + (p99 % 90 > 0)) * 90
}' | tee "$SUMMARY"

python3 "$GLS_ROOT/scripts/plotting/plot_inner_radius.py" "$RADII" "$OUT/radius_histogram.png"

rm -f "$GPKG"
log "Done: $SUMMARY"
