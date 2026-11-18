#!/bin/bash
# Build config/tiles.txt ("<TILE> <W> <S> <E> <N>") from a set of reference
# GeoTIFF tiles whose names end in _hXXvYY.tif (20 x 20 degree tiling of the
# 90 m layers, 116 land tiles).
#
# Usage: bash scripts/00_make_tile_list.sh

GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$GLS_ROOT/scripts/common.sh"

shopt -s nullglob
# shellcheck disable=SC2206  # the glob must expand
files=( $TILE_REF_GLOB )
[ ${#files[@]} -gt 0 ] || die "no reference tiles match $TILE_REF_GLOB"

{
    echo "# TILE W S E N"
    for f in $(printf '%s\n' "${files[@]}" | sort); do
        tile=$(basename "$f" .tif | grep -oE 'h[0-9]{2}v[0-9]{2}$')
        gdalinfo "$f" | awk -v t="$tile" -F'[(),]' '
            /^Upper Left/  {w=$2; n=$3}
            /^Lower Right/ {e=$2; s=$3}
            END {gsub(/ /,"",w); gsub(/ /,"",s); gsub(/ /,"",e); gsub(/ /,"",n); print t, w, s, e, n}'
    done
} > "$TILE_LIST"

log "Tiles written: $(n_tiles) -> $TILE_LIST"
