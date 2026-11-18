# shellcheck shell=bash
# Shared helpers for the bash steps of the GLS pipeline.
# Usage (first lines of each script):
#   GLS_ROOT="${GLS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
#   source "$GLS_ROOT/scripts/common.sh"
# Under SLURM, submit jobs from the repository root.

set -euo pipefail
export LC_ALL=C
export GLS_ROOT

# Optional site-specific paths and environment (not tracked by git), read
# before config.sh so that its values replace the defaults
# shellcheck disable=SC1091
[ -f "$GLS_ROOT/config/config.local.sh" ] && source "$GLS_ROOT/config/config.local.sh"
source "$GLS_ROOT/config/config.sh"
gls_setup_env

log() { echo "$(date +'%Y-%m-%d %H:%M:%S') - $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# Validate the model name used to select the pseudo-absence scheme.
check_model() {
    case "${1:-}" in
        slope_constrained|unconstrained) ;;
        *) die "model must be 'slope_constrained' or 'unconstrained' (got '${1:-}')" ;;
    esac
}

# Tile record "<TILE> <W> <S> <E> <N>" for a 1-based index of config/tiles.txt.
tile_record() {
    [ -s "$TILE_LIST" ] || die "tile list not found: $TILE_LIST (run scripts/00_make_tile_list.sh)"
    local rec
    rec=$(grep -v '^#' "$TILE_LIST" | awk 'NF' | sed -n "${1}p")
    [ -n "$rec" ] || die "no tile with index $1 in $TILE_LIST"
    echo "$rec"
}

n_tiles() { grep -v '^#' "$TILE_LIST" | awk 'NF' | wc -l; }

# Reproducible byte stream for `shuf --random-source` (GNU coreutils recipe).
seeded_random() {
    openssl enc -aes-256-ctr -pass pass:"$1" -nosalt </dev/zero 2>/dev/null
}

# Predictor list as "<name> <path>" pairs, with relative paths resolved.
predictor_pairs() {
    grep -v '^#' "$PREDICTOR_LIST" | awk 'NF' | while read -r name path; do
        case "$path" in /*) ;; *) path="$PREDICTOR_DIR/$path" ;; esac
        echo "$name $path"
    done
}

predictor_names() { grep -v '^#' "$PREDICTOR_LIST" | awk 'NF {print $1}'; }
