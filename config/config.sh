# shellcheck shell=bash
# =============================================================================
# GLS pipeline configuration
# Sourced by every script through scripts/common.sh. Edit the paths in
# section 1 and the environment set-up in section 2; the parameters in
# sections 3-9 are those used in the paper.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Paths
# -----------------------------------------------------------------------------
export DATA_DIR="${DATA_DIR:-$GLS_ROOT/data}"          # input data (read-only)
export WORK_DIR="${WORK_DIR:-$GLS_ROOT/work}"          # intermediate and final outputs
export PREDICTOR_DIR="${PREDICTOR_DIR:-$DATA_DIR/predictors}"

export PREDICTOR_LIST="${PREDICTOR_LIST:-$GLS_ROOT/config/predictors.tsv}"
export TILE_LIST="${TILE_LIST:-$GLS_ROOT/config/tiles.txt}"

# UGLC as published (Mancino et al., 2026, https://doi.org/10.5194/essd-18-4697-2026):
# point and polygon layers as CSV, geometry as WKT
export UGLC_POINTS_CSV="${UGLC_POINTS_CSV:-$DATA_DIR/UGLC/UGLC_point.csv}"
export UGLC_POLYGONS_CSV="${UGLC_POLYGONS_CSV:-$DATA_DIR/UGLC/UGLC_poly.csv}"
export UGLC_SEP="${UGLC_SEP:-|}"                                     # field delimiter of the CSVs
export UGLC_ID_FIELD="${UGLC_ID_FIELD:-ID}"
export UGLC_GEOM_FIELD="${UGLC_GEOM_FIELD:-WKT_GEOM}"
export UGLC_RELIABILITY_FIELD="${UGLC_RELIABILITY_FIELD:-RELIABILITY}"
export UGLC_TYPE_FIELD="${UGLC_TYPE_FIELD:-TYPE}"
export UGLC_FACTOR_FIELD="${UGLC_FACTOR_FIELD:-PHYSICAL FACTORS}"
# Derived UGLC inputs (scripts/01_uglc_points.py, scripts/02_uglc_polygons.py)
export UGLC_DIR="${UGLC_DIR:-$WORK_DIR/uglc}"
export UGLC_POINTS="${UGLC_POINTS:-$UGLC_DIR/UGLC_presence_points.txt}"
export POLYGON_MIN_DIST_M=90       # polygons with centroid closer to a presence point are excluded
export UGLC_POLYGONS_GPKG="${UGLC_POLYGONS_GPKG:-$UGLC_DIR/UGLC_poly_encoded.gpkg}"
export UGLC_POLYGONS_LAYER="${UGLC_POLYGONS_LAYER:-UGLC_poly}"

# Exclusion masks for pseudo-absence sampling (1 = not ice / not water)
export ICE_MASK="${ICE_MASK:-$DATA_DIR/masks/permanent_cryosphere_1992-2018_90m.tif}"
export WATER_MASK="${WATER_MASK:-$DATA_DIR/masks/permanent_water_bodies_1992-2018_90m.tif}"
export SLOPE_RASTER="${SLOPE_RASTER:-$PREDICTOR_DIR/slope.vrt}"
export ELEVATION_RASTER="${ELEVATION_RASTER:-$PREDICTOR_DIR/elevation.vrt}"

# Reference tiles used to build config/tiles.txt (scripts/00_make_tile_list.sh)
export TILE_REF_GLOB="${TILE_REF_GLOB:-$DATA_DIR/tiles/*_h??v??.tif}"

# -----------------------------------------------------------------------------
# 2. Software environment
# Adapt to your system. GDAL, GRASS GIS and pktools must be on PATH; the
# Python stack is described in environment.yml.
# -----------------------------------------------------------------------------
# (a gls_setup_env defined in config/config.local.sh takes precedence)
if ! declare -F gls_setup_env >/dev/null; then
    gls_setup_env() {
        # module load GDAL GRASS PKTOOLS
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook)"
            conda activate "${GLS_CONDA_ENV:-gls}" 2>/dev/null || true
        fi
    }
fi

# Python interpreter with pyjeo (map prediction, scripts/15_predict_map.sh);
# redefine in config/config.local.sh to run it in a container.
if ! declare -F gls_pyjeo_python >/dev/null; then
    gls_pyjeo_python() { python3 "$@"; }
fi

export N_JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export SEED=42

# -----------------------------------------------------------------------------
# 3. Pseudo-absence ring buffer (Sect. 4.2-4.3)
# -----------------------------------------------------------------------------
export INNER_RADIUS_M=360                   # P99 of UGLC polygon equivalent radius
export OUTER_RADIUS_M=6664                  # mean practical range, reliable semivariogram fits
export SLOPE_MAX_DEG=5                      # Slope Constrained scheme only
export MIN_ABSENCE_SPACING_DEG=0.0032393005 # ~360 m minimum spacing between pseudo-absences
export TILE_MARGIN_DEG=0.01                 # margin used to collect presences around each tile

# -----------------------------------------------------------------------------
# 4. Semivariogram analysis (Sect. 4.3)
# -----------------------------------------------------------------------------
export VARIO_N_SEEDS=5000
export VARIO_LAGS_M="0 180 360 720 1440 2880 5760 11520 23040"
export VARIO_MAX_ITER=20

# -----------------------------------------------------------------------------
# 5. Training table cleaning (Sect. 4.4)
# Rows containing any of these strings (no-data codes of the predictors) are removed.
# -----------------------------------------------------------------------------
export NODATA_PATTERNS="-9999 -9999999 72300000.000 61202000.000"

# -----------------------------------------------------------------------------
# 6. Thinning, spatial cross-validation and Random Forest (Sect. 4.4-4.5)
# -----------------------------------------------------------------------------
export THIN_KM=6.664
export BLOCK_SIZE_KM=1500
export N_FOLDS=10

# Block size selection (scripts/11_block_size_selection.py): candidate sizes and
# largest accepted share of evaluation records with a training record closer
# than OUTER_RADIUS_M across block boundaries
export BLOCK_SIZE_CANDIDATES_KM="250 500 750 1000 1250 1500 2000"
export BLOCK_PROXIMITY_MAX=0.02

export RF_N_ESTIMATORS=500
export RF_MAX_FEATURES=26
export RF_MAX_DEPTH=20
export RF_MIN_SAMPLES_LEAF=15
export RF_MIN_SAMPLES_SPLIT=50
export RF_CLASS_WEIGHT=balanced_subsample
export RF_BOOTSTRAP=true
export PERM_N_REPEATS=10

# Ensemble evaluation table (scripts/13_ensemble.py, 14_shap.py): all presences
# plus this fraction of the pseudo-absences of each parent table
export ENSEMBLE_ABSENCE_FRACTION=0.5

# -----------------------------------------------------------------------------
# 7. SHAP (Sect. 6.3)
# -----------------------------------------------------------------------------
export SHAP_MODEL="${SHAP_MODEL:-$WORK_DIR/ensemble/model/rf_model.joblib}"
export SHAP_DATA_MODE=THINNED       # tag of the output files
export SHAP_REGION=ALL              # ALL | EUROPE | AFRICA | NORTH_AMERICA | SOUTH_AMERICA | NORTH_ASIA | SOUTH_ASIA_OCEANIA
export SHAP_BALANCE=Stratified      # Stratified | P_only | None
export SHAP_SAMPLE_SIZE=10000
export SHAP_BATCH_SIZE=100

# -----------------------------------------------------------------------------
# 8. Susceptibility maps (0-100, Byte)
# -----------------------------------------------------------------------------
export MAP_NODATA=255
export MAP_DIR="${MAP_DIR:-$WORK_DIR/maps}"
# Per-tile map of each model: $MAP_DIR/<model>/susceptibility_<TILE>.tif
export PIXEL_DEG=0.000833333333333333       # 90 m grid (3 arc-seconds)
export ELEVATION_NODATA=-9999               # pixels with no elevation are set to MAP_NODATA
export PREDICT_CHUNK_ROWS=500               # tile rows per prediction batch

# -----------------------------------------------------------------------------
# 9. Susceptibility classes (Sect. 5.5)
# -----------------------------------------------------------------------------
export JENKS_CLASSES=5
export CLASS_BREAKS="31 37 55 72"           # upper bounds of classes 1-4 (inclusive)
export CLASS_NAMES="Very_Low Low Moderate High Very_High"
