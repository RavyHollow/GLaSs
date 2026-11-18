"""Shared definitions for the Python steps of the GLS pipeline.

Parameters are read from the environment exported by config/config.sh
(run the Python steps through slurm/python_job.sh, or `source` the config).
"""
import os

import numpy as np
import pandas as pd

KM_PER_DEG = 111.32
LABEL_COL = "P-A"
RC_COL = "RC"

# Record columns of the training table, before the predictors
META_COLS = ["ID", "lon", "lat", "P-A", "RC"]

# Thematic category of each predictor (Sect. 4.1).
CATEGORIES = {
    "Lithological": ["AWCtS", "silt", "sand", "clay", "unconsolidated_soils"],
    "Hydrological": [
        "CTI", "SPI", "STI", "outlet_diff_dw_scatch", "stream_diff_dw_near",
        "stream_diff_up_farth", "stream_diff_up_near", "stream_dist_dw_near",
        "stream_dist_proximity", "stream_dist_up_farth", "stream_dist_up_near",
        "slope_curv_max_dw_cel", "slope_curv_min_dw_cel", "slope_elv_dw_cel",
        "slope_grad_dw_cel", "accumulation",
    ],
    "Geomorphometrical": [
        "elevation", "aspect_cosine", "aspect_sine", "convergence", "dev_magnitude",
        "dev_scale", "dx", "dxx", "dxy", "dy", "dyy", "eastness", "northness",
        "elev_stdev", "pcurv", "rough_magnitude", "roughness", "rough_scale",
        "slope", "tcurv", "TPI", "TRI", "VRM",
    ],
    "Climatic": ["GRE", "GPM_mean", "GPM_stdev", "SSM_mean", "frozen_soil_proba"],
    "Seismotectonic": ["PGA", "VS30"],
    "Vegetational": ["NDVI_mean"],
}
CATEGORY_COLORS = {
    "Geomorphometrical": "#E07B39",
    "Hydrological": "#3B5BA5",
    "Climatic": "#4FB0C6",
    "Lithological": "#8B5A2B",
    "Seismotectonic": "#7B4B94",
    "Vegetational": "#4C9A4C",
}
FEATURE_CATEGORY = {f: c for c, feats in CATEGORIES.items() for f in feats}


def env(name, default=None, cast=str):
    """Read a configuration value from the environment."""
    value = os.environ.get(name, default)
    if value is None:
        raise KeyError(f"{name} is not set (source config/config.sh)")
    if cast is bool:
        return str(value).lower() in ("1", "true", "yes")
    return cast(value)


def n_jobs():
    return env("N_JOBS", os.cpu_count(), int)


def load_table(path):
    """Read a whitespace-separated training table."""
    return pd.read_csv(path, sep=r"\s+")


def predictor_names():
    """The 52 predictors, in model column order (config/predictors.tsv)."""
    with open(env("PREDICTOR_LIST")) as f:
        return [l.split()[0] for l in f if l.strip() and not l.startswith("#")]


def feature_columns(df):
    """Predictor columns of a table, in model column order."""
    names = predictor_names()
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise KeyError(f"predictors missing from the table: {missing}")
    return names


def add_km_coords(df):
    """Approximate planar coordinates (km) used for thinning and blocking."""
    df["x_km"] = df["lon"] * np.cos(np.radians(df["lat"])) * KM_PER_DEG
    df["y_km"] = df["lat"] * KM_PER_DEG
    return df


def thin(df, thin_km, seed):
    """Quality-based spatial thinning (Sect. 4.4).

    On a regular grid of `thin_km`, keep one presence per cell (best UGLC
    reliability class, i.e. lowest RC, ties broken at random) and one random
    pseudo-absence per cell.
    """
    df = df.copy()
    df["cell_x"] = (df["x_km"] // thin_km).astype(int)
    df["cell_y"] = (df["y_km"] // thin_km).astype(int)
    df["cell_id"] = df["cell_x"].astype(str) + "_" + df["cell_y"].astype(str)

    presence = df[df[LABEL_COL] == 1].sample(frac=1, random_state=seed)
    presence = presence.sort_values(by=["cell_id", RC_COL], ascending=[True, True])
    presence = presence.drop_duplicates(subset="cell_id", keep="first")

    absence = (
        df[df[LABEL_COL] == 0]
        .groupby("cell_id", group_keys=False)
        .apply(lambda g: g.sample(n=1, random_state=seed))
    )
    return pd.concat([presence, absence], axis=0).reset_index(drop=True)


def spatial_blocks(df, block_km):
    """Block identifier of each record for spatial cross-validation."""
    bx = (df["x_km"] // block_km).astype(int)
    by = (df["y_km"] // block_km).astype(int)
    return bx.astype(str) + "_" + by.astype(str)


def class_summary(y, label=""):
    n = len(y)
    p = int((y == 1).sum())
    print(f"{label}{n} records | presence={p} ({100 * p / n:.1f}%) | "
          f"absence={n - p} ({100 * (n - p) / n:.1f}%)", flush=True)


MODELS = ("slope_constrained", "unconstrained")


def rf_params(**overrides):
    """Random Forest parameters of the paper (Table 2), from the configuration."""
    max_depth = env("RF_MAX_DEPTH")
    max_features = env("RF_MAX_FEATURES")
    params = dict(
        n_estimators=env("RF_N_ESTIMATORS", cast=int),
        max_features=int(max_features) if max_features.isdigit() else max_features,
        max_depth=None if max_depth == "None" else int(max_depth),
        min_samples_leaf=env("RF_MIN_SAMPLES_LEAF", cast=int),
        min_samples_split=env("RF_MIN_SAMPLES_SPLIT", cast=int),
        bootstrap=env("RF_BOOTSTRAP", cast=bool),
        class_weight=None if env("RF_CLASS_WEIGHT") == "None" else env("RF_CLASS_WEIGHT"),
        random_state=env("SEED", 42, int),
        n_jobs=n_jobs(),
    )
    params.update(overrides)
    return params


def model_table(model):
    return os.path.join(env("WORK_DIR"), model, "training_table.txt")


def load_thinned(model, verbose=True):
    """Training table of a model after quality-based thinning (Sect. 4.4)."""
    df = add_km_coords(load_table(model_table(model)))
    df = thin(df, env("THIN_KM", cast=float), env("SEED", 42, int))
    if verbose:
        class_summary(df[LABEL_COL], f"{model}, thinned: ")
    return df


def nearest_distance_km(query_lonlat, ref_lonlat):
    """Great-circle distance (km) from each query point to the nearest reference point."""
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(np.asarray(ref_lonlat)[:, ::-1]), metric="haversine")
    dist, _ = tree.query(np.radians(np.asarray(query_lonlat)[:, ::-1]), k=1)
    return dist[:, 0] * 6371.0088


def macro_region(lat, lon):
    """Coarse continental region, for regional SHAP runs."""
    if lon < -30:
        return "NORTH_AMERICA" if lat >= 15 else "SOUTH_AMERICA"
    if lon <= 60:
        return "EUROPE" if lat >= 35 else "AFRICA"
    return "NORTH_ASIA" if lat >= 30 else "SOUTH_ASIA_OCEANIA"


def mixed_table(seed=None, absence_fraction=None):
    """Evaluation table of the ensemble (Sect. 5.3, 6.3).

    All presences (shared by the two parent tables, counted once) plus a random
    ABSENCE_FRACTION of the pseudo-absences of each parent table (Slope
    Constrained and Unconstrained), after the thinning used for training.
    The column "source" names the table each pseudo-absence comes from.
    """
    seed = env("SEED", 42, int) if seed is None else seed
    frac = env("ENSEMBLE_ABSENCE_FRACTION", 0.5, float) if absence_fraction is None \
        else absence_fraction
    parts, presences = [], []
    for model in MODELS:
        df = load_thinned(model).assign(source=model)
        presences.append(df[df[LABEL_COL] == 1])
        parts.append(df[df[LABEL_COL] == 0].sample(frac=frac, random_state=seed))
    pres = pd.concat(presences).drop_duplicates(subset=["lon", "lat"]).assign(source="presence")
    table = pd.concat([pres, *parts], ignore_index=True)
    class_summary(table[LABEL_COL], "Ensemble evaluation table: ")
    return table


def mixed_sample(n=None, seed=None, region=None, balance=None):
    """SHAP sample (Sect. 6.3): drawn from the ensemble evaluation table.

    Optionally restricted to a region (SHAP_REGION), balanced between presences
    and pseudo-absences (SHAP_BALANCE), and sampled to n records (SHAP_SAMPLE_SIZE).
    """
    n = n or env("SHAP_SAMPLE_SIZE", 10000, int)
    seed = env("SEED", 42, int) if seed is None else seed
    region = region or env("SHAP_REGION", "ALL")
    balance = balance or env("SHAP_BALANCE", "Stratified")

    df = mixed_table(seed=seed)
    if region != "ALL":
        df = df[[macro_region(la, lo) == region for la, lo in zip(df["lat"], df["lon"])]]
    pres, absn = df[df[LABEL_COL] == 1], df[df[LABEL_COL] == 0]
    if balance == "Stratified":
        k = min(len(pres), len(absn))
        pres = pres.sample(n=k, random_state=seed) if len(pres) > k else pres
        absn = absn.sample(n=k, random_state=seed) if len(absn) > k else absn
        df = pd.concat([pres, absn])
    elif balance == "P_only":
        df = pres
    elif balance != "None":
        raise ValueError(f"Unknown SHAP_BALANCE: {balance}")
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    class_summary(df[LABEL_COL], f"SHAP sample ({region}, {balance}): ")
    return df
