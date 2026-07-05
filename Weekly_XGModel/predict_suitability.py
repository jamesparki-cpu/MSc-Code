from __future__ import annotations
"""
predict_surfaces.py  —  STAGE 6b of the WEEKLY suitability pipeline.

Produces the two calibrated statewide suitability surfaces + the fade weight the
maps render with.

MODELS (deployment fit): calibrated XGBoost and calibrated RandomForest, each fit
on ALL trapped data (CV was for honest scoring in Stage 5; the map uses every
row). Calibration is the isotonic map learned from grouped OUT-OF-FOLD
predictions (so the probability scale is honest), then applied to a base model
fit on all data -- the standard "calibrate on CV, fit on all" pattern.

FADE (space x season sampling proximity): for each grid cell-week, how well
supported is it by real surveillance? = proximity to the nearest cell that was
TRAPPED in that season (+/- SEASON_WINDOW weeks). Rendered as opacity so dense-
data regions read bold and extrapolated ones read washed-out but still visible.
  * confidence_raw : true 0-1 support (gentle exp falloff) -- kept uncapped so a
                     future "fade to 0" figure needs no re-run.
  * opacity        : confidence floored at OPACITY_FLOOR so the WHOLE state stays
                     mapped (a legibility choice; caption must say faded =
                     extrapolated / lower confidence, spatial ROC ~0.69).

AGREEMENT: |prob_xgb - prob_rf| per cell-week (threshold-free model concordance;
extends to a 3-way spread when MaxEnt is added to this same table).

OUTPUT (to static_results_dir)
  surface_predictions.parquet
    [Grid_ID, iso_week, lat, lon, prob_xgb, prob_rf, agree_abs_diff,
     confidence_raw, opacity]
  --> MaxEnt later: add prob_maxent as a column, recompute agreement as spread.
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from scipy.spatial import cKDTree
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)
STATIC_DIR  = Path(config["static_dir"])
WEEKLY_DIR  = Path(config["weekly_xg_dir"])
MODEL_TABLE = WEEKLY_DIR / "weekly_model_table.parquet"
GRID_FILE   = WEEKLY_DIR / "statewide_weekly_features.parquet"
FEATURES    = WEEKLY_DIR / "model_features.json"
OUTPUT_DIR  = Path(config["weekly_results_dir"])

GRID_ID_COL, TARGET = "Grid_ID", "presence"
N_SPATIAL_BLOCKS = 6
CALIB_FOLDS = 5
RANDOM_STATE = 42

# fade knobs (gentle falloff + floor so all Florida stays visible)
SEASON_WINDOW = 2        # +/- ISO weeks counted as "same season" for support
FALLOFF_KM    = 120.0    # gentle: confidence ~0.43 at this distance, ~0.13 at 2x
OPACITY_FLOOR = 0.25     # most-extrapolated cells still render at 25% opacity

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss", tree_method="hist")
RF_PARAMS  = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                  class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
# ============================================================================

def log(m): print(m, flush=True)
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")

def parse_lonlat(grid_ids):
    lat = np.array([float(_GID.search(str(g)).group(1)) for g in grid_ids])
    lon = np.array([float(_GID.search(str(g)).group(2)) for g in grid_ids])
    return lat, lon

def ensure_blocks(df):
    if "st_block" in df.columns: return df
    cells = df.groupby(GRID_ID_COL)[["cell_lat","cell_lon"]].first()
    cells["spatial_block"] = KMeans(N_SPATIAL_BLOCKS, random_state=RANDOM_STATE,
                                    n_init=10).fit_predict(cells[["cell_lat","cell_lon"]])
    df = df.merge(cells["spatial_block"], on=GRID_ID_COL)
    df["st_block"] = df["spatial_block"].astype(str)+"_"+df["iso_year"].astype(str)
    return df


# ----- base fitters ---------------------------------------------------------
def fit_xgb(X, y, w):
    spw = (y==0).sum()/max((y==1).sum(),1)
    m = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS); m.fit(X, y, sample_weight=w); return m

def fit_rf(X, y, w, medians):
    m = RandomForestClassifier(**RF_PARAMS); m.fit(X.fillna(medians), y, sample_weight=w); return m


def calibrated_model(name, X, y, w, groups, medians):
    """Isotonic map learned from grouped OOF preds; base model fit on ALL data."""
    oof = np.full(len(X), np.nan)
    gkf = GroupKFold(n_splits=min(CALIB_FOLDS, pd.Series(groups).nunique()))
    for tr, te in gkf.split(X, y, groups):
        if name == "xgboost":
            b = fit_xgb(X.iloc[tr], y.iloc[tr], w.iloc[tr]); p = b.predict_proba(X.iloc[te])[:,1]
        else:
            b = fit_rf(X.iloc[tr], y.iloc[tr], w.iloc[tr], medians); p = b.predict_proba(X.iloc[te].fillna(medians))[:,1]
        oof[te] = p
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    base = fit_xgb(X, y, w) if name=="xgboost" else fit_rf(X, y, w, medians)
    log(f"[6b] {name}: calibrated (isotonic on {gkf.get_n_splits()}-fold grouped OOF) + fit on all data")
    return base, iso

def predict_surface(name, base, iso, Xg, medians):
    raw = base.predict_proba(Xg if name=="xgboost" else Xg.fillna(medians))[:,1]
    return iso.predict(raw)


# ----- space x season sampling-proximity fade -------------------------------
def compute_fade(grid, sampled):
    """sampled: DataFrame of trapped cell-weeks [Grid_ID, iso_week, cell_lat, cell_lon].
    For each grid cell-week, distance to nearest cell trapped within +/-SEASON_WINDOW
    weeks -> gentle exp falloff -> floored opacity."""
    glat, glon = parse_lonlat(grid[GRID_ID_COL].to_numpy())
    coslat = np.cos(np.radians(np.nanmean(glat)))
    gx, gy = glon*coslat*111.0, glat*111.0                       # km-ish planar
    conf = np.zeros(len(grid))

    for wk in sorted(grid["iso_week"].unique()):
        # season window with wraparound on 1..53
        lo, hi = wk-SEASON_WINDOW, wk+SEASON_WINDOW
        wks = {((w-1)%53)+1 for w in range(lo, hi+1)}
        samp = sampled[sampled["iso_week"].isin(wks)]
        rows = np.where(grid["iso_week"].to_numpy()==wk)[0]
        if samp.empty:
            continue
        sx = samp["cell_lon"].to_numpy()*coslat*111.0
        sy = samp["cell_lat"].to_numpy()*111.0
        tree = cKDTree(np.c_[sx, sy])
        d, _ = tree.query(np.c_[gx[rows], gy[rows]], k=1)
        conf[rows] = np.exp(-d / FALLOFF_KM)                     # gentle falloff
    opacity = OPACITY_FLOOR + (1.0 - OPACITY_FLOOR) * conf       # floored
    return conf, opacity


def run():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    feats = json.load(open(FEATURES))["model_features"]
    df   = ensure_blocks(pd.read_parquet(MODEL_TABLE))
    grid = pd.read_parquet(GRID_FILE)

    X = df[feats]; y = df[TARGET].astype(int); w = np.sqrt(df["n_events"].clip(lower=1))
    groups = df["st_block"].to_numpy()
    medians = X.median(numeric_only=True)                        # for RF imputation
    Xg = grid[feats]
    log(f"[6b] train {len(df):,} rows | grid {len(grid):,} cell-weeks | {len(feats)} features")

    surf = grid[[GRID_ID_COL, "iso_week"]].copy()
    surf["lat"], surf["lon"] = parse_lonlat(surf[GRID_ID_COL].to_numpy())

    names = (["xgboost"] if xgb is not None else []) + ["random_forest"]
    for name in names:
        base, iso = calibrated_model(name, X, y, w, groups, medians)
        surf[f"prob_{name}"] = predict_surface(name, base, iso, Xg, medians)
        log(f"[6b] {name} surface: mean {surf[f'prob_{name}'].mean():.3f} "
            f"| range {surf[f'prob_{name}'].min():.2f}-{surf[f'prob_{name}'].max():.2f}")

    if {"prob_xgboost","prob_random_forest"} <= set(surf.columns):
        surf["agree_abs_diff"] = (surf["prob_xgboost"] - surf["prob_random_forest"]).abs()
        log(f"[6b] model agreement: mean |diff| {surf['agree_abs_diff'].mean():.3f} "
            f"(lower = more concordant)")

    sampled = df.groupby([GRID_ID_COL,"iso_week"])[["cell_lat","cell_lon"]].first().reset_index()
    surf["confidence_raw"], surf["opacity"] = compute_fade(grid, sampled)
    log(f"[6b] fade: confidence_raw {surf.confidence_raw.min():.2f}-{surf.confidence_raw.max():.2f} "
        f"-> opacity floored at {OPACITY_FLOOR} ({surf.opacity.min():.2f}-{surf.opacity.max():.2f})")

    surf.to_parquet(out / "surface_predictions.parquet", index=False)
    log(f"[done] wrote surface_predictions.parquet ({len(surf):,} rows) -> 6c renders the maps.")
    return surf


if __name__ == "__main__":
    run()