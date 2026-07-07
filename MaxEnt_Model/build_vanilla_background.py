from __future__ import annotations
"""
build_vanilla_background.py  —  prerequisite for the VANILLA MaxEnt model.

Samples 10,000 random background cell-weeks from across Florida and builds their
31 features by the IDENTICAL leak-proof Stage 2 pipeline used for presences, so
vanilla MaxEnt runs through the same Stage 4/5 CV harness as everything else.

WHY REAL-YEAR, NOT CLIMATOLOGICAL (Option A):
  The statewide map grid is climatological (no iso_year) -> a background point
  there can't get a (spatial_block, iso_year) CV block. Presences live in real
  (cell, year, week) space, so background must too. We therefore sample real
  (Grid_ID, iso_year, iso_week) points and build real-year trailing features.

VANILLA CONVENTION: background is drawn from the WHOLE space-time cube (all
8,850 cells x 6 years x 53 weeks) -- it characterises the AVAILABLE environment,
including places/times never surveyed. Excludes only:
  * cell-weeks that are actual presences (background != known presence)
  * early-2013 warmup weeks with no lag window

LEAK-PROOFING: features are the trailing windows ending the day BEFORE each
point's week_start Monday -- same shift as training. Not climatological.

OUTPUT (to static_dir)
  vanilla_background.parquet   (presence = 0, 31 features + CV keys, iso_year kept)
"""
import json, glob
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)
STATIC_DIR = Path(config["static_dir"])
MAXENT_DIR = Path(config["maxent_dir"])
WEEKLY_DIR = Path(config["weekly_xg_dir"])
DAILY_DIR = Path(config["parquet_dir"])   # OuterMerged2 files
PRESENCE   = WEEKLY_DIR / "weekly_model_table.parquet"      # for presences + feature parity
FEATURES   = WEEKLY_DIR / "model_features.json"
OUTPUT_DIR = MAXENT_DIR

GRID_ID_COL = "Grid_ID"
YEAR_MIN, YEAR_MAX = config["start_year"], config["end_year"]
GLOBAL_START = pd.Timestamp(f"{YEAR_MIN}-01-01")
GLOBAL_END   = pd.Timestamp(f"{YEAR_MAX}-12-31")

N_BACKGROUND = 10_000
RANDOM_STATE = 42
CELL_BATCH   = 600     # process background cells in batches to bound memory

CLIMATE_COLS = ["tmax", "tmin", "tmean", "prcp", "vpd"]
VEG_COLS     = ["EVI", "NDWI"]
FFILL_LIMIT  = {"EVI": 16, "NDWI": 8}
CLIM_FFILL_LIMIT = 3
LAG_WINDOWS  = (7, 14, 28)
STATIC_COLS  = None   # filled from model_features.json
# ============================================================================

def log(m): print(m, flush=True)
def minp(w): return max(1, w - 2)


def sample_background(presence_keys, all_cells):
    """Draw N random (Grid_ID, iso_year, iso_week) points from the full cube,
    excluding presence cell-weeks. week_start = that ISO week's Monday."""
    rng = np.random.default_rng(RANDOM_STATE)
    years = list(range(YEAR_MIN, YEAR_MAX + 1))
    weeks = list(range(1, 53))          # 53 handled below only where it exists
    picks = set()
    # oversample then filter to hit N unique non-presence points
    while len(picks) < N_BACKGROUND:
        need = (N_BACKGROUND - len(picks)) * 2
        c = rng.choice(all_cells, need)
        y = rng.choice(years, need)
        w = rng.choice(weeks, need)
        for gi, yi, wi in zip(c, y, w):
            k = (gi, int(yi), int(wi))
            if k not in presence_keys:
                picks.add(k)
                if len(picks) >= N_BACKGROUND:
                    break
    bg = pd.DataFrame(list(picks), columns=[GRID_ID_COL, "iso_year", "iso_week"])
    bg["week_start"] = [pd.Timestamp(date.fromisocalendar(y, w, 1))
                        for y, w in zip(bg["iso_year"], bg["iso_week"])]
    log(f"[bg] sampled {len(bg):,} background cell-weeks "
        f"({bg[GRID_ID_COL].nunique():,} cells, years {YEAR_MIN}-{YEAR_MAX})")
    return bg


def build_daily_batch(cells_batch, files):
    """Daily lag pipeline (Stage 2) for ONE batch of cells: concat, reindex to
    gap-free calendar, ffill, roll, shift(1). Returns shifted daily features."""
    keep = [GRID_ID_COL, "Date"] + CLIMATE_COLS + VEG_COLS
    frames = []
    for f in files:
        d = pd.read_parquet(f, columns=keep)
        d = d[d[GRID_ID_COL].isin(cells_batch)]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = (df.dropna(subset=["Date"]).sort_values([GRID_ID_COL, "Date"])
            .groupby([GRID_ID_COL, "Date"], as_index=False).first())
    cells = sorted(df[GRID_ID_COL].unique())
    idx = pd.date_range(GLOBAL_START, GLOBAL_END, freq="D")
    mi = pd.MultiIndex.from_product([cells, idx], names=[GRID_ID_COL, "Date"])
    df = df.set_index([GRID_ID_COL, "Date"]).reindex(mi).reset_index()
    g = df.groupby(GRID_ID_COL, sort=False)
    for c in CLIMATE_COLS:
        df[c] = g[c].transform(lambda s: s.ffill(limit=CLIM_FFILL_LIMIT))
    return dynamic_features(df)


def dynamic_features(df):
    g = df.groupby(GRID_ID_COL, sort=False)
    feats = {}
    for w in LAG_WINDOWS:
        feats[f"prcp_sum_{w}d"]   = g["prcp"].transform(lambda s, w=w: s.rolling(w, min_periods=minp(w)).sum())
        feats[f"tmean_mean_{w}d"] = g["tmean"].transform(lambda s, w=w: s.rolling(w, min_periods=minp(w)).mean())
    feats["vpd_mean_14d"] = g["vpd"].transform(lambda s: s.rolling(14, min_periods=minp(14)).mean())
    feats["tmax_max_14d"] = g["tmax"].transform(lambda s: s.rolling(14, min_periods=minp(14)).max())
    feats["tmin_mean_14d"]= g["tmin"].transform(lambda s: s.rolling(14, min_periods=minp(14)).mean())
    feats["tmin_min_14d"] = g["tmin"].transform(lambda s: s.rolling(14, min_periods=minp(14)).min())
    for c in VEG_COLS:
        feats[c.lower()+"_level"] = g[c].transform(lambda s: s.ffill(limit=FFILL_LIMIT[c]))
    fd = pd.DataFrame(feats, index=df.index)
    shifted = fd.groupby(df[GRID_ID_COL], sort=False).shift(1)      # leak-proof
    return pd.concat([df[[GRID_ID_COL, "Date"]], shifted], axis=1)


def run():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    spec = json.load(open(FEATURES))
    model_features = spec["model_features"]
    static_cols = spec["static"]

    pres = pd.read_parquet(PRESENCE)
    presence_keys = set(map(tuple, pres.loc[pres.presence == 1,
                        [GRID_ID_COL, "iso_year", "iso_week"]].to_numpy()))
    # background sampling frame = all cells that exist in the daily files
    all_cells = pd.read_parquet(sorted(glob.glob(str(DAILY_DIR/"*OuterMerged2*.parquet")))[0],
                                columns=[GRID_ID_COL])[GRID_ID_COL].unique()

    bg = sample_background(presence_keys, all_cells)
    bg["week_start"] = pd.to_datetime(bg["week_start"]).dt.normalize()

    # build features batch-by-batch over the cells the background touches (bounds RAM)
    files = sorted(glob.glob(str(DAILY_DIR / "*OuterMerged2*.parquet")))
    bg_cells = sorted(bg[GRID_ID_COL].unique())
    log(f"[bg] building features over {len(bg_cells):,} cells in "
        f"{-(-len(bg_cells)//CELL_BATCH)} batches of {CELL_BATCH}")
    joined = []
    for i in range(0, len(bg_cells), CELL_BATCH):
        batch = bg_cells[i:i+CELL_BATCH]
        feat = build_daily_batch(set(batch), files)
        feat["week_start"] = pd.to_datetime(feat["Date"]).dt.normalize()
        sub = bg[bg[GRID_ID_COL].isin(batch)]
        joined.append(sub.merge(feat.drop(columns=["Date"]),
                                on=[GRID_ID_COL, "week_start"], how="left"))
    bg = pd.concat(joined, ignore_index=True)

    # static land cover (first epoch per cell) from the daily files (full coverage)
    one = pd.read_parquet(sorted(glob.glob(str(DAILY_DIR/"*OuterMerged2*.parquet")))[0],
                          columns=[GRID_ID_COL] + static_cols)
    per_cell = one.groupby(GRID_ID_COL)[static_cols].first().reset_index()
    bg = bg.merge(per_cell, on=GRID_ID_COL, how="left")

    # seasonality from week_start (leak-free)
    doy = pd.to_datetime(bg["week_start"]).dt.dayofyear
    bg["sin_doy"] = np.sin(2*np.pi*doy/365.25)
    bg["cos_doy"] = np.cos(2*np.pi*doy/365.25)

    # drop warmup points with no climate window
    before = len(bg)
    dyn = [c for c in model_features if c not in static_cols + ["sin_doy","cos_doy"]]
    bg = bg[bg[dyn].notna().any(axis=1)].reset_index(drop=True)
    if len(bg) != before:
        log(f"[bg] dropped {before-len(bg)} warmup points with no climate window")

    # cell centroids for spatial CV blocks
    cent = bg[GRID_ID_COL].str.extract(r'(-?\d+\.?\d*)_(-?\d+\.?\d*)$').astype(float)
    bg["cell_lat"], bg["cell_lon"] = cent[0], cent[1]
    bg["presence"] = 0

    # parity + column order to match the model feature list
    missing = [f for f in model_features if f not in bg.columns]
    if missing:
        raise AssertionError(f"[bg] background missing features: {missing}")
    keep = [GRID_ID_COL, "iso_year", "iso_week", "week_start",
            "cell_lat", "cell_lon", "presence"] + model_features
    bg = bg[keep]

    nan = bg[model_features].isna().mean()
    log(f"[bg] feature NaN max {nan.max():.1%} | fully-complete rows {bg[model_features].notna().all(axis=1).mean():.1%}")
    bg.to_parquet(out / "vanilla_background.parquet", index=False)
    log(f"[done] wrote vanilla_background.parquet ({len(bg):,} rows). "
        f"maxent_vanilla.py = presences + this background, through the shared harness.")
    return bg


if __name__ == "__main__":
    run()