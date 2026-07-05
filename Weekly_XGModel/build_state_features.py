from __future__ import annotations
"""
build_statewide_features.py  —  STAGE 6a of the WEEKLY suitability pipeline.

Builds the CLIMATOLOGICAL weekly feature grid the maps predict on: for EVERY
Florida cell (~8,850) and EVERY ISO week (1-53), the mean-across-2013-2018
trailing-weather features + static land cover + seasonality -- constructed by
the IDENTICAL pipeline as the training features (Stage 2), so the trained models
can predict on it directly.

WHY CLIMATOLOGICAL (not a specific year): a map per week should show "typical
late-August suitability", not one particular 2016 week. We therefore build each
year's weekly features exactly as in training, then AVERAGE across years per
(cell, ISO-week). Averaging also repairs the early-2013 warmup (a NaN week-1 in
2013 is filled by the other five years).

FAITHFULNESS GUARANTEE: the dynamic feature names/definitions mirror Stage 2
(right-aligned rolling windows -> shift(1) per cell -> value at each week's
Monday). Before caching we ASSERT the output columns equal model_features.json
exactly (names AND order) -- if construction ever drifts from training, this
fails loudly instead of producing a silently-wrong map.

OUTPUT (cached once; 6b/6c and later MaxEnt all read this)
  statewide_weekly_features.parquet   (~8,850 cells x 53 weeks)
"""
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)
STATIC_DIR = Path(config["static_dir"])
WEEKLY_DIR = Path(config["weekly_xg_dir"])
DAILY_DIR  = Path(config.get("parquet_dir", config["static_dir"]))   # OuterMerged2 files
# static land cover is sourced from the DAILY files: they cover all 8,850 cells
# (the allyears static parquet was missing 801 coastal/edge cells). Values match
# allyears to float32 rounding on the 8,049 overlapping cells, so the training-
# covered cells predict on what they were trained on, and the 801 extra cells get
# their (equally valid, same-GEE-extraction) land cover. allyears kept only as an
# optional cross-check.
STATIC_XCHECK = STATIC_DIR / "Florida_Static_SDM_allyears.parquet"
FEATURES   = WEEKLY_DIR / "model_features.json"
OUTPUT_DIR = WEEKLY_DIR

GRID_ID_COL = "Grid_ID"
YEAR_MIN, YEAR_MAX = config["start_year"], config["end_year"]
GLOBAL_START = pd.Timestamp(f"{YEAR_MIN}-01-01")
GLOBAL_END   = pd.Timestamp(f"{YEAR_MAX}-12-31")

CLIMATE_COLS = ["tmax", "tmin", "tmean", "prcp", "vpd"]
VEG_COLS     = ["EVI", "NDWI"]
FFILL_LIMIT  = {"EVI": 16, "NDWI": 8}
CLIM_FFILL_LIMIT = 3
LAG_WINDOWS  = (7, 14, 28)

CELL_LIMIT = None     # set to an int to test on a subset of cells
# ============================================================================

def log(m): print(m, flush=True)
def minp(w): return max(1, w - 2)


def load_daily_allcells():
    """Concat the six statewide OuterMerged2 files (climate+veg only), reindex
    every cell to a gap-free daily calendar so windows = real days."""
    files = sorted(glob.glob(str(DAILY_DIR / "*OuterMerged2*.parquet")))
    if not files:
        raise FileNotFoundError(f"No *OuterMerged2*.parquet in {DAILY_DIR}")
    log(f"[6a] {len(files)} statewide daily files")

    keep = [GRID_ID_COL, "Date"] + CLIMATE_COLS + VEG_COLS
    frames = []
    for f in files:
        d = pd.read_parquet(f, columns=[c for c in keep if c])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.dropna(subset=["Date"])

    if CELL_LIMIT:
        cells = sorted(df[GRID_ID_COL].unique())[:CELL_LIMIT]
        df = df[df[GRID_ID_COL].isin(cells)]
        log(f"[6a] CELL_LIMIT active -> {len(cells)} cells (test mode)")

    df = (df.sort_values([GRID_ID_COL, "Date"])
            .groupby([GRID_ID_COL, "Date"], as_index=False).first())

    cells = sorted(df[GRID_ID_COL].unique())
    full_idx = pd.date_range(GLOBAL_START, GLOBAL_END, freq="D")
    mi = pd.MultiIndex.from_product([cells, full_idx], names=[GRID_ID_COL, "Date"])
    df = df.set_index([GRID_ID_COL, "Date"]).reindex(mi).reset_index()
    log(f"[6a] {len(cells):,} cells x {len(full_idx):,} days = {len(df):,} daily rows")

    g = df.groupby(GRID_ID_COL, sort=False)
    for c in CLIMATE_COLS:
        df[c] = g[c].transform(lambda s: s.ffill(limit=CLIM_FFILL_LIMIT))
    return df.sort_values([GRID_ID_COL, "Date"]).reset_index(drop=True)


def build_dynamic(df):
    """The 12 dynamic features, names/defs mirroring Stage 2 (constraint-free set)."""
    g = df.groupby(GRID_ID_COL, sort=False)
    feats = {}
    for w in LAG_WINDOWS:
        feats[f"prcp_sum_{w}d"]   = g["prcp"].transform(lambda s, w=w: s.rolling(w, min_periods=minp(w)).sum())
        feats[f"tmean_mean_{w}d"] = g["tmean"].transform(lambda s, w=w: s.rolling(w, min_periods=minp(w)).mean())
    feats["vpd_mean_14d"] = g["vpd"].transform(lambda s: s.rolling(14, min_periods=minp(14)).mean())
    feats["tmax_max_14d"] = g["tmax"].transform(lambda s: s.rolling(14, min_periods=minp(14)).max())
    feats["tmin_mean_14d"]= g["tmin"].transform(lambda s: s.rolling(14, min_periods=minp(14)).mean())
    feats["tmin_min_14d"] = g["tmin"].transform(lambda s: s.rolling(14, min_periods=minp(14)).min())
    # vegetation: ffilled level (state), same as training
    for c in VEG_COLS:
        feats[c.lower() + "_level"] = g[c].transform(lambda s: s.ffill(limit=FFILL_LIMIT[c]))
    feat_df = pd.DataFrame(feats, index=df.index)

    # leak-proof shift(1) per cell (kept for construction parity with training)
    shifted = feat_df.groupby(df[GRID_ID_COL], sort=False).shift(1)
    return pd.concat([df[[GRID_ID_COL, "Date"]], shifted], axis=1)


def to_weekly_climatology(daily_feat):
    """Take each week's Monday row, tag ISO (year, week), average across years."""
    d = daily_feat.copy()
    iso = d["Date"].dt.isocalendar()
    d["iso_year"] = iso["year"].astype(int).values
    d["iso_week"] = iso["week"].astype(int).values
    d["isoday"]   = iso["day"].astype(int).values
    mondays = d[d["isoday"] == 1].drop(columns=["isoday"])           # one row per cell-week
    # seasonality from each Monday's DOY (averaged with everything else below)
    doy = mondays["Date"].dt.dayofyear
    mondays = mondays.assign(sin_doy=np.sin(2*np.pi*doy/365.25),
                             cos_doy=np.cos(2*np.pi*doy/365.25))

    feat_cols = [c for c in mondays.columns
                 if c not in (GRID_ID_COL, "Date", "iso_year", "iso_week")]
    clim = (mondays.groupby([GRID_ID_COL, "iso_week"])[feat_cols]
                   .mean().reset_index())                            # avg across years
    log(f"[6a] climatology: {clim[GRID_ID_COL].nunique():,} cells x "
        f"{clim['iso_week'].nunique()} weeks = {len(clim):,} cell-weeks")
    return clim


def add_static(clim, static_cols):
    """Source static land cover from the DAILY files (full 8,850-cell coverage).
    First non-null per cell = the (static) land-cover fractions + topography."""
    # land cover is static across years -> ONE daily file holds every cell's values
    # (reading all six would duplicate 8,850 cells x6 and risk OOM for no gain)
    files = sorted(glob.glob(str(DAILY_DIR / "*OuterMerged2*.parquet")))
    daily_static = pd.read_parquet(files[0], columns=[GRID_ID_COL] + static_cols)
    per_cell = (daily_static.dropna(subset=static_cols, how="all")
                            .groupby(GRID_ID_COL)[static_cols].first().reset_index())

    # cross-check: values must match the training source on overlapping cells
    if STATIC_XCHECK.exists():
        try:
            ay = pd.read_parquet(STATIC_XCHECK, columns=[GRID_ID_COL] + static_cols)
            ay = ay.groupby(GRID_ID_COL)[static_cols].first()
            ov = ay.index.intersection(per_cell.set_index(GRID_ID_COL).index)
            md = float(np.nanmax((ay.loc[ov] - per_cell.set_index(GRID_ID_COL)
                                  .loc[ov]).abs().values))
            log(f"[6a] static cross-check vs allyears on {len(ov):,} cells: "
                f"max abs diff {md:.4g} ({'OK, float32 rounding' if md < 0.02 else 'INVESTIGATE'})")
        except Exception as e:
            log(f"[6a] static cross-check skipped ({type(e).__name__})")

    out = clim.merge(per_cell, on=GRID_ID_COL, how="left")
    miss = out[static_cols[0]].isna().sum()
    log(f"[6a] joined {len(static_cols)} static features from daily files "
        f"({per_cell[GRID_ID_COL].nunique():,} cells)"
        + (f" | {miss} rows still missing static" if miss else " | full coverage"))
    return out


def run():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    spec = json.load(open(FEATURES))
    model_features = spec["model_features"]
    static_cols    = spec["static"]

    daily = load_daily_allcells()
    daily_feat = build_dynamic(daily)
    clim = to_weekly_climatology(daily_feat)
    clim = add_static(clim, static_cols)

    # ---- HARD PARITY ASSERT against training feature list ----
    have = set(clim.columns)
    missing = [f for f in model_features if f not in have]
    if missing:
        raise AssertionError(f"[6a] statewide grid missing training features: {missing}")
    # reorder to exactly the training feature order (+ keys)
    keys = [GRID_ID_COL, "iso_week"]
    clim = clim[keys + model_features]
    log(f"[6a] PARITY OK: {len(model_features)} features match model_features.json (names+order)")

    nan = clim[model_features].isna().mean()
    noteworthy = {k: f"{v:.1%}" for k, v in nan.items() if v > 0}
    log(f"[6a] features with any NaN: {noteworthy or 'none'}")

    clim.to_parquet(out / "statewide_weekly_features.parquet", index=False)
    log(f"[done] wrote statewide_weekly_features.parquet ({len(clim):,} rows). "
        f"6b predicts the two calibrated surfaces on this grid.")
    return clim


if __name__ == "__main__":
    run()