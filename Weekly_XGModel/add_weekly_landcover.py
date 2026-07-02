from __future__ import annotations
"""
add_static_features.py  —  STAGE 3c of the WEEKLY suitability pipeline.

Joins the PERSISTENT habitat features (land cover + topography) onto the weekly
model table. These explain WHERE a cell is structurally suitable, complementing
the dynamic lags that explain WHEN. They join on Grid_ID ALONE (no week_start):
they don't vary within a week, so there is no temporal leakage to guard against.

DESIGN DECISIONS (each defensible in methods):
  * PER-CLASS, NOT AGGREGATE. The parquet carries both the 4 coarse buckets
    (pct_urban/agriculture/wetland/water) AND the ~17 NLCD per-class fractions
    they are exact sums of. Feeding both = pure collinearity. We keep the
    per-class fractions and DROP the 4 aggregates, because for Cx. nigripalpus
    the sub-splits matter ecologically (emergent vs woody wetland; developed
    intensity) and the aggregates add no information their parts don't.
  * TRULY STATIC. Land cover is epoch-stepped in the source (NLCD 2013/2016/2019
    -> up to 3 values per cell). You asked for static land features, so we
    collapse to ONE row per cell (first epoch). To use the epoch signal instead,
    join on [Grid_ID, year] rather than [Grid_ID].
  * EXPLICIT ALLOWLIST. The model must see ONLY features, never trap metadata
    (n_events, n_species, other_count) or the target-in-disguise catch columns
    (mean_count/total_count/...). We write MODEL_FEATURES to a sidecar JSON so
    Stage 5 selects by allowlist, not "all columns except target".

OUTPUT (to static_dir)
  weekly_model_table.parquet   (rewritten, now with static features)
  model_features.json          (the explicit feature allowlist for Stage 5)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)

STATIC_DIR   = Path(config["static_dir"])
WEEKLY_DIR   = Path(config["weekly_xg_dir"])
MODEL_TABLE  = WEEKLY_DIR / "weekly_model_table.parquet"
STATIC_SRC   = STATIC_DIR / "Florida_Static_SDM_allyears.parquet"
OUTPUT_DIR   = WEEKLY_DIR

GRID_ID_COL  = "Grid_ID"

# the 4 aggregates to DROP (each an exact sum of per-class members already kept)
REDUNDANT_AGGREGATES = ["pct_urban", "pct_agriculture", "pct_wetland", "pct_water"]

TOPO = ["elevation", "slope"]

# the dynamic + seasonal features already in the table (from Stages 2/3b)
DYNAMIC_FEATURES = ["prcp_sum_7d", "tmean_mean_7d", "prcp_sum_14d", "tmean_mean_14d",
                    "prcp_sum_28d", "tmean_mean_28d", "vpd_mean_14d", "tmax_max_14d",
                    "tmin_mean_14d", "tmin_min_14d", "evi_level", "ndwi_level"]
SEASONAL = ["sin_doy", "cos_doy"]
# ============================================================================

def log(m): print(m, flush=True)


def load_static_per_cell():
    """One truly-static row per cell: topography + per-class land-cover fractions,
    aggregates dropped."""
    src = pd.read_parquet(STATIC_SRC)
    static_all = [c for c in src.columns if c.startswith("pct_") or c in TOPO]
    keep = [c for c in static_all if c not in REDUNDANT_AGGREGATES]

    # collapse epoch-stepped land cover -> one row per cell (first epoch)
    per_cell = src.groupby(GRID_ID_COL)[keep].first().reset_index()
    log(f"[static] {len(keep)} static features "
        f"({len(TOPO)} topo + {len(keep)-len(TOPO)} per-class land cover); "
        f"dropped aggregates {REDUNDANT_AGGREGATES}")
    log(f"[static] {len(per_cell):,} cells with static features")
    return per_cell, keep


def run():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(MODEL_TABLE)
    n_before_cols = df.shape[1]

    per_cell, static_cols = load_static_per_cell()

    # join on Grid_ID ONLY (static within a week -> no leakage)
    merged = df.merge(per_cell, on=GRID_ID_COL, how="left")

    # coverage check: every model cell should get static features
    miss = merged[static_cols[0]].isna().sum()
    covered = merged[GRID_ID_COL].nunique() - merged.loc[merged[static_cols[0]].isna(), GRID_ID_COL].nunique()
    log(f"[join] static features attached | {covered}/{merged[GRID_ID_COL].nunique()} cells covered"
        + (f" | {miss} rows missing static (kept as NaN)" if miss else ""))

    # assemble + persist the explicit allowlist
    model_features = static_cols + DYNAMIC_FEATURES + SEASONAL
    present = [f for f in model_features if f in merged.columns]
    missing = [f for f in model_features if f not in merged.columns]
    if missing:
        log(f"[warn] expected features not in table: {missing}")

    with open(out / "model_features.json", "w") as f:
        json.dump({"model_features": present,
                   "static": static_cols,
                   "dynamic": DYNAMIC_FEATURES,
                   "seasonal": SEASONAL,
                   "never_feed_to_model": ["Grid_ID","iso_year","iso_week","week_start",
                       "cell_lat","cell_lon","presence","low_n_flag","n_events",
                       "mean_log_count","mean_count","total_count","max_count",
                       "other_count","n_species"]}, f, indent=2)

    merged.to_parquet(out / "weekly_model_table.parquet", index=False)
    log(f"[final] table {df.shape} -> {merged.shape} (+{merged.shape[1]-n_before_cols} static cols)")
    log(f"[final] MODEL_FEATURES = {len(present)} "
        f"({len(static_cols)} static + {len(DYNAMIC_FEATURES)} dynamic + {len(SEASONAL)} seasonal)")
    log(f"[final] wrote weekly_model_table.parquet + model_features.json")
    log("[note] Stage 5 loads model_features.json and selects X = df[model_features] "
        "-> trap metadata & catch columns can never leak in.")
    return merged, present


if __name__ == "__main__":
    run()