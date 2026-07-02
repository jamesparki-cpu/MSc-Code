from __future__ import annotations
"""
build_weekly_model_table.py  —  STAGE 3b of the WEEKLY suitability pipeline.

Pure ASSEMBLY. No new features, no recomputation. It:
  1. UNIONs the two labelled cell-week sets:
        nigripalpus_presence_by_cellweek.parquet   (presence = 1, 16,771 rows)
        nigripalpus_absence_by_cellweek.parquet    (presence = 0,  4,849 rows)
     -> one table with a real 0/1 target the classifier can learn to separate.
  2. LEFT-JOINs the leak-proof lag features from
        daily_features_master.parquet
     on [Grid_ID, week_start] (master's Date == the cell-week's Monday), so the
     features attached are the trailing windows that ENDED the day before the
     week -- identical treatment for positives and negatives.
  3. Adds seasonality (sin/cos of day-of-year of week_start) -- known at
     prediction time, so it is NOT lagged/leaky.
  4. Drops the early-2013 warmup cell-weeks that have no prior climate to
     window over (no leakage created; those rows simply have no valid features).

OUTPUT (to static_dir)
  weekly_model_table.parquet   <- the single table Stage 5 trains on.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)

STATIC_DIR = Path(config["static_dir"])
WEEKLY_DIR = Path(config["weekly_xg_dir"])
PRESENCE   = WEEKLY_DIR / "nigripalpus_presence_by_cellweek.parquet"
ABSENCE    = WEEKLY_DIR / "nigripalpus_absence_by_cellweek.parquet"
MASTER     = WEEKLY_DIR / "daily_features_master.parquet"
OUTPUT_DIR = WEEKLY_DIR

GRID_ID_COL = "Grid_ID"

# the lag features to pull off the master (everything except its join keys)
JOIN_KEYS = [GRID_ID_COL, "week_start"]
# ============================================================================

def log(m): print(m, flush=True)


def load_labelled_union():
    """Stack presence + absence. Schemas share the same 13 core columns; the two
    absence-only diagnostics (other_count, n_species) become NaN for presence
    rows, which is fine -- they are diagnostics, not model inputs."""
    pres = pd.read_parquet(PRESENCE)
    absc = pd.read_parquet(ABSENCE)
    for df in (pres, absc):
        df["week_start"] = pd.to_datetime(df["week_start"]).dt.normalize()

    union = pd.concat([pres, absc], ignore_index=True, sort=False)
    log(f"[union] presence {len(pres):,} + absence {len(absc):,} = {len(union):,} cell-weeks "
        f"({len(pres)/(len(pres)+len(absc)):.1%} / {len(absc)/(len(pres)+len(absc)):.1%})")

    # safety: a (cell, year, week) must not appear in both classes
    dup = union.duplicated([GRID_ID_COL, "iso_year", "iso_week"]).sum()
    if dup:
        raise ValueError(f"[union] {dup} cell-weeks appear in BOTH classes -- not disjoint!")
    log(f"[union] disjoint check passed (0 cell-weeks in both classes)")
    return union


def attach_features(union):
    """Left-join the daily master's lag features at Date == week_start."""
    master = pd.read_parquet(MASTER)
    master["week_start"] = pd.to_datetime(master["Date"]).dt.normalize()
    feat_cols = [c for c in master.columns if c not in (GRID_ID_COL, "Date", "week_start")]
    master = master[[GRID_ID_COL, "week_start"] + feat_cols]

    merged = union.merge(master, on=JOIN_KEYS, how="left")
    log(f"[join] attached {len(feat_cols)} lag features on {JOIN_KEYS}")
    return merged, feat_cols


def add_seasonality(df):
    """Circular day-of-year of the TARGET week (leak-free: known at predict time)."""
    doy = pd.to_datetime(df["week_start"]).dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def drop_warmup(df, feat_cols):
    """Remove cell-weeks with no lag features at all (early-2013 warmup: no prior
    climate exists to build a window from). This is honest truncation, not leakage."""
    has_feat = df[feat_cols].notna().any(axis=1)
    n_drop = int((~has_feat).sum())
    if n_drop:
        by_year = df.loc[~has_feat, "iso_year"].value_counts().sort_index().to_dict()
        log(f"[warmup] dropping {n_drop} cell-weeks with no climate window "
            f"(by year: {by_year})")
    return df[has_feat].reset_index(drop=True)


def run():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)

    union = load_labelled_union()
    merged, feat_cols = attach_features(union)
    merged = add_seasonality(merged)
    merged = drop_warmup(merged, feat_cols)

    # final report
    P = int((merged["presence"] == 1).sum()); A = int((merged["presence"] == 0).sum())
    log(f"[final] {len(merged):,} rows | presence {P:,} / absence {A:,} "
        f"= {P/(P+A):.1%} / {A/(P+A):.1%}")
    log(f"[final] {merged[GRID_ID_COL].nunique()} cells | "
        f"years {merged['iso_year'].min()}-{merged['iso_year'].max()}")
    model_feats = feat_cols + ["sin_doy", "cos_doy"]
    log(f"[final] {len(model_feats)} model features: {model_feats}")
    # per-feature NaN (XGBoost handles these natively; just report them)
    nan = merged[model_feats].isna().mean().round(3)
    noteworthy = {k: f"{v:.1%}" for k, v in nan.items() if v > 0}
    log(f"[final] features with any NaN (kept for native handling): {noteworthy or 'none'}")

    merged.to_parquet(out / "weekly_model_table.parquet", index=False)
    log(f"[done] wrote weekly_model_table.parquet -> Stage 5 trains on this.")
    return merged


if __name__ == "__main__":
    run()