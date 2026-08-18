from __future__ import annotations
"""
seed_sensitivity.py  —  is the result an artefact of ONE spatial partition?

WHAT THIS TESTS (and why it is NOT the bootstrap)
  The block bootstrap asks: given THIS partition of Florida into 6 regions, how
  much would the metric move if we had sampled different region-years? It holds
  the partition fixed and resamples within it.

  This script asks a different question: how much does the metric move if the
  PARTITION ITSELF had been drawn differently? KMeans on cell centroids with
  RANDOM_STATE=42 produces one specific carve-up of the state. A different seed
  gives different region boundaries, hence different train/test geography, hence
  a different spatial-transfer test. If the headline numbers swing wildly across
  seeds, the reported result is a property of the partition rather than of the
  model.

  Both are needed. Neither substitutes for the other: the bootstrap addresses
  sampling noise, this addresses design arbitrariness.

WHY IT MATTERS MOST FOR THE SPATIAL SCHEME
  Temporal folds are fixed by the calendar -- iso_year is not a random choice, so
  seed has no effect on the temporal scheme. Spatial and spatiotemporal folds
  depend entirely on the KMeans solution, so those are where variation appears.

COST
  One full compare_models-style recompute of XGB + RF per seed, calibrated. On
  ~21.6k rows this is minutes, not hours. MaxEnt is EXCLUDED: at ~100-170 s per
  fit, re-running it across seeds would take days for a robustness check, and the
  tree models are sufficient to establish whether the partition drives the result.

OUTPUT (to comparison_dir)
  seed_sensitivity_runs.csv     every seed x model x scheme metric row
  seed_sensitivity_summary.csv  min / median / max / range per model x scheme
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import cv_harness as H

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
FEAT_DIR = Path(cfg["weekly_xg_dir"])
OUT_DIR = Path(cfg.get("comparison_dir", str(FEAT_DIR.parent / "Comparison_Results")))

SEEDS = [42, 7, 123, 2024, 31]          # 42 first: reproduces the headline run
SCHEMES = ("spatiotemporal", "spatial")  # match the four-model comparison
METRICS = ["roc_auc", "pr_lift", "bss"]

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=5.0, random_state=42,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist")
RF_PARAMS = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                 class_weight="balanced", n_jobs=-1, random_state=42)
# ============================================================================


def log(m): print(m, flush=True)


def run_one_seed(df_raw, feats, seed):
    """Rebuild blocks under `seed`, then score XGB + RF through the shared harness.

    NOTE: only the BLOCKING seed varies. Model seeds stay fixed at 42 so the only
    thing changing between runs is the spatial partition -- otherwise model
    stochasticity and partition choice would be confounded.
    """
    df = H.build_blocks(df_raw, random_state=seed)
    w = np.sqrt(df["n_events"].clip(lower=1))
    spw = (df.presence == 0).sum() / max((df.presence == 1).sum(), 1)

    def make_xgb():
        return xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)

    def make_rf():
        return RandomForestClassifier(**RF_PARAMS)

    frames = []
    if xgb is not None:
        r, _ = H.evaluate(df, feats, make_xgb, schemes=SCHEMES, sample_weight=w,
                          calibrate=True, impute=False, model_name="xgboost",
                          verbose=False)
        frames.append(r)
    r, _ = H.evaluate(df, feats, make_rf, schemes=SCHEMES, sample_weight=w,
                      calibrate=True, impute=True, model_name="random_forest",
                      verbose=False)
    frames.append(r)

    out = pd.concat(frames, ignore_index=True)
    out["block_seed"] = seed

    # record how the partition actually differs between seeds
    sizes = df.groupby("spatial_block")[H.GRID_ID_COL].nunique().sort_index()
    out["block_cells_min"] = int(sizes.min())
    out["block_cells_max"] = int(sizes.max())
    return out, sizes


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats = json.load(open(FEAT_DIR / "model_features.json"))["model_features"]
    df_raw = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet")
    log(f"[seed] {len(df_raw):,} rows | {len(feats)} features | "
        f"{len(SEEDS)} seeds x {len(SCHEMES)} schemes")

    runs = []
    for seed in SEEDS:
        log(f"\n[seed {seed}] rebuilding spatial blocks and rescoring...")
        res, sizes = run_one_seed(df_raw, feats, seed)
        log(f"[seed {seed}] cells per block: {sizes.to_dict()}")
        for _, row in res.iterrows():
            log(f"[seed {seed}] {row['model']:14s} {row['scheme']:16s} "
                f"ROC {row['roc_auc']:.3f} | PR-lift {row['pr_lift']:+.3f} "
                f"| BSS {row['bss']:+.3f}")
        runs.append(res)

    allruns = pd.concat(runs, ignore_index=True)
    allruns.to_csv(OUT_DIR / "seed_sensitivity_runs.csv", index=False)

    # summary: spread of each metric across seeds
    rows = []
    for (model, scheme), sub in allruns.groupby(["model", "scheme"], sort=False):
        rec = dict(model=model, scheme=scheme, n_seeds=len(sub))
        for m in METRICS:
            v = sub[m].astype(float)
            rec[f"{m}_seed42"] = round(float(sub.loc[sub.block_seed == 42, m].iloc[0]), 3)
            rec[f"{m}_min"] = round(float(v.min()), 3)
            rec[f"{m}_median"] = round(float(v.median()), 3)
            rec[f"{m}_max"] = round(float(v.max()), 3)
            rec[f"{m}_range"] = round(float(v.max() - v.min()), 3)
        rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "seed_sensitivity_summary.csv", index=False)

    log("\n[summary] metric spread across blocking seeds:")
    log(summary[["model", "scheme", "roc_auc_seed42", "roc_auc_min",
                 "roc_auc_max", "roc_auc_range"]].to_string(index=False))
    log(f"\n[done] wrote seed_sensitivity_runs.csv + seed_sensitivity_summary.csv to {OUT_DIR}")
    log("[read] a LARGE roc_auc_range on the spatial scheme means the transfer "
        "estimate depends on where the block boundaries fall, and should be "
        "reported as a range rather than a point estimate.")
    return summary


if __name__ == "__main__":
    run()