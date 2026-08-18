from __future__ import annotations
"""
maxent_common.py  —  shared logic for the two MaxEnt variants.

elapid's MaxentModel is already sklearn-style (fit/predict_proba, tolerates NaN),
so it plugs straight into cv_harness with no wrapper. The only thing that differs
between the two variants is the NEGATIVE/background class:

  * vanilla      : presences + 10k random statewide background (vanilla_background)
  * targetgroup  : presences + inferred target-group absences (from model table)

Both are UNWEIGHTED (MaxEnt convention; makes the vanilla-vs-targetgroup contrast
a clean test of the absence strategy). Both run the IDENTICAL cv_harness folds +
metrics as XGB/RF, and both are calibrated in-fold (isotonic) so BSS is
comparable across model families -- MaxEnt's raw cloglog output is a suitability
index, not a probability, and the calibration maps it onto observed frequency.

Each variant: evaluate (metrics + OOF) -> fit final calibrated model on all data
-> predict the statewide grid -> write a prob_<variant> column for the maps.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from elapid import MaxentModel
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression

import cv_harness as H

RANDOM_STATE = 42
CALIB_FOLDS = 5

# MaxEnt feature classes: linear + quadratic + hinge gives MaxEnt its flexible
# non-linear response curves (hinge) plus smooth curvature (quadratic). PRODUCT
# is deliberately dropped: on 31 predictors it creates hundreds of interaction
# terms -> intractably slow AND overfit-prone. This is the strong, defensible
# config, not a speed compromise.
def make_maxent():
    return MaxentModel(feature_types=["linear", "quadratic", "hinge"],
                       transform="cloglog", clamp=True)

# MaxEnt is ~100-170s per fit, so full LOGO x nested calibration is many hours.
# We run the two schemes that carry the comparison: spatiotemporal (headline,
# new place + new time) and spatial (the transfer test vs the static model).
# Add "temporal" / "spatiotemporal_3blocks" here later if you want the full set.
MAXENT_SCHEMES = ("spatiotemporal", "spatial")


def load_config():
    with open("config.json") as f:
        return json.load(f)


def evaluate_variant(df, feats, variant, out_dir):
    """Run the shared harness (unweighted, calibrated) and save metrics + OOF."""
    df = H.build_blocks(df)
    res, oof = H.evaluate(df, feats, make_maxent, schemes=MAXENT_SCHEMES,
                          sample_weight=None,
                          calibrate=True, impute=True,   # MaxEnt fit rejects NaN
                          model_name=f"maxent_{variant}")
    res.to_csv(out_dir / f"cv_metrics_maxent_{variant}.csv", index=False)
    diag = df[[H.GRID_ID_COL, "iso_year", "iso_week", "presence", "spatial_block"]].copy()
    for scheme, p in oof.items():
        diag[f"oof_{scheme}"] = p
    diag.to_csv(out_dir / f"oof_maxent_{variant}.csv", index=False)
    return res


def fit_final_calibrated(df, feats):
    """Isotonic map from grouped OOF (spatiotemporal) + base fit on all data.
    Median-imputes NaN (MaxEnt's LogisticRegression rejects NaN), matching the
    imputation the CV harness used, so the final model is consistent with eval."""
    X, y = df[feats].copy(), df["presence"].astype(int)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    groups = (df["spatial_block"].astype(str) + "_" + df["iso_year"].astype(str)).to_numpy()
    oof = np.full(len(X), np.nan)
    gkf = GroupKFold(n_splits=min(CALIB_FOLDS, pd.Series(groups).nunique()))
    for tr, te in gkf.split(X, y, groups):
        b = make_maxent(); b.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = b.predict_proba(X.iloc[te])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    base = make_maxent(); base.fit(X, y)
    return base, iso, medians


def predict_statewide(base, iso, medians, feats, grid_file, variant, out_dir):
    """Predict the statewide climatological grid; write prob_maxent_<variant>.
    Impute grid NaN with the SAME training medians used to fit the model."""
    grid = pd.read_parquet(grid_file)
    Xg = grid[feats].fillna(medians)
    p = iso.predict(base.predict_proba(Xg)[:, 1])
    surf = grid[[H.GRID_ID_COL, "iso_week"]].copy()
    surf[f"prob_maxent_{variant}"] = p
    surf.to_parquet(out_dir / f"surface_maxent_{variant}.parquet", index=False)
    print(f"[maxent_{variant}] statewide surface: mean {p.mean():.3f} "
          f"| range {p.min():.2f}-{p.max():.2f}", flush=True)
    return surf


def run_variant(variant, negative_df, feat_dir, out_dir, grid_dir=None):
    """Full variant run. Paths are passed in explicitly (no hidden config reads):
      feat_dir : holds weekly_model_table.parquet + model_features.json
      out_dir  : where metrics / OOF / surface are written
      grid_dir : holds statewide_weekly_features.parquet (defaults to feat_dir)
    """
    feat_dir = Path(feat_dir); out_dir = Path(out_dir)
    grid_dir = Path(grid_dir) if grid_dir else feat_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> maxent_{variant}: starting", flush=True)

    feats = json.load(open(feat_dir / "model_features.json"))["model_features"]

    pres = pd.read_parquet(feat_dir / "weekly_model_table.parquet")
    pres = pres[pres["presence"] == 1].copy()
    keep = [H.GRID_ID_COL, "iso_year", "iso_week", "cell_lat", "cell_lon", "presence"] + feats
    df = pd.concat([pres[keep], negative_df[keep]], ignore_index=True)
    print(f"[maxent_{variant}] {len(pres):,} presence + {len(negative_df):,} "
          f"{'background' if variant=='vanilla' else 'absence'} "
          f"= {len(df):,} rows ({df.presence.mean():.1%} presence)", flush=True)

    res = evaluate_variant(df, feats, variant, out_dir)
    df = H.build_blocks(df)
    base, iso, medians = fit_final_calibrated(df, feats)
    predict_statewide(base, iso, medians, feats,
                      grid_dir / "statewide_weekly_features.parquet", variant, out_dir)
    print(f"[maxent_{variant}] done -> {out_dir}/cv_metrics_maxent_{variant}.csv, "
          f"surface_maxent_{variant}.parquet", flush=True)
    return res