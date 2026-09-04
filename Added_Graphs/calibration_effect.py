from __future__ import annotations
"""
calibration_effect.py  —  raw vs calibrated metrics for EVERY model and scheme,
through the same harness that produced combined_metrics.csv.

WHY THIS EXISTS
  calibration_metrics.csv (from calibrate_weekly_models.py) covers XGBoost and
  Random Forest under spatiotemporal and temporal only. Table R2.0 therefore has
  no raw column for the spatial scheme or for either MaxEnt variant -- which is
  where the interesting behaviour is.

  This script runs cv_harness.evaluate TWICE per model per scheme, once with
  calibrate=False and once with calibrate=True, and reports the pair. Because it
  uses the same harness and the same blocks as compare_models.py, the calibrated
  column reconciles exactly with combined_metrics.csv and
  tree_scheme_metrics.csv.

WHAT THE RAW COLUMN IS FOR
  Isotonic regression is monotone, so it cannot change the ranking WITHIN a
  fold, and ROC should be unchanged. Under spatiotemporal CV it is: 0.849 raw
  against 0.850 calibrated.

  Under spatial CV it need not be, and that is not a contradiction. Each fold
  gets its OWN isotonic map, fitted on that fold's calibration slice. Pooling
  out-of-fold scores that have passed through different monotone maps yields a
  vector whose ordering is no longer meaningful ACROSS folds. With 32 folds the
  maps are similar and the effect is negligible; with 6 spatial folds they are
  not, and the pooled ROC can move a long way. The raw column is what isolates
  that effect, so it is the evidence for the "calibration has a cost under
  spatial CV" claim rather than an assertion of it.

COST
  Two full CV runs per model per scheme. Trees are minutes. MaxEnt is ~100-170 s
  per fit, so a single 32-fold scheme costs hours and it is OFF by default; set
  RUN_MAXENT = True only if you have the time, and consider restricting SCHEMES.

OUTPUT (to comparison_dir)
  calibration_effect.csv   model, scheme, brier_raw/cal, bss_raw/cal, roc_raw/cal,
                           and the deltas -- the source for Table R2.0
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import cv_harness as H
import report_style as S

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
FEAT_DIR = Path(cfg["weekly_xg_dir"])
MAXENT_DIR = Path(cfg.get("maxent_dir", str(FEAT_DIR)))
OUT_DIR = Path(cfg.get("comparison_dir", str(FEAT_DIR.parent / "Comparison_Results")))

SCHEMES = ("spatiotemporal", "spatial")
RUN_MAXENT = True          # see COST above
RANDOM_STATE = 42

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist")
RF_PARAMS = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                 class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

def log(m): print(m, flush=True)
# ============================================================================


def pair(df, feats, make_model, name, weight, impute):
    """One row per scheme: metrics with calibration off, then on."""
    rows = []
    for calibrate in (False, True):
        res, _ = H.evaluate(df, feats, make_model, schemes=SCHEMES,
                            sample_weight=weight, calibrate=calibrate,
                            impute=impute, model_name=name, verbose=False)
        res["calibrated"] = calibrate
        rows.append(res)
        log(f"[cal] {name:22s} calibrate={str(calibrate):5s} done "
            f"({len(SCHEMES)} schemes)")
    return pd.concat(rows, ignore_index=True)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats = json.load(open(FEAT_DIR / "model_features.json"))["model_features"]
    table = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet")
    df = H.build_blocks(table)          # the SAME partition as compare_models.py
    w = np.sqrt(df["n_events"].clip(lower=1))
    spw = (df.presence == 0).sum() / max((df.presence == 1).sum(), 1)
    log(f"[cal] {len(df):,} rows | schemes {SCHEMES}")

    frames = []
    if xgb is not None:
        frames.append(pair(df, feats,
                           lambda: xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS),
                           "xgboost", w, False))
    frames.append(pair(df, feats, lambda: RandomForestClassifier(**RF_PARAMS),
                       "random_forest", w, True))

    if RUN_MAXENT:
        try:
            from elapid import MaxentModel
            import maxent_common as MC

            def make_maxent():
                return MaxentModel(feature_types=["linear", "quadratic", "hinge"],
                                   transform="cloglog", clamp=True)

            # target-group uses the trap table directly, exactly as
            # maxent_common.evaluate_variant does (unweighted, imputed)
            frames.append(pair(df, feats, make_maxent,
                               "maxent_targetgroup", None, True))

            # vanilla needs its own presence + background frame; it is assembled
            # by the maxent scripts, not here, so it is only attempted if that
            # background file exists
            bg = Path(cfg.get("nowcast_background",
                              str(MAXENT_DIR / "vanilla_background.parquet")))
            if bg.exists():
                pres = table[table.presence == 1]
                b = pd.read_parquet(bg).assign(presence=0)
                keep = [c for c in ["Grid_ID", "iso_year", "iso_week",
                                    "cell_lat", "cell_lon", "presence"] + feats
                        if c in pres.columns and c in b.columns]
                van = H.build_blocks(pd.concat([pres[keep], b[keep]],
                                               ignore_index=True))
                frames.append(pair(van, feats, make_maxent,
                                   "maxent_vanilla", None, True))
            else:
                log(f"[cal] vanilla background not found at {bg} -- skipped")
        except Exception as e:
            log(f"[cal] MaxEnt skipped ({type(e).__name__}: {e})")

    allm = pd.concat(frames, ignore_index=True)
    wide = allm.pivot_table(index=["model", "scheme"], columns="calibrated",
                            values=["brier", "bss", "roc_auc"])
    wide.columns = [f"{a}_{'cal' if b else 'raw'}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["d_bss"] = (wide.bss_cal - wide.bss_raw).round(3)
    wide["d_roc"] = (wide.roc_auc_cal - wide.roc_auc_raw).round(3)
    wide["model"] = pd.Categorical(wide.model, S.models_in(wide.model), ordered=True)
    wide["scheme"] = pd.Categorical(wide.scheme, S.schemes_in(wide.scheme), ordered=True)
    wide = wide.sort_values(["scheme", "model"])

    cols = ["model", "scheme", "brier_raw", "brier_cal", "bss_raw", "bss_cal",
            "d_bss", "roc_auc_raw", "roc_auc_cal", "d_roc"]
    wide = wide[[c for c in cols if c in wide.columns]]
    wide.to_csv(OUT_DIR / "calibration_effect.csv", index=False)
    log("\n[cal] Table R2.0 source:")
    log(wide.to_string(index=False))

    big = wide[wide.d_roc.abs() > 0.01]
    if len(big):
        log("\n[cal] ROC moved by more than 0.01 under calibration for:")
        log(big[["model", "scheme", "roc_auc_raw", "roc_auc_cal", "d_roc"]]
            .to_string(index=False))
        log("[cal] Isotonic is monotone WITHIN a fold, so this is not a ranking "
            "change: each fold applies its own map, and pooling across folds "
            "mixes maps. The effect grows as the number of folds falls, which is "
            "why it shows up under the 6-fold spatial scheme and not the 32-fold "
            "spatiotemporal one.")
    log(f"\n[done] -> {OUT_DIR / 'calibration_effect.csv'}")
    return wide


if __name__ == "__main__":
    run()