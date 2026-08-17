from __future__ import annotations
"""
nowcast_run.py  —  DRIVER: run all four models through forward-chaining CV.

Self-contained (imports only nowcast_cv). Runs each model with expanding-window
folds and writes per-fold + pooled + headline metrics. The comparison FIGURES
are a separate file (nowcast_compare.py) that reads these CSVs.

Per-model data / weighting (as each was built):
  xgboost / random_forest / maxent_targetgroup : presences + target-group
      absences (weekly_model_table). XGB/RF weighted sqrt(n_events); MaxEnt not.
  maxent_vanilla : presences + random background (vanilla_background). Unweighted.

Expected files (put them in nowcast_dir, or point the config keys at them):
  weekly_model_table.parquet, model_features.json, vanilla_background.parquet

OUTPUT (to nowcast_results_dir)
  nowcast_perfold_all.csv        every fold, every model
  nowcast_summary.csv            pooled + headline row per model
  oof_nowcast_<variant>.parquet  the out-of-fold prediction for every row

WHY THE OOF FILES: the aggregate metrics cannot be re-used for the baseline
comparison (persistence is defined on only ~79% of rows, so it needs row-level
subsetting), for reliability diagrams, or for bootstrap confidence intervals.
Saving the vectors once means those never require a model re-run -- which matters
because the two MaxEnt variants take hours.

ROW ALIGNMENT: for xgboost / random_forest / maxent_targetgroup the assembled
frame is the model table in its original row order, so the OOF vector aligns
positionally with weekly_model_table.parquet. The keys are saved alongside so
alignment can be verified by merge rather than assumed. maxent_vanilla is the
exception: it trains on presences + random background, so its rows are a different
set and it CANNOT be row-aligned with the other models or with the baselines
(which are defined on model-table rows). Its OOF is saved for completeness but
must be excluded from the baseline comparison.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nowcast_cv as NC
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    from elapid import MaxentModel
except ImportError:
    MaxentModel = None

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
# everything for the nowcast lives together; fall back to the old dirs if unset
NOWCAST_DIR = Path(cfg["nowcast_dir"]) 
DATA_DIR    = Path(cfg.get("weekly_xg_dir", str(NOWCAST_DIR)))     # model table + features
BG_FILE     = Path(cfg.get("nowcast_background",
                           str(Path(cfg.get("maxent_dir", str(NOWCAST_DIR))) / "vanilla_background.parquet")))
OUT_DIR     = Path(cfg.get("nowcast_results_dir", str(NOWCAST_DIR / "Nowcast_Results")))

RANDOM_STATE = 42

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss", tree_method="hist")
RF_PARAMS  = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                  class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

# which models to run (drop entries to skip; MaxEnt is slow ~hours each)
RUN_MODELS = ["xgboost", "random_forest", "maxent_targetgroup", "maxent_vanilla"]
# ============================================================================

def log(m): print(m, flush=True)


def make_maxent():
    return MaxentModel(feature_types=["linear", "quadratic", "hinge"],
                       transform="cloglog", clamp=True)


def assemble(df_table, bg, variant, feats):
    """Return (df, sample_weight, impute) for a model variant."""
    keep = ["Grid_ID", "iso_year", "iso_week", "cell_lat", "cell_lon", "presence"] + feats
    if variant == "maxent_vanilla":
        pres = df_table[df_table.presence == 1][keep]
        neg = bg.copy(); neg["presence"] = 0
        df = pd.concat([pres, neg[keep]], ignore_index=True)
        return df, None, True                      # unweighted, impute (MaxEnt)
    df = df_table[keep].copy()                     # pres + target-group absences
    if variant in ("xgboost", "random_forest"):
        w = np.sqrt(df_table["n_events"].clip(lower=1).to_numpy())
        return df, pd.Series(w, index=df.index), (variant == "random_forest")
    return df, None, True                          # maxent_targetgroup: unweighted, impute


def make_factory(variant, df):
    if variant == "xgboost":
        spw = (df.presence == 0).sum() / max((df.presence == 1).sum(), 1)
        return lambda: xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    if variant == "random_forest":
        return lambda: RandomForestClassifier(**RF_PARAMS)
    return make_maxent                             # both MaxEnt variants


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats = json.load(open(DATA_DIR / "model_features.json"))["model_features"]
    table = pd.read_parquet(DATA_DIR / "weekly_model_table.parquet")
    bg = pd.read_parquet(BG_FILE) if BG_FILE.exists() else None
    log(f"[nowcast] table {len(table):,} rows | features {len(feats)} | "
        f"background {'loaded' if bg is not None else 'MISSING (skip vanilla)'}")

    perfold, summary = [], []
    for variant in RUN_MODELS:
        if variant.startswith("maxent") and MaxentModel is None:
            log(f"[nowcast] skip {variant}: elapid not installed"); continue
        if variant == "xgboost" and xgb is None:
            log(f"[nowcast] skip xgboost: not installed"); continue
        if variant == "maxent_vanilla" and bg is None:
            log(f"[nowcast] skip maxent_vanilla: no background file"); continue

        df, w, impute = assemble(table, bg, variant, feats)
        make_model = make_factory(variant, df)
        log(f"\n[nowcast] === {variant} === ({len(df):,} rows, "
            f"{df.presence.mean():.1%} presence, {'weighted' if w is not None else 'unweighted'})")
        pf, pooled, headline, oof = NC.evaluate_nowcast(
            df, feats, make_model, sample_weight=w, calibrate=True,
            impute=impute, model_name=variant, return_oof=True)
        perfold.append(pf)
        summary.append(pooled); summary.append(headline)

        # persist the OOF vector with its keys (see ROW ALIGNMENT in the header)
        keys = [c for c in ("Grid_ID", "iso_year", "iso_week", "presence") if c in df.columns]
        oof_df = df[keys].copy()
        oof_df["oof"] = oof
        oof_df["model"] = variant
        oof_path = OUT_DIR / f"oof_nowcast_{variant}.parquet"
        oof_df.to_parquet(oof_path, index=False)
        cov = float(np.isfinite(oof).mean())
        log(f"[nowcast] saved {oof_path.name} ({len(oof_df):,} rows, "
            f"{100*cov:.1f}% scored)"
            + ("  NOTE: different row set -- exclude from baseline comparison"
               if variant == "maxent_vanilla" else ""))

    pd.concat(perfold, ignore_index=True).to_csv(OUT_DIR / "nowcast_perfold_all.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT_DIR / "nowcast_summary.csv", index=False)
    log(f"\n[done] wrote nowcast_perfold_all.csv + nowcast_summary.csv + "
        f"oof_nowcast_*.parquet to {OUT_DIR}")
    log("[next] nowcast_compare.py builds the figures; run_baselines.py adds the "
        "reference forecasts; baselines.evaluate_with_baselines() consumes the OOF "
        "files for the matched-subset comparison and bootstrap CIs.")


if __name__ == "__main__":
    run()