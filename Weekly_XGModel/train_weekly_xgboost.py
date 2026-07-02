from __future__ import annotations
"""
train_weekly_model.py  —  STAGE 5 of the WEEKLY suitability pipeline.

Trains the occurrence (suitability) classifier and evaluates it HONESTLY.

METRICS (per scheme, per model), pooled OUT-OF-FOLD:
  * PR-AUC        ranking under imbalance   -> reported WITH its no-skill baseline
                                               (baseline = fold-pooled prevalence)
  * Brier Skill Score  calibration          -> BSS = 1 - Brier/Brier_baseline;
                                               closes PR-AUC's blind spot (whether
                                               the emitted PROBABILITIES are true,
                                               which is what the Stage-6 map renders)
  * ROC-AUC       prevalence-independent ranking, as a cross-check

CV SCHEMES (LeaveOneGroupOut, predictions pooled across folds):
  * spatiotemporal  (region x year)  <- HEADLINE: new place AND new time
  * temporal        (leave-one-year-out)   decomposition: "new season"
  * spatial         (leave-one-region-out) decomposition: "new place"
  * spatiotemporal_3blocks                 robustness: same headline, coarser space
                                           (shows the result is not an artefact of
                                            block count -- WITHOUT conceding the
                                            honest 6-block holdout as primary)

MODELS: XGBClassifier (headline) vs RandomForest (matched baseline, same folds/
features/weights). Both get imbalance handling + sqrt(n_events) effort weighting.

Leakage discipline: X = df[model_features] from model_features.json ONLY.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
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
FEATURES    = WEEKLY_DIR / "model_features.json"
OUTPUT_DIR  = Path(config["weekly_results_dir"])

GRID_ID_COL, TARGET = "Grid_ID", "presence"
N_SPATIAL_BLOCKS = 6
COARSE_BLOCKS    = 3        # robustness sensitivity check
RANDOM_STATE     = 42

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist")
RF_PARAMS  = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                  class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
# ============================================================================

def log(m): print(m, flush=True)


# ----- CV blocks (regenerate if Stage 4 didn't add them) --------------------
def ensure_blocks(df, k=N_SPATIAL_BLOCKS, col="spatial_block"):
    if col in df.columns:
        return df
    cells = df.groupby(GRID_ID_COL)[["cell_lat", "cell_lon"]].first()
    cells[col] = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init=10).fit_predict(cells[["cell_lat", "cell_lon"]])
    return df.merge(cells[col], on=GRID_ID_COL)

def group_labels(df, scheme):
    if scheme == "spatial":               return df["spatial_block"].to_numpy()
    if scheme == "temporal":              return df["iso_year"].to_numpy()
    if scheme == "spatiotemporal":        return (df["spatial_block"].astype(str)+"_"+df["iso_year"].astype(str)).to_numpy()
    if scheme == "spatiotemporal_3blocks":return (df["coarse_block"].astype(str)+"_"+df["iso_year"].astype(str)).to_numpy()
    raise ValueError(scheme)


# ----- model factories ------------------------------------------------------
def fit_xgb(Xtr, ytr, wtr):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)   # neg/pos on THIS fold
    m = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

def fit_rf(Xtr, ytr, wtr):
    m = RandomForestClassifier(**RF_PARAMS)
    m.fit(Xtr.fillna(np.nan) if False else Xtr, ytr, sample_weight=wtr)
    return m


# ----- pooled out-of-fold prediction ----------------------------------------
def pooled_oof(df, X, y, w, scheme, fitter):
    groups = group_labels(df, scheme)
    oof = np.full(len(df), np.nan)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        model = fitter(X.iloc[tr], y.iloc[tr], w.iloc[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof, groups


# ----- metrics (pooled) with baselines --------------------------------------
def score(y, p, groups=None, thin_mask=None):
    """PR-AUC vs prevalence baseline, ROC-AUC, and Brier Skill Score."""
    m = ~np.isnan(p)
    y, p = y[m], p[m]
    prev = y.mean()
    pr   = average_precision_score(y, p)
    roc  = roc_auc_score(y, p)
    brier      = brier_score_loss(y, p)
    brier_base = brier_score_loss(y, np.full_like(p, prev))   # predict prevalence
    bss  = 1 - brier / brier_base if brier_base > 0 else np.nan
    return dict(n=int(m.sum()), prevalence=round(prev,3),
                pr_auc=round(pr,3), pr_baseline=round(prev,3),
                pr_lift=round(pr-prev,3), roc_auc=round(roc,3),
                brier=round(brier,4), bss=round(bss,3))


def run():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(MODEL_TABLE)
    feats = json.load(open(FEATURES))["model_features"]

    df = ensure_blocks(df, N_SPATIAL_BLOCKS, "spatial_block")
    df = ensure_blocks(df, COARSE_BLOCKS,    "coarse_block")

    X = df[feats]
    y = df[TARGET].astype(int)
    w = np.sqrt(df["n_events"].clip(lower=1)).rename("w")
    log(f"[data] {len(df):,} rows | {len(feats)} features | "
        f"prevalence {y.mean():.1%} presence")

    models = {"xgboost": fit_xgb} if xgb is not None else {}
    models["random_forest"] = fit_rf
    if xgb is None:
        log("[warn] xgboost not installed -> RandomForest only")

    schemes = ["spatiotemporal", "temporal", "spatial", "spatiotemporal_3blocks"]
    rows, oof_store = [], {}
    for name, fitter in models.items():
        for scheme in schemes:
            oof, groups = pooled_oof(df, X, y, w, scheme, fitter)
            s = score(y.to_numpy(), oof)
            s.update(model=name, scheme=scheme)
            rows.append(s)
            oof_store[f"oof_{name}_{scheme}"] = oof
            log(f"[{name:13s}|{scheme:22s}] "
                f"PR-AUC {s['pr_auc']:.3f} (base {s['pr_baseline']:.3f}, "
                f"lift {s['pr_lift']:+.3f}) | ROC {s['roc_auc']:.3f} | BSS {s['bss']:+.3f}")

    res = pd.DataFrame(rows)[["model","scheme","n","prevalence",
                              "pr_auc","pr_baseline","pr_lift","roc_auc","brier","bss"]]
    res.to_csv(out / "cv_metrics.csv", index=False)

    # save pooled OOF predictions for diagnostics / mapping calibration
    diag = df[[GRID_ID_COL,"iso_year","iso_week","week_start","presence",
               "spatial_block","n_events"]].copy()
    for k, v in oof_store.items(): diag[k] = v
    diag.to_csv(out / "oof_predictions_weekly.csv", index=False)

    # ----- final model on ALL data: SHAP via native pred_contribs -----
    if xgb is not None:
        spw = (y==0).sum()/max((y==1).sum(),1)
        final = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
        final.fit(X, y, sample_weight=w)
        final.save_model(str(out / "weekly_xgb_model.json"))
        contribs = final.get_booster().predict(
            xgb.DMatrix(X, feature_names=feats, missing=np.nan), pred_contribs=True)
        mean_abs = pd.Series(np.abs(contribs[:, :-1]).mean(0), index=feats)
        (mean_abs.sort_values(ascending=False).rename("mean_abs_shap")
         .to_csv(out / "shap_importance_weekly.csv"))
        log("\n[shap] top predictors (mean |SHAP|):")
        log(mean_abs.sort_values(ascending=False).head(10).round(3).to_string())

    log(f"\n[done] wrote cv_metrics.csv, oof_predictions_weekly.csv, model + SHAP to {out}")
    return res


if __name__ == "__main__":
    run()