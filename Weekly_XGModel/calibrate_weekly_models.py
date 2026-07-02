from __future__ import annotations
"""
calibrate_weekly_model.py  —  STAGE 5b of the WEEKLY suitability pipeline.

Fixes the calibration failure Stage 5 exposed (XGBoost BSS negative: its ranking
is good but its emitted PROBABILITIES are untrustworthy -- fatal for a map that
renders probability as colour). Random Forest was already calibrated; we
calibrate both for parity and as a cross-check.

THE ONE RULE THAT KEEPS IT HONEST: calibration is fit INSIDE each CV fold, on a
held-out slice of the TRAINING data only -- never on the test fold. Otherwise we
would tune the probability scale on data the model saw, re-importing leakage.

Mechanism (nested, per outer fold):
  1. split the outer-train rows -> inner-fit (learn the model) + inner-cal (learn
     the probability mapping), grouped so inner-cal shares no CV-group with
     inner-fit (calibration is judged on transfer, matching the outer split).
  2. fit base model on inner-fit; fit isotonic mapping on its inner-cal scores.
  3. apply mapping to the OUTER-test scores -> calibrated OOF probability.

Compares raw vs calibrated via Brier Skill Score (BSS) and a reliability table,
per model, on the HEADLINE spatiotemporal scheme (and temporal for reference).

OUTPUT (to static_results_dir)
  calibration_metrics.csv        raw vs calibrated BSS / Brier / ROC (unchanged)
  reliability_<model>.csv        binned predicted-vs-observed (calibrated)
  oof_calibrated_weekly.csv      calibrated OOF probs (for the Stage-6 maps)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
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
RANDOM_STATE = 42
CAL_FRACTION = 0.25          # share of outer-train held out to fit the calibrator

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss", tree_method="hist")
RF_PARAMS  = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                  class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
# ============================================================================

def log(m): print(m, flush=True)

def ensure_blocks(df, k=N_SPATIAL_BLOCKS, col="spatial_block"):
    if col in df.columns: return df
    cells = df.groupby(GRID_ID_COL)[["cell_lat","cell_lon"]].first()
    cells[col] = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)\
                    .fit_predict(cells[["cell_lat","cell_lon"]])
    return df.merge(cells[col], on=GRID_ID_COL)

def group_labels(df, scheme):
    if scheme=="temporal":       return df["iso_year"].to_numpy()
    if scheme=="spatiotemporal": return (df["spatial_block"].astype(str)+"_"+df["iso_year"].astype(str)).to_numpy()
    raise ValueError(scheme)


def fit_base(name, Xtr, ytr, wtr):
    if name=="xgboost":
        spw=(ytr==0).sum()/max((ytr==1).sum(),1)
        m=xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    else:
        m=RandomForestClassifier(**RF_PARAMS)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m


def nested_calibrated_oof(df, X, y, w, name, scheme):
    """Return raw AND isotonic-calibrated pooled OOF probabilities."""
    groups = group_labels(df, scheme)
    raw = np.full(len(df), np.nan); cal = np.full(len(df), np.nan)
    outer = LeaveOneGroupOut()
    for tr, te in outer.split(X, y, groups):
        gtr = groups[tr]
        # inner split of TRAIN rows: fit vs calibrate, grouped (no shared group)
        gss = GroupShuffleSplit(n_splits=1, test_size=CAL_FRACTION, random_state=RANDOM_STATE)
        fit_idx, cal_idx = next(gss.split(X.iloc[tr], y.iloc[tr], gtr))
        Xf, yf, wf = X.iloc[tr].iloc[fit_idx], y.iloc[tr].iloc[fit_idx], w.iloc[tr].iloc[fit_idx]
        Xc, yc     = X.iloc[tr].iloc[cal_idx], y.iloc[tr].iloc[cal_idx]

        base = fit_base(name, Xf, yf, wf)
        # learn isotonic map on held-out calibration slice
        pc = base.predict_proba(Xc)[:,1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(pc, yc)

        pte = base.predict_proba(X.iloc[te])[:,1]
        raw[te] = pte
        cal[te] = iso.predict(pte)
    return raw, cal


def bss(y, p):
    m=~np.isnan(p); y,p=y[m],p[m]; prev=y.mean()
    b=brier_score_loss(y,p); b0=brier_score_loss(y,np.full_like(p,prev))
    return 1-b/b0 if b0>0 else np.nan, b, roc_auc_score(y,p)

def reliability(y, p, bins=10):
    m=~np.isnan(p); y,p=y[m],p[m]
    edges=np.linspace(0,1,bins+1); idx=np.clip(np.digitize(p,edges)-1,0,bins-1)
    rows=[]
    for b in range(bins):
        sel=idx==b
        if sel.sum():
            rows.append(dict(bin=f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                             n=int(sel.sum()), pred_mean=round(p[sel].mean(),3),
                             obs_freq=round(y[sel].mean(),3)))
    return pd.DataFrame(rows)


def run():
    out=Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_parquet(MODEL_TABLE)
    feats=json.load(open(FEATURES))["model_features"]
    df=ensure_blocks(df)
    X=df[feats]; y=df[TARGET].astype(int); w=np.sqrt(df["n_events"].clip(lower=1))

    models=[m for m in (["xgboost"] if xgb is not None else []) + ["random_forest"]]
    rows=[]; oof=df[[GRID_ID_COL,"iso_year","iso_week","week_start","presence","spatial_block"]].copy()
    for name in models:
        for scheme in ["spatiotemporal","temporal"]:
            raw,cal=nested_calibrated_oof(df,X,y,w,name,scheme)
            (bss_r,br_r,roc_r)=bss(y.to_numpy(),raw)
            (bss_c,br_c,roc_c)=bss(y.to_numpy(),cal)
            rows.append(dict(model=name,scheme=scheme,
                             bss_raw=round(bss_r,3),bss_cal=round(bss_c,3),
                             brier_raw=round(br_r,4),brier_cal=round(br_c,4),
                             roc_auc=round(roc_c,3)))
            log(f"[{name:13s}|{scheme:14s}] BSS {bss_r:+.3f} -> {bss_c:+.3f} (calibrated) | "
                f"Brier {br_r:.3f}->{br_c:.3f} | ROC {roc_c:.3f} (unchanged by calibration)")
            if scheme=="spatiotemporal":
                oof[f"cal_{name}"]=cal
                reliability(y.to_numpy(),cal).to_csv(out/f"reliability_{name}.csv",index=False)

    pd.DataFrame(rows).to_csv(out/"calibration_metrics.csv",index=False)
    oof.to_csv(out/"oof_calibrated_weekly.csv",index=False)
    log(f"\n[done] wrote calibration_metrics.csv, reliability_*.csv, "
        f"oof_calibrated_weekly.csv to {out}")
    log("[note] calibrated OOF probs feed the Stage-6 maps (one per model + agreement).")
    return pd.DataFrame(rows)

if __name__ == "__main__":
    run()