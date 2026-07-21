from __future__ import annotations
"""
predict_2025_maps.py  —  OUT-OF-PERIOD 2025 PREDICTION (no rendering).

Produces the 2025 suitability prediction surfaces for the full model set, saved
to parquet. Rendering is a SEPARATE step (run later) — this file makes no PNGs.

This is a DEPLOYMENT / CAPABILITY demonstration, not a validated result: there
are no 2025 trap catches to score against, so the outputs carry NO metric and
must be captioned as unverified predictions on out-of-period data.

Pipeline (reuses the training feature logic; no cleaning/labelling — 2025 has no
mosquito data):
  * daily climate/EVI/NDWI from the merged 2025 file,
  * STATIC land cover joined from the TRAINING source (the 2025 file carries the
    wrong 3 coarse columns; the model needs the 17 per-class fractions, which are
    static year-to-year so we reuse the training values),
  * Stage-2 leak-proof lags for each target week,
  * models fit on ALL 2013-2018 data, calibrated (isotonic on grouped OOF),
  * predict each 2025 week -> save.

Also saves the space x season opacity fade from the HISTORICAL trap footprint, so
the later render step is turn-key.

MODELS: xgboost, random_forest, maxent_targetgroup, maxent_vanilla.
MaxEnt is slow (~mins/fit x calibration folds) -- expect a long run.

OUTPUT (to out dir)
  surface_2025_predictions.parquet
    [Grid_ID, iso_week, lat, lon, prob_xgboost, prob_random_forest,
     prob_maxent_targetgroup, prob_maxent_vanilla, confidence_raw, opacity]
"""
import json, glob, re
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
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
DATA_DIR   = Path(cfg.get("weekly_xg_dir", "."))                 # model table + features
DAILY_DIR  = Path(cfg.get("parquet_dir", str(DATA_DIR)))        # 2025 merged + training OuterMerged2
BG_FILE    = Path(cfg.get("nowcast_background",
                          str(Path(cfg.get("maxent_dir", str(DATA_DIR))) / "vanilla_background.parquet")))
OUT_DIR    = Path(cfg["nowcast_dir"])
DAILY_2025 = Path(cfg.get("daily_2025",
                          str(DAILY_DIR / "Florida_Final_OuterMerged_2025.parquet")))

PRED_YEAR = 2025
WEEKS = [6, 20, 32, 43]           # winter / spring / summer / autumn
RUN_MODELS = ["xgboost", "random_forest", "maxent_targetgroup", "maxent_vanilla"]

GRID_ID_COL = "Grid_ID"
CLIMATE_COLS = ["tmax","tmin","tmean","prcp","vpd"]; VEG_COLS = ["EVI","NDWI"]
FFILL_LIMIT = {"EVI":16,"NDWI":8}; CLIM_FFILL_LIMIT = 3; LAG_WINDOWS = (7,14,28)
SEASON_WINDOW=2; FALLOFF_KM=120.0; OPACITY_FLOOR=0.25
CALIB_FOLDS = 5; RANDOM_STATE = 42

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss", tree_method="hist")
RF_PARAMS = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                 class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

def log(m): print(m, flush=True)
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
def lonlat(g):
    lat=np.array([float(_GID.search(str(x)).group(1)) for x in g])
    lon=np.array([float(_GID.search(str(x)).group(2)) for x in g]); return lat,lon
def make_maxent():
    return MaxentModel(feature_types=["linear","quadratic","hinge"], transform="cloglog", clamp=True)
# ============================================================================


def static_source(static_cols):
    """Per-class static land cover from a TRAINING file (static year-to-year).
    Prefers an OuterMerged2 training file (8,850 cells, per-class); falls back to
    the allyears static parquet."""
    tr = sorted(glob.glob(str(DAILY_DIR / "*OuterMerged2*.parquet")))
    src = tr[0] if tr else str(DATA_DIR / "Florida_Static_SDM_allyears.parquet")
    d = pd.read_parquet(src, columns=[GRID_ID_COL] + static_cols)
    return d.groupby(GRID_ID_COL)[static_cols].first().reset_index()


def build_week_features(week, feats, static_per_cell):
    """2025 week features: Stage-2 lags from the 2025 daily file + static join."""
    mon = pd.Timestamp(date.fromisocalendar(PRED_YEAR, week, 1))
    d = pd.read_parquet(DAILY_2025, columns=[GRID_ID_COL,"Date"]+CLIMATE_COLS+VEG_COLS)
    d["Date"]=pd.to_datetime(d["Date"]).dt.normalize(); d=d[d["Date"]<mon]
    d=(d.sort_values([GRID_ID_COL,"Date"]).groupby([GRID_ID_COL,"Date"],as_index=False).first())
    cells=sorted(d[GRID_ID_COL].unique())
    idx=pd.date_range(d["Date"].min(), mon-pd.Timedelta(days=1), freq="D")
    d=d.set_index([GRID_ID_COL,"Date"]).reindex(pd.MultiIndex.from_product([cells,idx],names=[GRID_ID_COL,"Date"])).reset_index()
    g=d.groupby(GRID_ID_COL,sort=False)
    for c in CLIMATE_COLS: d[c]=g[c].transform(lambda s: s.ffill(limit=CLIM_FFILL_LIMIT))
    for c in VEG_COLS: d[c+"_ff"]=g[c].transform(lambda s: s.ffill(limit=FFILL_LIMIT[c]))
    g=d.groupby(GRID_ID_COL,sort=False); agg={}
    for w in LAG_WINDOWS:
        agg[f"prcp_sum_{w}d"]=g["prcp"].apply(lambda s,w=w: s.iloc[-w:].sum(min_count=1))
        agg[f"tmean_mean_{w}d"]=g["tmean"].apply(lambda s,w=w: s.iloc[-w:].mean())
    agg["vpd_mean_14d"]=g["vpd"].apply(lambda s: s.iloc[-14:].mean())
    agg["tmax_max_14d"]=g["tmax"].apply(lambda s: s.iloc[-14:].max())
    agg["tmin_mean_14d"]=g["tmin"].apply(lambda s: s.iloc[-14:].mean())
    agg["tmin_min_14d"]=g["tmin"].apply(lambda s: s.iloc[-14:].min())
    agg["evi_level"]=g["EVI_ff"].apply(lambda s: s.iloc[-1]); agg["ndwi_level"]=g["NDWI_ff"].apply(lambda s: s.iloc[-1])
    feat=pd.DataFrame(agg).reset_index().merge(static_per_cell, on=GRID_ID_COL, how="left")
    doy=mon.dayofyear; feat["sin_doy"]=np.sin(2*np.pi*doy/365.25); feat["cos_doy"]=np.cos(2*np.pi*doy/365.25)
    return feat[[GRID_ID_COL]+feats]


# ----- model fitting (fit on ALL data, isotonic on grouped OOF) --------------
def fit_calibrated(variant, table, bg, feats):
    if variant == "maxent_vanilla":
        pres = table[table.presence==1]
        b = bg.copy(); b["presence"]=0
        keep=[GRID_ID_COL,"iso_year","presence"]+feats
        df = pd.concat([pres[keep], b[keep]], ignore_index=True); w=None; impute=True
    else:
        df = table.copy()
        w = np.sqrt(df["n_events"].clip(lower=1)) if variant in ("xgboost","random_forest") else None
        impute = variant != "xgboost"
    X, y = df[feats], df["presence"].astype(int)
    medians = X.median(numeric_only=True)
    groups = (df[GRID_ID_COL].astype(str)+"_"+df["iso_year"].astype(str)).to_numpy()

    def make(): 
        if variant=="xgboost":
            spw=(y==0).sum()/max((y==1).sum(),1); return xgb.XGBClassifier(scale_pos_weight=spw,**XGB_PARAMS)
        if variant=="random_forest": return RandomForestClassifier(**RF_PARAMS)
        return make_maxent()
    def fit(Xt,yt,wt):
        m=make(); Xf=Xt.fillna(medians) if impute else Xt
        try: m.fit(Xf,yt,sample_weight=wt) if wt is not None else m.fit(Xf,yt)
        except TypeError: m.fit(Xf,yt)
        return m

    oof=np.full(len(X),np.nan)
    gkf=GroupKFold(n_splits=min(CALIB_FOLDS,pd.Series(groups).nunique()))
    for tr,te in gkf.split(X,y,groups):
        wt = w.iloc[tr] if w is not None else None
        b=fit(X.iloc[tr],y.iloc[tr],wt)
        Xte=X.iloc[te].fillna(medians) if impute else X.iloc[te]
        oof[te]=b.predict_proba(Xte)[:,1]
    iso=IsotonicRegression(out_of_bounds="clip").fit(oof,y)
    base=fit(X,y,w)
    log(f"[2025] fitted+calibrated {variant}")
    return base, iso, medians, impute


def predict(base, iso, medians, impute, feats, gf):
    Xg = gf[feats].fillna(medians) if impute else gf[feats]
    return iso.predict(base.predict_proba(Xg)[:,1])


def fade(grid_ids, week, sampled):
    glat,glon=lonlat(grid_ids); coslat=np.cos(np.radians(np.nanmean(glat)))
    s={((w-1)%53)+1 for w in range(week-SEASON_WINDOW,week+SEASON_WINDOW+1)}
    samp=sampled[sampled.iso_week.isin(s)]
    if samp.empty: return np.zeros(len(grid_ids)), np.full(len(grid_ids),OPACITY_FLOOR)
    tree=cKDTree(np.c_[samp.cell_lon*coslat*111, samp.cell_lat*111])
    dch,_=tree.query(np.c_[glon*coslat*111, glat*111],k=1)
    conf=np.exp(-dch/FALLOFF_KM)
    return conf, OPACITY_FLOOR+(1-OPACITY_FLOOR)*conf


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec=json.load(open(DATA_DIR/"model_features.json")); feats,static_cols=spec["model_features"],spec["static"]
    table=pd.read_parquet(DATA_DIR/"weekly_model_table.parquet")
    bg=pd.read_parquet(BG_FILE) if BG_FILE.exists() else None
    static_per_cell=static_source(static_cols)
    sampled=table.groupby([GRID_ID_COL,"iso_week"])[["cell_lat","cell_lon"]].first().reset_index()

    # fit each model once (on ALL 2013-2018 data), reuse across weeks
    fitted={}
    for v in RUN_MODELS:
        if v=="xgboost" and xgb is None: log("skip xgboost"); continue
        if v.startswith("maxent") and MaxentModel is None: log(f"skip {v}"); continue
        if v=="maxent_vanilla" and bg is None: log("skip maxent_vanilla (no background)"); continue
        fitted[v]=fit_calibrated(v, table, bg, feats)

    # predict each week, assemble one long surface table
    out=[]
    for week in WEEKS:
        gf=build_week_features(week, feats, static_per_cell)
        lat,lon=lonlat(gf[GRID_ID_COL].to_numpy())
        conf,op=fade(gf[GRID_ID_COL].to_numpy(), week, sampled)
        wk=pd.DataFrame({GRID_ID_COL:gf[GRID_ID_COL],"iso_week":week,"lat":lat,"lon":lon,
                         "confidence_raw":conf,"opacity":op})
        for v,(base,iso,med,imp) in fitted.items():
            wk[f"prob_{v}"]=predict(base,iso,med,imp,feats,gf)
        log(f"[2025] week {week}: predicted {len(wk):,} cells for {len(fitted)} models")
        out.append(wk)

    surf=pd.concat(out, ignore_index=True)
    surf.to_parquet(OUT_DIR/"surface_2025_predictions.parquet", index=False)
    log(f"[done] wrote surface_2025_predictions.parquet ({len(surf):,} rows, weeks {WEEKS}) -> {OUT_DIR}")
    log("[note] NO metric attached (no 2025 catches). Render separately as a "
        "capability demonstration with an unverified-prediction caption.")


if __name__ == "__main__":
    run()