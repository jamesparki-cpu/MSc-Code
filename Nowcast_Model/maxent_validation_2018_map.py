from __future__ import annotations
"""
maxent_validation_map.py  —  MaxEnt version of the 2018 forward-chained
validation map, with PER-MAP metrics.

Mirrors validation_map_2018.py but uses elapid MaxentModel (unweighted, imputed).
Model fit on 2013-2017 only, predicts the chosen 2018 week's REAL conditions,
overlays observed catches. One MaxEnt fit per map (minutes, not the CV hours).

VARIANT:
  "targetgroup" (default) — trains on the model table (presence + target-group
      absences), a clean like-for-like with XGB/RF.
  "vanilla" — trains on presences + random background (needs vanilla_background).

Writes, alongside each map, a per-map metrics row (ROC / PR-lift / BSS / n /
mean-suitability at presence vs absence) computed on THAT week's observed points
only — the honest number to caption the figure with (NOT the pooled CV ROC).

OUTPUT (to validation_dir)
  maxent_validation_2018_week<NN>_<variant>.png
  validation_metrics_maxent.csv
"""
import json, glob, re
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from elapid import MaxentModel

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR  = Path(cfg.get("weekly_xg_dir", "."))
DAILY_DIR = Path(cfg.get("parquet_dir", str(DATA_DIR)))
BG_FILE   = Path(cfg.get("nowcast_background",
                         str(Path(cfg.get("maxent_dir", str(DATA_DIR))) / "vanilla_background.parquet")))
OUT_DIR    = Path(cfg["nowcast_dir"]) / "Validation_Results"  # where the maps go

VARIANT = "targetgroup"          # or "vanilla"
TEST_YEAR = 2018
TRAIN_YEARS = list(range(2013, TEST_YEAR))
WEEKS = [6, 20, 32, 43]                      # extend to [4, 17, 30, 43] for a four-season set
MASK_KM = 40.0                   # set e.g. 40.0 to hard-mask beyond N km of a trap

GRID_ID_COL = "Grid_ID"
CLIMATE_COLS = ["tmax","tmin","tmean","prcp","vpd"]; VEG_COLS = ["EVI","NDWI"]
FFILL_LIMIT = {"EVI":16,"NDWI":8}; CLIM_FFILL_LIMIT = 3; LAG_WINDOWS = (7,14,28)
CMAP="RdYlBu_r"; MARKER=6; DPI=130
SEASON_WINDOW=2; FALLOFF_KM=120.0; OPACITY_FLOOR=0.25

def log(m): print(m, flush=True)
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
def lonlat(g):
    lat=np.array([float(_GID.search(str(x)).group(1)) for x in g])
    lon=np.array([float(_GID.search(str(x)).group(2)) for x in g]); return lat,lon
def make_maxent():
    return MaxentModel(feature_types=["linear","quadratic","hinge"], transform="cloglog", clamp=True)
# ============================================================================


def build_week_features(week, feats, static_cols):
    mon = pd.Timestamp(date.fromisocalendar(TEST_YEAR, week, 1))
    f = sorted(glob.glob(str(DAILY_DIR / f"*OuterMerged2*{TEST_YEAR}*.parquet")))[0]
    d = pd.read_parquet(f, columns=[GRID_ID_COL,"Date"]+CLIMATE_COLS+VEG_COLS)
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
    feat=pd.DataFrame(agg).reset_index()
    stat=pd.read_parquet(f, columns=[GRID_ID_COL]+static_cols).groupby(GRID_ID_COL).first().reset_index()
    feat=feat.merge(stat,on=GRID_ID_COL,how="left")
    doy=mon.dayofyear; feat["sin_doy"]=np.sin(2*np.pi*doy/365.25); feat["cos_doy"]=np.cos(2*np.pi*doy/365.25)
    return feat[[GRID_ID_COL]+feats], mon


def training_set(feats):
    """2013-2017 training rows for the chosen variant."""
    table = pd.read_parquet(DATA_DIR / "weekly_model_table.parquet")
    if VARIANT == "vanilla":
        pres = table[(table.presence==1)&(table.iso_year.isin(TRAIN_YEARS))]
        bg = pd.read_parquet(BG_FILE); bg = bg[bg.iso_year.isin(TRAIN_YEARS)].copy(); bg["presence"]=0
        keep=[GRID_ID_COL,"iso_year","presence"]+feats
        return pd.concat([pres[keep], bg[keep]], ignore_index=True)
    return table[table.iso_year.isin(TRAIN_YEARS)].copy()   # targetgroup


def calibrated_predict(train, feats, grid_feat):
    """Unweighted MaxEnt fit on 2013-2016, isotonic on 2017, predict grid (imputed)."""
    X, y = train[feats], train["presence"].astype(int)
    medians = X.median(numeric_only=True)
    cal = train["iso_year"] == 2017
    base = make_maxent(); base.fit(X[~cal].fillna(medians), y[~cal])
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        base.predict_proba(X[cal].fillna(medians))[:,1], y[cal])
    return iso.predict(base.predict_proba(grid_feat[feats].fillna(medians))[:,1])


def fade(grid_ids, week, sampled):
    glat,glon=lonlat(grid_ids); coslat=np.cos(np.radians(np.nanmean(glat)))
    s={((w-1)%53)+1 for w in range(week-SEASON_WINDOW,week+SEASON_WINDOW+1)}
    samp=sampled[sampled.iso_week.isin(s)]
    if samp.empty: return np.full(len(grid_ids),OPACITY_FLOOR)
    tree=cKDTree(np.c_[samp.cell_lon*coslat*111, samp.cell_lat*111])
    dch,_=tree.query(np.c_[glon*coslat*111, glat*111],k=1)
    return OPACITY_FLOOR+(1-OPACITY_FLOOR)*np.exp(-dch/FALLOFF_KM)

def trap_mask(grid_ids, week, sampled):
    if MASK_KM is None: return np.ones(len(grid_ids),dtype=bool)
    glat,glon=lonlat(grid_ids); coslat=np.cos(np.radians(np.nanmean(glat)))
    s={((w-1)%53)+1 for w in range(week-SEASON_WINDOW,week+SEASON_WINDOW+1)}
    samp=sampled[sampled.iso_week.isin(s)]
    if samp.empty: return np.zeros(len(grid_ids),dtype=bool)
    tree=cKDTree(np.c_[samp.cell_lon*coslat*111, samp.cell_lat*111])
    dkm,_=tree.query(np.c_[glon*coslat*111, glat*111],k=1)
    return dkm<=MASK_KM


def map_metrics(grid, obs, week):
    """Metrics on THIS week's observed points only — the caption number."""
    gp = grid[[GRID_ID_COL,"pred"]].merge(obs, on=GRID_ID_COL)
    pres, absn = gp[gp.presence==1], gp[gp.presence==0]
    two = gp.presence.nunique() > 1
    prev = gp.presence.mean()
    return dict(model=f"maxent_{VARIANT}", week=week, n=len(gp),
                n_pres=len(pres), n_abs=len(absn),
                roc_auc=round(roc_auc_score(gp.presence, gp.pred),3) if two else np.nan,
                pr_lift=round(average_precision_score(gp.presence, gp.pred)-prev,3) if two else np.nan,
                bss=round(1-brier_score_loss(gp.presence,gp.pred)/brier_score_loss(gp.presence,np.full(len(gp),prev)),3) if two else np.nan,
                mean_suit_pres=round(pres.pred.mean(),3) if len(pres) else np.nan,
                mean_suit_abs=round(absn.pred.mean(),3) if len(absn) else np.nan)


def render(grid, obs, week, mon, path):
    lat,lon=lonlat(grid[GRID_ID_COL].to_numpy())
    fig,ax=plt.subplots(figsize=(7.5,8.5))
    ax.scatter(lon,lat,c=grid["pred"],s=MARKER,marker="s",cmap=CMAP,norm=Normalize(0,1),
               linewidths=0,alpha=grid["opacity"].to_numpy())
    pres,absn=obs[obs.presence==1],obs[obs.presence==0]
    ax.scatter(pres.cell_lon,pres.cell_lat,s=42,marker="o",facecolor="none",edgecolor="black",
               linewidths=1.3,label=f"observed presence (n={len(pres)})")
    ax.scatter(absn.cell_lon,absn.cell_lat,s=48,marker="x",color="black",linewidths=1.6,
               label=f"observed absence (n={len(absn)})")
    ax.set_aspect(1/np.cos(np.radians(lat.mean()))); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Cx. nigripalpus — MaxEnt ({VARIANT}) nowcast vs observed catches\n"
                 f"ISO week {week} {TEST_YEAR} (~{mon.strftime('%d %b')}) · trained on 2013–2017 only",fontsize=10.5)
    sm=plt.cm.ScalarMappable(cmap=CMAP,norm=Normalize(0,1)); sm.set_array([])
    fig.colorbar(sm,ax=ax,shrink=0.7).set_label("predicted suitability")
    ax.legend(loc="upper left",fontsize=8,framealpha=0.9)
    fig.tight_layout(); fig.savefig(path,dpi=DPI); plt.close(fig)
    log(f"[val] wrote {path.name}")


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec=json.load(open(DATA_DIR/"model_features.json")); feats,static_cols=spec["model_features"],spec["static"]
    table=pd.read_parquet(DATA_DIR/"weekly_model_table.parquet")
    train=training_set(feats)
    sampled=table[table.iso_year==TEST_YEAR].groupby([GRID_ID_COL,"iso_week"])[["cell_lat","cell_lon"]].first().reset_index()
    rows=[]
    for week in WEEKS:
        gf,mon=build_week_features(week,feats,static_cols)
        keep=trap_mask(gf[GRID_ID_COL].to_numpy(),week,sampled)
        gf=gf[keep].reset_index(drop=True)
        op=fade(gf[GRID_ID_COL].to_numpy(),week,sampled)
        obs=table[(table.iso_year==TEST_YEAR)&(table.iso_week==week)][[GRID_ID_COL,"cell_lat","cell_lon","presence"]]
        grid=gf[[GRID_ID_COL]].copy(); grid["pred"]=calibrated_predict(train,feats,gf); grid["opacity"]=op
        m=map_metrics(grid,obs,week); rows.append(m)
        log(f"[val] maxent_{VARIANT} week{week}: ROC {m['roc_auc']} (n={m['n']}, "
            f"pres suit {m['mean_suit_pres']} vs abs {m['mean_suit_abs']})")
        render(grid,obs,week,mon, OUT_DIR/f"maxent_validation_{TEST_YEAR}_week{week:02d}_{VARIANT}.png")
    mdf=pd.DataFrame(rows); out_csv=OUT_DIR/"validation_metrics_maxent.csv"
    if out_csv.exists():
        mdf=pd.concat([pd.read_csv(out_csv),mdf],ignore_index=True).drop_duplicates(["model","week"],keep="last")
    mdf.to_csv(out_csv,index=False)
    log(f"[done] maps + validation_metrics_maxent.csv -> {OUT_DIR}")


if __name__ == "__main__":
    run()