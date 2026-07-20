from __future__ import annotations
"""
validation_map_2018.py  —  the map that DEMONSTRATES nowcast success.

Honest forward-chained validation:
  * models fit on 2013-2017 ONLY (never see 2018),
  * predict the chosen 2018 week's REAL weekly conditions statewide (real-year
    lag features, not climatology),
  * overlay that week's OBSERVED trap outcomes (presence vs absence) as points.

Where high-suitability colour sits under presence points and low colour under
absence points, that's visible success -- prediction meeting reality on data the
model never trained on. This directly visualises the forward-forecast result
(ROC ~0.84 for 2018).

Built so extending to one-week-per-season is trivial: set WEEKS = [4, 17, 30, 43].

STYLE matches the other maps (RdYlBu_r, square markers, latitude aspect). The
suitability surface uses the sampling-proximity opacity fade; the overlaid
observation points are drawn at full opacity so they read clearly.

OUTPUT (to config nowcast_results_dir or validation_dir)
  validation_2018_week<NN>_<model>.png  per week per model
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR   = Path(cfg.get("weekly_xg_dir", "."))            # model table + features
DAILY_DIR  = Path(cfg.get("parquet_dir", str(DATA_DIR)))   # OuterMerged2 daily files
OUT_DIR    = Path(cfg["nowcast_dir"]) / "Validation_Results"  # where the maps go

TEST_YEAR  = 2018
TRAIN_YEARS = list(range(2013, TEST_YEAR))                 # 2013-2017
WEEKS = [6, 20, 32, 43]                                               # extend: [4, 17, 30, 43]
MODELS = ["xgboost", "random_forest"]

GRID_ID_COL = "Grid_ID"
CLIMATE_COLS = ["tmax", "tmin", "tmean", "prcp", "vpd"]
VEG_COLS = ["EVI", "NDWI"]
FFILL_LIMIT = {"EVI": 16, "NDWI": 8}
CLIM_FFILL_LIMIT = 3
LAG_WINDOWS = (7, 14, 28)
RANDOM_STATE = 42
MASK_KM = 40.0

CMAP = "RdYlBu_r"; MARKER = 6; DPI = 130
SEASON_WINDOW = 2; FALLOFF_KM = 120.0; OPACITY_FLOOR = 0.25

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss", tree_method="hist")
RF_PARAMS = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                 class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

def log(m): print(m, flush=True)
def minp(w): return max(1, w-2)
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
def lonlat(g):
    lat=np.array([float(_GID.search(str(x)).group(1)) for x in g])
    lon=np.array([float(_GID.search(str(x)).group(2)) for x in g]); return lat,lon
# ============================================================================


def build_week_features(week, feats, static_cols):
    """Real 2018-week features statewide: Stage-2 leak-proof lags ending the day
    before the week's Monday, + static + seasonality. Only the target week."""
    mon = pd.Timestamp(date.fromisocalendar(TEST_YEAR, week, 1))
    f = sorted(glob.glob(str(DAILY_DIR / f"*OuterMerged2*{TEST_YEAR}*.parquet")))[0]
    d = pd.read_parquet(f, columns=[GRID_ID_COL, "Date"] + CLIMATE_COLS + VEG_COLS)
    d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
    d = d[d["Date"] < mon]                                  # only days strictly before Monday
    d = (d.sort_values([GRID_ID_COL, "Date"])
           .groupby([GRID_ID_COL, "Date"], as_index=False).first())
    cells = sorted(d[GRID_ID_COL].unique())
    idx = pd.date_range(d["Date"].min(), mon - pd.Timedelta(days=1), freq="D")
    mi = pd.MultiIndex.from_product([cells, idx], names=[GRID_ID_COL, "Date"])
    d = d.set_index([GRID_ID_COL, "Date"]).reindex(mi).reset_index()
    g = d.groupby(GRID_ID_COL, sort=False)
    for c in CLIMATE_COLS:
        d[c] = g[c].transform(lambda s: s.ffill(limit=CLIM_FFILL_LIMIT))
    for c in VEG_COLS:
        d[c+"_ff"] = g[c].transform(lambda s: s.ffill(limit=FFILL_LIMIT[c]))

    # the feature row for each cell = the LAST day (Sunday before Monday) rolling value
    g = d.groupby(GRID_ID_COL, sort=False)
    feat = pd.DataFrame({GRID_ID_COL: cells})
    agg = {}
    for w in LAG_WINDOWS:
        agg[f"prcp_sum_{w}d"]   = g["prcp"].apply(lambda s, w=w: s.iloc[-w:].sum(min_count=1))
        agg[f"tmean_mean_{w}d"] = g["tmean"].apply(lambda s, w=w: s.iloc[-w:].mean())
    agg["vpd_mean_14d"] = g["vpd"].apply(lambda s: s.iloc[-14:].mean())
    agg["tmax_max_14d"] = g["tmax"].apply(lambda s: s.iloc[-14:].max())
    agg["tmin_mean_14d"]= g["tmin"].apply(lambda s: s.iloc[-14:].mean())
    agg["tmin_min_14d"] = g["tmin"].apply(lambda s: s.iloc[-14:].min())
    agg["evi_level"]    = g["EVI_ff"].apply(lambda s: s.iloc[-1])
    agg["ndwi_level"]   = g["NDWI_ff"].apply(lambda s: s.iloc[-1])
    dyn = pd.DataFrame(agg).reset_index()
    feat = feat.merge(dyn, on=GRID_ID_COL, how="left")

    # static (first-epoch) from the same daily file + seasonality from the Monday
    stat = pd.read_parquet(f, columns=[GRID_ID_COL]+static_cols).groupby(GRID_ID_COL).first().reset_index()
    feat = feat.merge(stat, on=GRID_ID_COL, how="left")
    doy = mon.dayofyear
    feat["sin_doy"] = np.sin(2*np.pi*doy/365.25); feat["cos_doy"] = np.cos(2*np.pi*doy/365.25)
    return feat[[GRID_ID_COL]+feats], mon


def fit_model(name, Xtr, ytr, wtr, medians, impute):
    if name == "xgboost":
        spw = (ytr==0).sum()/max((ytr==1).sum(),1)
        m = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    else:
        m = RandomForestClassifier(**RF_PARAMS)
    Xf = Xtr.fillna(medians) if impute else Xtr
    m.fit(Xf, ytr, sample_weight=wtr)
    return m


def calibrated_predict(name, train, feats, Xgrid, impute):
    """Fit on 2013-2017, isotonic on a held-out 2017 slice, predict the grid."""
    X, y = train[feats], train["presence"].astype(int)
    w = np.sqrt(train["n_events"].clip(lower=1))
    medians = X.median(numeric_only=True)
    # calibration slice = 2017 held out from the fit
    cal = train["iso_year"] == 2017
    base = fit_model(name, X[~cal], y[~cal], w[~cal], medians, impute)
    Xc = X[cal].fillna(medians) if impute else X[cal]
    iso = IsotonicRegression(out_of_bounds="clip").fit(base.predict_proba(Xc)[:,1], y[cal])
    Xg = Xgrid[feats].fillna(medians) if impute else Xgrid[feats]
    return iso.predict(base.predict_proba(Xg)[:,1])


def fade(grid_ids, week, sampled):
    glat, glon = lonlat(grid_ids); coslat=np.cos(np.radians(np.nanmean(glat)))
    s = {((w-1)%53)+1 for w in range(week-SEASON_WINDOW, week+SEASON_WINDOW+1)}
    samp = sampled[sampled.iso_week.isin(s)]
    if samp.empty: return np.full(len(grid_ids), OPACITY_FLOOR)
    tree = cKDTree(np.c_[samp.cell_lon*coslat*111, samp.cell_lat*111])
    dch,_ = tree.query(np.c_[glon*coslat*111, glat*111], k=1)
    return OPACITY_FLOOR + (1-OPACITY_FLOOR)*np.exp(-dch/FALLOFF_KM)

def trap_mask(grid_ids, week, sampled):
    """True = keep (within MASK_KM of a trap active that season); False = hide.
    Defined by SURVEILLANCE geometry only — never by prediction correctness."""
    glat, glon = lonlat(grid_ids); coslat = np.cos(np.radians(np.nanmean(glat)))
    s = {((w-1) % 53)+1 for w in range(week-SEASON_WINDOW, week+SEASON_WINDOW+1)}
    samp = sampled[sampled.iso_week.isin(s)]
    if samp.empty:
        return np.zeros(len(grid_ids), dtype=bool)
    tree = cKDTree(np.c_[samp.cell_lon*coslat*111, samp.cell_lat*111])
    dkm, _ = tree.query(np.c_[glon*coslat*111, glat*111], k=1)
    return dkm <= MASK_KM

def render(grid, obs, name, week, mon, path):
    lat, lon = lonlat(grid[GRID_ID_COL].to_numpy())
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    ax.scatter(lon, lat, c=grid["pred"], s=MARKER, marker="s", cmap=CMAP,
               norm=Normalize(0,1), linewidths=0, alpha=grid["opacity"].to_numpy())
    # observed trap outcomes on top, full opacity, outlined
    pres = obs[obs.presence==1]; absn = obs[obs.presence==0]
    ax.scatter(pres.cell_lon, pres.cell_lat, s=42, marker="o", facecolor="none",
               edgecolor="black", linewidths=1.3, label=f"observed presence (n={len(pres)})")
    ax.scatter(absn.cell_lon, absn.cell_lat, s=48, marker="x", color="black",
               linewidths=1.6, label=f"observed absence (n={len(absn)})")
    ax.set_aspect(1/np.cos(np.radians(lat.mean())))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Cx. nigripalpus — {name} nowcast vs observed catches\n"
                 f"ISO week {week} {TEST_YEAR} (~{mon.strftime('%d %b')}) · "
                 f"model trained on 2013–2017 only", fontsize=10.5)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0,1)); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7).set_label("predicted suitability")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)
    log(f"[val] wrote {path.name}")

def map_metrics(grid, obs, name, week):
    """Metrics on THIS week's observed points only — the honest caption number."""
    gp = grid[[GRID_ID_COL, "pred"]].merge(obs, on=GRID_ID_COL)
    pres, absn = gp[gp.presence == 1], gp[gp.presence == 0]
    two = gp.presence.nunique() > 1          # ROC undefined if only one class present
    prev = gp.presence.mean()
    return dict(model=name, week=week, n=len(gp), n_pres=len(pres), n_abs=len(absn),
        roc_auc=round(roc_auc_score(gp.presence, gp.pred), 3) if two else np.nan,
        pr_lift=round(average_precision_score(gp.presence, gp.pred) - prev, 3) if two else np.nan,
        bss=round(1 - brier_score_loss(gp.presence, gp.pred) /
                  brier_score_loss(gp.presence, np.full(len(gp), prev)), 3) if two else np.nan,
        mean_suit_pres=round(pres.pred.mean(), 3) if len(pres) else np.nan,
        mean_suit_abs=round(absn.pred.mean(), 3) if len(absn) else np.nan)

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = json.load(open(DATA_DIR / "model_features.json"))
    feats, static_cols = spec["model_features"], spec["static"]
    table = pd.read_parquet(DATA_DIR / "weekly_model_table.parquet")
    train = table[table.iso_year.isin(TRAIN_YEARS)].copy()
    sampled = table[table.iso_year==TEST_YEAR].groupby([GRID_ID_COL,"iso_week"])[["cell_lat","cell_lon"]].first().reset_index()
    rows = []

    for week in WEEKS:
        grid_feat, mon = build_week_features(week, feats, static_cols)
        obs = table[(table.iso_year==TEST_YEAR)&(table.iso_week==week)][[GRID_ID_COL,"cell_lat","cell_lon","presence"]]
        op = fade(grid_feat[GRID_ID_COL].to_numpy(), week, sampled)
        keep = trap_mask(grid_feat[GRID_ID_COL].to_numpy(), week, sampled)
        grid_feat = grid_feat[keep].reset_index(drop=True)
        op = op[keep]
        log(f"[val] hard mask: kept {keep.sum()}/{len(keep)} cells within {MASK_KM} km of a trap")
        for name in MODELS:
            if name=="xgboost" and xgb is None: continue
            grid = grid_feat[[GRID_ID_COL]].copy()
            grid["pred"] = calibrated_predict(name, train, feats, grid_feat, impute=(name=="random_forest"))
            grid["opacity"] = op
            m = map_metrics(grid, obs, name, week)
            rows.append(m)
            log(f"[val] {name} week{week}: ROC {m['roc_auc']} (n={m['n']}, "
                f"pres suit {m['mean_suit_pres']} vs abs {m['mean_suit_abs']})")
            render(grid, obs, {"xgboost":"XGBoost","random_forest":"Random Forest"}[name],
                   week, mon, OUT_DIR / f"validation_{TEST_YEAR}_week{week:02d}_{name}.png")
    pd.DataFrame(rows).to_csv(OUT_DIR / "validation_metrics_xgb_rf.csv", index=False)
    log(f"[done] validation maps -> {OUT_DIR}")


if __name__ == "__main__":
    run()