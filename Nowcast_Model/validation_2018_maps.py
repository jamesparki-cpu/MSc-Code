from __future__ import annotations
"""
validation_2018_maps.py  —  2018 forward-chained nowcast maps (XGBoost + RF).

Now renders THREE variants per week per model, so the 2018 set matches the 2025
set and adds the ground truth the 2025 maps cannot have:

  1. _full.png     statewide surface, sampling-proximity fade, NO overlay
                   -> directly comparable to the 2025 _full maps
  2. _masked.png   hard-masked to the surveyed footprint (<MASK_KM of a 2018
                   trap), NO overlay
                   -> directly comparable to the 2025 _masked maps
  3. _overlay.png  masked surface + OBSERVED presence/absence points, drawn on a
                   fixed close-up extent so the markers are legible
                   -> the validation figure: prediction meeting reality

Honest by construction: models are fit on 2013-2017 only (2017 held out for
isotonic calibration), and predict the chosen 2018 week's REAL weekly
conditions. 2018 is never trained on. The mask is defined by SURVEILLANCE
geometry, never by prediction values.

PER-MAP METRICS: computed on that week's own observed points (NOT the pooled CV
ROC) and written to validation_metrics_xgb_rf.csv. Weeks with few observed
ABSENCES give low-power metrics -- n_abs is in the CSV so you can flag them.

PERFORMANCE (vs the earlier single-week version):
  * each model is fit ONCE, not once per week
  * the 2018 daily file is read and gap-filled ONCE, then sliced per week
  * window aggregates are vectorised groupbys, not per-cell .apply
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR  = Path(cfg.get("weekly_xg_dir", "."))            # model table + features
DAILY_DIR = Path(cfg.get("parquet_dir", str(DATA_DIR)))    # OuterMerged2 daily files
OUT_DIR   = Path(cfg["nowcast_dir"]) / "Validation_Results"

TEST_YEAR   = 2018
TRAIN_YEARS = list(range(2013, TEST_YEAR))                 # 2013-2017
CALIB_YEAR  = 2017                                         # held out of the fit for isotonic
WEEKS  = [6, 20, 30, 32, 43]
MODELS = ["xgboost", "random_forest"]

MASK_KM  = 40.0                    # surveyed-footprint threshold
# Fixed extents. STATE_* is pinned on the _full AND _masked maps so they render
# at the same zoom (and therefore the same cell size) as the 2025 maps -- without
# this, matplotlib auto-zooms the masked map and the cells look blocky and gappy.
STATE_XLIM = (-88.0, -79.5)
STATE_YLIM = (24.3, 31.2)
MAP_XLIM = (-83.2, -80.3)          # close-up extent, used for the OVERLAY map only
MAP_YLIM = (25.8, 29.7)

# Which traps define the "surveyed footprint" for the fade and the hard mask:
#   True  -> ALL years 2013-2018. Matches how the 2025 maps were masked, and
#            reflects where the model's training support actually comes from.
#   False -> 2018 only, i.e. the traps actually active in the test year.
# Use True for a like-for-like pair with the 2025 maps; the 2018-only footprint
# is smaller (e.g. it drops the panhandle cluster) so the two would not be
# comparable. Whichever you pick, state it in the caption.
MASK_FROM_ALL_YEARS = True

GRID_ID_COL  = "Grid_ID"
CLIMATE_COLS = ["tmax", "tmin", "tmean", "prcp", "vpd"]
VEG_COLS     = ["EVI", "NDWI"]
FFILL_LIMIT  = {"EVI": 16, "NDWI": 8}
CLIM_FFILL_LIMIT = 3
LAG_WINDOWS  = (7, 14, 28)
RANDOM_STATE = 42

CMAP = "RdYlBu_r"; DPI = 130
MARKER_FULL = 6                    # statewide zoom: small cells read as a raster
MARKER_ZOOM = 9                    # close-up zoom: larger cells so there are no gaps
SEASON_WINDOW = 2; FALLOFF_KM = 120.0; OPACITY_FLOOR = 0.25

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss", tree_method="hist")
RF_PARAMS = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                 class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

PRETTY = {"xgboost": "XGBoost", "random_forest": "Random Forest"}

def log(m): print(m, flush=True)
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
def lonlat(g):
    lat = np.array([float(_GID.search(str(x)).group(1)) for x in g])
    lon = np.array([float(_GID.search(str(x)).group(2)) for x in g])
    return lat, lon
# ============================================================================


# ----- daily prep (ONCE) ----------------------------------------------------
def load_daily():
    """Read the 2018 daily file once, reindex every cell to a gap-free calendar
    and forward-fill. ffill is forward-only, so slicing to dates before a week's
    Monday afterwards introduces no leakage."""
    f = sorted(glob.glob(str(DAILY_DIR / f"*OuterMerged2*{TEST_YEAR}*.parquet")))[0]
    d = pd.read_parquet(f, columns=[GRID_ID_COL, "Date"] + CLIMATE_COLS + VEG_COLS)
    d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
    d = (d.sort_values([GRID_ID_COL, "Date"])
           .groupby([GRID_ID_COL, "Date"], as_index=False).first())
    cells = sorted(d[GRID_ID_COL].unique())
    idx = pd.date_range(d["Date"].min(), d["Date"].max(), freq="D")
    mi = pd.MultiIndex.from_product([cells, idx], names=[GRID_ID_COL, "Date"])
    d = d.set_index([GRID_ID_COL, "Date"]).reindex(mi).reset_index()
    g = d.groupby(GRID_ID_COL, sort=False)
    for c in CLIMATE_COLS:
        d[c] = g[c].transform(lambda s: s.ffill(limit=CLIM_FFILL_LIMIT))
    for c in VEG_COLS:
        d[c + "_ff"] = g[c].transform(lambda s: s.ffill(limit=FFILL_LIMIT[c]))
    log(f"[val] daily {TEST_YEAR}: {len(cells):,} cells x {len(idx)} days (gap-filled once)")
    return d, f


def static_per_cell(daily_file, static_cols):
    return (pd.read_parquet(daily_file, columns=[GRID_ID_COL] + static_cols)
              .groupby(GRID_ID_COL).first().reset_index())


def week_features(daily, week, feats, static_cells):
    """Statewide features for one 2018 week: trailing windows ending the day
    BEFORE the week's Monday (leak-proof), + static + seasonality."""
    mon = pd.Timestamp(date.fromisocalendar(TEST_YEAR, week, 1))
    parts = []
    for w in LAG_WINDOWS:
        win = daily[(daily.Date >= mon - pd.Timedelta(days=w)) & (daily.Date < mon)]
        g = win.groupby(GRID_ID_COL)
        p = pd.DataFrame({f"prcp_sum_{w}d": g["prcp"].sum(min_count=1),
                          f"tmean_mean_{w}d": g["tmean"].mean()})
        if w == 14:                       # the 14-day extras
            p["vpd_mean_14d"]  = g["vpd"].mean()
            p["tmax_max_14d"]  = g["tmax"].max()
            p["tmin_mean_14d"] = g["tmin"].mean()
            p["tmin_min_14d"]  = g["tmin"].min()
        parts.append(p)
    feat = pd.concat(parts, axis=1).reset_index()

    last = daily[daily.Date == mon - pd.Timedelta(days=1)][[GRID_ID_COL, "EVI_ff", "NDWI_ff"]]
    feat = feat.merge(last.rename(columns={"EVI_ff": "evi_level", "NDWI_ff": "ndwi_level"}),
                      on=GRID_ID_COL, how="left")
    feat = feat.merge(static_cells, on=GRID_ID_COL, how="left")
    doy = mon.dayofyear
    feat["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    feat["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    missing = [c for c in feats if c not in feat.columns]
    if missing:
        raise AssertionError(f"[val] week {week} missing features: {missing}")
    return feat[[GRID_ID_COL] + feats], mon


# ----- fitting (ONCE per model) ---------------------------------------------
def fit_calibrated(name, train, feats):
    """Fit on TRAIN_YEARS minus CALIB_YEAR, isotonic on CALIB_YEAR. 2018 unused."""
    X, y = train[feats], train["presence"].astype(int)
    w = np.sqrt(train["n_events"].clip(lower=1))
    medians = X.median(numeric_only=True)
    impute = (name == "random_forest")
    cal = (train["iso_year"] == CALIB_YEAR).to_numpy()

    Xf = X[~cal].fillna(medians) if impute else X[~cal]
    if name == "xgboost":
        spw = (y[~cal] == 0).sum() / max((y[~cal] == 1).sum(), 1)
        base = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    else:
        base = RandomForestClassifier(**RF_PARAMS)
    base.fit(Xf, y[~cal], sample_weight=w[~cal])

    Xc = X[cal].fillna(medians) if impute else X[cal]
    iso = IsotonicRegression(out_of_bounds="clip").fit(base.predict_proba(Xc)[:, 1], y[cal])
    log(f"[val] fitted {name} (fit {min(TRAIN_YEARS)}-{CALIB_YEAR-1}, calibrated on {CALIB_YEAR})")
    return base, iso, medians, impute


def predict_grid(model, feats, gf):
    base, iso, medians, impute = model
    Xg = gf[feats].fillna(medians) if impute else gf[feats]
    return iso.predict(base.predict_proba(Xg)[:, 1])


# ----- fade / mask ----------------------------------------------------------
def fade(grid_ids, week, sampled):
    glat, glon = lonlat(grid_ids); coslat = np.cos(np.radians(np.nanmean(glat)))
    s = {((w - 1) % 53) + 1 for w in range(week - SEASON_WINDOW, week + SEASON_WINDOW + 1)}
    samp = sampled[sampled.iso_week.isin(s)]
    if samp.empty:
        return np.full(len(grid_ids), OPACITY_FLOOR)
    tree = cKDTree(np.c_[samp.cell_lon * coslat * 111, samp.cell_lat * 111])
    dch, _ = tree.query(np.c_[glon * coslat * 111, glat * 111], k=1)
    return OPACITY_FLOOR + (1 - OPACITY_FLOOR) * np.exp(-dch / FALLOFF_KM)


def trap_mask(grid_ids, week, sampled):
    """True = keep (within MASK_KM of a trap active that season). Surveillance
    geometry ONLY -- never selected on whether the prediction was correct."""
    if MASK_KM is None:
        return np.ones(len(grid_ids), dtype=bool)
    glat, glon = lonlat(grid_ids); coslat = np.cos(np.radians(np.nanmean(glat)))
    s = {((w - 1) % 53) + 1 for w in range(week - SEASON_WINDOW, week + SEASON_WINDOW + 1)}
    samp = sampled[sampled.iso_week.isin(s)]
    if samp.empty:
        return np.zeros(len(grid_ids), dtype=bool)
    tree = cKDTree(np.c_[samp.cell_lon * coslat * 111, samp.cell_lat * 111])
    dkm, _ = tree.query(np.c_[glon * coslat * 111, glat * 111], k=1)
    return dkm <= MASK_KM


# ----- rendering ------------------------------------------------------------
def render(grid, title, path, obs=None, zoom=False):
    """zoom=False -> pinned statewide extent, small cells (matches the 2025 maps).
    zoom=True  -> close-up extent, larger cells, for the overlay figure."""
    lat, lon = lonlat(grid[GRID_ID_COL].to_numpy())
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    ax.scatter(lon, lat, c=grid["pred"], s=(MARKER_ZOOM if zoom else MARKER_FULL),
               marker="s", cmap=CMAP, norm=Normalize(0, 1), linewidths=0,
               alpha=grid["opacity"].to_numpy())
    if obs is not None:
        pres, absn = obs[obs.presence == 1], obs[obs.presence == 0]
        ax.scatter(pres.cell_lon, pres.cell_lat, s=70, marker="o", facecolor="none",
                   edgecolor="black", linewidths=1.3,
                   label=f"observed presence (n={len(pres)})")
        ax.scatter(absn.cell_lon, absn.cell_lat, s=80, marker="x", color="black",
                   linewidths=1.6, label=f"observed absence (n={len(absn)})")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    if zoom:
        ax.set_xlim(*MAP_XLIM); ax.set_ylim(*MAP_YLIM)
    else:
        ax.set_xlim(*STATE_XLIM); ax.set_ylim(*STATE_YLIM)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(title, fontsize=9.5)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1)); sm.set_array([])
    # pad keeps the colourbar clear of the (two-line) title
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.03).set_label("predicted suitability")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")   # bbox_inches stops title clipping
    plt.close(fig)
    log(f"[val] wrote {path.name}")


# ----- per-map metrics ------------------------------------------------------
def map_metrics(grid, obs, name, week):
    """Metrics on THIS week's observed points only -- the honest caption number."""
    gp = grid[[GRID_ID_COL, "pred"]].merge(obs, on=GRID_ID_COL)
    pres, absn = gp[gp.presence == 1], gp[gp.presence == 0]
    two = gp.presence.nunique() > 1        # ROC undefined with a single class
    prev = gp.presence.mean() if len(gp) else np.nan
    return dict(
        model=name, week=week, n=len(gp), n_pres=len(pres), n_abs=len(absn),
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
    foot = table if MASK_FROM_ALL_YEARS else table[table.iso_year == TEST_YEAR]
    sampled = (foot.groupby([GRID_ID_COL, "iso_week"])[["cell_lat", "cell_lon"]]
                   .first().reset_index())
    log(f"[val] footprint from {'all years 2013-2018' if MASK_FROM_ALL_YEARS else f'{TEST_YEAR} only'}"
        f" ({sampled[GRID_ID_COL].nunique():,} cells)")

    daily, daily_file = load_daily()
    static_cells = static_per_cell(daily_file, static_cols)

    fitted = {}
    for name in MODELS:
        if name == "xgboost" and xgb is None:
            log("[val] skip xgboost (not installed)"); continue
        fitted[name] = fit_calibrated(name, train, feats)

    rows = []
    for week in WEEKS:
        gf, mon = week_features(daily, week, feats, static_cells)
        ids = gf[GRID_ID_COL].to_numpy()
        op = fade(ids, week, sampled)
        keep = trap_mask(ids, week, sampled)
        obs = table[(table.iso_year == TEST_YEAR) & (table.iso_week == week)][
            [GRID_ID_COL, "cell_lat", "cell_lon", "presence"]]
        log(f"[val] week {week}: {len(gf):,} cells | masked {int(keep.sum()):,} "
            f"(<{MASK_KM:.0f} km) | observed {len(obs)}")

        for name, model in fitted.items():
            label = PRETTY.get(name, name)
            grid = gf[[GRID_ID_COL]].copy()
            grid["pred"] = predict_grid(model, feats, gf)
            grid["opacity"] = op
            stem = f"validation_{TEST_YEAR}_week{week:02d}_{name}"
            wk = f"ISO week {week} {TEST_YEAR} (~{mon.strftime('%d %b')})"
            trained = f"trained {min(TRAIN_YEARS)}–{max(TRAIN_YEARS)}"

            # 1. full statewide (comparable to the 2025 _full maps)
            render(grid,
                   f"Cx. nigripalpus — predicted suitability ({label})\n{wk} · {trained}",
                   OUT_DIR / f"{stem}_full.png")
            # 2. masked to the surveyed footprint (comparable to 2025 _masked)
            render(grid[keep].reset_index(drop=True),
                   f"Cx. nigripalpus — predicted suitability ({label}), surveyed footprint\n"
                   f"{wk} · masked <{MASK_KM:.0f} km of a trap · {trained}",
                   OUT_DIR / f"{stem}_masked.png")
            # 3. the validation figure: masked + observed catches, close-up
            render(grid[keep].reset_index(drop=True),
                   f"Cx. nigripalpus — {label} nowcast vs observed catches\n"
                   f"{wk} · {trained} (2018 never seen)",
                   OUT_DIR / f"{stem}_overlay.png", obs=obs, zoom=True)

            m = map_metrics(grid, obs, name, week)
            rows.append(m)
            log(f"[val] {name} week {week}: ROC {m['roc_auc']} (n={m['n']}, "
                f"{m['n_abs']} abs | pres suit {m['mean_suit_pres']} vs abs {m['mean_suit_abs']})")

    out_csv = OUT_DIR / "validation_metrics_xgb_rf.csv"
    mdf = pd.DataFrame(rows)
    if out_csv.exists():
        mdf = (pd.concat([pd.read_csv(out_csv), mdf], ignore_index=True)
                 .drop_duplicates(["model", "week"], keep="last"))
    mdf.to_csv(out_csv, index=False)
    log(f"\n[done] {len(rows)*3} maps + validation_metrics_xgb_rf.csv -> {OUT_DIR}")


if __name__ == "__main__":
    run()