from __future__ import annotations
"""
maxent_validation_2018_map.py  —  MaxEnt version of the 2018 nowcast maps.

Mirrors validation_2018_maps.py exactly (same extents, same mask, same fade, same
per-map metrics), so the MaxEnt and tree-model figures are directly comparable.
Renders THREE variants per week:

  1. _full.png     statewide surface, fade, no overlay      -> pairs with 2025 _full
  2. _masked.png   surveyed footprint only, no overlay      -> pairs with 2025 _masked
  3. _overlay.png  masked + OBSERVED presence/absence, close-up  -> the validation

VARIANT:
  "targetgroup" — trains on the model table (presence + target-group absences);
                  the clean like-for-like with XGB/RF.
  "vanilla"     — trains on presences + random background (needs the background
                  file). NOTE it answers a different question (presence vs the
                  average Florida landscape), so caption it separately.

MaxEnt is fit ONCE here, not once per week -- with 5 weeks that is the difference
between ~1 fit and ~5 fits, i.e. minutes rather than a quarter of an hour.
2018 is never trained on (fit 2013-2016, isotonic calibration on 2017).
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
OUT_DIR   = Path(cfg["nowcast_dir"]) / "Validation_Results"

VARIANT     = "targetgroup"        # or "vanilla"
TEST_YEAR   = 2018
TRAIN_YEARS = list(range(2013, TEST_YEAR))
CALIB_YEAR  = 2017
WEEKS       = [6, 20, 30, 32, 43]

MASK_KM = 40.0
STATE_XLIM = (-88.0, -79.5)        # pinned on _full and _masked (matches 2025)
STATE_YLIM = (24.3, 31.2)
MAP_XLIM = (-83.2, -80.3)          # close-up, for the _overlay figure only
MAP_YLIM = (25.8, 29.7)
MASK_FROM_ALL_YEARS = True         # keep True to match the 2025 masked footprint

GRID_ID_COL  = "Grid_ID"
CLIMATE_COLS = ["tmax", "tmin", "tmean", "prcp", "vpd"]
VEG_COLS     = ["EVI", "NDWI"]
FFILL_LIMIT  = {"EVI": 16, "NDWI": 8}
CLIM_FFILL_LIMIT = 3
LAG_WINDOWS  = (7, 14, 28)

CMAP = "RdYlBu_r"; DPI = 130
MARKER_FULL = 6; MARKER_ZOOM = 9
SEASON_WINDOW = 2; FALLOFF_KM = 120.0; OPACITY_FLOOR = 0.25

def log(m): print(m, flush=True)
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
def lonlat(g):
    lat = np.array([float(_GID.search(str(x)).group(1)) for x in g])
    lon = np.array([float(_GID.search(str(x)).group(2)) for x in g])
    return lat, lon
def make_maxent():
    return MaxentModel(feature_types=["linear", "quadratic", "hinge"],
                       transform="cloglog", clamp=True)
# ============================================================================


def load_daily():
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


def week_features(daily, week, feats, static_cells):
    mon = pd.Timestamp(date.fromisocalendar(TEST_YEAR, week, 1))
    parts = []
    for w in LAG_WINDOWS:
        win = daily[(daily.Date >= mon - pd.Timedelta(days=w)) & (daily.Date < mon)]
        g = win.groupby(GRID_ID_COL)
        p = pd.DataFrame({f"prcp_sum_{w}d": g["prcp"].sum(min_count=1),
                          f"tmean_mean_{w}d": g["tmean"].mean()})
        if w == 14:
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


def training_set(feats):
    """2013-2017 rows for the chosen variant."""
    table = pd.read_parquet(DATA_DIR / "weekly_model_table.parquet")
    if VARIANT == "vanilla":
        pres = table[(table.presence == 1) & (table.iso_year.isin(TRAIN_YEARS))]
        bg = pd.read_parquet(BG_FILE)
        bg = bg[bg.iso_year.isin(TRAIN_YEARS)].copy(); bg["presence"] = 0
        keep = [GRID_ID_COL, "iso_year", "presence"] + feats
        return pd.concat([pres[keep], bg[keep]], ignore_index=True)
    return table[table.iso_year.isin(TRAIN_YEARS)].copy()


def fit_calibrated(train, feats):
    """ONE unweighted MaxEnt fit on 2013-2016 + isotonic on 2017. Reused for all weeks."""
    X, y = train[feats], train["presence"].astype(int)
    medians = X.median(numeric_only=True)
    cal = (train["iso_year"] == CALIB_YEAR).to_numpy()
    base = make_maxent()
    base.fit(X[~cal].fillna(medians), y[~cal])
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        base.predict_proba(X[cal].fillna(medians))[:, 1], y[cal])
    log(f"[val] fitted maxent_{VARIANT} (fit {min(TRAIN_YEARS)}-{CALIB_YEAR-1}, "
        f"calibrated on {CALIB_YEAR})")
    return base, iso, medians


def predict_grid(model, feats, gf):
    base, iso, medians = model
    return iso.predict(base.predict_proba(gf[feats].fillna(medians))[:, 1])


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
    """Surveillance geometry only -- never selected on prediction correctness."""
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


def render(grid, title, path, obs=None, zoom=False):
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
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.03).set_label("predicted suitability")
    fig.tight_layout(); fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    log(f"[val] wrote {path.name}")


def map_metrics(grid, obs, week):
    """Metrics on THIS week's observed points only -- the caption number."""
    gp = grid[[GRID_ID_COL, "pred"]].merge(obs, on=GRID_ID_COL)
    pres, absn = gp[gp.presence == 1], gp[gp.presence == 0]
    two = gp.presence.nunique() > 1
    prev = gp.presence.mean() if len(gp) else np.nan
    return dict(
        model=f"maxent_{VARIANT}", week=week, n=len(gp),
        n_pres=len(pres), n_abs=len(absn),
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

    foot = table if MASK_FROM_ALL_YEARS else table[table.iso_year == TEST_YEAR]
    sampled = (foot.groupby([GRID_ID_COL, "iso_week"])[["cell_lat", "cell_lon"]]
                   .first().reset_index())
    log(f"[val] footprint from {'all years' if MASK_FROM_ALL_YEARS else f'{TEST_YEAR} only'}"
        f" ({sampled[GRID_ID_COL].nunique():,} cells)")

    daily, daily_file = load_daily()
    static_cells = (pd.read_parquet(daily_file, columns=[GRID_ID_COL] + static_cols)
                      .groupby(GRID_ID_COL).first().reset_index())

    model = fit_calibrated(training_set(feats), feats)      # ONCE
    label = f"MaxEnt ({VARIANT})"

    rows = []
    for week in WEEKS:
        gf, mon = week_features(daily, week, feats, static_cells)
        ids = gf[GRID_ID_COL].to_numpy()
        op = fade(ids, week, sampled)
        keep = trap_mask(ids, week, sampled)
        obs = table[(table.iso_year == TEST_YEAR) & (table.iso_week == week)][
            [GRID_ID_COL, "cell_lat", "cell_lon", "presence"]]

        grid = gf[[GRID_ID_COL]].copy()
        grid["pred"] = predict_grid(model, feats, gf)
        grid["opacity"] = op
        stem = f"maxent_validation_{TEST_YEAR}_week{week:02d}_{VARIANT}"
        wk = f"ISO week {week} {TEST_YEAR} (~{mon.strftime('%d %b')})"
        trained = f"trained {min(TRAIN_YEARS)}–{max(TRAIN_YEARS)}"

        render(grid, f"Cx. nigripalpus — predicted suitability ({label})\n{wk} · {trained}",
               OUT_DIR / f"{stem}_full.png")
        render(grid[keep].reset_index(drop=True),
               f"Cx. nigripalpus — predicted suitability ({label}), surveyed footprint\n"
               f"{wk} · masked <{MASK_KM:.0f} km of a trap · {trained}",
               OUT_DIR / f"{stem}_masked.png")
        render(grid[keep].reset_index(drop=True),
               f"Cx. nigripalpus — {label} nowcast vs observed catches\n"
               f"{wk} · {trained} (2018 never seen)",
               OUT_DIR / f"{stem}_overlay.png", obs=obs, zoom=True)

        m = map_metrics(grid, obs, week); rows.append(m)
        log(f"[val] maxent_{VARIANT} week {week}: ROC {m['roc_auc']} (n={m['n']}, "
            f"{m['n_abs']} abs | pres suit {m['mean_suit_pres']} vs abs {m['mean_suit_abs']})")

    out_csv = OUT_DIR / "validation_metrics_maxent.csv"
    mdf = pd.DataFrame(rows)
    if out_csv.exists():
        mdf = (pd.concat([pd.read_csv(out_csv), mdf], ignore_index=True)
                 .drop_duplicates(["model", "week"], keep="last"))
    mdf.to_csv(out_csv, index=False)
    log(f"\n[done] {len(rows)*3} maps + validation_metrics_maxent.csv -> {OUT_DIR}")


if __name__ == "__main__":
    run()