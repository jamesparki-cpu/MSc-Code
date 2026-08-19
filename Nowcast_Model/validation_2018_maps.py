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
# ===========================================================================
# THESIS MAP STYLING  —  paste this block into the script, just below its
# imports. Self-contained: needs only numpy, pandas and matplotlib (already
# imported by every map script). scipy is used if present, skipped if not.
#
# Provides: set_thesis_style(), build_raster(), draw_raster(),
#           footprint_outline(), add_graticule(), add_scalebar(),
#           add_north_arrow(), add_inset()
# ===========================================================================
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

CAPTION = ("Opacity encodes proximity to surveillance data (faded = extrapolated), "
           "not model certainty.")


def set_thesis_style():
    """Serif family, body-text-scaled labels. Call once at the top of run()."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


def _infer_step(vals, across=None):
    """Median neighbour spacing, measured WITHIN each line when possible.

    The prediction grid is metric (5 km), so longitude spacing in DEGREES
    widens with latitude and rows share no common longitude lattice. Taking
    np.unique across the whole state then returns the tiny gaps between
    near-duplicate values from different rows, over-resolving the raster.
    """
    vals = np.asarray(vals, float)
    if across is not None:
        d = []
        for k in np.unique(across):
            v = np.sort(vals[across == k])
            if v.size > 1:
                d.append(np.diff(v))
        if d:
            d = np.concatenate(d)
            d = d[d > 1e-9]
            if d.size:
                return float(np.median(d))
    u = np.unique(vals)
    if u.size < 2:
        return 0.05
    d = np.diff(u)
    d = d[d > 1e-9]
    return float(np.median(d)) if d.size else 0.05


def build_raster(lon, lat, values, alpha=None):
    """Bin an irregular point cloud onto a regular lat/lon grid.

    Returns (Z, A, extent). Replaces scatter(marker="s"): a true raster has no
    inter-marker gaps at any DPI or figure size.
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    val = np.asarray(values, float)

    dx = _infer_step(lon, across=lat)      # lon spacing within each latitude row
    dy = _infer_step(lat)                  # latitudes are a clean lattice

    lon0, lat0 = lon.min(), lat.min()
    nx = int(round((lon.max() - lon0) / dx)) + 1
    ny = int(round((lat.max() - lat0) / dy)) + 1
    ix = np.clip(np.round((lon - lon0) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.round((lat - lat0) / dy).astype(int), 0, ny - 1)

    vsum = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
    ok = np.isfinite(val)
    np.add.at(vsum, (iy[ok], ix[ok]), val[ok])
    np.add.at(cnt, (iy[ok], ix[ok]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.where(cnt > 0, vsum / np.maximum(cnt, 1), np.nan)

    A = None
    if alpha is not None:
        av = np.asarray(alpha, float)
        asum = np.zeros((ny, nx))
        np.add.at(asum, (iy[ok], ix[ok]), np.nan_to_num(av[ok]))
        A = np.where(cnt > 0, asum / np.maximum(cnt, 1), 0.0)

    # Fill interior gaps left by binning an irregular grid onto a regular
    # lattice. Only cells enclosed by real data are filled, from their nearest
    # valid neighbour. This is a rendering repair, not imputation: genuinely
    # missing regions stay masked and still render as gaps.
    try:
        from scipy.ndimage import binary_closing, distance_transform_edt
        real = cnt > 0
        fillable = binary_closing(real, structure=np.ones((3, 3)),
                                  iterations=2) & ~real
        if fillable.any():
            _, (jy, jx) = distance_transform_edt(~real, return_indices=True)
            Z[fillable] = Z[jy[fillable], jx[fillable]]
            if A is not None:
                A[fillable] = A[jy[fillable], jx[fillable]]
    except Exception:
        pass

    Z = np.ma.masked_invalid(Z)
    extent = (lon0 - dx / 2, lon0 + (nx - 0.5) * dx,
              lat0 - dy / 2, lat0 + (ny - 0.5) * dy)
    return Z, A, extent


def draw_raster(ax, Z, A, extent, cmap, norm):
    """RGBA imshow so per-cell alpha (the sampling fade) is reliable."""
    rgba = plt.get_cmap(cmap)(norm(Z.filled(np.nan)))
    rgba[..., 3] = 1.0 if A is None else np.clip(A, 0.0, 1.0)
    rgba[Z.mask, 3] = 0.0
    return ax.imshow(rgba, extent=extent, origin="lower",
                     interpolation="nearest", zorder=1)


def footprint_outline(ax, Z, extent, lw=0.8, color="0.15"):
    """Trace the edge of the valid-cell mask. The data footprint IS the coast,
    so this needs no shapefile and cannot disagree with the raster."""
    ny, nx = Z.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    valid = (~Z.mask).astype(float) if np.ma.isMaskedArray(Z) else np.isfinite(Z).astype(float)
    try:                       # close single-cell holes so no phantom coastline
        from scipy.ndimage import binary_closing
        valid = binary_closing(valid.astype(bool),
                               structure=np.ones((3, 3)), iterations=1).astype(float)
    except Exception:
        pass
    pad = np.zeros((ny + 2, nx + 2)); pad[1:-1, 1:-1] = valid
    xp = np.r_[x[0] - (x[1] - x[0]), x, x[-1] + (x[1] - x[0])]
    yp = np.r_[y[0] - (y[1] - y[0]), y, y[-1] + (y[1] - y[0])]
    ax.contour(xp, yp, pad, levels=[0.5], colors=color, linewidths=lw, zorder=4)


def add_graticule(ax, view, step=1.0):
    x0, x1, y0, y1 = view
    ax.set_xticks(np.arange(np.ceil(x0), np.floor(x1) + 0.01, step))
    ax.set_yticks(np.arange(np.ceil(y0), np.floor(y1) + 0.01, step))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{abs(v):.0f}\u00b0{'W' if v < 0 else 'E'}"))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{abs(v):.0f}\u00b0{'N' if v >= 0 else 'S'}"))
    ax.grid(True, lw=0.4, color="0.75", alpha=0.5, ls=":", zorder=5)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")


def add_scalebar(ax, view, lat_ref):
    """Latitude-corrected scale bar, snapped to a round distance."""
    x0, x1, y0, y1 = view
    kpd = 111.320 * np.cos(np.radians(lat_ref))
    span = (x1 - x0) * kpd
    km = 25
    for c in (25, 50, 100, 150, 200, 250, 500):
        if c <= span * 0.32:
            km = c
    dx = km / kpd
    bx = x0 + (x1 - x0) * 0.06; by = y0 + (y1 - y0) * 0.055
    h = (y1 - y0) * 0.009
    for i in range(2):
        ax.add_patch(Rectangle((bx + i * dx / 2, by), dx / 2, h,
                               facecolor="black" if i == 0 else "white",
                               edgecolor="black", lw=0.6, zorder=6))
    ax.text(bx + dx / 2, by + h * 1.9, f"{km} km", ha="center", va="bottom",
            fontsize=9, zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))


def add_north_arrow(ax, view):
    x0, x1, y0, y1 = view
    ax.annotate("N", xy=(x0 + (x1 - x0) * 0.94, y0 + (y1 - y0) * 0.955),
                xytext=(x0 + (x1 - x0) * 0.94, y0 + (y1 - y0) * 0.885),
                ha="center", va="bottom", fontsize=11, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))


def add_inset(ax, full_extent, view, full_Z):
    """Locator showing the cropped view inside the full grid. Skipped when the
    map is already full-extent."""
    if np.allclose(full_extent, view, atol=1e-6):
        return
    ins = ax.inset_axes([0.685, 0.015, 0.30, 0.30])
    ins.imshow(np.where(full_Z.mask, np.nan, 0.38), extent=full_extent,
               origin="lower", cmap="Greys", vmin=0, vmax=1,
               interpolation="nearest")
    ins.add_patch(Rectangle((view[0], view[2]), view[1] - view[0],
                            view[3] - view[2], fill=False, ec="crimson", lw=1.1))
    ins.set_xlim(full_extent[0], full_extent[1])
    ins.set_ylim(full_extent[2], full_extent[3])
    ins.set_aspect(1 / np.cos(np.radians(np.mean(full_extent[2:]))))
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_linewidth(0.7)


def finish_map(fig, ax, Z, extent, cmap, norm, cbar_label,
               view=None, caption=CAPTION, inset=True):
    """Boundary + furniture + colourbar + caption. Call after draw_raster."""
    footprint_outline(ax, Z, extent)
    view = view or extent
    lat_ref = float(np.mean(view[2:]))
    ax.set_aspect(1 / np.cos(np.radians(lat_ref)))
    ax.set_xlim(view[0], view[1]); ax.set_ylim(view[2], view[3])
    add_graticule(ax, view)
    add_scalebar(ax, view, lat_ref)
    add_north_arrow(ax, view)
    if inset:
        add_inset(ax, extent, view, Z)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.035, fraction=0.045)
    cb.set_label(cbar_label, labelpad=8)
    cb.outline.set_linewidth(0.6)
    if caption:
        ax.text(0.5, -0.105, caption, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5, color="0.35")
# ===================== END THESIS MAP STYLING BLOCK ========================
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

CMAP = "RdYlBu_r"; DPI = 220
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
def render(grid, title, path, obs=None, view=None, use_opacity=True):
    """Styled prediction map. Matches the 2025 map style exactly.

    obs   optional observed catches (needs cell_lon/cell_lat/presence)
    view  optional (x0, x1, y0, y1) crop; the inset then locates it
    """
    lat, lon = lonlat(grid[GRID_ID_COL].to_numpy())
    norm = Normalize(0, 1)
    Z, A, ext = build_raster(lon, lat, grid["pred"],
                             alpha=grid["opacity"] if use_opacity else None)
    fig, ax = plt.subplots(figsize=(6.6, 7.6))
    draw_raster(ax, Z, A, ext, CMAP, norm)

    has_obs = obs is not None and len(obs)
    if has_obs:
        pres = obs[obs.presence == 1]
        absn = obs[obs.presence == 0]
        ax.scatter(pres.cell_lon, pres.cell_lat, s=26, facecolors="none",
                   edgecolors="black", lw=0.9, marker="o", zorder=7,
                   label=f"observed presence (n={len(pres)})")
        ax.scatter(absn.cell_lon, absn.cell_lat, s=26, c="black", lw=0.9,
                   marker="x", zorder=7,
                   label=f"observed absence (n={len(absn)})")

    if has_obs:
        # crop to the supported footprint plus a margin. Auto rather than a fixed
        # box so no masked cell is ever cut: the winter footprint sits further
        # north and west than the summer one.
        view = (-83.4, -80.2, 25.7, 30.9)

    ax.set_title(title, fontsize=12, linespacing=1.35)
    finish_map(fig, ax, Z, ext, CMAP, norm, "predicted suitability", view=view)
    if has_obs:
        ax.legend(loc="upper left", framealpha=0.9)

    fig.savefig(path, dpi=DPI, bbox_inches="tight")
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
    set_thesis_style()
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
                   f"{wk} · masked <{MASK_KM:.0f} km of a trap · {trained} ({TEST_YEAR} never seen)",
                   OUT_DIR / f"{stem}_overlay.png",
                   obs=obs)

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