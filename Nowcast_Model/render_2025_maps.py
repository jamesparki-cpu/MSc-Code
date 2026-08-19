from __future__ import annotations
"""
render_2025_maps.py  —  render the 2025 out-of-period prediction surfaces.

Reads surface_2025_predictions.parquet (from predict_2025_maps.py) and renders,
for EVERY model and EVERY week, TWO versions:
  * full     — statewide, opacity = sampling-proximity fade (whole state visible)
  * masked   — hard-masked to the surveyed footprint (cells within MASK_KM of a
               historical trap); beyond that is not drawn.

These are DEPLOYMENT / CAPABILITY maps, NOT validated results: there are no 2025
catches, so NO metric and NO observation overlay. Captions say so explicitly.

The mask is defined by SURVEILLANCE geometry (trap proximity), never by
prediction values — it narrows the map's SCOPE to where the model is supported,
it does not select on outcome.

STYLE matches the other maps (RdYlBu_r, square markers, latitude aspect).

OUTPUT (to config predict2025_dir/maps)
  <model>/week<NN>_full.png  and  <model>/week<NN>_masked.png
"""
import json, re
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
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
DATA_DIR = Path(cfg.get("weekly_xg_dir", "."))
PRED_DIR = Path(cfg["nowcast_dir"])
SURFACE  = PRED_DIR / "surface_2025_predictions.parquet"
OUT_DIR  = PRED_DIR / "2025_maps"

PRED_YEAR = 2025
MASK_KM = 40.0                    # hard-mask threshold: cells within this of a trap
SEASON_WINDOW = 2
CMAP = "RdYlBu_r"; MARKER = 6; DPI = 220

PRETTY = {"xgboost": "XGBoost", "random_forest": "Random Forest",
          "maxent_targetgroup": "MaxEnt (target-group)", "maxent_vanilla": "MaxEnt (vanilla)"}

def log(m): print(m, flush=True)
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
# ============================================================================


def trap_mask(sub, week, sampled):
    """True = keep (within MASK_KM of a trap active that season). Surveillance
    geometry only — independent of the predicted values."""
    coslat = np.cos(np.radians(sub["lat"].mean()))
    s = {((w-1) % 53)+1 for w in range(week-SEASON_WINDOW, week+SEASON_WINDOW+1)}
    samp = sampled[sampled.iso_week.isin(s)]
    if samp.empty:
        return np.zeros(len(sub), dtype=bool)
    tree = cKDTree(np.c_[samp.cell_lon*coslat*111, samp.cell_lat*111])
    dkm, _ = tree.query(np.c_[sub["lon"]*coslat*111, sub["lat"]*111], k=1)
    return dkm <= MASK_KM

def render(sub, prob_col, title, path, use_opacity=True, view=None):
    norm = Normalize(0, 1)
    Z, A, ext = build_raster(sub["lon"], sub["lat"], sub[prob_col],
                             alpha=sub["opacity"] if use_opacity else None)
    fig, ax = plt.subplots(figsize=(6.6, 7.6))
    draw_raster(ax, Z, A, ext, CMAP, norm)
    ax.set_title(title, fontsize=12, linespacing=1.35)
    finish_map(fig, ax, Z, ext, CMAP, norm, "predicted suitability", view=view)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"[2025-maps] {path.parent.name}/{path.name}")

def run():
    set_thesis_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    surf = pd.read_parquet(SURFACE)
    prob_cols = [c for c in surf.columns if c.startswith("prob_")]
    # historical trap footprint for the mask (2013-2018 sampled cell-weeks)
    table = pd.read_parquet(DATA_DIR / "weekly_model_table.parquet")
    sampled = table.groupby(["Grid_ID", "iso_week"])[["cell_lat", "cell_lon"]].first().reset_index()

    for pc in prob_cols:
        model = pc.replace("prob_", ""); label = PRETTY.get(model, model)
        d = OUT_DIR / model; d.mkdir(parents=True, exist_ok=True)
        for week in sorted(surf["iso_week"].unique()):
            sub = surf[surf["iso_week"] == week].copy()
            mon = pd.Timestamp(date.fromisocalendar(PRED_YEAR, int(week), 1))
            base_title = (f"Cx. nigripalpus — predicted suitability ({label})\n"
                          f"ISO week {int(week)} {PRED_YEAR} (~{mon.strftime('%d %b')}) · "
                          f"out-of-period prediction — UNVALIDATED (no {PRED_YEAR} surveillance)")
            # full
            render(sub, pc, base_title, d / f"week{int(week):02d}_full.png")
            # masked
            keep = trap_mask(sub, int(week), sampled)
            title_m = (f"Cx. nigripalpus — predicted suitability ({label}), surveyed footprint\n"
                       f"ISO week {int(week)} {PRED_YEAR} (~{mon.strftime('%d %b')}) · "
                       f"masked to <{MASK_KM:.0f} km of a trap · out-of-period, UNVALIDATED")
            render(sub[keep], pc, title_m, d / f"week{int(week):02d}_masked.png",
                   view=(-83.2, -80.3, 25.8, 29.7))
            log(f"[2025-maps] {model} week {int(week)}: full ({len(sub)}) + masked ({int(keep.sum())})")

    log(f"[done] 2025 maps -> {OUT_DIR}  ({len(prob_cols)} models x weeks x [full, masked])")


if __name__ == "__main__":
    run()