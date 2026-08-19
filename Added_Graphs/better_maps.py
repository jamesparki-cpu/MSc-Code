from __future__ import annotations
"""
better_maps.py  —  STAGE 6d (trial): thesis-quality re-render of the 6b surfaces.

Does NOT replace render_maps.py. Reads the same surface_predictions.parquet and
writes a parallel set of figures to a NEW folder so the two can be compared
side by side before committing.

WHAT CHANGED vs render_maps.py
  1. TRUE RASTER      scatter(marker="s") -> gridded RGBA imshow. Points are
                      binned onto a regular lat/lon grid at the native cell
                      spacing, so there are no inter-marker gaps at any DPI or
                      figure size. (pcolormesh available via RENDER_BACKEND.)
  2. BOUNDARY         a coastline/state outline is drawn. Cartopy is used when
                      installed; otherwise the outline is derived from the data
                      footprint itself (contour of the valid-cell mask), which
                      needs no extra dependency and traces the real coast.
  3. MAP FURNITURE    scale bar (km, latitude-corrected), north arrow,
                      graticule with degree-formatted ticks, and an inset
                      locator showing the rendered extent within the full
                      statewide footprint (matters for the masked variants).
  4. TITLE CLIPPING   two-line short titles, constrained layout, colourbar pad,
                      bbox_inches="tight" on save.
  5. TYPOGRAPHY       serif family throughout, larger axis/tick labels sized for
                      a thesis body-text page.

Opacity semantics are unchanged and still mean PROXIMITY TO SURVEILLANCE, not
model certainty. Caption discipline from 6c carries over verbatim.

OUTPUT (to config["better_maps_dir"], default weekly_results_dir/maps_v2/)
  <series>/week_<NN>.png              statewide
  <series>_masked/week_<NN>.png       surveyed footprint only
  <series>.gif                        if RENDER_ALL_WEEKS and imageio present

USAGE
  python better_maps.py                    # default selection of weeks
  python better_maps.py --weeks 4 17 30 43
  python better_maps.py --all              # every week + GIFs
  # or, as a module:
  import better_maps as B; B.run(weeks=[32], series=["xgboost"])
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

# ============================== CONFIG ======================================
CONFIG_PATH = "config.json"

GRID_ID_COL = "Grid_ID"
CMAP_SUIT = "RdYlBu_r"
CMAP_AGREE = "magma_r"

# Which weeks to render when neither --weeks nor --all is given.
# One per season so the set demonstrates the model saying "no" as well as "yes".
DEFAULT_WEEKS = [4, 17, 30, 43]

RENDER_BACKEND = "imshow"     # "imshow" (RGBA, per-cell alpha) | "pcolormesh"
OPACITY_COL = "opacity"       # "opacity" (floored) | "confidence_raw" (fade to 0)

# Footprint mask: keep cells whose sampling support exceeds this. confidence_raw
# is an exponential falloff in distance-to-nearest-trap, so this approximates the
# hard <40 km mask used in the 2018 validation figure. Tune to match if needed.
MASK_MIN_CONFIDENCE = 0.05
MASK_CONFIDENCE_COL = "confidence_raw"

FIGSIZE = (6.6, 7.6)
DPI = 220
GIF_FPS = 4

SCALEBAR_KM = 100             # nominal; snapped to a round number at render time
DRAW_INSET = True             # inset locator (auto-skipped on full-extent maps)
DRAW_SCALEBAR = True
DRAW_NORTH_ARROW = True
DRAW_GRATICULE = True
USE_CARTOPY = False            # falls back silently to the derived footprint outline

CAPTION = ("Opacity encodes proximity to surveillance data (faded = extrapolated), "
           "not model certainty.")
# ============================================================================


# ----- typography -----------------------------------------------------------
def set_thesis_style():
    """Serif family and body-text-scaled labels. Called once by run()."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def log(m):
    print(m, flush=True)


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


# ----- column discovery -----------------------------------------------------
def detect_series(surf):
    """Map series name -> (column, cmap, vmin, vmax, use_opacity, cbar label).

    Model-agnostic: any prob_* column becomes a series, so MaxEnt columns are
    picked up automatically once they are joined into the surface table.
    """
    pretty = {
        "prob_xgb": "XGBoost", "prob_xgboost": "XGBoost",
        "prob_rf": "Random Forest", "prob_random_forest": "Random Forest",
        "prob_maxent_vanilla": "MaxEnt (vanilla)",
        "prob_maxent_targetgroup": "MaxEnt (target-group)",
    }
    out = {}
    for col in surf.columns:
        if not col.startswith("prob_"):
            continue
        name = col.replace("prob_", "")
        label = pretty.get(col, name.replace("_", " ").title())
        out[name] = dict(col=col, cmap=CMAP_SUIT, vmin=0.0, vmax=1.0,
                         use_opacity=True, cbar="predicted suitability",
                         label=f"predicted suitability ({label}, calibrated)")
    if "agree_abs_diff" in surf.columns:
        vmax = float(np.nanquantile(surf["agree_abs_diff"], 0.99))
        out["agreement"] = dict(col="agree_abs_diff", cmap=CMAP_AGREE,
                                vmin=0.0, vmax=vmax, use_opacity=False,
                                cbar="|XGB \u2212 RF|  (dark = disagree)",
                                label="model disagreement |XGB \u2212 RF|")
    return out


# ----- gridding: the fix for the inter-marker gaps --------------------------
def _infer_step(vals, across=None):
    """Median neighbour spacing.

    On a metric (5 km) grid, longitude spacing in DEGREES widens with latitude,
    so rows do not share a common longitude lattice. Taking np.unique across the
    whole state then returns the tiny gaps between near-duplicate values from
    different rows, which over-resolves the raster and leaves interior cells
    empty. Measuring WITHIN each line gives the true cell size.
    """
    if across is not None:
        d = []
        for k in np.unique(across):
            v = np.sort(np.asarray(vals)[across == k])
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


def build_raster(sub, value_col, alpha_col=None, step=None):
    """Bin an irregular point cloud onto a regular lat/lon grid.

    Returns (Z, A, extent) where Z is a masked 2D array of values, A a matching
    2D alpha array (or None), and extent the imshow/pcolormesh bounds. Binning
    (rather than pivoting on exact coordinates) means a projected 5 km grid whose
    longitude spacing drifts with latitude still rasterises cleanly.
    """
    lon = sub["lon"].to_numpy(float)
    lat = sub["lat"].to_numpy(float)
    val = sub[value_col].to_numpy(float)

    if step:
        dx, dy = step[0], step[1]
    else:
        dx = _infer_step(lon, across=lat)   # lon spacing within each latitude row
        dy = _infer_step(lat)               # latitudes are a clean lattice

    lon0, lat0 = lon.min(), lat.min()
    nx = int(round((lon.max() - lon0) / dx)) + 1
    ny = int(round((lat.max() - lat0) / dy)) + 1

    ix = np.clip(np.round((lon - lon0) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.round((lat - lat0) / dy).astype(int), 0, ny - 1)

    vsum = np.zeros((ny, nx))
    cnt = np.zeros((ny, nx))
    ok = np.isfinite(val)
    np.add.at(vsum, (iy[ok], ix[ok]), val[ok])
    np.add.at(cnt, (iy[ok], ix[ok]), 1.0)

    A = None
    if alpha_col is not None and alpha_col in sub.columns:
        av = sub[alpha_col].to_numpy(float)
        asum = np.zeros((ny, nx))
        np.add.at(asum, (iy[ok], ix[ok]), np.nan_to_num(av[ok]))
        A = np.where(cnt > 0, asum / np.maximum(cnt, 1), 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.where(cnt > 0, vsum / np.maximum(cnt, 1), np.nan)
    Z = np.ma.masked_invalid(Z)
    # Fill interior gaps left by binning an irregular metric grid onto a regular
    # lattice. Only cells enclosed by real data are filled, from their nearest
    # valid neighbour -- this is a rendering repair, not imputation: genuinely
    # missing regions stay masked and still render as gaps.
    try:
        from scipy.ndimage import binary_closing, distance_transform_edt
        real = cnt > 0
        fillable = binary_closing(real, structure=np.ones((3, 3)),
                                  iterations=2) & ~real
        if fillable.any():
            _, (jy, jx) = distance_transform_edt(~real, return_indices=True)
            Zf = Z.filled(np.nan)
            Zf[fillable] = Zf[jy[fillable], jx[fillable]]
            Z = np.ma.masked_invalid(Zf)
            if A is not None:
                A[fillable] = A[jy[fillable], jx[fillable]]
            cnt[fillable] = 1
    except Exception:
        pass

    extent = (lon0 - dx / 2, lon0 + (nx - 0.5) * dx,
              lat0 - dy / 2, lat0 + (ny - 0.5) * dy)
    return Z, A, extent


def draw_raster(ax, Z, A, extent, cmap, norm, backend=RENDER_BACKEND):
    """Draw the gridded surface with optional per-cell alpha."""
    cm = plt.get_cmap(cmap)
    if backend == "pcolormesh":
        ny, nx = Z.shape
        xe = np.linspace(extent[0], extent[1], nx + 1)
        ye = np.linspace(extent[2], extent[3], ny + 1)
        kw = dict(cmap=cm, norm=norm, shading="flat")
        if A is not None:
            kw["alpha"] = np.ma.masked_array(A, Z.mask)
        return ax.pcolormesh(xe, ye, Z, **kw)

    # imshow path: build RGBA explicitly so alpha is reliably per-cell
    rgba = cm(norm(Z.filled(np.nan)))
    rgba[..., 3] = 1.0 if A is None else np.clip(A, 0.0, 1.0)
    rgba[Z.mask, 3] = 0.0                      # sea / outside grid = transparent
    return ax.imshow(rgba, extent=extent, origin="lower",
                     interpolation="nearest", zorder=1)


def footprint_outline(ax, Z, extent, lw=0.8, color="0.15"):
    """Trace the edge of the valid-cell mask.

    The data footprint IS the Florida coast, so this gives a boundary with no
    external dependency and no risk of a coastline that disagrees with the grid.
    """
    ny, nx = Z.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    valid = (~Z.mask).astype(float) if np.ma.isMaskedArray(Z) else np.isfinite(Z).astype(float)
    pad = np.zeros((ny + 2, nx + 2))
    pad[1:-1, 1:-1] = valid
    xp = np.concatenate(([x[0] - (x[1] - x[0])], x, [x[-1] + (x[1] - x[0])]))
    yp = np.concatenate(([y[0] - (y[1] - y[0])], y, [y[-1] + (y[1] - y[0])]))
    ax.contour(xp, yp, pad, levels=[0.5], colors=color, linewidths=lw, zorder=4)


# ----- map furniture --------------------------------------------------------
def add_graticule(ax, extent, step=1.0):
    """Degree-labelled ticks plus a faint graticule."""
    x0, x1, y0, y1 = extent
    ax.set_xticks(np.arange(np.ceil(x0), np.floor(x1) + 0.01, step))
    ax.set_yticks(np.arange(np.ceil(y0), np.floor(y1) + 0.01, step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}\u00b0{'W' if v < 0 else 'E'}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}\u00b0{'N' if v >= 0 else 'S'}"))
    ax.grid(True, lw=0.4, color="0.75", alpha=0.5, ls=":", zorder=5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")


def add_scalebar(ax, extent, lat_ref, km=SCALEBAR_KM):
    """Latitude-corrected scale bar, snapped to a round distance."""
    x0, x1, y0, y1 = extent
    km_per_deg = 111.320 * np.cos(np.radians(lat_ref))
    span_km = (x1 - x0) * km_per_deg
    for cand in (25, 50, 100, 150, 200, 250, 500):          # pick a sensible length
        if cand <= span_km * 0.32:
            km = cand
    dx = km / km_per_deg

    bx = x0 + (x1 - x0) * 0.06
    by = y0 + (y1 - y0) * 0.055
    h = (y1 - y0) * 0.009
    for i in range(2):                                       # two-tone bar
        ax.add_patch(Rectangle((bx + i * dx / 2, by), dx / 2, h,
                               facecolor="black" if i == 0 else "white",
                               edgecolor="black", lw=0.6, zorder=6))
    ax.text(bx + dx / 2, by + h * 1.9, f"{km} km", ha="center", va="bottom",
            fontsize=9, zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))


def add_north_arrow(ax, extent):
    x0, x1, y0, y1 = extent
    ax.annotate("N", xy=(x0 + (x1 - x0) * 0.94, y0 + (y1 - y0) * 0.955),
                xytext=(x0 + (x1 - x0) * 0.94, y0 + (y1 - y0) * 0.885),
                ha="center", va="bottom", fontsize=11, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))


def add_inset(fig, ax, full_extent, view_extent, full_Z=None):
    """Small locator showing where the rendered extent sits in the full grid.

    Only meaningful when the view is a subset (the masked variants), so it is
    skipped automatically when the two extents match.
    """
    if np.allclose(full_extent, view_extent, atol=1e-6):
        return
    ins = ax.inset_axes([0.685, 0.015, 0.30, 0.30])
    if full_Z is not None:
        ins.imshow(np.where(full_Z.mask, np.nan, 0.38) if np.ma.isMaskedArray(full_Z)
                   else full_Z, extent=full_extent, origin="lower",
                   cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ins.add_patch(Rectangle((view_extent[0], view_extent[2]),
                            view_extent[1] - view_extent[0],
                            view_extent[3] - view_extent[2],
                            fill=False, ec="crimson", lw=1.1, zorder=3))
    ins.set_xlim(full_extent[0], full_extent[1])
    ins.set_ylim(full_extent[2], full_extent[3])
    ins.set_aspect(1 / np.cos(np.radians(np.mean(full_extent[2:]))))
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_linewidth(0.7)
    ins.patch.set_alpha(0.9)


# ----- frame ----------------------------------------------------------------
def render_frame(sub, spec, title_main, title_sub, path,
                 full_ref=None, observations=None, caption=CAPTION):
    """Render one week of one series.

    sub          rows for this week (already masked, if a masked variant)
    spec         entry from detect_series()
    full_ref     (Z, extent) of the unmasked statewide grid, for the inset
    observations DataFrame with lat/lon/presence, drawn at full opacity
    """
    norm = Normalize(spec["vmin"], spec["vmax"])
    Z, A, extent = build_raster(sub, spec["col"],
                                OPACITY_COL if spec["use_opacity"] else None)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_raster(ax, Z, A, extent, spec["cmap"], norm)

    if not _cartopy_boundary(ax):
        ref_Z, ref_ext = full_ref if full_ref is not None else (Z, extent)
        footprint_outline(ax, ref_Z, ref_ext)

    lat_ref = float(np.mean(extent[2:]))
    ax.set_aspect(1 / np.cos(np.radians(lat_ref)))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    if DRAW_GRATICULE:
        add_graticule(ax, extent)
    if DRAW_SCALEBAR:
        add_scalebar(ax, extent, lat_ref)
    if DRAW_NORTH_ARROW:
        add_north_arrow(ax, extent)
    if DRAW_INSET and full_ref is not None:
        add_inset(fig, ax, full_ref[1], extent, full_ref[0])

    if observations is not None and len(observations):
        pres = observations[observations["presence"] == 1]
        absn = observations[observations["presence"] == 0]
        ax.scatter(pres["lon"], pres["lat"], s=26, facecolors="none",
                   edgecolors="black", lw=0.9, marker="o", zorder=7,
                   label=f"observed presence (n={len(pres)})")
        ax.scatter(absn["lon"], absn["lat"], s=26, c="black", lw=0.9,
                   marker="x", zorder=7,
                   label=f"observed absence (n={len(absn)})")
        ax.legend(loc="upper left", framealpha=0.9, borderpad=0.4)

    # two-line title: short enough that nothing clips at this figure width
    ax.set_title(f"{title_main}\n{title_sub}", fontsize=12, linespacing=1.35)

    sm = plt.cm.ScalarMappable(cmap=spec["cmap"], norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.035, fraction=0.045)
    cb.set_label(spec["cbar"], labelpad=8)
    cb.outline.set_linewidth(0.6)

    if caption:
        # anchored to the axes (not the figure) so bbox_inches="tight" crops
        # snugly instead of leaving a band of whitespace below the map
        ax.text(0.5, -0.105, caption, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5, color="0.35")

    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    
# ----- boundary overlay -----------------------------------------------------
def _cartopy_boundary(ax):
    """True coastline + state borders. Returns True only if fully successful.

    Segments are collected first and drawn only once every geometry has been
    read, so a mid-loop failure cannot leave half a boundary on the axes while
    the footprint fallback also runs.
    """
    if not USE_CARTOPY:
        return False
    try:
        import cartopy.feature as cfeature
    except Exception:
        return False
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    pending = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for feat, lw, col in ((cfeature.COASTLINE, 0.7, "0.15"),
                                  (cfeature.STATES.with_scale("10m"), 0.5, "0.35")):
                for g in feat.geometries():
                    for poly in (g.geoms if hasattr(g, "geoms") else [g]):
                        xy = np.asarray(getattr(poly, "coords", None) or
                                        poly.exterior.coords)
                        # skip anything wholly outside the view
                        if (xy[:, 0].max() < x0 or xy[:, 0].min() > x1 or
                                xy[:, 1].max() < y0 or xy[:, 1].min() > y1):
                            continue
                        pending.append((xy, lw, col))
    except Exception:
        return False
    for xy, lw, col in pending:
        ax.plot(xy[:, 0], xy[:, 1], lw=lw, color=col, zorder=4)
    return bool(pending)

# ----- driver ---------------------------------------------------------------
def iso_monday_label(week, year=2015):
    from datetime import date
    try:
        return date.fromisocalendar(year, int(week), 1).strftime("%d %b")
    except ValueError:
        return f"wk{int(week)}"


def render_series(surf, name, spec, out_dir, weeks, masked=False,
                  period_label="climatological mean 2013\u20132018",
                  observations=None, make_gif=False):
    folder = out_dir / (f"{name}_masked" if masked else name)
    folder.mkdir(parents=True, exist_ok=True)

    frames = []
    for wk in weeks:
        sub_all = surf[surf["iso_week"] == wk]
        if sub_all.empty:
            log(f"  [skip] {name} week {wk}: no rows")
            continue
        full_Z, _, full_ext = build_raster(sub_all, spec["col"])

        sub = sub_all
        if masked:
            if MASK_CONFIDENCE_COL not in sub.columns:
                log(f"  [skip] masked {name}: '{MASK_CONFIDENCE_COL}' not in surface")
                return None
            sub = sub[sub[MASK_CONFIDENCE_COL] >= MASK_MIN_CONFIDENCE]
            if sub.empty:
                log(f"  [skip] masked {name} week {wk}: mask removed every cell")
                continue

        obs = None
        if observations is not None:
            obs = observations[observations["iso_week"] == wk]

        scope = "surveyed footprint" if masked else "statewide"
        title_main = f"$\\it{{Cx.\\ nigripalpus}}$ \u2014 {spec['label']}"
        title_sub = (f"ISO week {int(wk):02d} (~{iso_monday_label(wk)})  \u00b7  "
                     f"{scope}  \u00b7  {period_label}")

        p = folder / f"week_{int(wk):02d}.png"
        render_frame(sub, spec, title_main, title_sub, p,
                     full_ref=(full_Z, full_ext), observations=obs)
        frames.append(p)

    log(f"  [ok] {folder.name}: {len(frames)} PNG(s)")

    if make_gif and len(frames) > 1:
        try:
            import imageio.v2 as imageio
            gif = out_dir / f"{folder.name}.gif"
            imageio.mimsave(gif, [imageio.imread(f) for f in frames],
                            fps=GIF_FPS, loop=0)
            log(f"  [ok] {gif.name}")
        except Exception as e:
            log(f"  [warn] GIF skipped: {e}")
    return folder


def run(weeks=None, series=None, config_path=CONFIG_PATH, all_weeks=False,
        masked_variants=True, observations_file=None):
    set_thesis_style()
    cfg = load_config(config_path)
    results = Path(cfg["weekly_results_dir"])
    out_dir = Path(cfg.get("better_maps_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    surf = pd.read_parquet(results / "surface_predictions.parquet")
    log(f"[6d] {len(surf):,} cell-weeks | backend={RENDER_BACKEND} | out={out_dir}")

    specs = detect_series(surf)
    if series:
        specs = {k: v for k, v in specs.items() if k in series}
    if not specs:
        raise SystemExit("no renderable columns found in surface_predictions.parquet")

    all_wk = sorted(int(w) for w in surf["iso_week"].unique())
    wks = all_wk if all_weeks else [w for w in (weeks or DEFAULT_WEEKS) if w in all_wk]
    log(f"[6d] series={list(specs)} | weeks={wks}")

    obs = None
    if observations_file and Path(observations_file).exists():
        obs = pd.read_parquet(observations_file)
        log(f"[6d] observation overlay: {len(obs):,} trap records")

    for name, spec in specs.items():
        log(f"[6d] {name}")
        render_series(surf, name, spec, out_dir, wks,
                      masked=False, observations=obs, make_gif=all_weeks)
        if masked_variants and spec["use_opacity"]:
            render_series(surf, name, spec, out_dir, wks,
                          masked=True, observations=obs, make_gif=all_weeks)

    log(f"[done] {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Thesis-quality re-render of the 6b surfaces.")
    ap.add_argument("--weeks", type=int, nargs="+", default=None)
    ap.add_argument("--all", action="store_true", help="every week + GIFs")
    ap.add_argument("--series", nargs="+", default=None,
                    help="e.g. xgboost random_forest agreement")
    ap.add_argument("--no-masked", action="store_true")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--observations", default=None,
                    help="parquet with iso_week/lat/lon/presence to overlay")
    a = ap.parse_args()
    run(weeks=a.weeks, series=a.series, config_path=a.config, all_weeks=a.all,
        masked_variants=not a.no_masked, observations_file=a.observations)


if __name__ == "__main__":
    main()
