from __future__ import annotations
"""
render_maxent_maps.py  —  standalone renderer for the MaxEnt suitability maps.

Independent of the XGB/RF outputs (which live in a different folder): it reads
only the MaxEnt surfaces + the model table, and reproduces the IDENTICAL map
style and fade as render_maps.py. No merge step required.

For each surface_maxent_<variant>.parquet in maxent_dir it writes a 53-week PNG
series + GIF, plus one HTML week-slider over the MaxEnt variants. Cross-model
comparison (XGB/RF/MaxEnt metrics, agreement) is a SEPARATE script.

Self-contained because the MaxEnt surfaces carry only [Grid_ID, iso_week, prob]:
  * lat/lon are parsed from Grid_ID
  * the space x season opacity fade is RECOMPUTED here with the SAME parameters
    as predict_surfaces.py (6b), so the maps match XGB/RF exactly.

STYLE matches render_maps.py / static_map_builder.py (RdYlBu_r, square markers,
latitude-corrected aspect). Titles: "climatological mean 2013-2018".
"""
import json, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
import imageio.v2 as imageio
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
    config = json.load(f)
MAXENT_DIR = Path(config["maxent_dir"])                 # MaxEnt surfaces + output
FEAT_DIR   = Path(config["weekly_xg_dir"])              # weekly_model_table (fade points)
OUTPUT_DIR = MAXENT_DIR / "maps"

GRID_ID_COL = "Grid_ID"
CMAP_SUIT   = "RdYlBu_r"          # matches the static map builder
MARKER_SIZE = 6
DPI = 220
GIF_FPS = 4

# fade params — MUST match predict_surfaces.py (6b) so MaxEnt maps == XGB/RF maps
SEASON_WINDOW = 2
FALLOFF_KM    = 120.0
OPACITY_FLOOR = 0.25
OPACITY_COL   = "opacity"         # swap to "confidence_raw" for the fade-to-zero version

PRETTY = {"maxent_vanilla": "MaxEnt (vanilla)",
          "maxent_targetgroup": "MaxEnt (target-group)"}

_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
def _lonlat(gids):
    lat = np.array([float(_GID.search(str(g)).group(1)) for g in gids])
    lon = np.array([float(_GID.search(str(g)).group(2)) for g in gids])
    return lat, lon

def _iso_monday_label(week):
    from datetime import date
    try:   return date.fromisocalendar(2015, int(week), 1).strftime("%d %b")
    except ValueError: return f"wk{int(week)}"
# ============================================================================

def log(m): print(m, flush=True)


def compute_fade(surf, sampled):
    """Recompute the space x season sampling-proximity opacity, identical to 6b."""
    glat, glon = _lonlat(surf[GRID_ID_COL].to_numpy())
    coslat = np.cos(np.radians(np.nanmean(glat)))
    gx, gy = glon*coslat*111.0, glat*111.0
    conf = np.zeros(len(surf))
    weeks = surf["iso_week"].to_numpy()
    for wk in sorted(surf["iso_week"].unique()):
        s = {((w-1) % 53)+1 for w in range(wk-SEASON_WINDOW, wk+SEASON_WINDOW+1)}
        samp = sampled[sampled["iso_week"].isin(s)]
        rows = np.where(weeks == wk)[0]
        if samp.empty:
            continue
        tree = cKDTree(np.c_[samp["cell_lon"].to_numpy()*coslat*111.0,
                             samp["cell_lat"].to_numpy()*111.0])
        d, _ = tree.query(np.c_[gx[rows], gy[rows]], k=1)
        conf[rows] = np.exp(-d / FALLOFF_KM)
    surf = surf.copy()
    surf["lat"], surf["lon"] = glat, glon
    surf["confidence_raw"] = conf
    surf["opacity"] = OPACITY_FLOOR + (1.0 - OPACITY_FLOOR) * conf
    return surf

def render_frame(sub, value_col, title, cmap, vmin, vmax, path,
                 use_opacity=True, cbar_label="predicted suitability"):
    norm = Normalize(vmin, vmax)
    Z, A, ext = build_raster(sub["lon"], sub["lat"], sub[value_col],
                             alpha=sub[OPACITY_COL] if use_opacity else None)
    fig, ax = plt.subplots(figsize=(6.6, 7.6))
    draw_raster(ax, Z, A, ext, cmap, norm)
    ax.set_title(title, fontsize=12, linespacing=1.35)
    finish_map(fig, ax, Z, ext, cmap, norm, cbar_label)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def render_series(surf, value_col, folder, title_stub, cmap, vmin, vmax):
    d = OUTPUT_DIR / folder; d.mkdir(parents=True, exist_ok=True)
    frames = []
    for wk in sorted(surf["iso_week"].unique()):
        sub = surf[surf["iso_week"] == wk]
        title = (f"Cx. nigripalpus — {title_stub}\n"
                 f"ISO week {int(wk):02d} (~{_iso_monday_label(wk)}) · "
                 f"climatological mean 2013–2018")
        p = d / f"week_{int(wk):02d}.png"
        render_frame(sub, value_col, title, cmap, vmin, vmax, p)
        frames.append(imageio.imread(p))
    imageio.mimsave(OUTPUT_DIR / f"{folder}.gif", frames, fps=GIF_FPS, loop=0)
    log(f"[maxent-maps] {folder}: 53 PNGs + {folder}.gif")


def build_html_slider(weeks, variants):
    labels = {w: _iso_monday_label(w) for w in weeks}
    buttons = "\n".join(
        f'<button id="b_{v}" class="{"active" if i==0 else ""}" '
        f'onclick="setSeries(\'{v}\')">{PRETTY.get(v, v)}</button>'
        for i, v in enumerate(variants))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cx. nigripalpus — MaxEnt weekly suitability</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
 .row{{display:flex;gap:8px;align-items:center;margin:12px 0;flex-wrap:wrap}}
 img{{max-width:520px;border:1px solid #ddd}} button{{padding:6px 12px}}
 button.active{{background:#2166ac;color:#fff}} input[type=range]{{width:420px}}
</style></head><body>
<h2>Cx. nigripalpus — MaxEnt weekly suitability (climatological 2013–2018)</h2>
<div class="row">{buttons}</div>
<div class="row"><label>ISO week: <b id="wl"></b></label>
 <input type="range" id="wk" min="{weeks[0]}" max="{weeks[-1]}" value="34" step="1" oninput="update()"></div>
<img id="map" src="">
<p style="max-width:520px;color:#555;font-size:13px">Opacity = proximity to
surveillance data (faded = extrapolated). Titles state the calibrated model.</p>
<script>
 const labels={json.dumps(labels)}; const series={json.dumps(variants)};
 let cur=series[0];
 function pad(n){{return String(n).padStart(2,'0');}}
 function setSeries(s){{cur=s; for(const k of series)
   document.getElementById("b_"+k).classList.toggle("active",k===s); update();}}
 function update(){{const w=document.getElementById("wk").value;
   document.getElementById("wl").textContent=w+" (~"+labels[w]+")";
   document.getElementById("map").src=cur+"/week_"+pad(w)+".png";}}
 update();
</script></body></html>"""
    (OUTPUT_DIR / "maxent_explorer.html").write_text(html)
    log(f"[maxent-maps] maxent_explorer.html ({len(variants)} variant(s))")


def run():
    set_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # trapped cell-weeks = fade anchor points (both presence and absence were sampled)
    tbl = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet")
    sampled = tbl.groupby([GRID_ID_COL, "iso_week"])[["cell_lat", "cell_lon"]].first().reset_index()

    files = sorted(glob.glob(str(MAXENT_DIR / "surface_maxent_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No surface_maxent_*.parquet in {MAXENT_DIR}")
    variants = []
    for f in files:
        surf = pd.read_parquet(f)
        pcol = [c for c in surf.columns if c.startswith("prob_")][0]
        variant = pcol.replace("prob_", "")
        surf = compute_fade(surf, sampled)
        render_series(surf, pcol, variant,
                      f"predicted suitability ({PRETTY.get(variant, variant)}, calibrated)",
                      CMAP_SUIT, 0, 1)
        variants.append(variant)

    weeks = sorted(int(w) for w in surf["iso_week"].unique())
    build_html_slider(weeks, variants)
    log(f"[done] MaxEnt maps -> {OUTPUT_DIR}")


if __name__ == "__main__":
    run()