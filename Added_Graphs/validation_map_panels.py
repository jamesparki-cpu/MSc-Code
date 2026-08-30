from __future__ import annotations
"""
validation_2018_panels.py  —  three families of multi-panel figures over the
2018 held-out validation surfaces. Reads the cache; fits nothing.

  FAMILY 1  panel_validation_2018_<variant>.png        (variant = full | masked)
            every cached week x every model, rows are weeks, columns are models.
            One title, one colour bar, one scale bar. The same shape as the
            weekly-model panel figure, so the two read against each other
            directly.

  FAMILY 2  overlay_validation_2018_week<NN>.png       (one per week, 4 files)
            2x2 of the four models for ONE week, with that week's OBSERVED trap
            outcomes drawn on top: filled points are presences, open points are
            absences. Each panel's title carries its own ROC and BSS computed on
            those points, so a reader can see the metric and the reason for it
            in the same frame.

  FAMILY 3  seasons_validation_2018_<model>.png        (one per model)
            a square grid of the cached weeks for ONE model, observations
            overlaid. Answers "how does this model move through the season?",
            which the model-major layout of Family 2 cannot show.

            The frozen script maps FIVE weeks (6, 20, 30, 32, 43), so this is a
            3x2 grid with one empty cell rather than the 2x2 a four-week set
            would give. Set PANEL_WEEKS to restrict it.

  Families 2 and 3 are the same sixteen panels transposed. Both are produced
  because they answer different questions, and neither ordering serves both.

MASKING, STATED ONCE
  full    statewide; opacity encodes proximity to surveillance, so extrapolated
          areas fade rather than being hidden
  masked  restricted to the surveyed footprint, using the keep_masked flag the
          frozen validation scripts computed -- not recomputed here, so these
          panels agree with the published single-week maps cell for cell

  Overlay panels are always drawn on the FULL surface: the observation points
  are the evidence, and hiding the surface beneath some of them would be a
  strange thing to do.

PREREQUISITE
  validation_2018_cache.py, which persists the surfaces the frozen scripts
  build in memory.

OUTPUT (to <nowcast_dir>/Validation_Results/panels)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, brier_score_loss

import report_style as S
import map_style as M

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
RES = Path(cfg["nowcast_dir"]) / "Validation_Results"
OUT_DIR = RES / "panels"

SURFACE = RES / "surface_validation_2018.parquet"
OBS = RES / "observations_2018.parquet"

TEST_YEAR = 2018
SEASON = {6: "Winter", 20: "Spring", 30: "Summer", 32: "Summer", 43: "Autumn"}
CMAP = "RdYlBu_r"
DPI = 220
MARKER = 34              # observation points; the zoomed view gives them room
# Weeks plotted by ALL three families. The cache stores every week the frozen
# script maps (6, 20, 30, 32, 43); 30 and 32 are both summer, so plotting one of
# them keeps the seasonal grids at a clean 2x2 and loses nothing.
WEEKS_TO_PLOT = [6, 20, 32, 43]

# VIEW resolution:
#   "mask"          bounding box of the surveyed footprint, computed once and
#                   SHARED by every panel of every figure (default)
#   "observations"  bounding box of the trap points instead
#   "full"          the whole modelled grid
#   (x0, x1, y0, y1) an explicit crop
#
# A shared view matters: if each panel auto-scaled to its own week's mask, the
# panels would be at different scales and could not be compared by eye.
VIEW_MODE = "mask"
VIEW_MARGIN = 0.04       # fraction of span added on each side
_VIEW = None             # resolved in run()

# Overlay and seasonal panels are drawn on the MASKED surface, matching the
# published single-week overlay maps: at this zoom the surveyed footprint fills
# the panel, and the point-level agreement is the thing being shown.
OVERLAY_MASKED = True

# Panel width in inches; panel HEIGHT is derived from the view's true aspect so
# the figure has no white bands. The surveyed footprint is roughly twice as tall
# as it is wide, so a fixed square figure would waste most of the page.
PANEL_W_GRID = 2.5       # family 1 (weeks x models)
PANEL_W_SQUARE = 4.0     # families 2 and 3 (2x2)


def panel_hw_ratio(default: float = 1.15) -> float:
    """height / width of one panel, at the resolved view and its latitude."""
    if not _VIEW:
        return default
    x0, x1, y0, y1 = _VIEW
    w = (x1 - x0) * np.cos(np.radians((y0 + y1) / 2))
    return float((y1 - y0) / w) if w > 0 else default

def log(m): print(m, flush=True)
# ============================================================================


def resolve_view(surf, obs):
    """One view for every panel in every figure. See VIEW_MODE."""
    if isinstance(VIEW_MODE, (tuple, list)) and len(VIEW_MODE) == 4:
        return tuple(VIEW_MODE)
    if VIEW_MODE == "full":
        return None
    if VIEW_MODE == "observations" and len(obs):
        x, y = obs.cell_lon.to_numpy(), obs.cell_lat.to_numpy()
    else:
        d = surf[surf.keep_masked.astype(bool)] if "keep_masked" in surf.columns else surf
        if d.empty:
            return None
        x, y = d.lon.to_numpy(), d.lat.to_numpy()
    mx = (x.max() - x.min()) * VIEW_MARGIN
    my = (y.max() - y.min()) * VIEW_MARGIN
    v = (x.min() - mx, x.max() + mx, y.min() - my, y.max() + my)
    log(f"[view] {VIEW_MODE}: lon {v[0]:.2f} to {v[1]:.2f}, "
        f"lat {v[2]:.2f} to {v[3]:.2f}")
    return v


def load():
    if not SURFACE.exists():
        log(f"[load] {SURFACE.name} not found -- run validation_2018_cache.py first")
        return None, None
    surf = pd.read_parquet(SURFACE)
    obs = pd.read_parquet(OBS) if OBS.exists() else pd.DataFrame()
    log(f"[load] {len(surf):,} cell-weeks | weeks {sorted(surf.iso_week.unique())} | "
        f"{len(obs):,} trap points")
    return surf, obs


def week_metrics(surf, obs, week, model):
    """ROC and BSS on THIS week's observed points -- the honest caption number."""
    col = f"prob_{model}"
    g = (surf[surf.iso_week == week][["Grid_ID", col]]
         .merge(obs[obs.iso_week == week][["Grid_ID", "presence"]], on="Grid_ID"))
    g = g.dropna()
    if g.empty or g.presence.nunique() < 2:
        return None
    prev = g.presence.mean()
    return dict(
        n=len(g),
        roc=roc_auc_score(g.presence, g[col]),
        bss=1 - brier_score_loss(g.presence, g[col]) /
            brier_score_loss(g.presence, np.full(len(g), prev)),
    )


def draw_panel(ax, sub, col, norm, furniture=False, use_opacity=True):
    Z, A, ext = M.build_raster(sub["lon"], sub["lat"], sub[col],
                               alpha=sub["opacity"] if use_opacity else None)
    M.draw_raster(ax, Z, A, ext, CMAP, norm)
    M.footprint_outline(ax, Z, ext)
    v = _VIEW or ext
    lat_ref = float(np.mean(v[2:]))
    ax.set_aspect(1 / np.cos(np.radians(lat_ref)))
    ax.set_xlim(v[0], v[1]); ax.set_ylim(v[2], v[3])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    if furniture:
        M.add_scalebar(ax, v, lat_ref)
    return ext


def overlay_points(ax, o):
    """Observed outcomes on top. Marker convention matches the published
    single-week overlay maps: open circle = presence, cross = absence."""
    pres, absn = o[o.presence == 1], o[o.presence == 0]
    ax.scatter(pres.cell_lon, pres.cell_lat, s=MARKER, facecolors="none",
               edgecolors="#111111", linewidths=1.1, zorder=6)
    ax.scatter(absn.cell_lon, absn.cell_lat, s=MARKER, c="#111111",
               marker="x", linewidths=1.4, zorder=6)
    return len(pres), len(absn)


def shared_colourbar(fig, norm, rect=(0.92, 0.16, 0.015, 0.68)):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("predicted suitability", labelpad=8)
    cb.outline.set_linewidth(0.6)


def obs_legend(fig, n_pres=None, n_abs=None, y=0.055):
    handles = [
        Line2D([], [], marker="o", ls="none", ms=7, mfc="none",
               mec="#111111", mew=1.1,
               label="observed presence" + (f" (n={n_pres})" if n_pres else "")),
        Line2D([], [], marker="x", ls="none", ms=7, mec="#111111", mew=1.4,
               label="observed absence" + (f" (n={n_abs})" if n_abs else "")),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, y - 0.05))


# ------------------------------------------------------------------ FAMILY 1
def fig_week_model_grid(surf, models, masked: bool):
    weeks = sorted(surf.iso_week.unique())
    norm = Normalize(0, 1)
    variant = "masked" if masked else "full"

    r_hw = panel_hw_ratio()
    fig = plt.figure(figsize=(PANEL_W_GRID * len(models) + 1.6,
                              PANEL_W_GRID * r_hw * len(weeks) + 1.4))
    gs = GridSpec(len(weeks), len(models), figure=fig, left=0.07, right=0.90,
                  top=0.90, bottom=0.05, wspace=0.04, hspace=0.06)
    for r, wk in enumerate(weeks):
        w = surf[surf.iso_week == wk]
        if masked and "keep_masked" in w.columns:
            w = w[w.keep_masked.astype(bool)]
        for c, m in enumerate(models):
            ax = fig.add_subplot(gs[r, c])
            d = w.dropna(subset=[f"prob_{m}"])
            if d.empty:
                ax.axis("off"); continue
            draw_panel(ax, d, f"prob_{m}", norm,
                       furniture=(r == len(weeks) - 1 and c == 0))
            if r == 0:
                ax.set_title(S.model_label(m), fontsize=11, pad=6)
            if c == 0:
                ax.set_ylabel(f"{SEASON.get(wk, '')}\nISO week {wk}",
                              fontsize=10, labelpad=8)
    shared_colourbar(fig, norm)
    rule = ("restricted to the surveyed footprint (mask as computed by the "
            "validation scripts)" if masked else
            "statewide; opacity encodes proximity to surveillance data "
            "(faded = extrapolated), not model certainty")
    fig.suptitle(f"Cx. nigripalpus \u2014 {TEST_YEAR} held-out validation, four "
                 f"models across four weeks\n{rule}", fontsize=12)
    fig.text(0.5, 0.012, "All panels share the colour scale and extent; north is "
             "up and the scale bar applies throughout. Rows are weeks, columns "
             "are models.", ha="center", fontsize=8, color="0.35")
    out = OUT_DIR / f"panel_validation_{TEST_YEAR}_{variant}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    log(f"[fam1] wrote {out.name}")


# ------------------------------------------------------------------ FAMILY 2
def fig_overlay_by_week(surf, obs, models, week):
    norm = Normalize(0, 1)
    w = surf[surf.iso_week == week]
    if OVERLAY_MASKED and "keep_masked" in w.columns:
        w = w[w.keep_masked.astype(bool)]
    o = obs[obs.iso_week == week] if len(obs) else pd.DataFrame()

    ncol = 2
    nrow = int(np.ceil(len(models) / ncol))
    r_hw = panel_hw_ratio()
    fig = plt.figure(figsize=(PANEL_W_SQUARE * ncol + 2.2,
                              PANEL_W_SQUARE * r_hw * nrow + 2.0))
    gs = GridSpec(nrow, ncol, figure=fig, left=0.04, right=0.88, top=0.89,
                  bottom=0.10, wspace=0.04, hspace=0.10)
    np_, na_ = None, None
    for k, m in enumerate(models):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        d = w.dropna(subset=[f"prob_{m}"])
        if d.empty:
            ax.axis("off"); continue
        draw_panel(ax, d, f"prob_{m}", norm, furniture=(k == len(models) - ncol))
        if len(o):
            np_, na_ = overlay_points(ax, o)
        met = week_metrics(surf, obs, week, m) if len(o) else None
        t = S.model_label(m)
        if met:
            t += (f"\nROC {S.fmt(met['roc'], 'roc_auc')} · "
                  f"BSS {S.fmt(met['bss'], 'bss')} (n={S.fmt(met['n'], 'n')})")
        ax.set_title(t, fontsize=10, pad=5)

    shared_colourbar(fig, norm, rect=(0.90, 0.18, 0.018, 0.62))
    obs_legend(fig, np_, na_, y=0.09)
    fig.suptitle(f"Cx. nigripalpus \u2014 {TEST_YEAR} ISO week {week} "
                 f"({SEASON.get(week, '')}): predicted suitability vs observed "
                 f"catches\nmetrics are computed on this week's observed points "
                 f"only", fontsize=12)
    out = OUT_DIR / f"overlay_validation_{TEST_YEAR}_week{week:02d}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    log(f"[fam2] wrote {out.name}")


# ------------------------------------------------------------------ FAMILY 3
def fig_seasons_by_model(surf, obs, model):
    weeks = sorted(surf.iso_week.unique())
    norm = Normalize(0, 1)
    col = f"prob_{model}"
    if col not in surf.columns:
        return

    ncol = 2
    nrow = int(np.ceil(len(weeks) / ncol))
    r_hw = panel_hw_ratio()
    fig = plt.figure(figsize=(PANEL_W_SQUARE * ncol + 2.2,
                              PANEL_W_SQUARE * r_hw * nrow + 2.0))
    gs = GridSpec(nrow, ncol, figure=fig, left=0.04, right=0.88, top=0.89,
                  bottom=0.10, wspace=0.04, hspace=0.10)
    np_, na_ = None, None
    for k, wk in enumerate(weeks):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        d = surf[surf.iso_week == wk]
        if OVERLAY_MASKED and "keep_masked" in d.columns:
            d = d[d.keep_masked.astype(bool)]
        d = d.dropna(subset=[col])
        if d.empty:
            ax.axis("off"); continue
        draw_panel(ax, d, col, norm, furniture=(k == 2))
        o = obs[obs.iso_week == wk] if len(obs) else pd.DataFrame()
        if len(o):
            np_, na_ = overlay_points(ax, o)
        met = week_metrics(surf, obs, wk, model) if len(o) else None
        t = f"{SEASON.get(wk, '')} \u2014 ISO week {wk}"
        if met:
            t += (f"\nROC {S.fmt(met['roc'], 'roc_auc')} · "
                  f"BSS {S.fmt(met['bss'], 'bss')} (n={S.fmt(met['n'], 'n')})")
        ax.set_title(t, fontsize=10, pad=5)

    shared_colourbar(fig, norm, rect=(0.90, 0.18, 0.018, 0.62))
    obs_legend(fig, np_, na_, y=0.09)
    fig.suptitle(f"Cx. nigripalpus \u2014 {TEST_YEAR} held-out validation through "
                 f"the season ({S.model_label(model)})\npoint counts differ by "
                 f"week; metrics are computed on each week's observed points",
                 fontsize=12)
    out = OUT_DIR / f"seasons_validation_{TEST_YEAR}_{model}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    log(f"[fam3] wrote {out.name}")


def run():
    S.set_thesis_style(constrained=False)   # GridSpec geometry is set explicitly
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    surf, obs = load()
    if surf is None:
        return
    if WEEKS_TO_PLOT:
        missing = sorted(set(WEEKS_TO_PLOT) - set(surf.iso_week.unique()))
        if missing:
            log(f"[run] WARNING weeks {missing} are not in the cache")
        surf = surf[surf.iso_week.isin(WEEKS_TO_PLOT)]
        obs = obs[obs.iso_week.isin(WEEKS_TO_PLOT)] if len(obs) else obs
        log(f"[run] plotting weeks {sorted(surf.iso_week.unique())}")

    global _VIEW
    _VIEW = resolve_view(surf, obs)
    models = S.models_in([c.replace("prob_", "") for c in surf.columns
                          if c.startswith("prob_")])
    log(f"[run] models {models}")

    fig_week_model_grid(surf, models, masked=False)
    if "keep_masked" in surf.columns:
        fig_week_model_grid(surf, models, masked=True)
    else:
        log("[fam1] no keep_masked column -- masked variant skipped")

    for wk in sorted(surf.iso_week.unique()):
        fig_overlay_by_week(surf, obs, models, int(wk))
    for m in models:
        fig_seasons_by_model(surf, obs, m)

    log(f"\n[done] validation panels -> {OUT_DIR}")


if __name__ == "__main__":
    run()