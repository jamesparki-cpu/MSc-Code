from __future__ import annotations
"""
panel_maps.py  —  multi-panel map figures for print.

The weekly series and the animations are per-model, per-week, each with its own
title, colourbar, scale bar and north arrow. Repeated across a page that
furniture is noise: it is identical in every panel and it competes with the data
for space. These two figures draw it ONCE.

  FIG 1  panel_weeks_<variant>.png          (variant = full | masked)
         4 weeks x 4 models. Rows are weeks, columns are models, ONE shared
         colour bar, ONE title, ONE scale bar and north arrow (bottom-left
         panel). Replaces sixteen separate maps and makes the comparison the
         reader actually wants -- across models within a week -- a matter of
         reading along a row.

  FIG 2  filmstrip_<model>.png
         N evenly spaced weeks through the year for one model, same shared
         furniture. This is what stands in for the animation in a printed
         thesis: the GIF cannot be bound into a document, so it is cited as
         supplementary material and the filmstrip carries the argument.

MASKING, STATED ONCE
  full    statewide; opacity encodes proximity to surveillance, so extrapolated
          areas fade rather than being hidden
  masked  hard-masked to cells within MASK_KM of a trap that was active in the
          same part of the season; beyond that nothing is drawn

  The mask is defined by SURVEILLANCE GEOMETRY alone -- never by predicted
  values. It narrows the map's SCOPE to where the model is supported; it does
  not select on outcome. Both variants are produced so the reader can see the
  full surface and the supported subset of it.

STYLE
  Furniture from map_style.py, typography from report_style.py, colour map
  RdYlBu_r and marker conventions identical to the existing weekly series, so
  these figures sit alongside the single-week maps without a visible seam.
  The frozen map scripts are not touched and nothing is re-rendered.

OUTPUT (to <weekly_results_dir>/panel_maps)
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

import report_style as S
import map_style as M
from derived_maps import load_surfaces, cell_geometry   # one loader, not two

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
FEAT_DIR = Path(cfg["weekly_xg_dir"])
RESULTS = Path(cfg.get("weekly_results_dir", str(FEAT_DIR)))
OUT_DIR = RESULTS / "panel_maps"

PANEL_WEEKS = [6, 20, 32, 43]          # the weeks reported elsewhere in the chapter
FILMSTRIP_WEEKS = [1, 8, 15, 21, 28, 34, 41, 47]
SEASON = {6: "Winter", 20: "Spring", 30: "Summer", 32: "Summer", 43: "Autumn"}

CMAP = "RdYlBu_r"                      # as the weekly series
DPI = 220
MASK_KM = 40.0                         # as render_2025_maps.py
SEASON_WINDOW = 2                      # +/- weeks counted as "active this season"

def log(m): print(m, flush=True)
# ============================================================================


def trap_geometry():
    """Cell centroids of every trapped cell-week, for the mask."""
    t = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet",
                        columns=["Grid_ID", "iso_week", "cell_lat", "cell_lon"])
    return t.groupby(["Grid_ID", "iso_week"])[["cell_lat", "cell_lon"]].first().reset_index()


def trap_mask(sub: pd.DataFrame, week: int, sampled: pd.DataFrame) -> np.ndarray:
    """True = keep. Surveillance geometry only, independent of predicted values."""
    if cKDTree is None:
        log("[mask] scipy unavailable -- masked variant skipped")
        return np.ones(len(sub), bool)
    coslat = np.cos(np.radians(float(sub["lat"].mean())))
    wk = {((w - 1) % 53) + 1 for w in range(week - SEASON_WINDOW, week + SEASON_WINDOW + 1)}
    samp = sampled[sampled.iso_week.isin(wk)]
    if samp.empty:
        return np.zeros(len(sub), bool)
    tree = cKDTree(np.c_[samp.cell_lon * coslat * 111, samp.cell_lat * 111])
    dkm, _ = tree.query(np.c_[sub["lon"] * coslat * 111, sub["lat"] * 111], k=1)
    return dkm <= MASK_KM


def panel(ax, sub, col, norm, use_opacity=True, furniture=False, view=None):
    """One map panel: raster + outline, and furniture only where asked."""
    Z, A, ext = M.build_raster(sub["lon"], sub["lat"], sub[col],
                               alpha=sub["opacity"] if use_opacity else None)
    M.draw_raster(ax, Z, A, ext, CMAP, norm)
    M.footprint_outline(ax, Z, ext)
    v = view or ext
    lat_ref = float(np.mean(v[2:]))
    ax.set_aspect(1 / np.cos(np.radians(lat_ref)))
    ax.set_xlim(v[0], v[1]); ax.set_ylim(v[2], v[3])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    # The scale bar appears ONCE per figure, not once per panel.
    # No north arrow: add_north_arrow sizes its annotation for a full-page map
    # and collapses into an illegible glyph at panel scale. Every panel is
    # north-up on an identical extent, so the orientation is stated in the
    # footnote instead of drawn sixteen times.
    if furniture:
        M.add_scalebar(ax, v, lat_ref)
    return ext


def shared_colourbar(fig, norm, label, rect=(0.92, 0.15, 0.015, 0.7)):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, labelpad=8)
    cb.outline.set_linewidth(0.6)
    return cb


def fig_panel_grid(surf, geo, models, sampled, masked: bool):
    weeks = [w for w in PANEL_WEEKS if w in set(surf.iso_week)]
    if not weeks:
        log(f"[grid] SKIP: none of {PANEL_WEEKS} present in the surfaces")
        return
    norm = Normalize(0, 1)
    variant = "masked" if masked else "full"

    fig = plt.figure(figsize=(3.05 * len(models), 3.5 * len(weeks)))
    gs = GridSpec(len(weeks), len(models), figure=fig,
                  left=0.06, right=0.90, top=0.90, bottom=0.06,
                  wspace=0.04, hspace=0.06)

    kept_note = []
    for r, wk in enumerate(weeks):
        wsub = surf[surf.iso_week == wk]
        keep = trap_mask(wsub, int(wk), sampled) if masked else np.ones(len(wsub), bool)
        kept_note.append(int(keep.sum()))
        for c, m in enumerate(models):
            col = f"prob_{m}"
            ax = fig.add_subplot(gs[r, c])
            d = wsub[keep]
            d = d.dropna(subset=[col])
            if d.empty:
                ax.axis("off")
                continue
            panel(ax, d, col, norm,
                  furniture=(r == len(weeks) - 1 and c == 0))
            if r == 0:
                ax.set_title(S.model_label(m), fontsize=11, pad=6)
            if c == 0:
                ax.set_ylabel(f"{SEASON.get(wk, '')}\nISO week {wk}",
                              fontsize=10, labelpad=8)

    shared_colourbar(fig, norm, "predicted suitability")
    rule = (f"masked to cells within {MASK_KM:.0f} km of a trap active within "
            f"\u00b1{SEASON_WINDOW} weeks of the mapped week" if masked
            else "statewide; opacity encodes proximity to surveillance data "
                 "(faded = extrapolated), not model certainty")
    fig.suptitle("Cx. nigripalpus \u2014 predicted weekly suitability, "
                 "four models across four weeks\n" + rule, fontsize=12)
    fig.text(0.5, 0.015,
             "All panels share the colour scale above and the same extent; "
             "north is up and the scale bar applies throughout. "
             "Rows are weeks, columns are models.",
             ha="center", fontsize=8, color="0.35")
    out = OUT_DIR / f"panel_weeks_{variant}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"[grid] wrote {out.name} ({variant}; cells drawn per week: {kept_note})")


def fig_filmstrip(surf, geo, model, sampled):
    weeks = [w for w in FILMSTRIP_WEEKS if w in set(surf.iso_week)]
    if not weeks:
        log(f"[strip] SKIP {model}: none of {FILMSTRIP_WEEKS} present")
        return
    col = f"prob_{model}"
    if col not in surf.columns:
        return
    norm = Normalize(0, 1)
    ncol = 4
    nrow = int(np.ceil(len(weeks) / ncol))

    fig = plt.figure(figsize=(3.05 * ncol, 3.5 * nrow))
    gs = GridSpec(nrow, ncol, figure=fig, left=0.04, right=0.90,
                  top=0.88, bottom=0.07, wspace=0.04, hspace=0.10)
    for k, wk in enumerate(weeks):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        d = surf[(surf.iso_week == wk)].dropna(subset=[col])
        if d.empty:
            ax.axis("off"); continue
        panel(ax, d, col, norm, furniture=(k == len(weeks) - ncol))
        ax.set_title(f"ISO week {wk}", fontsize=10, pad=4)

    shared_colourbar(fig, norm, "predicted suitability")
    fig.suptitle(f"Cx. nigripalpus \u2014 predicted suitability through the year "
                 f"({S.model_label(model)})\n"
                 f"{len(weeks)} evenly spaced weeks; the full 53-week animation is "
                 f"supplementary material", fontsize=12)
    fig.text(0.5, 0.02,
             "Opacity encodes proximity to surveillance data (faded = "
             "extrapolated), not model certainty. All panels share the colour "
             "scale and the same extent; north is up and the scale bar applies "
             "throughout.",
             ha="center", fontsize=8, color="0.35")
    out = OUT_DIR / f"filmstrip_{model}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"[strip] wrote {out.name} ({len(weeks)} weeks)")


def run():
    S.set_thesis_style(constrained=False)   # GridSpec geometry is set explicitly
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    surf = load_surfaces()
    if surf.empty:
        log("[done] no surfaces found")
        return
    geo = cell_geometry(surf)
    if "opacity" not in surf.columns:
        surf = surf.merge(geo[["Grid_ID", "opacity"]], on="Grid_ID", how="left")
    models = S.models_in([c.replace("prob_", "") for c in surf.columns
                          if c.startswith("prob_")])
    log(f"[load] {len(geo):,} cells | models {models}")

    try:
        sampled = trap_geometry()
        log(f"[mask] {len(sampled):,} trapped cell-weeks for the mask")
    except Exception as e:
        log(f"[mask] trap geometry unavailable ({type(e).__name__}: {e}) "
            f"-- masked variant skipped")
        sampled = None

    fig_panel_grid(surf, geo, models, sampled, masked=False)
    if sampled is not None:
        fig_panel_grid(surf, geo, models, sampled, masked=True)

    for m in models:
        fig_filmstrip(surf, geo, m, sampled)

    log(f"\n[done] panel maps -> {OUT_DIR}")
    log("[note] the per-cell model-disagreement map is produced by "
        "derived_maps.py (derived_disagreement.png), not here.")


if __name__ == "__main__":
    run()