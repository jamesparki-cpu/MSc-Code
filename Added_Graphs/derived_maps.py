from __future__ import annotations
"""
derived_maps.py  —  three whole-year summary maps, one PNG each.

The weekly series and the animations show suitability week by week. These
collapse the year into single images that answer questions the series cannot:

  1. AMPLITUDE      derived_amplitude_<model>.png
     Per cell, max minus min predicted suitability across the 53 weeks. Where
     the surface is flat, the model is predicting from static cell properties
     alone; where it swings, weekly conditions are moving the prediction. This
     is the spatial counterpart of the static/dynamic SHAP split -- SHAP says
     how much weight the static block carries on average, this says WHERE.

  2. WEEK OF PEAK   derived_weekofpeak_<model>.png
     Per cell, the ISO week of maximum predicted suitability, on a CYCLIC
     colour map (week 52 and week 1 must not read as opposite ends of a ramp).
     One image summarising predicted phenology statewide.

     Cells whose amplitude is below AMPLITUDE_FLOOR are MASKED: if the annual
     range is 0.01, the argmax is noise and mapping it would invent a
     phenological pattern that the model is not making. The masked fraction is
     reported, and is itself a result.

  3. DISAGREEMENT   derived_disagreement.png
     Per cell, the standard deviation across models within each week, averaged
     over weeks. Identifies where the four families diverge, i.e. where an
     ensemble reading would be unreliable. Restricted to cells present in
     every model's surface; the count dropped is logged.

STYLE: furniture from map_style.py (the same helpers the frozen map scripts
vendor), typography from report_style.py. Nothing here re-renders or modifies
the existing maps.

INPUTS
  <weekly_results_dir>/surface_predictions.parquet    prob_xgboost, prob_random_forest
  <maxent_dir>/surface_maxent_<variant>.parquet       prob_maxent_<variant>
  each with Grid_ID, iso_week, lon, lat and (for the first) opacity

OUTPUT (to <weekly_results_dir>/derived_maps)
  derived_amplitude_<model>.png     + derived_amplitude.csv
  derived_weekofpeak_<model>.png    + derived_weekofpeak.csv
  derived_disagreement.png          + derived_disagreement.csv
"""
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import report_style as S
import map_style as M

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
RESULTS = Path(cfg.get("weekly_results_dir", cfg.get("weekly_xg_dir", ".")))
MAXENT_DIR = Path(cfg.get("maxent_dir", str(RESULTS)))
OUT_DIR = RESULTS / "derived_maps"

SURFACES = RESULTS / "surface_predictions.parquet"
GRID_ID_COL = "Grid_ID"
OPACITY_COL = "opacity"

DPI = 220                    # matches the existing map series
FIGSIZE = (6.6, 7.6)

CMAP_AMPLITUDE = "viridis"       # sequential; deliberately NOT the suitability map
CMAP_PEAK = "twilight_shifted"   # CYCLIC: week 53 sits next to week 1
CMAP_DISAGREE = "magma_r"        # dark = disagree, matching the agreement series

# Below this annual range, the week of peak suitability is not meaningful.
AMPLITUDE_FLOOR = 0.05

MONTH_STARTS = {1: "Jan", 6: "Feb", 10: "Mar", 14: "Apr", 19: "May", 23: "Jun",
                27: "Jul", 32: "Aug", 36: "Sep", 40: "Oct", 45: "Nov", 49: "Dec"}

def log(m): print(m, flush=True)
# ============================================================================


def load_surfaces() -> pd.DataFrame:
    """Wide frame: Grid_ID, iso_week, lon, lat, opacity, one prob_* per model."""
    if not SURFACES.exists():
        log(f"[load] {SURFACES} not found")
        return pd.DataFrame()
    surf = pd.read_parquet(SURFACES)
    keep = [c for c in [GRID_ID_COL, "iso_week", "lon", "lat", OPACITY_COL]
            if c in surf.columns] + [c for c in surf.columns if c.startswith("prob_")]
    out = surf[keep].copy()
    log(f"[load] surface_predictions.parquet: {len(out):,} cell-weeks, "
        f"{[c for c in out.columns if c.startswith('prob_')]}")

    for f in sorted(glob.glob(str(MAXENT_DIR / "surface_maxent_*.parquet"))):
        d = pd.read_parquet(f)
        pcol = [c for c in d.columns if c.startswith("prob_")]
        if not pcol:
            log(f"[load] {Path(f).name}: no prob_ column, skipped")
            continue
        pcol = pcol[0]
        # name the column after the model, so downstream code needs no aliasing
        variant = pcol.replace("prob_", "")
        name = variant if variant.startswith("maxent") else f"maxent_{variant}"
        d = d[[GRID_ID_COL, "iso_week", pcol]].rename(columns={pcol: f"prob_{name}"})
        before = len(out)
        out = out.merge(d, on=[GRID_ID_COL, "iso_week"], how="left")
        got = out[f"prob_{name}"].notna().sum()
        log(f"[load] {Path(f).name}: joined prob_{name} "
            f"({got:,}/{before:,} rows matched)")
    return out


def cell_geometry(surf: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: coordinates and an opacity for the fade."""
    agg = {"lon": ("lon", "first"), "lat": ("lat", "first")}
    if OPACITY_COL in surf.columns:
        agg["opacity"] = (OPACITY_COL, "mean")
    g = surf.groupby(GRID_ID_COL).agg(**agg).reset_index()
    if "opacity" not in g.columns:
        g["opacity"] = 1.0
    return g


def draw(geo: pd.DataFrame, values: pd.Series, cmap, norm, title, cbar_label,
         path, caption, cbar_ticks=None, cbar_ticklabels=None, use_opacity=True):
    sub = geo.assign(_v=values.to_numpy())
    Z, A, ext = M.build_raster(sub["lon"], sub["lat"], sub["_v"],
                               alpha=sub["opacity"] if use_opacity else None)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    M.draw_raster(ax, Z, A, ext, cmap, norm)
    ax.set_title(title, fontsize=12, linespacing=1.35)
    M.finish_map(fig, ax, Z, ext, cmap, norm, cbar_label, caption=caption)
    if cbar_ticks is not None:
        cb = fig.axes[-1]
        cb.set_yticks(cbar_ticks)
        if cbar_ticklabels is not None:
            cb.set_yticklabels(cbar_ticklabels, fontsize=8)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"[map] wrote {path.name}")


def run():
    S.set_thesis_style()          # constrained layout, as the map scripts use
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    surf = load_surfaces()
    if surf.empty:
        log("[done] nothing to render")
        return
    prob_cols = [c for c in surf.columns if c.startswith("prob_")]
    models = S.models_in([c.replace("prob_", "") for c in prob_cols])
    geo = cell_geometry(surf)
    n_weeks = surf.iso_week.nunique()
    log(f"[grid] {len(geo):,} cells x {n_weeks} weeks | models {models}")

    amp_rows, peak_rows = {}, {}

    # ------------------------------------------------ 1 + 2, per model
    for m in models:
        col = f"prob_{m}"
        d = surf[[GRID_ID_COL, "iso_week", col]].dropna(subset=[col])
        if d.empty:
            log(f"[skip] {m}: no predictions")
            continue
        g = d.groupby(GRID_ID_COL)[col]
        amp = (g.max() - g.min()).rename("amplitude")
        peak = d.loc[g.idxmax()].set_index(GRID_ID_COL)["iso_week"].rename("week_of_peak")
        cover = d.groupby(GRID_ID_COL).size().rename("n_weeks")

        per_cell = pd.concat([amp, peak, cover], axis=1).reindex(geo[GRID_ID_COL])
        amp_rows[m] = per_cell.amplitude
        peak_rows[m] = per_cell.week_of_peak

        log(f"[{m}] amplitude: median {per_cell.amplitude.median():.3f}, "
            f"p10 {per_cell.amplitude.quantile(.10):.3f}, "
            f"p90 {per_cell.amplitude.quantile(.90):.3f}")

        # ---- amplitude map
        vmax = float(np.nanpercentile(per_cell.amplitude, 99))
        draw(geo, per_cell.amplitude.fillna(np.nan),
             CMAP_AMPLITUDE, Normalize(0, max(vmax, 1e-3)),
             f"Cx. nigripalpus — seasonal amplitude ({S.model_label(m)})\n"
             f"max \u2212 min across {n_weeks} weeks · flat = driven by "
             f"static cell properties",
             "annual range of predicted suitability",
             OUT_DIR / f"derived_amplitude_{m}.png", M.CAPTION)

        # ---- week-of-peak map, masked where the range is too small to mean anything
        flat = per_cell.amplitude < AMPLITUDE_FLOOR
        pct_flat = 100 * flat.mean()
        pk = per_cell.week_of_peak.astype(float).copy()
        pk[flat] = np.nan
        log(f"[{m}] week-of-peak: {pct_flat:.1f}% of cells masked "
            f"(amplitude < {AMPLITUDE_FLOOR})")
        ticks = sorted(MONTH_STARTS)
        draw(geo, pk, CMAP_PEAK, Normalize(1, 53),
             f"Cx. nigripalpus — week of peak suitability ({S.model_label(m)})\n"
             f"cyclic scale · {pct_flat:.0f}% masked (annual range < "
             f"{AMPLITUDE_FLOOR:g}: peak week not meaningful)",
             "ISO week of maximum suitability",
             OUT_DIR / f"derived_weekofpeak_{m}.png", M.CAPTION,
             cbar_ticks=ticks,
             cbar_ticklabels=[MONTH_STARTS[t] for t in ticks])

    if amp_rows:
        pd.DataFrame(amp_rows).assign(Grid_ID=geo[GRID_ID_COL].to_numpy()).to_csv(
            OUT_DIR / "derived_amplitude.csv", index=False)
        pd.DataFrame(peak_rows).assign(Grid_ID=geo[GRID_ID_COL].to_numpy()).to_csv(
            OUT_DIR / "derived_weekofpeak.csv", index=False)

    # ------------------------------------------------ 3, across models
    have = [f"prob_{m}" for m in models if f"prob_{m}" in surf.columns]
    if len(have) < 2:
        log("[disagree] SKIP: need at least two models")
        return
    d = surf[[GRID_ID_COL, "iso_week"] + have].dropna()
    dropped = len(surf) - len(d)
    log(f"[disagree] {len(d):,} cell-weeks scored by all {len(have)} models "
        f"({dropped:,} dropped for incomplete coverage)")
    if d.empty:
        log("[disagree] SKIP: no cell-weeks are scored by every model")
        return

    # SD across models WITHIN each week, then averaged over weeks. Taking the SD
    # of annual means instead would hide weeks where the models diverge and then
    # reconverge.
    d = d.assign(_sd=d[have].std(axis=1, ddof=0))
    per_cell = d.groupby(GRID_ID_COL)._sd.agg(["mean", "max", "size"])
    per_cell.columns = ["mean_sd", "max_sd", "n_weeks"]
    per_cell = per_cell.reindex(geo[GRID_ID_COL])
    per_cell.reset_index().to_csv(OUT_DIR / "derived_disagreement.csv", index=False)
    log(f"[disagree] mean SD: median {per_cell.mean_sd.median():.3f}, "
        f"p90 {per_cell.mean_sd.quantile(.90):.3f}, "
        f"max {per_cell.mean_sd.max():.3f}")

    vmax = float(np.nanpercentile(per_cell.mean_sd, 99))
    draw(geo, per_cell.mean_sd, CMAP_DISAGREE, Normalize(0, max(vmax, 1e-3)),
         f"Cx. nigripalpus — between-model disagreement ({len(have)} models)\n"
         f"per-week SD across models, averaged over {n_weeks} weeks · "
         f"dark = ensemble unreliable",
         "mean between-model SD of predicted suitability",
         OUT_DIR / "derived_disagreement.png", M.CAPTION)

    log(f"\n[done] derived maps -> {OUT_DIR}")


if __name__ == "__main__":
    run()