from __future__ import annotations
"""
validation_2018_cache.py  —  compute the 2018 validation surfaces ONCE and
persist them, so the panel figures do not have to refit anything.

WHY THIS EXISTS
  validation_2018_maps.py and maxent_validation_2018_map.py each build their
  statewide 2018 surface in memory, render a PNG and discard it. Nothing is
  written to disk, so any new figure over those surfaces would have to refit all
  four models again -- and would risk drifting from what the published maps show.

  This script IMPORTS those two frozen scripts and calls their own builders
  (load_daily, week_features, fit_calibrated, predict_grid, fade, trap_mask), so
  the cached surface is produced by exactly the code behind the existing maps.
  Neither frozen script is modified and no existing PNG is re-rendered.

BOTH MaxEnt VARIANTS
  maxent_validation_2018_map.py selects its variant with a module-level constant,
  VARIANT, read inside training_set() and fit_calibrated(). Setting MV.VARIANT
  and refitting therefore yields the target-group and vanilla surfaces from the
  same code path. The constant is restored afterwards so importing the module
  later behaves as its author intended.

ALL-WEEK POINT PREDICTIONS
  The maps need a statewide surface, which is expensive, so they are built for
  V.WEEKS only. The matched target-group vs vanilla comparison needs something
  much cheaper: predictions AT THE TRAP POINTS, which already carry their
  features in weekly_model_table.parquet. Those rows can be scored directly by
  the fitted models, with no grid construction at all.

  So POINT_ALL_WEEKS scores every 2018 cell-week -- about 2,454 rows across all
  53 weeks rather than 355 across five -- for a few seconds of extra work. The
  comparison stays perfectly matched (identical rows for both variants) and
  stops resting on 20-102 points per week.

  This does NOT change any map: the same five weeks are still surfaced.

WHAT IT WRITES (to the same OUT_DIR the frozen scripts use)
  surface_validation_2018.parquet
      Grid_ID, iso_week, lat, lon, opacity, keep_masked, prob_<model> x4
  observations_2018.parquet
      Grid_ID, iso_week, cell_lat, cell_lon, presence   (the trap outcomes)
  validation_2018_points.csv
      EVERY 2018 cell-week: Grid_ID, iso_week, presence, n_events and one
      prob_<model> column per model. Feed this to maxent_background_compare.py.

  keep_masked is the surveyed-footprint flag from the frozen trap_mask, stored
  rather than recomputed, so masked figures agree with the published single-week
  maps cell for cell.

  Weeks come from validation_2018_maps.WEEKS, so the cache always covers exactly
  the weeks the frozen script maps.

RUN THIS BEFORE validation_2018_panels.py. It is the slow step -- four model
fits plus one daily-file pass per module -- so the figures can then be iterated
on cheaply.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import validation_2018_maps as V              # frozen: XGB + RF

try:
    import maxent_validation_2018_map as MV    # frozen: MaxEnt
except Exception as e:
    MV = None
    print(f"[cache] MaxEnt validation module unavailable ({type(e).__name__}: {e})",
          flush=True)

MAXENT_VARIANTS = ["targetgroup", "vanilla"]   # [] to skip MaxEnt entirely
POINT_ALL_WEEKS = True     # score every 2018 cell-week, not just V.WEEKS

def log(m): print(m, flush=True)


def run():
    OUT_DIR = V.OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = json.load(open(V.DATA_DIR / "model_features.json"))
    feats, static_cols = spec["model_features"], spec["static"]
    table = pd.read_parquet(V.DATA_DIR / "weekly_model_table.parquet")
    train = table[table.iso_year.isin(V.TRAIN_YEARS)].copy()

    # the footprint definition is the frozen script's, not a new one
    foot = table if V.MASK_FROM_ALL_YEARS else table[table.iso_year == V.TEST_YEAR]
    sampled = (foot.groupby([V.GRID_ID_COL, "iso_week"])[["cell_lat", "cell_lon"]]
                   .first().reset_index())
    log(f"[cache] train rows {len(train):,} | weeks {V.WEEKS} | "
        f"footprint cells {sampled[V.GRID_ID_COL].nunique():,}")

    # ---------------------------------------------------------- tree models
    daily, daily_file = V.load_daily()
    static_cells = V.static_per_cell(daily_file, static_cols)
    fitted = {}
    for name in V.MODELS:
        if name == "xgboost" and V.xgb is None:
            log("[cache] skip xgboost (not installed)"); continue
        fitted[name] = V.fit_calibrated(name, train, feats)

    frames = {}
    for week in V.WEEKS:
        gf, mon = V.week_features(daily, week, feats, static_cells)
        ids = gf[V.GRID_ID_COL].to_numpy()
        lat, lon = V.lonlat(ids)
        f = pd.DataFrame({V.GRID_ID_COL: ids, "iso_week": week,
                          "lat": lat, "lon": lon,
                          "opacity": V.fade(ids, week, sampled),
                          "keep_masked": V.trap_mask(ids, week, sampled)})
        for name, model in fitted.items():
            f[f"prob_{name}"] = V.predict_grid(model, feats, gf)
        frames[week] = f
        log(f"[cache] week {week:>2}: {len(f):,} cells, "
            f"masked {int(f.keep_masked.sum()):,}")

    # ------------------------------------------- all-week point predictions
    points = pd.DataFrame()
    if POINT_ALL_WEEKS:
        keep = [c for c in [V.GRID_ID_COL, "iso_week", "iso_year", "presence",
                            "n_events", "cell_lat", "cell_lon"] if c in table.columns]
        points = table[table.iso_year == V.TEST_YEAR][keep + feats].copy()
        for name, model in fitted.items():
            # predict_grid takes any frame carrying the feature columns, so the
            # observed rows are scored by the same fitted+calibrated model that
            # produced the maps -- no refit, no grid, no reimplementation
            points[f"prob_{name}"] = V.predict_grid(model, feats, points)
        log(f"[cache] point predictions: {len(points):,} cell-weeks across "
            f"{points.iso_week.nunique()} weeks "
            f"({points[V.GRID_ID_COL].nunique()} cells, "
            f"prevalence {points.presence.mean():.3f})")

    # --------------------------------------------------------------- MaxEnt
    if MV is not None and MAXENT_VARIANTS:
        original = MV.VARIANT
        m_daily, m_file = MV.load_daily()
        m_static = (pd.read_parquet(m_file, columns=[MV.GRID_ID_COL] + static_cols)
                      .groupby(MV.GRID_ID_COL).first().reset_index())
        try:
            for variant in MAXENT_VARIANTS:
                MV.VARIANT = variant          # read inside training_set/fit_calibrated
                try:
                    mtrain = MV.training_set(feats)
                    model = MV.fit_calibrated(mtrain, feats)
                except Exception as e:
                    log(f"[cache] maxent_{variant} FAILED to fit "
                        f"({type(e).__name__}: {e}) -- column omitted")
                    continue
                for week in V.WEEKS:
                    gf, _ = MV.week_features(m_daily, week, feats, m_static)
                    p = MV.predict_grid(model, feats, gf)
                    # align by Grid_ID, not by position: the two modules build
                    # their grids independently and need not share a row order
                    s = pd.Series(p, index=gf[MV.GRID_ID_COL].to_numpy())
                    col = f"prob_maxent_{variant}"
                    frames[week][col] = frames[week][V.GRID_ID_COL].map(s)
                    matched = frames[week][col].notna().mean() * 100
                    if matched < 99:
                        log(f"[cache] week {week:>2} maxent_{variant}: only "
                            f"{matched:.1f}% of cells matched by Grid_ID")
                if POINT_ALL_WEEKS and len(points):
                    points[f"prob_maxent_{variant}"] = MV.predict_grid(
                        model, feats, points)
                log(f"[cache] maxent_{variant} done for all weeks")
        finally:
            MV.VARIANT = original             # leave the module as we found it

    # --------------------------------------------------------------- write
    surf = pd.concat(frames.values(), ignore_index=True)
    obs = table[(table.iso_year == V.TEST_YEAR) & (table.iso_week.isin(V.WEEKS))][
        [V.GRID_ID_COL, "iso_week", "cell_lat", "cell_lon", "presence"]].copy()

    surf.to_parquet(OUT_DIR / "surface_validation_2018.parquet", index=False)
    obs.to_parquet(OUT_DIR / "observations_2018.parquet", index=False)
    if POINT_ALL_WEEKS and len(points):
        pcols = [c for c in points.columns if c.startswith("prob_")]
        out = points[[c for c in points.columns if not c in feats]]
        out.to_csv(OUT_DIR / "validation_2018_points.csv", index=False)
        log(f"[cache] wrote validation_2018_points.csv "
            f"({len(out):,} rows x {len(pcols)} models, all weeks)")
        per_wk = out.groupby("iso_week").agg(n=("presence", "size"),
                                             n_pres=("presence", "sum"))
        per_wk["n_abs"] = per_wk.n - per_wk.n_pres
        thin = int((per_wk.n_abs < 10).sum())
        log(f"[cache] {thin} of {len(per_wk)} weeks have fewer than 10 absences")

    probs = [c for c in surf.columns if c.startswith("prob_")]
    log(f"\n[cache] wrote surface_validation_2018.parquet "
        f"({len(surf):,} cell-weeks, {probs})")
    log(f"[cache] wrote observations_2018.parquet ({len(obs):,} trap points)")
    log(f"[cache] -> {OUT_DIR}")
    log("[next] run validation_2018_panels.py")
    return surf, obs


if __name__ == "__main__":
    run()