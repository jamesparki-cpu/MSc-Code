from __future__ import annotations
"""
build_oof_long.py  —  assemble every model's pooled OOF into ONE tidy file for
bootstrap_uncertainty.py.

Inputs
  <comparison_dir>/oof_long_trees.csv     written by the patched compare_models.py
                                          (calibrated XGB + RF, already long form)
  <maxent_dir>/oof_maxent_vanilla.csv     written by maxent_common.evaluate_variant
  <maxent_dir>/oof_maxent_targetgroup.csv

SPATIAL_BLOCK RECONSTRUCTION
  The MaxEnt OOF files predate the spatial_block column. Rather than re-running
  MaxEnt (hours), we rebuild the labels: H.build_blocks clusters CELL CENTROIDS
  with KMeans at a fixed seed, so re-running it on the same cell set reproduces
  the same labels exactly. Centroids are parsed back out of Grid_ID, which
  encodes them by construction.

  IMPORTANT: each variant is re-clustered on ITS OWN cell set. maxent_vanilla
  includes background cells the target-group set never contains, so its KMeans
  solution legitimately differs -- that is the partition its own evaluation used,
  and reusing the target-group partition would misstate its folds.

OUTPUT
  <comparison_dir>/oof_long.csv
    model, scheme, Grid_ID, iso_year, spatial_block, presence, p
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import cv_harness as H

GRID_ID_COL = "Grid_ID"
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")


def log(m): print(m, flush=True)


def add_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Recover cell_lat / cell_lon from Grid_ID (which encodes them)."""
    if {"cell_lat", "cell_lon"} <= set(df.columns):
        return df
    lat, lon = [], []
    for g in df[GRID_ID_COL].astype(str):
        m = _GID.search(g)
        if not m:
            raise ValueError(f"cannot parse centroid from Grid_ID: {g!r}")
        lat.append(float(m.group(1))); lon.append(float(m.group(2)))
    df = df.copy()
    df["cell_lat"], df["cell_lon"] = lat, lon
    return df


def load_maxent(path: Path, variant: str) -> pd.DataFrame:
    """Read one MaxEnt OOF file and melt its oof_<scheme> columns to long form."""
    df = pd.read_csv(path)
    oof_cols = [c for c in df.columns if c.startswith("oof_")]
    if not oof_cols:
        raise ValueError(f"{path.name} has no oof_* columns")

    if "spatial_block" not in df.columns:
        df = add_centroids(df)
        before = df.columns.tolist()
        df = H.build_blocks(df)                 # same KMeans, same seed
        log(f"[maxent_{variant}] rebuilt spatial_block "
            f"({df[GRID_ID_COL].nunique():,} cells -> "
            f"{df['spatial_block'].nunique()} blocks)")
        assert "spatial_block" in df.columns, "build_blocks did not add spatial_block"
    else:
        log(f"[maxent_{variant}] spatial_block already present")

    out = []
    for c in oof_cols:
        scheme = c.replace("oof_", "")
        r = df[[GRID_ID_COL, "iso_year", "spatial_block", "presence"]].copy()
        r["model"] = f"maxent_{variant}"
        r["scheme"] = scheme
        r["p"] = df[c].to_numpy()
        out.append(r)
    long = pd.concat(out, ignore_index=True)
    log(f"[maxent_{variant}] {len(long):,} rows across schemes {sorted(set(c.replace('oof_','') for c in oof_cols))}")
    return long


def run():
    with open("config.json") as f:
        cfg = json.load(f)
    FEAT_DIR = Path(cfg["weekly_xg_dir"])
    MAXENT_DIR = Path(cfg["maxent_dir"])
    OUT_DIR = Path(cfg.get("comparison_dir", str(FEAT_DIR.parent / "Comparison_Results")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []

    trees = OUT_DIR / "oof_long_trees.csv"
    if trees.exists():
        t = pd.read_csv(trees)
        frames.append(t)
        log(f"[trees] {len(t):,} rows | models {sorted(t['model'].unique())} "
            f"| schemes {sorted(t['scheme'].unique())}")
    else:
        log(f"[trees] MISSING {trees} -- patch compare_models.py and re-run it first")

    for variant in ("vanilla", "targetgroup"):
        f = MAXENT_DIR / f"oof_maxent_{variant}.csv"
        if f.exists():
            frames.append(load_maxent(f, variant))
        else:
            log(f"[maxent_{variant}] MISSING {f.name} -- skipping")

    if not frames:
        raise SystemExit("no OOF inputs found; nothing to assemble")

    cols = ["model", "scheme", GRID_ID_COL, "iso_year", "spatial_block", "presence", "p"]
    long = pd.concat([f[cols] for f in frames], ignore_index=True)

    n0 = len(long)
    long = long.dropna(subset=["p"])
    if len(long) != n0:
        log(f"[assemble] dropped {n0 - len(long):,} rows with NaN predictions")

    long.to_csv(OUT_DIR / "oof_long.csv", index=False)
    log(f"\n[done] wrote oof_long.csv ({len(long):,} rows) to {OUT_DIR}")

    # sanity summary: row counts must match what was scored
    summary = (long.groupby(["model", "scheme"])
                   .agg(n=("p", "size"), prevalence=("presence", "mean"))
                   .round(3).reset_index())
    log("\n[check] rows and prevalence per model x scheme:")
    log(summary.to_string(index=False))
    log("\n[note] maxent_vanilla prevalence should differ from the others "
        "(random background, not target-group absences) -- it is scored on a "
        "different evaluation set and must not be paired against them.")
    return long


if __name__ == "__main__":
    run()