from __future__ import annotations
"""
sample_effort_graphs.py  —  the figure that makes the spatial-CV collapse legible.

ARGUMENT THE FIGURE MAKES
  Spatial cross-validation fails not because the models are wrong but because the
  surveillance data cannot support a spatial-transfer test. Three panels build
  that case:

    (a) WHERE the traps are     surveyed cells against the full state grid,
                                sized and coloured by cell-weeks contributed
    (b) HOW UNEVEN effort is    distribution of cell-weeks per cell -- a long
                                right tail means a handful of stations dominate
    (c) WHEN the traps ran      cell-weeks per calendar month, STACKED BY REGION,
                                so seasonal effort and its geography are read
                                together: if northern regions contribute only in
                                part of the year, the model has no counter-
                                evidence for the rest of it

  Together: sampling is clustered in space, uneven between cells, and seasonally
  restricted in a region-dependent way.

  Per-BLOCK figures (cells, cell-weeks, presence rate per KMeans spatial block)
  are still computed and written to sampling_effort_summary.csv, but are no
  longer plotted -- the block-level story belongs with the CV design table in
  R1, not in this figure.

OUTPUT (to <weekly_xg_dir>)
  sampling_effort.png
  sampling_effort_summary.csv       per spatial block
  sampling_effort_regions.csv       per region: sites, cell-weeks, months, coverage
  sampling_effort_region_month.csv  the panel (c) matrix
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import report_style as S

GRID_ID_COL = "Grid_ID"
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")

# ---------------------------------------------------------------------------
# REGIONS
# Ordered; FIRST MATCH WINS, so the longitude test for the Panhandle must come
# before the latitude bands. Boundaries are conventional rather than official --
# state them in the caption so they are reproducible, and adjust freely.
#   Tallahassee 30.44N -84.28W | Jacksonville 30.33N -81.66W
#   Orlando     28.54N -81.38W | Tampa        27.95N -82.46W
#   Fort Myers  26.64N -81.87W | Miami        25.76N -80.19W
# ---------------------------------------------------------------------------
REGIONS = [
    ("Panhandle",        lambda lat, lon: lon < -84.0),
    ("North Florida",    lambda lat, lon: lat >= 29.6),
    ("Central",          lambda lat, lon: lat >= 27.8),
    ("Southwest",        lambda lat, lon: lat >= 25.9 and lon < -81.3),
    ("Southeast",        lambda lat, lon: lat >= 25.9),
    ("Keys / far south", lambda lat, lon: True),
]
REGION_ORDER = [r[0] for r in REGIONS]

# Region palette, deliberately disjoint from every palette in report_style
# (models / schemes / SHAP blocks), so no hex carries two meanings.
REGION_COLOURS = {
    "Panhandle":        "#E69F00",
    "North Florida":    "#56B4E9",
    "Central":          "#009E73",
    "Southwest":        "#0072B2",
    "Southeast":        "#D55E00",
    "Keys / far south": "#CC79A7",
}

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def log(m): print(m, flush=True)


def centroids(df: pd.DataFrame):
    """Prefer the table's own coordinates; fall back to parsing Grid_ID."""
    if {"cell_lat", "cell_lon"} <= set(df.columns):
        return df["cell_lat"].to_numpy(float), df["cell_lon"].to_numpy(float)
    lat, lon = [], []
    for g in df[GRID_ID_COL].astype(str):
        m = _GID.search(g)
        if not m:
            raise ValueError(f"cannot parse centroid from Grid_ID: {g!r}")
        lat.append(float(m.group(1))); lon.append(float(m.group(2)))
    return np.asarray(lat), np.asarray(lon)


def assign_region(lat, lon) -> str:
    for name, test in REGIONS:
        if test(lat, lon):
            return name
    return "unassigned"


def month_of(df: pd.DataFrame) -> pd.Series:
    """Calendar month. Uses week_start if present, else converts ISO week."""
    if "week_start" in df.columns:
        return pd.to_datetime(df["week_start"]).dt.month
    return pd.to_datetime(
        df["iso_year"].astype(str) + "-" +
        df["iso_week"].astype(int).astype(str) + "-1",
        format="%G-%V-%u", errors="coerce").dt.month


def run():
    S.set_thesis_style(constrained=False)

    with open("config.json") as f:
        cfg = json.load(f)
    FEAT_DIR = Path(cfg["weekly_xg_dir"])

    df = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet")
    log(f"[load] {len(df):,} cell-weeks x {df[GRID_ID_COL].nunique():,} surveyed cells")

    if "spatial_block" not in df.columns:
        import cv_harness as H
        df = H.build_blocks(df)
        log("[blocks] rebuilt spatial_block via cv_harness")

    df = df.copy()
    df["month"] = month_of(df)

    # ------------------------------------------------------- per-cell effort
    agg = {"cell_weeks": ("presence", "size"),
           "presence_rate": ("presence", "mean"),
           "spatial_block": ("spatial_block", "first")}
    if "n_events" in df.columns:
        agg["trap_events"] = ("n_events", "sum")
    for c in ("cell_lat", "cell_lon"):
        if c in df.columns:
            agg[c] = (c, "first")
    per_cell = df.groupby(GRID_ID_COL).agg(**agg).reset_index()

    lat, lon = centroids(per_cell)
    per_cell["lat"], per_cell["lon"] = lat, lon
    per_cell["region"] = [assign_region(a, o) for a, o in zip(lat, lon)]
    df = df.merge(per_cell[[GRID_ID_COL, "region"]], on=GRID_ID_COL, how="left")

    # ------------------------------------------------------ regional summary
    reg = (df.groupby("region")
             .agg(cell_weeks=("presence", "size"),
                  sites=(GRID_ID_COL, "nunique"),
                  months_covered=("month", "nunique"),
                  weeks_covered=("iso_week", "nunique"),
                  presence_rate=("presence", "mean")))
    reg = reg.reindex([r for r in REGION_ORDER if r in reg.index])
    reg["pct_of_cell_weeks"] = 100 * reg.cell_weeks / reg.cell_weeks.sum()
    reg["cell_weeks_per_site"] = reg.cell_weeks / reg.sites
    if "trap_events" in per_cell.columns:
        reg["trap_events"] = per_cell.groupby("region").trap_events.sum().reindex(reg.index)
    reg = reg.reset_index()
    reg.round(3).to_csv(FEAT_DIR / "sampling_effort_regions.csv", index=False)

    log("\n[regions] surveillance effort by region "
        "(first-match boundaries; see REGIONS in this script):")
    log(reg.round(2).to_string(index=False))
    unassigned = int((per_cell.region == "unassigned").sum())
    if unassigned:
        log(f"[regions] WARNING {unassigned} cells unassigned -- widen REGIONS")

    # ------------------------------------------------- region x month matrix
    rm = (df.groupby(["region", "month"]).size()
            .unstack("month")
            .reindex(index=[r for r in REGION_ORDER if r in set(df.region)],
                     columns=range(1, 13))
            .fillna(0.0))
    rm.to_csv(FEAT_DIR / "sampling_effort_region_month.csv")

    # ----------------------------------------------------- full state grid
    grid_path = FEAT_DIR / "statewide_weekly_features.parquet"
    state_lat = state_lon = n_state = None
    if grid_path.exists():
        allc = (pd.read_parquet(grid_path, columns=[GRID_ID_COL])[GRID_ID_COL]
                  .drop_duplicates().to_frame())
        state_lat, state_lon = centroids(allc)
        n_state = len(allc)
        log(f"\n[coverage] {len(per_cell):,} of {n_state:,} state cells surveyed "
            f"({100 * len(per_cell) / n_state:.1f}%)")
    else:
        log("\n[coverage] statewide grid not found -- panel (a) shows surveyed cells only")

    # ================================================================ FIGURE
    fig = plt.figure(figsize=(13, 8.4))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1], hspace=0.38, wspace=0.26)

    # ---------------------------------------------------------- (a) the map
    ax = fig.add_subplot(gs[:, 0])
    if state_lat is not None:
        ax.scatter(state_lon, state_lat, s=3, marker="s", c="0.90", linewidths=0,
                   label="Florida 5 km grid (unsurveyed)", zorder=1)
    sc = ax.scatter(per_cell.lon, per_cell.lat,
                    s=8 + 40 * per_cell.cell_weeks / per_cell.cell_weeks.max(),
                    c=per_cell.cell_weeks, cmap="viridis", alpha=0.85,
                    linewidths=0.3, edgecolors="white", zorder=2)
    ax.set_aspect(1 / np.cos(np.radians(float(per_cell.lat.mean()))))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    title = "(a) Surveillance effort by cell"
    if n_state:
        title += (f"\n{len(per_cell):,} of {n_state:,} state cells surveyed "
                  f"({100 * len(per_cell) / n_state:.1f}%)")
    ax.set_title(title, fontsize=11, loc="left")
    fig.colorbar(sc, ax=ax, shrink=0.55, label="cell-weeks contributed")
    if state_lat is not None:
        ax.legend(loc="lower left", fontsize=7, frameon=False)

    # ------------------------------------------- (b) effort is highly uneven
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(per_cell.cell_weeks, bins=40, color="#4C72B0",
            edgecolor="white", linewidth=0.4)
    med = per_cell.cell_weeks.median()
    p90 = per_cell.cell_weeks.quantile(0.90)
    ax.axvline(med, color="#C44E52", ls="--", lw=1.2, label=f"median {med:.0f}")
    ax.axvline(p90, color="#DD8452", ls=":", lw=1.2, label=f"90th pct {p90:.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("cell-weeks per cell"); ax.set_ylabel("number of cells (log)")
    top10 = 100 * per_cell.nlargest(max(1, len(per_cell) // 10),
                                    "cell_weeks").cell_weeks.sum() / per_cell.cell_weeks.sum()
    ax.set_title(f"(b) Effort is highly uneven\ntop 10% of cells contribute "
                 f"{top10:.0f}% of all cell-weeks", fontsize=11, loc="left")
    ax.legend(fontsize=7, frameon=False)

    # ------------------------------ (c) monthly effort, stacked by region
    ax = fig.add_subplot(gs[1, 1])
    months = np.arange(1, 13)
    bottom = np.zeros(12)
    for region in rm.index:
        vals = rm.loc[region].to_numpy(float)
        ax.bar(months, vals, width=0.82, bottom=bottom, label=region,
               color=REGION_COLOURS.get(region, "#999999"),
               edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_ABBR, fontsize=8)
    ax.set_xlabel("month"); ax.set_ylabel("cell-weeks")
    ax.grid(axis="y", alpha=0.3)
    peak, trough = int(np.argmax(bottom)) + 1, int(np.argmin(bottom)) + 1
    ratio = bottom.max() / max(bottom.min(), 1)
    ax.set_title(f"(c) Effort is seasonal, and its geography changes\n"
                 f"peak {MONTH_ABBR[peak - 1]} vs trough {MONTH_ABBR[trough - 1]}: "
                 f"{ratio:.1f}x more cell-weeks", fontsize=11, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")

    fig.suptitle("Surveillance effort and its consequences for spatial cross-validation",
                 fontsize=13, y=0.985)
    # no tight_layout: the colourbar on panel (a) creates axes it cannot lay
    # out; the GridSpec ratios above already fix the geometry
    fig.subplots_adjust(left=0.055, right=0.975, top=0.90, bottom=0.075)
    S.save(fig, FEAT_DIR / "sampling_effort.png")

    # --------------------------- per-block table (CSV only, no longer plotted)
    blk = (df.groupby("spatial_block")
             .agg(cell_weeks=("presence", "size"),
                  presence_rate=("presence", "mean"))
             .join(per_cell.groupby("spatial_block").size().rename("cells"))
             .reset_index().sort_values("spatial_block"))
    blk.round(3).to_csv(FEAT_DIR / "sampling_effort_summary.csv", index=False)
    log("\n[blocks] per-block figures (CSV only -- no longer plotted):")
    log(blk.round(3).to_string(index=False))
    spread = blk.presence_rate.max() - blk.presence_rate.min()
    log(f"[blocks] presence rate spans {blk.presence_rate.min():.3f}"
        f"-{blk.presence_rate.max():.3f} (spread {spread:.3f})")

    log("\n[read] (b) quantifies clustering between cells; (c) shows effort is "
        "seasonal AND that which regions contribute changes through the year, so "
        "a held-out block differs from the training blocks in when it was "
        "sampled as well as where.")
    return reg, blk


if __name__ == "__main__":
    run()