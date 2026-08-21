from __future__ import annotations
"""
sampling_effort_figure.py  —  the figure that makes the spatial-CV collapse legible.

ARGUMENT THE FIGURE MAKES
  Spatial cross-validation fails not because the models are wrong but because the
  surveillance data cannot support a spatial-transfer test. Four panels build
  that case:

    (a) WHERE the traps are      surveyed cells against the full state grid,
                                 sized by how many cell-weeks each contributes
    (b) HOW UNEVEN effort is     distribution of cell-weeks per cell -- a long
                                 right tail means a handful of stations dominate
    (c) HOW UNBALANCED blocks are cells and cell-weeks per KMeans spatial block
    (d) WHY held-out blocks fail  presence rate varies by block, so a model
                                 trained on five blocks meets a different base
                                 rate in the sixth

  Together: sampling is clustered, blocks inherit that clustering, and a held-out
  block is not exchangeable with the training blocks.

OUTPUT (to <weekly_xg_dir>)
  sampling_effort.png
  sampling_effort_summary.csv    per-block figures behind panels (c) and (d)
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

try:
    import report_style as RS
    RS.apply()
except Exception:
    RS = None

GRID_ID_COL = "Grid_ID"
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")


def log(m): print(m, flush=True)


def centroids(ids: pd.Series):
    lat, lon = [], []
    for g in ids.astype(str):
        m = _GID.search(g)
        if not m:
            raise ValueError(f"cannot parse centroid from Grid_ID: {g!r}")
        lat.append(float(m.group(1))); lon.append(float(m.group(2)))
    return np.asarray(lat), np.asarray(lon)


def run():
    with open("config.json") as f:
        cfg = json.load(f)
    FEAT_DIR = Path(cfg["weekly_xg_dir"])

    df = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet")
    log(f"[load] {len(df):,} cell-weeks × {df[GRID_ID_COL].nunique():,} surveyed cells")

    if "spatial_block" not in df.columns:
        import cv_harness as H
        df = H.build_blocks(df)
        log("[blocks] rebuilt spatial_block via cv_harness")

    # per-cell effort
    per_cell = (df.groupby(GRID_ID_COL)
                  .agg(cell_weeks=("presence", "size"),
                       trap_events=("n_events", "sum"),
                       presence_rate=("presence", "mean"),
                       spatial_block=("spatial_block", "first"))
                  .reset_index())
    lat, lon = centroids(per_cell[GRID_ID_COL])
    per_cell["lat"], per_cell["lon"] = lat, lon

    # full state grid for context, if built
    grid_path = FEAT_DIR / "statewide_weekly_features.parquet"
    state_lat = state_lon = None
    if grid_path.exists():
        allc = pd.read_parquet(grid_path, columns=[GRID_ID_COL])[GRID_ID_COL].drop_duplicates()
        state_lat, state_lon = centroids(allc)
        cov = 100 * len(per_cell) / len(allc)
        log(f"[coverage] {len(per_cell):,} of {len(allc):,} state cells surveyed ({cov:.1f}%)")
    else:
        cov = None
        log("[coverage] statewide grid not found — panel (a) shows surveyed cells only")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(13, 8.4))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1], hspace=0.32, wspace=0.26)

    # ---- (a) trap effort map
    ax = fig.add_subplot(gs[:, 0])
    if state_lat is not None:
        ax.scatter(state_lon, state_lat, s=3, marker="s", c="0.90",
                   linewidths=0, label="Florida 5 km grid (unsurveyed)", zorder=1)
    sc = ax.scatter(per_cell.lon, per_cell.lat,
                    s=8 + 40 * per_cell.cell_weeks / per_cell.cell_weeks.max(),
                    c=per_cell.cell_weeks, cmap="viridis", alpha=0.85,
                    linewidths=0.3, edgecolors="white", zorder=2)
    ax.set_aspect(1 / np.cos(np.radians(float(per_cell.lat.mean()))))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    title = "(a) Surveillance effort by cell"
    if cov is not None:
        title += f"\n{len(per_cell):,} of {len(allc):,} state cells surveyed ({cov:.1f}%)"
    ax.set_title(title, fontsize=11, loc="left")
    fig.colorbar(sc, ax=ax, shrink=0.55, label="cell-weeks contributed")
    if state_lat is not None:
        ax.legend(loc="lower left", fontsize=7, frameon=False)

    # ---- (b) effort distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(per_cell.cell_weeks, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
    med = per_cell.cell_weeks.median()
    p90 = per_cell.cell_weeks.quantile(0.90)
    ax.axvline(med, color="#C44E52", ls="--", lw=1.2, label=f"median {med:.0f}")
    ax.axvline(p90, color="#DD8452", ls=":", lw=1.2, label=f"90th pct {p90:.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("cell-weeks per cell"); ax.set_ylabel("number of cells (log)")
    top10 = 100 * per_cell.nlargest(max(1, len(per_cell) // 10), "cell_weeks"
                                    ).cell_weeks.sum() / per_cell.cell_weeks.sum()
    ax.set_title(f"(b) Effort is highly uneven\ntop 10% of cells contribute "
                 f"{top10:.0f}% of all cell-weeks", fontsize=11, loc="left")
    ax.legend(fontsize=7, frameon=False)

    # ---- (c) + (d) per-block structure
    blk = (df.groupby("spatial_block")
             .agg(cell_weeks=("presence", "size"),
                  presence_rate=("presence", "mean"))
             .join(per_cell.groupby("spatial_block").size().rename("cells"))
             .reset_index().sort_values("spatial_block"))

    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(blk))
    ax.bar(x - 0.2, blk.cells, width=0.4, color="#55A868", label="cells")
    ax.set_ylabel("cells", color="#55A868")
    ax.tick_params(axis="y", labelcolor="#55A868")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, blk.presence_rate, width=0.4, color="#C44E52", label="presence rate")
    ax2.set_ylabel("presence rate", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(0, 1)
    ax.set_xticks(x); ax.set_xticklabels(blk.spatial_block.astype(int))
    ax.set_xlabel("spatial block")
    spread = blk.presence_rate.max() - blk.presence_rate.min()
    ax.set_title(f"(c) Blocks are unbalanced and differ in base rate\n"
                 f"presence rate spans {blk.presence_rate.min():.2f}–"
                 f"{blk.presence_rate.max():.2f} (spread {spread:.2f})",
                 fontsize=11, loc="left")

    fig.suptitle("Surveillance effort and its consequences for spatial cross-validation",
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FEAT_DIR / "sampling_effort.png"
    fig.savefig(out, dpi=170); plt.close(fig)
    log(f"[figure] wrote {out.name}")

    blk.round(3).to_csv(FEAT_DIR / "sampling_effort_summary.csv", index=False)
    log(f"[table]  wrote sampling_effort_summary.csv")

    log("\n[read] panel (b) quantifies clustering; panel (c) shows that clustering "
        "propagates into the CV blocks, so a held-out block differs from the "
        "training blocks in both size and base rate.")
    log(blk.round(3).to_string(index=False))
    return blk


if __name__ == "__main__":
    run()