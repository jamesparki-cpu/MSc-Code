from __future__ import annotations
"""
collinearity_check.py  —  characterise the correlation structure of the 31 predictors.

WHY THIS EXISTS
  SHAP grouping was adopted because lagged climate features are correlated by
  construction. That mitigation is only defensible if the underlying condition is
  actually measured -- otherwise the grouping is an assertion. This script
  quantifies it.

WHAT IT DOES NOT DO
  It does not drop features. The lag structure is mechanistically motivated
  (development time, gonotrophic cycle), so removing prcp_sum_28d because it
  correlates with prcp_sum_14d would discard the biological rationale that
  justified the feature set. Tree ensembles are robust to collinearity for
  PREDICTION; it is ATTRIBUTION that degrades, which is what grouping addresses.

OUTPUTS (to weekly_xg_dir)
  predictor_correlation.csv        full 31 x 31 Spearman matrix
  predictor_correlation_summary.csv  max |r| within and between SHAP blocks
  predictor_correlation.png        heatmap ordered by SHAP block
  predictor_dendrogram.png         hierarchical clustering of |r| distance

STYLE: typography, block display names and block colours come from
report_style.py. Feature ORDERING still comes from shaps_common.GROUP_ORDER,
because it must match the SHAP figures exactly; run() warns if the two block
vocabularies disagree.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import shap_common2 as SC   # reuse the SAME block definitions used for SHAP
import report_style as S    # figure conventions (typography, block labels/colours)


def log(m): print(m, flush=True)


def run():
    S.set_thesis_style(constrained=False)   # these figures use tight_layout()
    # ordering is shaps_common's job (it must match the SHAP figures); this only
    # checks that report_style knows about every block shaps_common emits
    unknown = [b for b in SC.GROUP_ORDER if b not in S.BLOCK_ORDER]
    if unknown:
        log(f"[warn] blocks in shaps_common.GROUP_ORDER missing from "
            f"report_style.BLOCK_ORDER: {unknown} -- add them there")

    with open("config.json") as f:
        cfg = json.load(f)
    FEAT_DIR = Path(cfg["weekly_xg_dir"])

    feats = json.load(open(FEAT_DIR / "model_features.json"))["model_features"]
    df = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet", columns=feats)
    log(f"[load] {len(df):,} rows x {len(feats)} predictors")

    # Spearman: monotone but not necessarily linear association, appropriate for
    # skewed rainfall sums and bounded land-cover fractions alike
    corr = df[feats].corr(method="spearman")
    corr.to_csv(FEAT_DIR / "predictor_correlation.csv")

    # order columns by SHAP block so the block structure is visible in the figure
    gmap = SC.assign_groups(feats)
    order, labels = [], []
    for block in SC.GROUP_ORDER:
        present = [f for f in feats if gmap.get(f) == block]
        order += present
        labels += [block] * len(present)
    leftover = [f for f in feats if f not in order]
    if leftover:
        log(f"[warn] {len(leftover)} predictors outside GROUP_ORDER: {leftover}")
        order += leftover
        labels += [gmap.get(f, "other") for f in leftover]

    C = corr.loc[order, order]
    lab = pd.Series(labels, index=order)

    # ---- summary: max |r| within each block, and worst cross-block pair ----
    rows = []
    A = C.abs().to_numpy()
    np.fill_diagonal(A, np.nan)
    for block in lab.unique():
        idx = np.flatnonzero((lab == block).to_numpy())
        if len(idx) > 1:
            sub = A[np.ix_(idx, idx)]
            i, j = np.unravel_index(np.nanargmax(sub), sub.shape)
            rows.append(dict(scope="within", block=block, n_features=len(idx),
                             max_abs_r=round(float(np.nanmax(sub)), 3),
                             pair=f"{order[idx[i]]} ~ {order[idx[j]]}"))
        else:
            rows.append(dict(scope="within", block=block, n_features=len(idx),
                             max_abs_r=np.nan, pair="n/a (single feature)"))

    other = A.copy()
    for block in lab.unique():
        idx = np.flatnonzero((lab == block).to_numpy())
        other[np.ix_(idx, idx)] = np.nan
    i, j = np.unravel_index(np.nanargmax(other), other.shape)
    rows.append(dict(scope="between", block="ALL", n_features=len(order),
                     max_abs_r=round(float(np.nanmax(other)), 3),
                     pair=f"{order[i]} ~ {order[j]}"))

    summary = pd.DataFrame(rows)
    summary.to_csv(FEAT_DIR / "predictor_correlation_summary.csv", index=False)
    log("\n[summary] maximum |Spearman r|:")
    log(summary.to_string(index=False))

    # ---- heatmap ----
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(C.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6)
    # block boundaries
    bounds, seen = [], None
    for k, b in enumerate(labels):
        if b != seen:
            bounds.append(k); seen = b
    for b in bounds[1:]:
        ax.axhline(b - 0.5, color="k", lw=0.8)
        ax.axvline(b - 0.5, color="k", lw=0.8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman r")
    ax.set_title("Predictor correlation, ordered by SHAP block", fontsize=11)
    # block bands down the left edge, in the shared block colours
    for k0, k1, b in zip(bounds, bounds[1:] + [len(order)],
                         [labels[i] for i in bounds]):
        ax.annotate(S.block_label(b), xy=(-0.055, 1 - (k0 + k1) / 2 / len(order)),
                    xycoords="axes fraction", ha="right", va="center",
                    fontsize=7, color="0.25",
                    bbox=dict(fc=S.BLOCK_COLOURS.get(b, "#eee"), ec="none", pad=1.5))
    fig.tight_layout()
    S.save(fig, FEAT_DIR / "predictor_correlation.png")

    # ---- dendrogram: does the empirical structure recover the blocks? ----
    D = 1 - corr.abs()
    np.fill_diagonal(D.values, 0.0)
    Z = linkage(squareform(D.to_numpy(), checks=False), method="average")
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, labels=corr.columns.tolist(), leaf_rotation=90,
               leaf_font_size=6, ax=ax)
    ax.set_ylabel("1 - |r|")
    ax.set_title("Hierarchical clustering of predictors (average linkage)", fontsize=11)
    # colour each leaf label by its SHAP block, so the reader can see at a glance
    # which blocks the empirical structure actually recovers
    for t in ax.get_xticklabels():
        b = gmap.get(t.get_text(), "other")
        t.set_bbox(dict(fc=S.BLOCK_COLOURS.get(b, "#eee"), ec="none", pad=1.0))
    fig.tight_layout()
    S.save(fig, FEAT_DIR / "predictor_dendrogram.png")

    log(f"\n[done] wrote correlation matrix, summary, heatmap and dendrogram to {FEAT_DIR}")
    log("[read] high within-block |r| is EXPECTED and is what motivates grouping; "
        "high BETWEEN-block |r| is the number to report honestly, since it bounds "
        "how cleanly the blocks separate.")
    return summary


if __name__ == "__main__":
    run()