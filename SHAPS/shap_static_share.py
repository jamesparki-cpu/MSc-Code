from __future__ import annotations
"""
shap_static_share.py  —  is the model reading the CELL or the WEATHER?

THE ARGUMENT
  Grouped SHAP shows land cover taking 42-54% of explanatory weight, which
  suggests the models encode cell identity rather than transferable conditions.
  That figure is an average over the whole year, so it cannot distinguish

     (i)  the models rely on static site character ALL YEAR, from
     (ii) they rely on it in the off-season and switch to weather in summer.

  Reading (ii) would still permit transferable seasonal skill; reading (i) would
  not. This script separates them by computing, for every ISO week, the share of
  explanatory weight carried by STATIC features. A flat, high line is the
  numeric statement of the cell-identity result; a line that dips in the
  transmission season is a different and materially weaker finding.

THREE CATEGORIES, NOT TWO
  static     land_cover + terrain                                   the cell
  dynamic    temperature + precipitation + moisture + vegetation     the weather
  seasonal   sin_doy + cos_doy

  Seasonality is kept SEPARATE and is not folded into "dynamic". It is a
  deterministic function of ISO week -- the x-axis of this very figure -- and
  carries no weather information. Counting it as dynamic would inflate the
  dynamic share with a variable that cannot support spatial transfer, which is
  precisely the claim under test.

HOW MAGNITUDE IS AGGREGATED
  Exactly as shaps_common.grouped_importance: signed sum WITHIN each block per
  row, then absolute value. Block magnitudes are then added within a category.
  Summing raw absolutes inside a block would double-count opposing within-block
  effects; summing signed values across unrelated blocks would let temperature
  cancel land cover. Per-block-then-absolute avoids both.

  The share is computed PER ROW and then averaged within an ISO week, so a few
  rows with large total |SHAP| cannot dominate the week.

MODELS
  XGBoost and Random Forest by default (TreeSHAP, exact, fast). MaxEnt
  target-group is available but uses PermutationExplainer at ~0.9 s/row, so it
  runs on a much smaller per-week sample -- set INCLUDE_MAXENT_TG = True.

  MaxEnt-vanilla is EXCLUDED and cannot be added: it is trained against random
  background points, which carry no ISO week, so a weekly decomposition is
  undefined for it.

MODEL DEFINITIONS ARE IMPORTED, NOT RESTATED
  Fitting goes through shaps_run.assemble() and shaps_run.fit(), so these are
  the same fits as the SHAP chapter. No hyperparameters are declared here.

OUTPUT (to shap_dir)
  shap_static_share_weekly.csv    per model x ISO week: category magnitudes,
                                  shares, and n rows explained
  shap_static_share.png           (a) static share by week, one line per model
                                  (b) category composition by week, one model
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shaps_common as SC
import shaps_run as SR          # reuse ITS fit() so the models are identical
import report_style as S

# ============================== CONFIG ======================================
DATA_DIR = SR.DATA_DIR
OUT_DIR = SR.OUT_DIR

STATIC_BLOCKS = ("land_cover", "terrain")
DYNAMIC_BLOCKS = ("temperature", "precipitation", "moisture", "vegetation")
SEASONAL_BLOCKS = ("seasonality",)

# category palette: disjoint from models, schemes, SHAP blocks and regions
CATEGORY_COLOURS = {"static": "#7b5aa6", "dynamic": "#2f8fbf", "seasonal": "#d9a441"}

RUN_MODELS = ["xgboost", "random_forest"]
INCLUDE_MAXENT_TG = False       # ~0.9 s/row; roughly 6 minutes at 8 rows/week

N_PER_WEEK_TREE = 60            # rows explained per ISO week for XGB/RF
N_PER_WEEK_MAXENT = 8
MIN_ROWS_PER_WEEK = 10          # weeks below this are masked from the figure
COMPOSITION_MODEL = "xgboost"   # which model gets the stacked panel (b)
RANDOM_STATE = 42

def log(m): print(m, flush=True)
# ============================================================================


def sample_by_week(weeks: pd.Series, n_per_week: int) -> np.ndarray:
    """Stratify by ISO WEEK, not by presence.

    Every week needs enough rows to support its own mean, and thinly sampled
    weeks would otherwise disappear from the figure entirely -- which would hide
    exactly the seasonal coverage gaps documented in R1.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    pos = pd.Series(np.arange(len(weeks)), index=weeks.to_numpy())
    keep = []
    for _, idx in pos.groupby(level=0):
        arr = idx.to_numpy()
        keep.append(rng.choice(arr, size=min(len(arr), n_per_week), replace=False))
    return np.sort(np.concatenate(keep))


def category_magnitudes(sv: np.ndarray, feats) -> pd.DataFrame:
    """Per row: |signed sum within block|, then summed within category."""
    gmap = SC.assign_groups(list(feats))
    sv_df = pd.DataFrame(sv, columns=list(feats))
    block_mag = {}
    for g in sorted(set(gmap.values())):
        cols = [f for f in feats if gmap[f] == g]
        block_mag[g] = sv_df[cols].sum(axis=1).abs()

    out = pd.DataFrame(index=sv_df.index)
    for name, blocks in (("static", STATIC_BLOCKS),
                         ("dynamic", DYNAMIC_BLOCKS),
                         ("seasonal", SEASONAL_BLOCKS)):
        cols = [block_mag[b] for b in blocks if b in block_mag]
        out[name] = sum(cols) if cols else 0.0

    leftover = [g for g in block_mag
                if g not in STATIC_BLOCKS + DYNAMIC_BLOCKS + SEASONAL_BLOCKS]
    if leftover:
        log(f"[warn] blocks not assigned to any category, EXCLUDED from the "
            f"denominator: {leftover}")
    out["total"] = out[["static", "dynamic", "seasonal"]].sum(axis=1)
    return out


def run_model(variant, table, bg, feats):
    """Explain one model on a week-stratified sample; per-row category weights."""
    df = SR.assemble(variant, table, bg, feats)
    if len(df) != len(table):
        log(f"[{variant}] SKIP: assembled frame ({len(df):,}) does not align with "
            f"the model table ({len(table):,}), so ISO week cannot be attached")
        return None
    weeks = table["iso_week"].astype(int).reset_index(drop=True)

    model, medians, impute = SR.fit(variant, df, feats)
    n_per = N_PER_WEEK_MAXENT if variant.startswith("maxent") else N_PER_WEEK_TREE
    idx = sample_by_week(weeks, n_per)
    samp = df.iloc[idx]
    Xs = samp[feats].fillna(medians) if impute else samp[feats]
    log(f"[{variant}] explaining {len(Xs):,} rows across "
        f"{weeks.iloc[idx].nunique()} ISO weeks "
        f"({'Permutation' if variant.startswith('maxent') else 'TreeSHAP'})")

    sv = (SC.explain_permutation(model, Xs, feats) if variant.startswith("maxent")
          else SC.explain_tree(model, Xs))

    cat = category_magnitudes(np.asarray(sv), feats).reset_index(drop=True)
    cat["iso_week"] = weeks.iloc[idx].to_numpy()
    cat["model"] = variant
    return cat


def summarise(cat: pd.DataFrame) -> pd.DataFrame:
    """Per-row share first, then mean within week, so no single row dominates."""
    d = cat[cat.total > 0].copy()
    for c in ("static", "dynamic", "seasonal"):
        d[f"share_{c}"] = d[c] / d.total
    return (d.groupby(["model", "iso_week"])
              .agg(n=("total", "size"),
                   static_share=("share_static", "mean"),
                   dynamic_share=("share_dynamic", "mean"),
                   seasonal_share=("share_seasonal", "mean"),
                   static_mag=("static", "mean"),
                   dynamic_mag=("dynamic", "mean"),
                   seasonal_mag=("seasonal", "mean"))
              .reset_index())


def figure(g: pd.DataFrame, path):
    models = S.models_in(g.model.unique())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)

    # ---- (a) static share by week, one line per model
    ax = axes[0]
    for m in models:
        d = g[(g.model == m) & (g.n >= MIN_ROWS_PER_WEEK)].sort_values("iso_week")
        if d.empty:
            continue
        ax.plot(d.iso_week, d.static_share, marker="o", ms=3.4, lw=1.7,
                color=S.MODEL_COLOURS.get(m), label=S.model_label(m))
        mean = float(d.static_share.mean())
        ax.axhline(mean, color=S.MODEL_COLOURS.get(m), ls=":", lw=1.0, alpha=0.7)
        ax.annotate(f"mean {mean:.2f}", (52.6, mean), fontsize=7, va="center",
                    color=S.MODEL_COLOURS.get(m), annotation_clip=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("static share of |SHAP|")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(a) Share of explanatory weight carried by land cover and terrain",
                 fontsize=11, loc="left")

    # ---- (b) full composition for one model
    ax = axes[1]
    m = COMPOSITION_MODEL if COMPOSITION_MODEL in models else models[0]
    d = g[(g.model == m) & (g.n >= MIN_ROWS_PER_WEEK)].sort_values("iso_week")
    ax.stackplot(d.iso_week, d.static_share, d.dynamic_share, d.seasonal_share,
                 labels=["static (land cover + terrain)",
                         "dynamic (temperature, precipitation, moisture, vegetation)",
                         "seasonal (sin/cos day-of-year)"],
                 colors=[CATEGORY_COLOURS["static"], CATEGORY_COLOURS["dynamic"],
                         CATEGORY_COLOURS["seasonal"]], alpha=0.92)
    ax.set_ylim(0, 1); ax.set_xlim(1, 52)
    ax.set_xlabel("ISO week"); ax.set_ylabel("share of |SHAP|")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax.set_title(f"(b) Composition for {S.model_label(m)}", fontsize=11, loc="left")

    fig.suptitle("Static versus dynamic explanatory weight through the year.\n"
                 "A flat, high static share means the models read the cell rather "
                 "than the week's conditions.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    S.save(fig, path)


def run():
    S.set_thesis_style(constrained=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feats = json.load(open(DATA_DIR / "model_features.json"))["model_features"]
    table = pd.read_parquet(DATA_DIR / "weekly_model_table.parquet").reset_index(drop=True)
    bg = pd.read_parquet(SR.BG_FILE) if SR.BG_FILE.exists() else None
    log(f"[load] {len(table):,} rows | {len(feats)} features")

    variants = list(RUN_MODELS) + (["maxent_targetgroup"] if INCLUDE_MAXENT_TG else [])
    frames = []
    for v in variants:
        cat = run_model(v, table, bg, feats)
        if cat is not None:
            frames.append(cat)
    if not frames:
        log("[done] nothing explained")
        return

    g = summarise(pd.concat(frames, ignore_index=True))
    g.to_csv(OUT_DIR / "shap_static_share_weekly.csv", index=False)

    log("\n[static share] mean across weeks, and its range:")
    for m, d in g.groupby("model", sort=False):
        d = d[d.n >= MIN_ROWS_PER_WEEK]
        if d.empty:
            continue
        log(f"[static share]   {m:20s} mean {d.static_share.mean():.3f} | "
            f"min {d.static_share.min():.3f} "
            f"(wk {int(d.loc[d.static_share.idxmin(), 'iso_week'])}) | "
            f"max {d.static_share.max():.3f} "
            f"(wk {int(d.loc[d.static_share.idxmax(), 'iso_week'])}) | "
            f"range {d.static_share.max() - d.static_share.min():.3f}")

    thin = g[g.n < MIN_ROWS_PER_WEEK]
    if len(thin):
        log(f"[static share] {len(thin)} model-weeks below {MIN_ROWS_PER_WEEK} "
            f"rows, masked from the figure")

    figure(g, OUT_DIR / "shap_static_share.png")
    log("\n[read] a FLAT line in panel (a) means reliance on cell identity does not "
        "relax in the transmission season; a DIP would mean the models switch to "
        "weather when it matters, which is a materially weaker claim. Report the "
        "RANGE printed above, not just the mean.")
    return g


if __name__ == "__main__":
    run()