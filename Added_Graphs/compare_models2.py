from __future__ import annotations
"""
compare_models.py  —  four-model comparison (XGB / RF / MaxEnt-vanilla /
MaxEnt-targetgroup) on identical spatio-temporal CV.

VALIDITY: all four are scored by the same cv_harness on the same schemes
(spatiotemporal + spatial), calibrated. XGB/RF are RECOMPUTED here (fast) so
their calibrated spatial BSS exists; the slow MaxEnt runs are READ from their
saved CSVs. XGB/RF keep sqrt(n_events) weighting (as built); MaxEnt is unweighted
(as built) -- a disclosed difference, not a hidden one.

Lead metrics are ROC-AUC and BSS (fair across models); raw PR-AUC is NOT
comparable because vanilla's prevalence differs, so we report PR-AUC LIFT.

OOF FILES: cv_harness.evaluate already returns the out-of-fold vectors; they are
now saved rather than discarded, because the baseline comparison, reliability
diagrams and bootstrap confidence intervals all need row-level predictions and
none of them can be reconstructed from aggregate metrics.

OUTPUT (to comparison_dir; create the folder / add "comparison_dir" to config)
  combined_metrics.csv                 all four models x both schemes, tidy
  tree_scheme_metrics.csv              XGB + RF x all four schemes, tidy
  model_comparison.png                 grouped bars: ROC / PR-lift / BSS x scheme
  oof_weekly_<model>_<scheme>.parquet  out-of-fold predictions per model per scheme
  oof_long_trees.csv                   the same, tidy/long, for the bootstrap

STYLE: model ordering, display names, colours, decimals and axis labels come
from report_style.py. Nothing style-related is defined locally.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import cv_harness as H
import report_style as S
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
FEAT_DIR   = Path(cfg["weekly_xg_dir"])                 # weekly_model_table + features
MAXENT_DIR = Path(cfg["maxent_dir"])                    # maxent cv_metrics CSVs
OUTPUT_DIR = Path(cfg.get("comparison_dir",
                          str(FEAT_DIR.parent / "Comparison_Results")))

SCHEMES = ("spatiotemporal", "spatial")   # the schemes MaxEnt ran

# The trees are cheap, so they run every scheme. MaxEnt is not, so the
# four-model table stays restricted to SCHEMES above.
#
# WHY NOT REUSE cv_metrics.csv: train_weekly_xgboost.py writes that file using
# its OWN ensure_blocks() partition and scores the RAW pooled OOF. Both differ
# from this script (cv_harness.build_blocks, calibrate=True), which is why the
# two files disagree on spatial ROC (0.69 vs 0.41) and on BSS (raw vs
# calibrated). Producing the extra schemes HERE means every cell comes from one
# harness with one calibration setting, so the scheme table and the four-model
# table cannot drift apart.
TREE_SCHEMES = ("spatiotemporal", "temporal", "spatiotemporal_3blocks", "spatial")
RANDOM_STATE = 42

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss", tree_method="hist")
RF_PARAMS  = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                  class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

# Model ordering, display names, colours, decimal places and axis labels now
# come from report_style.py -- do not redefine them here.
# ============================================================================

def log(m): print(m, flush=True)


def save_oof(df, name, oof_dict):
    """Persist the harness's out-of-fold vectors, one file per scheme."""
    keys = [c for c in ("Grid_ID", "iso_year", "iso_week", "presence") if c in df.columns]
    for scheme, vec in oof_dict.items():
        o = df[keys].copy()
        o["oof"] = vec
        o["model"] = name
        o["scheme"] = scheme
        p = OUTPUT_DIR / f"oof_weekly_{name}_{scheme}.parquet"
        o.to_parquet(p, index=False)
        log(f"[compare] saved {p.name} ({np.isfinite(vec).mean()*100:.1f}% scored)")


def recompute_tree_models(df, feats):
    """Recompute XGB + RF through cv_harness (calibrated, weighted) so their
    numbers are directly comparable to the MaxEnt CSVs."""
    df = H.build_blocks(df)
    w = np.sqrt(df["n_events"].clip(lower=1))
    spw = (df.presence == 0).sum() / max((df.presence == 1).sum(), 1)

    def make_xgb():
        return xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    def make_rf():
        return RandomForestClassifier(**RF_PARAMS)

    out, oof_all = [], {}
    if xgb is not None:
        r, o = H.evaluate(df, feats, make_xgb, schemes=TREE_SCHEMES, sample_weight=w,
                          calibrate=True, impute=False, model_name="xgboost")
        out.append(r); oof_all["xgboost"] = o
    r, o = H.evaluate(df, feats, make_rf, schemes=TREE_SCHEMES, sample_weight=w,
                      calibrate=True, impute=True, model_name="random_forest")
    out.append(r); oof_all["random_forest"] = o
    return pd.concat(out, ignore_index=True), df, oof_all


def read_maxent_metrics():
    frames = []
    for variant in ("vanilla", "targetgroup"):
        f = MAXENT_DIR / f"cv_metrics_maxent_{variant}.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
            log(f"[compare] read {f.name}")
        else:
            log(f"[compare] WARNING missing {f.name} -- run maxent_{variant}.py first")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_bootstrap():
    """Block-bootstrap CIs, if bootstrap_uncertainty.py has already run.

    ORDER NOTE: on a first pass this file does NOT exist -- compare_models.py
    writes the OOF that bootstrap_uncertainty.py consumes, so the bootstrap can
    only run afterwards. The figure therefore degrades to no error bars rather
    than failing. Re-run compare_models.py after the bootstrap to get them.
    """
    f = OUTPUT_DIR / "bootstrap_marginal.csv"
    if not f.exists():
        log(f"[compare] {f.name} not found -- figure will have no error bars. "
            f"Run bootstrap_uncertainty.py, then re-run this script.")
        return None
    b = pd.read_csv(f)
    log(f"[compare] read {f.name} ({len(b)} model x scheme rows) for error bars")
    return b


def make_figure(combined, path, boot=None):
    """Grouped bars: ROC / PR-lift / BSS, four models x two CV schemes,
    with 95% block-bootstrap CIs where available."""
    metrics = ["roc_auc", "pr_lift", "bss"]
    models = S.models_in(combined["model"].unique())
    schemes = S.schemes_in(SCHEMES)
    x = np.arange(len(models)); width = 0.8 / max(len(schemes), 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, metrics):
        for i, scheme in enumerate(schemes):
            vals = [combined[(combined.model == m) & (combined.scheme == scheme)][col].values
                    for m in models]
            vals = [float(v[0]) if len(v) else np.nan for v in vals]

            err = None
            if boot is not None and {f"{col}_lo", f"{col}_hi"} <= set(boot.columns):
                lo, hi = [], []
                for m in models:
                    r = boot[(boot.model == m) & (boot.scheme == scheme)]
                    lo.append(float(r[f"{col}_lo"].iloc[0]) if len(r) else np.nan)
                    hi.append(float(r[f"{col}_hi"].iloc[0]) if len(r) else np.nan)
                # asymmetric: percentile intervals are not symmetric about the
                # point estimate, so lower and upper must be passed separately
                err = np.abs(np.vstack([np.asarray(vals, float) - np.asarray(lo, float),
                                        np.asarray(hi, float) - np.asarray(vals, float)]))

            bars = ax.bar(x + (i - (len(schemes) - 1) / 2) * width, vals, width,
                          label=S.scheme_label(scheme),
                          color=S.SCHEME_COLOURS.get(scheme),
                          hatch=S.SCHEME_HATCH.get(scheme, ""),
                          edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW,
                          yerr=err, capsize=2.5,
                          error_kw=dict(lw=0.9, ecolor="0.2", zorder=5))
            # value labels are dropped once error bars are present: the two
            # collide, and the interval is the more informative annotation
            if err is None:
                S.annotate_bars(ax, bars, vals, col)
        S.add_reference_line(ax, col)
        ax.set_xticks(x)
        ax.set_xticklabels([S.model_label(m, wrapped=True) for m in models], fontsize=9)
        ax.set_ylabel(S.metric_label(col))
        S.apply_ylim(ax, col)
        if boot is not None and col == "roc_auc":
            ax.set_ylim(0.15, 1.05)      # room for the wide spatial intervals
        ax.grid(axis="y", alpha=0.3)
    # figure-level legend: an in-axes legend collides with the bars in every
    # panel once value labels are on
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, title="CV scheme", fontsize=9, ncol=len(l),
               loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.01))
    sub = ("bars = 95% block-bootstrap CI" if boot is not None
           else "no bootstrap CIs available \u2014 run bootstrap_uncertainty.py")
    fig.suptitle("Cx. nigripalpus weekly suitability — four-model comparison "
                 f"(pooled out-of-fold, calibrated; {sub})", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    S.save(fig, path)


def run():
    S.set_thesis_style(constrained=False)   # these figures use tight_layout(rect=...)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feats = json.load(open(FEAT_DIR / "model_features.json"))["model_features"]
    df = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet")

    log("[compare] recomputing XGB + RF (fast) through the shared harness...")
    tree, blocked, oof_all = recompute_tree_models(df, feats)
    # persist calibrated OOF in long form for the bootstrap
    keys = blocked[[H.GRID_ID_COL, "iso_year", "spatial_block", "presence"]].reset_index(drop=True)
    rows = []
    for model, schemes in oof_all.items():
        for scheme, p in schemes.items():
            r = keys.copy()
            r["model"], r["scheme"], r["p"] = model, scheme, p
            rows.append(r)
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT_DIR / "oof_long_trees.csv", index=False)
    log(f"[compare] wrote oof_long_trees.csv ({sum(len(r) for r in rows):,} rows)")
    
    # scheme table: trees only, every scheme. This is the source for the
    # bottom row of the R2 synthesis figure and for Table R2.2. It replaces
    # cv_metrics.csv, which is blocked and calibrated differently.
    tree_cols = [c for c in ["model", "scheme", "n", "prevalence", "pr_auc",
                             "pr_baseline", "pr_lift", "roc_auc", "brier", "bss"]
                 if c in tree.columns]
    scheme_tbl = tree[tree_cols].copy()
    scheme_tbl["model"] = pd.Categorical(scheme_tbl["model"],
                                         S.models_in(scheme_tbl["model"]), ordered=True)
    scheme_tbl["scheme"] = pd.Categorical(scheme_tbl["scheme"],
                                          S.schemes_in(scheme_tbl["scheme"]), ordered=True)
    scheme_tbl = scheme_tbl.sort_values(["model", "scheme"])
    scheme_tbl.to_csv(OUTPUT_DIR / "tree_scheme_metrics.csv", index=False)
    log(f"[compare] wrote tree_scheme_metrics.csv "
        f"({len(scheme_tbl)} rows, {len(TREE_SCHEMES)} schemes)\n")
    log(scheme_tbl.to_string(index=False))
    log("")

    maxent = read_maxent_metrics()
    combined = pd.concat([tree, maxent], ignore_index=True)
    combined = combined[combined["scheme"].isin(SCHEMES)]

    tidy = combined[["model", "scheme", "prevalence", "roc_auc", "pr_auc",
                     "pr_baseline", "pr_lift", "bss"]].copy()
    # canonical row order, so the CSV and every figure agree
    tidy["model"] = pd.Categorical(tidy["model"], S.models_in(tidy["model"]), ordered=True)
    tidy["scheme"] = pd.Categorical(tidy["scheme"], S.schemes_in(tidy["scheme"]), ordered=True)
    tidy = tidy.sort_values(["scheme", "model"])
    tidy.to_csv(OUTPUT_DIR / "combined_metrics.csv", index=False)
    log(f"[compare] wrote combined_metrics.csv\n")
    log(tidy.to_string(index=False))

    make_figure(combined, OUTPUT_DIR / "model_comparison.png", boot=read_bootstrap())
    log(f"\n[done] comparison in {OUTPUT_DIR}")
    return tidy


if __name__ == "__main__":
    run()