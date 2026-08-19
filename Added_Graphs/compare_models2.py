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
        r, o = H.evaluate(df, feats, make_xgb, schemes=SCHEMES, sample_weight=w,
                          calibrate=True, impute=False, model_name="xgboost")
        out.append(r); oof_all["xgboost"] = o
    r, o = H.evaluate(df, feats, make_rf, schemes=SCHEMES, sample_weight=w,
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


def make_figure(combined, path):
    """Grouped bars: ROC / PR-lift / BSS, four models x two CV schemes."""
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
            bars = ax.bar(x + (i - (len(schemes) - 1) / 2) * width, vals, width,
                          label=S.scheme_label(scheme),
                          color=S.SCHEME_COLOURS.get(scheme),
                          hatch=S.SCHEME_HATCH.get(scheme, ""),
                          edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW)
            S.annotate_bars(ax, bars, vals, col)
        S.add_reference_line(ax, col)
        ax.set_xticks(x)
        ax.set_xticklabels([S.model_label(m, wrapped=True) for m in models], fontsize=9)
        ax.set_title(S.metric_label(col), fontsize=11)
        ax.set_ylabel(S.metric_label(col))
        S.apply_ylim(ax, col)
        ax.grid(axis="y", alpha=0.3)
    # figure-level legend: an in-axes legend collides with the bars in every
    # panel once value labels are on
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, title="CV scheme", fontsize=9, ncol=len(l),
               loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Cx. nigripalpus weekly suitability — four-model comparison "
                 "(pooled out-of-fold, calibrated)", fontsize=13)
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

    make_figure(combined, OUTPUT_DIR / "model_comparison.png")
    log(f"\n[done] comparison in {OUTPUT_DIR}")
    return tidy


if __name__ == "__main__":
    run()