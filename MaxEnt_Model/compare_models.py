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

OUTPUT (to comparison_dir; create the folder / add "comparison_dir" to config)
  combined_metrics.csv        all four models x both schemes, tidy
  model_comparison.png        grouped bars: ROC / PR-lift / BSS x scheme
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

MODEL_ORDER = ["xgboost", "random_forest", "maxent_vanilla", "maxent_targetgroup"]
PRETTY = {"xgboost": "XGBoost", "random_forest": "Random Forest",
          "maxent_vanilla": "MaxEnt\n(vanilla)", "maxent_targetgroup": "MaxEnt\n(target-group)"}
# ============================================================================

def log(m): print(m, flush=True)


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

    out = []
    if xgb is not None:
        r, _ = H.evaluate(df, feats, make_xgb, schemes=SCHEMES, sample_weight=w,
                          calibrate=True, impute=False, model_name="xgboost")
        out.append(r)
    r, _ = H.evaluate(df, feats, make_rf, schemes=SCHEMES, sample_weight=w,
                      calibrate=True, impute=True, model_name="random_forest")
    out.append(r)
    return pd.concat(out, ignore_index=True)


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
    metrics = [("roc_auc", "ROC-AUC", 0.5, (0.3, 1.0)),
               ("pr_lift", "PR-AUC lift over baseline", 0.0, None),
               ("bss", "Brier Skill Score", 0.0, None)]
    models = [m for m in MODEL_ORDER if m in combined["model"].unique()]
    x = np.arange(len(models)); width = 0.38
    colors = {"spatiotemporal": "#2166ac", "spatial": "#d7301f"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title, ref, ylim) in zip(axes, metrics):
        for i, scheme in enumerate(SCHEMES):
            vals = [combined[(combined.model == m) & (combined.scheme == scheme)][col].values
                    for m in models]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + (i - 0.5) * width, vals, width, label=scheme,
                   color=colors.get(scheme, None), edgecolor="black", linewidth=0.5)
        ax.axhline(ref, color="grey", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([PRETTY.get(m, m) for m in models], fontsize=9)
        ax.set_title(title, fontsize=11); ax.set_ylabel(col)
        if ylim: ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(title="CV scheme", fontsize=9)
    fig.suptitle("Cx. nigripalpus weekly suitability — four-model comparison "
                 "(pooled out-of-fold, calibrated)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150); plt.close(fig)
    log(f"[compare] wrote {path}")


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feats = json.load(open(FEAT_DIR / "model_features.json"))["model_features"]
    df = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet")

    log("[compare] recomputing XGB + RF (fast) through the shared harness...")
    tree = recompute_tree_models(df, feats)
    maxent = read_maxent_metrics()
    combined = pd.concat([tree, maxent], ignore_index=True)
    combined = combined[combined["scheme"].isin(SCHEMES)]

    tidy = combined[["model", "scheme", "prevalence", "roc_auc", "pr_auc",
                     "pr_baseline", "pr_lift", "bss"]].sort_values(["scheme", "model"])
    tidy.to_csv(OUTPUT_DIR / "combined_metrics.csv", index=False)
    log(f"[compare] wrote combined_metrics.csv\n")
    log(tidy.to_string(index=False))

    make_figure(combined, OUTPUT_DIR / "model_comparison.png")
    log(f"\n[done] comparison in {OUTPUT_DIR}")
    return tidy


if __name__ == "__main__":
    run()