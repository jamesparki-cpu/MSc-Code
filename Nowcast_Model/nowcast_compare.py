from __future__ import annotations
"""
nowcast_compare.py  —  FIGURES for the forward-chaining nowcast (reads the driver
outputs; makes no models). Two figures + a tidy headline table.

  1. nowcast_progression.png — ROC & BSS vs test year, one line per model. Shows
     performance improving as the training window expands (the nowcast story).
  2. nowcast_headline.png    — grouped bars of ROC / PR-lift / BSS for the final-
     year forward forecast (the deployment-realistic number), per model.
  3. nowcast_headline_table.csv — the headline row per model, tidy.

Reads nowcast_perfold_all.csv + nowcast_summary.csv from nowcast_results_dir.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("config.json") as f:
    cfg = json.load(f)
NOWCAST_DIR = Path(cfg["nowcast_dir"])
RES = Path(cfg.get("nowcast_results_dir", str(NOWCAST_DIR / "Nowcast_Results")))

MODEL_ORDER = ["xgboost", "random_forest", "maxent_targetgroup", "maxent_vanilla"]
PRETTY = {"xgboost": "XGBoost", "random_forest": "Random Forest",
          "maxent_targetgroup": "MaxEnt (target-group)", "maxent_vanilla": "MaxEnt (vanilla)"}
COLORS = {"xgboost": "#2166ac", "random_forest": "#238b45",
          "maxent_targetgroup": "#d7301f", "maxent_vanilla": "#6a51a3"}

def log(m): print(m, flush=True)


def progression_figure(pf, path):
    models = [m for m in MODEL_ORDER if m in pf["model"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (col, title, ref) in zip(axes, [("roc_auc", "ROC-AUC", 0.5),
                                            ("bss", "Brier Skill Score", 0.0)]):
        for m in models:
            d = pf[pf.model == m].sort_values("test_year")
            ax.plot(d["test_year"], d[col], marker="o", label=PRETTY.get(m, m),
                    color=COLORS.get(m), linewidth=2)
        ax.axhline(ref, color="grey", ls="--", lw=1)
        ax.set_xlabel("test year (trained on all prior years)")
        ax.set_ylabel(col); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("Forward-chaining nowcast — performance as training window expands", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(path, dpi=150); plt.close(fig)
    log(f"[compare] wrote {path}")


def headline_figure(summary, path):
    head = summary[summary["fold"] == "headline_final_year"].copy()
    models = [m for m in MODEL_ORDER if m in head["model"].unique()]
    metrics = [("roc_auc", "ROC-AUC", 0.5, (0.3, 1.0)),
               ("pr_lift", "PR-AUC lift", 0.0, None),
               ("bss", "Brier Skill Score", 0.0, None)]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title, ref, ylim) in zip(axes, metrics):
        vals = [head[head.model == m][col].values for m in models]
        vals = [float(v[0]) if len(v) else np.nan for v in vals]
        ax.bar(x, vals, color=[COLORS.get(m) for m in models], edgecolor="black", linewidth=0.5)
        ax.axhline(ref, color="grey", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([PRETTY.get(m, m).replace(" (", "\n(") for m in models], fontsize=9)
        ax.set_title(title); ax.set_ylabel(col)
        if ylim: ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
    yr = int(head["test_year"].iloc[0]) if "test_year" in head and len(head) else "final"
    fig.suptitle(f"Nowcast headline — forward forecast of {yr} (trained on all prior years)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(path, dpi=150); plt.close(fig)
    log(f"[compare] wrote {path}")


def run():
    RES.mkdir(parents=True, exist_ok=True)
    pf = pd.read_csv(RES / "nowcast_perfold_all.csv")
    summary = pd.read_csv(RES / "nowcast_summary.csv")

    progression_figure(pf, RES / "nowcast_progression.png")
    headline_figure(summary, RES / "nowcast_headline.png")

    head = summary[summary["fold"] == "headline_final_year"][
        ["model", "test_year", "prevalence", "roc_auc", "pr_lift", "bss"]]
    head.to_csv(RES / "nowcast_headline_table.csv", index=False)
    log("\n[headline] forward forecast of final year:")
    log(head.to_string(index=False))
    log(f"\n[done] figures + table in {RES}")


if __name__ == "__main__":
    run()