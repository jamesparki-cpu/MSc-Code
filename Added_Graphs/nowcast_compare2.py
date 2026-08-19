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

STYLE: ordering, display names, colours, decimals and axis labels come from
report_style.py. Nothing style-related is defined locally.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import report_style as S

with open("config.json") as f:
    cfg = json.load(f)
NOWCAST_DIR = Path(cfg["nowcast_dir"])
RES = Path(cfg.get("nowcast_results_dir", str(NOWCAST_DIR / "Nowcast_Results")))

def log(m): print(m, flush=True)


def progression_figure(pf, path):
    models = S.models_in(pf["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col in zip(axes, ["roc_auc", "bss"]):
        for m in models:
            d = pf[pf.model == m].sort_values("test_year")
            ax.plot(d["test_year"], d[col], marker="o", label=S.model_label(m),
                    color=S.MODEL_COLOURS.get(m), linewidth=2)
        S.add_reference_line(ax, col)
        ax.set_xlabel("test year (trained on all prior years)")
        ax.set_ylabel(S.metric_label(col)); ax.set_title(S.metric_label(col))
        ax.grid(alpha=0.3)
        # fold n varies 2,454-4,667 and confounds the trend -- state it
        ns = pf[pf.model == models[0]].sort_values("test_year")
        for _, r in ns.iterrows():
            ax.annotate(f"n={S.fmt(r['n'], 'n')}", (r["test_year"], ax.get_ylim()[0]),
                        ha="center", va="bottom", fontsize=7, color=S.GREY)
    axes[0].legend(fontsize=9)
    fig.suptitle("Forward-chaining nowcast — performance as training window expands", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    S.save(fig, path)


def headline_figure(summary, path):
    head = summary[summary["fold"] == "headline_final_year"].copy()
    models = S.models_in(head["model"].unique())
    metrics = ["roc_auc", "pr_lift", "bss"]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, metrics):
        vals = [head[head.model == m][col].values for m in models]
        vals = [float(v[0]) if len(v) else np.nan for v in vals]
        bars = ax.bar(x, vals, color=[S.MODEL_COLOURS.get(m) for m in models],
                      edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW)
        S.add_reference_line(ax, col)
        S.apply_ylim(ax, col)
        S.annotate_bars(ax, bars, vals, col)
        ax.set_xticks(x)
        ax.set_xticklabels([S.model_label(m, wrapped=True) for m in models], fontsize=9)
        ax.set_title(S.metric_label(col)); ax.set_ylabel(S.metric_label(col))
        ax.grid(axis="y", alpha=0.3)
    yr = int(head["test_year"].iloc[0]) if "test_year" in head and len(head) else "final"
    fig.suptitle(f"Nowcast headline — forward forecast of {yr} (trained on all prior years)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    S.save(fig, path)


def run():
    S.set_thesis_style(constrained=False)   # these figures use tight_layout(rect=...)
    RES.mkdir(parents=True, exist_ok=True)
    pf = pd.read_csv(RES / "nowcast_perfold_all.csv")
    summary = pd.read_csv(RES / "nowcast_summary.csv")

    progression_figure(pf, RES / "nowcast_progression.png")
    headline_figure(summary, RES / "nowcast_headline.png")

    head = summary[summary["fold"] == "headline_final_year"][
        ["model", "test_year", "prevalence", "roc_auc", "pr_lift", "bss"]].copy()
    head["model"] = pd.Categorical(head["model"], S.models_in(head["model"]), ordered=True)
    head = head.sort_values("model")
    head.to_csv(RES / "nowcast_headline_table.csv", index=False)
    log("\n[headline] forward forecast of final year:")
    log(head.to_string(index=False))
    log(f"\n[done] figures + table in {RES}")


if __name__ == "__main__":
    run()