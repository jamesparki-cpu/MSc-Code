from __future__ import annotations
"""
nowcast_compare.py  —  FIGURES for the forward-chaining nowcast (reads the driver
outputs; makes no models). Two figures + a tidy headline table.

  1. nowcast_progression.png — ROC & BSS vs test year, one line per model. Shows
     performance improving as the training window expands (the nowcast story).
  2. nowcast_headline.png    — grouped bars of ROC / PR-lift / BSS for the final-
     year forward forecast (the deployment-realistic number), per model.
  3. nowcast_headline_table.csv — the headline row per model, tidy.

Both figures now carry the three REFERENCE FORECASTS alongside the models,
read from baseline_metrics_nowcast.csv: climatology, persistence and the
constant prevalence forecast. Without them a BSS of +0.258 has no stated bar to
clear, and the reader cannot tell whether 0.839 is good.

  Baselines are drawn in GREY, so they read as the bar to beat rather than as a
  fifth and sixth model.
  Prevalence has NO ROC bar: a constant forecast cannot rank, so its ROC is
  undefined rather than zero, and the CSV carries NaN.
  Anything scored on a DIFFERENT ROW SET from the models is HATCHED (bars) or
  DASHED (lines) with its n stated -- persistence covers 1,893 of the 2,454
  rows in 2018, and MaxEnt-vanilla is on its own evaluation set throughout.

Reads nowcast_perfold_all.csv + nowcast_summary.csv from nowcast_results_dir,
and baseline_metrics_nowcast.csv from baselines_dir.

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
BASE_DIR = Path(cfg.get("baselines_dir", str(RES / "Baselines")))

SHOW_BASELINES = list(S.BASELINE_ORDER)     # None to omit them entirely


def load_baselines():
    """Per-fold reference forecasts; empty frame if run_baselines.py has not run."""
    f = BASE_DIR / "baseline_metrics_nowcast.csv"
    if not f.exists():
        log(f"[compare] {f.name} not found -- figures will show models only")
        return pd.DataFrame()
    b = pd.read_csv(f)
    b = b[b.baseline.isin(SHOW_BASELINES or [])]
    b = b[b.test_year.astype(str) != "POOLED"].copy()
    b["test_year"] = b.test_year.astype(int)
    # the models' BSS is against the constant prevalence forecast, so the
    # baselines must be read on the same reference or the panel is incoherent
    b["bss"] = b["bss_vs_prevalence"]
    log(f"[compare] read {f.name}: {sorted(b.baseline.unique())}")
    return b

def log(m): print(m, flush=True)


def progression_figure(pf, base, path):
    models = S.models_in(pf["model"].unique())
    n_ref = pf[pf.model == models[0]].set_index("test_year").n.to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, col in zip(axes, ["roc_auc", "bss"]):
        for m in models:
            d = pf[pf.model == m].sort_values("test_year")
            # dashed where the evaluation set differs from the reference model
            mismatch = any(n_ref.get(y) != n for y, n in zip(d.test_year, d.n))
            ax.plot(d["test_year"], d[col], marker="o", ms=5,
                    label=S.model_label(m), color=S.MODEL_COLOURS.get(m),
                    lw=2, ls="--" if mismatch else "-")
        for b in (S.baselines_in(base.baseline.unique()) if len(base) else []):
            d = base[base.baseline == b].sort_values("test_year")
            if d[col].isna().all():
                continue                     # prevalence has no ROC
            mismatch = any(n_ref.get(y) != n for y, n in zip(d.test_year, d.n_scored))
            ax.plot(d["test_year"], d[col], marker="s", ms=4.5,
                    label=S.baseline_label(b), color=S.BASELINE_COLOURS.get(b),
                    lw=1.6, ls="--" if mismatch else ":")
        S.add_reference_line(ax, col)
        ax.set_xlabel("test year (trained on all prior years)")
        ax.set_ylabel(S.metric_label(col) if col == "roc_auc"
                      else "BSS vs prevalence")
        ax.set_title(S.metric_label(col) if col == "roc_auc"
                     else "Brier Skill Score (vs prevalence)")
        ax.grid(alpha=0.3)
        # integer year ticks: the default put labels at 2014.5, 2015.5, ...
        yrs = sorted(pf.test_year.unique())
        ax.set_xticks(yrs); ax.set_xticklabels([int(y) for y in yrs])
        for y in yrs:
            ax.annotate(f"n={S.fmt(n_ref.get(y), 'n')}", (y, ax.get_ylim()[0]),
                        ha="center", va="bottom", fontsize=7, color=S.GREY)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=min(len(l), 4), loc="lower center",
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Forward-chaining nowcast — models and reference forecasts as "
                 "the training window expands\n"
                 "grey = reference forecasts; dashed = scored on a different row "
                 "set from the trap-restricted models (n stated per fold)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    S.save(fig, path)


def headline_figure(summary, base, path):
    head = summary[summary["fold"] == "headline_final_year"].copy()
    models = S.models_in(head["model"].unique())
    yr = int(head["test_year"].iloc[0]) if "test_year" in head and len(head) else None
    hb = base[base.test_year == yr] if (len(base) and yr) else pd.DataFrame()
    refs = S.baselines_in(hb.baseline.unique()) if len(hb) else []

    # the row set the models are scored on; anything else gets hatched
    n_ref = int(head[head.model == models[0]].n.iloc[0]) if "n" in head.columns else None

    entries = ([(m, "model") for m in models] + [(b, "baseline") for b in refs])
    metrics = ["roc_auc", "pr_lift", "bss"]
    x = np.arange(len(entries))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, col in zip(axes, metrics):
        vals, cols, hatch, ns = [], [], [], []
        for key, kind in entries:
            if kind == "model":
                r = head[head.model == key]
                vals.append(float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0])
                            else np.nan)
                cols.append(S.MODEL_COLOURS.get(key))
                ns.append(int(r.n.iloc[0]) if "n" in r.columns and len(r) else None)
            else:
                r = hb[hb.baseline == key]
                vals.append(float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0])
                            else np.nan)
                cols.append(S.BASELINE_COLOURS.get(key))
                ns.append(int(r.n_scored.iloc[0]) if len(r) else None)
            hatch.append("//" if (n_ref and ns[-1] and ns[-1] != n_ref) else "")
        bars = ax.bar(x, vals, 0.72, color=cols,
                      edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW)
        for b, h in zip(bars, hatch):
            if h:
                b.set_hatch(h)
        S.add_reference_line(ax, col)
        S.apply_ylim(ax, col)
        S.annotate_bars(ax, bars, vals, col, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [S.model_label(k, wrapped=True) if kd == "model"
             else S.baseline_label(k).replace(" (", "\n(") for k, kd in entries],
            fontsize=7.5, rotation=30, ha="right")
        ax.set_ylabel(S.metric_label(col) if col != "bss" else "BSS vs prevalence")
        ax.set_title(S.metric_label(col) if col != "bss"
                     else "Brier Skill Score (vs prevalence)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if col == "roc_auc":
            ax.annotate("prevalence: ROC undefined\n(constant forecast)",
                        (x[-1], 0.32), ha="center", fontsize=6.5, color=S.GREY)

    fig.suptitle(f"Nowcast headline — forward forecast of {yr} (trained on all "
                 f"prior years)\ngrey = reference forecasts; hatched = scored on "
                 f"a different row set from the {S.fmt(n_ref, 'n')} rows the "
                 f"trap-restricted models saw", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    S.save(fig, path)


def run():
    S.set_thesis_style(constrained=False)   # these figures use tight_layout(rect=...)
    RES.mkdir(parents=True, exist_ok=True)
    pf = pd.read_csv(RES / "nowcast_perfold_all.csv")
    summary = pd.read_csv(RES / "nowcast_summary.csv")

    base = load_baselines()
    progression_figure(pf, base, RES / "nowcast_progression.png")
    headline_figure(summary, base, RES / "nowcast_headline.png")

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