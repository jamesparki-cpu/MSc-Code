from __future__ import annotations
"""
maxent_background_compare.py  —  MaxEnt target-group vs MaxEnt vanilla.

THE PROBLEM THIS SOLVES
  The two MaxEnt variants differ only in what counts as an absence: effort-based
  trap absences (target-group) against random background points (vanilla). That
  is the cleanest experimental contrast in the project -- and it is the one the
  cross-validation tables CANNOT show, because the two are scored on different
  evaluation sets (21,608 rows at prevalence 0.776 against 26,667 at 0.628).
  Comparing 0.830 to 0.910 across those sets is exactly the comparison the rest
  of the chapter warns against.

  Both figures below sidestep that, in different ways.

  FIG A  maxent_background_matched.png
         The 2018 held-out point validation is the ONE place both variants are
         scored on identical observations: the same trap points, the same n,
         the same presence/absence counts, week by week. So a direct
         head-to-head is valid here and nowhere else. Paired bars for ROC and
         BSS, with n and the absence count on the axis.

  FIG B  maxent_background_shap.png
         SHAP composition does not depend on the evaluation set at all, so this
         comparison needs no caveat. (a) paired block shares; (b) the vanilla
         minus target-group difference as a diverging bar chart ordered by
         magnitude -- which is the mechanism, not just the outcome.

WHAT THE PAIR IS FOR
  If FIG A shows target-group ahead while the CV table shows vanilla ahead, the
  CV ranking was an artefact of scoring against random background rather than
  against observed absences. FIG B then supplies the reason: the two models
  weight the predictors differently, and in the direction the design predicts.
  Neither figure alone establishes that; together they do.

INPUTS
  validation_metrics_maxent.csv       (validation_dir)   -> FIG A
  shap_grouped_all_models.csv         (shap_dir)         -> FIG B

OUTPUT -> comparison_dir
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import report_style as S

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR = Path(cfg.get("weekly_xg_dir", "."))
NOWCAST_DIR = Path(cfg.get("nowcast_dir", str(DATA_DIR)))
VALIDATION_DIR = Path(cfg.get("validation_dir", str(NOWCAST_DIR / "Validation_Results")))
SHAP_DIR = Path(cfg.get("shap_dir", str(DATA_DIR / "SHAP_Results")))
OUT_DIR = Path(cfg.get("comparison_dir", str(DATA_DIR.parent / "Comparison_Results")))

TG, VAN = "maxent_targetgroup", "maxent_vanilla"
SEASON = {6: "Winter", 20: "Spring", 30: "Summer", 32: "Summer", 43: "Autumn"}
MIN_ABS_FOR_TRUST = 10       # weeks below this are shaded and hatched

def log(m): print(m, flush=True)


def _find(name, *dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None
# ============================================================================


# ------------------------------------------------------------------- FIGURE A
def fig_matched():
    f = _find("validation_metrics_maxent.csv", VALIDATION_DIR, OUT_DIR, DATA_DIR)
    if f is None:
        log("[figA] SKIP: validation_metrics_maxent.csv not found")
        return
    d = pd.read_csv(f).drop_duplicates(["model", "week"], keep="last")
    have = [m for m in (TG, VAN) if m in set(d.model)]
    if len(have) < 2:
        log(f"[figA] SKIP: need both variants, found {have}")
        return
    weeks = sorted(d.week.unique())

    # the whole point of this figure: verify the evaluation sets really match
    ok = True
    for w in weeks:
        sub = d[d.week == w]
        for col in ("n", "n_pres", "n_abs"):
            if col in sub.columns and sub[col].nunique() > 1:
                log(f"[figA] WARNING week {w}: {col} differs between variants "
                    f"({sub[col].tolist()}) -- the points are NOT matched")
                ok = False
    log(f"[figA] evaluation sets {'match' if ok else 'DO NOT match'} across "
        f"{len(weeks)} weeks")

    low = [w for w in weeks
           if "n_abs" in d.columns
           and int(d[d.week == w].n_abs.iloc[0]) < MIN_ABS_FOR_TRUST]

    x = np.arange(len(weeks)); width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
    for ax, col in zip(axes, ["roc_auc", "bss"]):
        for w in low:
            ax.axvspan(weeks.index(w) - 0.5, weeks.index(w) + 0.5,
                       color="#b03030", alpha=0.07, zorder=0)
        for i, m in enumerate(have):
            vals = [float(d[(d.model == m) & (d.week == w)][col].iloc[0])
                    if len(d[(d.model == m) & (d.week == w)]) else np.nan
                    for w in weeks]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=S.model_label(m), color=S.MODEL_COLOURS.get(m),
                          edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW)
            for b, w in zip(bars, weeks):
                if w in low:
                    b.set_hatch("//")
            S.annotate_bars(ax, bars, vals, col, fontsize=7)
        S.add_reference_line(ax, col)
        ax.set_ylabel(S.metric_label(col) if col == "roc_auc"
                      else "BSS vs prevalence")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0.4, 1.08)
    axes[0].set_title("(a) Discrimination on identical points", fontsize=11, loc="left")
    axes[1].set_title("(b) Calibration on identical points", fontsize=11, loc="left")

    labels = []
    for w in weeks:
        r = d[d.week == w].iloc[0]
        lab = f"{SEASON.get(w, '')}\nwk {w}"
        if "n" in d.columns:
            lab += f"\nn={int(r.n)}"
        if {"n_pres", "n_abs"} <= set(d.columns):
            lab += f"\n{int(r.n_pres)} pres / {int(r.n_abs)} abs"
        labels.append(lab)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, ncol=2, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Effort-based absences vs random background, scored on the SAME "
                 f"points ({max(weeks) and 2018} held-out validation)\n"
                 f"shaded/hatched = fewer than {MIN_ABS_FOR_TRUST} observed "
                 "absences: low power, do not over-read", fontsize=11.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    S.save(fig, OUT_DIR / "maxent_background_matched.png")

    # tidy table for the text
    t = (d[d.model.isin(have)]
         .pivot(index="week", columns="model", values=["roc_auc", "bss"]))
    t.columns = [f"{a}_{b.replace('maxent_', '')}" for a, b in t.columns]
    for met in ("roc_auc", "bss"):
        a, b = f"{met}_targetgroup", f"{met}_vanilla"
        if a in t.columns and b in t.columns:
            t[f"{met}_diff"] = (t[a] - t[b]).round(3)
    t.round(3).to_csv(OUT_DIR / "maxent_background_matched.csv")
    wins = int((t.get("roc_auc_diff", pd.Series(dtype=float)) > 0).sum())
    log(f"[figA] target-group ahead on ROC in {wins}/{len(t)} weeks")
    log(t.round(3).to_string())


# ------------------------------------------------------------------- FIGURE B
def fig_shap():
    f = _find("shap_grouped_all_models.csv", SHAP_DIR, OUT_DIR, DATA_DIR)
    if f is None:
        log("[figB] SKIP: shap_grouped_all_models.csv not found")
        return
    d = pd.read_csv(f)
    if not {TG, VAN} <= set(d.model):
        log(f"[figB] SKIP: need both variants in {f.name}")
        return
    p = (d[d.model.isin([TG, VAN])]
         .pivot(index="group", columns="model", values="pct_of_total"))
    p["diff"] = p[VAN] - p[TG]
    p = p.reindex(p["diff"].abs().sort_values(ascending=False).index)
    p.round(2).to_csv(OUT_DIR / "maxent_background_shap.csv")
    log("[figB] block shares and vanilla-minus-target-group difference:")
    log(p.round(1).to_string())

    blocks = list(p.index)
    y = np.arange(len(blocks))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # (a) paired shares
    ax = axes[0]; h = 0.38
    for i, m in enumerate([TG, VAN]):
        ax.barh(y + (0.5 - i) * h, p[m], h, label=S.model_label(m),
                color=S.MODEL_COLOURS.get(m), edgecolor=S.BAR_EDGE,
                linewidth=S.BAR_EDGE_LW)
    for k, b in enumerate(blocks):
        for i, m in enumerate([TG, VAN]):
            ax.text(p[m].iloc[k] + 0.6, y[k] + (0.5 - i) * h,
                    f"{p[m].iloc[k]:.1f}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([S.block_label(b) for b in blocks], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(S.metric_label("pct_of_total"))
    ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(a) Where each variant puts its explanatory weight", fontsize=11)

    # (b) the difference, ordered by magnitude
    ax = axes[1]
    cols = [S.MODEL_COLOURS[VAN] if v > 0 else S.MODEL_COLOURS[TG]
            for v in p["diff"]]
    ax.barh(y, p["diff"], 0.66, color=cols, edgecolor=S.BAR_EDGE,
            linewidth=S.BAR_EDGE_LW)
    ax.axvline(0, color=S.GREY, lw=1.1)
    for k, v in enumerate(p["diff"]):
        ax.text(v + (0.4 if v >= 0 else -0.4), y[k], f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([S.block_label(b) for b in blocks], fontsize=9)
    ax.invert_yaxis()
    lim = float(np.abs(p["diff"]).max()) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("percentage points: vanilla \u2212 target-group")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("(b) Difference, ordered by magnitude", fontsize=11)
    ax.annotate("\u2190 target-group weights this more", (-lim * 0.96, len(blocks) - 0.4),
                fontsize=7.5, color=S.MODEL_COLOURS[TG], ha="left")
    ax.annotate("vanilla weights this more \u2192", (lim * 0.96, len(blocks) - 0.4),
                fontsize=7.5, color=S.MODEL_COLOURS[VAN], ha="right")

    fig.suptitle("What each MaxEnt variant learned. SHAP composition does not "
                 "depend on the evaluation set,\nso this comparison carries none "
                 "of the caveats attached to the cross-validation metrics.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    S.save(fig, OUT_DIR / "maxent_background_shap.png")


def run():
    S.set_thesis_style(constrained=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_matched()
    fig_shap()
    log(f"\n[done] MaxEnt background comparison -> {OUT_DIR}")


if __name__ == "__main__":
    run()