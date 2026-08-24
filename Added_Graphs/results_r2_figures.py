from __future__ import annotations
"""
results_r2_figures.py  —  the three R2 figures. Reads persisted outputs only;
fits nothing. Each figure skips independently (with a message) if its inputs are
missing, so this can be run before every stage is finished.

  FIG 1  r2_synthesis.png          THE chapter figure. 2x3.
         Top row    ROC / PR-lift / BSS, four models, spatiotemporal vs spatial,
                    with block-bootstrap 95% CIs.
         Bottom row the same three metrics for XGB and RF across ALL FOUR
                    schemes, so the temporal / spatiotemporal / 3-block bars
                    visibly collapse onto one another while the spatial bar
                    falls off. That contrast is the result; currently the reader
                    has to assemble it from separate figures.

  FIG 2  r2_scheme_slopegraph.png  One panel, ROC-AUC, spatiotemporal -> spatial,
         one line per model. Four steep near-parallel descents say "this is a
         property of the data, not of any one algorithm" more directly than bars.

  FIG 3  r2_roc_pr_curves.png      Pooled out-of-fold ROC and PR curves per
         scheme, four models each, from oof_long.csv. The AUCs are reported
         throughout the chapter but never shown as curves.

--------------------------------------------------------------------------
CONSISTENCY GUARD (read this before trusting FIG 1)
  The top row comes from combined_metrics.csv; the bottom row from
  tree_scheme_metrics.csv (compare_models.py, TREE_SCHEMES) -- the only source
  where all four schemes are blocked and calibrated the same way as the
  four-model table.

  DO NOT substitute cv_metrics.csv. That file is written by
  train_weekly_xgboost.py using its own ensure_blocks() partition and scoring
  the RAW pooled OOF, so it disagrees with combined_metrics.csv on two counts:
  spatial ROC (~0.69 vs 0.405, different KMeans partition) and BSS (-0.025 vs
  +0.282, uncalibrated vs isotonic). It is accepted here only as a fallback,
  and the guard below will reject it.

  Plotting both rows from disagreeing sources would put two different values for
  the SAME model-scheme-metric in one figure. So the overlapping cells are
  compared before drawing, and if they disagree beyond TOLERANCE the bottom row
  is REFUSED rather than drawn wrong. Regenerate cv_metrics.csv from the same
  harness and calibration settings as combined_metrics.csv, then re-run.
--------------------------------------------------------------------------

OUTPUT -> comparison_dir
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

import report_style as S

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
FEAT_DIR = Path(cfg["weekly_xg_dir"])
MAXENT_DIR = Path(cfg.get("maxent_dir", str(FEAT_DIR)))
OUT_DIR = Path(cfg.get("comparison_dir", str(FEAT_DIR.parent / "Comparison_Results")))

METRICS = ["roc_auc", "pr_lift", "bss"]
TOP_SCHEMES = ["spatiotemporal", "spatial"]          # the four-model comparison
BOTTOM_SCHEMES = ["spatiotemporal", "temporal", "spatiotemporal_3blocks", "spatial"]
BOTTOM_MODELS = ["xgboost", "random_forest"]         # only these ran every scheme

TOLERANCE = 0.02        # max allowed |combined - cv_metrics| on shared cells
STRICT = True           # True: refuse the bottom row on mismatch. False: draw + warn

def log(m): print(m, flush=True)


def _find(name, *dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None
# ============================================================================


# ------------------------------------------------------------------ FIGURE 1
def check_consistency(combined: pd.DataFrame, cvm: pd.DataFrame):
    """Compare the cells both files claim to describe. Returns (ok, report)."""
    rows = []
    for model in BOTTOM_MODELS:
        for scheme in TOP_SCHEMES:                    # the overlap
            for met in METRICS:
                a = combined[(combined.model == model) & (combined.scheme == scheme)][met]
                b = cvm[(cvm.model == model) & (cvm.scheme == scheme)][met]
                if not len(a) or not len(b):
                    continue
                a, b = float(a.iloc[0]), float(b.iloc[0])
                rows.append(dict(model=model, scheme=scheme, metric=met,
                                 combined=a, cv_metrics=b, diff=abs(a - b)))
    rep = pd.DataFrame(rows)
    if rep.empty:
        return False, rep
    return bool((rep["diff"] <= TOLERANCE).all()), rep


def fig_synthesis():
    f_comb = _find("combined_metrics.csv", OUT_DIR, FEAT_DIR)
    if f_comb is None:
        log("[fig1] SKIP: combined_metrics.csv not found -- run compare_models.py")
        return
    combined = pd.read_csv(f_comb)

    f_boot = _find("bootstrap_marginal.csv", OUT_DIR, FEAT_DIR)
    boot = pd.read_csv(f_boot) if f_boot else None
    if boot is None:
        log("[fig1] no bootstrap_marginal.csv -- top row will have no error bars")

    # tree_scheme_metrics.csv first: same harness, same calibration, same
    # blocking as combined_metrics.csv, so the shared cells agree by
    # construction. cv_metrics.csv is a legacy fallback only.
    f_cv = _find("tree_scheme_metrics.csv", OUT_DIR, FEAT_DIR)
    if f_cv is None:
        f_cv = _find("cv_metrics.csv", OUT_DIR, FEAT_DIR)
        if f_cv is not None:
            log("[fig1] falling back to cv_metrics.csv -- expect the consistency "
                "check to fail; run compare_models.py to write "
                "tree_scheme_metrics.csv instead")
    cvm = pd.read_csv(f_cv) if f_cv else None
    draw_bottom, rep = False, pd.DataFrame()
    if cvm is None:
        log("[fig1] no scheme table found -- bottom row unavailable")
    else:
        ok, rep = check_consistency(combined, cvm)
        if ok:
            draw_bottom = True
            log(f"[fig1] consistency OK: {f_comb.name} and {f_cv.name} agree on all "
                f"{len(rep)} shared cells (max diff {rep['diff'].max():.4f})")
        else:
            bad = rep[rep["diff"] > TOLERANCE]
            log(f"\n[fig1] *** CONSISTENCY FAILURE *** {len(bad)} shared cell(s) "
                f"disagree by more than {TOLERANCE}:")
            log(bad.to_string(index=False))
            log(f"[fig1] {f_cv.name} was produced with a different partition "
                "and/or without calibration. Run compare_models.py, which writes "
                "tree_scheme_metrics.csv from the same harness as "
                "combined_metrics.csv.")
            if STRICT:
                log("[fig1] STRICT=True -> bottom row REFUSED (a figure showing two "
                    "different values for one cell is worse than a missing panel)")
            else:
                draw_bottom = True
                log("[fig1] STRICT=False -> drawing anyway, panels annotated")
        rep.to_csv(OUT_DIR / "r2_source_consistency.csv", index=False)

    models = S.models_in(combined.model.unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # ---- top row: four models x {spatiotemporal, spatial}, with CIs ----
    schemes = S.schemes_in(TOP_SCHEMES)
    x = np.arange(len(models)); width = 0.8 / len(schemes)
    for ax, met in zip(axes[0], METRICS):
        for i, sch in enumerate(schemes):
            vals, lo, hi = [], [], []
            for m in models:
                r = combined[(combined.model == m) & (combined.scheme == sch)]
                vals.append(float(r[met].iloc[0]) if len(r) else np.nan)
                if boot is not None:
                    b = boot[(boot.model == m) & (boot.scheme == sch)]
                    lo.append(float(b[f"{met}_lo"].iloc[0]) if len(b) else np.nan)
                    hi.append(float(b[f"{met}_hi"].iloc[0]) if len(b) else np.nan)
            pos = x + (i - (len(schemes) - 1) / 2) * width
            err = None
            if boot is not None and len(lo) == len(vals):
                err = np.abs(np.vstack([np.array(vals) - np.array(lo),
                                        np.array(hi) - np.array(vals)]))
            ax.bar(pos, vals, width, label=S.scheme_label(sch),
                   color=S.SCHEME_COLOURS.get(sch),
                   hatch=S.SCHEME_HATCH.get(sch, ""),
                   edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW,
                   yerr=err, capsize=2.5,
                   error_kw=dict(lw=0.9, ecolor="0.2"))
        S.add_reference_line(ax, met)
        ax.set_xticks(x)
        ax.set_xticklabels([S.model_label(m, wrapped=True) for m in models], fontsize=8)
        ax.set_ylabel(S.metric_label(met))
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylim(0.15, 1.05)     # room for the wide spatial CIs
    axes[0, 1].set_title("(a) Four models, spatiotemporal versus spatial "
                         "(bars = 95% block-bootstrap CI)", fontsize=11)
    # one figure-level legend: an in-axes legend lands on the bars in every panel
    h1, l1 = axes[0, 0].get_legend_handles_labels()

    # ---- bottom row: XGB + RF across all four schemes ----
    if draw_bottom:
        bmods = [m for m in BOTTOM_MODELS if m in cvm.model.unique()]
        bsch = S.schemes_in([s for s in BOTTOM_SCHEMES if s in cvm.scheme.unique()])
        xb = np.arange(len(bmods)); wb = 0.8 / max(len(bsch), 1)
        for ax, met in zip(axes[1], METRICS):
            for i, sch in enumerate(bsch):
                vals = []
                for m in bmods:
                    r = cvm[(cvm.model == m) & (cvm.scheme == sch)]
                    vals.append(float(r[met].iloc[0]) if len(r) else np.nan)
                bars = ax.bar(xb + (i - (len(bsch) - 1) / 2) * wb, vals, wb,
                              label=S.scheme_label(sch),
                              color=S.SCHEME_COLOURS.get(sch),
                              hatch=S.SCHEME_HATCH.get(sch, ""),
                              edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW)
                S.annotate_bars(ax, bars, vals, met, fontsize=6)
            S.add_reference_line(ax, met)
            ax.set_xticks(xb)
            ax.set_xticklabels([S.model_label(m, wrapped=True) for m in bmods], fontsize=8)
            ax.set_ylabel(S.metric_label(met))
            ax.grid(axis="y", alpha=0.3)
        axes[1, 1].set_title("(b) Tree models across all four schemes \u2014 the first "
                             "three coincide, the spatial bar does not", fontsize=11)
    else:
        for ax in axes[1]:
            ax.axis("off")
        msg = ("Bottom row not drawn: the scheme table disagrees with "
               "combined_metrics.csv on shared cells\n"
               "(see r2_source_consistency.csv). Run compare_models.py to write "
               "tree_scheme_metrics.csv.")
        if cvm is None:
            msg = ("Bottom row not drawn: no tree_scheme_metrics.csv.\n"
                   "Run compare_models.py.")
        axes[1, 1].text(0.5, 0.5, msg, ha="center", va="center", fontsize=10,
                        color="#b03030", transform=axes[1, 1].transAxes)

    handles, labels = h1, l1
    if draw_bottom:
        h2, l2 = axes[1, 0].get_legend_handles_labels()
        seen = dict(zip(l1, h1))
        for lab, hh in zip(l2, h2):
            seen.setdefault(lab, hh)
        labels, handles = list(seen), list(seen.values())
    fig.legend(handles, labels, title="CV scheme", fontsize=9, ncol=len(labels),
               loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Cx. nigripalpus weekly suitability \u2014 cross-validated performance",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    S.save(fig, OUT_DIR / "r2_synthesis.png")


# ------------------------------------------------------------------ FIGURE 2
def fig_slopegraph():
    f = _find("combined_metrics.csv", OUT_DIR, FEAT_DIR)
    if f is None:
        log("[fig2] SKIP: combined_metrics.csv not found")
        return
    d = pd.read_csv(f)
    models = S.models_in(d.model.unique())

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    xs = [0, 1]

    pts = []
    for m in models:
        a = d[(d.model == m) & (d.scheme == "spatiotemporal")]["roc_auc"]
        b = d[(d.model == m) & (d.scheme == "spatial")]["roc_auc"]
        if not len(a) or not len(b):
            continue
        pts.append((m, float(a.iloc[0]), float(b.iloc[0])))

    for m, a, b in pts:
        ax.plot(xs, [a, b], marker="o", ms=7, lw=2.2,
                color=S.MODEL_COLOURS.get(m), zorder=3)

    def nudge(values, min_gap):
        """Push overlapping label y-positions apart, preserving their order.
        Three of the four spatiotemporal values sit within 0.02 of each other,
        so unadjusted labels overprint."""
        idx = np.argsort(values)
        out = np.array(values, dtype=float)
        for k in range(1, len(idx)):
            lo, hi = idx[k - 1], idx[k]
            if out[hi] - out[lo] < min_gap:
                out[hi] = out[lo] + min_gap
        return out

    span = max(max(a, b) for _, a, b in pts) - min(min(a, b) for _, a, b in pts)
    gap = span * 0.045
    ay = nudge([a for _, a, _ in pts], gap)
    by = nudge([b for _, _, b in pts], gap)

    for (m, a, b), ya, yb in zip(pts, ay, by):
        c = S.MODEL_COLOURS.get(m)
        ax.annotate(f"{S.model_label(m)}  {S.fmt(a, 'roc_auc')}",
                    (0, ya), xytext=(-12, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, color=c)
        ax.annotate(f"{S.fmt(b, 'roc_auc')}  ({b - a:+.3f})",
                    (1, yb), xytext=(12, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color=c)

    S.add_reference_line(ax, "roc_auc")
    ax.annotate("chance (0.5)", (0.5, 0.5), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=8, color=S.GREY)
    ax.set_xticks(xs)
    ax.set_xticklabels([S.scheme_label("spatiotemporal"), S.scheme_label("spatial")])
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylabel(S.metric_label("roc_auc"))
    ax.set_title("Discrimination collapses under spatial holdout in every model family",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    S.save(fig, OUT_DIR / "r2_scheme_slopegraph.png")


# ------------------------------------------------------------------ FIGURE 3
def fig_curves():
    f = _find("oof_long.csv", OUT_DIR, FEAT_DIR)
    if f is None:
        log("[fig3] SKIP: oof_long.csv not found -- run compare_models.py then "
            "build_oof_long.py")
        return
    d = pd.read_csv(f).dropna(subset=["p"])
    schemes = S.schemes_in(d.scheme.unique())
    models = S.models_in(d.model.unique())
    log(f"[fig3] {len(d):,} OOF rows | models {models} | schemes {schemes}")

    fig, axes = plt.subplots(2, len(schemes), figsize=(6.0 * len(schemes), 10),
                             squeeze=False)
    for j, sch in enumerate(schemes):
        ax_roc, ax_pr = axes[0, j], axes[1, j]
        baselines = {}
        for m in models:
            sub = d[(d.model == m) & (d.scheme == sch)]
            if sub.empty or sub.presence.nunique() < 2:
                continue
            y, p = sub.presence.to_numpy(int), sub.p.to_numpy()
            c = S.MODEL_COLOURS.get(m)

            fpr, tpr, _ = roc_curve(y, p)
            ax_roc.plot(fpr, tpr, lw=1.8, color=c, label=S.model_label(m))

            prec, rec, _ = precision_recall_curve(y, p)
            ax_pr.plot(rec, prec, lw=1.8, color=c, label=S.model_label(m))
            baselines[round(float(y.mean()), 3)] = None

        ax_roc.plot([0, 1], [0, 1], ls="--", lw=1.1, color=S.GREY, label="chance")
        ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
        ax_roc.set_xlabel("false positive rate"); ax_roc.set_ylabel("true positive rate")
        ax_roc.set_title(f"ROC \u2014 {S.scheme_label(sch)}", fontsize=11)
        ax_roc.grid(alpha=0.3); ax_roc.legend(fontsize=8, loc="lower right")

        # PR baselines differ BY MODEL because vanilla is scored on a different
        # evaluation set (prevalence 0.628 vs 0.776). Draw every baseline present,
        # or the vanilla curve looks better than it is.
        for prev in sorted(baselines):
            ax_pr.axhline(prev, ls=":", lw=1.1, color=S.GREY)
            ax_pr.annotate(f"baseline {S.fmt(prev, 'prevalence')}", (0.02, prev),
                           xytext=(0, 4), textcoords="offset points",
                           fontsize=8, color=S.GREY)
        ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)
        ax_pr.set_xlabel("recall"); ax_pr.set_ylabel("precision")
        ax_pr.set_title(f"Precision\u2013recall \u2014 {S.scheme_label(sch)}", fontsize=11)
        ax_pr.grid(alpha=0.3); ax_pr.legend(fontsize=8, loc="lower left")

    fig.suptitle("Pooled out-of-fold discrimination. Precision\u2013recall baselines "
                 "differ between models: MaxEnt-vanilla is scored on a different "
                 "evaluation set.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    S.save(fig, OUT_DIR / "r2_roc_pr_curves.png")


def run():
    S.set_thesis_style(constrained=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_synthesis()
    fig_slopegraph()
    fig_curves()
    log(f"\n[done] R2 figures -> {OUT_DIR}")


if __name__ == "__main__":
    run()