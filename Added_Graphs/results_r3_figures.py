from __future__ import annotations
"""
results_r3_figures.py  —  the two R3 figures. Reads persisted outputs only.

  FIG 1  r3_models_vs_baselines.png    (3 rows x one column per scheme)
         (a) ROC-AUC        models and reference forecasts on one axis
         (b) BSS vs prevalence   ditto -- the reference every model BSS in the
                                 thesis is computed against
         (c) BSS vs CLIMATOLOGY  models only, derived arithmetically as
                                 1 - brier_model / brier_climatology.
                                 Zero on this axis IS climatology, so the sign
                                 of the bar is the headline result: positive
                                 under spatiotemporal CV, negative under spatial
                                 holdout.
         Also writes r3_bss_vs_climatology.csv with the derived values.

  FIG 2  r3_forest.png                 one panel per metric, every model x
         scheme on a shared vertical axis with its 95% block-bootstrap CI and
         its resampling-group count. Replaces the marginal CI table.
         Also writes r3_paired_table.csv, the four-row paired-difference table.

--------------------------------------------------------------------------
TWO DENOMINATOR TRAPS, both guarded below

1. PERSISTENCE IS NOT DENOMINATOR-MATCHED. It scores 17,166 rows at prevalence
   0.805; the models and the other baselines score 21,608 at 0.776. Its bars are
   therefore HATCHED and its n annotated. Do not compute model-minus-persistence
   differences from this figure.

2. MAXENT-VANILLA CANNOT BE PUT ON THE CLIMATOLOGY AXIS. It is scored on 26,667
   rows (presences + random background); the climatology baseline was computed
   on the 21,608 trap rows. 1 - brier_model/brier_clim across different row sets
   is not a skill score. Panel (c) therefore excludes it, and says so.
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

import report_style as S

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
FEAT_DIR = Path(cfg["weekly_xg_dir"])
MAXENT_DIR = Path(cfg.get("maxent_dir", str(FEAT_DIR)))
OUT_DIR = Path(cfg.get("comparison_dir", str(FEAT_DIR.parent / "Comparison_Results")))
NOWCAST_RES = Path(cfg.get("nowcast_results_dir", str(FEAT_DIR)))
BASE_DIR = Path(cfg.get("baselines_dir", str(NOWCAST_RES / "Baselines")))

# which persistence variant is THE persistence baseline in the text; the other
# variants stay in the CSV and go to an appendix
BASELINES = ["climatology", "persistence", "prevalence"]
BASELINE_LABELS = {"climatology": "Climatology", "persistence": "Persistence",
                   "prevalence": "Prevalence (constant)"}
# greys, disjoint from every palette in report_style
BASELINE_COLOURS = {"climatology": "#4f4f4f", "persistence": "#8c8c8c",
                    "prevalence": "#c4c4c4"}
# models whose evaluation set matches the baselines' 21,608 rows
CLIM_COMPARABLE = ["xgboost", "random_forest", "maxent_targetgroup"]

def log(m): print(m, flush=True)


def _find(name, *dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None
# ============================================================================


def load_model_metrics() -> pd.DataFrame:
    """Model metrics with a brier column, across every scheme available.

    combined_metrics.csv has no brier, so trees come from tree_scheme_metrics.csv
    and MaxEnt from its own cv_metrics files.
    """
    frames = []
    f = _find("tree_scheme_metrics.csv", OUT_DIR, FEAT_DIR)
    if f:
        frames.append(pd.read_csv(f))
        log(f"[load] {f.name}")
    else:
        log("[load] tree_scheme_metrics.csv missing -- run compare_models.py")
    for variant in ("vanilla", "targetgroup"):
        g = _find(f"cv_metrics_maxent_{variant}.csv", MAXENT_DIR, OUT_DIR, FEAT_DIR)
        if g:
            frames.append(pd.read_csv(g))
            log(f"[load] {g.name}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ FIGURE 1
def fig_baselines():
    fb = _find("baseline_metrics_weekly.csv", BASE_DIR, OUT_DIR, FEAT_DIR)
    if fb is None:
        log("[fig1] SKIP: baseline_metrics_weekly.csv not found -- run run_baselines.py")
        return
    base = pd.read_csv(fb)
    mod = load_model_metrics()
    if mod.empty:
        log("[fig1] SKIP: no model metrics with a brier column")
        return

    schemes = [s for s in S.schemes_in(base.scheme.unique()) if s in set(mod.scheme)]
    models = S.models_in(mod.model.unique())
    log(f"[fig1] schemes {schemes} | models {models}")

    # ---- derive BSS vs climatology, arithmetically, no refit -------------
    clim = (base[base.baseline == "climatology"]
            .set_index("scheme")[["brier", "n_scored"]])
    rows = []
    for sch in schemes:
        if sch not in clim.index:
            continue
        bc, nc = float(clim.loc[sch, "brier"]), int(clim.loc[sch, "n_scored"])
        for m in models:
            r = mod[(mod.model == m) & (mod.scheme == sch)]
            if not len(r):
                continue
            r = r.iloc[0]
            comparable = (m in CLIM_COMPARABLE) and int(r.n) == nc
            rows.append(dict(model=m, scheme=sch, n_model=int(r.n), n_clim=nc,
                             brier_model=float(r.brier), brier_clim=bc,
                             bss_vs_climatology=round(1 - float(r.brier) / bc, 3)
                             if comparable else np.nan,
                             comparable=comparable))
    derived = pd.DataFrame(rows)
    derived.to_csv(OUT_DIR / "r3_bss_vs_climatology.csv", index=False)
    log("\n[fig1] BSS vs climatology (blank = different evaluation set):")
    log(derived.to_string(index=False))

    # ---------------------------------------------------------------- plot
    fig, axes = plt.subplots(3, len(schemes), figsize=(4.6 * len(schemes), 11),
                             squeeze=False)
    bss_panels = []      # (axis, entries, values) for the shared-limit pass

    for j, sch in enumerate(schemes):
        bsub = base[(base.scheme == sch) & (base.baseline.isin(BASELINES))]
        present_b = [b for b in BASELINES if b in set(bsub.baseline)]
        n_match = None
        cm = bsub[bsub.baseline == "climatology"]
        if len(cm):
            n_match = int(cm.n_scored.iloc[0])

        entries = ([(m, "model") for m in models if
                    len(mod[(mod.model == m) & (mod.scheme == sch)])]
                   + [(b, "baseline") for b in present_b])
        x = np.arange(len(entries))

        def value(key, kind, col_model, col_base):
            if kind == "model":
                r = mod[(mod.model == key) & (mod.scheme == sch)]
                return float(r[col_model].iloc[0]) if len(r) else np.nan
            r = bsub[bsub.baseline == key]
            if not len(r) or col_base not in r.columns:
                return np.nan
            v = r[col_base].iloc[0]
            return float(v) if pd.notna(v) else np.nan

        def draw(ax, col_model, col_base, metric):
            vals, cols, hatches, ns = [], [], [], []
            for key, kind in entries:
                vals.append(value(key, kind, col_model, col_base))
                if kind == "model":
                    cols.append(S.MODEL_COLOURS.get(key))
                    n = mod[(mod.model == key) & (mod.scheme == sch)].n.iloc[0]
                else:
                    cols.append(BASELINE_COLOURS.get(key, "#999"))
                    n = bsub[bsub.baseline == key].n_scored.iloc[0]
                ns.append(int(n))
                # hatch anything not scored on the matched row set
                hatches.append("//" if (n_match and int(n) != n_match) else "")
            bars = ax.bar(x, vals, 0.72, color=cols,
                          edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW)
            for b, h in zip(bars, hatches):
                if h:
                    b.set_hatch(h)
            S.add_reference_line(ax, metric)
            S.annotate_bars(ax, bars, vals, metric, fontsize=6)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [S.model_label(k, wrapped=True) if kd == "model"
                 else BASELINE_LABELS.get(k, k).replace(" (", "\n(")
                 for k, kd in entries], fontsize=7, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.3)
            return ns, vals

        ns, _ = draw(axes[0, j], "roc_auc", "roc_auc", "roc_auc")
        axes[0, j].set_title(S.scheme_label(sch), fontsize=12)
        axes[0, j].set_ylim(0.30, 1.0)
        if j == 0:
            axes[0, j].set_ylabel(S.metric_label("roc_auc"))

        _, bss_vals = draw(axes[1, j], "bss", "bss_vs_prevalence", "bss")
        bss_panels.append((axes[1, j], entries, bss_vals))
        if j == 0:
            axes[1, j].set_ylabel("BSS vs prevalence")

        # ---- (c) models only, climatology as the origin
        d = derived[(derived.scheme == sch) & derived.comparable]
        mm = [m for m in models if m in set(d.model)]
        xc = np.arange(len(mm))
        vals = [float(d[d.model == m].bss_vs_climatology.iloc[0]) for m in mm]
        ax = axes[2, j]
        bars = ax.bar(xc, vals, 0.6, color=[S.MODEL_COLOURS.get(m) for m in mm],
                      edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW)
        ax.axhline(0.0, color=BASELINE_COLOURS["climatology"], lw=1.4)
        ax.annotate("climatology", (len(mm) - 0.5, 0), xytext=(0, 3),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=7, color=BASELINE_COLOURS["climatology"])
        S.annotate_bars(ax, bars, vals, "bss", fontsize=6)
        ax.set_xticks(xc)
        ax.set_xticklabels([S.model_label(m, wrapped=True) for m in mm],
                           fontsize=7, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
        if j == 0:
            ax.set_ylabel("BSS vs climatology")

    # Shared y-limits per row so panels are comparable across schemes.
    # The BSS row is scaled to the COMPARABLE entries only: MaxEnt-vanilla's
    # spatial BSS of -1.103 is on a different evaluation set, and letting it set
    # the scale flattens every other bar into invisibility. It is drawn clipped
    # with its true value printed instead.
    keep = []
    for ax, entries, vals in bss_panels:
        for (key, kind), v in zip(entries, vals):
            if np.isfinite(v) and not (kind == "model" and key not in CLIM_COMPARABLE):
                keep.append(v)
    if keep:
        lo, hi = min(keep), max(keep)
        pad = 0.14 * (hi - lo or 1)
        for ax, entries, vals in bss_panels:
            ylo, yhi = lo - pad, hi + pad
            ax.set_ylim(ylo, yhi)
            # value labels written by annotate_bars sit at the bar height; once
            # the axis is clipped, an out-of-range bar's label lands outside the
            # figure. Confine every label, then re-label the clipped bars
            # explicitly, with the arrow pointing the way the bar actually runs.
            for t in ax.texts:
                t.set_clip_on(True)
            for k, ((key, kind), v) in enumerate(zip(entries, vals)):
                if not np.isfinite(v):
                    continue
                if v > yhi:
                    ax.annotate(f"\u2191 {S.fmt(v, 'bss')}", (k, yhi),
                                xytext=(0, -10), textcoords="offset points",
                                ha="center", va="top", fontsize=7,
                                color="#b03030", zorder=6)
                elif v < ylo:
                    ax.annotate(f"\u2193 {S.fmt(v, 'bss')}", (k, ylo),
                                xytext=(0, 10), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7,
                                color="#b03030", zorder=6)
    lo2 = min(a.get_ylim()[0] for a in axes[2]); hi2 = max(a.get_ylim()[1] for a in axes[2])
    for a in axes[2]:
        a.set_ylim(lo2, hi2)

    fig.suptitle("Models against reference forecasts. HATCHED bars are scored on a "
                 "different row set from the reference \u2014 persistence (17,166 rows) "
                 "and\nMaxEnt-vanilla (26,667) \u2014 and are not directly comparable. "
                 "The bottom row, referenced to climatology, excludes MaxEnt-vanilla "
                 "for the same reason.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    S.save(fig, OUT_DIR / "r3_models_vs_baselines.png")


# ------------------------------------------------------------------ FIGURE 2
def fig_forest():
    fm = _find("bootstrap_marginal.csv", OUT_DIR, FEAT_DIR)
    if fm is None:
        log("[fig2] SKIP: bootstrap_marginal.csv not found -- run bootstrap_uncertainty.py")
        return
    b = pd.read_csv(fm)
    metrics = [m for m in ("roc_auc", "pr_lift", "bss") if m in b.columns]

    # order: scheme block, models canonical within it
    b["_s"] = pd.Categorical(b.scheme, S.schemes_in(b.scheme), ordered=True)
    b["_m"] = pd.Categorical(b.model, S.models_in(b.model), ordered=True)
    b = b.sort_values(["_s", "_m"]).reset_index(drop=True)
    rows = list(b.index)[::-1]           # top of plot = first row
    ypos = {i: k for k, i in enumerate(rows)}

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 8),
                             squeeze=False, sharey=True)
    for ax, met in zip(axes[0], metrics):
        for i, r in b.iterrows():
            y = ypos[i]
            lo, hi = float(r[f"{met}_lo"]), float(r[f"{met}_hi"])
            ax.plot([lo, hi], [y, y], lw=1.6, color=S.MODEL_COLOURS.get(r.model),
                    solid_capstyle="butt", zorder=2)
            ax.plot([lo, lo, hi, hi], [y - .16, y + .16, y - .16, y + .16],
                    ls="none", marker="|", ms=6,
                    color=S.MODEL_COLOURS.get(r.model), zorder=2)
            ax.plot([float(r[met])], [y], marker="o", ms=6,
                    color=S.MODEL_COLOURS.get(r.model), zorder=3)
        # NOTE: S.add_reference_line draws a HORIZONTAL line, which is correct
        # for bar charts and wrong here -- on a forest plot the no-skill value
        # is a vertical line. Drawn directly, using the same constant.
        ref = S.REFERENCE_LINE.get(met)
        if ref is not None:
            ax.axvline(ref, color=S.GREY, ls="--", lw=1.0, zorder=0)
        ax.set_xlabel(S.metric_label(met))
        ax.grid(axis="x", alpha=0.3)

    labels = [f"{S.model_label(b.loc[i, 'model'])}  \u2014  "
              f"{S.scheme_label(b.loc[i, 'scheme'])}   "
              f"({b.loc[i, 'n_groups']} groups, n={S.fmt(b.loc[i, 'n'], 'n')})"
              for i in rows]
    axes[0, 0].set_yticks(range(len(rows)))
    axes[0, 0].set_yticklabels(labels, fontsize=7)
    axes[0, 0].set_ylim(-0.6, len(rows) - 0.4)

    fig.suptitle("95% block-bootstrap intervals. Width is governed by the number of "
                 "resampling groups, printed beside each row.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    S.save(fig, OUT_DIR / "r3_forest.png")

    # ---- paired-difference table ----
    fp = _find("bootstrap_paired.csv", OUT_DIR, FEAT_DIR)
    if fp is None:
        log("[fig2] no bootstrap_paired.csv -- paired table skipped")
        return
    p = pd.read_csv(fp)
    out = p.assign(
        comparison=lambda d: d.model_a.map(S.PRETTY).fillna(d.model_a) + " \u2212 "
                             + d.model_b.map(S.PRETTY).fillna(d.model_b),
        scheme_label=lambda d: d.scheme.map(S.scheme_label),
        d_roc=lambda d: [f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]" for v, lo, hi in
                         zip(d.d_roc_auc, d.d_roc_auc_lo, d.d_roc_auc_hi)],
        d_bss_fmt=lambda d: [f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]" for v, lo, hi in
                             zip(d.d_bss, d.d_bss_lo, d.d_bss_hi)],
    )[["scheme_label", "comparison", "n", "d_roc", "d_roc_auc_frac_a_better",
       "d_bss_fmt", "d_bss_frac_a_better"]]
    out.to_csv(OUT_DIR / "r3_paired_table.csv", index=False)
    log("\n[fig2] paired differences (every interval spanning zero means no "
        "model separates from another):")
    log(out.to_string(index=False))
    spans_zero = ((p.d_roc_auc_lo < 0) & (p.d_roc_auc_hi > 0)).sum()
    log(f"[fig2] {spans_zero} of {len(p)} ROC differences have a CI spanning zero")


def run():
    S.set_thesis_style(constrained=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_baselines()
    fig_forest()
    log(f"\n[done] R3 figures -> {OUT_DIR}")


if __name__ == "__main__":
    run()