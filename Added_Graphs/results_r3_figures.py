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
CONFIDENCE INTERVALS
  Every bar carries a 95% block-bootstrap interval, resampled on the CV group
  that formed the fold (spatial_block, iso_year, or the pair), never on rows --
  rows within a group are dependent and a row bootstrap gives intervals that are
  far too narrow.
  MODEL intervals are read from bootstrap_marginal.csv, the same file the R2
  figure plots, so the two figures cannot show different whiskers for the same
  number. BASELINE intervals come from bootstrap_weekly_baselines.py, which
  resamples the same unit and reports its agreement with bootstrap_marginal.csv
  on the shared model cells -- read that [check] line before trusting the bars.
  Panel (c) needs a PAIRED interval: BSS-vs-climatology is a ratio against the
  climatology Brier, so both forecasts must see the same drawn groups. Those
  bounds come from bootstrap_weekly_model_vs_baseline.csv.
  A baseline forecast is deterministic, but its SCORE is not: it was computed on
  the cells that happened to be surveyed, and the interval quantifies
  sensitivity to that. Intervals under the SPATIAL scheme rest on six resampling
  groups and are coarse; n_groups is carried in the CSVs so this stays visible.
PREVALENCE IS NOT DRAWN
  It is the denominator of panel (b), so its bar is 0 by construction, and a
  constant forecast cannot rank, so it has no ROC. Its value is printed in the
  panel (b) axis label instead of occupying an empty column in two of three
  panels.
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
# baseline ordering, labels and colours now live in report_style, since the
# nowcast figures use them too
# Prevalence is excluded: it is the denominator of panel (b) and has no ROC.
# See PREVALENCE IS NOT DRAWN in the header.
BASELINES = [b for b in S.BASELINE_ORDER if b != "prevalence"]
BASELINE_LABELS = S.BASELINE_PRETTY
BASELINE_COLOURS = S.BASELINE_COLOURS
# models whose evaluation set matches the baselines' 21,608 rows
CLIM_COMPARABLE = ["xgboost", "random_forest", "maxent_targetgroup"]
# Which CV schemes get a column. Temporal is omitted by default: its numbers sit
# within 0.006 of spatiotemporal on every metric, so the column adds a third of
# the figure's width and no information. Set to None to show every scheme found
# in baseline_metrics_weekly.csv.
SHOW_SCHEMES = ["spatiotemporal"]
# With a SINGLE scheme the metric panels would stack into a tall column, which
# wastes a portrait page. Lay them out as one row instead. With two or more
# schemes the columns-are-schemes layout is kept, because that is the axis the
# reader compares along.
ROW_LAYOUT_WHEN_SINGLE_SCHEME = True
# Which schemes appear as rows in the forest plot. Kept separate from
# SHOW_SCHEMES because the forest has room for more rows than the baseline
# figure has columns.
#
# spatiotemporal_3blocks is EXCLUDED, and not only for space:
# bootstrap_uncertainty.group_key() resamples it on spatial_block x iso_year --
# the SIX-block unit -- rather than on coarse_block x iso_year. Its rows in
# bootstrap_marginal.csv therefore report 32 groups, identical to ordinary
# spatiotemporal CV, when three coarse blocks over six years can yield at most
# 18. The intervals are computed on the wrong resampling unit and should not be
# plotted until group_key is corrected.
FOREST_SCHEMES = ["spatiotemporal", "temporal", "spatial"]
def log(m): print(m, flush=True)

def _find(name, *dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None

def load_intervals():
    """(scheme, method) -> interval record, for models AND baselines.
    MODEL bounds come from bootstrap_marginal.csv -- the file the R2 figure
    plots -- so a given model/scheme cell shows the same whisker in both
    figures. BASELINE bounds come from bootstrap_weekly_baselines.py, which
    resamples the same CV group.
    """
    out = {}
    fm = _find("bootstrap_marginal.csv", OUT_DIR, FEAT_DIR, BASE_DIR)
    if fm is not None:
        b = pd.read_csv(fm)
        out.update({(r.scheme, r.model): r._asdict() for r in b.itertuples()})
        log(f"[fig1] model intervals from {fm.name}")
    else:
        log("[fig1] no bootstrap_marginal.csv -- model bars will have no whiskers")
    fb = _find("bootstrap_weekly_baselines.csv", BASE_DIR, OUT_DIR, FEAT_DIR)
    if fb is not None:
        b = pd.read_csv(fb)
        if "kind" in b.columns:
            b = b[b.kind == "baseline"]
        out.update({(r.scheme, r.method): r._asdict() for r in b.itertuples()})
        log(f"[fig1] baseline intervals from {fb.name}")
    else:
        log("[fig1] no bootstrap_weekly_baselines.csv -- baseline bars will have "
            "no whiskers; run run_baselines.py then bootstrap_weekly_baselines.py")
    return out

def load_paired():
    """(scheme, model) -> BSS-vs-climatology interval, for panel (c).
    Not obtainable from bootstrap_marginal.csv: the quantity is a ratio against
    the climatology Brier, so its interval requires a PAIRED resample in which
    both forecasts are scored on the same drawn groups.
    """
    f = _find("bootstrap_weekly_model_vs_baseline.csv", BASE_DIR, OUT_DIR, FEAT_DIR)
    if f is None:
        log("[fig1] no bootstrap_weekly_model_vs_baseline.csv -- panel (c) will "
            "have no whiskers")
        return {}
    b = pd.read_csv(f)
    b = b[b.baseline == "climatology"]
    log(f"[fig1] paired BSS-vs-climatology intervals from {f.name}")
    return {(r.scheme, r.model): r._asdict() for r in b.itertuples()}

def _errbars(vals, bounds):
    """Two-row yerr array from (lo, hi) pairs; a missing bound gives no whisker."""
    lo, hi = [], []
    for v, (l, h) in zip(vals, bounds):
        l = v if l is None or not np.isfinite(l) else float(l)
        h = v if h is None or not np.isfinite(h) else float(h)
        lo.append(l); hi.append(h)
    return np.nan_to_num(np.abs(np.vstack([
        np.asarray(vals, float) - np.asarray(lo, float),
        np.asarray(hi, float) - np.asarray(vals, float)])))
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
    if SHOW_SCHEMES is not None:
        dropped = [s for s in schemes if s not in SHOW_SCHEMES]
        schemes = [s for s in schemes if s in SHOW_SCHEMES]
        if dropped:
            log(f"[fig1] omitting scheme column(s) {dropped} (SHOW_SCHEMES); "
                f"their values remain in r3_bss_vs_climatology.csv")
    models = S.models_in(mod.model.unique())
    log(f"[fig1] schemes {schemes} | models {models}")
    boot = load_intervals()
    pair = load_paired()
    # ---- derive BSS vs climatology, arithmetically, no refit -------------
    clim = (base[base.baseline == "climatology"]
            .set_index("scheme")[["brier", "n_scored"]])
    rows = []
    for sch in [x for x in S.schemes_in(base.scheme.unique()) if x in set(mod.scheme)]:
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
    single = len(schemes) == 1 and ROW_LAYOUT_WHEN_SINGLE_SCHEME
    if single:
        fig, row = plt.subplots(1, 3, figsize=(16.5, 5.6))
        # keep the (metric, scheme) indexing the rest of the function uses
        axes = np.array([[row[0]], [row[1]], [row[2]]], dtype=object)
    else:
        fig, axes = plt.subplots(3, len(schemes), figsize=(5.2 * len(schemes), 11),
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
            err = None
            if boot:
                err = _errbars(vals, [
                    (boot.get((sch, k), {}).get(f"{metric}_lo"),
                     boot.get((sch, k), {}).get(f"{metric}_hi"))
                    for k, _ in entries])
            bars = ax.bar(x, vals, 0.72, color=cols,
                          edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW,
                          yerr=err, capsize=2.5,
                          error_kw=dict(lw=0.9, ecolor="0.2", zorder=5))
            for b, h in zip(bars, hatches):
                if h:
                    b.set_hatch(h)
            S.add_reference_line(ax, metric)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [S.model_label(k, wrapped=True) if kd == "model"
                 else BASELINE_LABELS.get(k, k).replace(" (", "\n(")
                 for k, kd in entries], fontsize=7, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.3)
            return ns, vals
        ns, _ = draw(axes[0, j], "roc_auc", "roc_auc", "roc_auc")
        axes[0, j].set_ylim(0.30, 1.0)
        axes[0, j].set_title(S.metric_label("roc_auc") if single
                             else S.scheme_label(sch), fontsize=12)
        if j == 0:
            axes[0, j].set_ylabel(S.metric_label("roc_auc"))
        _, bss_vals = draw(axes[1, j], "bss", "bss_vs_prevalence", "bss")
        axes[1, j]._r3_scheme = sch          # for the shared-limit pass below
        bss_panels.append((axes[1, j], entries, bss_vals))
        if single:
            axes[1, j].set_title("BSS vs prevalence", fontsize=12)
        if j == 0:
            # prevalence has no bar of its own (see header); its value is stated
            # here, because it is the denominator this panel is measured against
            pv = None
            pr = base[(base.scheme == sch) & (base.baseline == "climatology")]
            if len(pr) and "prevalence" in pr.columns and pd.notna(pr.prevalence.iloc[0]):
                pv = float(pr.prevalence.iloc[0])
            axes[1, j].set_ylabel("BSS vs prevalence"
                                  + (f" ({pv:.3f})" if pv is not None else ""))
        # ---- (c) models only, climatology as the origin
        d = derived[(derived.scheme == sch) & derived.comparable]
        mm = [m for m in models if m in set(d.model)]
        xc = np.arange(len(mm))
        vals = [float(d[d.model == m].bss_vs_climatology.iloc[0]) for m in mm]
        ax = axes[2, j]
        cerr = None
        if pair:
            cerr = _errbars(vals, [
                (pair.get((sch, m), {}).get("bss_vs_b_lo"),
                 pair.get((sch, m), {}).get("bss_vs_b_hi")) for m in mm])
        bars = ax.bar(xc, vals, 0.6, color=[S.MODEL_COLOURS.get(m) for m in mm],
                      edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW,
                      yerr=cerr, capsize=2.5,
                      error_kw=dict(lw=0.9, ecolor="0.2", zorder=5))
        ax.axhline(0.0, color=BASELINE_COLOURS["climatology"], lw=1.4)
        ax.annotate("climatology", (len(mm) - 0.5, 0), xytext=(0, 3),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=7, color=BASELINE_COLOURS["climatology"])
        ax.set_xticks(xc)
        ax.set_xticklabels([S.model_label(m, wrapped=True) for m in mm],
                           fontsize=7, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
        if single:
            ax.set_title("BSS vs climatology", fontsize=12)
        if j == 0:
            ax.set_ylabel("BSS vs climatology")
    # Shared y-limits per row so panels are comparable across schemes.
    # The BSS row is scaled to the COMPARABLE entries only: MaxEnt-vanilla's
    # spatial BSS of -1.103 is on a different evaluation set, and letting it set
    # the scale flattens every other bar into invisibility. It is drawn clipped
    # with its true value printed instead.
    # Bounds are included alongside bar heights: with whiskers drawn, a limit
    # computed from bar tops alone would clip the intervals it exists to show.
    keep, everything = [], []
    for ax, entries, vals in bss_panels:
        sch_of = getattr(ax, "_r3_scheme", None)
        for (key, kind), v in zip(entries, vals):
            if not np.isfinite(v):
                continue
            cand = [v]
            r = boot.get((sch_of, key), {}) if sch_of else {}
            for side in ("bss_lo", "bss_hi"):
                b_ = r.get(side)
                if b_ is not None and np.isfinite(b_):
                    cand.append(float(b_))
            everything.extend(cand)
            if not (kind == "model" and key not in CLIM_COMPARABLE):
                keep.extend(cand)
    # Only clip when the excluded values would actually flatten the panel. Under
    # spatial CV, MaxEnt-vanilla's -1.103 is four times the range of everything
    # else (ratio 6.9) and must be clipped; under spatiotemporal CV its +0.516 is
    # only 1.8x, so clipping there would hide a real value for no benefit.
    if keep and everything:
        span_keep = max(keep) - min(keep)
        span_all = max(everything) - min(everything)
        if span_all <= 2.5 * (span_keep or 1e-9):
            keep = everything
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
                # white backing box: the label sits ON TOP of the clipped bar,
                # which is dark and often hatched, so plain text is unreadable
                box = dict(fc="white", ec="#b03030", lw=0.5, pad=1.6, alpha=0.95)
                if v > yhi:
                    ax.annotate(f"\u2191 {S.fmt(v, 'bss')}", (k, yhi),
                                xytext=(0, -11), textcoords="offset points",
                                ha="center", va="top", fontsize=7,
                                color="#b03030", zorder=6, bbox=box)
                elif v < ylo:
                    ax.annotate(f"\u2193 {S.fmt(v, 'bss')}", (k, ylo),
                                xytext=(0, 11), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7,
                                color="#b03030", zorder=6, bbox=box)
    lo2 = min(a.get_ylim()[0] for a in axes[2]); hi2 = max(a.get_ylim()[1] for a in axes[2])
    pad2 = 0.10 * (hi2 - lo2 or 1)      # headroom: value labels sit outside the bar
    for a in axes[2]:
        a.set_ylim(lo2 - pad2, hi2 + pad2)
    where = (f" \u2014 {S.scheme_label(schemes[0])}" if single else "")
    panel_word = "right-hand panel" if single else "bottom row"
    fig.suptitle(f"Models against reference forecasts{where}. Whiskers are 95% "
                 "block-bootstrap intervals, resampled on each scheme's CV group. "
                 "HATCHED bars are\nscored on a different row set from the "
                 "reference \u2014 persistence (17,166 rows) and MaxEnt-vanilla "
                 f"(26,667) \u2014 and are not directly comparable. The {panel_word}, "
                 "referenced to climatology, excludes MaxEnt-vanilla for the same "
                 "reason.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88 if single else 0.93])
    S.save(fig, OUT_DIR / "r3_models_vs_baselines.png")

# ------------------------------------------------------------------ FIGURE 2
def fig_forest():
    fm = _find("bootstrap_marginal.csv", OUT_DIR, FEAT_DIR)
    if fm is None:
        log("[fig2] SKIP: bootstrap_marginal.csv not found -- run bootstrap_uncertainty.py")
        return
    b = pd.read_csv(fm)
    if FOREST_SCHEMES is not None:
        dropped = sorted(set(b.scheme) - set(FOREST_SCHEMES))
        b = b[b.scheme.isin(FOREST_SCHEMES)]
        if dropped:
            log(f"[fig2] omitting scheme row(s) {dropped} (FOREST_SCHEMES)")
    if b.empty:
        log("[fig2] SKIP: no rows left after the scheme filter")
        return
    metrics = [m for m in ("roc_auc", "pr_lift", "bss") if m in b.columns]
    # order: scheme block, models canonical within it
    b["_s"] = pd.Categorical(b.scheme, S.schemes_in(b.scheme), ordered=True)
    b["_m"] = pd.Categorical(b.model, S.models_in(b.model), ordered=True)
    b = b.sort_values(["_s", "_m"]).reset_index(drop=True)
    rows = list(b.index)[::-1]           # top of plot = first row
    ypos = {i: k for k, i in enumerate(rows)}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics),
                                                      2.2 + 0.52 * len(b)),
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