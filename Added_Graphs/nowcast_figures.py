from __future__ import annotations
"""
nowcast_figures.py  —  three diagnostics for the forward-chaining nowcast,
all on the HEADLINE FOLD (train 2013-2017, test 2018). Reads persisted output;
fits nothing.

  FIG 1  nowcast_roc_pr_2018.png
         Pooled ROC and precision-recall curves, four models on one axis. The
         chapter reports AUCs throughout but never shows a curve.
         PR baselines are drawn PER EVALUATION SET: the trap-restricted trio
         sits at prevalence 0.813, MaxEnt-vanilla at 0.539. With one baseline
         drawn, vanilla's curve looks stronger than it is.

  FIG 2  nowcast_weekly_skill_2018.png
         The 2018 fold decomposed by ISO week: ROC-AUC and Brier per week, four
         models, with the weekly sample size beneath. Answers "WHEN in the
         season does the nowcast work?", which the pooled 2018 number cannot,
         and connects directly to the week-43 collapse in the validation maps.

         Weeks with only one observed class give an undefined ROC and are left
         blank; weeks below MIN_N_WEEK are drawn faint and flagged, because a
         ROC on nine observations is not a seasonal signal.

  FIG 3  nowcast_reliability_2018.png
         Reliability diagram plus prediction histogram. BSS is the headline
         claim of the nowcast section and its reliability component has never
         been shown. Quantile (equal-count) bins, as elsewhere: predictions
         concentrate near the top at prevalence 0.813, so equal-width bins
         would be mostly empty.

--------------------------------------------------------------------------
ISO WEEK IS NOT IN THE OOF FILE

nowcast_oof_long.csv carries Grid_ID, presence, model, p, test_year -- the week
was dropped when it was written, so FIG 2 cannot be built from it directly.

This script reconstructs the week from weekly_model_table.parquet BY POSITION,
and only after verifying that, for the fold in question, the Grid_ID sequence
AND the presence sequence match the table exactly. If either differs the figure
is SKIPPED rather than drawn from a guessed alignment.

The durable fix is one line in whatever writes the OOF file: carry iso_week
through alongside Grid_ID. Do that and the check below becomes a no-op.
--------------------------------------------------------------------------

OUTPUT -> nowcast_results_dir
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, brier_score_loss

import report_style as S

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
FEAT_DIR = Path(cfg["weekly_xg_dir"])
NOWCAST_DIR = Path(cfg.get("nowcast_dir", str(FEAT_DIR)))
RES = Path(cfg.get("nowcast_results_dir", str(NOWCAST_DIR / "Nowcast_Results")))

TEST_YEAR = 2018          # the headline fold
MIN_N_WEEK = 20           # weeks below this are drawn faint and ringed
N_BINS = 10               # reliability bins (quantile)
MIN_BIN = 15              # minimum observations per reliability bin

def log(m): print(m, flush=True)


def _find(name, *dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None
# ============================================================================


def load_oof() -> pd.DataFrame:
    f = _find("nowcast_oof_long.csv", RES, NOWCAST_DIR, FEAT_DIR)
    if f is None:
        log("[load] nowcast_oof_long.csv not found")
        return pd.DataFrame()
    d = pd.read_csv(f).dropna(subset=["p"])
    d = d[d.test_year == TEST_YEAR].copy()
    log(f"[load] {f.name}: {len(d):,} rows for test year {TEST_YEAR} | "
        f"models {sorted(d.model.unique())}")
    return d


def attach_iso_week(oof: pd.DataFrame) -> pd.DataFrame | None:
    """Recover iso_week by position, ONLY if the alignment is provably correct."""
    if "iso_week" in oof.columns:
        log("[week] iso_week already present -- no reconstruction needed")
        return oof

    tbl_path = FEAT_DIR / "weekly_model_table.parquet"
    if not tbl_path.exists():
        log(f"[week] SKIP: {tbl_path.name} not found, cannot recover iso_week")
        return None
    tbl = pd.read_parquet(tbl_path, columns=["Grid_ID", "iso_year", "iso_week",
                                             "presence"])
    year = tbl[tbl.iso_year == TEST_YEAR].reset_index(drop=True)
    log(f"[week] model table has {len(year):,} rows for {TEST_YEAR}")

    out = []
    for m, sub in oof.groupby("model", sort=False):
        sub = sub.reset_index(drop=True)
        if len(sub) != len(year):
            log(f"[week] {m}: {len(sub):,} OOF rows vs {len(year):,} table rows "
                f"-- lengths differ, alignment NOT verifiable")
            continue
        same_gid = (sub.Grid_ID.to_numpy() == year.Grid_ID.to_numpy()).all()
        same_y = (sub.presence.to_numpy() == year.presence.to_numpy()).all()
        if not (same_gid and same_y):
            log(f"[week] {m}: Grid_ID match={same_gid}, presence match={same_y} "
                f"-- positional alignment REJECTED")
            continue
        s = sub.copy()
        s["iso_week"] = year.iso_week.to_numpy()
        out.append(s)
        log(f"[week] {m}: alignment verified, iso_week attached")

    if not out:
        log("[week] no model could be aligned -- FIG 2 will be skipped.\n"
            "[week] FIX: carry iso_week through when writing nowcast_oof_long.csv.")
        return None
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ FIGURE 1
def fig_roc_pr(oof: pd.DataFrame):
    models = S.models_in(oof.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    prevs = {}
    for m in models:
        sub = oof[oof.model == m]
        if sub.presence.nunique() < 2:
            continue
        y, p = sub.presence.to_numpy(int), sub.p.to_numpy()
        c = S.MODEL_COLOURS.get(m)
        auc = roc_auc_score(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, lw=1.9, color=c,
                     label=f"{S.model_label(m)}  {S.fmt(auc, 'roc_auc')}")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, lw=1.9, color=c, label=S.model_label(m))
        prevs[round(float(y.mean()), 3)] = None

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color=S.GREY, label="chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"(a) ROC — {TEST_YEAR} forward forecast", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for pv in sorted(prevs):
        ax.axhline(pv, ls=":", lw=1.1, color=S.GREY)
        ax.annotate(f"baseline {S.fmt(pv, 'prevalence')}", (0.02, pv),
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=8, color=S.GREY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("(b) Precision–recall", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(f"Nowcast, {TEST_YEAR} fold (trained on 2013–2017). "
                 "Precision–recall baselines differ between models: "
                 "MaxEnt-vanilla is scored on a different evaluation set.",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    S.save(fig, RES / f"nowcast_roc_pr_{TEST_YEAR}.png")


# ------------------------------------------------------------------ FIGURE 2
def fig_weekly(oof_wk: pd.DataFrame):
    models = S.models_in(oof_wk.model.unique())
    rows = []
    for m in models:
        for wk, sub in oof_wk[oof_wk.model == m].groupby("iso_week"):
            y, p = sub.presence.to_numpy(int), sub.p.to_numpy()
            rec = dict(model=m, iso_week=int(wk), n=len(y),
                       prevalence=round(float(y.mean()), 3),
                       brier=round(float(brier_score_loss(y, p)), 4))
            # ROC is undefined when a week contains only presences or only absences
            rec["roc_auc"] = (round(float(roc_auc_score(y, p)), 3)
                              if len(np.unique(y)) == 2 else np.nan)
            rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(RES / f"nowcast_weekly_skill_{TEST_YEAR}.csv", index=False)

    single = d.roc_auc.isna().sum()
    thin = (d.n < MIN_N_WEEK).sum()
    log(f"[fig2] {len(d)} model-weeks | {single} with a single class (ROC blank) "
        f"| {thin} below n={MIN_N_WEEK}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 2.2, 0.9]})
    for ax, met in zip(axes[:2], ["roc_auc", "brier"]):
        for m in models:
            s = d[d.model == m].sort_values("iso_week")
            c = S.MODEL_COLOURS.get(m)
            ax.plot(s.iso_week, s[met], marker="o", ms=3.5, lw=1.7, color=c,
                    label=S.model_label(m))
            low = s[s.n < MIN_N_WEEK]
            if len(low):
                ax.scatter(low.iso_week, low[met], s=48, facecolors="none",
                           edgecolors="#b03030", lw=0.9, zorder=5)
        S.add_reference_line(ax, met)
        ax.set_ylabel(S.metric_label(met))
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, ncol=len(models), loc="lower left")
    axes[0].set_title(f"(a) Discrimination by ISO week — {TEST_YEAR} forward forecast\n"
                      "gaps = only one class observed that week; red rings = "
                      f"fewer than {MIN_N_WEEK} observations", fontsize=11)
    axes[1].set_title("(b) Brier score by week (lower is better)", fontsize=11)

    # sample size, from any one model on the shared evaluation set
    ax = axes[2]
    ref = d[d.model == models[0]].sort_values("iso_week")
    ax.bar(ref.iso_week, ref.n, width=0.85, color="0.55", linewidth=0)
    ax.axhline(MIN_N_WEEK, color="#b03030", ls=":", lw=1.0)
    ax.set_ylabel("n"); ax.set_xlabel("ISO week"); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"(c) Observations per week ({S.model_label(models[0])} "
                 "evaluation set)", fontsize=10)

    fig.tight_layout()
    S.save(fig, RES / f"nowcast_weekly_skill_{TEST_YEAR}.png")
    return d


# ------------------------------------------------------------------ FIGURE 3
def reliability(y, p, n_bins=N_BINS, min_count=MIN_BIN):
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).dropna()
    if df.empty:
        return None
    try:
        df["bin"] = pd.qcut(df.p, n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df.p, n_bins)
    g = (df.groupby("bin", observed=True)
           .agg(mean_pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")))
    return g[g.n >= min_count]


def fig_reliability(oof: pd.DataFrame):
    models = S.models_in(oof.model.unique())
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 9),
                             gridspec_kw={"height_ratios": [2.6, 1.0]})
    ax, axh = axes
    ax.plot([0, 1], [0, 1], color=S.GREY, ls="--", lw=1.2,
            label="perfect calibration")
    for m in models:
        sub = oof[oof.model == m]
        g = reliability(sub.presence, sub.p)
        if g is None or g.empty:
            continue
        c = S.MODEL_COLOURS.get(m)
        ax.plot(g.mean_pred, g.obs, marker="o", ms=5, lw=1.8, color=c,
                label=S.model_label(m))
        axh.hist(sub.p.dropna(), bins=30, histtype="step", lw=1.5, color=c,
                 density=True)
        # each model's own observed base rate: the flat forecast it must beat
        ax.axhline(float(sub.presence.mean()), color=c, ls=":", lw=0.9, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted suitability")
    ax.set_ylabel("observed presence frequency")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Reliability — {TEST_YEAR} forward forecast\n"
                 "points below the diagonal = over-predicting suitability; "
                 "dotted horizontals = each model's observed base rate\n"
                 f"(quantile bins, \u2265{MIN_BIN} observations each)", fontsize=11)
    axh.set_xlim(0, 1); axh.set_xlabel("predicted suitability")
    axh.set_ylabel("density"); axh.grid(alpha=0.3)
    axh.set_title("where the predictions sit", fontsize=9)
    fig.tight_layout()
    S.save(fig, RES / f"nowcast_reliability_{TEST_YEAR}.png")


def run():
    S.set_thesis_style(constrained=False)
    RES.mkdir(parents=True, exist_ok=True)

    oof = load_oof()
    if oof.empty:
        log("[done] no OOF predictions for the headline fold")
        return

    fig_roc_pr(oof)
    fig_reliability(oof)

    oof_wk = attach_iso_week(oof)
    if oof_wk is None:
        log("[fig2] SKIPPED (see the ISO WEEK note at the top of this file)")
    else:
        fig_weekly(oof_wk)

    log(f"\n[done] nowcast figures -> {RES}")


if __name__ == "__main__":
    run()