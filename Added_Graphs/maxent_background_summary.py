from __future__ import annotations
"""
maxent_background_summary.py  —  the three-panel MaxEnt background figure.

Draws the strongest panels from maxent_background_compare.py and boyce_index.py
into one figure, so the whole argument sits on a single page:

  (a) POOLED, MATCHED         ROC, PR-lift and BSS over every 2018 cell-week,
                              both variants on identical rows at one prevalence.
                              The result.

  (b) PER-WEEK DIFFERENCE     target-group minus vanilla, week by week. Shows
                              the result is not an artefact of pooling. Weeks
                              with fewer than MIN_ABS_FOR_TRUST observed
                              absences are hatched.

  (c) CONTINUOUS BOYCE INDEX  a presence-only metric that uses NO absences, so
                              it cannot reward a model for having been trained
                              on the kind of negative it is tested against. The
                              neutrality check.

WHY (c) IS VISUALLY SEPARATED
  It is on a different scale (-1 to +1) and answers a different question from
  (a) and (b). Sharing an axis or a visual grouping with them would invite the
  reader to compare 0.94 against 0.821 as though they were the same kind of
  quantity. It therefore sits behind a vertical rule, on its own scale, with its
  own caption line.

READING ORDER
  (a) the result -> (b) it holds week by week -> (c) it is not an artefact of a
  test that suits target-group. Each panel answers the objection the previous
  one raises.

INPUTS
  validation_2018_points.csv        (preferred) every 2018 cell-week
  surface_validation_2018.parquet   availability + fallback presences
  observations_2018.parquet         presences
  all written by validation_2018_cache.py

OUTPUT -> comparison_dir
  maxent_background_summary.png
  maxent_background_summary.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)

import report_style as S
from boyce_index import boyce, boyce_ci

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR = Path(cfg.get("weekly_xg_dir", "."))
NOWCAST_DIR = Path(cfg.get("nowcast_dir", str(DATA_DIR)))
VALIDATION_DIR = Path(cfg.get("validation_dir", str(NOWCAST_DIR / "Validation_Results")))
OUT_DIR = Path(cfg.get("comparison_dir", str(DATA_DIR.parent / "Comparison_Results")))

TG, VAN = "maxent_targetgroup", "maxent_vanilla"
MIN_ABS_FOR_TRUST = 10
AVAILABILITY = "surveyed"     # "surveyed" | "state"; see boyce_index.py
TEST_YEAR = 2018

def log(m): print(m, flush=True)


def _find(name, *dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def _metrics(y, p):
    prev = y.mean()
    return dict(roc_auc=roc_auc_score(y, p),
                pr_lift=average_precision_score(y, p) - prev,
                bss=1 - brier_score_loss(y, p) /
                    brier_score_loss(y, np.full(len(y), prev)))
# ============================================================================


def load_points() -> pd.DataFrame:
    pts = _find("validation_2018_points.csv", VALIDATION_DIR, NOWCAST_DIR, OUT_DIR)
    if pts is not None:
        g = pd.read_csv(pts)
        cols = [f"prob_{m}" for m in (TG, VAN) if f"prob_{m}" in g.columns]
        if len(cols) == 2:
            return g.dropna(subset=cols)
    sp = _find("surface_validation_2018.parquet", VALIDATION_DIR, NOWCAST_DIR)
    op = _find("observations_2018.parquet", VALIDATION_DIR, NOWCAST_DIR)
    if sp is None or op is None:
        return pd.DataFrame()
    surf, obs = pd.read_parquet(sp), pd.read_parquet(op)
    cols = [f"prob_{m}" for m in (TG, VAN)]
    return (obs.merge(surf[["Grid_ID", "iso_week"] + cols],
                      on=["Grid_ID", "iso_week"], how="inner").dropna(subset=cols))


def load_availability():
    """Surface + presences joined to it, for the Boyce panel."""
    sp = _find("surface_validation_2018.parquet", VALIDATION_DIR, NOWCAST_DIR)
    op = _find("observations_2018.parquet", VALIDATION_DIR, NOWCAST_DIR)
    if sp is None or op is None:
        return None, None
    surf, obs = pd.read_parquet(sp), pd.read_parquet(op)
    if AVAILABILITY == "surveyed" and "keep_masked" in surf.columns:
        surf = surf[surf.keep_masked.astype(bool)]
    pres = (obs[obs.presence == 1]
            .merge(surf[["Grid_ID", "iso_week"] + [f"prob_{m}" for m in (TG, VAN)]],
                   on=["Grid_ID", "iso_week"], how="inner"))
    return surf, pres


def run():
    S.set_thesis_style(constrained=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    g = load_points()
    if g.empty or g.presence.nunique() < 2:
        log("[summary] SKIP: run validation_2018_cache.py first")
        return
    y = g.presence.to_numpy(int)
    prev = float(y.mean())
    pooled = {m: _metrics(y, g[f"prob_{m}"].to_numpy()) for m in (TG, VAN)}
    log(f"[summary] pooled over {len(y):,} matched points (prevalence {prev:.3f})")

    # 1 x 3, with panel (c) pushed right and separated by a rule
    fig = plt.figure(figsize=(16.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 0.85],
                          left=0.05, right=0.985, top=0.80, bottom=0.20,
                          wspace=0.30)
    axA, axB, axC = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    # ---------------------------------------------------------- (a) pooled
    mets = ["roc_auc", "pr_lift", "bss"]
    x = np.arange(len(mets)); w = 0.38
    for i, m in enumerate((TG, VAN)):
        vals = [pooled[m][k] for k in mets]
        bars = axA.bar(x + (i - 0.5) * w, vals, w, label=S.model_label(m),
                       color=S.MODEL_COLOURS.get(m), edgecolor=S.BAR_EDGE,
                       linewidth=S.BAR_EDGE_LW)
        S.annotate_bars(axA, bars, vals, "roc_auc", fontsize=8)
    axA.axhline(0, color=S.GREY, lw=1.0)
    axA.axhline(0.5, color=S.GREY, ls="--", lw=1.0)
    axA.annotate("0.5 chance (ROC only)", (2.45, 0.5), xytext=(0, 4),
                 textcoords="offset points", ha="right", fontsize=7, color=S.GREY)
    axA.set_xticks(x)
    axA.set_xticklabels(["ROC-AUC", "PR-AUC lift", "BSS vs\nprevalence"], fontsize=9)
    axA.grid(axis="y", alpha=0.3); axA.legend(fontsize=8, loc="upper right")
    axA.set_title(f"(a) Pooled over {S.fmt(len(y), 'n')} identical points\n"
                  f"prevalence {S.fmt(prev, 'prevalence')}", fontsize=11)

    # ------------------------------------------------------- (b) per week
    rows = []
    for wk, sub in g.groupby("iso_week"):
        yy = sub.presence.to_numpy(int)
        if len(np.unique(yy)) < 2:
            continue
        a = _metrics(yy, sub[f"prob_{TG}"].to_numpy())
        b = _metrics(yy, sub[f"prob_{VAN}"].to_numpy())
        rows.append(dict(iso_week=int(wk), n=len(yy), n_abs=int((yy == 0).sum()),
                         d_roc=a["roc_auc"] - b["roc_auc"],
                         d_bss=a["bss"] - b["bss"]))
    wk = pd.DataFrame(rows)
    xw = np.arange(len(wk)); w = 0.40
    for i, (col, alpha) in enumerate([("d_roc", 1.0), ("d_bss", 0.55)]):
        cols = [S.MODEL_COLOURS[TG] if v >= 0 else S.MODEL_COLOURS[VAN]
                for v in wk[col]]
        bars = axB.bar(xw + (i - 0.5) * w, wk[col], w, color=cols, alpha=alpha,
                       edgecolor=S.BAR_EDGE, linewidth=0.3)
        for b_, low in zip(bars, wk.n_abs < MIN_ABS_FOR_TRUST):
            if low:
                b_.set_hatch("//")
    axB.axhline(0, color=S.GREY, lw=1.2)
    axB.set_xticks(xw)
    axB.set_xticklabels([str(r.iso_week) if k % 4 == 0 else ""
                         for k, r in enumerate(wk.itertuples())], fontsize=7)
    axB.set_xlabel("ISO week")
    axB.set_ylabel("difference: target-group \u2212 vanilla")
    axB.grid(axis="y", alpha=0.3)
    axB.legend(handles=[
        Patch(fc=S.MODEL_COLOURS[TG], ec=S.BAR_EDGE, label="\u0394 ROC-AUC"),
        Patch(fc=S.MODEL_COLOURS[TG], ec=S.BAR_EDGE, alpha=0.55, label="\u0394 BSS"),
        Patch(fc=S.MODEL_COLOURS[VAN], ec=S.BAR_EDGE, label="below 0: vanilla ahead"),
        Patch(fc="white", ec=S.BAR_EDGE, hatch="//",
              label=f"<{MIN_ABS_FOR_TRUST} absences"),
    ], fontsize=7, ncol=2, loc="upper left", framealpha=0.9)
    up = int((wk.d_roc > 0).sum())
    axB.set_title(f"(b) Per-week difference \u2014 target-group ahead on ROC\n"
                  f"in {up} of {len(wk)} weeks", fontsize=11)

    # ------------------------------------------------------------ (c) Boyce
    surf, pres = load_availability()
    boy = {}
    if surf is not None and pres is not None and len(pres) > 20:
        for m in (TG, VAN):
            col = f"prob_{m}"
            bidx, _, _ = boyce(pres[col].to_numpy(), surf[col].to_numpy())
            lo, hi = boyce_ci(pres[col].to_numpy(), surf[col].to_numpy())
            boy[m] = dict(boyce=bidx, lo=lo, hi=hi)
            log(f"[summary] Boyce {m:22s} {bidx:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        xb = np.arange(2)
        vals = [boy[m]["boyce"] for m in (TG, VAN)]
        err = np.abs(np.vstack([[vals[i] - boy[m]["lo"] for i, m in enumerate((TG, VAN))],
                                [boy[m]["hi"] - vals[i] for i, m in enumerate((TG, VAN))]]))
        axC.bar(xb, vals, 0.55, color=[S.MODEL_COLOURS[TG], S.MODEL_COLOURS[VAN]],
                edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW, yerr=err,
                capsize=3, error_kw=dict(lw=0.9, ecolor="0.2"))
        for k, v in enumerate(vals):
            axC.text(k, v + 0.04, S.fmt(v, "roc_auc"), ha="center", fontsize=8)
        axC.set_xticks(xb)
        axC.set_xticklabels([S.model_label(m, wrapped=True) for m in (TG, VAN)],
                            fontsize=8)
        axC.set_ylim(-0.2, 1.28)
        axC.axhline(0, color=S.GREY, lw=1.2)
        axC.set_ylabel("Continuous Boyce Index")
        axC.grid(axis="y", alpha=0.3)
        axC.set_title(f"(c) Presence-only check \u2014 NO absences used\n"
                      f"availability: {AVAILABILITY}; {len(pres):,} presences",
                      fontsize=11)
    else:
        axC.axis("off")
        axC.text(0.5, 0.5, "Boyce panel needs\nsurface_validation_2018.parquet",
                 ha="center", va="center", fontsize=9, color="#b03030")

    # the separating rule: (c) is on a different scale and answers a different
    # question, so it must not read as a third bar of the same comparison
    xr = (axB.get_position().x1 + axC.get_position().x0) / 2
    fig.add_artist(Line2D([xr, xr], [0.20, 0.84], color="0.6", lw=1.1,
                          ls=(0, (5, 4)), transform=fig.transFigure))
    fig.text(xr, 0.155, "new scale",
             ha="center", va="top", fontsize=7.5, color="0.45",
             linespacing=1.4)

    fig.suptitle("Effort-based absences vs random background.  (a) and (b) score "
                 "both variants on identical observations; (c) uses no absences "
                 "at all,\nso it cannot reward a model for being tested against "
                 "the kind of negative it was trained on.", fontsize=11)
    S.save(fig, OUT_DIR / "maxent_background_summary.png")

    out = pd.DataFrame([dict(panel="a_pooled", model=m, n=len(y),
                             prevalence=round(prev, 3), **pooled[m])
                        for m in (TG, VAN)])
    if boy:
        out = pd.concat([out, pd.DataFrame(
            [dict(panel="c_boyce", model=m, availability=AVAILABILITY, **boy[m])
             for m in (TG, VAN)])], ignore_index=True)
    out = pd.concat([out, wk.assign(panel="b_per_week")], ignore_index=True)
    out.round(4).to_csv(OUT_DIR / "maxent_background_summary.csv", index=False)


if __name__ == "__main__":
    run()