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
         The 2018 held-out validation POOLED ACROSS WEEKS: 355 points scored
         by both variants on identical rows. This is the only pooled comparison
         of the two that needs no caveat.

         Cross-validation and nowcast metrics are deliberately NOT shown. There
         the variants are scored on different sets (21,608 rows at prevalence
         0.776 against 26,667 at 0.628; 2,454 against 3,702), so vanilla's
         higher score is partly a lower baseline rather than a better model, and
         putting those bars beside a matched comparison invites the reader to
         average across the two.

         Per-week numbers rest on 20-102 points each and are noisy, which is why
         the headline is pooled. Panel (d) still shows the per-week differences,
         because a result that holds in every week separately is stronger than
         one that only survives pooling -- but it plots DIFFERENCES rather than
         levels, since the difference is what the matched design licenses.

         Four panels: (a) pooled metrics; (b) pooled ROC curves; (c) pooled
         reliability, which is where a BSS collapse becomes visible; (d) the
         per-week target-group minus vanilla difference.

         Requires validation_2018_cache.py to have run. It prefers
         validation_2018_points.csv, which covers EVERY 2018 cell-week (about
         2,454 rows over 53 weeks). If only the map surfaces are present it
         falls back to joining those, which restricts the comparison to the five
         mapped weeks (355 points) -- still matched, but a seventh of the data.

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

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, roc_curve)

import report_style as S

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR = Path(cfg.get("weekly_xg_dir", "."))
NOWCAST_DIR = Path(cfg.get("nowcast_dir", str(DATA_DIR)))
VALIDATION_DIR = Path(cfg.get("validation_dir", str(NOWCAST_DIR / "Validation_Results")))
SHAP_DIR = Path(cfg.get("shap_dir", str(DATA_DIR / "SHAP_Results")))
MAXENT_DIR = Path(cfg.get("maxent_dir", str(DATA_DIR)))
NOWCAST_RES = Path(cfg.get("nowcast_results_dir", str(NOWCAST_DIR / "Nowcast_Results")))
OUT_DIR = Path(cfg.get("comparison_dir", str(DATA_DIR.parent / "Comparison_Results")))

TG, VAN = "maxent_targetgroup", "maxent_vanilla"
SEASON = {6: "Winter", 20: "Spring", 30: "Summer", 32: "Summer", 43: "Autumn"}
MIN_ABS_FOR_TRUST = 10       # weeks below this are hatched in panel (d)
TEST_LABEL = "2018"

def log(m): print(m, flush=True)


def _find(name, *dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None
# ============================================================================


# ------------------------------------------------------------------- FIGURE A
def matched_points() -> pd.DataFrame:
    """Row-level 2018 predictions for both variants on identical rows.

    Preference order:
      1. validation_2018_points.csv   every 2018 cell-week, all weeks
      2. surface + observations       the five MAPPED weeks only

    Option 1 is not a different analysis, just a larger sample of the same one:
    the map surfaces exist for five weeks because surfaces are expensive, while
    scoring the observed rows is nearly free.
    """
    pts = _find("validation_2018_points.csv", VALIDATION_DIR, NOWCAST_DIR, OUT_DIR)
    if pts is not None:
        g = pd.read_csv(pts)
        cols = [f"prob_{m}" for m in (TG, VAN) if f"prob_{m}" in g.columns]
        if len(cols) == 2:
            g = g.dropna(subset=cols)
            log(f"[figA] {pts.name}: {len(g):,} matched points over "
                f"{g.iso_week.nunique()} weeks "
                f"(prevalence {g.presence.mean():.3f})")
            return g
        log(f"[figA] {pts.name} lacks both variants ({cols}); falling back")

    surf_p = _find("surface_validation_2018.parquet", VALIDATION_DIR, NOWCAST_DIR)
    obs_p = _find("observations_2018.parquet", VALIDATION_DIR, NOWCAST_DIR)
    if surf_p is None or obs_p is None:
        log("[figA] SKIP: run validation_2018_cache.py to write "
            "surface_validation_2018.parquet + observations_2018.parquet")
        return pd.DataFrame()
    surf, obs = pd.read_parquet(surf_p), pd.read_parquet(obs_p)
    cols = [f"prob_{m}" for m in (TG, VAN) if f"prob_{m}" in surf.columns]
    if len(cols) < 2:
        log(f"[figA] SKIP: need both variants in the cache, found {cols}")
        return pd.DataFrame()
    g = (obs.merge(surf[["Grid_ID", "iso_week"] + cols],
                   on=["Grid_ID", "iso_week"], how="inner")
            .dropna(subset=cols))
    log(f"[figA] FALLBACK to map surfaces: {len(g):,} matched points over "
        f"{g.iso_week.nunique()} MAPPED weeks (prevalence "
        f"{g.presence.mean():.3f}). Re-run validation_2018_cache.py with "
        f"POINT_ALL_WEEKS=True for the full-year sample.")
    return g


def _metrics(y, p):
    prev = y.mean()
    return dict(roc_auc=roc_auc_score(y, p),
                pr_lift=average_precision_score(y, p) - prev,
                bss=1 - brier_score_loss(y, p) /
                    brier_score_loss(y, np.full(len(y), prev)))


def fig_matched():
    g = matched_points()
    if g.empty or g.presence.nunique() < 2:
        return
    y = g.presence.to_numpy(int)
    prev = float(y.mean())
    P = {m: g[f"prob_{m}"].to_numpy() for m in (TG, VAN)}
    pooled = {m: _metrics(y, P[m]) for m in (TG, VAN)}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))

    # ---- (a) pooled metrics
    ax = axes[0, 0]
    mets = ["roc_auc", "pr_lift", "bss"]
    x = np.arange(len(mets)); w = 0.38
    for i, m in enumerate((TG, VAN)):
        vals = [pooled[m][k] for k in mets]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=S.model_label(m),
                      color=S.MODEL_COLOURS.get(m), edgecolor=S.BAR_EDGE,
                      linewidth=S.BAR_EDGE_LW)
        S.annotate_bars(ax, bars, vals, "roc_auc", fontsize=8)
    ax.axhline(0, color=S.GREY, lw=1.0)
    ax.axhline(0.5, color=S.GREY, ls="--", lw=1.0)
    ax.annotate("0.5 (chance, ROC only)", (2.45, 0.5), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=7, color=S.GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(["ROC-AUC", "PR-AUC lift", "BSS vs prevalence"], fontsize=9)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"(a) Pooled over {S.fmt(len(y), 'n')} identical points "
                 f"(prevalence {S.fmt(prev, 'prevalence')})", fontsize=11)

    # ---- (b) ROC curves
    ax = axes[0, 1]
    for m in (TG, VAN):
        fpr, tpr, _ = roc_curve(y, P[m])
        ax.plot(fpr, tpr, lw=2, color=S.MODEL_COLOURS.get(m),
                label=f"{S.model_label(m)}  {S.fmt(pooled[m]['roc_auc'], 'roc_auc')}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color=S.GREY, label="chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(b) Pooled ROC \u2014 one baseline, one row set", fontsize=11)

    # ---- (c) reliability
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=S.GREY,
            label="perfect calibration")
    for m in (TG, VAN):
        d = pd.DataFrame({"y": y, "p": P[m]})
        try:
            d["bin"] = pd.qcut(d.p, 8, duplicates="drop")
        except ValueError:
            d["bin"] = pd.cut(d.p, 8)
        b = (d.groupby("bin", observed=True)
               .agg(mp=("p", "mean"), obs=("y", "mean"), n=("y", "size")))
        b = b[b.n >= 15]
        ax.plot(b.mp, b.obs, marker="o", ms=5, lw=1.8,
                color=S.MODEL_COLOURS.get(m), label=S.model_label(m))
    ax.axhline(prev, color=S.GREY, ls=":", lw=1.0)
    ax.annotate(f"observed base rate {S.fmt(prev, 'prevalence')}", (0.02, prev),
                xytext=(0, 4), textcoords="offset points", fontsize=7,
                color=S.GREY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted suitability")
    ax.set_ylabel("observed presence frequency")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(c) Pooled reliability \u2014 below the diagonal is "
                 "over-prediction", fontsize=11)

    # ---- (d) per-week difference
    ax = axes[1, 1]
    weeks, rows = sorted(g.iso_week.unique()), []
    for wk in weeks:
        sub = g[g.iso_week == wk]
        yy = sub.presence.to_numpy(int)
        if len(np.unique(yy)) < 2:
            continue
        a = _metrics(yy, sub[f"prob_{TG}"].to_numpy())
        b = _metrics(yy, sub[f"prob_{VAN}"].to_numpy())
        rows.append(dict(iso_week=int(wk), n=len(yy),
                         n_abs=int((yy == 0).sum()),
                         d_roc=a["roc_auc"] - b["roc_auc"],
                         d_bss=a["bss"] - b["bss"]))
    wk = pd.DataFrame(rows)
    xw = np.arange(len(wk)); w = 0.38
    # Colour by WHICH VARIANT THE DIFFERENCE FAVOURS, using the model palette:
    # above zero the bar is target-group's colour, below it is vanilla's. Grey
    # bars would have been neutral but would also have thrown away the one thing
    # the panel is for -- direction -- and disconnected it from panels (a)-(c).
    # The two metrics are separated by alpha, not by hue, so hue stays free to
    # carry the sign.
    for i, (col, lab, alpha) in enumerate([("d_roc", "\u0394 ROC-AUC", 1.0),
                                           ("d_bss", "\u0394 BSS", 0.55)]):
        cols = [S.MODEL_COLOURS[TG] if v >= 0 else S.MODEL_COLOURS[VAN]
                for v in wk[col]]
        bars = ax.bar(xw + (i - 0.5) * w, wk[col], w, label=lab, color=cols,
                      alpha=alpha, edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW)
        for b_, low in zip(bars, wk.n_abs < MIN_ABS_FOR_TRUST):
            if low:
                b_.set_hatch("//")
    ax.axhline(0, color=S.GREY, lw=1.2)

    # the legend must explain hue AND alpha, so it is built by hand
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(fc=S.MODEL_COLOURS[TG], ec=S.BAR_EDGE, label="\u0394 ROC-AUC"),
        Patch(fc=S.MODEL_COLOURS[TG], ec=S.BAR_EDGE, alpha=0.55, label="\u0394 BSS"),
        Patch(fc=S.MODEL_COLOURS[TG], ec=S.BAR_EDGE,
              label="above 0: target-group ahead"),
        Patch(fc=S.MODEL_COLOURS[VAN], ec=S.BAR_EDGE,
              label="below 0: vanilla ahead"),
    ], fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    ax.set_xticks(xw)
    if len(wk) <= 8:
        ax.set_xticklabels([f"{SEASON.get(r.iso_week, '')}\nwk {r.iso_week}\n"
                            f"n={r.n} ({r.n_abs} abs)" for r in wk.itertuples()],
                           fontsize=7.5)
    else:
        # a full year of weeks: labels every fourth, and n moves to the caption
        ax.set_xticklabels([str(r.iso_week) if k % 4 == 0 else ""
                            for k, r in enumerate(wk.itertuples())], fontsize=7)
        ax.set_xlabel("ISO week")
    ax.set_ylabel("difference: target-group \u2212 vanilla")
    ax.grid(axis="y", alpha=0.3)
    up = int((wk.d_roc > 0).sum())
    thin = int((wk.n_abs < MIN_ABS_FOR_TRUST).sum())
    ax.set_title(f"(d) Per-week difference \u2014 target-group ahead on ROC in "
                 f"{up}/{len(wk)} weeks\nhatched = fewer than "
                 f"{MIN_ABS_FOR_TRUST} observed absences ({thin} weeks)",
                 fontsize=11)

    fig.suptitle("Effort-based absences vs random background, on identical "
                 "observations\n"
                 f"{TEST_LABEL} held-out validation; cross-validation metrics are "
                 "excluded because there the two variants are scored on "
                 "different rows at different prevalence", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    S.save(fig, OUT_DIR / "maxent_background_matched.png")

    out = pd.DataFrame([dict(scope="pooled", iso_week="all", n=len(y),
                             prevalence=round(prev, 3), model=m, **pooled[m])
                        for m in (TG, VAN)])
    out = pd.concat([out, wk.assign(scope="per_week")], ignore_index=True)
    out.round(4).to_csv(OUT_DIR / "maxent_background_matched.csv", index=False)
    log(out.round(3).to_string(index=False))


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