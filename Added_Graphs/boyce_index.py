from __future__ import annotations
"""
boyce_index.py  —  Continuous Boyce Index: a presence-only evaluation that
privileges neither absence convention.

WHY THIS EXISTS
  Target-group and vanilla MaxEnt differ only in what counts as an absence, so
  every absence-based metric favours whichever model was trained on that
  definition:

    * scored against random background, vanilla wins (nowcast ROC 0.914)
    * scored against observed trap absences, target-group wins (matched 0.821)

  Both are real results, but neither is neutral -- each test agrees with one
  model's training distribution. The Boyce index sidesteps this because it uses
  NO ABSENCES AT ALL. It asks only whether presences fall in high-suitability
  environment more often than the landscape makes available.

HOW IT WORKS (Hirzel et al. 2006, Ecological Modelling)
  Slide a window of width W across the suitability range in steps of STEP. For
  each window i:

      F_i = (presences in window / all presences)
            ---------------------------------------
            (available cells in window / all available cells)

  F_i is a predicted-to-expected ratio: 1 means presences fall there exactly as
  often as the landscape offers, above 1 means they are concentrated there. A
  good model has F increasing monotonically with suitability, so the index is
  the Spearman correlation between F_i and the window midpoint, on [-1, 1].

PRESENCE-ONLY, AND WHAT THAT MEANS HERE
  Only rows with presence == 1 are used. Every observed absence is discarded, so
  neither model's absence convention enters the calculation and both are scored
  on identical presences against identical availability.

WHICH AVAILABILITY? THE CHOICE MATTERS
  Boyce compares where presences fall against what the landscape OFFERS, so the
  answer depends on what counts as on offer:

    "state"     every modelled grid cell. The textbook definition -- but the
                presences come from ~113 trapped cells while availability spans
                the whole state, so the two are drawn from different domains. A
                model that simply predicts high suitability inside the trapped
                footprint scores well whether or not it has learned any ecology.
                That is the sampling-bias confound target-group correction
                exists to address, so leaving it in place undercuts the point.

    "surveyed"  availability restricted to the surveyed footprint (keep_masked).
                Presences and availability now come from the same domain, and
                the index measures ranking WITHIN the region the models were
                actually trained on.

    "both"      compute both and plot them side by side (default). If the
                ranking is the same under each, it does not depend on the
                availability definition, which is a stronger statement than
                either alone.

WHAT IT DOES NOT DO
  It cannot detect over-prediction of suitability in unsampled areas: a model
  that calls the whole state suitable still scores well if presences fall in its
  highest band. So it complements the matched comparison rather than replacing
  it, and the two should be reported together.

  It is also computed against MODELLED availability, so it inherits the sampling
  footprint: availability is the modelled grid, not true environmental
  availability across the species' range.

INPUTS (from validation_2018_cache.py)
  surface_validation_2018.parquet   availability: every grid cell, per week
  observations_2018.parquet         presences: observed trap catches
  (validation_2018_points.csv is NOT used -- it holds no availability sample)

OUTPUT -> comparison_dir
  boyce_index.png    F-ratio curves, the index with bootstrap CIs, per-week spread
  boyce_index.csv    index and CI per model, plus the per-week values
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import report_style as S

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR = Path(cfg.get("weekly_xg_dir", "."))
NOWCAST_DIR = Path(cfg.get("nowcast_dir", str(DATA_DIR)))
VALIDATION_DIR = Path(cfg.get("validation_dir", str(NOWCAST_DIR / "Validation_Results")))
OUT_DIR = Path(cfg.get("comparison_dir", str(DATA_DIR.parent / "Comparison_Results")))

# Which models appear. None = every model in the surface.
#   ["maxent_targetgroup", "maxent_vanilla"]  restricts to the absence-convention
#   contrast, which is what this metric was added to settle. The tree models are
#   worth showing once, but they were trained with effort weighting on
#   trap-restricted rows and their output is compressed into a narrow high band,
#   so a low Boyce score for them reflects that compression as much as poor
#   ranking -- a caveat that does not apply to the MaxEnt pair.
MODELS_TO_PLOT = ["maxent_targetgroup", "maxent_vanilla"] 

# "state" | "surveyed" | "both"  -- see the docstring
AVAILABILITY = "both"
PRIMARY_AVAIL = "surveyed"      # which one drives panels (a) and (c)

WINDOW = 0.20        # window width, as a fraction of the suitability range
STEP = 0.01          # window step
MIN_AVAIL = 30       # windows with fewer available cells are dropped as unstable
N_BOOT = 500         # bootstrap replicates, resampling PRESENCES
MIN_PRES_WEEK = 15   # weeks with fewer presences are excluded from panel (c)
RANDOM_STATE = 42

def log(m): print(m, flush=True)


def _find(name, *dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None
# ============================================================================


def boyce(pres: np.ndarray, avail: np.ndarray, window: float = WINDOW,
          step: float = STEP, min_avail: int = MIN_AVAIL):
    """Continuous Boyce Index. Returns (index, midpoints, F ratios).

    pres  predicted suitability AT PRESENCE LOCATIONS
    avail predicted suitability across the AVAILABLE landscape
    """
    pres = np.asarray(pres, float); pres = pres[np.isfinite(pres)]
    avail = np.asarray(avail, float); avail = avail[np.isfinite(avail)]
    if len(pres) < 10 or len(avail) < 100:
        return np.nan, np.array([]), np.array([])

    lo, hi = float(min(avail.min(), pres.min())), float(max(avail.max(), pres.max()))
    if hi - lo < 1e-9:
        return np.nan, np.array([]), np.array([])
    starts = np.arange(lo, hi - window + 1e-12, step)
    mids, ratios = [], []
    for a in starts:
        b = a + window
        e = int(((avail >= a) & (avail < b)).sum())
        if e < min_avail:
            continue                      # too little landscape here to be stable
        p = int(((pres >= a) & (pres < b)).sum())
        mids.append(a + window / 2)
        ratios.append((p / len(pres)) / (e / len(avail)))
    if len(mids) < 5:
        return np.nan, np.array(mids), np.array(ratios)
    rho, _ = spearmanr(mids, ratios)
    return float(rho), np.asarray(mids), np.asarray(ratios)


def boyce_ci(pres, avail, n_boot=N_BOOT):
    """Percentile CI, resampling PRESENCES only: availability is the landscape,
    which is fixed, not a sample to be resampled."""
    rng = np.random.default_rng(RANDOM_STATE)
    pres = np.asarray(pres, float); pres = pres[np.isfinite(pres)]
    draws = []
    for _ in range(n_boot):
        b, _, _ = boyce(rng.choice(pres, len(pres), replace=True), avail)
        if np.isfinite(b):
            draws.append(b)
    if not draws:
        return np.nan, np.nan
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def run():
    S.set_thesis_style(constrained=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sp = _find("surface_validation_2018.parquet", VALIDATION_DIR, NOWCAST_DIR)
    op = _find("observations_2018.parquet", VALIDATION_DIR, NOWCAST_DIR)
    if sp is None or op is None:
        log("[boyce] SKIP: run validation_2018_cache.py first")
        return
    surf, obs = pd.read_parquet(sp), pd.read_parquet(op)
    models = S.models_in([c.replace("prob_", "") for c in surf.columns
                          if c.startswith("prob_")])
    if MODELS_TO_PLOT is not None:
        dropped = [m for m in models if m not in MODELS_TO_PLOT]
        models = [m for m in models if m in MODELS_TO_PLOT]
        if dropped:
            log(f"[boyce] restricted to {models} (MODELS_TO_PLOT); "
                f"omitted {dropped}")
    if not models:
        log("[boyce] SKIP: no models left to score")
        return

    # PRESENCE-ONLY: absences are dropped here and never used again
    pres_obs = obs[obs.presence == 1]
    log(f"[boyce] {len(obs):,} observations -> {len(pres_obs):,} presences kept, "
        f"{len(obs) - len(pres_obs):,} absences discarded (presence-only metric)")

    avail_sets = {}
    if AVAILABILITY in ("state", "both"):
        avail_sets["state"] = surf
    if AVAILABILITY in ("surveyed", "both"):
        if "keep_masked" in surf.columns:
            avail_sets["surveyed"] = surf[surf.keep_masked.astype(bool)]
        else:
            log("[boyce] no keep_masked column -- 'surveyed' availability skipped")
    for k, v in avail_sets.items():
        log(f"[boyce] availability '{k}': {len(v):,} grid cell-weeks "
            f"({100 * len(v) / len(surf):.0f}% of the modelled grid)")
    primary = PRIMARY_AVAIL if PRIMARY_AVAIL in avail_sets else list(avail_sets)[0]
    joined = pres_obs.merge(
        surf[["Grid_ID", "iso_week"] + [f"prob_{m}" for m in models]],
        on=["Grid_ID", "iso_week"], how="inner")
    log(f"[boyce] {len(joined):,} presences | availability = "
        f"{len(surf):,} grid cell-weeks | models {models}")

    rows, curves, weekly = [], {}, []
    for m in models:
        col = f"prob_{m}"
        for aname, adf in avail_sets.items():
            b, mids, F = boyce(joined[col].to_numpy(), adf[col].to_numpy())
            lo, hi = boyce_ci(joined[col].to_numpy(), adf[col].to_numpy())
            rows.append(dict(model=m, availability=aname, scope="pooled",
                             boyce=round(b, 3), lo=round(lo, 3), hi=round(hi, 3),
                             n_presences=len(joined), n_available=len(adf)))
            log(f"[boyce] {m:22s} [{aname:8s}] B = {b:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            if aname == primary:
                curves[m] = (mids, F)

        for wk, g in joined.groupby("iso_week"):
            if len(g) < MIN_PRES_WEEK:
                continue
            a = avail_sets[primary]
            bw, _, _ = boyce(g[col].to_numpy(),
                             a[a.iso_week == wk][col].to_numpy())
            weekly.append(dict(model=m, iso_week=int(wk), boyce=bw,
                               n_presences=len(g), availability=primary))

    wk = pd.DataFrame(weekly)
    pd.concat([pd.DataFrame(rows),
               wk.assign(scope="per_week")], ignore_index=True
              ).round(4).to_csv(OUT_DIR / "boyce_index.csv", index=False)

    # ------------------------------------------------------------- figure
    ncol = 3 if len(wk) else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.3 * ncol, 5.4))

    ax = axes[0]
    for m in models:
        mids, F = curves[m]
        if len(mids):
            ax.plot(mids, F, lw=2, color=S.MODEL_COLOURS.get(m),
                    label=S.model_label(m))
    ax.axhline(1.0, color=S.GREY, ls="--", lw=1.1)
    ax.annotate("F = 1: presences as common as the landscape offers",
                (0.02, 1.0), xytext=(0, 5), textcoords="offset points",
                fontsize=7, color=S.GREY)
    ax.set_xlabel("predicted suitability (window midpoint)")
    ax.set_ylabel("predicted-to-expected ratio F")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"(a) Are presences concentrated where suitability is high?\n"
                 f"availability: {primary}", fontsize=11)

    ax = axes[1]
    R = pd.DataFrame(rows)
    anames = list(avail_sets)
    x = np.arange(len(models)); w = 0.8 / max(len(anames), 1)
    for i, aname in enumerate(anames):
        sub = R[R.availability == aname].set_index("model")
        vals = [float(sub.loc[m, "boyce"]) for m in models]
        los = [float(sub.loc[m, "lo"]) for m in models]
        his = [float(sub.loc[m, "hi"]) for m in models]
        err = np.abs(np.vstack([np.array(vals) - np.array(los),
                                np.array(his) - np.array(vals)]))
        # hatch the statewide bars: their availability spans regions no trap has
        # ever occupied, so presences and availability are not drawn alike
        ax.bar(x + (i - (len(anames) - 1) / 2) * w, vals, w,
               color=[S.MODEL_COLOURS.get(m) for m in models],
               alpha=1.0 if aname == "surveyed" else 0.55,
               hatch="" if aname == "surveyed" else "//",
               edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_LW,
               yerr=err, capsize=2.5, error_kw=dict(lw=0.8, ecolor="0.2"),
               label=f"availability: {aname}")
    ax.axhline(0, color=S.GREY, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([S.model_label(m, wrapped=True) for m in models],
                       fontsize=8, rotation=20, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Continuous Boyce Index")
    ax.grid(axis="y", alpha=0.3)
    if len(anames) > 1:
        ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(b) Index with 95% CI (bootstrap over presences)\n"
                 "hatched = statewide availability, unhatched = surveyed only",
                 fontsize=10)

    if len(wk):
        ax = axes[2]
        for m in models:
            d = wk[wk.model == m].sort_values("iso_week")
            if len(d):
                ax.plot(d.iso_week, d.boyce, marker="o", ms=3.5, lw=1.6,
                        color=S.MODEL_COLOURS.get(m), label=S.model_label(m))
        ax.axhline(0, color=S.GREY, lw=1.1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("ISO week"); ax.set_ylabel("Continuous Boyce Index")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"(c) By week, availability: {primary}\n"
                     f"(weeks with at least {MIN_PRES_WEEK} presences)",
                     fontsize=11)

    only_maxent = all(m.startswith("maxent") for m in models)
    who = "Both MaxEnt variants are" if only_maxent else "All models are"
    fig.suptitle("Continuous Boyce Index \u2014 a presence-only evaluation.\n"
                 f"{who} scored on the SAME presences against the SAME "
                 "availability, so neither absence convention is favoured.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    S.save(fig, OUT_DIR / "boyce_index.png")

    log("\n[read] the Boyce index uses no absences, so it does not reward a model "
        "for having been trained on the same kind of negative it is tested "
        "against. It cannot, however, penalise over-prediction in unsampled "
        "areas -- report it alongside the matched comparison, not instead of it.")


if __name__ == "__main__":
    run()