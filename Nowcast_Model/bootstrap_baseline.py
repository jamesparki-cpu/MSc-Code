from __future__ import annotations
"""
bootstrap_baselines.py  —  cell-resampled 95% intervals for the REFERENCE
FORECASTS on the nowcast headline fold, plus paired model-vs-baseline
differences. Fills the gap noted in nowcast_compare.py: run_baselines.py
persisted baseline metrics but not their row-level predictions, so the
reference bars carried no whiskers.

WHY BASELINES GET INTERVALS AT ALL
  A baseline forecast is deterministic -- climatology for a given cell-week is a
  fixed number. Its SCORE is not. That score was computed on the 2,454 cell-weeks
  from the 113 cells Florida happened to trap in 2018; had a different set of
  cells been surveyed, the same deterministic forecast would have scored
  differently. The interval quantifies sensitivity to WHICH LOCATIONS were
  evaluated, exactly as it does for the models.

RESAMPLING UNIT
  Grid cells, with all of a drawn cell's weeks kept together. Rows within a cell
  are dependent (same trap, same landscape), so a row bootstrap treats dependent
  observations as independent and produces intervals roughly half as wide.
  Region-year is not usable here: the headline fold is a single test year, so
  between-year variation is not present in the sample and cannot be estimated.

WHAT CAN AND CANNOT BE BOOTSTRAPPED
  climatology, persistence*        all three metrics
  prevalence                       Brier only -- a constant forecast cannot rank,
                                   so ROC and PR-lift are undefined (not zero),
                                   and its BSS against itself is 0 by definition
  Anything on a different row set from the models is flagged, not silently
  compared: paired differences are computed on the intersection only, and the
  intersection size is reported.

INPUTS   baselines_dir/baseline_oof_long.csv     (from run_baselines.py)
         nowcast_results_dir/nowcast_oof_long.csv (from nowcast_run.py)
OUTPUTS  baselines_dir/bootstrap_baselines.csv    marginal intervals, per method
         baselines_dir/bootstrap_model_vs_baseline.csv  paired differences

USAGE    python bootstrap_baselines.py
         python bootstrap_baselines.py --year 2018 --n-boot 2000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)

N_BOOT = 1000
RANDOM_STATE = 42
CONSTANT_FORECASTS = {"prevalence"}      # ROC / PR-lift undefined for these


def log(m):
    print(m, flush=True)


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


# ----- metrics ---------------------------------------------------------------
def _metrics(y, p, constant=False):
    """ROC, PR-lift and BSS-vs-prevalence for one forecast on one row set."""
    prev = y.mean()
    brier = brier_score_loss(y, p)
    ref = brier_score_loss(y, np.full(len(y), prev))
    bss = 1 - brier / ref if ref > 0 else np.nan
    if constant:
        # A constant forecast has no discrimination: ROC is 0.5 by construction
        # and PR-AUC is prevalence. Reporting them as numbers would invite the
        # reader to compare them with the models', so they are left undefined.
        return np.nan, np.nan, bss, brier
    return (roc_auc_score(y, p),
            average_precision_score(y, p) - prev,
            bss, brier)


METRICS = ("roc_auc", "pr_lift", "bss", "brier")


def cell_bootstrap(y, p, cells, constant=False, n_boot=N_BOOT,
                   random_state=RANDOM_STATE):
    """95% percentile intervals, resampling whole cells with replacement.

    Percentile rather than BCa: with ~100 units the bias-correction and
    acceleration terms are themselves noisy, so the plain interval is the
    defensible conservative choice.
    """
    uq = np.unique(cells)
    idx_by = {c: np.flatnonzero(cells == c) for c in uq}
    rng = np.random.default_rng(random_state)

    draws = {k: [] for k in METRICS}
    for _ in range(n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ix = np.concatenate([idx_by[c] for c in pick])
        if len(np.unique(y[ix])) < 2:
            continue                     # a resample with one class is unscorable
        for k, v in zip(METRICS, _metrics(y[ix], p[ix], constant)):
            draws[k].append(v)

    rec = {"n": int(len(y)), "n_cells": int(len(uq)), "n_boot_used": len(draws["bss"])}
    for k, v in zip(METRICS, _metrics(y, p, constant)):
        arr = np.asarray(draws[k], float)
        arr = arr[np.isfinite(arr)]
        rec[k] = v
        rec[f"{k}_lo"] = float(np.percentile(arr, 2.5)) if arr.size else np.nan
        rec[f"{k}_hi"] = float(np.percentile(arr, 97.5)) if arr.size else np.nan
    return rec


def paired_cell_bootstrap(y, p_a, p_b, cells, n_boot=N_BOOT,
                          random_state=RANDOM_STATE):
    """Paired difference (a - b) on identically resampled cells.

    Pairing matters: both forecasts are scored on the SAME resampled cells every
    draw, so the noise they share cancels instead of being counted twice. The
    reported fraction is the proportion of resamples favouring a; it summarises
    resample agreement and is NOT a p-value.
    """
    uq = np.unique(cells)
    idx_by = {c: np.flatnonzero(cells == c) for c in uq}
    rng = np.random.default_rng(random_state)

    def diffs(ix):
        prev = y[ix].mean()
        ba = brier_score_loss(y[ix], p_a[ix])
        bb = brier_score_loss(y[ix], p_b[ix])
        ref = brier_score_loss(y[ix], np.full(len(ix), prev))
        d_bss = ((1 - ba / ref) - (1 - bb / ref)) if ref > 0 else np.nan
        d_brier = bb - ba                        # >0 : a better (lower Brier)
        try:
            d_roc = roc_auc_score(y[ix], p_a[ix]) - roc_auc_score(y[ix], p_b[ix])
        except ValueError:
            d_roc = np.nan
        return d_roc, d_bss, d_brier

    all_ix = np.arange(len(y))
    point = diffs(all_ix)
    keys = ("d_roc_auc", "d_bss", "d_brier")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ix = np.concatenate([idx_by[c] for c in pick])
        if len(np.unique(y[ix])) < 2:
            continue
        for k, v in zip(keys, diffs(ix)):
            draws[k].append(v)

    rec = {"n": int(len(y)), "n_cells": int(len(uq))}
    for k, v in zip(keys, point):
        arr = np.asarray(draws[k], float)
        arr = arr[np.isfinite(arr)]
        rec[k] = v
        rec[f"{k}_lo"] = float(np.percentile(arr, 2.5)) if arr.size else np.nan
        rec[f"{k}_hi"] = float(np.percentile(arr, 97.5)) if arr.size else np.nan
        rec[f"{k}_frac_a_better"] = float((arr > 0).mean()) if arr.size else np.nan
        # "excludes zero" is a descriptive statement about the interval, not a
        # significance test; blocked CV violates the independence assumptions
        # those tests require.
        rec[f"{k}_excludes_zero"] = bool(
            arr.size and (rec[f"{k}_lo"] > 0 or rec[f"{k}_hi"] < 0))
    return rec


# ----- driver ----------------------------------------------------------------
def run(config_path="config.json", year=None, n_boot=N_BOOT):
    cfg = load_config(config_path)
    nowcast_dir = Path(cfg.get("nowcast_dir", "."))
    res = Path(cfg.get("nowcast_results_dir", str(nowcast_dir / "Nowcast_Results")))
    base_dir = Path(cfg.get("baselines_dir", str(res / "Baselines")))

    bl_file = base_dir / "baseline_oof_long.csv"
    md_file = res / "nowcast_oof_long.csv"
    if not bl_file.exists():
        raise SystemExit(f"{bl_file} not found -- re-run run_baselines.py after "
                         "adding the prediction-persistence block.")

    bl = pd.read_csv(bl_file)
    md = pd.read_csv(md_file) if md_file.exists() else pd.DataFrame()

    yr = int(year) if year else int(bl.test_year.max())
    bl = bl[bl.test_year == yr]
    if len(md):
        md = md[md.test_year == yr]
    log(f"[boot] headline fold: test year {yr} | {n_boot} resamples, unit = cell")

    frames = [bl[["Grid_ID", "presence", "p"]].assign(
                  method=bl.baseline, kind="baseline")]
    if len(md):
        frames.append(md[["Grid_ID", "presence", "p"]].assign(
                          method=md.model, kind="model"))
    long = pd.concat(frames, ignore_index=True).dropna(subset=["p"])

    # ---- marginal intervals -------------------------------------------------
    rows = []
    for (method, kind), sub in long.groupby(["method", "kind"], sort=False):
        y = sub.presence.to_numpy(int)
        p = sub.p.to_numpy(float)
        cells = sub.Grid_ID.to_numpy()
        if len(np.unique(y)) < 2:
            log(f"  [skip] {method}: single class")
            continue
        rec = cell_bootstrap(y, p, cells, constant=method in CONSTANT_FORECASTS,
                             n_boot=n_boot)
        rec.update(method=method, kind=kind, test_year=yr)
        rows.append(rec)
        roc = "undefined" if not np.isfinite(rec["roc_auc"]) else \
            f"{rec['roc_auc']:.3f} [{rec['roc_auc_lo']:.3f}, {rec['roc_auc_hi']:.3f}]"
        log(f"  {method:24s} n={rec['n']:5d} cells={rec['n_cells']:4d} | "
            f"ROC {roc} | BSS {rec['bss']:.3f} "
            f"[{rec['bss_lo']:.3f}, {rec['bss_hi']:.3f}]")

    marg = pd.DataFrame(rows)
    cols = ["method", "kind", "test_year", "n", "n_cells", "n_boot_used"] + \
           [c for m in METRICS for c in (m, f"{m}_lo", f"{m}_hi")]
    marg = marg[[c for c in cols if c in marg.columns]]
    base_dir.mkdir(parents=True, exist_ok=True)
    marg.to_csv(base_dir / "bootstrap_baselines.csv", index=False)
    log(f"[boot] wrote bootstrap_baselines.csv ({len(marg)} rows)")

    # ---- paired model vs baseline -------------------------------------------
    if not len(md):
        log("[boot] no model OOF found -- skipping paired comparisons")
        return marg, pd.DataFrame()

    # A paired comparison needs the two forecasts aligned ROW BY ROW. Grid_ID
    # alone is not a row key -- a cell contributes many weeks -- so merging on it
    # produces a cartesian product. Both long files must carry the week.
    key = [c for c in ("iso_week", "week_start") if c in bl.columns and
           (not len(md) or c in md.columns)]
    if not key:
        log("[boot] paired comparisons SKIPPED: baseline_oof_long.csv and "
            "nowcast_oof_long.csv need a shared row key (iso_week or "
            "week_start) in addition to Grid_ID. Marginal intervals are "
            "unaffected.")
        return marg, pd.DataFrame()
    kcols = ["Grid_ID"] + key
    log(f"[boot] pairing on {kcols}")

    mwide = md.pivot_table(index=kcols + ["presence"], columns="model",
                           values="p", aggfunc="first").reset_index()
    bwide = bl.pivot_table(index=kcols + ["presence"], columns="baseline",
                           values="p", aggfunc="first").reset_index()
    both = mwide.merge(bwide, on=kcols + ["presence"], how="inner")

    pairs = []
    models = [m for m in md.model.unique() if m in both.columns]
    bases = [b for b in bl.baseline.unique()
             if b in both.columns and b not in CONSTANT_FORECASTS]
    for m in models:
        for b in bases:
            sub = both[[*kcols, "presence", m, b]].dropna()
            if sub.empty or sub.presence.nunique() < 2:
                continue
            rec = paired_cell_bootstrap(
                sub.presence.to_numpy(int), sub[m].to_numpy(float),
                sub[b].to_numpy(float), sub.Grid_ID.to_numpy(), n_boot=n_boot)
            rec.update(model=m, baseline=b, test_year=yr)
            pairs.append(rec)
            log(f"  {m:22s} vs {b:22s} n={rec['n']:5d} "
                f"dBSS {rec['d_bss']:+.3f} "
                f"[{rec['d_bss_lo']:+.3f}, {rec['d_bss_hi']:+.3f}] "
                f"excl0={rec['d_bss_excludes_zero']}")

    pdf = pd.DataFrame(pairs)
    if len(pdf):
        front = ["model", "baseline", "test_year", "n", "n_cells"]
        pdf = pdf[front + [c for c in pdf.columns if c not in front]]
        pdf.to_csv(base_dir / "bootstrap_model_vs_baseline.csv", index=False)
        log(f"[boot] wrote bootstrap_model_vs_baseline.csv ({len(pdf)} rows)")
    return marg, pdf


def main():
    ap = argparse.ArgumentParser(
        description="Cell-resampled intervals for baselines and nowcast models.")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--year", type=int, default=None,
                    help="headline test year (default: latest present)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    run(config_path=a.config, year=a.year, n_boot=a.n_boot)


if __name__ == "__main__":
    main()