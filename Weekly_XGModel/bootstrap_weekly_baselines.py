from __future__ import annotations
"""
bootstrap_weekly_baselines.py  —  the weekly (blocked-CV) counterpart of
bootstrap_baselines.py. Produces cell-block-resampled 95% intervals for the
REFERENCE FORECASTS under each blocked scheme, plus paired model-vs-baseline
differences, so the r3 figure can carry whiskers on every bar.

RESAMPLING UNIT = THE SCHEME'S OWN CV GROUP
  Unlike the nowcast (one test year, so cells are the only usable unit), the
  blocked schemes have a natural resampling unit: the group that defined the
  fold. Resampling that unit reproduces the dependence the scheme was built
  around.
      spatial          -> spatial_block            (6 units)
      temporal         -> iso_year                 (6 units)
      spatiotemporal   -> (spatial_block, iso_year) (36 units)
  Row resampling is not used: rows within a group are dependent, and treating
  them as independent gives intervals that are far too narrow.

  DISCLOSE THE SMALL-GROUP PROBLEM: with six units, a resample draws six with
  replacement, so many draws duplicate or omit whole blocks. Spatial intervals
  are coarse and should be read as indicative, not precise. n_groups is written
  to the output so this is visible in the table.

PREVALENCE IS EXCLUDED
  It is the denominator of the BSS panel, so its bar is 0 by construction and it
  has no ROC or PR-lift. Its value belongs in the axis label, not as a bar.

INPUTS   baselines_dir/baseline_oof_weekly_long.csv  (from run_baselines.py)
         weekly_xg_dir/oof_long.csv                  (model OOF, all schemes)
OUTPUTS  baselines_dir/bootstrap_weekly_baselines.csv        marginal intervals
         baselines_dir/bootstrap_weekly_model_vs_baseline.csv paired differences

CONSISTENCY WITH bootstrap_marginal.csv
  The models already have intervals from bootstrap_uncertainty.py, and the r2
  figure plots those. To avoid two figures showing different whiskers for the
  same number, the r3 figure should also read MODEL intervals from
  bootstrap_marginal.csv and take only the BASELINE intervals from here.

  This script still recomputes the model intervals, purely to CHECK that its
  resampling unit matches: on completion it compares its own model bounds
  against bootstrap_marginal.csv and reports the largest disagreement. A large
  difference means the two scripts group differently and the baseline intervals
  are not on the same footing as the models' -- fix before plotting them
  together.

USAGE    python bootstrap_weekly_baselines.py
         python bootstrap_weekly_baselines.py --schemes spatiotemporal spatial
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
DROP_BASELINES = {"prevalence"}          # denominator of the BSS panel, not a bar
METRICS = ("roc_auc", "pr_lift", "bss", "brier")

# The unit resampled for each scheme; must match how the folds were formed.
GROUP_COLS = {
    "spatial":                ["spatial_block"],
    "temporal":               ["iso_year"],
    "spatiotemporal":         ["spatial_block", "iso_year"],
    "spatiotemporal_3blocks": ["spatial_block", "iso_year"],
}


def log(m):
    print(m, flush=True)


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def _group_key(df, scheme):
    """Single string label per resampling unit for `scheme`."""
    cols = [c for c in GROUP_COLS.get(scheme, ["spatial_block", "iso_year"])
            if c in df.columns]
    if not cols:
        return df["Grid_ID"].to_numpy()          # last-resort fallback
    return df[cols].astype(str).agg("|".join, axis=1).to_numpy()


def _metrics(y, p):
    prev = y.mean()
    brier = brier_score_loss(y, p)
    ref = brier_score_loss(y, np.full(len(y), prev))
    return (roc_auc_score(y, p),
            average_precision_score(y, p) - prev,
            1 - brier / ref if ref > 0 else np.nan,
            brier)

def _find(name, *dirs):
    """First existing copy of `name`, searching in the order given.

    Mirrors results_r2_figures.py: oof_long.csv is written to comparison_dir,
    not weekly_xg_dir, so that directory must be searched first.
    """
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def group_bootstrap(y, p, groups, n_boot=N_BOOT, random_state=RANDOM_STATE):
    """95% percentile intervals, resampling whole CV groups with replacement."""
    uq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uq}
    rng = np.random.default_rng(random_state)

    draws = {k: [] for k in METRICS}
    for _ in range(n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ix = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y[ix])) < 2:
            continue
        for k, v in zip(METRICS, _metrics(y[ix], p[ix])):
            draws[k].append(v)

    rec = {"n": int(len(y)), "n_groups": int(len(uq)),
           "n_boot_used": len(draws["bss"])}
    for k, v in zip(METRICS, _metrics(y, p)):
        arr = np.asarray(draws[k], float)
        arr = arr[np.isfinite(arr)]
        rec[k] = v
        rec[f"{k}_lo"] = float(np.percentile(arr, 2.5)) if arr.size else np.nan
        rec[f"{k}_hi"] = float(np.percentile(arr, 97.5)) if arr.size else np.nan
    return rec


def paired_group_bootstrap(y, p_a, p_b, groups, n_boot=N_BOOT,
                           random_state=RANDOM_STATE):
    """Paired difference (a - b) on identically resampled groups.

    Includes d_bss_vs_b: the BSS of a REFERENCED TO b, which is what the
    'BSS vs climatology' panel plots. Its interval comes free from the same
    resampling, so that panel can carry whiskers too.
    """
    uq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uq}
    rng = np.random.default_rng(random_state)
    keys = ("d_roc_auc", "d_bss", "bss_vs_b")

    def diffs(ix):
        prev = y[ix].mean()
        ba = brier_score_loss(y[ix], p_a[ix])
        bb = brier_score_loss(y[ix], p_b[ix])
        ref = brier_score_loss(y[ix], np.full(len(ix), prev))
        d_bss = ((1 - ba / ref) - (1 - bb / ref)) if ref > 0 else np.nan
        try:
            d_roc = roc_auc_score(y[ix], p_a[ix]) - roc_auc_score(y[ix], p_b[ix])
        except ValueError:
            d_roc = np.nan
        return d_roc, d_bss, (1 - ba / bb if bb > 0 else np.nan)

    point = diffs(np.arange(len(y)))
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ix = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y[ix])) < 2:
            continue
        for k, v in zip(keys, diffs(ix)):
            draws[k].append(v)

    rec = {"n": int(len(y)), "n_groups": int(len(uq))}
    for k, v in zip(keys, point):
        arr = np.asarray(draws[k], float)
        arr = arr[np.isfinite(arr)]
        rec[k] = v
        rec[f"{k}_lo"] = float(np.percentile(arr, 2.5)) if arr.size else np.nan
        rec[f"{k}_hi"] = float(np.percentile(arr, 97.5)) if arr.size else np.nan
        rec[f"{k}_frac_a_better"] = float((arr > 0).mean()) if arr.size else np.nan
        # descriptive statement about the interval, NOT a significance test
        rec[f"{k}_excludes_zero"] = bool(
            arr.size and (rec[f"{k}_lo"] > 0 or rec[f"{k}_hi"] < 0))
    return rec


def _check_against_marginal(marg, *dirs):
    """Compare this script's MODEL bounds against bootstrap_uncertainty.py's.

    Both should resample the same unit. A material disagreement means the two
    group differently, in which case the baseline intervals produced here are
    not on the same footing as the model intervals the r2 figure plots, and the
    two must not be drawn side by side until reconciled.
    """
    f = None
    for d in dirs:
        cand = Path(d) / "bootstrap_marginal.csv"
        if cand.exists():
            f = cand
            break
    if f is None or marg.empty:
        log("[check] bootstrap_marginal.csv not found in "
            + ", ".join(str(d) for d in dirs)
            + " -- cannot verify that the baseline intervals use the same "
              "resampling unit as the models'")
        return
    log(f"[check] comparing against {f}")
    ref = pd.read_csv(f)
    mine = marg[marg.kind == "model"]
    rows = []
    for r in mine.itertuples():
        m = ref[(ref.model == r.method) & (ref.scheme == r.scheme)]
        if not len(m):
            continue
        for met in ("roc_auc", "bss"):
            for side in ("lo", "hi"):
                a = getattr(r, f"{met}_{side}", np.nan)
                col = f"{met}_{side}"
                if col not in m.columns:
                    continue
                b = float(m[col].iloc[0])
                if np.isfinite(a) and np.isfinite(b):
                    rows.append({"model": r.method, "scheme": r.scheme,
                                 "bound": col, "here": a, "marginal": b,
                                 "diff": abs(a - b)})
    if not rows:
        log("[check] no shared (model, scheme) cells to compare")
        return
    rep = pd.DataFrame(rows)
    worst = rep.loc[rep["diff"].idxmax()]
    log(f"[check] model bounds vs bootstrap_marginal.csv: {len(rep)} compared, "
        f"max diff {worst['diff']:.4f} "
        f"({worst['model']} / {worst['scheme']} / {worst['bound']})")
    if worst["diff"] > 0.02:
        log("[check] *** LARGE DISAGREEMENT *** the two scripts are almost "
            "certainly resampling different units. Do NOT plot these baseline "
            "intervals alongside bootstrap_marginal.csv model intervals until "
            "GROUP_COLS here matches bootstrap_uncertainty.group_key().")
    return rep


def run(config_path="config.json", schemes=None, n_boot=N_BOOT):
    cfg = load_config(config_path)
    data_dir = Path(cfg.get("weekly_xg_dir", "."))
    res = Path(cfg.get("nowcast_results_dir",
                       str(Path(cfg.get("nowcast_dir", str(data_dir))) / "Nowcast_Results")))
    base_dir = Path(cfg.get("baselines_dir", str(res / "Baselines")))
    # bootstrap_marginal.csv is written to comparison_dir; resolve it exactly as
    # results_r2_figures.py and results_r3_figures.py do, or the consistency
    # check silently reports "not found" and the whiskers go unverified.
    comp_dir = Path(cfg.get("comparison_dir",
                            str(data_dir.parent / "Comparison_Results")))

    bl_file = base_dir / "baseline_oof_weekly_long.csv"
    md_file = _find("oof_long.csv", comp_dir, data_dir, res, base_dir)
    bl_file = base_dir / "baseline_oof_weekly_long.csv"
    if not bl_file.exists():
        raise SystemExit(f"{bl_file} not found -- add the persistence block to "
                         "weekly_baselines() in run_baselines.py and re-run it.")
    bl = pd.read_csv(bl_file)
    if md_file is None:
        log("[boot] oof_long.csv not found -- baselines only, no paired "
            "comparisons and no consistency check")
        md = pd.DataFrame()
    else:
        log(f"[boot] model OOF from {md_file}")
        md = pd.read_csv(md_file)

    want = schemes or sorted(bl.scheme.unique())
    log(f"[boot] schemes {want} | {n_boot} resamples | unit = scheme CV group")

    marg_rows, pair_rows = [], []
    for scheme in want:
        b = bl[bl.scheme == scheme]
        m = md[md.scheme == scheme] if len(md) else pd.DataFrame()
        if b.empty:
            log(f"  [skip] {scheme}: no baseline rows")
            continue

        frames = [b[["Grid_ID", "iso_year", "presence", "p"]].assign(
                      method=b.baseline, kind="baseline",
                      spatial_block=b.get("spatial_block"))]
        if len(m):
            frames.append(m[["Grid_ID", "iso_year", "spatial_block",
                             "presence", "p"]].assign(method=m.model, kind="model"))
        long = pd.concat(frames, ignore_index=True).dropna(subset=["p"])

        # ---- marginal ----
        for (method, kind), sub in long.groupby(["method", "kind"], sort=False):
            y = sub.presence.to_numpy(int)
            if len(np.unique(y)) < 2:
                continue
            g = _group_key(sub, scheme)
            rec = group_bootstrap(y, sub.p.to_numpy(float), g, n_boot=n_boot)
            rec.update(method=method, kind=kind, scheme=scheme)
            marg_rows.append(rec)
            log(f"  {scheme:16s} {method:22s} n={rec['n']:6d} groups={rec['n_groups']:3d}"
                f" | ROC {rec['roc_auc']:.3f} [{rec['roc_auc_lo']:.3f}, "
                f"{rec['roc_auc_hi']:.3f}] | BSS {rec['bss']:+.3f} "
                f"[{rec['bss_lo']:+.3f}, {rec['bss_hi']:+.3f}]")

        # ---- paired model vs baseline, on the matched row set ----
        if not len(m):
            continue
        kcols = ["Grid_ID", "iso_year", "iso_week"]
        mw = m.pivot_table(index=kcols + ["presence", "spatial_block"],
                           columns="model", values="p", aggfunc="first").reset_index()
        bw = b.pivot_table(index=kcols + ["presence"],
                           columns="baseline", values="p", aggfunc="first").reset_index()
        both = mw.merge(bw, on=kcols + ["presence"], how="inner")
        for mm in [c for c in m.model.unique() if c in both.columns]:
            for bb in [c for c in b.baseline.unique() if c in both.columns]:
                sub = both[[*kcols, "spatial_block", "presence", mm, bb]].dropna()
                if sub.empty or sub.presence.nunique() < 2:
                    continue
                g = _group_key(sub, scheme)
                rec = paired_group_bootstrap(
                    sub.presence.to_numpy(int), sub[mm].to_numpy(float),
                    sub[bb].to_numpy(float), g, n_boot=n_boot)
                rec.update(model=mm, baseline=bb, scheme=scheme)
                pair_rows.append(rec)
                log(f"  {scheme:16s} {mm:20s} vs {bb:14s} "
                    f"BSS_vs_ref {rec['bss_vs_b']:+.3f} "
                    f"[{rec['bss_vs_b_lo']:+.3f}, {rec['bss_vs_b_hi']:+.3f}]"
                    f" excl0={rec['bss_vs_b_excludes_zero']}")

    base_dir.mkdir(parents=True, exist_ok=True)
    marg = pd.DataFrame(marg_rows)
    _check_against_marginal(marg, comp_dir, data_dir, res, base_dir)
    if len(marg):
        front = ["scheme", "method", "kind", "n", "n_groups", "n_boot_used"]
        marg = marg[front + [c for c in marg.columns if c not in front]]
        marg.to_csv(base_dir / "bootstrap_weekly_baselines.csv", index=False)
        log(f"[boot] wrote bootstrap_weekly_baselines.csv ({len(marg)} rows)")

    pdf = pd.DataFrame(pair_rows)
    if len(pdf):
        front = ["scheme", "model", "baseline", "n", "n_groups"]
        pdf = pdf[front + [c for c in pdf.columns if c not in front]]
        pdf.to_csv(base_dir / "bootstrap_weekly_model_vs_baseline.csv", index=False)
        log(f"[boot] wrote bootstrap_weekly_model_vs_baseline.csv ({len(pdf)} rows)")
    return marg, pdf


def main():
    ap = argparse.ArgumentParser(
        description="Group-resampled intervals for weekly baselines and models.")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--schemes", nargs="+", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    run(config_path=a.config, schemes=a.schemes, n_boot=a.n_boot)


if __name__ == "__main__":
    main()
