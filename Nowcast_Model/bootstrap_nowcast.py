from __future__ import annotations
"""
bootstrap_nowcast.py  —  uncertainty on the forward-chained headline forecast.

WHAT IS BEING BOOTSTRAPPED
  The headline nowcast fold: models trained on all years strictly before the
  final year, scored on the final year alone. That single number is the
  deployment-realistic claim, so it is the one that most needs an interval.

RESAMPLING UNIT = CELL (not row, not year)
  Within one test year there is only one year, so year cannot be the resampling
  unit. Rows are not independent -- repeated cell-weeks at the same trap site
  share site effects and land cover -- so row resampling would understate the
  interval. We therefore resample CELLS (Grid_ID) with replacement, taking all
  of a drawn cell's cell-weeks together. This preserves within-site dependence,
  which is the dominant structure once the year is fixed.

  This is a weaker guarantee than the block bootstrap used for the CV
  comparison, where whole region-years were resampled. State that honestly: the
  interval accounts for which SITES were surveyed, not for which YEAR was held
  out. With a single test year, the latter is not estimable.

INPUT
  <nowcast_results_dir>/nowcast_oof_long.csv, written by the patched
  nowcast_run.py, with columns:
    model, test_year, Grid_ID, presence, p

OUTPUT
  <nowcast_results_dir>/bootstrap_nowcast.csv
    per model: point estimate + 95% CI for ROC-AUC, PR-lift, BSS
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

RANDOM_STATE = 42
N_BOOT = 1000


def log(m): print(m, flush=True)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Same definitions as nowcast_cv.score, so intervals match the point estimates."""
    if len(np.unique(y)) < 2:
        return dict(roc_auc=np.nan, pr_lift=np.nan, bss=np.nan)
    prev = y.mean()
    pr = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    brier_base = brier_score_loss(y, np.full_like(p, prev))
    return dict(
        roc_auc=roc_auc_score(y, p),
        pr_lift=pr - prev,
        bss=1 - brier / brier_base if brier_base > 0 else np.nan,
    )


def bootstrap_by_cell(y, p, cells, n_boot=N_BOOT, seed=RANDOM_STATE):
    """Resample whole cells with replacement; return percentile CIs."""
    uniq = np.unique(cells)
    idx_by_cell = {c: np.flatnonzero(cells == c) for c in uniq}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ("roc_auc", "pr_lift", "bss")}
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cell[c] for c in drawn])
        m = metrics(y[idx], p[idx])
        for k in draws:
            draws[k].append(m[k])
    out = {}
    for k, v in draws.items():
        arr = np.asarray(v, dtype=float)
        arr = arr[~np.isnan(arr)]
        out[f"{k}_lo"] = round(float(np.percentile(arr, 2.5)), 3) if arr.size else np.nan
        out[f"{k}_hi"] = round(float(np.percentile(arr, 97.5)), 3) if arr.size else np.nan
    return out


def run():
    with open("config.json") as f:
        cfg = json.load(f)
    NOWCAST_DIR = Path(cfg["nowcast_dir"])
    RES = Path(cfg.get("nowcast_results_dir", str(NOWCAST_DIR / "Nowcast_Results")))

    src = RES / "nowcast_oof_long.csv"
    if not src.exists():
        raise SystemExit(f"missing {src} -- patch nowcast_run.py and re-run it first")

    df = pd.read_csv(src).dropna(subset=["p"])
    final_year = int(df["test_year"].max())
    df = df[df["test_year"] == final_year]
    log(f"[load] headline fold = test year {final_year} | {len(df):,} rows | "
        f"{df['model'].nunique()} models")

    rows = []
    for model, sub in df.groupby("model", sort=False):
        sub = sub.reset_index(drop=True)
        y = sub["presence"].to_numpy().astype(int)
        p = sub["p"].to_numpy()
        cells = sub["Grid_ID"].to_numpy()

        point = metrics(y, p)
        ci = bootstrap_by_cell(y, p, cells)

        rec = dict(model=model, test_year=final_year, n=len(sub),
                   n_cells=int(pd.Series(cells).nunique()),
                   prevalence=round(float(y.mean()), 3))
        for k, v in point.items():
            rec[k] = round(float(v), 3)
            rec[f"{k}_lo"] = ci[f"{k}_lo"]
            rec[f"{k}_hi"] = ci[f"{k}_hi"]
        rows.append(rec)
        log(f"[nowcast] {model:22s} ROC {rec['roc_auc']:.3f} "
            f"[{rec['roc_auc_lo']:.3f}, {rec['roc_auc_hi']:.3f}] | "
            f"BSS {rec['bss']:+.3f} [{rec['bss_lo']:+.3f}, {rec['bss_hi']:+.3f}] "
            f"| {rec['n_cells']} cells")

    out = pd.DataFrame(rows)
    out.to_csv(RES / "bootstrap_nowcast.csv", index=False)
    log(f"\n[done] wrote bootstrap_nowcast.csv to {RES}")
    log("[note] the interval reflects WHICH SITES were surveyed, not which year "
        "was held out -- with one test year, between-year variation is not "
        "estimable from this fold and must be read off the per-fold progression.")
    return out


if __name__ == "__main__":
    run()