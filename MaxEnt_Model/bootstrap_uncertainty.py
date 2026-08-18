from __future__ import annotations
"""
bootstrap_uncertainty.py  —  uncertainty intervals for the CV metric comparison.

WHY BLOCK BOOTSTRAP (and not row bootstrap):
  The whole point of blocked CV is that rows within a spatial region and year are
  dependent. Resampling ROWS would destroy that dependence and give intervals
  that are far too narrow -- the same error that random k-fold makes. We
  therefore resample whole CV GROUPS with replacement, so a region-year either
  appears (possibly several times) or does not appear at all.

WHY PAIRED DIFFERENCES (and not two marginal CIs):
  Two overlapping marginal intervals do NOT establish that a difference is
  unreliable. The correct object is the distribution of the DIFFERENCE computed
  on the SAME resampled groups, which cancels the shared fold-composition noise.

PAIRING CONSTRAINT:
  xgboost / random_forest / maxent_targetgroup are scored on identical rows, so
  they can be paired. maxent_vanilla uses presences + random background: a
  DIFFERENT evaluation set with a different prevalence (0.628 vs 0.776), so it
  gets marginal intervals only and must not be paired against the others.

INPUT (one tidy CSV):
  oof_long.csv with columns:
    model, scheme, Grid_ID, iso_year, spatial_block, presence, p
  where p is the pooled out-of-fold predicted probability that was used to
  produce the reported metric (i.e. the CALIBRATED one if that is what the
  results table reports).

OUTPUT (to --out):
  bootstrap_marginal.csv    per model x scheme: point estimate + 95% CI
  bootstrap_paired.csv      per model-pair x scheme: mean difference + 95% CI
                            + fraction of replicates favouring each model
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

RANDOM_STATE = 42
N_BOOT = 1000

# models scored on identical rows -> paired comparison is valid
PAIRABLE = ["xgboost", "random_forest", "maxent_targetgroup"]


def group_key(df: pd.DataFrame, scheme: str) -> pd.Series:
    """The resampling unit = the CV group for that scheme."""
    if scheme == "spatial":
        return df["spatial_block"].astype(str)
    if scheme == "temporal":
        return df["iso_year"].astype(str)
    if scheme in ("spatiotemporal", "spatiotemporal_3blocks"):
        return df["spatial_block"].astype(str) + "_" + df["iso_year"].astype(str)
    raise ValueError(f"unknown scheme: {scheme}")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Same definitions as cv_harness.score, so intervals match the point estimates."""
    if len(np.unique(y)) < 2:
        return dict(roc_auc=np.nan, pr_auc=np.nan, pr_lift=np.nan, bss=np.nan)
    prev = y.mean()
    pr = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    brier_base = brier_score_loss(y, np.full_like(p, prev))
    return dict(
        roc_auc=roc_auc_score(y, p),
        pr_auc=pr,
        pr_lift=pr - prev,
        bss=1 - brier / brier_base if brier_base > 0 else np.nan,
    )


def _resample_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw groups with replacement; return concatenated row indices."""
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    drawn = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([idx_by_group[g] for g in drawn])


def bootstrap_marginal(df: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Per model x scheme: point estimate and percentile CI for each metric."""
    rows = []
    for (model, scheme), sub in df.groupby(["model", "scheme"], sort=False):
        sub = sub.reset_index(drop=True)
        y = sub["presence"].to_numpy().astype(int)
        p = sub["p"].to_numpy()
        groups = group_key(sub, scheme).to_numpy()

        point = metrics(y, p)
        rng = np.random.default_rng(RANDOM_STATE)
        draws = {k: [] for k in point}
        for _ in range(n_boot):
            idx = _resample_indices(groups, rng)
            m = metrics(y[idx], p[idx])
            for k, v in m.items():
                draws[k].append(v)

        rec = dict(model=model, scheme=scheme, n=len(sub),
                   n_groups=int(pd.Series(groups).nunique()))
        for k, v in point.items():
            arr = np.asarray(draws[k], dtype=float)
            arr = arr[~np.isnan(arr)]
            rec[f"{k}"] = round(float(v), 3)
            rec[f"{k}_lo"] = round(float(np.percentile(arr, 2.5)), 3) if arr.size else np.nan
            rec[f"{k}_hi"] = round(float(np.percentile(arr, 97.5)), 3) if arr.size else np.nan
        rows.append(rec)
        print(f"[marginal] {model:22s} {scheme:22s} "
              f"ROC {rec['roc_auc']:.3f} [{rec['roc_auc_lo']:.3f}, {rec['roc_auc_hi']:.3f}]",
              flush=True)
    return pd.DataFrame(rows)


def bootstrap_paired(df: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Paired differences on identical resampled groups, for pairable models only."""
    rows = []
    for scheme, sub_s in df[df["model"].isin(PAIRABLE)].groupby("scheme", sort=False):
        # wide: one row per observation, one column per model
        wide = sub_s.pivot_table(
            index=["Grid_ID", "iso_year", "spatial_block", "presence"],
            columns="model", values="p", aggfunc="first").reset_index()
        present = [m for m in PAIRABLE if m in wide.columns]
        wide = wide.dropna(subset=present)
        if len(present) < 2:
            continue

        y = wide["presence"].to_numpy().astype(int)
        groups = group_key(wide, scheme).to_numpy()

        for i, a in enumerate(present):
            for b in present[i + 1:]:
                pa, pb = wide[a].to_numpy(), wide[b].to_numpy()
                point = {k: metrics(y, pa)[k] - metrics(y, pb)[k]
                         for k in ("roc_auc", "pr_lift", "bss")}
                rng = np.random.default_rng(RANDOM_STATE)
                draws = {k: [] for k in point}
                for _ in range(n_boot):
                    idx = _resample_indices(groups, rng)     # SAME idx for both models
                    ma, mb = metrics(y[idx], pa[idx]), metrics(y[idx], pb[idx])
                    for k in point:
                        draws[k].append(ma[k] - mb[k])

                rec = dict(scheme=scheme, model_a=a, model_b=b, n=len(wide))
                for k, v in point.items():
                    arr = np.asarray(draws[k], dtype=float)
                    arr = arr[~np.isnan(arr)]
                    rec[f"d_{k}"] = round(float(v), 3)
                    rec[f"d_{k}_lo"] = round(float(np.percentile(arr, 2.5)), 3)
                    rec[f"d_{k}_hi"] = round(float(np.percentile(arr, 97.5)), 3)
                    # fraction of replicates where a > b: a descriptive statistic,
                    # NOT a p-value -- do not report it as one
                    rec[f"d_{k}_frac_a_better"] = round(float((arr > 0).mean()), 3)
                rows.append(rec)
                print(f"[paired]   {scheme:22s} {a} - {b}: "
                      f"dROC {rec['d_roc_auc']:+.3f} "
                      f"[{rec['d_roc_auc_lo']:+.3f}, {rec['d_roc_auc_hi']:+.3f}]",
                      flush=True)
    return pd.DataFrame(rows)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=None, help="path to oof_long.csv (default: from config.json)")
    ap.add_argument("--out", default=None, help="output directory (default: from config.json)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

        # fall back to config.json so the script runs with no arguments (IDE-friendly)
    if args.oof is None or args.out is None:
        import json
        with open("config.json") as f:
            cfg = json.load(f)
        feat_dir = Path(cfg["weekly_xg_dir"])
        default_out = Path(cfg.get("comparison_dir", str(feat_dir.parent / "Comparison_Results")))
        if args.out is None:
            args.out = str(default_out)
        if args.oof is None:
            args.oof = str(default_out / "oof_long.csv")
        print(f"[config] oof = {args.oof}\n[config] out = {args.out}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.oof)
    required = {"model", "scheme", "Grid_ID", "iso_year", "spatial_block", "presence", "p"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"oof_long.csv missing columns: {sorted(missing)}")
    df = df.dropna(subset=["p"])
    print(f"[load] {len(df):,} OOF rows | {df['model'].nunique()} models | "
              f"{df['scheme'].nunique()} schemes\n", flush=True)

    marg = bootstrap_marginal(df, args.n_boot)
    marg.to_csv(out / "bootstrap_marginal.csv", index=False)

    print()
    paired = bootstrap_paired(df, args.n_boot)
    paired.to_csv(out / "bootstrap_paired.csv", index=False)

    print(f"\n[done] wrote bootstrap_marginal.csv + bootstrap_paired.csv to {out}")
    print("[note] maxent_vanilla appears in the marginal table only: it is scored "
          "on a different evaluation set (presences + random background), so it "
          "cannot be paired against the target-group models.")


if __name__ == "__main__":
    main()