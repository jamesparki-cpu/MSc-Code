from __future__ import annotations
"""
run_baselines.py  —  standalone baseline driver. NEEDS NO MODELS.

Produces the reference-forecast tables immediately, because climatology,
persistence and prevalence depend only on the observed labels. Run this before
committing to any model re-run.

WHAT IT DOES
  1. NOWCAST folds (forward-chaining by year, matching nowcast_cv):
       baseline_metrics_nowcast.csv      per-fold + pooled, every baseline
       baseline_diagnostics_nowcast.csv  climatology tier usage + coverage
  2. WEEKLY blocked CV schemes (spatial / temporal / spatiotemporal), if
     cv_harness is importable:
       baseline_metrics_weekly.csv       per scheme, pooled out-of-fold
     PERSISTENCE IS OMITTED FOR THE SPATIAL SCHEME on purpose -- in a spatial
     holdout the previous week at a held-out cell lies inside the held-out block,
     so persistence would read labels from the withheld region and would not be a
     spatial-transfer baseline. Climatology is the meaningful spatial reference,
     and note it degrades to a SEASONAL-ONLY reference there (tier 2), because a
     held-out cell has no training history.
  3. DERIVED model skill without re-running anything:
       derived_bss_vs_climatology.csv
     Climatology has 100% test-row coverage (tier fallback guarantees a value),
     so where an existing per-fold model Brier was computed on the same rows,
       BSS_vs_climatology = 1 - brier_model / brier_climatology
     can be computed arithmetically from nowcast_perfold_all.csv. Row counts are
     checked and any mismatch is refused rather than silently divided.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import baselines as B

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR    = Path(cfg.get("weekly_xg_dir", "."))
NOWCAST_RES = Path(cfg.get("nowcast_results_dir",
                           str(Path(cfg.get("nowcast_dir", str(DATA_DIR))) / "Nowcast_Results")))
OUT_DIR     = Path(cfg.get("baselines_dir", str(NOWCAST_RES / "Baselines")))

PREVALENCE_SOURCE = "train"      # 'train' (operationally honest) or 'test'
WEEKLY_SCHEMES = ("spatiotemporal", "temporal", "spatial")

def log(m): print(m, flush=True)
# ============================================================================


def load_table():
    df = pd.read_parquet(DATA_DIR / "weekly_model_table.parquet").reset_index(drop=True)
    need = [B.TARGET, B.YEAR, B.WEEK, B.CELL, B.WEEK_START]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise AssertionError(f"model table missing {missing}")
    return df


# ----------------------------- 1. nowcast folds -----------------------------
def nowcast_baselines(df):
    import nowcast_cv as NC
    years = sorted(df[B.YEAR].unique())
    folds = list(NC.forward_chain_folds(years))
    y = df[B.TARGET].astype(int).to_numpy()

    rows, diag_rows = [], []
    oof = {}
    for train_years, test_year in folds:
        preds, diags, test_mask = B.build_fold_baselines(
            df, train_years, test_year, PREVALENCE_SOURCE)
        for k, v in preds.items():
            oof.setdefault(k, np.full(len(df), np.nan))
            oof[k][test_mask] = v[test_mask]
        for k, d in diags.items():
            diag_rows.append({"scope": "nowcast", "test_year": test_year,
                              "baseline": k, **d})
        refs = {k: preds[k] for k in ("prevalence", "climatology", "persistence")}
        for name, vec in preds.items():
            v = vec.copy(); v[~test_mask] = np.nan
            s = B.score_forecast(y, v, refs, constant_forecast=(name == "prevalence"))
            s.update(baseline=name, test_year=test_year,
                     train_years=f"{min(train_years)}-{max(train_years)}")
            rows.append(s)
        c, p = diags["climatology"], diags["persistence"]
        log(f"[nowcast] test {test_year}: clim tier1 {c['tier1_cell_week_pct']}% / "
            f"tier2 {c['tier2_week_only_pct']}% | persistence cov {p['coverage_pct']}%")

    refs_all = {k: oof[k] for k in ("prevalence", "climatology", "persistence")}
    for name, vec in oof.items():
        s = B.score_forecast(y, vec, refs_all, constant_forecast=(name == "prevalence"))
        s.update(baseline=name, test_year="POOLED", train_years="all folds")
        rows.append(s)

    per = pd.DataFrame(rows)
    per.to_csv(OUT_DIR / "baseline_metrics_nowcast.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT_DIR / "baseline_diagnostics_nowcast.csv", index=False)
    log(f"[nowcast] wrote baseline_metrics_nowcast.csv ({len(per)} rows)")
    return per, oof


# ----------------------------- 2. weekly schemes ----------------------------
def weekly_baselines(df):
    try:
        import cv_harness as H
        from sklearn.model_selection import LeaveOneGroupOut
    except ImportError as e:
        log(f"[weekly] SKIP ({e}) — put cv_harness.py alongside this script "
            f"or add its folder to sys.path")
        return None

    blocked = H.build_blocks(df)
    y = df[B.TARGET].astype(int).to_numpy()
    rows, diag_rows = [], []

    for scheme in WEEKLY_SCHEMES:
        groups = H.group_labels(blocked, scheme)
        # persistence is invalid for a spatial holdout (see build_baselines_for_masks)
        use_persist = scheme != "spatial"
        oof = {}
        for tr, te in LeaveOneGroupOut().split(np.zeros(len(df)), y, groups):
            train_mask = np.zeros(len(df), bool); train_mask[tr] = True
            test_mask = np.zeros(len(df), bool); test_mask[te] = True
            preds, diags = B.build_baselines_for_masks(
                df, train_mask, test_mask, PREVALENCE_SOURCE,
                include_persistence=use_persist)
            for k, v in preds.items():
                oof.setdefault(k, np.full(len(df), np.nan))
                oof[k][test_mask] = v[test_mask]
            diag_rows.append({"scope": scheme, "fold": str(groups[te][0]),
                              "baseline": "climatology", **diags["climatology"]})

        refs = {k: oof[k] for k in ("prevalence", "climatology") if k in oof}
        if "persistence" in oof:
            refs["persistence"] = oof["persistence"]
        for name, vec in oof.items():
            s = B.score_forecast(y, vec, refs, constant_forecast=(name == "prevalence"))
            s.update(baseline=name, scheme=scheme)
            rows.append(s)
        t1 = np.mean([d["tier1_cell_week_pct"] for d in diag_rows if d["scope"] == scheme])
        log(f"[weekly] {scheme}: persistence {'included' if use_persist else 'OMITTED (invalid)'} "
            f"| mean climatology tier1 {t1:.1f}%"
            + ("  <-- climatology is SEASONAL-ONLY here" if scheme == "spatial" and t1 < 20 else ""))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "baseline_metrics_weekly.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT_DIR / "baseline_diagnostics_weekly.csv", index=False)
    log(f"[weekly] wrote baseline_metrics_weekly.csv ({len(out)} rows)")
    return out


# ------------------- 3. derive model BSS vs climatology ---------------------
def derive_model_bss(nowcast_per):
    """No model re-run: BSS = 1 - brier_model/brier_climatology, per fold."""
    f = NOWCAST_RES / "nowcast_perfold_all.csv"
    if not f.exists():
        log(f"[derive] SKIP: {f.name} not found")
        return None
    mods = pd.read_csv(f)
    if "brier" not in mods.columns:
        log("[derive] SKIP: no 'brier' column in nowcast_perfold_all.csv")
        return None
    clim = (nowcast_per[(nowcast_per.baseline == "climatology") &
                        (nowcast_per.test_year != "POOLED")]
            [["test_year", "brier", "n_scored"]]
            .rename(columns={"brier": "brier_clim", "n_scored": "n_clim"}))
    clim["test_year"] = clim["test_year"].astype(int)
    m = mods.merge(clim, on="test_year", how="left")

    bad = m[(m.n.notna()) & (m.n_clim.notna()) & (m.n != m.n_clim)]
    if len(bad):
        log(f"[derive] REFUSED for {len(bad)} row(s): model n != climatology n "
            f"(different denominators -> ratio not interpretable). Re-run those.")
    ok = m.n == m.n_clim
    m["bss_vs_climatology"] = np.where(ok, 1 - m.brier / m.brier_clim, np.nan)
    keep = ["model", "test_year", "n", "roc_auc", "brier", "brier_clim",
            "bss", "bss_vs_climatology"]
    out = m[[c for c in keep if c in m.columns]]
    out.to_csv(OUT_DIR / "derived_bss_vs_climatology.csv", index=False)
    log(f"[derive] wrote derived_bss_vs_climatology.csv "
        f"({int(ok.sum())}/{len(m)} rows derivable, no model re-run)")
    return out


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_table()
    log(f"[base] {len(df):,} rows | {df[B.CELL].nunique()} cells | "
        f"years {sorted(df[B.YEAR].unique())} | prevalence source '{PREVALENCE_SOURCE}'")
    per, _ = nowcast_baselines(df)
    weekly_baselines(df)
    derive_model_bss(per)
    log(f"\n[done] baseline tables -> {OUT_DIR}")
    log("[note] interpret against the MATCHED subset when comparing to models; "
        "see BASELINES.md B4.")


if __name__ == "__main__":
    run()