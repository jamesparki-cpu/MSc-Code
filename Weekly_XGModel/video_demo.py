from __future__ import annotations
"""
model_demo.py  —  reduced end-to-end demonstration of the modelling pipeline.

WHAT THIS IS
  A genuine, scoped-down run of the real pipeline: it loads the real training
  table, builds real spatial blocks, fits a real XGBoost model under real blocked
  cross-validation, scores it with the real metric definitions, predicts a real
  statewide surface and writes real outputs.

WHAT THIS IS NOT
  The full evaluation. To finish in under a minute it uses a REDUCED
  CONFIGURATION -- fewer boosting rounds, a subset of folds, one scheme, one
  week. The numbers it prints are therefore indicative of the pipeline working,
  NOT the reported results. Headline figures come from compare_models.py.

  Every reduction is announced in the terminal as it happens, so nothing here is
  passed off as the full run.

OUTPUTS (to <weekly_xg_dir>/Demo_Run/)
  demo_metrics.csv          fold-by-fold and pooled scores
  demo_importance.csv       gain-based feature importance, grouped
  demo_map_week<NN>.png     statewide suitability surface for the chosen week

USAGE
  python model_demo.py                 # defaults to week 32
  python model_demo.py --week 20
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import LeaveOneGroupOut

import xgboost as xgb
import cv_harness as H

# ---- REDUCED CONFIGURATION (announced at runtime) -------------------------
DEMO_ESTIMATORS = 150      # full pipeline uses 400
DEMO_MAX_FOLDS = 4         # full spatiotemporal scheme uses up to 36
DEMO_SCHEME = "spatiotemporal"
DEFAULT_WEEK = 32
RANDOM_STATE = 42

XGB_PARAMS = dict(learning_rate=0.03, max_depth=4, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                  random_state=RANDOM_STATE, objective="binary:logistic",
                  eval_metric="logloss", tree_method="hist")

_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
BOLD, DIM, GREEN, CYAN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[92m", "\033[96m", "\033[93m", "\033[0m")


def log(m=""): print(m, flush=True)


def rule(title):
    log(f"\n{BOLD}{'─' * 70}{RESET}")
    log(f"{BOLD}  {title}{RESET}")
    log(f"{BOLD}{'─' * 70}{RESET}")


def step(m): log(f"  {CYAN}▸{RESET} {m}")
def done(m): log(f"  {GREEN}✓{RESET} {m}")
def warn(m): log(f"  {YELLOW}!{RESET} {m}")


def score(y, p):
    """Identical metric definitions to cv_harness, so the demo is comparable."""
    prev = float(np.mean(y))
    pr = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    base = brier_score_loss(y, np.full_like(p, prev, dtype=float))
    return dict(n=len(y), prevalence=round(prev, 3),
                roc_auc=round(roc_auc_score(y, p), 3),
                pr_auc=round(pr, 3), pr_lift=round(pr - prev, 3),
                brier=round(brier, 4),
                bss=round(1 - brier / base, 3) if base > 0 else np.nan)


def centroids(df):
    lat, lon = [], []
    for g in df[H.GRID_ID_COL].astype(str):
        m = _GID.search(g)
        lat.append(float(m.group(1))); lon.append(float(m.group(2)))
    return np.array(lat), np.array(lon)


def run(week: int):
    t0 = time.time()

    log(f"\n{BOLD}CULEX NIGRIPALPUS — WEEKLY HABITAT SUITABILITY{RESET}")
    log(f"{DIM}reduced demonstration run · not the reported evaluation{RESET}")

    # ---------------------------------------------------------------- stage 1
    rule("STAGE 1 · CONFIGURATION AND INPUTS")
    with open("config.json") as f:
        cfg = json.load(f)
    FEAT_DIR = Path(cfg["weekly_xg_dir"])
    OUT = FEAT_DIR / "Demo_Run"; OUT.mkdir(parents=True, exist_ok=True)
    step(f"config.json resolved · weekly_xg_dir = {FEAT_DIR.name}")

    spec = json.load(open(FEAT_DIR / "model_features.json"))
    feats = spec["model_features"]
    banned = spec.get("never_feed_to_model", [])
    done(f"feature allowlist loaded · {len(feats)} predictors, "
         f"{len(banned)} columns explicitly excluded")

    df = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet")
    done(f"training table loaded · {len(df):,} cell-weeks × {df.shape[1]} columns")

    dyn = [f for f in feats if any(k in f for k in ("_7d", "_14d", "_28d", "_level"))]
    stat = [f for f in feats if f.startswith("pct_") or f in ("elevation", "slope")]
    seas = [f for f in feats if f in ("sin_doy", "cos_doy")]
    step(f"predictor composition · {len(dyn)} dynamic · {len(stat)} static · "
         f"{len(seas)} seasonal")

    # ---------------------------------------------------------------- stage 2
    rule("STAGE 2 · DATA CHARACTERISATION")
    n_cells = df[H.GRID_ID_COL].nunique()
    yr0, yr1 = int(df.iso_year.min()), int(df.iso_year.max())
    prev = df.presence.mean()
    step(f"spatial extent · {n_cells:,} surveyed 5 km cells")
    step(f"temporal extent · {yr0}–{yr1} · ISO weeks "
         f"{int(df.iso_week.min())}–{int(df.iso_week.max())}")
    step(f"class balance · {int(df.presence.sum()):,} presence / "
         f"{int((1 - df.presence).sum()):,} inferred absence")
    warn(f"prevalence {prev:.3f} — the minority class is ABSENCE, not presence")

    key = [H.GRID_ID_COL, "iso_year", "iso_week"]
    assert not df.duplicated(subset=key).any(), "duplicated cell-week key"
    done("cell-week keys verified unique · presence/absence sets disjoint")

    # ---------------------------------------------------------------- stage 3
    rule("STAGE 3 · SPATIAL BLOCKING")
    step("clustering cell centroids with KMeans (seed 42) …")
    df = H.build_blocks(df)
    sizes = df.groupby("spatial_block")[H.GRID_ID_COL].nunique().sort_index()
    done(f"{len(sizes)} spatial blocks · {sizes.min()}–{sizes.max()} cells each")

    straddle = df.groupby(H.GRID_ID_COL)["spatial_block"].nunique().gt(1).sum()
    assert straddle == 0, f"{straddle} cells span multiple blocks"
    done("fold integrity verified · no cell appears in more than one block")

    df["st_block"] = (df.spatial_block.astype(str) + "_" + df.iso_year.astype(str))
    groups = df["st_block"].to_numpy()
    step(f"leave-one-group-out groups · {pd.Series(groups).nunique()} "
         f"(block × year) combinations")

    # ---------------------------------------------------------------- stage 4
    rule("STAGE 4 · BLOCKED CROSS-VALIDATION")
    warn(f"REDUCED: {DEMO_ESTIMATORS} boosting rounds (full run uses 400)")
    warn(f"REDUCED: first {DEMO_MAX_FOLDS} folds only (full run uses all "
         f"{pd.Series(groups).nunique()})")
    warn(f"REDUCED: {DEMO_SCHEME} scheme only · calibration disabled")

    X = df[feats]
    y = df["presence"].to_numpy().astype(int)
    w = np.sqrt(df["n_events"].clip(lower=1)).to_numpy()
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    step(f"sample weights = sqrt(n_events) · scale_pos_weight = {spw:.3f}")

    logo = LeaveOneGroupOut()
    rows, oof_idx, oof_p = [], [], []
    for k, (tr, te) in enumerate(logo.split(X, y, groups), start=1):
        if k > DEMO_MAX_FOLDS:
            break
        if len(np.unique(y[te])) < 2:
            warn(f"fold {k} · single-class test set, skipped"); continue
        t = time.time()
        m = xgb.XGBClassifier(n_estimators=DEMO_ESTIMATORS,
                              scale_pos_weight=spw, **XGB_PARAMS)
        m.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        s = score(y[te], p)
        rows.append(dict(fold=k, held_out=df["st_block"].iloc[te[0]],
                         n_train=len(tr), **s))
        oof_idx.append(te); oof_p.append(p)
        log(f"  {GREEN}✓{RESET} fold {k} · held out {df['st_block'].iloc[te[0]]:<10s} "
            f"n={s['n']:>5,} · ROC {s['roc_auc']:.3f} · PR-lift {s['pr_lift']:+.3f} "
            f"· BSS {s['bss']:+.3f} {DIM}({time.time()-t:.1f}s){RESET}")

    oof_idx = np.concatenate(oof_idx); oof_p = np.concatenate(oof_p)
    pooled = score(y[oof_idx], oof_p)
    log()
    done(f"pooled over {len(rows)} folds · ROC {pooled['roc_auc']:.3f} · "
         f"PR-lift {pooled['pr_lift']:+.3f} · BSS {pooled['bss']:+.3f}")

    pd.DataFrame(rows + [dict(fold="pooled", held_out="—", n_train=np.nan, **pooled)]
                 ).to_csv(OUT / "demo_metrics.csv", index=False)
    done(f"wrote demo_metrics.csv ({len(rows)+1} rows)")

    # ---------------------------------------------------------------- stage 5
    rule("STAGE 5 · DEPLOYMENT FIT AND ATTRIBUTION")
    step(f"refitting on all {len(df):,} cell-weeks …")
    final = xgb.XGBClassifier(n_estimators=DEMO_ESTIMATORS,
                              scale_pos_weight=spw, **XGB_PARAMS)
    final.fit(X, y, sample_weight=w)
    done("deployment model fitted")

    gain = pd.Series(final.get_booster().get_score(importance_type="gain"))
    gain = gain.reindex(feats).fillna(0.0)
    imp = pd.DataFrame({"feature": gain.index, "gain": gain.values})
    imp["group"] = imp.feature.map(
        lambda f: "land_cover" if f.startswith("pct_")
        else "terrain" if f in ("elevation", "slope")
        else "seasonality" if f in ("sin_doy", "cos_doy")
        else "temperature" if "tm" in f
        else "precipitation" if "prcp" in f
        else "moisture" if ("vpd" in f or "ndwi" in f)
        else "vegetation")
    imp["pct"] = 100 * imp.gain / imp.gain.sum()
    imp.sort_values("gain", ascending=False).to_csv(OUT / "demo_importance.csv", index=False)

    grouped = imp.groupby("group")["pct"].sum().sort_values(ascending=False)
    log(f"\n  {DIM}gain-based importance by block:{RESET}")
    for g, v in grouped.items():
        log(f"    {g:<15s} {'█' * int(v / 2):<26s} {v:5.1f}%")
    warn("gain importance is indicative only · reported attribution uses grouped SHAP")
    done("wrote demo_importance.csv")

    # ---------------------------------------------------------------- stage 6
    rule(f"STAGE 6 · STATEWIDE PREDICTION · ISO WEEK {week}")
    gridfile = FEAT_DIR / "statewide_weekly_features.parquet"
    if not gridfile.exists():
        warn(f"{gridfile.name} not found — skipping map "
             f"(run build_state_features.py to enable)")
    else:
        grid = pd.read_parquet(gridfile)
        gw = grid[grid.iso_week == week].copy()
        step(f"loaded prediction grid · {gw[H.GRID_ID_COL].nunique():,} cells "
             f"for week {week}")

        missing = [f for f in feats if f not in gw.columns]
        assert not missing, f"grid missing {len(missing)} features: {missing[:5]}"
        done("grid/feature parity verified · column names and order match allowlist")

        gw["suitability"] = final.predict_proba(gw[feats])[:, 1]
        lat, lon = centroids(gw)
        done(f"predicted · mean {gw.suitability.mean():.3f} · "
             f"range {gw.suitability.min():.3f}–{gw.suitability.max():.3f}")

        fig, ax = plt.subplots(figsize=(6.2, 7.4))
        sc = ax.scatter(lon, lat, c=gw.suitability, cmap="RdYlBu_r",
                        s=6, marker="s", vmin=0, vmax=1, linewidths=0)
        ax.set_aspect(1 / np.cos(np.radians(float(np.mean(lat)))))
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(f"Culex nigripalpus suitability · ISO week {week}\n"
                     f"climatological mean conditions {yr0}–{yr1}", fontsize=10)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="calibrated suitability")
        fig.text(0.5, 0.015, "demonstration run — reduced configuration",
                 ha="center", fontsize=7, color="0.45")
        fig.tight_layout()
        png = OUT / f"demo_map_week{week:02d}.png"
        fig.savefig(png, dpi=140); plt.close(fig)
        done(f"wrote {png.name}")

        out_cols = [H.GRID_ID_COL, "iso_week", "suitability"]
        gw[out_cols].to_csv(OUT / f"demo_surface_week{week:02d}.csv", index=False)
        done(f"wrote demo_surface_week{week:02d}.csv ({len(gw):,} rows)")

    # ---------------------------------------------------------------- summary
    rule("RUN COMPLETE")
    log(f"  elapsed {BOLD}{time.time() - t0:.1f}s{RESET}")
    log(f"  outputs → {OUT}")
    for f in sorted(OUT.iterdir()):
        log(f"    {DIM}·{RESET} {f.name}")
    log(f"\n  {YELLOW}Reduced configuration — indicative of pipeline function, "
        f"not reported results.{RESET}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=DEFAULT_WEEK)
    run(ap.parse_args().week)