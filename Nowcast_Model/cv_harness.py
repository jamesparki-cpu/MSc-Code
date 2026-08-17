from __future__ import annotations
"""
cv_harness.py  —  shared, model-agnostic evaluation harness.

One place that defines HOW every model is judged, so XGBoost, Random Forest,
MaxEnt-vanilla and MaxEnt-targetgroup are compared on byte-identical folds and
metrics. Consolidates Stage 4 (blocks) + Stage 5/5b (pooled OOF, calibration,
scoring) behind a small importable API.

Any sklearn-style estimator plugs in via a factory `make_model() -> estimator`
with `.fit(X, y[, sample_weight])` and `.predict_proba(X)`. elapid's MaxentModel,
XGBClassifier and RandomForestClassifier all satisfy this.

CORE GUARANTEES (identical for every model):
  * blocks: 6 KMeans spatial regions on cell centroids; temporal = iso_year;
    combined = (spatial_block, iso_year); + a 3-block combined robustness scheme.
  * scoring: pooled OUT-OF-FOLD -> one PR-AUC (vs prevalence baseline), ROC-AUC,
    and Brier Skill Score per scheme. Every row predicted once by a model that
    never saw it.
  * calibration (optional, on by default): isotonic fit INSIDE each fold on a
    grouped held-out slice of TRAIN only -- never the test fold.

Typical use:
    import cv_harness as H
    df = H.build_blocks(df)
    res, oof = H.evaluate(df, feats, make_model, sample_weight=w, calibrate=True)
"""
from typing import Callable, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

GRID_ID_COL = "Grid_ID"
TARGET = "presence"
DEFAULT_SCHEMES = ("spatiotemporal", "temporal", "spatial", "spatiotemporal_3blocks")
RANDOM_STATE = 42


# ----- blocks ---------------------------------------------------------------
def build_blocks(df: pd.DataFrame, n_spatial: int = 6, n_coarse: int = 3,
                 random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Add spatial_block (n_spatial), coarse_block (n_coarse), st_block, and
    st_block_3 columns. Cells clustered on centroids so a held-out block shares
    no cell with training. Uses a Grid_ID->label map (not a merge) so it behaves
    identically across pandas versions."""
    df = df.copy()
    cells = df.groupby(GRID_ID_COL)[["cell_lat", "cell_lon"]].first()
    for k, col in ((n_spatial, "spatial_block"), (n_coarse, "coarse_block")):
        labels = KMeans(n_clusters=k, random_state=random_state,
                        n_init=10).fit_predict(cells[["cell_lat", "cell_lon"]])
        mapping = dict(zip(cells.index, labels))          # Grid_ID -> cluster
        df[col] = df[GRID_ID_COL].map(mapping).astype(int)
    df["st_block"]   = df["spatial_block"].astype(str) + "_" + df["iso_year"].astype(str)
    df["st_block_3"] = df["coarse_block"].astype(str)  + "_" + df["iso_year"].astype(str)
    return df


def group_labels(df: pd.DataFrame, scheme: str) -> np.ndarray:
    return {
        "spatial":                df["spatial_block"],
        "temporal":               df["iso_year"],
        "spatiotemporal":         df["st_block"],
        "spatiotemporal_3blocks": df["st_block_3"],
    }[scheme].to_numpy()


# ----- pooled out-of-fold (optionally calibrated) ---------------------------
def _fit_predict(make_model, Xtr, ytr, wtr, Xte, impute, medians):
    model = make_model()
    if impute:
        Xtr, Xte = Xtr.fillna(medians), Xte.fillna(medians)
    try:
        model.fit(Xtr, ytr, sample_weight=wtr) if wtr is not None else model.fit(Xtr, ytr)
    except TypeError:               # estimator without sample_weight support
        model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def pooled_oof(df, X, y, groups, make_model, sample_weight=None,
               calibrate=True, cal_fraction=0.25, impute=False):
    """Pooled OOF probabilities. If calibrate, learn an isotonic map inside each
    fold on a grouped held-out slice of TRAIN, then apply to the test fold."""
    medians = X.median(numeric_only=True) if impute else None
    oof = np.full(len(X), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        wtr = sample_weight.iloc[tr] if sample_weight is not None else None
        gtr = groups[tr]
        if not calibrate:
            oof[te] = _fit_predict(make_model, Xtr, ytr, wtr, X.iloc[te], impute, medians)
            continue
        # nested: split TRAIN into fit vs calibrate (grouped, no shared group)
        gss = GroupShuffleSplit(n_splits=1, test_size=cal_fraction, random_state=RANDOM_STATE)
        fi, ci = next(gss.split(Xtr, ytr, gtr))
        wfi = wtr.iloc[fi] if wtr is not None else None
        p_cal = _fit_predict(make_model, Xtr.iloc[fi], ytr.iloc[fi], wfi,
                             Xtr.iloc[ci], impute, medians)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, ytr.iloc[ci])
        p_te = _fit_predict(make_model, Xtr.iloc[fi], ytr.iloc[fi], wfi,
                            X.iloc[te], impute, medians)
        oof[te] = iso.predict(p_te)
    return oof


# ----- scoring --------------------------------------------------------------
def score(y: np.ndarray, p: np.ndarray) -> dict:
    m = ~np.isnan(p); y, p = np.asarray(y)[m], p[m]
    prev = y.mean()
    brier = brier_score_loss(y, p)
    brier_base = brier_score_loss(y, np.full_like(p, prev))
    return dict(n=int(m.sum()), prevalence=round(float(prev), 3),
                pr_auc=round(average_precision_score(y, p), 3),
                pr_baseline=round(float(prev), 3),
                pr_lift=round(average_precision_score(y, p) - prev, 3),
                roc_auc=round(roc_auc_score(y, p), 3),
                brier=round(brier, 4),
                bss=round(1 - brier / brier_base, 3) if brier_base > 0 else np.nan)


# ----- orchestrator ---------------------------------------------------------
def evaluate(df: pd.DataFrame, feats: Sequence[str], make_model: Callable,
             schemes: Sequence[str] = DEFAULT_SCHEMES,
             sample_weight: Optional[pd.Series] = None,
             calibrate: bool = True, impute: bool = False,
             model_name: str = "model", verbose: bool = True):
    """Run every scheme; return (metrics_df, oof_dict). Assumes build_blocks ran."""
    X = df[list(feats)]; y = df[TARGET].astype(int)
    rows, oof_store = [], {}
    for scheme in schemes:
        groups = group_labels(df, scheme)
        oof = pooled_oof(df, X, y, groups, make_model, sample_weight, calibrate, impute=impute)
        s = score(y.to_numpy(), oof); s.update(model=model_name, scheme=scheme)
        rows.append(s); oof_store[scheme] = oof
        if verbose:
            print(f"[{model_name:16s}|{scheme:22s}] "
                  f"PR-AUC {s['pr_auc']:.3f} (base {s['pr_baseline']:.3f}, "
                  f"lift {s['pr_lift']:+.3f}) | ROC {s['roc_auc']:.3f} | BSS {s['bss']:+.3f}",
                  flush=True)
    cols = ["model","scheme","n","prevalence","pr_auc","pr_baseline","pr_lift",
            "roc_auc","brier","bss"]
    return pd.DataFrame(rows)[cols], oof_store