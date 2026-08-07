from __future__ import annotations
"""
shap_common.py  —  shared SHAP machinery for the four models.

Explainer dispatch (they are NOT the same method, and that matters for methods
reporting):
  * XGBoost / Random Forest -> TreeSHAP: EXACT, fast.
  * MaxEnt (elapid)         -> PermutationExplainer: model-agnostic and
        APPROXIMATE. elapid fits a regularised logistic model over internally
        expanded hinge/quadratic features, so there is no tree structure to
        exploit. ~0.9 s/row, hence the smaller MaxEnt explain sample.

GROUPED IMPORTANCE (the headline output):
  Individual SHAP is misleading here because the lag features are strongly
  correlated (tmean_7/14/28d, prcp_7/14/28d) -- SHAP splits credit between them,
  so temperature looks weaker than it is. We therefore sum the SIGNED SHAP values
  within each block PER ROW (SHAP is additive, so this recovers the block's net
  contribution), then take the mean absolute across rows. Summing absolutes
  instead would double-count opposing within-block effects.

Groups: temperature, precipitation, moisture (VPD/NDWI), vegetation (EVI),
seasonality, land_cover, terrain.
"""
from typing import Dict, Sequence
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

RANDOM_STATE = 42

# ----- feature grouping -----------------------------------------------------
# NOTE: elevation/slope are static but are TERRAIN, not land COVER. Kept
# separate by default; set MERGE_TERRAIN_INTO_LANDCOVER = True to fold them in.
MERGE_TERRAIN_INTO_LANDCOVER = False

GROUP_DEFS: Dict[str, Sequence[str]] = {
    "temperature":   ["tmean_mean_7d", "tmean_mean_14d", "tmean_mean_28d",
                      "tmax_max_14d", "tmin_mean_14d", "tmin_min_14d"],
    "precipitation": ["prcp_sum_7d", "prcp_sum_14d", "prcp_sum_28d"],
    "moisture":      ["vpd_mean_14d", "ndwi_level"],
    "vegetation":    ["evi_level"],
    "seasonality":   ["sin_doy", "cos_doy"],
    "terrain":       ["elevation", "slope"],
}
GROUP_ORDER = ["temperature", "precipitation", "moisture", "vegetation",
               "seasonality", "land_cover", "terrain"]


def assign_groups(feats: Sequence[str]) -> Dict[str, str]:
    """feature -> group. Anything starting pct_ falls to land_cover; anything
    unmatched is reported so nothing is silently dropped."""
    m = {}
    for g, cols in GROUP_DEFS.items():
        for c in cols:
            if c in feats:
                m[c] = g
    for f in feats:
        if f not in m:
            m[f] = "land_cover" if f.startswith("pct_") else "other"
    if MERGE_TERRAIN_INTO_LANDCOVER:
        m = {k: ("land_cover" if v == "terrain" else v) for k, v in m.items()}
    unknown = [f for f, g in m.items() if g == "other"]
    if unknown:
        print(f"[shap] WARNING ungrouped features -> 'other': {unknown}", flush=True)
    return m


# ----- sampling -------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, n: int, target: str = "presence"):
    """Subsample preserving the presence/absence balance (SHAP plots need a
    representative sample, not the full table). Written without groupby.apply so
    it is free of the pandas grouping-columns FutureWarning."""
    if len(df) <= n:
        return df.copy()
    frac = n / len(df)
    parts = []
    for _, d in df.groupby(target, sort=False):
        k = min(len(d), max(1, int(round(len(d) * frac))))
        parts.append(d.sample(k, random_state=RANDOM_STATE))
    return pd.concat(parts, ignore_index=True)


# ----- explainers -----------------------------------------------------------
def explain_tree(model, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP. Returns (n_rows, n_features) for the positive class.

    XGBoost is routed through its OWN pred_contribs=True rather than
    shap.TreeExplainer: shap's XGBoost parser breaks on newer XGBoost, which
    stores base_score as an array ("could not convert string to float:
    '[4.3e-01]'"). pred_contribs is the same exact TreeSHAP, computed inside
    XGBoost, so it is immune to shap/xgboost version drift.
    """
    try:
        import xgboost as xgb
        if isinstance(model, xgb.XGBModel):
            contribs = model.get_booster().predict(
                xgb.DMatrix(X, missing=np.nan), pred_contribs=True)
            return np.asarray(contribs)[:, :-1]      # last column = bias/base
    except ImportError:
        pass
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X, check_additivity=False)
    sv = np.array(sv)
    if sv.ndim == 3:                 # (n, features, classes) -> positive class
        sv = sv[:, :, 1]
    return sv


def explain_permutation(model, X: pd.DataFrame, feats, n_background: int = 50):
    """Model-agnostic (approximate) SHAP for MaxEnt. Slow: ~0.9 s/row."""
    f = lambda Z: model.predict_proba(pd.DataFrame(Z, columns=list(feats)))[:, 1]
    bg = shap.utils.sample(X, min(n_background, len(X)), random_state=RANDOM_STATE)
    ex = shap.PermutationExplainer(f, bg)
    return np.array(ex(X).values)


# ----- importance tables ----------------------------------------------------
def individual_importance(sv: np.ndarray, feats: Sequence[str]) -> pd.DataFrame:
    imp = np.abs(sv).mean(axis=0)
    d = pd.DataFrame({"feature": list(feats), "mean_abs_shap": imp})
    d["group"] = d["feature"].map(assign_groups(feats))
    d["pct_of_total"] = 100 * d.mean_abs_shap / d.mean_abs_shap.sum()
    return d.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def grouped_importance(sv: np.ndarray, feats: Sequence[str]) -> pd.DataFrame:
    """Sum SIGNED shap within each group per row, then mean |.| across rows.
    This is what defuses the correlated-lag credit splitting."""
    gmap = assign_groups(feats)
    sv_df = pd.DataFrame(sv, columns=list(feats))
    rows = {}
    for g in sorted(set(gmap.values())):
        cols = [f for f in feats if gmap[f] == g]
        rows[g] = np.abs(sv_df[cols].sum(axis=1)).mean()   # signed sum, then |.|
    d = pd.DataFrame({"group": list(rows), "mean_abs_group_shap": list(rows.values())})
    d["n_features"] = d["group"].map(lambda g: sum(1 for f in feats if gmap[f] == g))
    d["pct_of_total"] = 100 * d.mean_abs_group_shap / d.mean_abs_group_shap.sum()
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    return d.sort_values("mean_abs_group_shap", ascending=False).reset_index(drop=True)


# ----- plots ----------------------------------------------------------------
def plot_beeswarm(sv, X, title, path, max_display=20):
    plt.figure()
    shap.summary_plot(sv, X, max_display=max_display, show=False)
    plt.title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_bar_individual(imp: pd.DataFrame, title, path, top=20):
    d = imp.head(top).iloc[::-1]
    colors = {"temperature": "#d7301f", "precipitation": "#2166ac", "moisture": "#41b6c4",
              "vegetation": "#238b45", "seasonality": "#6a51a3", "land_cover": "#b8860b",
              "terrain": "#777777", "other": "#cccccc"}
    plt.figure(figsize=(8, max(4, 0.32*len(d))))
    plt.barh(d.feature, d.mean_abs_shap, color=[colors.get(g, "#999") for g in d.group],
             edgecolor="black", linewidth=0.4)
    plt.xlabel("mean |SHAP| (contribution to predicted suitability)")
    plt.title(title, fontsize=10); plt.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_bar_grouped(gimp: pd.DataFrame, title, path):
    d = gimp.iloc[::-1]
    colors = {"temperature": "#d7301f", "precipitation": "#2166ac", "moisture": "#41b6c4",
              "vegetation": "#238b45", "seasonality": "#6a51a3", "land_cover": "#b8860b",
              "terrain": "#777777", "other": "#cccccc"}
    plt.figure(figsize=(7.5, 4.5))
    plt.barh(d.group, d.mean_abs_group_shap,
             color=[colors.get(g, "#999") for g in d.group], edgecolor="black", linewidth=0.5)
    for y, (v, p, n) in enumerate(zip(d.mean_abs_group_shap, d.pct_of_total, d.n_features)):
        plt.text(v, y, f"  {p:.0f}%  (n={n})", va="center", fontsize=9)
    plt.xlabel("mean |summed SHAP| within block")
    plt.title(title, fontsize=10); plt.grid(axis="x", alpha=0.3)
    plt.xlim(0, d.mean_abs_group_shap.max()*1.25)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()