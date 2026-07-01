"""
static_xgboost_climate.py — DIAGNOSTIC pass: climate-only, decorrelated, tighter.

Purpose: test whether removing the land-cover 'fingerprint' features and the
redundant correlated climate twins — plus stronger regularisation tuned at the
SPATIAL fold — recovers any spatial generalisation. This does NOT rebuild the
parquet; it simply selects a subset of predictors at model time (reversible).

Realistic expectation: temporal R2 will DROP (the memorisation fuel is gone) and
spatial R2 should rise toward ~0. That trade = the model becoming honest, not a fix.
Closing the spatial gap properly is the weekly dynamic model's job.
"""
import numpy as np, pandas as pd
from pathlib import Path
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# reuse the validated machinery from the main script
from Static_Model.static_xgboost_claude import (load_data, add_spatial_blocks, run_cv,
                            correlated_groups, shap_importance,
                            TARGET_COL, METADATA_COLS, OUTPUT_DIR, TRAINING_PARQUET)

# ---- feature policy (edit here, not the parquet) --------------------------
DROP_PREFIXES = ("pct_",)     # remove land cover -> align with CLIMATIC suitability
KEEP_VEGETATION = False       # True re-adds EVI*/NDWI* (greenness; more land-surface)
CORR_THRESH = 0.90            # collapse climate families: keep one per correlated cluster

# ---- regularisation, tuned for SPATIAL transfer (shallower, stronger penalty) ----
TIGHT_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, max_depth=3,   # was 5
    min_child_weight=5,                                  # was default 1
    subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, # added L2
    random_state=42, objective="reg:squarederror", importance_type="gain",
)
def make_xgb_tight():
    if xgb is None:
        raise ImportError("xgboost not installed — `pip install xgboost`.")
    return xgb.XGBRegressor(**TIGHT_PARAMS)

def decorrelate(X, thresh=CORR_THRESH):
    """Greedy: keep a feature only if it isn't >=thresh correlated with one already kept."""
    corr = X.corr().abs()
    kept = []
    for c in X.columns:
        if all(corr.loc[c, k] < thresh for k in kept):
            kept.append(c)
    return kept

def select_climate_predictors(df):
    cols = [c for c in df.columns
            if c not in set(METADATA_COLS) | {TARGET_COL, "block"}]
    cols = [c for c in cols if not c.startswith(DROP_PREFIXES)]          # no land cover
    if not KEEP_VEGETATION:
        cols = [c for c in cols if not (c.startswith("EVI") or c.startswith("NDWI"))]
    kept = decorrelate(df[cols])
    dropped_corr = [c for c in cols if c not in kept]
    print(f"[features] {len(cols)} climate/topo candidates -> {len(kept)} after "
          f"decorrelation (dropped {len(dropped_corr)} redundant twins)")
    print(f"[features] using: {kept}")
    return kept

def main():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    df = load_data(TRAINING_PARQUET)
    df = add_spatial_blocks(df)

    predictors = select_climate_predictors(df)
    X = df[predictors]; y = df[TARGET_COL]
    weights = np.sqrt(df["n_events"]).rename("w")

    oof_t, r2_t, _ = run_cv(make_xgb_tight, X, y, df["year"],  weights, "year")
    oof_s, r2_s, _ = run_cv(make_xgb_tight, X, y, df["block"], weights, "block")

    print("\n=== CLIMATE-ONLY / DECORRELATED / REGULARISED ===")
    print(f"  temporal R2 = {r2_t:.3f}   (expect lower than the 46-feature run)")
    print(f"  spatial  R2 = {r2_s:.3f}   (the number that matters; hope: toward 0+)")
    print(f"  predictors  = {len(predictors)} (was 46)")

    df.assign(oof_temporal=oof_t, oof_spatial=oof_s)[
        ["Grid_ID", "year", "cell_lat", "cell_lon", TARGET_COL,
         "n_events", "oof_temporal", "oof_spatial"]
    ].to_csv(out / "oof_predictions_climate.csv", index=False)

    final = make_xgb_tight(); final.fit(X, y, sample_weight=weights)
    final.save_model(str(out / "static_xgb_climate_model.json"))
    shap_importance(final, X, str(out))
    print(f"\n[done] wrote climate-only model + oof + importances to {out}/")

if __name__ == "__main__":
    main()