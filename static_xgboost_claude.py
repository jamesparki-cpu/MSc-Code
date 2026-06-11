"""
static_xgboost.py — XGBoost baseline for Cx. nigripalpus abundance (panel: cell x year)

Implements:
  * correct training-table path + keep NaN predictors (XGBoost handles them)
  * sample weighting by sqrt(n_events) (dampens huge-catch cell-years)
  * TWO honest CV schemes, reported side by side:
        - temporal  : Leave-One-Year-Out  ("known place, new year")
        - spatial   : Leave-One-Block-Out ("new place")  <- what the map relies on
  * NO eval_set inside CV folds (avoids selecting on the test fold)
  * SHAP feature importance with correlated-feature grouping (falls back to gain)
  * saves out-of-fold predictions, importances, the final model, and plots
"""
"""
static_xgboost.py — XGBoost baseline for Cx. nigripalpus abundance (panel: cell x year)

Implements:
  * correct training-table path + keep NaN predictors (XGBoost handles them)
  * sample weighting by sqrt(n_events) (dampens huge-catch cell-years)
  * TWO honest CV schemes, reported side by side:
        - temporal  : Leave-One-Year-Out  ("known place, new year")
        - spatial   : Leave-One-Block-Out ("new place")  <- what the map relies on
  * NO eval_set inside CV folds (avoids selecting on the test fold)
  * SHAP feature importance with correlated-feature grouping (falls back to gain)
  * saves out-of-fold predictions, importances, the final model, and plots
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except ImportError:
    xgb = None      # clear message raised in make_xgb() if actually used
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")           # save figures without a display
import matplotlib.pyplot as plt

# ----------------------------- config --------------------------------------
# The cleaning script writes this into ./outputs/. Point at wherever it landed.
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Assign the data folder dynamically from your config
TRAINING_PARQUET = config['static_dir'] + "/static_training_table.parquet"
OUTPUT_DIR       = config['static_results_dir']

N_SPATIAL_BLOCKS = 6            # regions for Leave-One-Block-Out spatial CV
RANDOM_STATE     = 42

METADATA_COLS = ["Grid_ID", "year", "cell_lat", "cell_lon",
                 "n_events", "mean_count", "low_n_flag"]
TARGET_COL    = "response"

XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, objective="reg:squarederror",
    importance_type="gain",      # match the "Gain" label; make it explicit
)

def make_xgb():
    if xgb is None:
        raise ImportError("xgboost is not installed — run `pip install xgboost`.")
    return xgb.XGBRegressor(**XGB_PARAMS)

# ----------------------------- data ----------------------------------------
def load_data(path):
    df = pd.read_parquet(path)
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)  # ONLY require a response
    print(f"[data] {before} rows -> {len(df)} with a response "
          f"(NaN predictors kept; XGBoost learns a default split direction)")
    return df

def get_predictors(df):
    drop = set(METADATA_COLS) | {TARGET_COL, "block"}
    cols = [c for c in df.columns if c not in drop]
    print(f"[data] {len(cols)} predictors (lat/lon, mean_count, block excluded -> "
          f"model explains abundance via climate, not geography)")
    return cols

def add_spatial_blocks(df, k=N_SPATIAL_BLOCKS):
    """Cluster whole CELLS into k contiguous regions so a held-out block shares
    no cell with the training set (true spatial transfer test)."""
    cells = df.groupby("Grid_ID")[["cell_lat", "cell_lon"]].first()
    cells["block"] = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                            n_init=10).fit_predict(cells[["cell_lat", "cell_lon"]])
    df = df.merge(cells["block"], on="Grid_ID")
    print(f"[cv] {df['Grid_ID'].nunique()} cells grouped into {k} spatial blocks "
          f"(sizes: {df.groupby('block')['Grid_ID'].nunique().to_dict()})")
    return df

# ----------------------------- CV ------------------------------------------
def run_cv(make_model, X, y, groups, weights, label):
    """Model-agnostic Leave-One-Group-Out. Returns out-of-fold predictions.
    No eval_set: every fold trains the full model, nothing is tuned on test."""
    logo = LeaveOneGroupOut()
    oof = np.full(len(X), np.nan)
    print(f"\n[cv:{label}] {logo.get_n_splits(groups=groups)} folds")
    for fold, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        model = make_model()
        model.fit(X.iloc[tr], y.iloc[tr], sample_weight=weights.iloc[tr])
        oof[te] = model.predict(X.iloc[te])
        held = groups.iloc[te].iloc[0]
        print(f"  fold {fold} (held out {label}={held}): "
              f"R2={r2_score(y.iloc[te], oof[te]):.3f}  n_test={len(te)}")
    rmse = np.sqrt(mean_squared_error(y, oof))
    r2 = r2_score(y, oof)
    print(f"[cv:{label}] OVERALL  R2={r2:.3f}  RMSE={rmse:.3f} (log1p units)")
    return oof, r2, rmse

# ----------------------------- importance ----------------------------------
def correlated_groups(X, thresh=0.9):
    """Cluster predictors whose |corr| >= thresh so importance can be read by group."""
    corr = X.corr().abs()
    remaining = list(X.columns)
    groups = []
    while remaining:
        seed = remaining.pop(0)
        grp = [seed] + [c for c in remaining if corr.loc[seed, c] >= thresh]
        remaining = [c for c in remaining if c not in grp]
        groups.append(grp)
    return groups

def shap_importance(model, X, out):
    """SHAP values straight from XGBoost (pred_contribs) — version-proof, no
    dependency on shap's model loader. shap is used only (optionally) to draw
    the beeswarm from the precomputed values."""
    if xgb is None:
        return gain_importance(model, X, out)
    booster = model.get_booster()
    dm = xgb.DMatrix(X, feature_names=list(X.columns), missing=np.nan)
    contribs = booster.predict(dm, pred_contribs=True)   # (n, n_feat+1); last col = bias
    sv = contribs[:, :-1]
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)

    # grouped importance: pool |SHAP| within correlated clusters
    grouped = []
    for grp in correlated_groups(X):
        grouped.append(("; ".join(grp[:3]) + ("..." if len(grp) > 3 else ""),
                        float(mean_abs[grp].sum())))
    gtab = (pd.DataFrame(grouped, columns=["feature_group", "mean_abs_shap"])
              .sort_values("mean_abs_shap", ascending=False))
    gtab.to_csv(f"{out}/shap_importance_grouped.csv", index=False)
    mean_abs.sort_values(ascending=False).to_csv(f"{out}/shap_importance_single.csv")
    print("\n[imp] top grouped SHAP importance (correlated features pooled):")
    print(gtab.head(8).to_string(index=False))

    # beeswarm from precomputed values (no TreeExplainer); skip quietly if shap absent
    try:
        import shap
        plt.figure()
        shap.summary_plot(sv, X, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(f"{out}/shap_beeswarm.png", dpi=150); plt.close()
    except Exception as e:
        print(f"[imp] beeswarm skipped ({type(e).__name__}); CSVs still written.")
    return gtab

def gain_importance(model, X, out):
    imp = (pd.DataFrame({"feature": X.columns,
                         "gain": model.feature_importances_})
             .sort_values("gain", ascending=False))
    imp.to_csv(f"{out}/gain_importance.csv", index=False)
    top = imp.tail(10) if len(imp) > 10 else imp
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["gain"], color="skyblue")
    plt.title("Top environmental predictors — Cx. nigripalpus abundance")
    plt.xlabel("XGBoost gain"); plt.tight_layout()
    plt.savefig(f"{out}/gain_importance.png", dpi=150); plt.close()
    return imp

# ----------------------------- main ----------------------------------------
def main():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    df = load_data(TRAINING_PARQUET)
    df = add_spatial_blocks(df)

    predictors = get_predictors(df)
    X = df[predictors]
    y = df[TARGET_COL]
    weights = np.sqrt(df["n_events"]).rename("w")   # sqrt dampens huge-catch cells

    # --- the two honest CV runs ---
    oof_t, r2_t, _ = run_cv(make_xgb, X, y, df["year"],  weights, "year")
    oof_s, r2_s, _ = run_cv(make_xgb, X, y, df["block"], weights, "block")

    print("\n=== HONEST PERFORMANCE SUMMARY ===")
    print(f"  temporal (new year, known place): R2 = {r2_t:.3f}")
    print(f"  spatial  (new place)            : R2 = {r2_s:.3f}")
    print("  -> the gap is the finding: temporal transfer is the easy case; "
          "spatial transfer is what the suitability map actually needs.")

    # save out-of-fold predictions for the mapping/diagnostic step
    df.assign(oof_temporal=oof_t, oof_spatial=oof_s)[
        ["Grid_ID", "year", "cell_lat", "cell_lon", TARGET_COL,
         "n_events", "oof_temporal", "oof_spatial"]
    ].to_csv(out / "oof_predictions.csv", index=False)

    # --- final model on ALL data (for importances + prediction surface) ---
    final = make_xgb()
    final.fit(X, y, sample_weight=weights)
    final.save_model(str(out / "static_xgb_model.json"))
    shap_importance(final, X, str(out))
    print(f"\n[done] wrote model, oof_predictions.csv, importances + plots to {out}/")

if __name__ == "__main__":
    main()