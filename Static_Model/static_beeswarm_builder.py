"""
shap_beeswarm_climate.py — beeswarm of SHAP values for the CLIMATE-only model,
showing just the climatic factors (temperature / precipitation / VPD).

SHAP values come straight from XGBoost (pred_contribs) so this never hits the
shap-vs-xgboost base_score bug. shap is used ONLY to draw the beeswarm from the
precomputed values.
"""
import numpy as np, pandas as pd
from pathlib import Path
try:
    import xgboost as xgb
except ImportError:
    xgb = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ----------------------------- config --------------------------------------
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

TRAINING_PARQUET = config['static_dir'] + "/static_training_table.parquet"
OUTPUT_DIR       = config['static_results_dir']
CLIMATE_PARQUET = config['static_dir'] + "/Florida_Static_SDM_allyears.parquet" # Update if file name differs
MODEL_PATH       = config['static_results_dir'] + "/static_xgb_climate_model.json"  
TARGET_COL       = "response"
# show only these (purely climatic). Set to None to show every model feature
# (which would also include elevation/slope if the model used them).
CLIMATE_PREFIXES = ("tmax", "tmin", "tmean", "prcp", "vpd")
MAX_DISPLAY      = 15
# ---------------------------------------------------------------------------

def main():
    if xgb is None:
        raise SystemExit("xgboost not installed — `pip install xgboost`.")
    try:
        import shap
    except ImportError:
        raise SystemExit("shap not installed — `pip install shap` (only the plot needs it).")

    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)

    model = xgb.XGBRegressor(); model.load_model(MODEL_PATH)
    feats = model.get_booster().feature_names           # exact training features/order

    df = pd.read_parquet(TRAINING_PARQUET).dropna(subset=[TARGET_COL])
    X = df[feats]

    # SHAP values from XGBoost itself (version-proof)
    contribs = model.get_booster().predict(
        xgb.DMatrix(X, feature_names=feats, missing=np.nan), pred_contribs=True)
    sv = contribs[:, :-1]                               # drop bias column

    # keep only the climatic columns for display
    if CLIMATE_PREFIXES is not None:
        keep = [i for i, f in enumerate(feats) if f.startswith(CLIMATE_PREFIXES)]
        sv_show, X_show = sv[:, keep], X.iloc[:, keep]
    else:
        sv_show, X_show = sv, X
    print(f"[shap] showing {X_show.shape[1]} climatic features over {len(X_show):,} rows")

    plt.figure()
    shap.summary_plot(sv_show, X_show, show=False, max_display=MAX_DISPLAY,
                      plot_type="dot")            # dot = beeswarm
    plt.title("SHAP — climatic drivers of Cx. nigripalpus abundance")
    plt.tight_layout()
    plt.savefig(out / "shap_beeswarm_climate.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[shap] wrote {out/'shap_beeswarm_climate.png'}")

if __name__ == "__main__":
    main()