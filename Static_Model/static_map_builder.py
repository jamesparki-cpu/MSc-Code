"""
map_static.py — (1) label the spatial CV blocks (find what block 5 is), and
                (2) render a proof-of-concept abundance heat map from a saved model.

Dependency-light: matplotlib only (no geopandas). Cells are drawn as small squares
at their centroids, which at 5 km reads as a raster surface.

CAPTION YOUR FIGURE HONESTLY: this is the temporally-validated model
(temporal R2~0.5-0.6); spatial CV is negative, so the map is a pipeline
proof-of-concept, not a validated suitability claim.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import xgboost as xgb
except ImportError:
    xgb = None
import json

# ----------------------------- config --------------------------------------
# ----------------------------- config --------------------------------------
# The cleaning script writes this into ./outputs/. Point at wherever it landed.
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Assign the data folder dynamically from your config
TRAINING_PARQUET = config['static_dir'] + "/static_training_table.parquet"
OUTPUT_DIR       = config['static_results_dir']
CLIMATE_PARQUET = config['static_dir'] + "/Florida_Static_SDM_allyears.parquet" # Update if file name differs
# vvvvvvv CURRENTLY CLIMATE ONLY (will need changes to output names etc at bottom) vvvvvvvvvvvvvv
MODEL_PATH       = config['static_results_dir'] + "/static_xgb_climate_model.json"           # the model whose surface you want to render
N_SPATIAL_BLOCKS = 6
RANDOM_STATE     = 42
GRID_ID_COL, YEAR_COL = "Grid_ID", "year"

def region_label(lat, lon):
    """Very rough Florida region from a centroid (approximate; eyeball on a map)."""
    if lat < 25.3:  return "Florida Keys / far south tip"
    if lat < 27.0:  return "South FL (Everglades / SE coast)" if lon > -81.8 else "SW coast (Naples area)"
    if lat < 29.0:  return "Central Florida"
    if lon < -83.5: return "Western Panhandle"
    if lon < -82.5: return "Big Bend / north-central"
    return "Northeast Florida"

# Colour ramp for the heat map (low -> high abundance).
# "RdYlBu_r" runs blue -> yellow -> red (clean, perceptual, built-in).
# For a DARKER navy start through yellow into deep red, use CMAP = DARK_BYR below.
from matplotlib.colors import LinearSegmentedColormap
DARK_BYR = LinearSegmentedColormap.from_list(
    "dark_blue_yellow_red",
    ["#01184f", "#2166ac", "#67a9cf", "#ffffbf", "#fdae61", "#d7301f", "#7f0000"])
CMAP = "RdYlBu_r"        # <- swap to DARK_BYR for the darker navy/red version


# ----------------------------- (1) block labels ----------------------------
def label_blocks(train_path, df=None):
    if df is None:
        df = pd.read_parquet(train_path)
    cells = df.groupby(GRID_ID_COL)[["cell_lat", "cell_lon"]].first()
    cells["block"] = KMeans(n_clusters=N_SPATIAL_BLOCKS, random_state=RANDOM_STATE,
                            n_init=10).fit_predict(cells[["cell_lat", "cell_lon"]])
    print("=== spatial CV block locations ===")
    for b, g in cells.groupby("block"):
        clat, clon = g["cell_lat"].mean(), g["cell_lon"].mean()
        print(f"  block {b}: {len(g):3d} cells | centroid ({clat:.2f}, {clon:.2f}) | "
              f"lat {g.cell_lat.min():.2f}-{g.cell_lat.max():.2f} "
              f"lon {g.cell_lon.min():.2f}-{g.cell_lon.max():.2f} | "
              f"~{region_label(clat, clon)}")
    return cells
 
# ----------------------------- (2) heat map --------------------------------
def per_cell_climate(climate_path):
    """One static feature row per cell = mean of its yearly climate."""
    clim = pd.read_parquet(climate_path)
    feat = clim.drop(columns=[c for c in [YEAR_COL] if c in clim.columns])
    per_cell = feat.groupby(GRID_ID_COL).mean(numeric_only=True).reset_index()
    cent = per_cell[GRID_ID_COL].astype(str).str.extract(r'(-?\d+\.?\d*)_(-?\d+\.?\d*)$').astype(float)
    per_cell["lat"], per_cell["lon"] = cent[0], cent[1]
    return per_cell
 
def render_map(per_cell, preds, out):
    fig, ax = plt.subplots(figsize=(7, 8))
    sc = ax.scatter(per_cell["lon"], per_cell["lat"], c=preds, s=14, marker="s",
                    cmap=CMAP, linewidths=0)
    ax.set_aspect(1 / np.cos(np.radians(per_cell["lat"].mean())))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Cx. nigripalpus — predicted relative abundance\n"
                 "(proof-of-concept; temporally validated, spatial CV negative)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7); cb.set_label("predicted log(1+catch)")
    fig.tight_layout(); fig.savefig(f"{out}/heatmap_static_climate.png", dpi=160); plt.close()
    print(f"[map] wrote {out}/heatmap_static_climate.png ({len(per_cell):,} cells)")
 
def main():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    label_blocks(TRAINING_PARQUET)
 
    if xgb is None:
        print("[map] xgboost missing -> skipping map."); return
    model = xgb.XGBRegressor(); model.load_model(MODEL_PATH)
    feats = model.get_booster().feature_names         # exact training feature order
    per_cell = per_cell_climate(CLIMATE_PARQUET)
    missing = [f for f in feats if f not in per_cell.columns]
    if missing:
        print(f"[map] !! model expects features absent from climate parquet: {missing[:5]}...")
        return
    preds = model.predict(per_cell[feats])            # NaNs handled natively
    render_map(per_cell, preds, str(out))
    per_cell.assign(pred_log1p=preds)[[GRID_ID_COL, "lat", "lon", "pred_log1p"]] \
            .to_csv(out / "heatmap_climate_predictions.csv", index=False)
 
if __name__ == "__main__":
    main()