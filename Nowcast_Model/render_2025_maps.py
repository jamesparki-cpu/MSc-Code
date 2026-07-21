from __future__ import annotations
"""
render_2025_maps.py  —  render the 2025 out-of-period prediction surfaces.

Reads surface_2025_predictions.parquet (from predict_2025_maps.py) and renders,
for EVERY model and EVERY week, TWO versions:
  * full     — statewide, opacity = sampling-proximity fade (whole state visible)
  * masked   — hard-masked to the surveyed footprint (cells within MASK_KM of a
               historical trap); beyond that is not drawn.

These are DEPLOYMENT / CAPABILITY maps, NOT validated results: there are no 2025
catches, so NO metric and NO observation overlay. Captions say so explicitly.

The mask is defined by SURVEILLANCE geometry (trap proximity), never by
prediction values — it narrows the map's SCOPE to where the model is supported,
it does not select on outcome.

STYLE matches the other maps (RdYlBu_r, square markers, latitude aspect).

OUTPUT (to config predict2025_dir/maps)
  <model>/week<NN>_full.png  and  <model>/week<NN>_masked.png
"""
import json, re
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR = Path(cfg.get("weekly_xg_dir", "."))
PRED_DIR = Path(cfg["nowcast_dir"])
SURFACE  = PRED_DIR / "surface_2025_predictions.parquet"
OUT_DIR  = PRED_DIR / "2025_maps"

PRED_YEAR = 2025
MASK_KM = 40.0                    # hard-mask threshold: cells within this of a trap
SEASON_WINDOW = 2
CMAP = "RdYlBu_r"; MARKER = 6; DPI = 130

PRETTY = {"xgboost": "XGBoost", "random_forest": "Random Forest",
          "maxent_targetgroup": "MaxEnt (target-group)", "maxent_vanilla": "MaxEnt (vanilla)"}

def log(m): print(m, flush=True)
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
# ============================================================================


def trap_mask(sub, week, sampled):
    """True = keep (within MASK_KM of a trap active that season). Surveillance
    geometry only — independent of the predicted values."""
    coslat = np.cos(np.radians(sub["lat"].mean()))
    s = {((w-1) % 53)+1 for w in range(week-SEASON_WINDOW, week+SEASON_WINDOW+1)}
    samp = sampled[sampled.iso_week.isin(s)]
    if samp.empty:
        return np.zeros(len(sub), dtype=bool)
    tree = cKDTree(np.c_[samp.cell_lon*coslat*111, samp.cell_lat*111])
    dkm, _ = tree.query(np.c_[sub["lon"]*coslat*111, sub["lat"]*111], k=1)
    return dkm <= MASK_KM


def render(sub, prob_col, title, path, use_opacity=True):
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    alpha = sub["opacity"].to_numpy() if use_opacity else None
    ax.scatter(sub["lon"], sub["lat"], c=sub[prob_col], s=MARKER, marker="s",
               cmap=CMAP, norm=Normalize(0, 1), linewidths=0, alpha=alpha)
    ax.set_aspect(1 / np.cos(np.radians(sub["lat"].mean())))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(title, fontsize=10.5)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1)); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7).set_label("predicted suitability")
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)
    log(f"[2025-maps] {path.parent.name}/{path.name}")


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    surf = pd.read_parquet(SURFACE)
    prob_cols = [c for c in surf.columns if c.startswith("prob_")]
    # historical trap footprint for the mask (2013-2018 sampled cell-weeks)
    table = pd.read_parquet(DATA_DIR / "weekly_model_table.parquet")
    sampled = table.groupby(["Grid_ID", "iso_week"])[["cell_lat", "cell_lon"]].first().reset_index()

    for pc in prob_cols:
        model = pc.replace("prob_", ""); label = PRETTY.get(model, model)
        d = OUT_DIR / model; d.mkdir(parents=True, exist_ok=True)
        for week in sorted(surf["iso_week"].unique()):
            sub = surf[surf["iso_week"] == week].copy()
            mon = pd.Timestamp(date.fromisocalendar(PRED_YEAR, int(week), 1))
            base_title = (f"Cx. nigripalpus — predicted suitability ({label})\n"
                          f"ISO week {int(week)} {PRED_YEAR} (~{mon.strftime('%d %b')}) · "
                          f"out-of-period prediction — UNVALIDATED (no {PRED_YEAR} surveillance)")
            # full
            render(sub, pc, base_title, d / f"week{int(week):02d}_full.png")
            # masked
            keep = trap_mask(sub, int(week), sampled)
            title_m = (f"Cx. nigripalpus — predicted suitability ({label}), surveyed footprint\n"
                       f"ISO week {int(week)} {PRED_YEAR} (~{mon.strftime('%d %b')}) · "
                       f"masked to <{MASK_KM:.0f} km of a trap · out-of-period, UNVALIDATED")
            render(sub[keep], pc, title_m, d / f"week{int(week):02d}_masked.png")
            log(f"[2025-maps] {model} week {int(week)}: full ({len(sub)}) + masked ({int(keep.sum())})")

    log(f"[done] 2025 maps -> {OUT_DIR}  ({len(prob_cols)} models x weeks x [full, masked])")


if __name__ == "__main__":
    run()