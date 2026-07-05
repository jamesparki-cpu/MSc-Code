from __future__ import annotations
"""
render_maps.py  —  STAGE 6c of the WEEKLY suitability pipeline.

Renders the surfaces from 6b as thesis-ready figures. Model-agnostic: any
prob_* column in surface_predictions.parquet becomes a series, so MaxEnt joins
for free later.

Three series (each = 53 weekly PNGs + an assembled GIF):
  * xgboost        calibrated suitability, opacity = sampling-proximity fade
  * random_forest  same, matched model
  * agreement      |xgb - rf| (model concordance / uncertainty; low = agree)

Plus ONE interactive HTML with a WEEK SLIDER (exploration / viva) -- the GIF
auto-plays, the slider lets you scrub to any week.

STYLE matches static_map_builder.py exactly (RdYlBu_r, square markers, latitude-
corrected aspect) for figure continuity.

CAPTION DISCIPLINE: opacity encodes proximity to surveillance data (a data-
support cue), NOT model certainty. Faded = extrapolated. Temporal validation is
strong (ROC ~0.85, calibrated); spatial transfer is limited (ROC ~0.69). The
titles state "climatological mean conditions 2013-2018".

OUTPUT (to static_results_dir/maps/)
  xgboost/week_01.png ... + xgboost.gif   (and random_forest/, agreement/)
  suitability_explorer.html               (week slider over all three)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio.v2 as imageio

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)
STATIC_DIR = Path(config["static_dir"])
RESULTS    = Path(config["weekly_results_dir"])
SURFACES   = RESULTS / "surface_predictions.parquet"
OUTPUT_DIR = RESULTS / "maps"

GRID_ID_COL = "Grid_ID"
CMAP_SUIT  = "RdYlBu_r"          # matches the static map builder
CMAP_AGREE = "magma_r"          # dark = high disagreement
MARKER_SIZE = 6                  # square marker; 8,850 cells at 5 km reads as raster
DPI = 130
GIF_FPS = 4

# opacity source: "opacity" (floored, whole state visible) or "confidence_raw"
# (honest fade to ~0). Your default = floored; swap for the fade-to-zero figure.
OPACITY_COL = "opacity"

def _iso_monday_label(week):
    from datetime import date
    try:   return date.fromisocalendar(2015, int(week), 1).strftime("%d %b")
    except ValueError: return f"wk{int(week)}"
# ============================================================================

def log(m): print(m, flush=True)


def render_frame(sub, value_col, title, cmap, vmin, vmax, path,
                 use_opacity=True, cbar_label="predicted suitability"):
    fig, ax = plt.subplots(figsize=(7, 8))
    alpha = sub[OPACITY_COL].to_numpy() if use_opacity else None
    sc = ax.scatter(sub["lon"], sub["lat"], c=sub[value_col], s=MARKER_SIZE, marker="s",
                    cmap=cmap, norm=Normalize(vmin, vmax), linewidths=0, alpha=alpha)
    ax.set_aspect(1 / np.cos(np.radians(sub["lat"].mean())))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(title, fontsize=11)
    # colourbar at full alpha (independent of the per-point fade)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7); cb.set_label(cbar_label)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def render_series(surf, value_col, folder, title_stub, cmap, vmin, vmax,
                  use_opacity=True, cbar_label="predicted suitability"):
    d = OUTPUT_DIR / folder; d.mkdir(parents=True, exist_ok=True)
    frames = []
    for wk in sorted(surf["iso_week"].unique()):
        sub = surf[surf["iso_week"] == wk]
        title = (f"Cx. nigripalpus — {title_stub}\n"
                 f"ISO week {int(wk):02d} (~{_iso_monday_label(wk)}) · "
                 f"climatological mean 2013–2018")
        p = d / f"week_{int(wk):02d}.png"
        render_frame(sub, value_col, title, cmap, vmin, vmax, p, use_opacity, cbar_label)
        frames.append(imageio.imread(p))
    gif = OUTPUT_DIR / f"{folder}.gif"
    imageio.mimsave(gif, frames, fps=GIF_FPS, loop=0)
    log(f"[6c] {folder}: 53 PNGs + {folder}.gif")
    return gif


def build_html_slider(surf):
    """One interactive HTML: week slider, toggle XGB/RF/agreement. Reads the
    already-rendered PNGs so it's a pure viewer (no recompute)."""
    weeks = sorted(int(w) for w in surf["iso_week"].unique())
    labels = {w: _iso_monday_label(w) for w in weeks}
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cx. nigripalpus weekly suitability</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
 .row{{display:flex;gap:8px;align-items:center;margin:12px 0}}
 img{{max-width:520px;border:1px solid #ddd}} button{{padding:6px 12px}}
 button.active{{background:#2166ac;color:#fff}}
 input[type=range]{{width:420px}}
</style></head><body>
<h2>Cx. nigripalpus — weekly habitat suitability (climatological 2013–2018)</h2>
<div class="row">
 <button id="b_xgboost" class="active" onclick="setSeries('xgboost')">XGBoost</button>
 <button id="b_random_forest" onclick="setSeries('random_forest')">Random Forest</button>
 <button id="b_agreement" onclick="setSeries('agreement')">Model agreement</button>
</div>
<div class="row">
 <label>ISO week: <b id="wl"></b></label>
 <input type="range" id="wk" min="{weeks[0]}" max="{weeks[-1]}" value="34" step="1"
        oninput="update()">
</div>
<img id="map" src="">
<p style="max-width:520px;color:#555;font-size:13px">Opacity = proximity to
surveillance data (faded = extrapolated, lower support). Temporal validation
strong (ROC~0.85, calibrated); spatial transfer limited (ROC~0.69).</p>
<script>
 const labels = {json.dumps(labels)};
 let series = "xgboost";
 function pad(n){{return String(n).padStart(2,'0');}}
 function setSeries(s){{series=s;
   for (const k of ["xgboost","random_forest","agreement"])
     document.getElementById("b_"+k).classList.toggle("active",k===s);
   update();}}
 function update(){{const w=document.getElementById("wk").value;
   document.getElementById("wl").textContent = w+" (~"+labels[w]+")";
   document.getElementById("map").src = series+"/week_"+pad(w)+".png";}}
 update();
</script></body></html>"""
    (OUTPUT_DIR / "suitability_explorer.html").write_text(html)
    log(f"[6c] suitability_explorer.html (week slider over all three series)")


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    surf = pd.read_parquet(SURFACES)
    log(f"[6c] {len(surf):,} cell-weeks | opacity source = '{OPACITY_COL}'")

    render_series(surf, "prob_xgboost", "xgboost",
                  "predicted suitability (XGBoost, calibrated)", CMAP_SUIT, 0, 1)
    render_series(surf, "prob_random_forest", "random_forest",
                  "predicted suitability (Random Forest, calibrated)", CMAP_SUIT, 0, 1)
    render_series(surf, "agree_abs_diff", "agreement",
                  "model disagreement |XGB − RF|", CMAP_AGREE, 0,
                  float(surf["agree_abs_diff"].quantile(0.99)),
                  use_opacity=False, cbar_label="|XGB − RF| (dark = disagree)")

    build_html_slider(surf)
    log(f"[done] maps written to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()