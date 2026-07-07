from __future__ import annotations
"""
render_maxent_maps.py  —  standalone renderer for the MaxEnt suitability maps.

Independent of the XGB/RF outputs (which live in a different folder): it reads
only the MaxEnt surfaces + the model table, and reproduces the IDENTICAL map
style and fade as render_maps.py. No merge step required.

For each surface_maxent_<variant>.parquet in maxent_dir it writes a 53-week PNG
series + GIF, plus one HTML week-slider over the MaxEnt variants. Cross-model
comparison (XGB/RF/MaxEnt metrics, agreement) is a SEPARATE script.

Self-contained because the MaxEnt surfaces carry only [Grid_ID, iso_week, prob]:
  * lat/lon are parsed from Grid_ID
  * the space x season opacity fade is RECOMPUTED here with the SAME parameters
    as predict_surfaces.py (6b), so the maps match XGB/RF exactly.

STYLE matches render_maps.py / static_map_builder.py (RdYlBu_r, square markers,
latitude-corrected aspect). Titles: "climatological mean 2013-2018".
"""
import json, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
import imageio.v2 as imageio

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)
MAXENT_DIR = Path(config["maxent_dir"])                 # MaxEnt surfaces + output
FEAT_DIR   = Path(config["weekly_xg_dir"])              # weekly_model_table (fade points)
OUTPUT_DIR = MAXENT_DIR / "maps"

GRID_ID_COL = "Grid_ID"
CMAP_SUIT   = "RdYlBu_r"          # matches the static map builder
MARKER_SIZE = 6
DPI = 130
GIF_FPS = 4

# fade params — MUST match predict_surfaces.py (6b) so MaxEnt maps == XGB/RF maps
SEASON_WINDOW = 2
FALLOFF_KM    = 120.0
OPACITY_FLOOR = 0.25
OPACITY_COL   = "opacity"         # swap to "confidence_raw" for the fade-to-zero version

PRETTY = {"maxent_vanilla": "MaxEnt (vanilla)",
          "maxent_targetgroup": "MaxEnt (target-group)"}

_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
def _lonlat(gids):
    lat = np.array([float(_GID.search(str(g)).group(1)) for g in gids])
    lon = np.array([float(_GID.search(str(g)).group(2)) for g in gids])
    return lat, lon

def _iso_monday_label(week):
    from datetime import date
    try:   return date.fromisocalendar(2015, int(week), 1).strftime("%d %b")
    except ValueError: return f"wk{int(week)}"
# ============================================================================

def log(m): print(m, flush=True)


def compute_fade(surf, sampled):
    """Recompute the space x season sampling-proximity opacity, identical to 6b."""
    glat, glon = _lonlat(surf[GRID_ID_COL].to_numpy())
    coslat = np.cos(np.radians(np.nanmean(glat)))
    gx, gy = glon*coslat*111.0, glat*111.0
    conf = np.zeros(len(surf))
    weeks = surf["iso_week"].to_numpy()
    for wk in sorted(surf["iso_week"].unique()):
        s = {((w-1) % 53)+1 for w in range(wk-SEASON_WINDOW, wk+SEASON_WINDOW+1)}
        samp = sampled[sampled["iso_week"].isin(s)]
        rows = np.where(weeks == wk)[0]
        if samp.empty:
            continue
        tree = cKDTree(np.c_[samp["cell_lon"].to_numpy()*coslat*111.0,
                             samp["cell_lat"].to_numpy()*111.0])
        d, _ = tree.query(np.c_[gx[rows], gy[rows]], k=1)
        conf[rows] = np.exp(-d / FALLOFF_KM)
    surf = surf.copy()
    surf["lat"], surf["lon"] = glat, glon
    surf["confidence_raw"] = conf
    surf["opacity"] = OPACITY_FLOOR + (1.0 - OPACITY_FLOOR) * conf
    return surf


def render_frame(sub, value_col, title, cmap, vmin, vmax, path,
                 use_opacity=True, cbar_label="predicted suitability"):
    fig, ax = plt.subplots(figsize=(7, 8))
    alpha = sub[OPACITY_COL].to_numpy() if use_opacity else None
    sc = ax.scatter(sub["lon"], sub["lat"], c=sub[value_col], s=MARKER_SIZE, marker="s",
                    cmap=cmap, norm=Normalize(vmin, vmax), linewidths=0, alpha=alpha)
    ax.set_aspect(1 / np.cos(np.radians(sub["lat"].mean())))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(title, fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7); cb.set_label(cbar_label)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def render_series(surf, value_col, folder, title_stub, cmap, vmin, vmax):
    d = OUTPUT_DIR / folder; d.mkdir(parents=True, exist_ok=True)
    frames = []
    for wk in sorted(surf["iso_week"].unique()):
        sub = surf[surf["iso_week"] == wk]
        title = (f"Cx. nigripalpus — {title_stub}\n"
                 f"ISO week {int(wk):02d} (~{_iso_monday_label(wk)}) · "
                 f"climatological mean 2013–2018")
        p = d / f"week_{int(wk):02d}.png"
        render_frame(sub, value_col, title, cmap, vmin, vmax, p)
        frames.append(imageio.imread(p))
    imageio.mimsave(OUTPUT_DIR / f"{folder}.gif", frames, fps=GIF_FPS, loop=0)
    log(f"[maxent-maps] {folder}: 53 PNGs + {folder}.gif")


def build_html_slider(weeks, variants):
    labels = {w: _iso_monday_label(w) for w in weeks}
    buttons = "\n".join(
        f'<button id="b_{v}" class="{"active" if i==0 else ""}" '
        f'onclick="setSeries(\'{v}\')">{PRETTY.get(v, v)}</button>'
        for i, v in enumerate(variants))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cx. nigripalpus — MaxEnt weekly suitability</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
 .row{{display:flex;gap:8px;align-items:center;margin:12px 0;flex-wrap:wrap}}
 img{{max-width:520px;border:1px solid #ddd}} button{{padding:6px 12px}}
 button.active{{background:#2166ac;color:#fff}} input[type=range]{{width:420px}}
</style></head><body>
<h2>Cx. nigripalpus — MaxEnt weekly suitability (climatological 2013–2018)</h2>
<div class="row">{buttons}</div>
<div class="row"><label>ISO week: <b id="wl"></b></label>
 <input type="range" id="wk" min="{weeks[0]}" max="{weeks[-1]}" value="34" step="1" oninput="update()"></div>
<img id="map" src="">
<p style="max-width:520px;color:#555;font-size:13px">Opacity = proximity to
surveillance data (faded = extrapolated). Titles state the calibrated model.</p>
<script>
 const labels={json.dumps(labels)}; const series={json.dumps(variants)};
 let cur=series[0];
 function pad(n){{return String(n).padStart(2,'0');}}
 function setSeries(s){{cur=s; for(const k of series)
   document.getElementById("b_"+k).classList.toggle("active",k===s); update();}}
 function update(){{const w=document.getElementById("wk").value;
   document.getElementById("wl").textContent=w+" (~"+labels[w]+")";
   document.getElementById("map").src=cur+"/week_"+pad(w)+".png";}}
 update();
</script></body></html>"""
    (OUTPUT_DIR / "maxent_explorer.html").write_text(html)
    log(f"[maxent-maps] maxent_explorer.html ({len(variants)} variant(s))")


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # trapped cell-weeks = fade anchor points (both presence and absence were sampled)
    tbl = pd.read_parquet(FEAT_DIR / "weekly_model_table.parquet")
    sampled = tbl.groupby([GRID_ID_COL, "iso_week"])[["cell_lat", "cell_lon"]].first().reset_index()

    files = sorted(glob.glob(str(MAXENT_DIR / "surface_maxent_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No surface_maxent_*.parquet in {MAXENT_DIR}")
    variants = []
    for f in files:
        surf = pd.read_parquet(f)
        pcol = [c for c in surf.columns if c.startswith("prob_")][0]
        variant = pcol.replace("prob_", "")
        surf = compute_fade(surf, sampled)
        render_series(surf, pcol, variant,
                      f"predicted suitability ({PRETTY.get(variant, variant)}, calibrated)",
                      CMAP_SUIT, 0, 1)
        variants.append(variant)

    weeks = sorted(int(w) for w in surf["iso_week"].unique())
    build_html_slider(weeks, variants)
    log(f"[done] MaxEnt maps -> {OUTPUT_DIR}")


if __name__ == "__main__":
    run()