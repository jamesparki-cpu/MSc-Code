from __future__ import annotations

"""
clean_mosquito_static.py  —  STATIC ABUNDANCE baseline, PANEL (cell x year) version.

Target  : Culex nigripalpus
Response: mean of log1p(individualCount) per (Grid_ID, year)   [crude CPUE]
Join    : nearest-centroid snap to the climate grid, then inner-join on [Grid_ID, year]

The climate parquet encodes the cell centroid inside Grid_ID as 'Grid_{lat}_{lon}'
(e.g. 'Grid_24.525_-81.904'), and has one row per cell per year. We parse those
centroids, snap each mosquito record to the nearest one (guaranteeing the join key
matches exactly), aggregate abundance per cell-year, and join on BOTH [Grid_ID, year].
"""
import re, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== CONFIG ======================================
# 1. Load Main Project Directories & Years from JSON
with open("config.json") as f:
    config = json.load(f)

LOCAL_DIR  = Path(config["local_data_dir"])
STATIC_DIR = Path(config["static_dir"])

# Construct file paths dynamically
INPUT_PATH      = LOCAL_DIR / "Mosquito_data_full_raw.csv"
CLIMATE_PARQUET = STATIC_DIR / "Florida_Static_SDM_allyears.parquet" # Update if file name differs
OUTPUT_DIR      = STATIC_DIR

YEAR_MIN        = config["start_year"]
YEAR_MAX        = config["end_year"]

# 2. Local Biological & Data Cleaning Parameters
FOCAL_SPECIES   = "Culex nigripalpus"
GRID_ID_COL     = "Grid_ID"
YEAR_COL        = "year"

FL_BBOX = dict(lat_min=24.0, lat_max=31.5, lon_min=-88.0, lon_max=-79.5)
MAX_COORD_UNCERTAINTY_M = 5000
DEDUP_KEYS = ["eventDate", "decimalLatitude", "decimalLongitude", "individualCount"]
WINSOR_PCT = 99.5
MIN_EVENTS_FLAG = 5          # cell-years below this are flagged (noisy), not dropped
MAX_SNAP_KM = 6.0            # warn if a record snaps farther than this to its cell
# ============================================================================

def log(m): print(m, flush=True)

# ---------- cleaning ----------
def load_raw(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    log(f"[load] {len(df):,} rows x {df.shape[1]} cols"); return df

def clean_whole_file(df):
    n0 = len(df)
    keep = [c for c in ["gbifID","species","individualCount","decimalLatitude",
            "decimalLongitude","coordinateUncertaintyInMeters","eventDate",
            "day","month","year"] if c in df.columns]
    df = df[keep].copy()
    df["individualCount"]  = pd.to_numeric(df["individualCount"],  errors="coerce")
    df["decimalLatitude"]  = pd.to_numeric(df["decimalLatitude"],  errors="coerce")
    df["decimalLongitude"] = pd.to_numeric(df["decimalLongitude"], errors="coerce")
    df["eventDate"]        = pd.to_datetime(df["eventDate"], errors="coerce")
    df["species"]          = df["species"].astype("string").str.strip()
    df = df.dropna(subset=["decimalLatitude","decimalLongitude","eventDate","individualCount"])
    df = df[df["individualCount"] > 0]
    b = FL_BBOX
    df = df[df["decimalLatitude"].between(b["lat_min"],b["lat_max"]) &
            df["decimalLongitude"].between(b["lon_min"],b["lon_max"])]
    df = df[df["eventDate"].dt.year.between(YEAR_MIN, YEAR_MAX)]
    if "coordinateUncertaintyInMeters" in df.columns:
        unc = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
        df = df[~(unc > MAX_COORD_UNCERTAINTY_M)]   # NaN kept
    df["year"] = df["eventDate"].dt.year
    log(f"[clean] {n0-len(df):,} dropped -> {len(df):,} remain")
    return df.reset_index(drop=True)

def subset_focal(df):
    out = df[df["species"] == FOCAL_SPECIES].copy()
    log(f"[focal] {FOCAL_SPECIES}: {len(out):,} records"); return out

def dedup_and_winsor(df):
    n0 = len(df)
    df = df.drop_duplicates(subset=[k for k in DEDUP_KEYS if k in df.columns])
    log(f"[dedup] dropped {n0-len(df):,} repeat reports")
    if WINSOR_PCT is not None:
        cap = np.percentile(df["individualCount"], WINSOR_PCT)
        log(f"[winsor] capped {int((df['individualCount']>cap).sum()):,} catches > P{WINSOR_PCT} (cap={cap:,.0f})")
        df["individualCount"] = df["individualCount"].clip(upper=cap)
    df["log1p_count"] = np.log1p(df["individualCount"])
    return df.reset_index(drop=True)

# ---------- grid join (centroid encoded in Grid_ID) ----------
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
def parse_centroids(grid_ids):
    """Return DataFrame[Grid_ID, clat, clon] parsed from 'Grid_{lat}_{lon}' strings."""
    rows = []
    for g in pd.unique(grid_ids):
        m = _GID.search(str(g))
        if m:
            rows.append((g, float(m.group(1)), float(m.group(2))))
    out = pd.DataFrame(rows, columns=[GRID_ID_COL, "clat", "clon"])
    if out.empty:
        sys.exit("[grid] could not parse any centroids from Grid_ID — check format.")
    return out

def snap_to_cells(rec, cells):
    """Assign each record the nearest climate-cell Grid_ID (equirectangular dist)."""
    glat = cells["clat"].to_numpy(); glon = cells["clon"].to_numpy()
    gid  = cells[GRID_ID_COL].to_numpy()
    rlat = rec["decimalLatitude"].to_numpy(); rlon = rec["decimalLongitude"].to_numpy()
    coslat = np.cos(np.radians(rlat.mean()))
    
    idx = np.empty(len(rec), dtype=int); dist_km = np.empty(len(rec))
    CH = 4000
    for s in range(0, len(rec), CH):
        e = min(s+CH, len(rec))
        dlat = rlat[s:e,None] - glat[None,:]
        dlon = (rlon[s:e,None] - glon[None,:]) * coslat
        d2 = dlat*dlat + dlon*dlon
        j = d2.argmin(1)
        idx[s:e] = j
        dist_km[s:e] = np.sqrt(d2[np.arange(e-s), j]) * 111.0
    rec = rec.copy()
    rec[GRID_ID_COL] = gid[idx]
    rec["cell_lat"]  = glat[idx]; rec["cell_lon"] = glon[idx]
    rec["snap_km"]   = dist_km
    far = int((dist_km > MAX_SNAP_KM).sum())
    log(f"[snap] {len(rec):,} records snapped | median {np.median(dist_km):.2f} km | "
        f"{far} beyond {MAX_SNAP_KM} km")
    return rec

def aggregate_panel(rec):
    g = rec.groupby([GRID_ID_COL, YEAR_COL]).agg(
        response=("log1p_count","mean"),
        n_events=("log1p_count","size"),
        mean_count=("individualCount","mean"),
        cell_lat=("cell_lat","first"),
        cell_lon=("cell_lon","first"),
    ).reset_index()
    g["low_n_flag"] = g["n_events"] < MIN_EVENTS_FLAG
    log(f"[agg] {len(g):,} cell-years over {g[GRID_ID_COL].nunique()} cells | "
        f"response {g['response'].min():.2f}-{g['response'].max():.2f} | "
        f"{int(g['low_n_flag'].sum())} flagged low-n")
    return g

def join_panel(panel, clim):
    """Inner-join abundance cell-years to climate on [Grid_ID, year]."""
    merged = panel.merge(clim, on=[GRID_ID_COL, YEAR_COL], how="inner")
    cov = len(merged)/len(panel) if len(panel) else 0
    log(f"[join] {len(merged):,}/{len(panel):,} cell-years matched ({cov:.0%})")
    if cov < 0.95:
        log("[join] !! WARNING: <95% matched — check snap distances / year overlap.")
    return merged

# ---------- driver ----------
def run(climate_df=None):
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    rec = dedup_and_winsor(subset_focal(clean_whole_file(load_raw(INPUT_PATH))))

    if climate_df is None:
        climate_df = pd.read_parquet(CLIMATE_PARQUET)   
    log(f"[clim] climate panel: {len(climate_df):,} rows, "
        f"{climate_df[GRID_ID_COL].nunique()} cells, years "
        f"{sorted(climate_df[YEAR_COL].unique())}")

    cells = parse_centroids(climate_df[GRID_ID_COL])
    rec = snap_to_cells(rec, cells)
    rec.to_csv(out/"nigripalpus_clean_records.csv", index=False)

    panel = aggregate_panel(rec)
    panel.to_csv(out/"nigripalpus_abundance_by_cellyear.csv", index=False)

    train = join_panel(panel, climate_df)
    bad = int((~np.isfinite(train["response"])).sum())
    log(f"[check] non-finite responses: {bad}")
    train.to_parquet(out/"static_training_table.parquet", index=False)
    log("[done] keep Grid_ID + year + cell_lat/lon for year-block CV and maps; "
        "n_events is your effort diagnostic / sample_weight.")
    return train

if __name__ == "__main__":
    run()