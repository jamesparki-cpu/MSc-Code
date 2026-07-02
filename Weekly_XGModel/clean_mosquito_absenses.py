from __future__ import annotations
"""
clean_mosquito_absence.py  —  STAGE 3 of the WEEKLY suitability pipeline.

Builds the INFERRED-ABSENCE cell-weeks that the classifier needs as its negative
class, as a mirror image of clean_mosquito_static.py's presence cleaner.

WHY INFERRED, NOT OBSERVED
  The raw GBIF file is occurrenceStatus == PRESENT for every row: it records only
  what was CAUGHT, never an explicit zero. So a nigripalpus absence cannot be read
  directly -- it is inferred by TARGET-GROUP BACKGROUND:

      a (cell, ISO-week) is a trap-active event if ANY of the ~40 mosquito
      species was caught there. If nigripalpus is NOT among them, that cell-week
      is an inferred absence -- the trap demonstrably ran (other species prove
      it) but nigripalpus was not present.

  Using other-species catch as the effort signal holds trap PLACEMENT roughly
  constant between the positive and negative classes, forcing the model to
  explain presence through climate/season rather than through where traps sit.

FALSE-ABSENCE GUARD (config-driven, not hard-coded)
  A trap that caught only 1-2 other mosquitoes is weak evidence nigripalpus was
  truly absent. We keep an absence only if the other-species catch that cell-week
  clears `absence_min_other_count` (read from config.json; default 3). Sweeping
  this value is a one-line robustness check.

SYMMETRY WITH THE PRESENCE CLEANER
  Identical clean gates (bbox / date / count>0 / coord-uncertainty / dedup) and
  the identical nearest-centroid snap, so presence and absence are treated the
  same way. Output columns match nigripalpus_presence_by_cellweek EXACTLY (plus
  two trailing diagnostics), so the union in Stage 3b is a straight concat and
  the daily-feature join is unchanged.

OUTPUT (to static_dir)
  nigripalpus_absence_by_cellweek.csv / .parquet   (presence = 0)
"""
import re, json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)

LOCAL_DIR  = Path(config["local_data_dir"])
STATIC_DIR = Path(config["static_dir"])
WEEKLY_DIR = Path(config["weekly_xg_dir"])

INPUT_PATH      = LOCAL_DIR / "Mosquito_data_full_raw.csv"
# same grid the presence cleaner snapped to (all FL cells, centroid encoded in Grid_ID)
GRID_PARQUET    = STATIC_DIR / "Florida_Static_SDM_allyears.parquet"
PRESENCE_TABLE  = WEEKLY_DIR / "nigripalpus_presence_by_cellweek.parquet"
OUTPUT_DIR      = WEEKLY_DIR / "nigripalpus_absence_by_cellweek.csv"

YEAR_MIN, YEAR_MAX = config["start_year"], config["end_year"]
FOCAL_SPECIES = "Culex nigripalpus"
GRID_ID_COL   = "Grid_ID"

# ---- cleaning params (mirror the presence cleaner) ----
FL_BBOX = dict(lat_min=24.0, lat_max=31.5, lon_min=-88.0, lon_max=-79.5)
MAX_COORD_UNCERTAINTY_M = 5000
# dedup MUST include species here (multi-species file), unlike the nig-only cleaner
DEDUP_KEYS = ["eventDate", "decimalLatitude", "decimalLongitude", "species", "individualCount"]
MIN_EVENTS_FLAG = 2
MAX_SNAP_KM = 6.0

# ---- absence-design params (config-driven; add to config.json to override) ----
# false-absence guard: min TOTAL other-species catch to trust a zero
MIN_OTHER_COUNT   = config.get("absence_min_other_count", 3)
# optional thinning cap: max absence cell-weeks per cell (None = keep all)
MAX_ABS_PER_CELL  = config.get("absence_max_per_cell", None)
# ============================================================================

def log(m): print(m, flush=True)


# ---------- cleaning (identical gates to the presence cleaner) --------------
def load_raw(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    log(f"[load] {len(df):,} rows x {df.shape[1]} cols")
    return df

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
    df = df.dropna(subset=["decimalLatitude","decimalLongitude","eventDate",
                           "individualCount","species"])
    df = df[df["individualCount"] > 0]
    b = FL_BBOX
    df = df[df["decimalLatitude"].between(b["lat_min"],b["lat_max"]) &
            df["decimalLongitude"].between(b["lon_min"],b["lon_max"])]
    df = df[df["eventDate"].dt.year.between(YEAR_MIN, YEAR_MAX)]
    if "coordinateUncertaintyInMeters" in df.columns:
        unc = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
        df = df[~(unc > MAX_COORD_UNCERTAINTY_M)]         # NaN kept
    df["year"] = df["eventDate"].dt.year
    log(f"[clean] {n0-len(df):,} dropped -> {len(df):,} remain "
        f"({df['species'].nunique()} species)")
    return df.reset_index(drop=True)

def dedup(df):
    n0 = len(df)
    df = df.drop_duplicates(subset=[k for k in DEDUP_KEYS if k in df.columns])
    log(f"[dedup] dropped {n0-len(df):,} repeat reports -> {len(df):,}")
    return df.reset_index(drop=True)


# ---------- grid snap (identical machinery to the presence cleaner) ---------
_GID = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")
def load_grid_centroids(path):
    gid = pd.read_parquet(path, columns=[GRID_ID_COL])[GRID_ID_COL].drop_duplicates()
    rows = []
    for g in gid:
        m = _GID.search(str(g))
        if m:
            rows.append((g, float(m.group(1)), float(m.group(2))))
    cells = pd.DataFrame(rows, columns=[GRID_ID_COL, "clat", "clon"])
    log(f"[grid] {len(cells):,} climate cells to snap against")
    return cells

def snap_to_cells(rec, cells):
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
        j = d2.argmin(1); idx[s:e] = j
        dist_km[s:e] = np.sqrt(d2[np.arange(e-s), j]) * 111.0
    rec = rec.copy()
    rec[GRID_ID_COL] = gid[idx]
    rec["cell_lat"] = glat[idx]; rec["cell_lon"] = glon[idx]
    rec["snap_km"]  = dist_km
    far = int((dist_km > MAX_SNAP_KM).sum())
    log(f"[snap] {len(rec):,} records | median {np.median(dist_km):.2f} km | "
        f"{far} beyond {MAX_SNAP_KM} km")
    return rec


# ---------- build trap-activity, then infer absences ------------------------
def _monday(iso_year, iso_week):
    return pd.Timestamp(date.fromisocalendar(int(iso_year), int(iso_week), 1))

def build_cellweek_activity(rec):
    """One row per (cell, ISO-week): nigripalpus vs other-species catch."""
    iso = rec["eventDate"].dt.isocalendar()
    rec = rec.copy()
    rec["iso_year"] = iso["year"].astype(int).values
    rec["iso_week"] = iso["week"].astype(int).values
    rec["is_nig"]   = rec["species"].eq(FOCAL_SPECIES)

    nig   = rec[rec["is_nig"]]
    other = rec[~rec["is_nig"]]

    o = (other.groupby([GRID_ID_COL, "iso_year", "iso_week"])
              .agg(other_count=("individualCount", "sum"),
                   n_other_events=("individualCount", "size"),
                   n_species=("species", "nunique"),
                   cell_lat=("cell_lat", "first"),
                   cell_lon=("cell_lon", "first")).reset_index())
    # nigripalpus catch per cell-week (to identify where it WAS present -> exclude)
    n = (nig.groupby([GRID_ID_COL, "iso_year", "iso_week"])
             .agg(nig_count=("individualCount", "sum")).reset_index())
    cw = o.merge(n, on=[GRID_ID_COL, "iso_year", "iso_week"], how="left")
    cw["nig_count"] = cw["nig_count"].fillna(0)
    log(f"[activity] {len(cw):,} trap-active cell-weeks with other-species catch")
    return cw

def infer_absences(cw, presence):
    """Keep cell-weeks where nigripalpus is absent AND the trap clears the
    false-absence guard; then make disjoint from the presence table."""
    n0 = len(cw)
    absent = cw[cw["nig_count"] == 0].copy()          # nig not caught here that week
    log(f"[absence] {len(absent):,}/{n0:,} cell-weeks have zero nigripalpus")

    absent = absent[absent["other_count"] >= MIN_OTHER_COUNT]
    log(f"[absence] {len(absent):,} survive false-absence guard "
        f"(other_count >= {MIN_OTHER_COUNT})")

    # disjointness: a cell-week caught by the presence table can't be an absence
    pkey = set(map(tuple, presence[[GRID_ID_COL, "iso_year", "iso_week"]].to_numpy()))
    mask = [ (g, y, w) not in pkey for g, y, w in
             zip(absent[GRID_ID_COL], absent["iso_year"], absent["iso_week"]) ]
    dropped = len(absent) - int(np.sum(mask))
    absent = absent[mask]
    if dropped:
        log(f"[absence] removed {dropped} cell-weeks already in the presence table")

    if MAX_ABS_PER_CELL is not None:                  # optional thinning
        before = len(absent)
        absent = (absent.sort_values("other_count", ascending=False)
                        .groupby(GRID_ID_COL, group_keys=False)
                        .head(MAX_ABS_PER_CELL))
        log(f"[absence] thinned {before:,} -> {len(absent):,} "
            f"(cap {MAX_ABS_PER_CELL}/cell)")
    return absent.reset_index(drop=True)

def to_union_schema(absent):
    """Emit EXACTLY the presence table's columns (+ 2 diagnostics), presence=0.
    nigripalpus catch metrics are genuinely 0 (it wasn't caught). n_events uses
    the OTHER-species record count as the effort proxy, so sqrt(n_events)
    weighting is well-defined for negatives (never 0)."""
    out = pd.DataFrame({
        GRID_ID_COL:      absent[GRID_ID_COL],
        "iso_year":       absent["iso_year"].astype(int),
        "iso_week":       absent["iso_week"].astype(int),
        "week_start":     [ _monday(y, w) for y, w in
                            zip(absent["iso_year"], absent["iso_week"]) ],
        "cell_lat":       absent["cell_lat"],
        "cell_lon":       absent["cell_lon"],
        "presence":       0,
        "n_events":       absent["n_other_events"].astype(int),   # effort proxy
        "mean_log_count": 0.0,     # nigripalpus abundance is zero here
        "mean_count":     0.0,
        "total_count":    0.0,
        "max_count":      0.0,
        "low_n_flag":     absent["n_other_events"] < MIN_EVENTS_FLAG,
        # trailing diagnostics (absence-specific; ignored by the feature join)
        "other_count":    absent["other_count"],
        "n_species":      absent["n_species"].astype(int),
    })
    return out.sort_values([GRID_ID_COL, "iso_year", "iso_week"]).reset_index(drop=True)


# ---------- driver ----------------------------------------------------------
def run():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    rec   = dedup(clean_whole_file(load_raw(INPUT_PATH)))
    cells = load_grid_centroids(GRID_PARQUET)
    rec   = snap_to_cells(rec, cells)

    presence = pd.read_parquet(PRESENCE_TABLE)
    cw       = build_cellweek_activity(rec)
    absent   = infer_absences(cw, presence)
    tab      = to_union_schema(absent)

    tab.to_csv(out / "nigripalpus_absence_by_cellweek.csv", index=False)
    tab.to_parquet(out / "nigripalpus_absence_by_cellweek.parquet", index=False)

    # report the class balance the union will have
    P, A = len(presence), len(tab)
    new_cells = set(tab[GRID_ID_COL]) - set(presence[GRID_ID_COL])
    log(f"[done] wrote {A:,} absence cell-weeks -> nigripalpus_absence_by_cellweek.*")
    log(f"[balance] presence {P:,} / absence {A:,} "
        f"= {P/(P+A):.1%} presence, {A/(P+A):.1%} absence")
    log(f"[cells] {len(new_cells)} absence-only cells not in presence set "
        f"-> add to Stage-2 cells_needed & re-run master for these: {sorted(new_cells)}")
    return tab


if __name__ == "__main__":
    run()