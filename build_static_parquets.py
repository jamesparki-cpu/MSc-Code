from __future__ import annotations

"""
build_static_sdm.py
-------------------
Turn the validated per-year merged Parquets (one row per Grid_ID x Date) into
STATIC, model-ready SDM tables (one row per Grid_ID per year) for an XGBoost /
Random Forest habitat-suitability baseline.

What this resolves:
  * 2b (EVI/NDWI cadence NaN): time is collapsed by per-cell aggregation.
    mean/min/max/sum skip NaN by default, so each cell's EVI summary is simply
    the average of the composites that exist -- no gap-filling needed, and the
    output has near-zero NaN in the vegetation columns.
  * 1b (water cells): cells with pct_water > WATER_THRESHOLD are dropped,
    defining the study area as land (+ partial-water coastal cells, which are
    kept on purpose -- coastal wetland is real Culex habitat).

Output per year: one row per land cell, with aggregated climate/vegetation
features + static land-cover/topography + coordinates. Ready to join to
presence/background labels for a static SDM.

Run:  python build_static_sdm.py            # all years found
      python build_static_sdm.py 2013       # one year
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG -- adjust to your paths / filenames.
# ---------------------------------------------------------------------------
with open("config.json") as f:
    config = json.load(f)

# directory holding the merged per-year Parquets (input)
IN_DIR = config.get("parquet_dir") or os.path.join(
    os.path.dirname(config["local_data_dir"]), "Florida_Master_Parquets")
# input filename pattern -- {year} is substituted. EDIT to match your files.
IN_PATTERN = "Florida_Final_OuterMerged2_{year}.parquet"
# output directory + pattern
OUT_DIR = config["static_dir"]
OUT_PATTERN = "Florida_Static_Parquets_{year}.parquet"

KEY = "Grid_ID"
DATE = "Date"
WATER_THRESHOLD = 0.90          # drop cells with pct_water above this

# Which dynamic columns to aggregate, and how. Stats skip NaN by default.
TEMP_VARS = ["tmax", "tmin", "tmean"]   # -> mean/min/max/std
PRCP_VARS = ["prcp"]                    # -> sum/mean/max
VPD_VARS  = ["vpd"]                     # -> mean/max
VEG_VARS  = ["EVI", "NDWI"]             # -> mean/min/max/std
# Static columns carried through unchanged (one value per cell):
STATIC_COLS = ["elevation", "slope", "latitude", "longitude"]
# (all pct_* land-cover columns are auto-detected and carried through)


def _build_dynamic_agg(cols):
    agg = {}
    for c in TEMP_VARS:
        if c in cols:
            agg[c] = ["mean", "min", "max", "std"]
    for c in PRCP_VARS:
        if c in cols:
            agg[c] = ["sum", "mean", "max"]
    for c in VPD_VARS:
        if c in cols:
            agg[c] = ["mean", "max"]
    for c in VEG_VARS:
        if c in cols:
            agg[c] = ["mean", "min", "max", "std"]
    return agg


def build_static_year(in_path: str, year: int) -> pd.DataFrame:
    print(f"\n{'=' * 60}\nStatic build {year}: {os.path.basename(in_path)}\n{'=' * 60}")
    df = pd.read_parquet(in_path)
    df[KEY] = df[KEY].astype(str)
    cols = set(df.columns)
    n_cells_in = df[KEY].nunique()
    print(f"  input: {len(df):,} rows, {n_cells_in:,} cells")

    # --- aggregate the time-varying climate/vegetation (collapses time, 2b) --
    agg = _build_dynamic_agg(cols)
    dyn = df.groupby(KEY).agg(agg)
    dyn.columns = [f"{var}_{stat}" for var, stat in dyn.columns]  # flatten
    dyn = dyn.reset_index()

    # --- carry static columns through unchanged ----------------------------
    pct_cols = sorted(c for c in df.columns if c.startswith("pct_"))
    static_cols = [c for c in STATIC_COLS if c in cols] + pct_cols
    static = df.groupby(KEY)[static_cols].first().reset_index()

    out = dyn.merge(static, on=KEY, how="left")
    out["year"] = year

    # --- resolve 1b: drop > WATER_THRESHOLD water cells AND no-land-cover ---
    if "pct_water" in out.columns:
        # cell has NO land cover if every pct_ column is NaN (empty histogram,
        # i.e. offshore / outside the land-cover raster) -- unusable, drop it.
        no_landcover = out[pct_cols].isna().all(axis=1)
        too_wet = out["pct_water"] > WATER_THRESHOLD
        drop_mask = too_wet | no_landcover
 
        n_water = int((too_wet & ~no_landcover).sum())
        n_nan_lc = int(no_landcover.sum())
        out = out[~drop_mask].reset_index(drop=True)
        print(f"  dropped {n_water:,} cells > {WATER_THRESHOLD:.0%} water")
        print(f"  dropped {n_nan_lc:,} cells with no land cover (NaN histogram)")
        print(f"  -> {int(drop_mask.sum()):,} cells removed total")
    else:
        print("  WARNING: no pct_water column -- water filter skipped")
 
    # --- lightweight static sanity report ----------------------------------
    assert out[KEY].is_unique, "FAIL: Grid_ID not unique after aggregation!"
    print(f"  output: {len(out):,} land cells, {out.shape[1]} columns")
    for c in ("EVI_mean", "NDWI_mean", "tmean_mean", "prcp_sum"):
        if c in out.columns:
            print(f"    {c:<12} null={out[c].isna().mean():5.1%}  "
                  f"[{out[c].min():.3g}, {out[c].max():.3g}]")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if len(sys.argv) > 1:
        years = [int(sys.argv[1])]
    else:
        # discover years from files present
        found = glob.glob(os.path.join(IN_DIR, IN_PATTERN.replace("{year}", "*")))
        years = sorted(int("".join(filter(str.isdigit, os.path.basename(f)))[-4:])
                       for f in found)
        print(f"Found years: {years}")
        if not years:
            print(f"No inputs matching {IN_PATTERN} in {IN_DIR}")
            return
 
    for year in years:
        in_path = os.path.join(IN_DIR, IN_PATTERN.format(year=year))
        if not os.path.exists(in_path):
            print(f"  skip {year}: {in_path} not found")
            continue
        out = build_static_year(in_path, year)
        out_path = os.path.join(OUT_DIR, OUT_PATTERN.format(year=year))
        out.to_parquet(out_path, engine="pyarrow", index=False)
        print(f"  wrote -> {out_path}")
 
    print(f"\nDone. Static SDM files in {OUT_DIR}")


if __name__ == "__main__":
    main()