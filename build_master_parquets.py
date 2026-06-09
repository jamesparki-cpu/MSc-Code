import pandas as pd
import numpy as np
import json
import gc
import os
import re

# =====================================================================
# FLAWS FIXED vs the original script (see inline ">>> FIX" markers):
#
#   1. CRITICAL -- Land cover was read as if each NLCD class were its own
#      digit-named column ('21','22',...). In reality EE exports a single
#      'histogram' string column like "{21=22.9, 11=26383.7, ...}", so the
#      digit-column scan found nothing, total_pixels became 0, and every
#      pct_ column divided to NaN. This is the 100%-NaN FAIL you saw. Now
#      parsed properly with a regex (the format uses '=', not valid JSON).
#
#   2. Only urban/wetland/agri were computed. Now we emit a pct_ column for
#      EVERY class present (full granularity, nothing discarded) PLUS the
#      grouped categories incl. pct_water -- per your request to keep the
#      data fully realised and do trimming later.
#
#   3. Temporal frames were merged with ALL their columns, risking _x/_y
#      suffix collisions and dragging EE junk ('system:index','.geo') into
#      the output. Now each product is reduced to its key + value columns,
#      and latitude/longitude are captured once and reattached at the end.
#
#   4. No cells are dropped (no water trimming) -- intentional, per request.
#
#   5. Optional validation pass after each year (uses validate_parquet if
#      present) so a bad build is caught, not silently written.
# =====================================================================

# =====================================================================
# 1. PATH CONFIGURATION & OUTPUT SETUP (Top of Script)
# =====================================================================
print("Loading configuration...")
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Assign the data folder dynamically from your config
DATA_DIR = config['local_data_dir']

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"CRITICAL: The data folder was not found at {DATA_DIR}. Check config.json.")

# Create the output folder right next to your data folder
output_folder = os.path.join(os.path.dirname(DATA_DIR), 'Florida_Master_Parquets')
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder ready at: {output_folder}")

RUN_VALIDATION = True   # set False to skip the per-year validation pass

# =====================================================================
# LAND-COVER CONFIG (NLCD). Confirm codes against your printed histogram.
# =====================================================================
# Readable name per NLCD class code -> column becomes pct_<name>.
NLCD_NAMES = {
    11: "open_water",       12: "perennial_ice_snow",
    21: "developed_open",   22: "developed_low",
    23: "developed_med",    24: "developed_high",
    31: "barren",
    41: "deciduous_forest", 42: "evergreen_forest", 43: "mixed_forest",
    51: "dwarf_scrub",      52: "shrub_scrub",
    71: "grassland",        72: "sedge",            73: "lichens", 74: "moss",
    81: "pasture_hay",      82: "cultivated_crops",
    90: "woody_wetland",    95: "emergent_wetland",
}
# Grouped, model-ready categories (sum of member classes / total).
CATEGORY_GROUPS = {
    "urban":       {21, 22, 23, 24},
    "agriculture": {81, 82},
    "wetland":     {90, 95},
    "water":       {11, 12},
}
# EE histogram is Java-style "{code=count, ...}" -- NOT JSON. Regex-parse it.
_PAIR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*=\s*([0-9eE.+-]+)")


def parse_histogram(v):
    """Parse one EE frequencyHistogram value into {class_code: count}."""
    if isinstance(v, dict):
        return {int(float(k)): float(x) for k, x in v.items()}
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return {}
    out = {}
    for code, count in _PAIR_RE.findall(str(v)):
        try:
            out[int(float(code))] = float(count)
        except ValueError:
            continue
    return out


# =====================================================================
# 2. STANDARDIZATION FUNCTIONS
# =====================================================================
def standardize_temporal_csv(filepath, value_cols, date_col='Date', id_col='Grid_ID'):
    """Load a dynamic time-series CSV, keep only key + requested value cols.

    >>> FIX 3: select columns explicitly so EE junk ('system:index','.geo')
    and lat/lon don't ride into the merge and create _x/_y collisions.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col])
    df[id_col] = df[id_col].astype(str)
    df = df.replace([-9999, -9999.0], np.nan)

    present = [c for c in value_cols if c in df.columns]
    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        print(f"    WARNING: {os.path.basename(filepath)} missing {missing}")
    keep = [id_col, date_col] + present
    df = df[keep]

    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df


def standardize_static_csv(filepath, id_col='Grid_ID'):
    """Loads and standardizes a dataset that has no temporal date."""
    df = pd.read_csv(filepath)
    df[id_col] = df[id_col].astype(str)
    df = df.replace([-9999, -9999.0], np.nan)

    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df


def engineer_landcover(filepath, id_col='Grid_ID'):
    """Parse the histogram column into pct_ fractions: one per class present,
    plus grouped categories (incl. pct_water). No cells dropped; a cell with a
    zero-total histogram yields NaN fractions rather than 0/0.

    >>> FIX 1 & 2: replaces the broken digit-column scan entirely.
    """
    df = pd.read_csv(filepath)
    df[id_col] = df[id_col].astype(str)
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
    # land cover is static within a year -> one row per cell
    df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)

    if 'histogram' not in df.columns:
        raise KeyError(f"'histogram' not in {filepath}; cols={df.columns.tolist()}")

    hist = df['histogram'].apply(parse_histogram)
    totals = hist.apply(lambda d: sum(d.values())).replace(0, np.nan)

    out = pd.DataFrame({id_col: df[id_col].values})

    # one fraction column per class actually present in the data
    all_codes = sorted({c for d in hist for c in d})
    for code in all_codes:
        name = NLCD_NAMES.get(code, f"class_{code}")
        out[f"pct_{name}"] = (hist.apply(lambda d: d.get(code, 0.0)).values / totals.values)

    # grouped, model-ready categories
    for cat, codes in CATEGORY_GROUPS.items():
        grouped = hist.apply(lambda d: sum(v for k, v in d.items() if k in codes))
        out[f"pct_{cat}"] = grouped.values / totals.values

    pct_cols = [c for c in out.columns if c.startswith("pct_")]
    out[pct_cols] = out[pct_cols].astype('float32')
    n_classes = len(all_codes)
    print(f"    land cover: {n_classes} classes -> "
          f"{len(pct_cols)} pct_ columns over {len(out):,} cells")
    return out


# =====================================================================
# 3. STATIC DATA PREPARATION
# =====================================================================
print("\nStandardizing Static Topography...")
topo_path = os.path.join(DATA_DIR, 'FL_Topography_5km.csv')
topo_df = standardize_static_csv(topo_path)
topo_df = topo_df[['Grid_ID', 'elevation', 'slope']].drop_duplicates(subset=['Grid_ID'])

# =====================================================================
# 4. THE MASTER MERGE LOOP
# =====================================================================
years = range(config['start_year'], config['end_year'] + 1)

DAYMET_COLS = ['tmax', 'tmin', 'tmean', 'prcp', 'vpd']
EVI_COLS = ['EVI']
NDWI_COLS = ['NDWI']

for year in years:
    print(f"\n{'='*40}\nProcessing Year: {year}\n{'='*40}")

    # --- Load Temporal Data (column-reduced) ---
    print("Loading and Standardizing Temporal CSVs...")
    daymet_path = os.path.join(DATA_DIR, f'FL_Daymet_5km_{year}.csv')
    daymet_df = standardize_temporal_csv(daymet_path, DAYMET_COLS)
    evi_df = standardize_temporal_csv(os.path.join(DATA_DIR, f'FL_EVI_5km_{year}.csv'), EVI_COLS)
    ndwi_df = standardize_temporal_csv(os.path.join(DATA_DIR, f'FL_NDWI_5km_{year}.csv'), NDWI_COLS)

    # >>> FIX 3: capture coordinates once (static per cell) for the heat map,
    # from the raw Daymet export, rather than letting them collide in merges.
    raw_daymet = pd.read_csv(daymet_path)
    raw_daymet['Grid_ID'] = raw_daymet['Grid_ID'].astype(str)
    coord_cols = [c for c in ['latitude', 'longitude'] if c in raw_daymet.columns]
    coords_df = (raw_daymet[['Grid_ID'] + coord_cols].drop_duplicates(subset=['Grid_ID'])
                 if coord_cols else None)
    del raw_daymet

    # --- Outer Merge Temporal Data ---
    print("Executing Outer Joins on temporal data...")
    master_df = pd.merge(daymet_df, evi_df, on=['Grid_ID', 'Date'], how='outer')
    master_df = pd.merge(master_df, ndwi_df, on=['Grid_ID', 'Date'], how='outer')

    del daymet_df, evi_df, ndwi_df
    gc.collect()

    # --- Engineer Land Cover from the histogram (FIX 1 & 2) ---
    print("Standardizing and Merging Land Cover...")
    land_clean = engineer_landcover(os.path.join(DATA_DIR, f'FL_LandCover_5km_{year}.csv'))

    # --- Final Left Merges (Static onto Temporal) ---
    master_df = pd.merge(master_df, topo_df, on='Grid_ID', how='left')
    master_df = pd.merge(master_df, land_clean, on='Grid_ID', how='left')
    if coords_df is not None:
        master_df = pd.merge(master_df, coords_df, on='Grid_ID', how='left')

    master_df = master_df.sort_values(by=['Grid_ID', 'Date']).reset_index(drop=True)

    del land_clean
    gc.collect()

    # =====================================================================
    # 5. OUTPUT CREATION (Bottom of Loop)
    # =====================================================================
    output_file = os.path.join(output_folder, f'Florida_Final_OuterMerged2_{year}.parquet')
    print(f"Compressing missingness-preserved dataset to {output_file}...")
    master_df.to_parquet(output_file, engine='pyarrow', index=False)

    # --- Optional validation pass (FIX 5) ---
    if RUN_VALIDATION:
        try:
            from validate_parquets import validate_parquet
            report = validate_parquet(
                output_file,
                cell_col='Grid_ID', date_col='Date',
                static_cols=('elevation', 'slope'),
                bounded={
                    'EVI': (-1.0, 1.0), 'NDWI': (-1.0, 1.0),
                    'prcp': (0.0, None), 'vpd': (0.0, None),
                    'elevation': (-10.0, 110.0),
                    'pct_urban': (0.0, 1.0), 'pct_wetland': (0.0, 1.0),
                    'pct_agriculture': (0.0, 1.0), 'pct_water': (0.0, 1.0),
                },
            )
            if not report['ok']:
                print(f"  *** {year}: validation reported {report['n_fail']} "
                      f"FAIL(s) -- inspect before using this file ***")
        except ImportError:
            print("  (validate_parquet not found on path -- skipping validation)")

    del master_df
    gc.collect()

print("\nPipeline Complete! Missingness is now visible and inspectable for XGBoost.")