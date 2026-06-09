import pandas as pd
import numpy as np
import json
import gc
import os

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

# =====================================================================
# 2. STANDARDIZATION FUNCTIONS
# =====================================================================
def standardize_temporal_csv(filepath, date_col='Date', id_col='Grid_ID'):
    """Loads, cleans, and standardizes a dynamic time-series dataset."""
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col])
    df[id_col] = df[id_col].astype(str)
    
    # Convert specific NoData/fill values to true NaN
    df = df.replace([-9999, -9999.0], np.nan)
    
    # Downcast floats to float32 to save RAM
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

def standardize_static_csv(filepath, id_col='Grid_ID'):
    """Loads and standardizes dataset that has no temporal date."""
    df = pd.read_csv(filepath)
    df[id_col] = df[id_col].astype(str)
    df = df.replace([-9999, -9999.0], np.nan)
    
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

# =====================================================================
# 3. STATIC DATA PREPARATION
# =====================================================================
print("\nStandardizing Static Topography...")
topo_path = os.path.join(DATA_DIR, 'FL_Topography_5km.csv')
topo_df = standardize_static_csv(topo_path)
topo_df = topo_df[['Grid_ID', 'elevation', 'slope']]

# =====================================================================
# 4. THE MASTER MERGE LOOP
# =====================================================================
years = range(config['start_year'], config['end_year'] + 1)

for year in years:
    print(f"\n{'='*40}\nProcessing Year: {year}\n{'='*40}")
    
    # --- Load Temporal Data ---
    print("Loading and Standardizing Temporal CSVs...")
    daymet_df = standardize_temporal_csv(os.path.join(DATA_DIR, f'FL_Daymet_5km_{year}.csv'))
    evi_df = standardize_temporal_csv(os.path.join(DATA_DIR, f'FL_EVI_5km_{year}.csv'))
    ndwi_df = standardize_temporal_csv(os.path.join(DATA_DIR, f'FL_NDWI_5km_{year}.csv'))
    
    # --- Outer Merge Temporal Data ---
    print("Executing Outer Joins on temporal data...")
    master_df = pd.merge(daymet_df, evi_df, on=['Grid_ID', 'Date'], how='outer')
    master_df = pd.merge(master_df, ndwi_df, on=['Grid_ID', 'Date'], how='outer')
    
    del daymet_df, evi_df, ndwi_df
    gc.collect()
    
    # --- Load and Engineer Land Cover ---
    print("Standardizing and Merging Land Cover...")
    land_df = standardize_static_csv(os.path.join(DATA_DIR, f'FL_LandCover_5km_{year}.csv'))
    
    if 'Date' in land_df.columns:
        land_df = land_df.drop(columns=['Date'])
    
    land_df = land_df.fillna(0)
    
    class_columns = [col for col in land_df.columns if col.isdigit()]
    land_df['total_pixels'] = land_df[class_columns].sum(axis=1)
    
    urban = land_df.get('21', 0) + land_df.get('22', 0) + land_df.get('23', 0) + land_df.get('24', 0)
    wetland = land_df.get('90', 0) + land_df.get('95', 0)
    agri = land_df.get('81', 0) + land_df.get('82', 0)
    
    land_df['pct_urban'] = (urban / land_df['total_pixels']).astype('float32')
    land_df['pct_wetland'] = (wetland / land_df['total_pixels']).astype('float32')
    land_df['pct_agriculture'] = (agri / land_df['total_pixels']).astype('float32')
    
    land_clean = land_df[['Grid_ID', 'pct_urban', 'pct_wetland', 'pct_agriculture']]
    
    # --- Final Left Merge (Static onto Temporal) ---
    master_df = pd.merge(master_df, topo_df, on='Grid_ID', how='left')
    master_df = pd.merge(master_df, land_clean, on='Grid_ID', how='left')
    
    master_df = master_df.sort_values(by=['Grid_ID', 'Date']).reset_index(drop=True)
    
    del land_df, land_clean
    
    # =====================================================================
    # 5. OUTPUT CREATION (Bottom of Loop)
    # =====================================================================
    output_file = os.path.join(output_folder, f'Florida_Final_OuterMerged_{year}.parquet')
    print(f"Compressing missingness-preserved dataset to {output_file}...")
    
    master_df.to_parquet(output_file, engine='pyarrow', index=False)
    
    del master_df
    gc.collect()

print("\nPipeline Complete! Missingness is now visible and inspectable for XGBoost.")