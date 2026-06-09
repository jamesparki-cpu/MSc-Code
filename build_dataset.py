from process_landcover import process_landcover_file, drop_water_cells
import pandas as pd
import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Assign the data folder dynamically from your config
DATA_DIR = config['local_data_dir']
PARQUET_DIR = config['parquet_dir']

from process_landcover import process_landcover_file, drop_water_cells
from validate_parquets import validate_parquet
df = pd.read_parquet(f"{PARQUET_DIR}/2016_Florida_Final_OuterMerged.parquet")  # your existing table with Grid_ID

lc   = process_landcover_file(f"{DATA_DIR}/FL_LandCover_5km_2016.csv")   # parse + fractions
df = df.merge(lc, on="Grid_ID", how="left")          # onto your main table
df, n = drop_water_cells(df)        # once you've checked the threshold
df.to_parquet(f"{PARQUET_DIR}/florida_grid_2016.parquet", index=False)
report = validate_parquet(f"{PARQUET_DIR}/florida_grid_2016.parquet")