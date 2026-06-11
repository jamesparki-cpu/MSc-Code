import pandas as pd
import json

print("Loading configuration...")
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Assign the data folder dynamically from your config
DATA_DIR = config['local_data_dir']
lc   = pd.read_csv(f"{DATA_DIR}/FL_LandCover_5km_2016.csv")          # the failing file
elev = pd.read_csv(f"{DATA_DIR}/FL_Topography_5km.csv")          # the working one

print(lc.columns.tolist())                   # is the column even named what you think?
print(lc["Grid_ID"].head().tolist())         # what does its key actually look like?
print(elev["Grid_ID"].head().tolist())       # vs the one that worked

a, b = set(lc["Grid_ID"]), set(elev["Grid_ID"])
print("overlap:", len(a & b), " lc-only:", len(a - b), " elev-only:", len(b - a))
print("examples lc-only:", list(a - b)[:5])