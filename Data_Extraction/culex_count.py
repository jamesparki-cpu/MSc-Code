from pathlib import Path
import pandas as pd
import json

with open("config.json") as f:
    config = json.load(f)

LOCAL_DIR  = Path(config["local_data_dir"])


INPUT_PATH      = LOCAL_DIR / "Mosquito_data_full_raw.csv"

# Load your original full dataset
df = pd.read_csv(INPUT_PATH, sep="\t")  # Adjust separator if it's CSV or text

# 1. Total absolute count of Culex nigripalpus across the entire initial dataset
total_nigripalpus = df[df['species'] == 'Culex nigripalpus'].shape[0]
print(f"Total Culex nigripalpus in full dataset: {total_nigripalpus}")

# 2. Total count that survives the final Florida-only Culex filter criteria
florida_culex_full = df[
    (df['stateProvince'] == 'Florida') & 
    (df['species'] == 'Culex nigripalpus')
].shape[0]

print(f"Total records surviving to Florida_culex_full.csv: {florida_culex_full}")
