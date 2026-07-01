import os, glob, json
import pandas as pd

# --- read static_dir from config ------------------------------------------
with open("config.json") as f:
    config = json.load(f)
STATIC_DIR = config["static_dir"]        # folder holding the per-year files

OUT_FILE = os.path.join(STATIC_DIR, "Florida_Static_SDM_allyears.parquet")
COLLAPSE_TO_ONE_ROW_PER_CELL = False     # False = keep (cell, year) rows
                                         # True  = average years -> one row per cell

# --- find the per-year files (exclude any existing combined file) ---------
files = sorted(glob.glob(os.path.join(STATIC_DIR, "Florida_Static_Parquets_2*.parquet")))
files = [f for f in files if "allyears" not in f]
if not files:
    raise FileNotFoundError(
        f"No per-year files found in {STATIC_DIR} "
        f"(looked for Florida_Static_Parquets_2*.parquet). Check static_dir in config.json."
    )
print(f"Combining {len(files)} yearly files from {STATIC_DIR}:")
for f in files:
    print("  ", os.path.basename(f))

df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
print(f"Stacked: {len(df):,} rows, {df['Grid_ID'].nunique():,} unique cells")

if COLLAPSE_TO_ONE_ROW_PER_CELL:
    # average the numeric climate features across years; keep static cols as-is
    static_like = [c for c in df.columns
                   if c.startswith("pct_") or c in
                   ("elevation", "slope", "latitude", "longitude")]
    feature_cols = [c for c in df.columns
                    if c not in static_like + ["Grid_ID", "year"]]
    agg = {c: "mean" for c in feature_cols}
    agg.update({c: "first" for c in static_like})
    df = df.groupby("Grid_ID").agg(agg).reset_index()
    print(f"Collapsed -> {len(df):,} rows (one per cell)")

df.to_parquet(OUT_FILE, index=False)
print(f"Wrote -> {OUT_FILE}")