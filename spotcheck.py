"""
spotcheck.py
------------
Warning-8 fix: trace a few (Grid_ID, Date) rows from a merged Parquet back to
the raw source CSVs and print them side by side, so you can confirm no column
was swapped or mis-aligned during the merge.
 
Run:  python spotcheck.py 2013
"""
 
import os
import sys
import json
import pandas as pd
 
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
 
# --- resolve paths the same way the build script does ----------------------
with open("config.json") as f:
    config = json.load(f)
DATA_DIR = config["local_data_dir"]
PARQUET_DIR = config["parquet_dir"]
OUT_DIR = PARQUET_DIR
 
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else config["start_year"]
N_SAMPLES = 3
 
# raw files keyed by the merged columns they feed
RAW_SOURCES = {
    "Daymet":    (f"FL_Daymet_5km_{YEAR}.csv",   ["tmax", "tmin", "tmean", "prcp", "vpd"]),
    "EVI":       (f"FL_EVI_5km_{YEAR}.csv",      ["EVI"]),
    "NDWI":      (f"FL_NDWI_5km_{YEAR}.csv",     ["NDWI"]),
    "Topography":(f"FL_Topography_5km.csv",      ["elevation", "slope"]),   # static, no Date
    # land cover (histogram) is verified separately below
}
LANDCOVER_FILE = f"FL_LandCover_5km_{YEAR}.csv"
 
 
def load_raw(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["Grid_ID"] = df["Grid_ID"].astype(str)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df
 
 
def main():
    pq = pd.read_parquet(os.path.join(OUT_DIR, f"Florida_Final_OuterMerged2_{YEAR}.parquet"))
    pq["Grid_ID"] = pq["Grid_ID"].astype(str)
    pq["Date"] = pd.to_datetime(pq["Date"])
 
    # pick samples where EVI and NDWI are PRESENT (they're mostly NaN, so random
    # rows wouldn't test those columns). Fall back to any rows if none found.
    have_veg = pq[pq["EVI"].notna() & pq["NDWI"].notna()]
    pool = have_veg if len(have_veg) >= N_SAMPLES else pq
    samples = pool.sample(N_SAMPLES, random_state=1)[["Grid_ID", "Date"]]
 
    raw = {name: load_raw(fn) for name, (fn, _) in RAW_SOURCES.items()}
 
    for i, (_, key) in enumerate(samples.iterrows(), 1):
        gid, date = key["Grid_ID"], key["Date"]
        print("\n" + "=" * 78)
        print(f"SAMPLE {i}:  Grid_ID={gid}   Date={date.date()}")
        print("=" * 78)
 
        merged_row = pq[(pq["Grid_ID"] == gid) & (pq["Date"] == date)]
 
        for name, (_, cols) in RAW_SOURCES.items():
            df = raw[name]
            if "Date" in df.columns:
                src = df[(df["Grid_ID"] == gid) & (df["Date"] == date)]
            else:  # static product: match on Grid_ID only
                src = df[df["Grid_ID"] == gid]
 
            print(f"\n  [{name}]")
            for c in cols:
                m = merged_row[c].iloc[0] if c in merged_row and len(merged_row) else "MISSING"
                s = src[c].iloc[0] if (c in src.columns and len(src)) else "MISSING"
                flag = "" if _close(m, s) else "   <-- MISMATCH"
                print(f"    {c:<12} parquet={_fmt(m):>14}   raw={_fmt(s):>14}{flag}")
 
        # land cover: show the parquet pct_ columns + the raw histogram to read by eye
        lc = load_raw(LANDCOVER_FILE)
        lc_row = lc[lc["Grid_ID"] == gid]
        pct_cols = [c for c in merged_row.columns if c.startswith("pct_")]
        print("\n  [LandCover]  parquet pct_ columns:")
        for c in pct_cols:
            v = merged_row[c].iloc[0]
            if pd.notna(v) and v > 0:
                print(f"    {c:<22} = {v:.4f}")
        if len(lc_row) and "histogram" in lc_row.columns:
            print(f"    raw histogram = {lc_row['histogram'].iloc[0]}")
 
 
def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
 
 
def _close(a, b):
    try:
        if pd.isna(a) and pd.isna(b):
            return True
        return abs(float(a) - float(b)) < 1e-3
    except (TypeError, ValueError):
        return str(a) == str(b)
 
 
if __name__ == "__main__":
    main()