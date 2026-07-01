import os, json, glob
import pandas as pd

with open("config.json") as f:
    config = json.load(f)
STATIC_DIR = config["static_dir"]          # your new config key

def validate_static(path):
    print(f"\n=== {os.path.basename(path)} ===")
    df = pd.read_parquet(path)
    ok = True

    # 1. one row per cell (per year file) / per cell-year (combined)
    keys = ["Grid_ID"] + (["year"] if "year" in df.columns else [])
    dup = df.groupby(keys).size().max()
    print(f"[{'PASS' if dup==1 else 'FAIL'}] one row per {keys}: max {dup}")
    ok &= dup == 1

    # 2. water + no-land-cover cells removed
    if "pct_water" in df:
        n_wet = int((df["pct_water"] > 0.90).sum())
        print(f"[{'PASS' if n_wet==0 else 'FAIL'}] no >90% water cells: {n_wet}")
        ok &= n_wet == 0
    pct = [c for c in df.columns if c.startswith("pct_")]
    n_nolc = int(df[pct].isna().all(axis=1).sum())
    print(f"[{'PASS' if n_nolc==0 else 'FAIL'}] no empty-land-cover cells: {n_nolc}")
    ok &= n_nolc == 0

    # 3. aggregated features in plausible range
    checks = {"EVI_mean": (-1, 1), "NDWI_mean": (-1, 1),
              "tmean_mean": (-15, 50), "prcp_sum": (0, None),
              "pct_water": (0, 1), "pct_urban": (0, 1)}
    for col, (lo, hi) in checks.items():
        if col in df:
            mn, mx = df[col].min(), df[col].max()
            bad = (lo is not None and mn < lo) or (hi is not None and mx > hi)
            print(f"[{'FAIL' if bad else 'PASS'}] {col} in [{lo},{hi}]: "
                  f"[{mn:.3g}, {mx:.3g}]")
            ok &= not bad

    # 4. veg NaN should now be low (2b resolved)
    for col in ("EVI_mean", "NDWI_mean"):
        if col in df:
            frac = df[col].isna().mean()
            print(f"[{'PASS' if frac<0.05 else 'WARN'}] {col} null: {frac:.1%}")

    print("RESULT:", "PASS" if ok else "FAIL")
    return ok

for f in sorted(glob.glob(os.path.join(STATIC_DIR, "*.parquet"))):
    validate_static(f)