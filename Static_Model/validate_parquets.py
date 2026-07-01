from __future__ import annotations
"""
validate_parquet.py
-------------------
Validation suite for the merged Florida grid Parquet dataset
(climate + vegetation + static layers joined on cell_id / date).

Usage
-----
    from validate_parquet import validate_parquet

    report = validate_parquet(
        "florida_grid/",                 # path or glob to parquet
        expected_cells=None,             # e.g. 6800 if you know it
        date_col="date",
        cell_col="cell_id",
        coord_cols=("latitude", "longitude"),
        static_cols=("elevation",),
        bounded={                        # physically-valid ranges
            "ndvi": (-1.0, 1.0),
            "evi": (-1.0, 1.0),
            "precip": (0.0, None),       # None = no bound on that side
            "temp": (-15.0, 50.0),       # Florida-ish, in deg C
        },
        fill_sentinels=(-3000, -9999, -32768),  # values that should be NaN now
    )

Each check prints PASS / WARN / FAIL with detail. The function returns a dict
so you can assert on it in a pipeline, e.g. `assert report["ok"]`.

A WARN is something expected-but-worth-noting (e.g. 16-day NDVI mostly NaN
against a daily climate join). A FAIL is a structural problem you must fix.
"""
"""
validate_parquet.py
-------------------
Validation suite for the merged Florida grid Parquet dataset
(Daymet + MODIS + Static layers joined on Grid_ID / Date).

Usage
-----
    python validate_parquet.py Florida_Master_Parquets/Florida_Final_OuterMerged_2015.parquet
"""

import pandas as pd
import numpy as np
import sys
import re

# --- Reporting Helpers -------------------------------------------------

class _Report:
    def __init__(self):
        self.results = []   # list of (level, check, message)

    def log(self, level, check, message):
        self.results.append((level, check, message))
        symbol = {"PASS": "  [PASS]", "WARN": "  [WARN]", "FAIL": "  [FAIL]"}[level]
        print(f"{symbol} {check}: {message}")

    def passed(self, check, msg):  self.log("PASS", check, msg)
    def warned(self, check, msg):  self.log("WARN", check, msg)
    def failed(self, check, msg):  self.log("FAIL", check, msg)

    @property
    def n_fail(self):  return sum(1 for lvl, *_ in self.results if lvl == "FAIL")
    @property
    def n_warn(self):  return sum(1 for lvl, *_ in self.results if lvl == "WARN")

# --- Main Entry Point -------------------------------------------------------

def validate_parquet(
    path,
    expected_cells=6800,           # Roughly the 5km grid count for Florida
    expected_timesteps=None,       # e.g., 365 or 366
    date_col="Date",
    cell_col="Grid_ID",
    static_cols=("elevation", "slope", "pct_urban", "pct_wetland", "pct_agriculture"),
    bounded=None,
    fill_sentinels=(-3000, -9999, -32768, -9999.0),
    date_range=("2013-01-01", "2018-12-31"),
):
    """Run all structural and value checks on the merged Parquet dataset."""
    
    # Custom bounds for your specific Florida mosquito dataset
    if bounded is None:
        bounded = {
            "NDWI": (-1.0, 1.0),
            "EVI": (-1.0, 1.0),
            "prcp": (0.0, None),          # Rain cannot be negative
            "vpd": (0.0, None),           # Vapor Pressure Deficit cannot be negative
            "elevation": (-10.0, 110.0),  # Florida max elevation is ~105m (Britton Hill)
            "pct_urban": (0.0, 1.0),      # Percentages must be 0 to 1
            "pct_wetland": (0.0, 1.0),
            "pct_agriculture": (0.0, 1.0)
        }

    r = _Report()

    print("=" * 64)
    print(f"Validating: {path}")
    print("=" * 64)

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        r.failed("0. file load", f"Failed to load parquet file: {e}")
        return _finish(r)

    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns\n")

    # Confirm the exact keys we rely on actually exist
    for col in (cell_col, date_col):
        if col not in df.columns:
            r.failed("0. keys present", f"required column '{col}' missing -- cannot continue")
            return _finish(r)

    # ---------------------------------------------------------------- CHECK 1
    # Key uniqueness: exactly one row per (Grid_ID, Date).
    dup = df.groupby([cell_col, date_col]).size()
    max_dup = int(dup.max())
    if max_dup == 1:
        r.passed("1. key uniqueness", "one row per (Grid_ID, Date)")
    else:
        n_bad = int((dup > 1).sum())
        r.failed("1. key uniqueness",
                 f"{n_bad:,} (cell,date) keys duplicated "
                 f"(max {max_dup} rows) -- join is duplicating rows")

    # ---------------------------------------------------------------- CHECK 1b
    # Dimensions vs expectation.
    n_cells = df[cell_col].nunique()
    n_dates = df[date_col].nunique()
    print(f"    cells={n_cells:,}  distinct dates={n_dates:,}")
    if expected_cells is not None:
        # Give a 5% buffer in case coastal cells were trimmed differently
        if expected_cells * 0.95 <= n_cells <= expected_cells * 1.05:
            r.passed("1b. cell count", f"matches expected ~{expected_cells:,}")
        else:
            r.warned("1b. cell count", f"got {n_cells:,}, expected ~{expected_cells:,}")

    # ---------------------------------------------------------------- CHECK 2
    # Per-column null fractions; ~100% null = a failed (float) join.
    nulls = df.isna().mean().sort_values(ascending=False)
    print("\n    null fraction by column:")
    for col, frac in nulls.items():
        print(f"       {col:<28} {frac:6.1%}")
    dead = nulls[nulls >= 0.999].index.tolist()
    
    if dead:
        r.failed("2. null fractions", f"column(s) ~100% NaN: {dead} -- merge key never matched")
    else:
        r.passed("2. null fractions", "no column is entirely NaN")
        
    high = nulls[(nulls >= 0.5) & (nulls < 0.999)].index.tolist()
    if high:
        r.warned("2b. high nulls",
                 f"{high} >50% NaN -- expected for 16-day MODIS vs daily Daymet, "
                 "but confirm it is cadence and not a partial key failure")

    # ---------------------------------------------------------------- CHECK 3
    # Fill sentinels should already be NaN, not surviving as real numbers.
    numeric = df.select_dtypes(include="number")
    survived = {}
    for col in numeric.columns:
        hits = sum(int((df[col] == s).sum()) for s in fill_sentinels)
        if hits:
            survived[col] = hits
    if survived:
        r.failed("3. fill sentinels",
                 f"unconverted fill values remain: {survived} "
                 "-- convert to NaN per-product before merge")
    else:
        r.passed("3. fill sentinels", f"no {fill_sentinels} values left in numeric columns")

    # ---------------------------------------------------------------- CHECK 4
    # Physical range / bounds (Is Florida actually flat? Is rain positive?).
    for col, (lo, hi) in bounded.items():
        if col not in df.columns:
            r.warned("4. ranges", f"'{col}' not in data -- skipped")
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
            
        cmin, cmax = s.min(), s.max()
        bad = False
        if lo is not None and cmin < lo: bad = True
        if hi is not None and cmax > hi: bad = True
            
        if bad:
            r.failed("4. ranges", f"'{col}' min={cmin:.4g} max={cmax:.4g} outside [{lo}, {hi}] -- scaling error?")
        else:
            r.passed("4. ranges", f"'{col}' within [{lo}, {hi}] (min={cmin:.4g}, max={cmax:.4g})")

    # ---------------------------------------------------------------- CHECK 5
    # Dtypes and date span.
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        r.passed("5. date dtype", f"'{date_col}' is datetime64")
        dmin, dmax = df[date_col].min(), df[date_col].max()
        print(f"    date span: {dmin.date()} -> {dmax.date()}")
        lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        if dmin < lo or dmax > hi:
            r.warned("5b. date span", f"dates fall outside {date_range} -- check for misparsed formats")
        else:
            r.passed("5b. date span", "all dates within expected window")
    else:
        r.failed("5. date dtype", f"'{date_col}' is {df[date_col].dtype}, not datetime -- rolling math will break")

    # ---------------------------------------------------------------- CHECK 6
    # Static variables constant within each cell across time.
    for col in static_cols:
        if col not in df.columns:
            r.warned("6. static broadcast", f"'{col}' not in data -- skipped")
            continue
        varying = (df.groupby(cell_col)[col].nunique(dropna=True) > 1).sum()
        if varying == 0:
            r.passed("6. static broadcast", f"'{col}' constant within every cell")
        else:
            r.failed("6. static broadcast", f"'{col}' varies across time in {int(varying):,} cells -- static join misbehaved")

    # ---------------------------------------------------------------- CHECK 7
    # Grid_ID Structural Integrity (Are they perfectly formatted coords?)
    sample_ids = df[cell_col].dropna().sample(min(1000, len(df)))
    # Regex expects "Grid_XX.XXX_-YY.YYY"
    pattern = re.compile(r"^Grid_\d{1,2}\.\d{3}_-\d{1,3}\.\d{3}$")
    
    malformed = sample_ids[~sample_ids.str.match(pattern)]
    if len(malformed) == 0:
        r.passed("7. Grid_ID format", "Coordinates perfectly encoded in string (e.g. Grid_28.530_-81.370)")
    else:
        r.failed("7. Grid_ID format", f"Found malformed IDs (e.g., {malformed.iloc[0]}) -- coordinate extraction will fail later")

    # Check 8 (manual spot-check vs raw CSVs)
    r.warned("8. manual spot-check",
             "NOT automated -- hand-trace 2-3 rows back to the raw CSVs "
             "to catch swapped/mis-aligned columns")

    return _finish(r)

def _finish(r):
    print("\n" + "=" * 64)
    print(f"SUMMARY: {r.n_fail} fail, {r.n_warn} warn")
    print("PASSED -- dataset looks structurally sound." if r.n_fail == 0
          else "FAILED -- fix the items above before modelling.")
    print("=" * 64)
    return {
        "ok": r.n_fail == 0,
        "n_fail": r.n_fail,
        "n_warn": r.n_warn,
        "results": r.results,
    }

if __name__ == "__main__":
    # If you run it without typing a filename, it defaults to checking 2013.
    target = sys.argv[1] if len(sys.argv) > 1 else "Florida_Master_Parquets/Florida_Final_OuterMerged_2013.parquet"
    rep = validate_parquet(target)
    sys.exit(0 if rep["ok"] else 1)