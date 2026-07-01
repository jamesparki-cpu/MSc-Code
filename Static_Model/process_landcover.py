from __future__ import annotations
"""
process_landcover.py
--------------------
Turn the Earth Engine frequencyHistogram land-cover export into tidy
per-cell area fractions (pct_urban, pct_agriculture, pct_wetland, pct_water),
and optionally flag/drop water-dominated grid cells.

Handles the quirks found in the real data:
  * histogram stored as a Java-style string  "{21=22.9, 11=26383.7, ...}"
    (NOT valid JSON -- needs a regex parser, not json.loads / literal_eval)
  * fractional (area-weighted) pixel counts, not integers
  * NLCD class codes (11 water, 21-24 developed, 81-82 ag, 90/95 wetland)
  * keys that may arrive as "11" or "11.0" or already as a dict

Designed to run identically across all six yearly files.

Usage
-----
    from process_landcover import process_landcover_file, process_many

    # one file -> tidy per-cell fractions
    lc = process_landcover_file("landcover_2013.csv")

    # merge onto your main table (static join, one row per cell)
    df = df.merge(lc, on="Grid_ID", how="left")

    # apply across all years at once
    lc_all = process_many({
        2013: "landcover_2013.csv",
        2014: "landcover_2014.csv",
        # ...
    })

NLCD codes are the default. If your export used a different product
(MODIS IGBP small integers 1-17, ESA WorldCover 10-100, etc.) pass your
own CLASS_MAP -- the code-to-category numbers are the ONLY thing that
changes between products.
"""

from logging import config
import re
import pandas as pd
import json

# --- NLCD class -> category mapping (confirm against your printed codes) ----
# 11 open water | 21-24 developed | 81-82 cultivated/pasture | 90,95 wetland
NLCD_CLASS_MAP = {
    "urban":       {21, 22, 23, 24},
    "agriculture": {81, 82},
    "wetland":     {90, 95},
    "water":       {11},
}

# Matches "21=22.9" style pairs; tolerant of decimals/scientific notation.
_PAIR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*=\s*([0-9eE.+-]+)")


# --- histogram parsing ------------------------------------------------------

def parse_histogram(v):
    """Parse one Earth Engine frequencyHistogram value into {class_code: count}.

    Accepts the Java-style string form, an already-parsed dict, or a missing
    value. Returns {} for anything unparseable so the row survives as NaN
    fractions rather than crashing the batch.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return {}
    if isinstance(v, dict):
        return {int(float(k)): float(val) for k, val in v.items()}
    pairs = _PAIR_RE.findall(str(v))
    out = {}
    for code, count in pairs:
        try:
            out[int(float(code))] = float(count)
        except ValueError:
            continue
    return out


def _fraction(hist: dict, classes: set, ignore_codes: set | None = None):
    """Share of a cell's pixels falling in `classes`, as a fraction of all
    counted pixels. Returns NaN for an empty/zero-total histogram."""
    if not hist:
        return float("nan")
    ignore_codes = ignore_codes or set()
    total = sum(c for k, c in hist.items() if k not in ignore_codes)
    if total <= 0:
        return float("nan")
    num = sum(c for k, c in hist.items() if k in classes and k not in ignore_codes)
    return num / total


# --- single-file processing -------------------------------------------------

def process_landcover_file(
    path,
    hist_col="histogram",
    cell_col="Grid_ID",
    class_map=None,
    ignore_codes=None,
    keep_extra_cols=("latitude", "longitude"),
):
    """Read one land-cover CSV and return tidy per-cell fraction columns.

    Returns a DataFrame with one row per cell: cell_col + pct_<category> for
    every category in class_map, plus any keep_extra_cols that are present.
    Land cover is static, so the result is de-duplicated to one row per cell.

    `ignore_codes`: class codes excluded from the denominator (e.g. a
    no-data/masked bucket). Leave as None unless Step-1 inspection shows one.
    """
    class_map = class_map or NLCD_CLASS_MAP
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    DATA_DIR = config['local_data_dir']
    df = pd.read_csv(f"{DATA_DIR}/FL_LandCover_5km_2016.csv")

    if hist_col not in df.columns:
        raise KeyError(
            f"'{hist_col}' not in {path}; columns = {df.columns.tolist()}. "
            "Did the export use a different reducer / column name?"
        )

    hist = df[hist_col].apply(parse_histogram)

    # warn on rows that produced an empty histogram (parse miss or truly empty)
    n_empty = int(hist.apply(len).eq(0).sum())
    if n_empty:
        print(f"[process_landcover] {path}: {n_empty:,} rows parsed to empty "
              f"histogram -> NaN fractions (inspect if unexpected)")

    out = pd.DataFrame({cell_col: df[cell_col]})
    for category, codes in class_map.items():
        out[f"pct_{category}"] = hist.apply(lambda h: _fraction(h, codes, ignore_codes))

    for col in keep_extra_cols:
        if col in df.columns:
            out[col] = df[col]

    # land cover is static: collapse to one row per cell
    before = len(out)
    out = out.drop_duplicates(subset=[cell_col]).reset_index(drop=True)
    if len(out) != before:
        print(f"[process_landcover] {path}: collapsed {before:,} -> "
              f"{len(out):,} rows (one per cell)")

    return out


# --- water-cell filtering ---------------------------------------------------

def flag_water_cells(df, water_col="pct_water", threshold=0.9, flag_col="is_water"):
    """Add a boolean column marking cells whose water fraction exceeds
    `threshold`. Flagging (not dropping) lets you inspect before committing."""
    df = df.copy()
    df[flag_col] = df[water_col] > threshold
    n = int(df[flag_col].sum())
    print(f"[process_landcover] {n:,} cells flagged as >{threshold:.0%} water "
          f"({n / len(df):.1%} of cells)")
    return df


def drop_water_cells(df, water_col="pct_water", threshold=0.9):
    """Remove water-dominated cells. Returns (kept_df, n_dropped).

    NOTE: this is a documented modelling decision, not a silent correctness
    fix. A coastal cell that is, say, 60% water but contains marsh is prime
    Culex habitat -- only the near-fully-water cells are safe to drop. Tune
    `threshold` and record the value you used in your methods.
    """
    before = len(df)
    kept = df[df[water_col] <= threshold].reset_index(drop=True)
    dropped = before - len(kept)
    print(f"[process_landcover] dropped {dropped:,} cells >{threshold:.0%} "
          f"water; {len(kept):,} cells remain")
    return kept, dropped


# --- multi-year convenience -------------------------------------------------

def process_many(year_to_path: dict, add_year_col=True, **kwargs):
    """Process several yearly land-cover files with identical settings.

    Land cover usually changes little year to year, so you may prefer one
    file applied to all years; but if you exported per-year NLCD, this keeps
    them consistent. Returns a single concatenated DataFrame with a 'year'
    column (if add_year_col), one row per (cell, year).
    """
    frames = []
    for year, path in sorted(year_to_path.items()):
        print(f"\n--- {year}: {path} ---")
        lc = process_landcover_file(path, **kwargs)
        if add_year_col:
            lc["year"] = year
        frames.append(lc)
    combined = pd.concat(frames, ignore_index=True)
    print(f"\n[process_landcover] combined {len(frames)} file(s) -> "
          f"{len(combined):,} rows")
    return combined


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "landcover.csv"
    lc = process_landcover_file(p)
    lc = flag_water_cells(lc)
    print("\nSample:")
    print(lc.head(10).to_string(index=False))
    print("\nFraction summary:")
    print(lc.filter(like="pct_").describe().to_string())