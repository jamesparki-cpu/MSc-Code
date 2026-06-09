from __future__ import annotations
"""
Mosquito Alert -> Culex presence table (2014-2018)
==================================================

Downloads the Mosquito Alert `reports` dataset from its stable Zenodo DOI,
filters to adult-mosquito reports validated as *Culex* in the years 2014-2018,
and writes a flat presence table: latitude | longitude | date | species.

Why Zenodo and not the GBIF export?
  The GBIF dataset is a frozen, versioned snapshot whose records cluster in
  ~2019-2021. The Zenodo `reports` distribution is the live, nightly-updated
  source and the DOI below always resolves to the latest version.

Why auto-detect columns?
  The portal's public docs show the report `type`, `creation_year/month` and
  nested `responses`, but do NOT pin down the exact lat/lon or species-label
  field names in the raw JSON. This script therefore *discovers* those columns
  and prints what it found. Run once, read the INSPECT output, and (if needed)
  set the *_COL constants explicitly near the top.

Requires: pandas, requests, beautifulsoup4, pyarrow
    pip install pandas requests beautifulsoup4 pyarrow
"""


import io
import os
import re
import glob
import json
import zipfile
import shutil
import urllib.parse
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ZENODO_DOI_URL = "https://doi.org/10.5281/zenodo.597466"  # always latest version
YEARS = range(2014, 2019)            # 2014, 2015, 2016, 2017, 2018 (inclusive)
GENUS_REGEX = re.compile(r"culex", re.IGNORECASE)  # matches "Culex", "Culex pipiens", etc.
ADULT_TYPES = {"adult", "Adult", "ADULT"}  # report types treated as a mosquito sighting

WORK_DIR = "./mosquito_alert_data"
OUT_CSV = "./culex_presence_2014_2018.csv"
OUT_PARQUET = "./culex_presence_2014_2018.parquet"

# Leave as None to auto-detect; set to a string to force a specific column.
LAT_COL: str | None = None
LON_COL: str | None = None
SPECIES_COL: str | None = None
TYPE_COL: str | None = None


# ----------------------------------------------------------------------------
# 1. Download + unpack the reports archive from Zenodo
# ----------------------------------------------------------------------------
def download_reports(work_dir: str) -> str:
    os.makedirs(work_dir, exist_ok=True)

    # Resolve the Zenodo record page, then find the archive download link on it.
    page = requests.get(ZENODO_DOI_URL, timeout=120)
    page.raise_for_status()
    link = BeautifulSoup(page.content, "html.parser").find(
        "a", {"class": "ui compact mini button"}
    )
    if link is None:
        raise RuntimeError(
            "Could not locate the Zenodo download link automatically. "
            "Open the record manually and download the .zip into WORK_DIR."
        )
    file_url = urllib.parse.urljoin(page.url, link["href"])

    print(f"Downloading archive from: {file_url}")
    blob = requests.get(file_url, timeout=600)
    blob.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(blob.content))
    z.extractall(work_dir)

    # The archive may contain nested zips; flatten any inner *.zip too.
    for inner in glob.glob(f"{work_dir}/**/*.zip", recursive=True):
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(os.path.dirname(inner))

    return work_dir


# ----------------------------------------------------------------------------
# 2. Load all_reports*.json into a flat (root-level) dataframe
# ----------------------------------------------------------------------------
def load_reports(work_dir: str) -> pd.DataFrame:
    files = glob.glob(f"{work_dir}/**/all_reports*.json", recursive=True)
    if not files:
        files = glob.glob(f"{work_dir}/**/*.json", recursive=True)
    if not files:
        raise FileNotFoundError(f"No report JSON files found under {work_dir}")

    frames = []
    for fp in files:
        if os.path.basename(fp).startswith("translation"):
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        # max_level=0 keeps it compact: nested fields (e.g. responses) stay as objects.
        frames.append(pd.json_normalize(data, max_level=0))

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} raw report rows from {len(files)} file(s).")
    return df


# ----------------------------------------------------------------------------
# 3. Figure out which columns hold coordinates, species and report type
# ----------------------------------------------------------------------------
def pick(df: pd.DataFrame, forced: str | None, patterns: list[str]) -> str | None:
    if forced and forced in df.columns:
        return forced
    for pat in patterns:
        for c in df.columns:
            if re.search(pat, c, re.IGNORECASE):
                return c
    return None


def inspect_and_resolve(df: pd.DataFrame):
    print("\n--- INSPECT: available columns ---")
    print(list(df.columns))

    lat = pick(df, LAT_COL, [r"^lat", r"latitude", r"_lat"])
    lon = pick(df, LON_COL, [r"^lon", r"longitude", r"_lon"])
    typ = pick(df, TYPE_COL, [r"^type$", r"report_type"])
    sp = pick(df, SPECIES_COL,
              [r"simplified.?annotation", r"movelab.?annotation",
               r"annotation", r"species", r"taxon", r"validation"])

    print("\n--- Resolved columns (override the *_COL constants if wrong) ---")
    print(f"  latitude : {lat}")
    print(f"  longitude: {lon}")
    print(f"  type     : {typ}")
    print(f"  species  : {sp}")

    if sp and sp in df.columns:
        vals = df[sp].dropna().astype(str).value_counts().head(20)
        print(f"\n--- Top values in species column '{sp}' (confirm Culex appears here) ---")
        print(vals.to_string())
    if typ and typ in df.columns:
        print(f"\n--- Values in type column '{typ}' ---")
        print(df[typ].dropna().astype(str).value_counts().to_string())

    missing = [n for n, c in [("lat", lat), ("lon", lon), ("species", sp)] if c is None]
    if missing:
        raise RuntimeError(
            f"Could not auto-detect: {missing}. Inspect the column list above and "
            f"set the corresponding *_COL constant(s) at the top of the script."
        )
    return lat, lon, typ, sp


# ----------------------------------------------------------------------------
# 4. Build a date column from whatever the reports provide
# ----------------------------------------------------------------------------
def build_date(df: pd.DataFrame) -> pd.Series:
    # Prefer a full timestamp if present...
    for c in df.columns:
        if re.search(r"creation.?time|observation.?date|event.?date|^date$", c, re.IGNORECASE):
            return pd.to_datetime(df[c], errors="coerce")
    # ...otherwise reconstruct from year/month/(day) parts.
    y = pick(df, None, [r"creation.?year", r"^year$"])
    m = pick(df, None, [r"creation.?month", r"^month$"])
    d = pick(df, None, [r"creation.?day", r"^day$"])
    if y is None:
        raise RuntimeError("No usable date column or creation_year found.")
    parts = pd.DataFrame({
        "year": pd.to_numeric(df[y], errors="coerce"),
        "month": pd.to_numeric(df[m], errors="coerce") if m else 1,
        "day": pd.to_numeric(df[d], errors="coerce") if d else 1,
    })
    return pd.to_datetime(parts, errors="coerce")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    work_dir = download_reports(WORK_DIR)
    df = load_reports(work_dir)

    lat_c, lon_c, type_c, sp_c = inspect_and_resolve(df)
    df["_date"] = build_date(df)

    mask = df["_date"].dt.year.isin(list(YEARS))
    mask &= df[sp_c].astype(str).str.contains(GENUS_REGEX)
    if type_c:  # keep only adult sightings if a type column exists
        mask &= df[type_c].astype(str).isin(ADULT_TYPES)

    out = (
        df.loc[mask, [lat_c, lon_c, "_date", sp_c]]
        .rename(columns={lat_c: "latitude", lon_c: "longitude",
                         "_date": "date", sp_c: "species"})
        .dropna(subset=["latitude", "longitude", "date"])
        .reset_index(drop=True)
    )
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    out.to_csv(OUT_CSV, index=False)
    out.to_parquet(OUT_PARQUET, index=False)

    csv_kb = os.path.getsize(OUT_CSV) / 1024
    pq_kb = os.path.getsize(OUT_PARQUET) / 1024
    print("\n==================== RESULT ====================")
    print(f"Culex presence records, 2014-2018: {len(out):,}")
    print(f"  {OUT_CSV}     -> {csv_kb:8.1f} KB")
    print(f"  {OUT_PARQUET} -> {pq_kb:8.1f} KB")
    print("\nYou can delete the raw download to reclaim space:")
    print(f"  shutil.rmtree('{WORK_DIR}')   # the filtered table is what you keep")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()