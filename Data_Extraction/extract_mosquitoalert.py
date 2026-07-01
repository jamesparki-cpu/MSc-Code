from __future__ import annotations
"""
Mosquito Alert -> full reports CSV (2014-2018)
==============================================

Downloads the Mosquito Alert `reports` dataset from its stable Zenodo DOI and
writes ALL reports from 2014-2018 (every type: adult sightings, bites,
breeding sites; every field) to a single local CSV.

Repo-friendly: the raw multi-file JSON archive is unpacked into a SYSTEM TEMP
directory (outside your project), and that temp directory is deleted
automatically when the script finishes - even on error. The ONLY file written
to your working directory is the final CSV, so nothing extra shows up in git.

Note: "all data" here = the full `reports` dataset. Other Mosquito Alert
datasets (sampling_effort, tigapics, etc.) are separate downloads.

Requires: pandas, requests, beautifulsoup4
    pip install pandas requests beautifulsoup4
"""


import io
import os
import re
import glob
import json
import shutil
import zipfile
import tempfile
import urllib.parse

import requests
import pandas as pd
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ZENODO_DOI_URL = "https://doi.org/10.5281/zenodo.597466"  # always latest version
YEARS = range(2014, 2019)            # 2014-2018 inclusive
OUT_CSV = "/Users/jamesparkinson/Library/CloudStorage/OneDrive-TheUniversityofLiverpool/Year 4/COMP702- MSc Project/Msc-data/mosquito_alert_all_2014_2018.csv"   # written to your working dir
DATE_COL: str | None = None          # leave None to auto-detect; else force a column


# ----------------------------------------------------------------------------
# 1. Download + unpack the reports archive into a temp dir (outside the repo)
# ----------------------------------------------------------------------------
def download_reports(work_dir: str) -> str:
    page = requests.get(ZENODO_DOI_URL, timeout=120)
    page.raise_for_status()
    link = BeautifulSoup(page.content, "html.parser").find(
        "a", {"class": "ui compact mini button"}
    )
    if link is None:
        raise RuntimeError(
            "Could not locate the Zenodo download link automatically. "
            "Open the record manually and place the .zip into work_dir."
        )
    file_url = urllib.parse.urljoin(page.url, link["href"])

    print(f"Downloading archive from: {file_url}")
    blob = requests.get(file_url, timeout=600)
    blob.raise_for_status()

    zipfile.ZipFile(io.BytesIO(blob.content)).extractall(work_dir)
    # Flatten any nested zips inside the archive
    for inner in glob.glob(f"{work_dir}/**/*.zip", recursive=True):
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(os.path.dirname(inner))
    return work_dir


# ----------------------------------------------------------------------------
# 2. Load all reports (root level, every column kept)
# ----------------------------------------------------------------------------
def load_reports(work_dir: str) -> pd.DataFrame:
    files = glob.glob(f"{work_dir}/**/all_reports*.json", recursive=True)
    if not files:
        files = [f for f in glob.glob(f"{work_dir}/**/*.json", recursive=True)
                 if "translation" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No report JSON files found under {work_dir}")

    frames = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            frames.append(pd.json_normalize(json.load(f), max_level=0))
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} raw report rows across all years.")
    print("Columns:", list(df.columns))
    return df


# ----------------------------------------------------------------------------
# 3. Build a year series to filter on
# ----------------------------------------------------------------------------
def report_year(df: pd.DataFrame) -> pd.Series:
    if DATE_COL and DATE_COL in df.columns:
        return pd.to_datetime(df[DATE_COL], errors="coerce").dt.year
    for c in df.columns:  # prefer a full timestamp
        if re.search(r"creation.?time|observation.?date|event.?date|^date$", c, re.IGNORECASE):
            return pd.to_datetime(df[c], errors="coerce").dt.year
    for c in df.columns:  # else a year field
        if re.search(r"creation.?year|^year$", c, re.IGNORECASE):
            return pd.to_numeric(df[c], errors="coerce").astype("Int64")
    raise RuntimeError("No date/year column found. Set DATE_COL at the top.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # Unpack into a unique system temp folder, NOT the current directory.
    work_dir = tempfile.mkdtemp(prefix="mosquito_alert_")
    try:
        df = load_reports(download_reports(work_dir))
        df = df[report_year(df).isin(list(YEARS))].reset_index(drop=True)

        # JSON-encode nested (object/list/dict) columns so the CSV stays valid.
        for col in df.columns:
            if df[col].map(lambda v: isinstance(v, (list, dict))).any():
                df[col] = df[col].map(
                    lambda v: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, dict)) else v
                )

        out_path = os.path.abspath(OUT_CSV)
        df.to_csv(out_path, index=False)

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print("\n==================== RESULT ====================")
        print(f"All reports, 2014-2018: {len(df):,} rows x {df.shape[1]} columns")
        print(f"CSV saved to: {out_path}  ({size_mb:.2f} MB)")
    finally:
        # Always remove the temp data, even if an error occurred above.
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"Cleaned up temp download dir: {work_dir}")


if __name__ == "__main__":
    main()