from __future__ import annotations
"""
aggregate_weekly_presence.py  —  STAGE 1 of the WEEKLY suitability pipeline.

Re-aggregates the already-cleaned, already-snapped Culex nigripalpus records
from cell x YEAR (static abundance) down to cell x ISO-WEEK (trap-event grain),
the resolution the dynamic suitability model needs.

What this DOES:
  * read nigripalpus_clean_records.csv (output of clean_mosquito_static.py):
    one row per trap record, snapped to a 5 km Grid_ID, with log1p_count + eventDate
  * key each record to its ISO (year, week) and the Monday that week starts on
    (week_start = the anchor date Stage-2 lag windows must END BEFORE)
  * collapse to one row per (Grid_ID, iso_year, iso_week)

What this DOES NOT do (by design):
  * It emits POSITIVES ONLY (presence = 1). The cleaned records are
    nigripalpus-only with individualCount > 0, so zero-catch (sampled-absence)
    cell-weeks are NOT recoverable here -- they are built in Stage 3 from the
    full multi-species trap file and UNION-ed onto this table. Columns are laid
    out so that union is a straight concat (presence flips to 0, counts -> 0).

  * It computes NO climate / lag features. Those join on
    [Grid_ID, iso_year, iso_week] (or week_start) in Stage 2.

Outputs (to static_dir):
  nigripalpus_presence_by_cellweek.csv / .parquet
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)

STATIC_DIR = Path(config["static_dir"])
WEEKLY_XG_DIR = Path(config["weekly_xg_dir"])

# clean_mosquito_static.py writes nigripalpus_clean_records.csv into static_dir
INPUT_PATH  = STATIC_DIR / "nigripalpus_clean_records.csv"
OUTPUT_DIR = WEEKLY_XG_DIR / "nigripalpus_presence_by_cellweek.csv"

YEAR_MIN = config["start_year"]
YEAR_MAX = config["end_year"]

GRID_ID_COL = "Grid_ID"

# At weekly grain a single trap-night is the norm (median n_events == 1), so a
# static-style "flag < 5" would flag almost everything. We flag SINGLETONS only:
# informative without nuking the panel. Tune in methods if you change it.
MIN_EVENTS_FLAG = 2          # cell-weeks with n_events < this are flagged noisy
# ============================================================================

def log(m): print(m, flush=True)


def load_clean_records(path):
    df = pd.read_csv(path)
    # eventDate round-trips through CSV as a string; re-parse it.
    df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["eventDate", "log1p_count", GRID_ID_COL])
    # belt-and-braces: stay inside the study window
    df = df[df["eventDate"].dt.year.between(YEAR_MIN, YEAR_MAX)].reset_index(drop=True)
    log(f"[load] {n0:,} clean records -> {len(df):,} usable "
        f"({df[GRID_ID_COL].nunique()} cells, "
        f"{df['eventDate'].dt.year.min()}-{df['eventDate'].dt.year.max()})")
    return df


def add_iso_week_keys(df):
    """Attach ISO (year, week) and the Monday date the week starts on.

    ISO year is used (not calendar year) so the few records in a year-boundary
    week land in the correct week-1/week-53 bucket. week_start (Monday 00:00) is
    the temporal anchor Stage 2 hangs lag windows off: lags accumulate over the
    days *strictly before* week_start, so no same-week catch ever informs its
    own predictors.
    """
    iso = df["eventDate"].dt.isocalendar()      # columns: year, week, day (1=Mon..7=Sun)
    df = df.copy()
    df["iso_year"] = iso["year"].astype(int).values
    df["iso_week"] = iso["week"].astype(int).values
    # Monday of that ISO week = the record's date minus (isoday - 1) days
    df["week_start"] = (df["eventDate"]
                        - pd.to_timedelta(iso["day"].astype(int).values - 1, unit="D")
                        ).dt.normalize()
    return df


def aggregate_cellweek(df):
    """One row per (Grid_ID, iso_year, iso_week). Positives only."""
    g = (df.groupby([GRID_ID_COL, "iso_year", "iso_week"], as_index=False)
           .agg(week_start=("week_start", "first"),
                cell_lat=("cell_lat", "first"),
                cell_lon=("cell_lon", "first"),
                n_events=("log1p_count", "size"),          # trap records in cell-week
                mean_log_count=("log1p_count", "mean"),    # secondary abundance target
                mean_count=("individualCount", "mean"),
                total_count=("individualCount", "sum"),    # for Stage-3 min-catch rule
                max_count=("individualCount", "max")))
    g["presence"] = 1                                      # this table is positives only
    g["low_n_flag"] = g["n_events"] < MIN_EVENTS_FLAG
    # stable, union-ready column order (Stage 3 absences concat straight on)
    cols = [GRID_ID_COL, "iso_year", "iso_week", "week_start",
            "cell_lat", "cell_lon", "presence",
            "n_events", "mean_log_count", "mean_count",
            "total_count", "max_count", "low_n_flag"]
    g = g[cols].sort_values([GRID_ID_COL, "iso_year", "iso_week"]).reset_index(drop=True)
    return g


def report(g):
    log(f"[agg] {len(g):,} positive cell-weeks over {g[GRID_ID_COL].nunique()} cells")
    log(f"[agg] trap-records per cell-week: "
        f"median {g['n_events'].median():.0f}, mean {g['n_events'].mean():.2f}, "
        f"max {g['n_events'].max()}")
    log(f"[agg] singletons (n_events==1): {int((g['n_events']==1).sum()):,} "
        f"({(g['n_events']==1).mean():.0%})  |  low_n_flagged: {int(g['low_n_flag'].sum()):,}")
    by_year = g.groupby("iso_year").size()
    log(f"[agg] cell-weeks per ISO year: {by_year.to_dict()}")
    wk = g.groupby("iso_week").size()
    log(f"[agg] ISO weeks present: {wk.index.min()}-{wk.index.max()} "
        f"(peak week {int(wk.idxmax())} with {int(wk.max())} cell-weeks)")


def run():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    df = load_clean_records(INPUT_PATH)
    df = add_iso_week_keys(df)
    g = aggregate_cellweek(df)
    report(g)

    g.to_csv(out / "nigripalpus_presence_by_cellweek.csv", index=False)
    g.to_parquet(out / "nigripalpus_presence_by_cellweek.parquet", index=False)
    log(f"[done] wrote nigripalpus_presence_by_cellweek.csv/.parquet to {out}")
    log("[next] Stage 2 joins daily-climate lag windows on "
        "[Grid_ID, iso_year, iso_week]; Stage 3 unions sampled-absence cell-weeks.")
    return g


if __name__ == "__main__":
    run()