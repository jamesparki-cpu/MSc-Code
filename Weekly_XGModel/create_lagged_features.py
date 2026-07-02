from __future__ import annotations
"""
build_daily_features.py  —  STAGE 2 of the WEEKLY suitability pipeline.

Turns the daily OuterMerged climate series into LEAK-PROOF, trailing weather
features and joins them onto the Stage-1 cell-week table.

Follows your 7-step plan. Three places are HARDENED beyond the plan; each is
marked [HARDENED] with the reason. The hardening is what makes window=14 mean
"14 calendar days" and guarantees the join never silently drops a cell-week.

Inputs
  * <daily_dir>/*OuterMerged*.parquet   (one per year; daily, all FL cells)
  * <static_dir>/nigripalpus_presence_by_cellweek.parquet   (Stage 1 output)

Outputs (to static_dir)
  * daily_features_master.parquet   -> shifted daily features, trapped cells only.
                                       Build ONCE; Stage 3/5 just load + join.
  * weekly_training_features.parquet-> Stage-1 positives with features attached.
"""
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)

# ===== EXPERIMENTAL GATE ====================================================
# True  : Adds explicit GDD and cold-day counts (Ecological constraint arm)
# False : Relies purely on continuous rolling features (Data-driven arm)
USE_ECOLOGICAL_FEATURES = False
# ============================================================================

STATIC_DIR = Path(config["static_dir"])
WEEKLY_DIR = Path(config["weekly_xg_dir"])
DAILY_DIR  = Path(config.get("parquet_dir", config["static_dir"]))   # where *OuterMerged* live
PRESENCE   = WEEKLY_DIR / "nigripalpus_presence_by_cellweek.parquet"
OUTPUT_DIR = WEEKLY_DIR / "daily_features"   # where to write daily_features_master.parquet

GRID_ID_COL = "Grid_ID"
YEAR_MIN, YEAR_MAX = config["start_year"], config["end_year"]
GLOBAL_START = pd.Timestamp(f"{YEAR_MIN}-01-01")
GLOBAL_END   = pd.Timestamp(f"{YEAR_MAX}-12-31")

# the only climate columns we lag (drop static elevation/slope/landcover here;
# static features join later, they don't belong in a rolling window)
CLIMATE_COLS = ["tmax", "tmin", "tmean", "prcp", "vpd"]
VEG_COLS     = ["EVI", "NDWI"]            # already present in OuterMerged, just sparse
FFILL_LIMIT  = {"EVI": 16, "NDWI": 8}     # carry last satellite obs to next revisit
CLIM_FFILL_LIMIT = 3                       # patch rare 1-3 day Daymet merge gaps only

LAG_WINDOWS  = (7, 14, 28)                 # days
DD_BASELINE  = 10.0                        # Culex developmental threshold (deg C)
COLD_THRESH  = 10.0                        # "cold day" if tmin < this
# ============================================================================

def log(m): print(m, flush=True)
def minp(w): return max(1, w - 2)          # your slight tolerance: window-2 non-NaN


# ===== STEP 1 — Prep the daily climate master ===============================
def load_daily_master(cells_needed):
    """Concat yearly OuterMerged files, keep ONLY trapped cells + the columns we
    lag, coerce dtypes, and [HARDENED] reindex each cell to a gap-free daily
    calendar so every rolling window is measured in real days, not rows."""
    files = sorted(glob.glob(str(DAILY_DIR / "*OuterMerged*.parquet")))
    if not files:
        raise FileNotFoundError(f"No *OuterMerged*.parquet in {DAILY_DIR}")
    log(f"[step1] {len(files)} daily files in {DAILY_DIR}")

    keep = [GRID_ID_COL, "Date"] + CLIMATE_COLS + VEG_COLS
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        d = d[[c for c in keep if c in d.columns]]
        d = d[d[GRID_ID_COL].isin(cells_needed)]          # filter to trapped cells on read (RAM)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])

    # Crucial check you asked for: is the core weather actually populated?
    nan = df[CLIMATE_COLS].isna().mean().round(4).to_dict()
    log(f"[step1] core-weather NaN fraction (pre-reindex): {nan}")

    # collapse any accidental duplicate (cell, day) rows before reindex
    df = (df.sort_values([GRID_ID_COL, "Date"])
            .groupby([GRID_ID_COL, "Date"], as_index=False).first())

    # [HARDENED] reindex to the full daily calendar x trapped cells. Missing days
    # become NaN rows so window=N == N calendar days and shift(1) == exactly 1 day.
    full_idx = pd.date_range(GLOBAL_START, GLOBAL_END, freq="D")
    mi = pd.MultiIndex.from_product([sorted(cells_needed), full_idx],
                                    names=[GRID_ID_COL, "Date"])
    df = (df.set_index([GRID_ID_COL, "Date"]).reindex(mi).reset_index())
    log(f"[step1] reindexed -> {len(df):,} rows "
        f"({df[GRID_ID_COL].nunique()} cells x {len(full_idx)} days)")

    # patch only tiny Daymet gaps; longer gaps stay NaN (XGBoost handles natively)
    g = df.groupby(GRID_ID_COL, sort=False)
    for c in CLIMATE_COLS:
        df[c] = g[c].transform(lambda s: s.ffill(limit=CLIM_FFILL_LIMIT))
    df = df.sort_values([GRID_ID_COL, "Date"]).reset_index(drop=True)
    return df


# ===== STEP 2 — Integrate & forward-fill MODIS =============================
def ffill_vegetation(df):
    """EVI/NDWI are already in OuterMerged but only land on satellite-revisit
    days. Carry the last obs forward (cell-wise) up to one revisit cycle: the
    mosquito experiences the last measured greenness/wetness until the next pass."""
    g = df.groupby(GRID_ID_COL, sort=False)
    for c in VEG_COLS:
        if c in df.columns:
            df[c + "_ff"] = g[c].transform(lambda s: s.ffill(limit=FFILL_LIMIT[c]))
    return df


# ===== STEP 3 & 4 — Feature Engineering (Gated) ============================
def build_rolling_features(df):
    """All windows are RIGHT-aligned (value at day D covers D-window+1 .. D).
    Step 6 then shifts them one day so a Monday row sees only up to the prior
    Sunday. Built per cell so windows never bleed across cell boundaries."""
    g = df.groupby(GRID_ID_COL, sort=False)
    feats = {}

    # --- Step 3: standard accumulations / means ---
    for w in LAG_WINDOWS:
        feats[f"prcp_sum_{w}d"]  = g["prcp"].transform(lambda s, w=w: s.rolling(w, min_periods=minp(w)).sum())
        feats[f"tmean_mean_{w}d"]= g["tmean"].transform(lambda s, w=w: s.rolling(w, min_periods=minp(w)).mean())
    feats["vpd_mean_14d"] = g["vpd"].transform(lambda s: s.rolling(14, min_periods=minp(14)).mean())
    feats["tmax_max_14d"] = g["tmax"].transform(lambda s: s.rolling(14, min_periods=minp(14)).max())

    # ADDED: Continuous tmin signals (always included so XGBoost can find its own cold thresholds)
    feats["tmin_mean_14d"] = g["tmin"].transform(lambda s: s.rolling(14, min_periods=minp(14)).mean())
    feats["tmin_min_14d"]  = g["tmin"].transform(lambda s: s.rolling(14, min_periods=minp(14)).min())

    # --- Step 4: ecological threshold counts (GATED EXPERIMENT) ---
    if USE_ECOLOGICAL_FEATURES:
        log("[step4] GATED: Generating ecological bounds (GDD, cold counts).")
        df["_is_cold"] = (df["tmin"] < COLD_THRESH).astype("float")   # NaN tmin -> NaN, not False
        df.loc[df["tmin"].isna(), "_is_cold"] = np.nan
        feats["cold_days_14d"] = df.groupby(GRID_ID_COL, sort=False)["_is_cold"] \
                                   .transform(lambda s: s.rolling(14, min_periods=minp(14)).sum())

        df["_dd"] = (df["tmean"] - DD_BASELINE).clip(lower=0)
        gdd = df.groupby(GRID_ID_COL, sort=False)["_dd"]
        feats["gdd_sum_14d"] = gdd.transform(lambda s: s.rolling(14, min_periods=minp(14)).sum())
        feats["gdd_sum_28d"] = gdd.transform(lambda s: s.rolling(28, min_periods=minp(28)).sum())

        df = df.drop(columns=["_is_cold", "_dd"])
    else:
        log("[step4] GATED: Ecological features skipped. Model will rely on continuous trailing metrics.")

    feat_df = pd.DataFrame(feats, index=df.index)
    # vegetation enters as its forward-filled level (a state, not an accumulation)
    for c in VEG_COLS:
        if c + "_ff" in df.columns:
            feat_df[c.lower() + "_level"] = df[c + "_ff"]
    return df, feat_df, list(feat_df.columns)


# ===== STEP 6 — Leak-proof shift ===========================================
def shift_features(df, feat_df, feat_cols):
    """[Plan Step 6] Push every feature down one row PER CELL so the value on day
    D becomes the window that ENDED on D-1. After the calendar reindex, one row
    == one day, so this is an exact 'strictly before' guarantee.
    Seasonality is intentionally NOT built here — it's a property of the target
    week (known at prediction time, not a lagged climate window), so it's added
    unshifted at join time in Step 7."""
    out = df[[GRID_ID_COL, "Date"]].copy()
    shifted = feat_df.groupby(df[GRID_ID_COL], sort=False).shift(1)
    out = pd.concat([out, shifted], axis=1)
    return out


# ===== STEP 7 — Final merge (+ Step 5 seasonality, unshifted) ===============
def join_to_weeks(master, presence, feat_cols):
    """Left-join shifted daily features onto each cell-week at Date == week_start.
    Then attach sin/cos day-of-year from week_start itself (Step 5)."""
    pres = presence.copy()
    pres["week_start"] = pd.to_datetime(pres["week_start"]).dt.normalize()
    merged = pres.merge(master.rename(columns={"Date": "week_start"}),
                        on=[GRID_ID_COL, "week_start"], how="left")

    # --- Step 5: seasonality on the TARGET week (circular, leak-free) ---
    doy = pd.to_datetime(merged["week_start"]).dt.dayofyear
    merged["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    merged["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    matched = merged[feat_cols].notna().any(axis=1).mean()
    log(f"[step7] {len(merged):,} cell-weeks joined | "
        f"{matched:.1%} have >=1 climate feature "
        f"(unmatched = pre-2013-warmup or daily gap)")
    return merged


# ===== leak self-check ======================================================
def leak_check(master_raw_daily, weekly, n=3):
    """Independently recompute prcp_sum_7d for a few cell-weeks from the RAW daily
    series (sum over the 7 days ending the Sunday before week_start) and confirm
    it equals the joined feature. If this passes, the shift/join is leak-free."""
    log("[check] verifying prcp_sum_7d against an independent recompute:")
    daily = master_raw_daily.set_index([GRID_ID_COL, "Date"])["prcp"]
    ok = 0
    sample = weekly.dropna(subset=["prcp_sum_7d"]).sample(min(n, len(weekly)), random_state=0)
    for _, r in sample.iterrows():
        end = pd.Timestamp(r["week_start"]) - pd.Timedelta(days=1)   # Sunday before
        start = end - pd.Timedelta(days=6)
        try:
            window = daily.loc[(r[GRID_ID_COL], slice(start, end))]
            indep = float(window.sum(min_count=1))
        except KeyError:
            continue
        good = np.isclose(indep, r["prcp_sum_7d"], rtol=1e-4, atol=1e-4)
        ok += good
        log(f"   {r[GRID_ID_COL]} wk{int(r['iso_week'])}/{int(r['iso_year'])}: "
            f"joined={r['prcp_sum_7d']:.2f} independent={indep:.2f} {'OK' if good else 'MISMATCH'}")
    log(f"[check] {ok}/{len(sample)} matched -> "
        f"{'leak-free' if ok == len(sample) else 'INVESTIGATE'}")


# ===== driver ===============================================================
def run():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    presence = pd.read_parquet(PRESENCE)
    absence  = pd.read_parquet(WEEKLY_DIR / "nigripalpus_absence_by_cellweek.parquet")
    cells_needed = set(presence[GRID_ID_COL]) | set(absence[GRID_ID_COL])
    log(f"[run] {len(cells_needed)} trapped cells needed "
        f"(extend this set in Stage 3 if absences add new cells)")

    df = load_daily_master(cells_needed)                  # Step 1 (+reindex)
    raw_daily = df[[GRID_ID_COL, "Date", "prcp"]].copy()  # keep raw prcp for leak-check
    df = ffill_vegetation(df)                             # Step 2
    df, feat_df, feat_cols = build_rolling_features(df)   # Steps 3-4
    master = shift_features(df, feat_df, feat_cols)       # Step 6

    # Optional: Tag filename so you don't overwrite if doing back-to-back testing
    eco_tag = "_ECO" if USE_ECOLOGICAL_FEATURES else ""

    # write the daily master ONCE; Stage 3/5 reload + join in seconds
    master.to_parquet(out / f"daily_features_master{eco_tag}.parquet", index=False)
    log(f"[run] wrote daily_features_master{eco_tag}.parquet "
        f"({len(master):,} rows x {len(feat_cols)} features)")

    weekly = join_to_weeks(master, presence, feat_cols)   # Step 7 (+seasonality)
    weekly.to_parquet(out / f"weekly_training_features{eco_tag}.parquet", index=False)
    log(f"[run] wrote weekly_training_features{eco_tag}.parquet ({len(weekly):,} rows)")

    leak_check(raw_daily, weekly)
    log("[done] Stage 2 complete. Next: Stage 3 unions sampled-absence cell-weeks, "
        "then re-joins daily_features_master.parquet on [Grid_ID, week_start].")
    return weekly


if __name__ == "__main__":
    run()