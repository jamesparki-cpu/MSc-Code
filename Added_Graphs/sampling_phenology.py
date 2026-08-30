from __future__ import annotations
"""
sampling_phenology.py  —  R1 descriptive: WHEN and WHERE the traps actually ran.

WHY THIS EXISTS
  The weekly suitability maps show elevated northern suitability in winter and
  spring. That pattern is a candidate SURVEILLANCE artefact rather than an
  ecological signal: if northern cells are trapped only in a narrow part of the
  year, the model has no counter-evidence for the rest of it. This figure states
  the sampling design as a measured result, so the artefact is disclosed in
  Results rather than conceded in Discussion.

WHAT IT SHOWS
  (a) EFFORT   trap-weeks per (ISO week x latitude band). This is the design.
  (b) OUTCOME  observed presence rate per cell, same axes, MASKED where effort
               is below MIN_TRAP_WEEKS -- a 100% presence rate on 2 trap-weeks
               is not a phenological signal and must not be read as one.

  Panel (b) uses the same colour map as the suitability maps (RdYlBu_r, 0-1) so
  that the OBSERVED seasonal pattern can be compared by eye against the
  PREDICTED surfaces. Panel (a) deliberately uses a different, sequential map:
  effort is a count, not a probability, and must not be confusable with it.

  Marginals: total trap-weeks per ISO week along the top. Per-band totals are
  folded into the y-tick labels rather than drawn as a second bar axis -- a
  side axis cannot be kept in register with the heatmap rows once a colourbar
  takes height from the panel.

WHY LATITUDE BANDS AND NOT COUNTIES
  The model table carries cell_lat / cell_lon, not a county field, and latitude
  is the axis the artefact runs along. If a county or station column exists in
  your table, set BAND_COL to it and the script will group by that instead.

RAW TRAP FILE (optional)
  Set "raw_trap_file" in config.json to the VectorBase/GBIF export and the
  script also reports what the MODEL TABLE CANNOT SHOW: the number of physical
  trap stations, the counties they sit in, and how stations collapse into 5 km
  cells. The model table has already aggregated stations to cells, so the
  station count and the station-to-cell ratio are unrecoverable from it.

  Expected columns (GBIF export, tab-separated): stateProvince, locality,
  decimalLatitude, decimalLongitude, year, species.

OUTPUT (to comparison_dir)
  sampling_phenology.png     two-panel heatmap with marginals
  sampling_phenology.csv     tidy: band, iso_week, trap_weeks, n_cells,
                             n_presence, presence_rate
  r1_summary.csv             one row per headline statistic -> Table R1.1 notes
  r1_per_year.csv            per year: cell-weeks, cells, weeks, prevalence,
                             and (if the raw file is given) stations + counties
  r1_stations_by_county.csv  county x year station counts (raw file only)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

import report_style as S

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
FEAT_DIR = Path(cfg["weekly_xg_dir"])
OUT_DIR = Path(cfg.get("comparison_dir", str(FEAT_DIR.parent / "Comparison_Results")))

TABLE = FEAT_DIR / "weekly_model_table.parquet"
RAW_TRAP_FILE = cfg.get("raw_trap_file")     # optional; see docstring
RAW_SEP = "\t"                               # GBIF exports are tab-separated
RAW_STATE = "Florida"
TARGET_SPECIES = "Culex nigripalpus"

BAND_COL = None          # set to e.g. "county" to group by that column instead
BAND_WIDTH = 0.5         # degrees of latitude per band
MIN_TRAP_WEEKS = 5       # below this, panel (b) is masked as low-power
EFFORT_CMAP = "YlGnBu"   # counts -- deliberately NOT the suitability map
RATE_CMAP = "RdYlBu_r"   # matches the suitability maps, for eyeball comparison
MASK_COLOUR = "0.88"

# weeks that carry a validation figure elsewhere in the chapter, marked so the
# reader can connect the week-43 metric collapse to the effort behind it
MARKED_WEEKS = (6, 20, 30, 32, 43)

def log(m): print(m, flush=True)
# ============================================================================


def latitude_bands(lat: pd.Series, width: float = BAND_WIDTH):
    """Half-open bands of `width` degrees, labelled by their southern edge."""
    edge = np.floor(lat / width) * width
    return edge.round(2)


def band_label(v) -> str:
    try:
        return f"{float(v):.1f}\u00b0N"
    except (TypeError, ValueError):
        return str(v)


def describe(df: pd.DataFrame):
    """The R1 denominators, returned as tidy frames AND logged."""
    weeks = sorted(df["iso_week"].unique())
    per_cell = df.groupby("Grid_ID").size()
    missing = sorted(set(range(1, max(weeks) + 1)) - set(weeks))

    stats = [
        ("cell_weeks_total", len(df), "rows in the model table; the modelling unit"),
        ("grid_cells_total", df["Grid_ID"].nunique(), "distinct 5 km cells"),
        ("years", f"{min(df.iso_year)}-{max(df.iso_year)}", "range of iso_year"),
        ("iso_weeks_present", len(weeks), f"min {min(weeks)}, max {max(weeks)}"),
        ("iso_weeks_never_sampled", len(missing), str(missing) if missing else "none"),
        ("prevalence", round(float(df["presence"].mean()), 3), "share of cell-weeks with a presence"),
        ("cell_weeks_per_cell_median", int(per_cell.median()), "median observations per cell"),
        ("cell_weeks_per_cell_min", int(per_cell.min()), ""),
        ("cell_weeks_per_cell_max", int(per_cell.max()), ""),
        ("cell_weeks_per_cell_mean", round(len(df) / df["Grid_ID"].nunique(), 1),
         "the clustering behind the spatial-CV result"),
    ]

    per_year = (df.groupby("iso_year")
                  .agg(cell_weeks=("presence", "size"),
                       grid_cells=("Grid_ID", "nunique"),
                       iso_weeks=("iso_week", "nunique"),
                       prevalence=("presence", "mean"))
                  .round(3).reset_index())

    log("\n[R1] ---- sampling denominators ----")
    for k, v, note in stats:
        log(f"[R1] {k:28s} {str(v):>10s}   {note}")
    log("\n[R1] per year:")
    log(per_year.to_string(index=False))
    return pd.DataFrame(stats, columns=["statistic", "value", "note"]), per_year


def describe_raw():
    """Station and county counts. Only the RAW file can give these: the model
    table has already collapsed stations into 5 km cells, so neither the station
    count nor the station-to-cell ratio survives into it."""
    if not RAW_TRAP_FILE or not Path(RAW_TRAP_FILE).exists():
        log("\n[R1-raw] no raw_trap_file in config.json -- station and county "
            "counts unavailable (they cannot be recovered from the model table)")
        return None, None, None
    want = ["species", "stateProvince", "locality", "decimalLatitude",
            "decimalLongitude", "year"]
    raw = pd.read_csv(RAW_TRAP_FILE, sep=RAW_SEP, usecols=want, low_memory=False)
    raw = raw[(raw.stateProvince == RAW_STATE)].dropna(
        subset=["decimalLatitude", "decimalLongitude"])
    log(f"\n[R1-raw] {len(raw):,} occurrence records in {RAW_STATE}")

    def station_count(d):
        return d.groupby(["decimalLatitude", "decimalLongitude"]).ngroups

    # effort is defined by ALL trapping, not by the target species alone: a trap
    # that ran and caught nothing else still constitutes surveillance effort
    stations = station_count(raw)
    counties = raw.locality.nunique()
    # station -> cell collapse, at the model grid's rounding
    cells = (raw.decimalLatitude.round(2).astype(str) + "_" +
             raw.decimalLongitude.round(2).astype(str)).nunique()
    tgt = raw[raw.species == TARGET_SPECIES]

    stats = [
        ("trap_stations", stations, "distinct coordinate pairs, all species"),
        ("counties", counties, "distinct locality values"),
        ("approx_cells_from_stations", cells, "stations rounded to 0.01 degrees"),
        ("stations_per_cell_mean", round(stations / max(cells, 1), 2),
         "aggregation ratio: stations collapse into cells"),
        ("records_target_species", len(tgt), TARGET_SPECIES),
        ("records_all_species", len(raw), "defines surveillance effort"),
        ("species_recorded", raw.species.nunique(), ""),
    ]
    log("\n[R1-raw] ---- station-level denominators ----")
    for k, v, note in stats:
        log(f"[R1-raw] {k:28s} {str(v):>10s}   {note}")

    by_year = (raw.groupby("year")
                 .apply(lambda g: pd.Series(
                     {"stations": station_count(g),
                      "counties": g.locality.nunique(),
                      "records": len(g)}), include_groups=False)
                 .reset_index())

    by_cy = (raw.groupby(["locality", "year"])
               .apply(station_count, include_groups=False)
               .unstack(fill_value=0))
    log("\n[R1-raw] trap stations by county and year:")
    log(by_cy.to_string())
    dropped = by_cy.columns[-1]
    gone = by_cy.index[(by_cy[dropped] == 0) & (by_cy.iloc[:, 0:-1].sum(axis=1) > 0)]
    if len(gone):
        log(f"[R1-raw] counties sampled earlier but ABSENT in {dropped}: "
            f"{list(gone)} -- this bears directly on any {dropped} holdout")
    return pd.DataFrame(stats, columns=["statistic", "value", "note"]), by_year, by_cy


def build_grid(df: pd.DataFrame):
    """Tidy table + the two matrices (effort, presence rate)."""
    key = BAND_COL if BAND_COL and BAND_COL in df.columns else "_band"
    if key == "_band":
        df = df.assign(_band=latitude_bands(df["cell_lat"]))

    tidy = (df.groupby([key, "iso_week"])
              .agg(trap_weeks=("presence", "size"),
                   n_cells=("Grid_ID", "nunique"),
                   n_presence=("presence", "sum"))
              .reset_index())
    tidy["presence_rate"] = tidy.n_presence / tidy.trap_weeks

    bands = sorted(tidy[key].unique())
    weeks = list(range(1, int(tidy.iso_week.max()) + 1))
    effort = (tidy.pivot(index=key, columns="iso_week", values="trap_weeks")
                  .reindex(index=bands, columns=weeks))
    rate = (tidy.pivot(index=key, columns="iso_week", values="presence_rate")
                .reindex(index=bands, columns=weeks))
    return tidy.rename(columns={key: "band"}), effort, rate, bands, weeks


def figure(effort: pd.DataFrame, rate: pd.DataFrame, bands, weeks, path):
    E = effort.to_numpy(float)
    R = rate.to_numpy(float)
    R_masked = np.where(np.nan_to_num(E) >= MIN_TRAP_WEEKS, R, np.nan)

    band_totals = np.nansum(E, axis=1)

    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(3, 1, height_ratios=[0.7, 3, 3], hspace=0.30, figure=fig)

    # ---- top marginal: total trap-weeks per ISO week ----
    axm = fig.add_subplot(gs[0])
    tot_week = np.nansum(E, axis=0)
    axm.bar(weeks, tot_week, width=0.9, color="0.45", linewidth=0)
    axm.set_ylabel("trap-weeks", fontsize=8)
    axm.set_xlim(0.5, len(weeks) + 0.5)
    axm.tick_params(labelbottom=False, labelsize=8)
    axm.grid(axis="y", alpha=0.3)
    axm.set_title("Surveillance effort by ISO week and latitude, 2013\u20132018",
                  fontsize=12, pad=8)

    def heat(ax, M, cmap, norm, label, totals=False):
        ax.set_facecolor(MASK_COLOUR)          # shows through NaN cells
        im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                       interpolation="nearest",
                       extent=(0.5, len(weeks) + 0.5, -0.5, len(bands) - 0.5))
        ax.set_yticks(range(len(bands)))
        # per-band effort in the tick label: always in register with its row
        labs = [f"{band_label(b)}  ({S.fmt(t, 'n')})" if totals else band_label(b)
                for b, t in zip(bands, band_totals)]
        ax.set_yticklabels(labs, fontsize=8)
        ax.set_ylabel("latitude band" + ("  (total trap-weeks)" if totals else ""))
        for w in MARKED_WEEKS:
            if w in weeks:
                ax.axvline(w, color="0.25", lw=0.7, ls=":", alpha=0.8)
        cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.030)
        cb.set_label(label, fontsize=9)
        cb.outline.set_linewidth(0.6)
        return im

    # ---- (a) effort ----
    axa = fig.add_subplot(gs[1])
    vmax = np.nanpercentile(E, 99) if np.isfinite(E).any() else 1
    heat(axa, E, EFFORT_CMAP, Normalize(0, max(vmax, 1)), "trap-weeks", totals=True)
    axa.set_title("(a) Effort \u2014 grey = never trapped", fontsize=10, loc="left")
    axa.tick_params(labelbottom=False)

    # ---- (b) observed presence rate ----
    axb = fig.add_subplot(gs[2])
    heat(axb, R_masked, RATE_CMAP, Normalize(0, 1), "observed presence rate")
    axb.set_title(f"(b) Outcome \u2014 grey = never trapped or fewer than "
                  f"{MIN_TRAP_WEEKS} trap-weeks", fontsize=10, loc="left")
    axb.set_xlabel("ISO week")

    fig.text(0.01, 0.005,
             "Dotted lines mark the weeks reported in the 2018 validation. "
             "Panel (b) shares the suitability colour map so observed and "
             "predicted seasonality can be compared directly; it is an "
             "OBSERVED rate, not a model output.",
             fontsize=8, color="0.35", ha="left", va="bottom")
    # no tight_layout: the colourbars create axes it cannot lay out, and the
    # GridSpec ratios above already fix the geometry
    fig.subplots_adjust(left=0.135, right=0.95, top=0.94, bottom=0.10)
    S.save(fig, path)


def run():
    S.set_thesis_style(constrained=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["Grid_ID", "iso_year", "iso_week", "presence", "cell_lat", "cell_lon"]
    if BAND_COL:
        cols.append(BAND_COL)
    df = pd.read_parquet(TABLE, columns=[c for c in cols])
    log(f"[load] {len(df):,} rows from {TABLE.name}")

    stats, per_year = describe(df)
    raw_stats, raw_year, raw_cy = describe_raw()

    if raw_stats is not None:
        stats = pd.concat([stats, raw_stats], ignore_index=True)
        per_year = per_year.merge(raw_year, left_on="iso_year", right_on="year",
                                  how="left").drop(columns=["year"])
        raw_cy.to_csv(OUT_DIR / "r1_stations_by_county.csv")
    stats.to_csv(OUT_DIR / "r1_summary.csv", index=False)
    per_year.to_csv(OUT_DIR / "r1_per_year.csv", index=False)
    log(f"\n[R1] wrote r1_summary.csv and r1_per_year.csv to {OUT_DIR}")

    tidy, effort, rate, bands, weeks = build_grid(df)
    tidy.to_csv(OUT_DIR / "sampling_phenology.csv", index=False)
    log(f"\n[grid] {len(bands)} bands x {len(weeks)} weeks | "
        f"{int((effort.notna()).to_numpy().sum())} band-weeks sampled of "
        f"{len(bands) * len(weeks)} possible "
        f"({100 * effort.notna().to_numpy().mean():.1f}% coverage)")

    low = (effort.to_numpy(float) < MIN_TRAP_WEEKS) & effort.notna().to_numpy()
    log(f"[grid] {int(low.sum())} sampled band-weeks have fewer than "
        f"{MIN_TRAP_WEEKS} trap-weeks and are masked in panel (b)")

    # the artefact, as a number: which bands lose coverage in which weeks
    per_band = effort.notna().sum(axis=1)
    log("\n[artefact] weeks sampled per latitude band:")
    for b, n in per_band.items():
        log(f"[artefact]   {band_label(b):>8}  {int(n):>2} / {len(weeks)} weeks")

    figure(effort, rate, bands, weeks, OUT_DIR / "sampling_phenology.png")
    log(f"\n[done] -> {OUT_DIR}")
    return tidy


if __name__ == "__main__":
    run()