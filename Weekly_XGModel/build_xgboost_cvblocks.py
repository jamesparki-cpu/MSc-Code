from __future__ import annotations
"""
build_cv_blocks.py  —  STAGE 4 of the WEEKLY suitability pipeline.

The HONEST evaluation scaffold Stage 5 plugs into. Produces THREE grouping
schemes and a reusable splitter, plus the per-fold prevalence diagnostics that
tell you whether any fold is too thin to trust.

SCHEMES
  spatial   : leave-one-region-out   -> "new place"      (6 KMeans blocks on centroids)
  temporal  : leave-one-year-out     -> "new season"     (iso_year)
  spatiotemporal : leave-(region,year)-out -> "new place AND new time"  <- HEADLINE
                   the deployment-realistic case the weekly model actually claims.

The weekly model is judged PRIMARILY on `spatiotemporal`; `spatial` and
`temporal` are the diagnostic decomposition that explains where its skill comes
from (seasonality vs transferable habitat).

THIN-FOLD POLICY (spatiotemporal crosses up to 6x6=36 folds; some are small)
  * METRIC (Stage 5): pool OUT-OF-FOLD predictions -> one PR-AUC over all rows.
    Every row is predicted once by a model that never saw it; pooling avoids
    averaging 36 noisy per-fold numbers and never hits an undefined-PR-AUC fold.
  * DIAGNOSIS (here): flag folds below MIN_FOLD_N rows or MIN_CLASS_N of either
    class, so Stage 5 can report the pooled metric with AND without them
    (a sensitivity check, not a silent drop).

OUTPUT (to static_dir)
  weekly_model_table.parquet   (rewritten: + spatial_block, + st_block columns)
  cv_fold_report.csv           (per-fold n / prevalence / thin flag, all schemes)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneGroupOut

# ============================== CONFIG ======================================
with open("config.json") as f:
    config = json.load(f)

STATIC_DIR  = Path(config["static_dir"])
WEEKLY_DIR  = Path(config["weekly_xg_dir"])
MODEL_TABLE = WEEKLY_DIR / "weekly_model_table.parquet"
OUTPUT_DIR  = WEEKLY_DIR

GRID_ID_COL   = "Grid_ID"
N_SPATIAL_BLOCKS = 6
RANDOM_STATE  = 42

# thin-fold thresholds (diagnosis only; nothing is dropped here)
MIN_FOLD_N  = 30     # a fold with fewer rows than this is flagged thin
MIN_CLASS_N = 5      # ... or fewer than this of EITHER class
# ============================================================================

def log(m): print(m, flush=True)


# ----- spatial blocks: cluster whole CELLS so a block shares no cell w/ train
def add_spatial_blocks(df, k=N_SPATIAL_BLOCKS):
    cells = df.groupby(GRID_ID_COL)[["cell_lat", "cell_lon"]].first()
    cells["spatial_block"] = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                                    n_init=10).fit_predict(cells[["cell_lat", "cell_lon"]])
    df = df.merge(cells["spatial_block"], on=GRID_ID_COL)
    sizes = df.groupby("spatial_block")[GRID_ID_COL].nunique().to_dict()
    log(f"[spatial] {df[GRID_ID_COL].nunique()} cells -> {k} blocks (cells/block: {sizes})")
    for b, g in df.groupby("spatial_block"):
        log(f"   block {b}: {g[GRID_ID_COL].nunique():3d} cells | "
            f"lat {g.cell_lat.min():.2f}-{g.cell_lat.max():.2f} "
            f"lon {g.cell_lon.min():.2f}-{g.cell_lon.max():.2f}")
    return df


def add_combined_block(df):
    """Combined spatio-temporal group = (spatial_block, iso_year) as one label."""
    df["st_block"] = (df["spatial_block"].astype(str) + "_" + df["iso_year"].astype(str))
    log(f"[spatiotemporal] {df['st_block'].nunique()} (region x year) folds "
        f"(max possible {N_SPATIAL_BLOCKS * df['iso_year'].nunique()})")
    return df


# ----- per-fold prevalence diagnostics (all three schemes) ------------------
def fold_report(df, scheme, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        n = len(sub); pos = int((sub.presence == 1).sum()); neg = n - pos
        thin = (n < MIN_FOLD_N) or (pos < MIN_CLASS_N) or (neg < MIN_CLASS_N)
        rows.append(dict(scheme=scheme, fold=str(g), n=n, presence=pos, absence=neg,
                         prevalence=round(pos / n, 3), thin=thin))
    rep = pd.DataFrame(rows).sort_values("n")
    n_thin = int(rep["thin"].sum())
    log(f"[report:{scheme}] {len(rep)} folds | {n_thin} flagged thin "
        f"(<{MIN_FOLD_N} rows or <{MIN_CLASS_N} of a class) | "
        f"prevalence range {rep.prevalence.min():.2f}-{rep.prevalence.max():.2f}")
    if n_thin:
        log(f"   thin folds: {rep.loc[rep.thin, 'fold'].tolist()}")
    return rep


# ----- reusable splitter Stage 5 imports ------------------------------------
def make_splitter(df, scheme):
    """Return (splitter, groups) for a scheme. Stage 5:
        for tr, te in splitter.split(X, y, groups): ...
    Pool the te-fold predictions across all folds, then compute ONE PR-AUC."""
    col = {"spatial": "spatial_block", "temporal": "iso_year",
           "spatiotemporal": "st_block"}[scheme]
    return LeaveOneGroupOut(), df[col].to_numpy()


def run():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(MODEL_TABLE)

    df = add_spatial_blocks(df)
    df = add_combined_block(df)

    reports = [
        fold_report(df, "spatial",        "spatial_block"),
        fold_report(df, "temporal",       "iso_year"),
        fold_report(df, "spatiotemporal", "st_block"),
    ]
    rep = pd.concat(reports, ignore_index=True)
    rep.to_csv(out / "cv_fold_report.csv", index=False)

    df.to_parquet(out / "weekly_model_table.parquet", index=False)
    log(f"\n[done] added spatial_block + st_block; wrote cv_fold_report.csv")
    log(f"[headline] Stage 5 leads on 'spatiotemporal' (pooled OOF PR-AUC), "
        f"reports 'temporal' & 'spatial' as the decomposition.")

    # quick guidance on whether thin folds will matter
    st = rep[rep.scheme == "spatiotemporal"]
    thin_rows = int(st.loc[st.thin, "n"].sum()); total = int(st["n"].sum())
    log(f"[headline] spatiotemporal: {int(st.thin.sum())}/{len(st)} folds thin, "
        f"covering {thin_rows}/{total} rows ({thin_rows/total:.1%}). "
        f"-> Stage 5 reports pooled PR-AUC with and without thin folds.")
    return df, rep


if __name__ == "__main__":
    run()