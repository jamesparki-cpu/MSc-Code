from __future__ import annotations
"""
maxent_targetgroup.py  —  TARGET-GROUP MaxEnt (presence + inferred absences).

Uses your effort-based absences (other species trapped, nigripalpus not) as the
background, testing whether that surveillance information beats vanilla's random
background. Unweighted, same cv_harness as everything else.

Takes hours; prints ">>> maxent_targetgroup: starting" immediately, then a line
per CV scheme.
"""
from pathlib import Path
import pandas as pd
import maxent_common as MC

if __name__ == "__main__":
    cfg = MC.load_config()
    feat_dir   = Path(cfg["weekly_xg_dir"])   # weekly_model_table + model_features + grid
    maxent_dir = Path(cfg["maxent_dir"])      # outputs

    tbl = pd.read_parquet(feat_dir / "weekly_model_table.parquet")
    absc = tbl[tbl["presence"] == 0].copy()   # the target-group absences
    MC.run_variant("targetgroup", absc,
                   feat_dir=feat_dir, out_dir=maxent_dir, grid_dir=feat_dir)