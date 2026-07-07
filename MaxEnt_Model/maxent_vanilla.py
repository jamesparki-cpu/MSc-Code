from __future__ import annotations
"""
maxent_vanilla.py  —  VANILLA MaxEnt (presence + random statewide background).

Standard SDM baseline: presences vs 10k random background cell-weeks from across
Florida (vanilla_background.parquet). Unweighted, run through cv_harness so
metrics are directly comparable to XGB / RF / target-group MaxEnt.

Run AFTER build_vanilla_background.py. Takes hours (MaxEnt is slow); the console
prints ">>> maxent_vanilla: starting" immediately, then a line per CV scheme.
"""
from pathlib import Path
import pandas as pd
import maxent_common as MC

if __name__ == "__main__":
    cfg = MC.load_config()
    feat_dir   = Path(cfg["weekly_xg_dir"])   # weekly_model_table + model_features + grid
    maxent_dir = Path(cfg["maxent_dir"])      # vanilla_background + outputs

    bg = pd.read_parquet(maxent_dir / "vanilla_background.parquet")
    bg["presence"] = 0
    MC.run_variant("vanilla", bg,
                   feat_dir=feat_dir, out_dir=maxent_dir, grid_dir=feat_dir)