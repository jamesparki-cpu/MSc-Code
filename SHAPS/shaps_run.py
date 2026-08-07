from __future__ import annotations
"""
shap_run.py  —  DRIVER: SHAP for all four models, fitted on ALL 2013-2018 data.

For each model writes:
  shap_beeswarm_<model>.png      per-feature direction + magnitude (top 20)
  shap_bar_<model>.png           per-feature mean |SHAP|, coloured by block
  shap_grouped_<model>.png       BLOCK importance (the headline figure)
  shap_importance_<model>.csv    per-feature table
  shap_grouped_<model>.csv       per-block table
and one combined shap_grouped_all_models.csv for cross-model comparison.

MODELS: xgboost, random_forest, maxent_targetgroup, maxent_vanilla.

READ THIS BEFORE INTERPRETING maxent_vanilla: it is trained against RANDOM
BACKGROUND, so its SHAP answers "what separates presences from the average
Florida landscape", whereas the other three answer "what separates presence from
a trapped absence". Do not put it in the same ranking table as the others.

Explainers differ by necessity (disclose in methods): TreeSHAP (exact) for
XGB/RF; PermutationExplainer (approximate, ~0.9 s/row) for MaxEnt -- hence the
smaller MaxEnt sample. Expect the MaxEnt runs to take several minutes each,
plus the MaxEnt fits themselves.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shaps_common as SC
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    from elapid import MaxentModel
except ImportError:
    MaxentModel = None

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR = Path(cfg.get("weekly_xg_dir", "."))
BG_FILE  = Path(cfg.get("nowcast_background",
                        str(Path(cfg.get("maxent_dir", str(DATA_DIR))) / "vanilla_background.parquet")))
OUT_DIR  = Path(cfg.get("shap_dir", str(DATA_DIR / "SHAP_Results")))

RUN_MODELS = ["xgboost", "random_forest", "maxent_targetgroup", "maxent_vanilla"]
N_EXPLAIN_TREE   = 2000     # rows explained for XGB/RF (TreeSHAP is fast)
N_EXPLAIN_MAXENT = 400      # rows for MaxEnt (~0.9 s/row -> ~6 min each)
RANDOM_STATE = 42

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss", tree_method="hist")
RF_PARAMS = dict(n_estimators=400, max_depth=12, min_samples_leaf=5,
                 class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

PRETTY = {"xgboost": "XGBoost", "random_forest": "Random Forest",
          "maxent_targetgroup": "MaxEnt (target-group)", "maxent_vanilla": "MaxEnt (vanilla)"}

def log(m): print(m, flush=True)
# ============================================================================


def assemble(variant, table, bg, feats):
    """Training frame per variant (matches how each model was built)."""
    if variant == "maxent_vanilla":
        pres = table[table.presence == 1]
        b = bg.copy(); b["presence"] = 0
        keep = ["presence"] + feats
        return pd.concat([pres[keep], b[keep]], ignore_index=True)
    return table[["presence", "n_events"] + feats].copy()


def fit(variant, df, feats):
    X, y = df[feats], df["presence"].astype(int)
    impute = variant != "xgboost"
    medians = X.median(numeric_only=True)
    Xf = X.fillna(medians) if impute else X
    if variant == "xgboost":
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        m = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
        m.fit(Xf, y, sample_weight=np.sqrt(df["n_events"].clip(lower=1)))
    elif variant == "random_forest":
        m = RandomForestClassifier(**RF_PARAMS)
        m.fit(Xf, y, sample_weight=np.sqrt(df["n_events"].clip(lower=1)))
    else:
        m = MaxentModel(feature_types=["linear", "quadratic", "hinge"],
                        transform="cloglog", clamp=True)
        m.fit(Xf, y)                      # unweighted, as built
    log(f"[shap] fitted {variant}")
    return m, medians, impute


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats = json.load(open(DATA_DIR / "model_features.json"))["model_features"]
    table = pd.read_parquet(DATA_DIR / "weekly_model_table.parquet")
    bg = pd.read_parquet(BG_FILE) if BG_FILE.exists() else None

    grouped_all = []
    for variant in RUN_MODELS:
        if variant == "xgboost" and xgb is None: log("skip xgboost"); continue
        if variant.startswith("maxent") and MaxentModel is None: log(f"skip {variant}"); continue
        if variant == "maxent_vanilla" and bg is None: log("skip maxent_vanilla (no background)"); continue

        df = assemble(variant, table, bg, feats)
        model, medians, impute = fit(variant, df, feats)

        n = N_EXPLAIN_MAXENT if variant.startswith("maxent") else N_EXPLAIN_TREE
        samp = SC.stratified_sample(df, n)
        Xs = samp[feats].fillna(medians) if impute else samp[feats]
        log(f"[shap] explaining {variant} on {len(Xs):,} rows "
            f"({'Permutation ~min' if variant.startswith('maxent') else 'TreeSHAP'})")

        if variant.startswith("maxent"):
            sv = SC.explain_permutation(model, Xs, feats)
        else:
            sv = SC.explain_tree(model, Xs)

        imp = SC.individual_importance(sv, feats)
        gimp = SC.grouped_importance(sv, feats)
        imp.to_csv(OUT_DIR / f"shap_importance_{variant}.csv", index=False)
        gimp.to_csv(OUT_DIR / f"shap_grouped_{variant}.csv", index=False)
        g2 = gimp.copy(); g2.insert(0, "model", variant); grouped_all.append(g2)

        label = PRETTY.get(variant, variant)
        SC.plot_beeswarm(sv, Xs, f"SHAP — {label} (n={len(Xs)})",
                         OUT_DIR / f"shap_beeswarm_{variant}.png")
        SC.plot_bar_individual(imp, f"Feature importance (mean |SHAP|) — {label}",
                               OUT_DIR / f"shap_bar_{variant}.png")
        SC.plot_bar_grouped(gimp, f"Block importance (summed SHAP within block) — {label}",
                            OUT_DIR / f"shap_grouped_{variant}.png")
        log(f"[shap] {variant} top blocks: " +
            ", ".join(f"{r.group} {r.pct_of_total:.0f}%" for r in gimp.head(3).itertuples()))

    if grouped_all:
        allg = pd.concat(grouped_all, ignore_index=True)
        allg.to_csv(OUT_DIR / "shap_grouped_all_models.csv", index=False)
        log("\n[shap] block importance (% of total) by model:")
        log(allg.pivot(index="group", columns="model", values="pct_of_total")
                .round(1).to_string())
    log(f"\n[done] SHAP outputs -> {OUT_DIR}")
    log("[note] maxent_vanilla is trained vs random background — interpret separately.")


if __name__ == "__main__":
    run()