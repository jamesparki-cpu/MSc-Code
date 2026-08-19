# spatial_roc_diagnosis.py
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import cv_harness as H
import xgboost as xgb

cfg = json.load(open("config.json"))
FEAT = Path(cfg["weekly_xg_dir"])
df = pd.read_parquet(FEAT / "weekly_model_table.parquet")
feats = json.load(open(FEAT / "model_features.json"))["model_features"]

df = H.build_blocks(df)
print(H.build_blocks.__doc__)
w = np.sqrt(df["n_events"].clip(lower=1))
spw = (df.presence == 0).sum() / max((df.presence == 1).sum(), 1)
mk_xgb = lambda: xgb.XGBClassifier(
    scale_pos_weight=spw, n_estimators=400, learning_rate=0.03, max_depth=4,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
    random_state=42, objective="binary:logistic", eval_metric="logloss",
    tree_method="hist")
mk_rf = lambda: RandomForestClassifier(
    n_estimators=400, max_depth=12, min_samples_leaf=5,
    class_weight="balanced", n_jobs=-1, random_state=42)

rows = []
for name, mk, imp in (("xgboost", mk_xgb, False), ("random_forest", mk_rf, True)):
    for cal in (False, True):
        r, _ = H.evaluate(df, feats, mk, schemes=("spatial", "spatiotemporal"),
                          sample_weight=w, calibrate=cal, impute=imp,
                          model_name=name)
        r["calibrate"] = cal
        rows.append(r)
        print(f"  done {name} calibrate={cal}", flush=True)

out = pd.concat(rows, ignore_index=True)[
    ["model", "scheme", "calibrate", "n", "roc_auc", "bss"]]
print(out.to_string(index=False))
out.to_csv("spatial_roc_diagnosis.csv", index=False)