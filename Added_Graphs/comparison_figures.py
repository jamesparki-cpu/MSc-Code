from __future__ import annotations
"""
comparison_figures.py  —  three synthesis figures for the results chapter.

Standalone; reads outputs the other scripts already wrote. Each figure is
independent and skips (with a message) if its inputs are missing, so you can run
it before every stage is finished.

  FIG 1  seasonal_validation.png
         Per-week 2018 validation ROC and BSS, one bar per model, from
         validation_metrics_*.csv. Answers "does skill hold through the seasonal
         cycle, or only at peak?" -- new information the pooled metrics hide.
         x labels carry n and the ABSENCE count, because a week with few
         absences gives a near-meaningless ROC and must not be over-read.

  FIG 2  shap_blocks_cross_model.png
         (a) grouped bars: block importance per model; (b) 100% stacked
         composition per model. Uses pct_of_total ONLY -- raw mean|SHAP| is not
         comparable across model families (XGB contributions are log-odds,
         RF probability), so percentages are the only honest cross-model unit.

  FIG 3  calibration_curves.png
         Reliability diagrams (predicted vs observed frequency) per model, for
         spatiotemporal and spatial CV, with a prediction-histogram panel.
         Makes the BSS collapse visible instead of abstract.
         Bins are QUANTILE (equal-count), not equal-width: predictions are
         concentrated near the top (prevalence ~0.78), so equal-width bins would
         be mostly empty and noisy.
         MaxEnt OOF is read from oof_maxent_*.csv; XGB/RF OOF is recomputed via
         cv_harness (fast) because compare_models.py does not save it.
"""
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR       = Path(cfg.get("weekly_xg_dir", "."))
MAXENT_DIR     = Path(cfg.get("maxent_dir", str(DATA_DIR)))
VALIDATION_DIR = Path(cfg.get("validation_dir", str(DATA_DIR / "Validation_Maps")))
SHAP_DIR       = Path(cfg.get("shap_dir", str(DATA_DIR / "SHAP_Results")))
OUT_DIR        = Path(cfg["comparison_dir"])

MODEL_ORDER = ["xgboost", "random_forest", "maxent_targetgroup", "maxent_vanilla"]
PRETTY = {"xgboost": "XGBoost", "random_forest": "Random Forest",
          "maxent_targetgroup": "MaxEnt (target-group)", "maxent_vanilla": "MaxEnt (vanilla)"}
COLORS = {"xgboost": "#2166ac", "random_forest": "#238b45",
          "maxent_targetgroup": "#d7301f", "maxent_vanilla": "#6a51a3"}
BLOCK_COLORS = {"temperature": "#d7301f", "precipitation": "#2166ac", "moisture": "#41b6c4",
                "vegetation": "#238b45", "seasonality": "#6a51a3", "land_cover": "#b8860b",
                "terrain": "#777777", "other": "#cccccc"}
SEASON_OF_WEEK = {6: "Winter", 20: "Spring", 30: "Summer", 32: "Summer", 43: "Autumn"}
MIN_ABS_FOR_TRUST = 10        # weeks with fewer absences are flagged as low power
N_BINS = 10

def log(m): print(m, flush=True)
# ============================================================================


# ----------------------------------------------------------------- FIGURE 1
def fig_seasonal_validation():
    files = sorted(glob.glob(str(VALIDATION_DIR / "validation_metrics_*.csv")))
    if not files:
        log(f"[fig1] SKIP: no validation_metrics_*.csv in {VALIDATION_DIR}")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.drop_duplicates(["model", "week"], keep="last")
    weeks = sorted(d.week.unique())
    models = [m for m in MODEL_ORDER if m in d.model.unique()]
    log(f"[fig1] {len(d)} rows | weeks {weeks} | models {models}")

    x = np.arange(len(weeks)); width = 0.8 / max(len(models), 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

    for panel, (col, title, ref, ylab) in enumerate(
            [("roc_auc", "Discrimination — ROC-AUC per week", 0.5, "ROC-AUC"),
             ("bss", "Calibration — Brier Skill Score per week", 0.0, "BSS")]):
        ax = axes[panel]
        for i, m in enumerate(models):
            vals, hatches = [], []
            for w in weeks:
                r = d[(d.model == m) & (d.week == w)]
                vals.append(float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else np.nan)
                nab = int(r.n_abs.iloc[0]) if len(r) else 0
                hatches.append("//" if nab < MIN_ABS_FOR_TRUST else "")
            bars = ax.bar(x + (i - (len(models)-1)/2)*width, vals, width,
                          label=PRETTY.get(m, m), color=COLORS.get(m),
                          edgecolor="black", linewidth=0.5)
            for b, h in zip(bars, hatches):
                if h: b.set_hatch(h)
        ax.axhline(ref, color="grey", ls="--", lw=1)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if col == "roc_auc": ax.set_ylim(0.4, 1.05)

    labels = []
    for w in weeks:
        r = d[d.week == w].iloc[0]
        labels.append(f"{SEASON_OF_WEEK.get(w, '')}\nwk {w}\n(n={int(r.n)}, {int(r.n_abs)} abs)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("2018 forward-chained validation, per week (models trained on 2013–2017)\n"
                 "hatched bars = fewer than "
                 f"{MIN_ABS_FOR_TRUST} observed absences: metric is low-power, do not over-read",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "seasonal_validation.png", dpi=150); plt.close(fig)
    log("[fig1] wrote seasonal_validation.png")


# ----------------------------------------------------------------- FIGURE 2
def fig_shap_blocks():
    f = SHAP_DIR / "shap_grouped_all_models.csv"
    if not f.exists():
        log(f"[fig2] SKIP: {f} not found")
        return
    d = pd.read_csv(f)
    models = [m for m in MODEL_ORDER if m in d.model.unique()]
    blocks = (d.groupby("group").pct_of_total.mean()
                .sort_values(ascending=False).index.tolist())
    log(f"[fig2] models {models} | blocks {blocks}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})

    # (a) grouped bars: block on x, one bar per model
    ax = axes[0]; x = np.arange(len(blocks)); width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [float(d[(d.model == m) & (d.group == b)].pct_of_total.iloc[0])
                if len(d[(d.model == m) & (d.group == b)]) else np.nan for b in blocks]
        ax.bar(x + (i - (len(models)-1)/2)*width, vals, width, label=PRETTY.get(m, m),
               color=COLORS.get(m), edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([b.replace("_", " ") for b in blocks],
                                         rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("% of total |SHAP|"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("(a) Block importance by model", fontsize=11); ax.legend(fontsize=9)

    # (b) 100% stacked composition per model
    ax = axes[1]; bottoms = np.zeros(len(models))
    for b in blocks:
        vals = np.array([float(d[(d.model == m) & (d.group == b)].pct_of_total.iloc[0])
                         if len(d[(d.model == m) & (d.group == b)]) else 0.0 for m in models])
        ax.bar(range(len(models)), vals, 0.6, bottom=bottoms,
               label=b.replace("_", " "), color=BLOCK_COLORS.get(b, "#999"),
               edgecolor="white", linewidth=0.6)
        bottoms += vals
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([PRETTY.get(m, m).replace(" (", "\n(") for m in models], fontsize=8)
    ax.set_ylabel("% of total |SHAP|"); ax.set_ylim(0, 100)
    ax.set_title("(b) Composition", fontsize=11)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.suptitle("Grouped SHAP: where each model's explanatory weight sits\n"
                 "(percentages only — raw |SHAP| is not comparable across model families)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT_DIR / "shap_blocks_cross_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("[fig2] wrote shap_blocks_cross_model.png")


# ----------------------------------------------------------------- FIGURE 3
def reliability(y, p, n_bins=N_BINS, min_count=15):
    """Quantile-binned reliability: equal-count bins suit skewed predictions."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).dropna()
    if df.empty: return None
    try:
        df["bin"] = pd.qcut(df.p, n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df.p, n_bins)
    g = (df.groupby("bin", observed=True)
           .agg(mean_pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")))
    return g[g.n >= min_count]


def collect_oof():
    """{model: DataFrame[presence, oof_spatiotemporal, oof_spatial]}."""
    out = {}
    for f in sorted(glob.glob(str(MAXENT_DIR / "oof_maxent_*.csv"))):
        name = "maxent_" + Path(f).stem.replace("oof_maxent_", "")
        out[name] = pd.read_csv(f)
        log(f"[fig3] read {Path(f).name}")
    # XGB/RF: recompute (compare_models.py doesn't persist OOF)
    try:
        import cv_harness as H
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb
        feats = json.load(open(DATA_DIR / "model_features.json"))["model_features"]
        tbl = H.build_blocks(pd.read_parquet(DATA_DIR / "weekly_model_table.parquet"))
        w = np.sqrt(tbl["n_events"].clip(lower=1))
        spw = (tbl.presence == 0).sum() / max((tbl.presence == 1).sum(), 1)
        specs = {
            "xgboost": (lambda: xgb.XGBClassifier(
                n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=42,
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                scale_pos_weight=spw), False),
            "random_forest": (lambda: RandomForestClassifier(
                n_estimators=400, max_depth=12, min_samples_leaf=5,
                class_weight="balanced", n_jobs=-1, random_state=42), True),
        }
        for name, (mk, imp) in specs.items():
            log(f"[fig3] recomputing OOF for {name} (this takes a few minutes)")
            _, oof = H.evaluate(tbl, feats, mk, schemes=("spatiotemporal", "spatial"),
                                sample_weight=w, calibrate=True, impute=imp,
                                model_name=name, verbose=False)
            out[name] = pd.DataFrame({"presence": tbl.presence.values,
                                      "oof_spatiotemporal": oof["spatiotemporal"],
                                      "oof_spatial": oof["spatial"]})
    except Exception as e:
        log(f"[fig3] XGB/RF OOF unavailable ({type(e).__name__}: {e}) — plotting MaxEnt only")
    return out


def fig_calibration():
    oof = collect_oof()
    if not oof:
        log("[fig3] SKIP: no OOF predictions found")
        return
    models = [m for m in MODEL_ORDER if m in oof]
    schemes = [("oof_spatiotemporal", "Spatiotemporal CV"), ("oof_spatial", "Spatial CV")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [2.4, 1.0]})

    for j, (col, title) in enumerate(schemes):
        ax, axh = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1.2, label="perfect calibration")
        for m in models:
            d = oof[m]
            if col not in d.columns: continue
            g = reliability(d["presence"], d[col])
            if g is None or g.empty: continue
            ax.plot(g.mean_pred, g.obs, marker="o", ms=5, lw=1.8,
                    color=COLORS.get(m), label=PRETTY.get(m, m))
            axh.hist(d[col].dropna(), bins=30, histtype="step", lw=1.5,
                     color=COLORS.get(m), density=True)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted suitability"); ax.set_ylabel("observed presence frequency")
        ax.set_title(title, fontsize=11); ax.grid(alpha=0.3)
        axh.set_xlim(0, 1); axh.set_xlabel("predicted suitability")
        axh.set_ylabel("density"); axh.grid(alpha=0.3)
        axh.set_title("where the predictions sit", fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Reliability diagrams — points on the diagonal are well calibrated;\n"
                 "below it = over-predicting suitability (quantile bins, ≥15 obs each)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "calibration_curves.png", dpi=150); plt.close(fig)
    log("[fig3] wrote calibration_curves.png")


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_seasonal_validation()
    fig_shap_blocks()
    fig_calibration()
    log(f"\n[done] figures -> {OUT_DIR}")


if __name__ == "__main__":
    run()