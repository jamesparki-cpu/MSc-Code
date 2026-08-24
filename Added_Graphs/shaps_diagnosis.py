from __future__ import annotations
"""
shap_diagnostics.py  —  two checks on the SHAP comparison itself.

  DIAG 1  shap_concordance.png / shap_concordance.csv          FAST (reads CSVs)
          Do the four models agree on WHICH predictors matter, or do they reach
          similar performance by different routes? Pairwise Spearman rank
          correlation of pct_of_total across the 31 features, plus top-10
          overlap. Nothing in the chapter currently answers this: the grouped
          block chart shows composition, not agreement.

  DIAG 2  shap_explainer_check.png / shap_explainer_check.csv  SLOW (refits XGB)
          Is that agreement structure an ARTEFACT OF THE EXPLAINER?

          This matters because the observed pattern splits by model family --
          trees agree with each other, MaxEnts agree with each other, and
          cross-family agreement is much weaker -- and the family boundary is
          EXACTLY the explainer boundary (exact TreeSHAP for XGB/RF,
          approximate PermutationExplainer for MaxEnt). The two explanations
          are perfectly confounded in the existing outputs.

          The test: run BOTH explainers on the SAME model (XGBoost) and the SAME
          rows, then compare. High concordance means the explainer is not
          driving the family split and the cross-model comparison stands. Low
          concordance means part of DIAG 1 is measuring the explainer, and must
          be reported as such.

          One model, one row set, both explainers: the only thing that varies is
          the method, so whatever difference appears is attributable to it.

OUTPUT -> shap_dir
"""
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import report_style as S

try:
    import shap_common2 as SC
except ImportError:
    SC = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ============================== CONFIG ======================================
with open("config.json") as f:
    cfg = json.load(f)
DATA_DIR = Path(cfg.get("weekly_xg_dir", "."))
OUT_DIR = Path(cfg.get("shap_dir", str(DATA_DIR / "SHAP_Results")))

TOP_K = 10                 # overlap is reported over the top-K features
N_EXPLAINER_CHECK = 200    # rows for DIAG 2 (~0.9 s/row on the permutation side)
RUN_EXPLAINER_CHECK = True # set False to run DIAG 1 only (instant)
RANDOM_STATE = 42

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=5.0, random_state=RANDOM_STATE,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist")

def log(m): print(m, flush=True)
# ============================================================================


def load_importances() -> dict:
    """{model: DataFrame indexed by feature} from shap_importance_<model>.csv."""
    out = {}
    for m in S.MODEL_ORDER:
        f = OUT_DIR / f"shap_importance_{m}.csv"
        if f.exists():
            out[m] = pd.read_csv(f).set_index("feature")
            log(f"[diag1] read {f.name} ({len(out[m])} features)")
    return out


# ------------------------------------------------------------------- DIAG 1
def diag_concordance():
    imp = load_importances()
    if len(imp) < 2:
        log("[diag1] SKIP: need at least two shap_importance_<model>.csv files")
        return
    models = S.models_in(imp.keys())
    feats = sorted(set.intersection(*[set(imp[m].index) for m in models]))
    log(f"[diag1] {len(models)} models over {len(feats)} common features")

    rows, R = [], pd.DataFrame(np.eye(len(models)), index=models, columns=models)
    OV = pd.DataFrame(np.nan, index=models, columns=models)
    for a, b in itertools.combinations(models, 2):
        rho, _ = spearmanr(imp[a].loc[feats, "pct_of_total"],
                           imp[b].loc[feats, "pct_of_total"])
        ta = set(imp[a].nlargest(TOP_K, "pct_of_total").index)
        tb = set(imp[b].nlargest(TOP_K, "pct_of_total").index)
        ov = len(ta & tb)
        R.loc[a, b] = R.loc[b, a] = round(float(rho), 3)
        OV.loc[a, b] = OV.loc[b, a] = ov
        rows.append(dict(model_a=a, model_b=b, spearman_rho=round(float(rho), 3),
                         top_k=TOP_K, top_k_overlap=ov,
                         shared_top_features=", ".join(sorted(ta & tb))))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_DIR / "shap_concordance.csv", index=False)
    log("\n[diag1] pairwise agreement on feature importance:")
    log(tab[["model_a", "model_b", "spearman_rho", "top_k_overlap"]].to_string(index=False))

    # ---- figure: rho above the diagonal, top-K overlap below ----
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    M = np.full((len(models), len(models)), np.nan)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i < j:
                M[i, j] = R.loc[a, b]
            elif i > j:
                M[i, j] = OV.loc[a, b] / TOP_K      # rescaled to share the map
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j:
                ax.text(j, i, "\u2014", ha="center", va="center", fontsize=11,
                        color="0.5")
            elif i < j:
                ax.text(j, i, f"\u03c1 = {R.loc[a, b]:.3f}", ha="center",
                        va="center", fontsize=9)
            else:
                ax.text(j, i, f"{int(OV.loc[a, b])}/{TOP_K}", ha="center",
                        va="center", fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([S.model_label(m, wrapped=True) for m in models],
                       fontsize=8, rotation=25, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([S.model_label(m, wrapped=True) for m in models], fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, fraction=0.045, pad=0.03)
    cb.set_label("agreement (Spearman \u03c1 above, overlap fraction below)")
    cb.outline.set_linewidth(0.6)
    ax.set_title("Do the models agree on which predictors matter?\n"
                 f"upper triangle: Spearman \u03c1 over all features; "
                 f"lower: shared features in each model's top {TOP_K}",
                 fontsize=11)
    fig.tight_layout()
    S.save(fig, OUT_DIR / "shap_concordance.png")

    # the structural reading, stated numerically
    fam = {"xgboost": "tree", "random_forest": "tree",
           "maxent_targetgroup": "maxent", "maxent_vanilla": "maxent"}
    within = [r.spearman_rho for r in tab.itertuples()
              if fam.get(r.model_a) == fam.get(r.model_b)]
    across = [r.spearman_rho for r in tab.itertuples()
              if fam.get(r.model_a) != fam.get(r.model_b)]
    if within and across:
        log(f"\n[diag1] within-family mean \u03c1 = {np.mean(within):.3f} "
            f"| across-family mean \u03c1 = {np.mean(across):.3f}")
        log("[diag1] NOTE the family boundary is also the EXPLAINER boundary "
            "(TreeSHAP vs PermutationExplainer). DIAG 2 tests whether the split "
            "is attributable to the method rather than to the models.")
    return tab


# ------------------------------------------------------------------- DIAG 2
def diag_explainer_check():
    if SC is None or xgb is None:
        log("[diag2] SKIP: needs shaps_common and xgboost")
        return
    feats = json.load(open(DATA_DIR / "model_features.json"))["model_features"]
    table = pd.read_parquet(DATA_DIR / "weekly_model_table.parquet")

    y = table["presence"].astype(int)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    model = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    w = np.sqrt(table["n_events"].clip(lower=1)) if "n_events" in table else None
    model.fit(table[feats], y, sample_weight=w)
    log("[diag2] fitted XGBoost (same params as shaps_run.py)")

    samp = SC.stratified_sample(table, N_EXPLAINER_CHECK)
    Xs = samp[feats]
    log(f"[diag2] explaining the SAME {len(Xs):,} rows with both methods")

    sv_tree = SC.explain_tree(model, Xs)
    log("[diag2] TreeSHAP done (exact)")
    sv_perm = SC.explain_permutation(model, Xs, feats)
    log("[diag2] PermutationExplainer done (approximate)")

    t = np.abs(sv_tree).mean(axis=0)
    p = np.abs(sv_perm).mean(axis=0)
    d = pd.DataFrame({"feature": list(feats),
                      "mean_abs_shap_tree": t,
                      "mean_abs_shap_permutation": p})
    d["pct_tree"] = 100 * d.mean_abs_shap_tree / d.mean_abs_shap_tree.sum()
    d["pct_permutation"] = 100 * d.mean_abs_shap_permutation / d.mean_abs_shap_permutation.sum()
    d["rank_tree"] = d.pct_tree.rank(ascending=False).astype(int)
    d["rank_permutation"] = d.pct_permutation.rank(ascending=False).astype(int)
    d["rank_shift"] = (d.rank_permutation - d.rank_tree)
    if SC is not None:
        d["group"] = d.feature.map(SC.assign_groups(feats))
    d = d.sort_values("pct_tree", ascending=False)
    d.round(4).to_csv(OUT_DIR / "shap_explainer_check.csv", index=False)

    rho, _ = spearmanr(d.pct_tree, d.pct_permutation)
    tt = set(d.nlargest(TOP_K, "pct_tree").feature)
    tp = set(d.nlargest(TOP_K, "pct_permutation").feature)
    log(f"\n[diag2] TreeSHAP vs PermutationExplainer on ONE model, SAME rows:")
    log(f"[diag2]   Spearman rho = {rho:+.3f}")
    log(f"[diag2]   top-{TOP_K} overlap = {len(tt & tp)}/{TOP_K}")
    log(f"[diag2]   largest rank shift = {int(d.rank_shift.abs().max())} places")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    cols = ([S.BLOCK_COLOURS.get(g, "#999") for g in d.group]
            if "group" in d.columns else S.MODEL_COLOURS["xgboost"])
    ax.scatter(d.pct_tree, d.pct_permutation, s=44, c=cols,
               edgecolors=S.BAR_EDGE, linewidths=0.5, zorder=3)
    hi = float(max(d.pct_tree.max(), d.pct_permutation.max())) * 1.08
    ax.plot([0, hi], [0, hi], ls="--", lw=1.1, color=S.GREY, zorder=1)
    for r in d.head(6).itertuples():
        ax.annotate(r.feature, (r.pct_tree, r.pct_permutation), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points", color="0.3")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("% of total |SHAP| — TreeSHAP (exact)")
    ax.set_ylabel("% of total |SHAP| — PermutationExplainer")
    ax.grid(alpha=0.3)
    ax.set_title(f"(a) Same model, same rows, two methods\n"
                 f"Spearman \u03c1 = {rho:+.3f}, top-{TOP_K} overlap "
                 f"{len(tt & tp)}/{TOP_K}", fontsize=10)

    ax = axes[1]
    dd = d.head(15).iloc[::-1]
    yy = np.arange(len(dd))
    ax.hlines(yy, dd.pct_tree, dd.pct_permutation, color="0.75", lw=1.2, zorder=1)
    ax.scatter(dd.pct_tree, yy, s=40, color="#2166ac", zorder=3, label="TreeSHAP")
    ax.scatter(dd.pct_permutation, yy, s=40, color="#e08214", zorder=3,
               label="Permutation")
    ax.set_yticks(yy); ax.set_yticklabels(dd.feature, fontsize=7)
    ax.set_xlabel(S.metric_label("pct_of_total"))
    ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("(b) Per-feature disagreement, top 15 by TreeSHAP", fontsize=10)

    fig.suptitle("Explainer check — is the cross-model SHAP comparison an "
                 "artefact of using different explainers?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    S.save(fig, OUT_DIR / "shap_explainer_check.png")

    log("\n[read] HIGH rho means the explainer is not driving the family split "
        "in DIAG 1, and the cross-model comparison stands. LOW rho means part "
        "of DIAG 1 measures the METHOD rather than the models, and every "
        "tree-versus-MaxEnt SHAP statement needs that caveat.")
    return d


def run():
    S.set_thesis_style(constrained=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diag_concordance()
    if RUN_EXPLAINER_CHECK:
        diag_explainer_check()
    else:
        log("[diag2] skipped (RUN_EXPLAINER_CHECK = False)")
    log(f"\n[done] SHAP diagnostics -> {OUT_DIR}")


if __name__ == "__main__":
    run()