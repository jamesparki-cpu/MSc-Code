from __future__ import annotations
"""
baselines.py  —  reference forecasts for the nowcast harness.

Provides the three baselines a forecast-verification reviewer expects, each
produced as an out-of-fold PREDICTION VECTOR so it flows through the same
forward-chaining folds and the same scoring code as the ML models:

  1. CONSTANT PREVALENCE  — the unconditional climatology. Two variants:
       'train' = prevalence of the training years (the honest operational
                 reference; a real forecaster does not know the test base rate)
       'test'  = prevalence of the test fold itself (the conventional
                 "sample climatology" reference; harsher on the model)
  2. CLIMATOLOGY          — historical mean presence rate for (Grid_ID, iso_week)
       computed from TRAIN YEARS ONLY, with fallback tiers:
         tier 1  (cell, week) seen in train
         tier 2  statewide mean for that week-of-year
         tier 3  train prevalence
       Resolving the reference by location AND week-of-year (rather than one
       constant) follows standard verification practice; see BASELINES.md.
  3. PERSISTENCE          — last observation carried forward.
       mode='prev_week'   the observed label at that cell one week earlier
       mode='prior_year'  the observed label at that cell, same week, last year
       raw=True   issues the observed 0/1 (conventional persistence)
       raw=False  issues train-estimated P(presence | previous state), i.e. a
                  Markov / conditional-climatology forecast. A hard 0/1 forecast
                  contributes 1.0 to Brier whenever wrong, so the raw variant is
                  punished severely; the calibrated variant is the fair Brier
                  comparator. Report both.

INFORMATION ASYMMETRY (must be disclosed, not hidden): mode='prev_week' reads
observed trap outcomes from INSIDE the test year. The ML models never see those --
they get climate lags only. So prev-week persistence has information the models
lack. It is the operationally honest baseline (at week t you do know week t-1's
catch), but if it wins, the finding is "knowing last week's catch beats modelling
this week's climate". mode='prior_year' is the information-matched variant when
the prior year is inside the training window.

COVERAGE: persistence and climatology are not defined for every row. Skill scores
computed on different denominators are not comparable, so this module always
reports coverage and provides a MATCHED SUBSET (rows where every method is
defined) -- interpret that table.

CONVENTIONS AND CAVEATS, with citations, are in BASELINES.md.
"""
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

TARGET = "presence"
YEAR   = "iso_year"
WEEK   = "iso_week"
CELL   = "Grid_ID"
WEEK_START = "week_start"
RANDOM_STATE = 42


# ============================ 1. CONSTANT PREVALENCE ========================
def constant_prevalence(df, train_mask, test_mask, source: str = "train"):
    """Unconditional reference. source='train' (operationally honest) or
    'test' (conventional sample climatology)."""
    y = df[TARGET].astype(int).to_numpy()
    p = np.full(len(df), np.nan)
    base = y[train_mask].mean() if source == "train" else y[test_mask].mean()
    p[test_mask] = base
    return p, {"prevalence_source": source, "value": round(float(base), 4)}


# ============================ 2. CLIMATOLOGY ================================
def climatology_forecast(df, train_mask, test_mask, min_obs: int = 1):
    """Mean presence rate for (cell, week) from TRAIN rows only, with fallbacks.
    Returns (predictions, diagnostics) where diagnostics records how often each
    fallback tier fired -- report this, because tier 2/3 use means a cell had no
    training history and the 'climatology' is really a seasonal average."""
    tr = df.loc[train_mask, [CELL, WEEK, TARGET]]
    cw = (tr.groupby([CELL, WEEK])[TARGET].agg(["mean", "size"])
            .rename(columns={"mean": "clim_cw", "size": "n_cw"}))
    wk = tr.groupby(WEEK)[TARGET].mean().rename("clim_w")
    grand = tr[TARGET].mean()

    te = df.loc[test_mask, [CELL, WEEK]].copy()
    te = te.join(cw, on=[CELL, WEEK]).join(wk, on=WEEK)
    ok_cw = te["clim_cw"].notna() & (te["n_cw"] >= min_obs)
    pred = np.where(ok_cw, te["clim_cw"],
                    np.where(te["clim_w"].notna(), te["clim_w"], grand))

    p = np.full(len(df), np.nan)
    p[test_mask] = pred
    n = len(te)
    diag = {"tier1_cell_week_pct": round(100 * float(ok_cw.mean()), 1),
            "tier2_week_only_pct": round(100 * float((~ok_cw & te["clim_w"].notna()).mean()), 1),
            "tier3_grand_pct": round(100 * float((~ok_cw & te["clim_w"].isna()).mean()), 1),
            "n_test": int(n)}
    return p, diag


# ============================ 3. PERSISTENCE ================================
def _lookup_previous(df, mode):
    """Map every row to the observed label of its predecessor, or NaN.
    prev_week uses week_start - 7 days, which handles ISO year boundaries
    (week 1 correctly points at week 52/53 of the previous ISO year)."""
    obs = df[[CELL, TARGET]].copy()
    if mode == "prev_week":
        ws = pd.to_datetime(df[WEEK_START])
        key = pd.MultiIndex.from_arrays([df[CELL], ws])
        want = pd.MultiIndex.from_arrays([df[CELL], ws - pd.Timedelta(days=7)])
    elif mode == "prior_year":
        key = pd.MultiIndex.from_arrays([df[CELL], df[YEAR], df[WEEK]])
        want = pd.MultiIndex.from_arrays([df[CELL], df[YEAR] - 1, df[WEEK]])
    else:
        raise ValueError("mode must be 'prev_week' or 'prior_year'")
    lut = pd.Series(obs[TARGET].to_numpy(), index=key)
    lut = lut[~lut.index.duplicated(keep="first")]
    return lut.reindex(want).to_numpy(dtype=float)      # NaN where unavailable


def persistence_forecast(df, train_mask, test_mask, mode: str = "prev_week",
                         raw: bool = False):
    """Last-observation-carried-forward. raw=True issues 0/1; raw=False issues
    train-estimated P(presence | previous state)."""
    prev = _lookup_previous(df, mode)
    y = df[TARGET].astype(int).to_numpy()
    p = np.full(len(df), np.nan)

    if raw:
        vals = prev
        diag = {"variant": f"{mode}_raw01"}
    else:
        # transition probabilities from TRAIN rows only
        m1 = train_mask & (prev == 1)
        m0 = train_mask & (prev == 0)
        p1 = y[m1].mean() if m1.sum() else np.nan
        p0 = y[m0].mean() if m0.sum() else np.nan
        vals = np.where(prev == 1, p1, np.where(prev == 0, p0, np.nan))
        diag = {"variant": f"{mode}_calibrated",
                "P(pres|prev_pres)": round(float(p1), 4) if np.isfinite(p1) else None,
                "P(pres|prev_abs)": round(float(p0), 4) if np.isfinite(p0) else None}

    p[test_mask] = vals[test_mask]
    cov = np.isfinite(p[test_mask]).mean() if test_mask.sum() else np.nan
    diag["coverage_pct"] = round(100 * float(cov), 1)
    return p, diag


# ============================ SCORING =======================================
def brier_skill(y, p, ref):
    """BSS = 1 - Brier(forecast)/Brier(reference), on rows where BOTH are
    defined. Reference Brier must be computed on the same rows as the forecast
    Brier or the skill score is not interpretable."""
    m = np.isfinite(p) & np.isfinite(ref) & np.isfinite(y)
    if m.sum() == 0:
        return np.nan, 0
    b = brier_score_loss(y[m], p[m]); br = brier_score_loss(y[m], ref[m])
    return (1 - b / br) if br > 0 else np.nan, int(m.sum())


def score_forecast(y, p, refs: Optional[Dict[str, np.ndarray]] = None,
                   constant_forecast: bool = False):
    """Discrimination + calibration for one forecast, plus BSS against each
    supplied reference. NOTE the ROC/PR baselines: ROC's no-skill value is 0.5
    by construction, and the PR no-skill value is prevalence -- but a resolved
    climatology is a MUCH stronger reference than either, which is why the
    baselines are scored on these metrics too and compared directly."""
    y = np.asarray(y, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    out = {"n_scored": int(m.sum()),
           "coverage_pct": round(100 * float(m.mean()), 1)}
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return out
    ys, ps = y[m].astype(int), p[m]
    prev = ys.mean()
    out.update(prevalence=round(float(prev), 3),
               brier=round(brier_score_loss(ys, ps), 4))
    if constant_forecast:
        # A constant forecast has NO discrimination: its true ROC is 0.5 and its
        # PR-AUC is prevalence. But a per-fold constant, POOLED across folds,
        # becomes a step function whose steps can correlate with the outcome,
        # yielding meaningless pooled ROC values (e.g. 0.22). Those must not be
        # tabulated, so discrimination metrics are suppressed for such forecasts.
        out.update(roc_auc=np.nan, pr_auc=np.nan,
                   pr_baseline=round(float(prev), 3), pr_lift=np.nan,
                   discrimination="undefined (constant forecast)")
    else:
        out.update(roc_auc=round(roc_auc_score(ys, ps), 3),
                   pr_auc=round(average_precision_score(ys, ps), 3),
                   pr_baseline=round(float(prev), 3),
                   pr_lift=round(average_precision_score(ys, ps) - prev, 3))
    for name, ref in (refs or {}).items():
        bss, n = brier_skill(y, p, np.asarray(ref, dtype=float))
        out[f"bss_vs_{name}"] = round(bss, 3) if np.isfinite(bss) else np.nan
        out[f"n_vs_{name}"] = n
    return out


def matched_mask(*preds):
    """Rows where every supplied forecast is defined. Scores computed here are
    comparable across methods; scores on each method's own coverage are not."""
    m = np.isfinite(np.asarray(preds[0], dtype=float))
    for p in preds[1:]:
        m &= np.isfinite(np.asarray(p, dtype=float))
    return m


# ============================ UNCERTAINTY ===================================
def paired_block_bootstrap(y, p_model, p_ref, groups, n_boot: int = 2000,
                           statistic: str = "brier_diff", random_state=RANDOM_STATE):
    """Paired bootstrap CI on (reference Brier - model Brier); positive means the
    model is better. Resamples whole CELLS (blocks) with replacement rather than
    rows, because trap cell-weeks are strongly spatially and serially correlated
    and a row bootstrap would give intervals that are far too narrow.
    Paired = the same resampled cells are used for both forecasts."""
    y = np.asarray(y, dtype=float)
    m = np.isfinite(p_model) & np.isfinite(p_ref) & np.isfinite(y)
    ys, pm, pr, gs = y[m].astype(int), np.asarray(p_model)[m], np.asarray(p_ref)[m], np.asarray(groups)[m]
    cells = np.unique(gs)
    idx_by_cell = {c: np.where(gs == c)[0] for c in cells}
    rng = np.random.default_rng(random_state)

    def stat(ix):
        if len(np.unique(ys[ix])) < 2:
            return np.nan
        bm = brier_score_loss(ys[ix], pm[ix]); br = brier_score_loss(ys[ix], pr[ix])
        if statistic == "brier_diff":
            return br - bm                       # >0 : model better
        if statistic == "bss":
            return 1 - bm / br if br > 0 else np.nan
        if statistic == "roc_diff":
            return roc_auc_score(ys[ix], pm[ix]) - roc_auc_score(ys[ix], pr[ix])
        raise ValueError(statistic)

    point = stat(np.arange(len(ys)))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(cells, size=len(cells), replace=True)
        ix = np.concatenate([idx_by_cell[c] for c in pick])
        draws[b] = stat(ix)
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan)
    return {"statistic": statistic,
            "point": round(float(point), 4) if np.isfinite(point) else np.nan,
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "n_blocks": int(len(cells)), "n_boot": int(len(draws)),
            "significant": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))}


# ============================ FOLD-WISE ORCHESTRATION =======================
def build_baselines_for_masks(df, train_mask, test_mask, prevalence_source="train",
                              include_persistence=True):
    """Scheme-agnostic baseline builder: takes boolean train/test masks, so it
    works with ANY cross-validation scheme -- forward-chaining year folds, or the
    blocked spatial / temporal / spatiotemporal folds of the weekly pipeline.

    include_persistence=False MUST be used for a SPATIAL holdout. In spatial CV a
    whole cell block is withheld, so the 'previous week' at a held-out cell is
    itself inside the held-out block: a persistence forecast there reads observed
    labels from the very region the scheme is meant to withhold. It would not be
    a spatial-transfer baseline at all. See BASELINES.md B8.
    """
    preds, diags = {}, {}
    preds["prevalence"], diags["prevalence"] = constant_prevalence(
        df, train_mask, test_mask, source=prevalence_source)
    preds["climatology"], diags["climatology"] = climatology_forecast(
        df, train_mask, test_mask)
    if include_persistence:
        preds["persistence"], diags["persistence"] = persistence_forecast(
            df, train_mask, test_mask, mode="prev_week", raw=False)
        preds["persistence_raw"], diags["persistence_raw"] = persistence_forecast(
            df, train_mask, test_mask, mode="prev_week", raw=True)
        preds["persistence_prioryear"], diags["persistence_prioryear"] = persistence_forecast(
            df, train_mask, test_mask, mode="prior_year", raw=False)
    return preds, diags


def build_fold_baselines(df, train_years, test_year, prevalence_source="train"):
    """All baselines for ONE forward-chaining fold. Every baseline is built from
    TRAIN YEARS ONLY (except persistence's within-test-year lookback, which is
    the disclosed asymmetry). Returns (predictions, diagnostics, test_mask)."""
    train_mask = df[YEAR].isin(train_years).to_numpy()
    test_mask = (df[YEAR] == test_year).to_numpy()
    preds, diags = build_baselines_for_masks(
        df, train_mask, test_mask, prevalence_source, include_persistence=True)
    return preds, diags, test_mask


def evaluate_with_baselines(df, model_oof: Dict[str, np.ndarray], folds,
                            prevalence_source="train", n_boot=1000, verbose=True):
    """Score models AND baselines on identical folds.

    df         : the model table (needs presence, iso_year, iso_week, Grid_ID, week_start)
    model_oof  : {model_name: full-length OOF prediction vector} from the nowcast
                 harness -- pass nowcast_cv's oof arrays straight in
    folds      : iterable of (train_years, test_year), e.g.
                 nowcast_cv.forward_chain_folds(sorted(df.iso_year.unique()))

    Returns (per_fold_df, pooled_df, matched_df, diagnostics_df, bootstrap_df):
      per_fold / pooled : each method on its OWN coverage (coverage_pct reported)
      matched           : every method on the COMMON subset -- interpret this one
      bootstrap         : paired block-bootstrap CI on model-vs-climatology
    """
    y = df[TARGET].astype(int).to_numpy()
    cells = df[CELL].to_numpy()
    rows, diag_rows = [], []
    base_oof = {}                      # accumulate baseline predictions across folds

    for train_years, test_year in folds:
        preds, diags, test_mask = build_fold_baselines(
            df, train_years, test_year, prevalence_source)
        for k, v in preds.items():
            base_oof.setdefault(k, np.full(len(df), np.nan))
            base_oof[k][test_mask] = v[test_mask]
        for k, d in diags.items():
            diag_rows.append({"test_year": test_year, "baseline": k, **d})

        refs = {"prevalence": preds["prevalence"],
                "climatology": preds["climatology"],
                "persistence": preds["persistence"]}
        for name, vec in list(model_oof.items()) + [(f"BASELINE_{k}", v) for k, v in preds.items()]:
            v = np.asarray(vec, dtype=float).copy()
            v[~test_mask] = np.nan                       # score this fold only
            s = score_forecast(y, v, refs)
            s.update(method=name, test_year=test_year,
                     train_years=f"{min(train_years)}-{max(train_years)}")
            rows.append(s)
        if verbose:
            c = diags["climatology"]; p = diags["persistence"]
            print(f"[base] test {test_year}: climatology tier1 {c['tier1_cell_week_pct']}% "
                  f"/ tier2 {c['tier2_week_only_pct']}% | persistence coverage "
                  f"{p['coverage_pct']}%", flush=True)

    per_fold = pd.DataFrame(rows)

    # ---- pooled, each method on its own coverage
    refs_all = {k: base_oof[k] for k in ("prevalence", "climatology", "persistence")}
    pooled = []
    for name, vec in list(model_oof.items()) + [(f"BASELINE_{k}", v) for k, v in base_oof.items()]:
        s = score_forecast(y, np.asarray(vec, dtype=float), refs_all)
        s["method"] = name; pooled.append(s)
    pooled = pd.DataFrame(pooled)

    # ---- matched subset: every model and every baseline defined
    all_vecs = list(model_oof.values()) + list(base_oof.values())
    mm = matched_mask(*all_vecs)
    matched = []
    for name, vec in list(model_oof.items()) + [(f"BASELINE_{k}", v) for k, v in base_oof.items()]:
        v = np.asarray(vec, dtype=float).copy(); v[~mm] = np.nan
        r = {k: (np.asarray(r_, dtype=float).copy()) for k, r_ in refs_all.items()}
        for k in r: r[k][~mm] = np.nan
        s = score_forecast(y, v, r)
        s["method"] = name; matched.append(s)
    matched = pd.DataFrame(matched)
    if verbose:
        print(f"[base] matched subset: {int(mm.sum()):,}/{len(df):,} rows "
              f"({100*mm.mean():.1f}%) -- interpret the matched table", flush=True)

    # ---- paired block bootstrap: each model vs climatology, on the matched subset
    boot = []
    for name, vec in model_oof.items():
        v = np.asarray(vec, dtype=float).copy(); v[~mm] = np.nan
        ref = base_oof["climatology"].copy(); ref[~mm] = np.nan
        for stat in ("brier_diff", "roc_diff"):
            b = paired_block_bootstrap(y, v, ref, cells, n_boot=n_boot, statistic=stat)
            b.update(model=name, reference="climatology"); boot.append(b)
    boot = pd.DataFrame(boot)

    return per_fold, pooled, matched, pd.DataFrame(diag_rows), boot