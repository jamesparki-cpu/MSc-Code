from __future__ import annotations
"""
nowcast_cv.py  —  forward-chaining (expanding-window) CV for NOWCASTING.

Self-contained: imports nothing from the climatological pipeline. Validates the
operational claim "given data up to now, predict the upcoming week in a surveyed
region" -- so folds train ONLY on years strictly BEFORE the test year, mirroring
deployment. This is different from the earlier leave-one-year-out CV, which let
future years train on past ones.

TEMPORAL-ONLY (no spatial holdout): the nowcast claim is about surveyed Florida,
where the models work. The spatial-transfer failure is already established
elsewhere and is not re-tested here.

REPORTING (both, deliberately):
  * per-fold  -- early folds (train on 1-2 years) are weak, late folds strong;
                 a pooled-only number would misrepresent the model. You MUST see
                 the progression.
  * pooled    -- all forward-chained OOF predictions in one metric.
  * headline  -- a clean single split: train 2013..(N-1), test final year N.

Calibration: isotonic fit inside each fold on a held-out slice of that fold's
TRAIN years only (never the test year) -- honest, leak-free.

Metrics: ROC-AUC, PR-AUC + baseline + lift, Brier Skill Score (same definitions
as the main pipeline, so numbers are comparable to the climatological results).

NOTE ON THE BSS REFERENCE: score() builds the Brier reference from the prevalence
of the rows being scored, i.e. the TEST fold's own base rate. That is the
conventional "sample climatology" reference and it is conservative (harder for the
model to beat) because it already knows the test period's prevalence. The
operationally honest alternative uses the TRAINING prevalence, and stronger
references -- a resolved (cell, week) climatology, and persistence -- are provided
by baselines.py. Whichever is quoted, state which. See BASELINES.md.

OOF ACCESS: pass return_oof=True to evaluate_nowcast to also receive the pooled
out-of-fold prediction vector. It is required for the baseline comparison,
reliability diagrams and bootstrap confidence intervals, none of which can be
reconstructed from the aggregate metrics. The default return signature is
unchanged, so existing callers keep working.
"""
from typing import Callable, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

TARGET = "presence"
YEAR = "iso_year"
RANDOM_STATE = 42
MIN_TRAIN_YEARS = 1        # smallest expanding window (train >=1 year before test)


# ----- forward-chaining fold generator --------------------------------------
def forward_chain_folds(years: Sequence[int], min_train: int = MIN_TRAIN_YEARS):
    """Yield (train_years, test_year) expanding-window pairs. e.g. for
    2013..2018: (2013->2014), (2013-14->2015), ... (2013-17->2018)."""
    ys = sorted(set(int(y) for y in years))
    for i in range(min_train, len(ys)):
        yield ys[:i], ys[i]


# ----- one fold: fit (+optional calibrate) on train years, predict test year -
def _fit_predict(make_model, Xtr, ytr, wtr, Xte, impute, medians):
    m = make_model()
    if impute:
        Xtr, Xte = Xtr.fillna(medians), Xte.fillna(medians)
    try:
        m.fit(Xtr, ytr, sample_weight=wtr) if wtr is not None else m.fit(Xtr, ytr)
    except TypeError:
        m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def _fold_predict(df, X, y, w, train_years, test_year, make_model,
                  calibrate, impute, medians, cal_fraction=0.25):
    tr = df[YEAR].isin(train_years).to_numpy()
    te = (df[YEAR] == test_year).to_numpy()
    Xtr, ytr = X[tr], y[tr]
    wtr = w[tr] if w is not None else None
    if not calibrate or len(train_years) < 2:
        # need >=2 train years to hold one out for calibration; else skip calib
        return _fit_predict(make_model, Xtr, ytr, wtr, X[te], impute, medians), te
    # calibrate on the LAST train year held out (closest to the test year)
    cal_year = max(train_years); fit_years = [yv for yv in train_years if yv != cal_year]
    fmask = df[YEAR].isin(fit_years).to_numpy()[tr]
    cmask = (df[YEAR] == cal_year).to_numpy()[tr]
    wfit = wtr[fmask] if wtr is not None else None
    p_cal = _fit_predict(make_model, Xtr[fmask], ytr[fmask], wfit, Xtr[cmask], impute, medians)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, ytr[cmask])
    p_te = _fit_predict(make_model, Xtr[fmask], ytr[fmask], wfit, X[te], impute, medians)
    return iso.predict(p_te), te


# ----- metrics --------------------------------------------------------------
def score(y, p):
    m = ~np.isnan(p); y, p = np.asarray(y)[m], p[m]
    if len(np.unique(y)) < 2:
        return dict(n=int(m.sum()), prevalence=round(float(y.mean()), 3),
                    roc_auc=np.nan, pr_auc=np.nan, pr_baseline=round(float(y.mean()),3),
                    pr_lift=np.nan, bss=np.nan)
    prev = y.mean()
    brier = brier_score_loss(y, p); brier_base = brier_score_loss(y, np.full_like(p, prev))
    return dict(n=int(m.sum()), prevalence=round(float(prev), 3),
                roc_auc=round(roc_auc_score(y, p), 3),
                pr_auc=round(average_precision_score(y, p), 3),
                pr_baseline=round(float(prev), 3),
                pr_lift=round(average_precision_score(y, p) - prev, 3),
                brier=round(brier, 4),
                bss=round(1 - brier / brier_base, 3) if brier_base > 0 else np.nan)


# ----- orchestrator ---------------------------------------------------------
def evaluate_nowcast(df, feats, make_model, sample_weight=None,
                     calibrate=True, impute=False, model_name="model", verbose=True,
                     return_oof=False):
    """Forward-chaining evaluation. Returns (per_fold_df, pooled_row, headline_row),
    or (per_fold_df, pooled_row, headline_row, oof) when return_oof=True.
      per_fold : one row per (train_years -> test_year) fold
      pooled   : one metric over all forward-chained OOF predictions
      headline : the final split train[:-1] -> last year (the deployment case)
      oof      : float array of len(df), the pooled out-of-fold prediction per row,
                 NaN for rows never in a test fold (the first year is train-only
                 under forward chaining). Positionally aligned with df.
    """
    X = df[list(feats)]; y = df[TARGET].astype(int)
    w = sample_weight
    medians = X.median(numeric_only=True) if impute else None
    years = sorted(df[YEAR].unique())

    oof = np.full(len(df), np.nan)
    test_of = np.full(len(df), np.nan)
    rows = []
    for train_years, test_year in forward_chain_folds(years):
        p, te = _fold_predict(df, X, y, w, train_years, test_year, make_model,
                              calibrate, impute, medians)
        oof[te] = p
        test_of[te] = test_year
        s = score(y[te].to_numpy(), p)
        s.update(model=model_name, train_years=f"{min(train_years)}-{max(train_years)}",
                 n_train_years=len(train_years), test_year=test_year)
        rows.append(s)
        if verbose:
            print(f"[{model_name}] train {min(train_years)}-{max(train_years)} "
                  f"({len(train_years)}y) -> test {test_year}: "
                  f"ROC {s['roc_auc']} | PR-lift {s['pr_lift']} | BSS {s['bss']}", flush=True)

    per_fold = pd.DataFrame(rows)
    pooled = score(y.to_numpy(), oof); pooled.update(model=model_name, fold="pooled_OOF")
    headline = rows[-1].copy(); headline["fold"] = "headline_final_year"
    if verbose:
        print(f"[{model_name}] POOLED forward-chain: ROC {pooled['roc_auc']} | "
              f"BSS {pooled['bss']}", flush=True)
        print(f"[{model_name}] HEADLINE (train ->{years[-1]}): ROC {headline['roc_auc']} | "
              f"BSS {headline['bss']}", flush=True)
    return per_fold, pd.Series(pooled), pd.Series(headline), oof, test_of  