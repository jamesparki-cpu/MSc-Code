from __future__ import annotations
"""
report_style.py  —  single source of truth for figure and table conventions.

Imported by every script that writes a figure or a metrics table, so that
ordering, colour, labelling, decimal places and typography are decided ONCE and
cannot drift between chapters.

No config.json dependency and no project imports: this module is pure
constants + matplotlib helpers, so it is safe to import from anywhere in the
pipeline without circularity.

CONTENTS
  set_thesis_style()   typography (lifted verbatim from the map scripts)
  MODEL_ORDER          canonical model ordering, used by every figure
  SCHEME_ORDER         canonical CV-scheme ordering
  PRETTY / SCHEME_PRETTY / METRIC_LABELS / BLOCK_PRETTY   display names
  MODEL_COLOURS / SCHEME_COLOURS / BLOCK_COLOURS          three DISJOINT palettes
  DECIMALS / fmt()     fixed decimal places per metric
  REFERENCE_LINE       no-skill reference value per metric
  helpers              models_in, schemes_in, wrap, add_reference_line,
                       annotate_bars, annotate_n, save

SCOPE
  Applies to METRIC FIGURES ONLY -- bar charts, forest plots, reliability
  diagrams, SHAP panels, correlation/dendrogram plots.

  The four existing map scripts (render_maps.py, render_maxent_maps.py,
  validation_2018_maps.py, maxent_validation_2018_map.py) are FROZEN and are
  NOT migrated: they already vendor an identical copy of set_thesis_style(),
  so their output matches these figures without re-rendering. Do not import
  this module into them.

  New map scripts written from here on SHOULD import set_thesis_style() from
  here rather than pasting another copy.

  CONSEQUENCE: set_thesis_style() below is duplicated in five places. If you
  ever change the font stack or label sizes here, the four frozen map scripts
  will silently diverge -- either re-render them at that point, or accept the
  mismatch deliberately. Nothing else in this module is duplicated anywhere.

USAGE
    import report_style as S
    S.set_thesis_style(constrained=False)      # metric figures
    models = S.models_in(df.model)
    ax.bar(x, vals, color=[S.MODEL_COLOURS[m] for m in models])
    ax.set_ylabel(S.metric_label("roc_auc"))
    S.add_reference_line(ax, "roc_auc")
    S.save(fig, OUT_DIR / "model_comparison.png")
"""
import matplotlib.pyplot as plt
import numpy as np

# ===========================================================================
# TYPOGRAPHY  —  copied verbatim from the existing map scripts so that metric
# figures match the already-rendered maps WITHOUT re-rendering them. The map
# scripts keep their own copy and are not touched; see SCOPE above.
# ===========================================================================
def set_thesis_style(constrained: bool = True):
    """Serif family, body-text-scaled labels. Call once at the top of run().

    constrained=True   matches the existing maps (constrained_layout on).
    constrained=False  for figures that call fig.tight_layout(rect=...) with a
                       suptitle -- matplotlib warns and ignores tight_layout if
                       constrained_layout is also on, so the metric-figure
                       scripts pass False.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "figure.constrained_layout.use": bool(constrained),
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


# ===========================================================================
# PAGE GEOMETRY
# A figure saved 15 in wide and then scaled to fit a 6.3 in text column renders
# its 11 pt labels at roughly 4.6 pt. Author figures at final print size.
# ===========================================================================
FIG_W_FULL = 6.3          # inches: A4 portrait, 2.5 cm margins
FIG_W_WIDE = 9.0          # landscape / fold-out / appendix figure
FIG_W_HALF = 3.05         # two side by side
FIG_DPI    = 300          # vector-ish line art destined for print
MAP_DPI    = 220          # what the frozen map scripts already used; recorded
                          # here for reference and for any FUTURE map script


# ===========================================================================
# CANONICAL ORDERING
# ===========================================================================
# DECISION: MaxEnt-vanilla LAST. The first three share n = 21,608 and
# prevalence 0.776; vanilla is evaluated on n = 26,667 at prevalence 0.628 and
# is therefore not directly comparable. Placing it last keeps the comparable
# trio adjacent and visually isolates the caveated model.
# NOTE: compare_models.py currently orders vanilla third -- change it there.
MODEL_ORDER = ["xgboost", "random_forest", "maxent_targetgroup", "maxent_vanilla"]

# Degradation order: the schemes that behave alike, then the one that collapses.
SCHEME_ORDER = ["spatiotemporal", "temporal", "spatiotemporal_3blocks",
                "spatial", "forward_chaining"]

BLOCK_ORDER = ["land_cover", "seasonality", "terrain", "precipitation",
               "temperature", "vegetation", "moisture", "other"]


# ===========================================================================
# DISPLAY NAMES
# Store names UNWRAPPED. Insert line breaks at plot time with wrap(), so the
# same constant serves table captions, legends and tick labels.
# ===========================================================================
PRETTY = {
    "xgboost":            "XGBoost",
    "random_forest":      "Random Forest",
    "maxent_targetgroup": "MaxEnt (target-group)",
    "maxent_vanilla":     "MaxEnt (vanilla)",
}

SCHEME_PRETTY = {
    "spatiotemporal":          "Spatiotemporal CV",
    "temporal":                "Temporal CV",
    "spatiotemporal_3blocks":  "Spatiotemporal CV (3 blocks)",
    "spatial":                 "Spatial CV",
    "forward_chaining":        "Forward-chaining",
}

BLOCK_PRETTY = {
    "land_cover": "land cover", "seasonality": "seasonality", "terrain": "terrain",
    "precipitation": "precipitation", "temperature": "temperature",
    "vegetation": "vegetation", "moisture": "moisture", "other": "other",
}

# Fixes ax.set_ylabel(col) currently printing raw column names.
METRIC_LABELS = {
    "roc_auc":     "ROC-AUC",
    "pr_auc":      "PR-AUC",
    "pr_lift":     "PR-AUC lift over prevalence baseline",
    "pr_baseline": "prevalence baseline",
    "bss":         "Brier Skill Score",
    "brier":       "Brier score",
    "prevalence":  "prevalence",
    "n":           "n observations",
    "pct_of_total": "% of total |SHAP|",
    "mean_abs_shap": "mean |SHAP|",
}


# ===========================================================================
# THREE DISJOINT PALETTES
#
# The previous clash: #2166ac meant XGBoost in comparison_figures.py, the
# spatiotemporal SCHEME in compare_models.py, and precipitation in
# BLOCK_COLORS. #d7301f meant MaxEnt-TG, the spatial scheme, and temperature.
#
# Resolution:
#   models  -> saturated qualitative hues (unchanged; most widely used)
#   schemes -> greyscale ramp + one orange for the collapsing scheme
#   blocks  -> pastels, so that in a figure containing BOTH model bars and
#              block stacks (e.g. shap_blocks_cross_model.png) saturation
#              alone distinguishes the two encodings
# No hex string appears in more than one dictionary.
# ===========================================================================
MODEL_COLOURS = {
    "xgboost":            "#2166ac",   # blue
    "random_forest":      "#238b45",   # green
    "maxent_targetgroup": "#d7301f",   # red
    "maxent_vanilla":     "#6a51a3",   # purple
}

# The three schemes that behave alike form a periwinkle ramp; spatial is the
# warm outlier. The collapse is then a single orange bar in every figure.
SCHEME_COLOURS = {
    "spatiotemporal":         "#a3b3fd",
    "temporal":               "#7b8fe0",
    "spatiotemporal_3blocks": "#c7d0fe",
    "spatial":                "#faa665",   # the odd one out, deliberately
    "forward_chaining":       "#5aa77a",
}

# WHY HATCHING: #a3b3fd and #faa665 have almost the same relative luminance
# (0.471 vs 0.485), so in greyscale print or a photocopy they are the SAME grey
# -- and this pair carries the headline result. Hatching the spatial series
# keeps the distinction in monochrome. Delete SCHEME_HATCH usage if the thesis
# will only ever be read in colour.
SCHEME_HATCH = {
    "spatiotemporal":         "",
    "temporal":               "",
    "spatiotemporal_3blocks": "",
    "spatial":                "//",
    "forward_chaining":       "",
}

BLOCK_COLOURS = {
    "land_cover":    "#ffe08a",
    "seasonality":   "#cab2d6",
    "terrain":       "#d9d9d9",
    "precipitation": "#a6cee3",
    "temperature":   "#fb9a99",
    "vegetation":    "#b2df8a",
    "moisture":      "#99d8c9",
    "other":         "#f0f0f0",
}

GREY = "#808080"          # reference lines, perfect-calibration diagonal
BAR_EDGE = "black"
BAR_EDGE_LW = 0.5


# ===========================================================================
# NUMERIC CONVENTIONS
# ===========================================================================
# Brier is 4 dp, not 3: values cluster around 0.11-0.25 and the calibrated /
# raw distinction turns on the fourth place (0.1126 vs 0.1147). Everything
# else is 3 dp. This is the only exception; do not add more.
DECIMALS = {
    "roc_auc": 3, "pr_auc": 3, "pr_lift": 3, "pr_baseline": 3,
    "bss": 3, "brier": 4, "prevalence": 3,
    "pct_of_total": 1, "mean_abs_shap": 3,
}
DEFAULT_DECIMALS = 3

# No-skill value per metric. None = no meaningful reference line.
REFERENCE_LINE = {
    "roc_auc": 0.5, "pr_lift": 0.0, "bss": 0.0,
    "pr_auc": None, "brier": None, "pct_of_total": None,
}

# Sensible axis limits so panels are comparable between figures.
YLIM = {
    "roc_auc": (0.30, 1.00),
    "pct_of_total": (0, 100),
}


def fmt(x, metric: str = None, na: str = "--") -> str:
    """Format a metric value at its canonical decimal places.

    Integers (n, counts) get thousands separators; everything else uses
    DECIMALS[metric], defaulting to 3.
    """
    if x is None:
        return na
    try:
        if isinstance(x, (int, np.integer)) or metric in ("n", "n_pres", "n_abs",
                                                          "n_cells", "n_groups",
                                                          "n_features"):
            return f"{int(x):,}"
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return na
    return f"{x:.{DECIMALS.get(metric, DEFAULT_DECIMALS)}f}"


def ci(lo, hi, metric: str = None) -> str:
    """Bracketed confidence interval at the metric's decimal places."""
    return f"[{fmt(lo, metric)}, {fmt(hi, metric)}]"


# ===========================================================================
# HELPERS
# ===========================================================================
def models_in(values) -> list:
    """Canonical-order subset of MODEL_ORDER actually present in `values`."""
    present = set(map(str, values))
    ordered = [m for m in MODEL_ORDER if m in present]
    extra = sorted(present - set(MODEL_ORDER))          # never silently drop
    return ordered + extra


def schemes_in(values) -> list:
    """Canonical-order subset of SCHEME_ORDER actually present in `values`."""
    present = set(map(str, values))
    ordered = [s for s in SCHEME_ORDER if s in present]
    return ordered + sorted(present - set(SCHEME_ORDER))


def blocks_in(values) -> list:
    present = set(map(str, values))
    ordered = [b for b in BLOCK_ORDER if b in present]
    return ordered + sorted(present - set(BLOCK_ORDER))


def wrap(label: str) -> str:
    """Break a display name before its parenthetical, for tick labels."""
    return label.replace(" (", "\n(")


def model_label(model: str, wrapped: bool = False) -> str:
    lab = PRETTY.get(model, model)
    return wrap(lab) if wrapped else lab


def scheme_label(scheme: str) -> str:
    return SCHEME_PRETTY.get(scheme, scheme.replace("_", " "))


def block_label(block: str) -> str:
    return BLOCK_PRETTY.get(block, block.replace("_", " "))


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def add_reference_line(ax, metric: str, **kw):
    """Draw the no-skill line for `metric`, if one is defined."""
    ref = REFERENCE_LINE.get(metric)
    if ref is None:
        return None
    opts = dict(color=GREY, ls="--", lw=1.0, zorder=0)
    opts.update(kw)
    return ax.axhline(ref, **opts)


def apply_ylim(ax, metric: str):
    if metric in YLIM:
        ax.set_ylim(*YLIM[metric])


def annotate_bars(ax, bars, values, metric: str = None, fontsize: int = 7,
                  offset: float = 0.012):
    """Print each bar's value above (or below, if negative) the bar."""
    span = np.diff(ax.get_ylim())[0] or 1.0
    for b, v in zip(bars, values):
        if v is None or not np.isfinite(v):
            continue
        up = v >= 0
        ax.text(b.get_x() + b.get_width() / 2,
                v + (offset * span if up else -offset * span),
                fmt(v, metric), ha="center",
                va="bottom" if up else "top",
                fontsize=fontsize, clip_on=False)


def annotate_n(ax, positions, ns, y=None, fontsize: int = 7):
    """Print the sample size under each x position.

    Every figure reporting a metric should carry n: several of the strongest
    numbers in this project rest on n = 20-47.
    """
    lo = ax.get_ylim()[0] if y is None else y
    for x, n in zip(positions, ns):
        if n is None:
            continue
        ax.text(x, lo, f"n={fmt(n, 'n')}", ha="center", va="bottom",
                fontsize=fontsize, color=GREY)


def n_label(**counts) -> str:
    """Compact 'n=2,454 · 113 cells' style annotation for captions/titles."""
    return " \u00b7 ".join(f"{k}={fmt(v, 'n')}" for k, v in counts.items()
                           if v is not None)


def save(fig, path, dpi: int = FIG_DPI, close: bool = True):
    """Single save path so DPI and bbox are never set per-script."""
    fig.savefig(path, dpi=dpi)
    if close:
        plt.close(fig)
    print(f"[fig] wrote {path}", flush=True)
    return path


# ===========================================================================
# SELF-CHECK  —  `python report_style.py` verifies the palettes are disjoint
# and that every ordered key has a display name and a colour.
# ===========================================================================
def _self_check():
    ok = True
    pools = {"MODEL_COLOURS": MODEL_COLOURS,
             "SCHEME_COLOURS": SCHEME_COLOURS,
             "BLOCK_COLOURS": BLOCK_COLOURS}
    seen = {}
    for name, pool in pools.items():
        for key, hexval in pool.items():
            h = hexval.lower()
            if h in seen:
                print(f"COLLISION {h}: {seen[h]} vs {name}.{key}")
                ok = False
            seen[h] = f"{name}.{key}"
    for m in MODEL_ORDER:
        if m not in PRETTY or m not in MODEL_COLOURS:
            print(f"MISSING label/colour for model {m}"); ok = False
    for s in SCHEME_ORDER:
        if s not in SCHEME_PRETTY or s not in SCHEME_COLOURS:
            print(f"MISSING label/colour for scheme {s}"); ok = False
    for b in BLOCK_ORDER:
        if b not in BLOCK_PRETTY or b not in BLOCK_COLOURS:
            print(f"MISSING label/colour for block {b}"); ok = False
    print("report_style self-check:", "PASS" if ok else "FAIL", flush=True)
    return ok


if __name__ == "__main__":
    _self_check()