"""Figure helpers shared by the notebooks, with one consistent, accessible style.

Colour roles: categorical series use the first three validated palette slots in a fixed
order (blue, orange, aqua); context/baselines are gray; values and labels use ink
colours, never series colours.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from lifelines import KaplanMeierFitter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from churn.config import DURATION_COL, FIGURES_DIR, TARGET_COL

# Palette ------------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
RED = "#e34948"  # diverging pole (pushes towards churn)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#b5b4ad"  # de-emphasised series / baselines

DIVERGING_CMAP = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, "#a9a8a2", RED])

usd = FuncFormatter(lambda x, _: f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "axes.prop_cycle": cycler(color=SERIES),
        }
    )


def save_figure(fig: plt.Figure, name: str, figures_dir: Path = FIGURES_DIR) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path)
    return path


def set_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title with an optional secondary-ink subtitle underneath."""
    ax.set_title(title, pad=26 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def format_p(p: float) -> str:
    """p-values that underflow to 0.0 are reported as a bound, not as 'p = 0'."""
    return "p < 1e-300" if p == 0 else f"p = {p:.1e}" if p < 1e-3 else f"p = {p:.3f}"


# --------------------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------------------
def plot_churn_rate_by(
    df: pd.DataFrame,
    col: str,
    order: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    xmax: float = 0.75,
) -> plt.Axes:
    """Horizontal bars of churn rate per level, labelled with rate and group size."""
    ax = ax or plt.gca()
    stats = df.groupby(col, observed=True)[TARGET_COL].agg(rate="mean", n="size")
    stats = stats.reindex(order) if order is not None else stats.sort_values("rate")
    y = np.arange(len(stats))
    ax.barh(y, stats["rate"], height=0.5, color=BLUE)
    overall = df[TARGET_COL].mean()
    ax.axvline(overall, color=MUTED, linewidth=1, zorder=1)
    ax.text(overall, len(stats) - 0.45, f" overall {overall:.1%}", color=MUTED, fontsize=8)
    for yi, (rate, n) in zip(y, stats[["rate", "n"]].itertuples(index=False), strict=True):
        # A surface-coloured box keeps the reference line from running through the label.
        ax.text(
            rate + 0.01,
            yi,
            f"{rate:.0%}  (n={n:,})",
            va="center",
            color=INK_2,
            fontsize=9,
            zorder=3,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1},
        )
    ax.set_yticks(y, stats.index)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return ax


def plot_distribution_by_churn(
    df: pd.DataFrame, col: str, bins: int | Sequence[float] = 24, ax: plt.Axes | None = None
) -> plt.Axes:
    """Overlaid histograms of ``col``, each normalised to its own group's size."""
    ax = ax or plt.gca()
    for label, value, color in [("Stayed", 0, BLUE), ("Churned", 1, ORANGE)]:
        data = df.loc[df[TARGET_COL] == value, col]
        weights = np.full(len(data), 1 / len(data))
        ax.hist(data, bins=bins, weights=weights, histtype="stepfilled", alpha=0.10, color=color)
        ax.hist(
            data, bins=bins, weights=weights, histtype="step", linewidth=2, color=color, label=label
        )
    ax.set_ylabel("Share of group")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="upper right")
    return ax


def plot_km_by_group(
    df: pd.DataFrame,
    group_col: str,
    order: Sequence[str],
    ax: plt.Axes | None = None,
    show_ci: bool = True,
) -> plt.Axes:
    """Kaplan-Meier retention curves per group, direct-labelled at the right edge."""
    ax = ax or plt.gca()
    for level, color in zip(order, SERIES, strict=False):
        sub = df[df[group_col] == level]
        km = KaplanMeierFitter(label=level).fit(sub[DURATION_COL], sub[TARGET_COL])
        sf = km.survival_function_[level]
        ax.step(sf.index, sf.values, where="post", color=color, label=level)
        if show_ci:
            ci = km.confidence_interval_survival_function_
            ax.fill_between(
                ci.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post", color=color, alpha=0.12, lw=0
            )
        ax.text(
            sf.index[-1] + 1,
            sf.values[-1],
            f"{level}  {sf.values[-1]:.0%}",
            va="center",
            color=INK_2,
            fontsize=9,
        )
    ax.set_xlim(0, df[DURATION_COL].max() * 1.22)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Share of customers still active")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="lower left")
    return ax


def plot_log_minus_log(
    df: pd.DataFrame, group_col: str, order: Sequence[str], ax: plt.Axes | None = None
) -> plt.Axes:
    """log(-log S(t)) vs log t: parallel curves are consistent with proportional hazards."""
    ax = ax or plt.gca()
    for level, color in zip(order, SERIES, strict=False):
        sub = df[df[group_col] == level]
        sf = KaplanMeierFitter().fit(sub[DURATION_COL], sub[TARGET_COL]).survival_function_
        sf = sf[(sf.index > 0) & (sf.iloc[:, 0] < 1) & (sf.iloc[:, 0] > 0)]
        ax.step(
            np.log(sf.index), np.log(-np.log(sf.iloc[:, 0])), where="post", color=color, label=level
        )
    ax.set_xlabel("log(tenure)")
    ax.set_ylabel("log(-log S(t))")
    ax.legend(loc="lower right")
    return ax


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------
def plot_optuna_history(trials: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    ax.scatter(
        trials["number"],
        trials["value"],
        s=24,
        color=CONTEXT,
        label="Trial",
        zorder=2,
        edgecolor=SURFACE,
        linewidth=1,
    )
    ax.plot(trials["number"], trials["value"].cummax(), color=BLUE, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean CV ROC-AUC")
    ax.legend(loc="lower right")
    return ax


def plot_roc_pr(y_true, p, axes: Sequence[plt.Axes]) -> None:
    """ROC and precision-recall curves with their no-skill baselines."""
    ax_roc, ax_pr = axes
    fpr, tpr, _ = roc_curve(y_true, p)
    ax_roc.plot([0, 1], [0, 1], color=CONTEXT, linewidth=1)
    ax_roc.plot(fpr, tpr, color=BLUE)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")

    precision, recall, _ = precision_recall_curve(y_true, p)
    base = float(np.mean(y_true))
    ax_pr.axhline(base, color=CONTEXT, linewidth=1)
    ax_pr.text(1, base, f"no skill ({base:.0%})", ha="right", va="bottom", color=MUTED, fontsize=8)
    ax_pr.plot(recall, precision, color=BLUE)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-recall (AP = {average_precision_score(y_true, p):.3f})")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")


def plot_reliability(calibration: dict[str, pd.DataFrame], ax: plt.Axes | None = None) -> plt.Axes:
    """Reliability diagram: observed churn rate vs mean predicted probability per bin."""
    ax = ax or plt.gca()
    ax.plot([0, 1], [0, 1], color=CONTEXT, linewidth=1, label="Perfect calibration")
    for (label, table), color in zip(calibration.items(), SERIES, strict=False):
        ax.plot(
            table["mean_predicted"],
            table["observed_rate"],
            color=color,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="upper left")
    return ax


def plot_local_explanation(
    contributions: pd.Series,
    feature_values: pd.Series,
    ax: plt.Axes | None = None,
    top_k: int = 10,
) -> plt.Axes:
    """Top SHAP contributions (log-odds) for one customer as diverging bars.

    Red bars push the prediction towards churn, blue bars away from it. The remaining
    features are folded into a single gray "other features" bar.
    """
    ax = ax or plt.gca()
    order = contributions.abs().sort_values(ascending=False).index
    top, rest = order[:top_k], order[top_k:]
    values = contributions[top]
    labels = [f"{f} = {_fmt_value(feature_values[f])}" for f in top]
    colors = [RED if v > 0 else BLUE for v in values]
    if len(rest):
        values = pd.concat([values, pd.Series({"other": contributions[rest].sum()})])
        labels.append(f"{len(rest)} other features")
        colors.append(CONTEXT)
    y = np.arange(len(values))[::-1]
    ax.barh(y, values.to_numpy(), height=0.55, color=colors)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("SHAP contribution (log-odds of churn)")
    return ax


def _fmt_value(v) -> str:
    if isinstance(v, int | float | np.floating) and float(v).is_integer():
        return f"{int(v)}"
    if isinstance(v, int | float | np.floating):
        return f"{v:.1f}"
    return str(v)


# --------------------------------------------------------------------------------------
# Business
# --------------------------------------------------------------------------------------
def plot_profit_curves(
    curves: dict[str, pd.DataFrame], threshold: float, ax: plt.Axes | None = None
) -> plt.Axes:
    """Expected profit per customer in the base vs targeting threshold.

    Profit is normalised per customer so curves from data sets of different size share
    one axis.
    """
    ax = ax or plt.gca()
    for (label, curve), color in zip(curves.items(), SERIES, strict=False):
        n = curve["n_targeted"].iloc[0]  # threshold 0 targets everyone
        ax.plot(curve["threshold"], curve["profit"] / n, color=color, label=label)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axvline(threshold, color=INK_2, linewidth=1)
    # Label sits low on the line, where the curves (which peak near the threshold) are not.
    ymin, ymax = ax.get_ylim()
    ax.text(
        threshold + 0.01,
        ymin + 0.08 * (ymax - ymin),
        f"chosen threshold {threshold:.2f}",
        va="bottom",
        color=INK_2,
        fontsize=9,
    )
    ax.set_xlabel("Target customers with P(churn) at or above")
    ax.set_ylabel("Expected profit per customer")
    ax.yaxis.set_major_formatter(usd)
    ax.legend(loc="upper right")
    return ax


# --------------------------------------------------------------------------------------
# Survival
# --------------------------------------------------------------------------------------
def plot_hazard_ratios(summary: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Forest plot of Cox hazard ratios with 95% CIs (log scale)."""
    ax = ax or plt.gca()
    s = summary.sort_values("exp(coef)")
    y = np.arange(len(s))
    lo, hi = s["exp(coef) lower 95%"], s["exp(coef) upper 95%"]
    ax.hlines(y, lo, hi, color=BLUE, linewidth=2)
    ax.scatter(s["exp(coef)"], y, color=BLUE, s=36, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(1, color=MUTED, linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, s.index)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hazard ratio (log scale): < 1 retains, > 1 accelerates churn")
    return ax


def plot_brier_over_time(
    times: np.ndarray, scores: dict[str, np.ndarray], baseline: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """Time-dependent Brier score; ``baseline`` is drawn in gray as context."""
    ax = ax or plt.gca()
    ax.plot(times, scores[baseline], color=CONTEXT, label=baseline)
    others = [k for k in scores if k != baseline]
    for name, color in zip(others, SERIES, strict=False):
        ax.plot(times, scores[name], color=color, label=name)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.legend(loc="upper left")
    return ax


def plot_customer_survival(
    times: np.ndarray,
    curves: dict[str, np.ndarray],
    current_tenure: dict[str, int],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Conditional survival curves for a few customers, starting at their current tenure."""
    ax = ax or plt.gca()
    for (label, curve), color in zip(curves.items(), SERIES, strict=False):
        t0 = current_tenure[label]
        mask = times >= t0
        ax.step(times[mask], curve[mask], where="post", color=color, label=label)
        ax.scatter([t0], [1.0], color=color, s=40, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("P(still a customer | active today)")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="lower left")
    return ax
