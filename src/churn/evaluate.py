"""Evaluation metrics for the classification and survival models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

from churn.config import CONFIG

# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------
CLASSIFICATION_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "brier": brier_score_loss,
    "log_loss": log_loss,
}

# sklearn scorer names for the same metrics (used by cross_validate). Losses are negated.
SKLEARN_SCORERS = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "neg_brier_score",
    "log_loss": "neg_log_loss",
}
LOWER_IS_BETTER = {"brier", "log_loss"}


def classification_metrics(y_true, p) -> dict[str, float]:
    y_true, p = np.asarray(y_true), np.asarray(p, dtype=float)
    return {name: float(fn(y_true, p)) for name, fn in CLASSIFICATION_METRICS.items()}


def bootstrap_ci(
    y_true,
    p,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = CONFIG.classification.n_bootstrap,
    alpha: float = 0.05,
    seed: int = CONFIG.seed,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a metric on a fixed test set."""
    y_true, p = np.asarray(y_true), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():  # resample has a single class
            continue
        stats.append(metric(y_true[idx], p[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def classification_metrics_with_ci(
    y_true, p, n_boot: int = CONFIG.classification.n_bootstrap, seed: int = CONFIG.seed
) -> dict[str, dict[str, float]]:
    point = classification_metrics(y_true, p)
    out = {}
    for name, fn in CLASSIFICATION_METRICS.items():
        lo, hi = bootstrap_ci(y_true, p, fn, n_boot=n_boot, seed=seed)
        out[name] = {"value": point[name], "ci_low": lo, "ci_high": hi}
    return out


def calibration_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    """Observed churn rate vs mean predicted probability in equal-count bins."""
    observed, predicted = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": predicted, "observed_rate": observed})


def summarise_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std per model of the per-fold metrics produced by ``churn.train``."""
    metrics = [m for m in CLASSIFICATION_METRICS if m in cv_results.columns]
    summary = cv_results.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    return summary


# --------------------------------------------------------------------------------------
# Survival
# --------------------------------------------------------------------------------------
def concordance_index(duration, event, risk) -> float:
    """Harrell's C: share of comparable pairs where the higher-risk customer churns first."""
    return float(
        concordance_index_censored(
            np.asarray(event, dtype=bool), np.asarray(duration, dtype=float), np.asarray(risk)
        )[0]
    )


def integrated_brier(
    duration_train,
    event_train,
    duration_test,
    event_test,
    surv_at_times: np.ndarray,
    times: np.ndarray,
) -> float:
    """Integrated Brier score (IPCW; censoring distribution estimated on the train set).

    ``surv_at_times`` has shape ``(n_test, len(times))``: predicted S(t) for each customer.
    Lower is better; a Kaplan-Meier curve with no covariates is the natural baseline.
    """
    y_train = Surv.from_arrays(np.asarray(event_train, bool), np.asarray(duration_train, float))
    y_test = Surv.from_arrays(np.asarray(event_test, bool), np.asarray(duration_test, float))
    return float(integrated_brier_score(y_train, y_test, surv_at_times, times))
