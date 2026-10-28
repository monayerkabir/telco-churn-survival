"""Time-to-churn models: Kaplan-Meier baseline, Cox PH (lifelines), random survival forest.

Framing: ``tenure`` (months) is the time axis. Customers who churned contribute an event
at their tenure; active customers are right-censored at their current tenure. Customers
with ``tenure == 0`` carry no follow-up time and are dropped from survival fitting.

All models share one interface: ``fit(X, duration, event)`` and
``predict_survival(X, times) -> (n_customers, n_times)`` so they can be evaluated and
compared with the same code. Covariates exclude tenure and everything derived from it
(see :data:`churn.features.TENURE_DERIVED_COLS`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from sklearn.base import BaseEstimator
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from churn.config import CONFIG, DURATION_COL, MAX_TENURE, TARGET_COL, SurvivalConfig
from churn.evaluate import concordance_index, integrated_brier
from churn.features import build_preprocessor

TIME_GRID = np.arange(0, MAX_TENURE + 1)  # monthly grid 0..72; tenure is in whole months
# Brier scores need times strictly inside the test follow-up window.
IBS_TIMES = np.arange(1, MAX_TENURE)

_EVENT = "_event"
_DURATION = "_duration"


def survival_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for survival fitting (positive follow-up time)."""
    return df[df[DURATION_COL] > 0].reset_index(drop=True)


def survival_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[DURATION_COL].to_numpy(dtype=float), df[TARGET_COL].to_numpy(dtype=bool)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class KaplanMeierModel(BaseEstimator):
    """Covariate-free baseline: every customer gets the population survival curve."""

    def fit(self, X: pd.DataFrame, duration, event):
        self.km_ = KaplanMeierFitter().fit(duration, event)
        return self

    def predict_survival(self, X: pd.DataFrame, times=TIME_GRID) -> np.ndarray:
        curve = self.km_.survival_function_at_times(times).to_numpy()
        return np.tile(curve, (len(X), 1))


class CoxPHModel(BaseEstimator):
    """Cox proportional hazards with its own preprocessing.

    ``strata`` columns are not given coefficients; instead each stratum gets its own
    baseline hazard. This is the standard remedy when a covariate violates the
    proportional-hazards assumption.
    """

    def __init__(self, penalizer: float = 0.01, strata: tuple[str, ...] = ()):
        self.penalizer = penalizer
        self.strata = strata

    def fit(self, X: pd.DataFrame, duration, event):
        self.preprocessor_ = build_preprocessor(
            "linear", include_tenure=False, drop="first", exclude=tuple(self.strata)
        )
        design = self._design(X, fit=True)
        design[_DURATION] = np.asarray(duration, dtype=float)
        design[_EVENT] = np.asarray(event, dtype=int)
        self.cph_ = CoxPHFitter(penalizer=self.penalizer).fit(
            design, duration_col=_DURATION, event_col=_EVENT, strata=list(self.strata) or None
        )
        self.training_design_ = design
        return self

    def _design(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        Z = self.preprocessor_.fit_transform(X) if fit else self.preprocessor_.transform(X)
        Z = Z.reset_index(drop=True)
        for col in self.strata:
            Z[col] = X[col].to_numpy()
        return Z

    def predict_survival(self, X: pd.DataFrame, times=TIME_GRID) -> np.ndarray:
        design = self._design(X)
        sf = self.cph_.predict_survival_function(design, times=times)
        # Stratified predictions come back grouped by stratum: restore the input order.
        return sf.reindex(columns=design.index).to_numpy().T

    @property
    def summary(self) -> pd.DataFrame:
        """Coefficients, hazard ratios and confidence intervals."""
        return self.cph_.summary

    def proportional_hazards_test(self) -> pd.DataFrame:
        """Schoenfeld-residual test (rank time transform) on the training data."""
        result = proportional_hazard_test(self.cph_, self.training_design_, time_transform="rank")
        return result.summary.sort_values("test_statistic", ascending=False)


class RandomSurvivalForestModel(BaseEstimator):
    """scikit-survival random survival forest on the tree preprocessor's features."""

    def __init__(
        self,
        n_estimators: int = 300,
        min_samples_leaf: int = 15,
        max_features: str | float = "sqrt",
        random_state: int = CONFIG.seed,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, duration, event):
        self.preprocessor_ = build_preprocessor("tree", include_tenure=False)
        Z = self.preprocessor_.fit_transform(X)
        self.rsf_ = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(Z, Surv.from_arrays(np.asarray(event, bool), np.asarray(duration, float)))
        return self

    def predict_survival(self, X: pd.DataFrame, times=TIME_GRID) -> np.ndarray:
        Z = self.preprocessor_.transform(X)
        surv = self.rsf_.predict_survival_function(Z, return_array=True)
        # Step function: S(t) is the value at the last event time <= t (1.0 before any).
        idx = np.searchsorted(self.rsf_.unique_times_, np.asarray(times), side="right") - 1
        padded = np.hstack([np.ones((surv.shape[0], 1)), surv])
        return padded[:, idx + 1]


def build_survival_models(cfg: SurvivalConfig = CONFIG.survival, seed: int = CONFIG.seed):
    """The model line-up compared in the report, keyed by display name."""
    return {
        "Kaplan-Meier (no covariates)": KaplanMeierModel(),
        "Cox PH": CoxPHModel(penalizer=cfg.cox_penalizer),
        "Cox PH (stratified by contract)": CoxPHModel(
            penalizer=cfg.cox_penalizer, strata=cfg.cox_strata
        ),
        "Random survival forest": RandomSurvivalForestModel(
            n_estimators=cfg.rsf_n_estimators,
            min_samples_leaf=cfg.rsf_min_samples_leaf,
            max_features=cfg.rsf_max_features,
            random_state=seed,
        ),
    }


# --------------------------------------------------------------------------------------
# Derived quantities from survival curves on TIME_GRID
# --------------------------------------------------------------------------------------
def restricted_mean_survival(surv: np.ndarray, tau: int = MAX_TENURE) -> np.ndarray:
    """Area under S(t) on [0, tau] for curves on :data:`TIME_GRID` (step functions)."""
    return surv[:, :tau].sum(axis=1)


def risk_score(surv: np.ndarray) -> np.ndarray:
    """Model-agnostic risk: minus the restricted mean survival time.

    For an unstratified Cox model this ranks customers exactly like the partial hazard,
    and unlike the partial hazard it stays valid for stratified Cox and forests.
    """
    return -restricted_mean_survival(surv)


def conditional_survival(surv: np.ndarray, current_tenure) -> np.ndarray:
    """S(t | T > t0) = S(t) / S(t0) for t >= t0; NaN for t < t0."""
    t0 = np.asarray(current_tenure, dtype=int)
    s_t0 = surv[np.arange(len(t0)), t0]
    cond = surv / np.clip(s_t0, 1e-12, None)[:, None]
    cond[TIME_GRID[None, :] < t0[:, None]] = np.nan
    return np.clip(cond, 0.0, 1.0)


def expected_remaining_tenure(
    surv: np.ndarray, current_tenure, tau: int = MAX_TENURE
) -> np.ndarray:
    """Restricted mean residual tenure: E[min(T, tau) - t0 | T > t0], in months.

    Restricted to the 72-month observation window because the curves cannot be
    extrapolated beyond the longest observed tenure. The window shrinks as t0 grows, so
    long-tenure customers get small values by construction; use
    :func:`expected_months_within` to compare customers on a common horizon.
    """
    cond = conditional_survival(surv, current_tenure)
    return np.nansum(cond[:, :tau], axis=1)


def expected_months_within(surv: np.ndarray, current_tenure, horizon: int) -> np.ndarray:
    """E[min(T - t0, horizon) | T > t0]: expected active months over the next ``horizon``.

    Unlike :func:`expected_remaining_tenure`, the window has the same length for every
    customer, so values are comparable across tenures (and times MonthlyCharges give
    expected revenue). NaN if t0 + horizon > 72.
    """
    t0 = np.asarray(current_tenure, dtype=int)
    cond = conditional_survival(surv, t0)
    window = (TIME_GRID[None, :] >= t0[:, None]) & (TIME_GRID[None, :] < (t0 + horizon)[:, None])
    out = np.where(window, cond, 0.0).sum(axis=1)
    return np.where(t0 + horizon <= MAX_TENURE, out, np.nan)


def churn_probability_within(surv: np.ndarray, current_tenure, horizon: int) -> np.ndarray:
    """P(churn within ``horizon`` months | active at t0). NaN if t0 + horizon > 72."""
    t0 = np.asarray(current_tenure, dtype=int)
    end = t0 + horizon
    valid = end <= MAX_TENURE
    cond = conditional_survival(surv, t0)
    out = np.full(len(t0), np.nan)
    rows = np.flatnonzero(valid)
    out[rows] = 1.0 - cond[rows, end[rows]]
    return out


# --------------------------------------------------------------------------------------
# Evaluation and tests
# --------------------------------------------------------------------------------------
def evaluate_survival_model(
    model, train_df: pd.DataFrame, test_df: pd.DataFrame, times=IBS_TIMES
) -> dict[str, float]:
    """Harrell's C-index and integrated Brier score on the test set."""
    d_train, e_train = survival_arrays(train_df)
    d_test, e_test = survival_arrays(test_df)
    surv = model.predict_survival(test_df, TIME_GRID)
    return {
        "c_index": concordance_index(d_test, e_test, risk_score(surv)),
        "integrated_brier_score": integrated_brier(
            d_train, e_train, d_test, e_test, surv[:, np.asarray(times)], np.asarray(times)
        ),
    }


def logrank_by_group(df: pd.DataFrame, group_col: str):
    """Multivariate log-rank test of equal survival across the levels of ``group_col``."""
    return multivariate_logrank_test(df[DURATION_COL], df[group_col], df[TARGET_COL])
