"""Business layer: revenue at risk and profit-maximising retention targeting.

Profit model (assumptions live in :class:`churn.config.BusinessConfig`): every targeted
customer costs ``offer_cost``; a targeted customer who would otherwise churn is retained
with probability ``offer_success_rate``, saving ``MonthlyCharges * horizon_months``.

    profit(targeted set) = sum_i [ churn_i * success_rate * value_i - offer_cost ]

``churn_i`` can be the observed 0/1 outcome (evaluating a policy on held-out data) or a
calibrated probability (forecasting the profit of a campaign before outcomes are known).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config import CONFIG, BusinessConfig

DEFAULT_THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)


def customer_value(monthly_charges, horizon_months: int) -> np.ndarray:
    """Revenue a customer generates over the horizon if they stay."""
    return np.asarray(monthly_charges, dtype=float) * horizon_months


def revenue_at_risk(p_churn, monthly_charges, horizon_months: int) -> np.ndarray:
    """Expected revenue lost to churn: ``P(churn) * MonthlyCharges * horizon``."""
    p = np.asarray(p_churn, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_churn must be probabilities in [0, 1]")
    return p * customer_value(monthly_charges, horizon_months)


def expected_profit(
    churn,
    p_churn,
    monthly_charges,
    threshold: float,
    cfg: BusinessConfig = CONFIG.business,
) -> float:
    """Expected campaign profit when targeting every customer with ``p_churn >= threshold``.

    Args:
        churn: observed outcomes (0/1) or calibrated churn probabilities.
        p_churn: model scores used to decide who is targeted.
        monthly_charges: per-customer monthly revenue.
        threshold: target customers whose score is at least this value.
    """
    targeted = np.asarray(p_churn, dtype=float) >= threshold
    return policy_profit(churn, targeted, monthly_charges, cfg)


def policy_profit(churn, targeted, monthly_charges, cfg: BusinessConfig = CONFIG.business) -> float:
    """Expected profit of offering the incentive to an arbitrary set of customers."""
    churn = np.asarray(churn, dtype=float)
    value = customer_value(monthly_charges, cfg.horizon_months)
    gain = churn * cfg.offer_success_rate * value - cfg.offer_cost
    return float(gain[np.asarray(targeted, dtype=bool)].sum())


def profit_curve(
    churn,
    p_churn,
    monthly_charges,
    thresholds=DEFAULT_THRESHOLDS,
    cfg: BusinessConfig = CONFIG.business,
) -> pd.DataFrame:
    """Profit and campaign size for each candidate threshold."""
    churn = np.asarray(churn, dtype=float)
    p = np.asarray(p_churn, dtype=float)
    value = customer_value(monthly_charges, cfg.horizon_months)
    saved = churn * cfg.offer_success_rate * value

    rows = []
    for t in thresholds:
        targeted = p >= t
        n = int(targeted.sum())
        retained = float(saved[targeted].sum())
        cost = n * cfg.offer_cost
        rows.append(
            {
                "threshold": float(t),
                "n_targeted": n,
                "share_targeted": n / len(p) if len(p) else 0.0,
                "churners_reached": float(churn[targeted].sum()),
                "retained_revenue": retained,
                "offer_cost": cost,
                "profit": retained - cost,
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(curve: pd.DataFrame) -> pd.Series:
    """Row of ``curve`` with maximum profit; ties go to the higher (cheaper) threshold."""
    best = curve["profit"].max()
    return curve[curve["profit"] == best].sort_values("threshold").iloc[-1]


def breakeven_probability(monthly_charges, cfg: BusinessConfig = CONFIG.business) -> np.ndarray:
    """Churn probability above which targeting a given customer has positive expected value.

    From ``p * success_rate * value - cost > 0``. A single global threshold approximates
    this per-customer rule; the two coincide only if every customer has the same value.
    """
    value = customer_value(monthly_charges, cfg.horizon_months)
    return cfg.offer_cost / (cfg.offer_success_rate * value)


def value_aware_targets(p_churn, monthly_charges, cfg: BusinessConfig = CONFIG.business):
    """Target exactly the customers whose own expected profit is positive."""
    return np.asarray(p_churn, dtype=float) > breakeven_probability(monthly_charges, cfg)
