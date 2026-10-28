import numpy as np
import pytest

from churn.business import (
    breakeven_probability,
    expected_profit,
    optimal_threshold,
    policy_profit,
    profit_curve,
    revenue_at_risk,
    value_aware_targets,
)
from churn.config import BusinessConfig

CFG = BusinessConfig(horizon_months=10, offer_cost=20.0, offer_success_rate=0.5)


def test_revenue_at_risk():
    rar = revenue_at_risk([0.5, 0.1, 0.0], [100.0, 50.0, 80.0], horizon_months=12)
    np.testing.assert_allclose(rar, [600.0, 60.0, 0.0])


def test_revenue_at_risk_rejects_non_probabilities():
    with pytest.raises(ValueError):
        revenue_at_risk([1.2], [100.0], horizon_months=12)


def test_expected_profit_hand_computed():
    churn = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.3, 0.1])
    charges = np.array([10.0, 10.0, 20.0, 10.0])
    # threshold 0.5 targets customers 0 and 1:
    #   customer 0 churns -> 0.5 * (10 * 10) - 20 = 30 ; customer 1 stays -> -20
    assert expected_profit(churn, p, charges, 0.5, CFG) == pytest.approx(10.0)
    # threshold 0.2 adds customer 2 (churns, value 200): 0.5 * 200 - 20 = 80
    assert expected_profit(churn, p, charges, 0.2, CFG) == pytest.approx(90.0)


def test_nobody_targeted_means_zero_profit():
    assert expected_profit([1, 1], [0.9, 0.95], [50.0, 50.0], 1.01, CFG) == 0.0


def test_profit_with_probabilities_is_the_expectation():
    """Using calibrated probabilities as outcomes gives the ex-ante expected profit."""
    p = np.array([0.4])
    value = 30.0 * CFG.horizon_months
    expected = 0.4 * CFG.offer_success_rate * value - CFG.offer_cost
    assert expected_profit(p, p, [30.0], 0.0, CFG) == pytest.approx(expected)


def test_profit_curve_matches_expected_profit():
    rng = np.random.default_rng(0)
    churn = rng.integers(0, 2, 200)
    p = rng.random(200)
    charges = rng.uniform(20, 100, 200)

    curve = profit_curve(churn, p, charges, cfg=CFG)

    for t in [0.0, 0.25, 0.5, 0.9]:
        row = curve.loc[curve["threshold"] == t].iloc[0]
        assert row["profit"] == pytest.approx(expected_profit(churn, p, charges, t, CFG))
        assert row["n_targeted"] == (p >= t).sum()
    assert curve["n_targeted"].is_monotonic_decreasing


def test_optimal_threshold_breaks_ties_towards_fewer_offers():
    churn = np.array([1, 0])
    p = np.array([0.9, 0.2])
    charges = np.array([10.0, 0.0])
    curve = profit_curve(churn, p, charges, thresholds=[0.1, 0.5, 0.8, 0.95], cfg=CFG)
    best = optimal_threshold(curve)
    # 0.5 and 0.8 both target only customer 0 (profit 30); pick the higher threshold.
    assert best["threshold"] == 0.8
    assert best["profit"] == pytest.approx(30.0)


def test_breakeven_probability_has_zero_expected_profit():
    charges = np.array([40.0])
    p_star = breakeven_probability(charges, CFG)
    assert expected_profit(p_star, p_star, charges, 0.0, CFG) == pytest.approx(0.0)


def test_value_aware_targets_only_profitable_customers():
    p = np.array([0.2, 0.2, 0.05])
    charges = np.array([100.0, 10.0, 100.0])
    # breakeven p* = 20 / (0.5 * charges * 10): 0.04 for $100 customers, 0.4 for $10 ones
    targeted = value_aware_targets(p, charges, CFG)
    np.testing.assert_array_equal(targeted, [True, False, True])
    # Ex-ante, the value-aware rule is never worse than any global threshold.
    best_threshold = max(expected_profit(p, p, charges, t, CFG) for t in [0, 0.05, 0.1, 0.2, 1.1])
    assert policy_profit(p, targeted, charges, CFG) >= best_threshold
