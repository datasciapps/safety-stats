"""Detectable rates planned under the interval method analyze uses."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from inspect_safety_stats.stats.direct_mean import (
    DirectMeanBounds,
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
)
from inspect_safety_stats.stats.planning import (
    DirectMeanMethod,
    detectable_failure_rates,
    fixed_design_detectable_failure_rates,
)

ANALYSIS: dict[DirectMeanMethod, Callable[..., DirectMeanBounds]] = {
    "weighted_hoeffding": weighted_bounded_hoeffding_mean_bounds,
    "weighted_empirical_bernstein": weighted_bounded_empirical_bernstein_mean_bounds,
    "weighted_kl_chernoff": weighted_bounded_kl_chernoff_mean_bounds,
}
METHODS = list(ANALYSIS)
ALPHA = 0.0125


def _analysis(
    method: DirectMeanMethod, values: np.ndarray, weights: np.ndarray
) -> DirectMeanBounds:
    return ANALYSIS[method](
        values,
        values,
        weights,
        alpha_lower=ALPHA,
        alpha_upper=ALPHA,
        independence_profile="test",
    )


def _weights(kind: str) -> np.ndarray:
    if kind == "equal":
        return np.full(800, 1 / 800)
    weights = np.random.default_rng(7).uniform(0.5, 1.5, size=800)
    return weights / weights.sum()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["equal", "unequal"])
def test_planned_rates_reach_the_target_power_under_the_analysis_interval(
    method: DirectMeanMethod, kind: str
) -> None:
    weights = _weights(kind)
    rates = detectable_failure_rates(
        weights,
        method=method,
        alpha=ALPHA,
        power=0.8,
        acceptable_upper=0.08,
        unacceptable_lower=0.08,
    )
    assert rates.maximum_certifiable_failure_rate is not None
    assert rates.minimum_detectable_failure_rate is not None
    rng = np.random.default_rng(11)
    for rate, decided in (
        (rates.maximum_certifiable_failure_rate, lambda bounds: bounds.upper <= 0.08),
        (rates.minimum_detectable_failure_rate, lambda bounds: bounds.lower >= 0.08),
    ):
        # Heterogeneous block rates with the planned weighted mean: the guarantee does not
        # assume a common rate.
        spread = rng.uniform(0.5, 1.5, size=len(weights))
        block_rates = np.clip(rate * spread / float(np.dot(weights, spread)), 0.0, 1.0)
        hits = sum(
            decided(_analysis(method, (rng.random(len(weights)) < block_rates) * 1.0, weights))
            for _ in range(300)
        )
        assert hits / 300 >= 0.8


@pytest.mark.parametrize("method", METHODS)
def test_certification_floor_is_the_analysis_upper_bound_on_all_safe_blocks(
    method: DirectMeanMethod,
) -> None:
    weights = _weights("unequal")
    rates = detectable_failure_rates(weights, method=method, alpha=ALPHA, power=0.9)
    zeros = np.zeros(len(weights))
    ones = np.ones(len(weights))
    assert rates.certification_floor == pytest.approx(
        _analysis(method, zeros, weights).upper, rel=1e-12, abs=0
    )
    assert rates.rejection_ceiling == pytest.approx(
        _analysis(method, ones, weights).lower, rel=1e-12, abs=0
    )


@pytest.mark.parametrize("kind", ["equal", "unequal"])
def test_empirical_bernstein_planning_takes_the_widest_interval_of_an_estimate(kind: str) -> None:
    # The planner bounds the EB interval of any data by the one at the largest second moments
    # the estimate allows; observed data never gives a wider one.
    from inspect_safety_stats.stats.planning import _planned_endpoints

    weights = _weights(kind)
    rng = np.random.default_rng(3)
    for _ in range(50):
        values = (rng.random(len(weights)) < rng.uniform(0.0, 0.3)) * rng.uniform(
            0, 1, size=len(weights)
        )
        observed = _analysis("weighted_empirical_bernstein", values, weights)
        lower, upper = _planned_endpoints(
            "weighted_empirical_bernstein", weights, float(np.dot(weights, values)), ALPHA
        )
        assert upper >= observed.upper - 1e-12
        assert lower <= observed.lower + 1e-12


def test_kl_chernoff_planning_is_less_pessimistic_than_the_hoeffding_reference() -> None:
    weights = np.full(2000, 1 / 2000)
    reference = fixed_design_detectable_failure_rates(
        [1] * 2000,
        weights,
        threshold=0.03,
        support_alpha=ALPHA,
        rejection_alpha=ALPHA,
        power=0.9,
        independence_unit="prompt_cluster",
    )
    planned = detectable_failure_rates(
        weights,
        method="weighted_kl_chernoff",
        alpha=ALPHA,
        power=0.9,
        acceptable_upper=0.03,
        unacceptable_lower=0.03,
    )
    # At a rare-event threshold Hoeffding cannot certify any rate, while KL-Chernoff can.
    assert reference.maximum_certifiable_failure_rate is None
    assert planned.maximum_certifiable_failure_rate is not None
    assert 0 < planned.maximum_certifiable_failure_rate < 0.03
    assert reference.minimum_detectable_failure_rate is not None
    assert planned.minimum_detectable_failure_rate is not None
    assert planned.minimum_detectable_failure_rate < reference.minimum_detectable_failure_rate


@pytest.mark.parametrize("method", METHODS)
def test_a_threshold_no_outcome_can_cross_is_unattainable(method: DirectMeanMethod) -> None:
    weights = np.full(10, 0.1)
    rates = detectable_failure_rates(
        weights,
        method=method,
        alpha=ALPHA,
        power=0.9,
        acceptable_upper=0.01,
        unacceptable_lower=0.99,
    )
    assert rates.certification_floor > 0.01
    assert rates.maximum_certifiable_failure_rate is None
    assert rates.rejection_ceiling < 0.99
    assert rates.minimum_detectable_failure_rate is None


def test_planning_rejects_weights_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        detectable_failure_rates([0.5, 0.4], method="weighted_kl_chernoff", alpha=0.05, power=0.9)
