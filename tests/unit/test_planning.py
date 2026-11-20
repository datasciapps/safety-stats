"""Tests for the weighted fixed-design planning helper."""

import math

import pytest

from inspect_safety_stats.stats.binomial import binomial_upper_bound
from inspect_safety_stats.stats.planning import (
    expected_upper_bound,
    fixed_design_detectable_failure_rates,
)


def test_equal_weight_rollout_geometry() -> None:
    rates = fixed_design_detectable_failure_rates(
        [20, 20, 20],
        [1 / 3, 1 / 3, 1 / 3],
        threshold=0.5,
        support_alpha=0.05,
        rejection_alpha=0.05,
        power=0.9,
    )
    assert rates.independence_unit == "rollout"
    assert math.isclose(rates.effective_sample_size, 60.0)
    assert rates.maximum_certifiable_failure_rate is not None
    assert rates.minimum_detectable_failure_rate is not None
    assert rates.maximum_certifiable_failure_rate < 0.5 < rates.minimum_detectable_failure_rate
    assert rates.support_gap is not None
    expected_support_gap = math.sqrt(0.5 * (1 / 60) * math.log(1 / 0.05)) + math.sqrt(
        0.5 * (1 / 60) * math.log(1 / 0.1)
    )
    assert math.isclose(rates.support_gap, expected_support_gap)


def test_prompt_cluster_geometry_ignores_rollout_counts() -> None:
    shared = dict(threshold=0.5, support_alpha=0.05, rejection_alpha=0.05, power=0.9)
    many = fixed_design_detectable_failure_rates(
        [20, 20, 20],
        [1 / 3, 1 / 3, 1 / 3],
        independence_unit="prompt_cluster",
        **shared,
    )
    single = fixed_design_detectable_failure_rates(
        [1, 1, 1],
        [1 / 3, 1 / 3, 1 / 3],
        independence_unit="prompt_cluster",
        **shared,
    )
    assert math.isclose(many.effective_sample_size, 3.0)
    assert math.isclose(single.effective_sample_size, 3.0)
    assert many.support_gap == single.support_gap
    assert many.rejection_gap == single.rejection_gap


def test_unattainable_support_side_is_none() -> None:
    rates = fixed_design_detectable_failure_rates(
        [5, 5, 5],
        [1 / 3, 1 / 3, 1 / 3],
        threshold=0.01,
        support_alpha=0.05,
        rejection_alpha=0.05,
        power=0.9,
    )
    assert rates.maximum_certifiable_failure_rate is None
    assert rates.support_gap is None
    assert rates.minimum_detectable_failure_rate is not None


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        fixed_design_detectable_failure_rates(
            [10, 10],
            [0.5, 0.4],
            threshold=0.1,
            support_alpha=0.05,
            rejection_alpha=0.05,
            power=0.9,
        )


def test_the_expected_bound_is_the_exact_bound_at_whole_counts() -> None:
    for trials, rate in [(1200, 0.03), (100, 0.05), (16, 0.0), (8, 0.25)]:
        failures = round(rate * trials)
        assert expected_upper_bound(draws=trials, rate=rate, alpha=0.05) == pytest.approx(
            binomial_upper_bound(failures, trials, alpha=0.05), rel=1e-12, abs=0
        )


def test_the_expected_bound_falls_as_the_draws_grow() -> None:
    """With rounded counts, 16 trials at 3% expected no failure and 20 expected one, so the
    smaller sample forecast the tighter bound."""

    for rate in (0.0, 0.01, 0.03, 0.2):
        bounds = [
            expected_upper_bound(draws=draws / 4, rate=rate, alpha=0.05) for draws in range(4, 4000)
        ]
        assert all(later < earlier for earlier, later in zip(bounds, bounds[1:], strict=False))
    assert expected_upper_bound(draws=16, rate=0.03, alpha=0.05) > expected_upper_bound(
        draws=20, rate=0.03, alpha=0.05
    )
    # Fewer than one draw counts as one.
    assert expected_upper_bound(draws=0.25, rate=0.03, alpha=0.05) == expected_upper_bound(
        draws=1, rate=0.03, alpha=0.05
    )
