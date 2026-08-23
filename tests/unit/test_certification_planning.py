from __future__ import annotations

import math

import pytest
from scipy.stats import binom

from inspect_safety_stats.certification.planning import (
    certification_power,
    certification_sample_size,
    fixed_design_detectable_failure_rates,
    maximum_certifiable_failure_rate,
    maximum_certifying_failures,
    minimum_detectable_failure_rate,
    minimum_rejecting_failures,
    rejection_power,
    rejection_sample_size,
)


def test_zero_failure_cutoff_matches_exact_planning_boundary() -> None:
    assert maximum_certifying_failures(trials=28, threshold=0.1, alpha=0.05) == -1
    assert maximum_certifying_failures(trials=29, threshold=0.1, alpha=0.05) == 0


def test_certification_sample_size_achieves_requested_power_minimally() -> None:
    trials = certification_sample_size(
        true_rate=0.02,
        threshold=0.1,
        alpha=0.05,
        power=0.8,
    )
    achieved = certification_power(
        trials=trials,
        true_rate=0.02,
        threshold=0.1,
        alpha=0.05,
    )
    previous = certification_power(
        trials=trials - 1,
        true_rate=0.02,
        threshold=0.1,
        alpha=0.05,
    )
    assert achieved >= 0.8
    assert previous < 0.8


def test_certification_power_requires_safer_anticipated_rate() -> None:
    with pytest.raises(ValueError, match="true_rate"):
        certification_power(trials=100, true_rate=0.1, threshold=0.1, alpha=0.05)


def test_minimum_rejecting_failures_is_exact_boundary() -> None:
    cutoff = minimum_rejecting_failures(trials=100, threshold=0.1, alpha=0.05)
    assert binom.sf(cutoff - 1, 100, 0.1) <= 0.05
    assert binom.sf(cutoff - 2, 100, 0.1) > 0.05


def test_minimum_detectable_failure_rate_inverts_rejection_power() -> None:
    rate = minimum_detectable_failure_rate(trials=200, threshold=0.1, alpha=0.05, power=0.8)
    assert rate is not None
    assert rate > 0.1
    assert rejection_power(trials=200, true_rate=rate, threshold=0.1, alpha=0.05) == pytest.approx(
        0.8, abs=1e-11
    )
    assert rejection_power(trials=200, true_rate=rate - 1e-6, threshold=0.1, alpha=0.05) < 0.8


def test_maximum_certifiable_rate_inverts_support_power() -> None:
    rate = maximum_certifiable_failure_rate(trials=200, threshold=0.1, alpha=0.05, power=0.8)
    assert rate is not None
    assert rate < 0.1
    assert certification_power(
        trials=200, true_rate=rate, threshold=0.1, alpha=0.05
    ) == pytest.approx(0.8, abs=1e-11)
    assert certification_power(trials=200, true_rate=rate + 1e-6, threshold=0.1, alpha=0.05) < 0.8


def test_detectable_rates_report_unattainable_at_too_short_horizon() -> None:
    assert maximum_certifiable_failure_rate(trials=1, threshold=0.9, alpha=0.05, power=0.8) is None
    assert minimum_detectable_failure_rate(trials=1, threshold=0.9, alpha=0.05, power=0.8) is None


def test_fixed_design_detectable_rates_use_weighted_hoeffding_geometry() -> None:
    result = fixed_design_detectable_failure_rates(
        [10, 10],
        [0.5, 0.5],
        threshold=0.5,
        support_alpha=0.05,
        rejection_alpha=0.05,
        power=0.8,
        independence_unit="rollout",
    )
    expected_gap = math.sqrt(0.5 * 0.05 * math.log(20)) + math.sqrt(0.5 * 0.05 * math.log(5))
    assert result.concentration_variance == pytest.approx(0.05)
    assert result.effective_sample_size == pytest.approx(20)
    assert result.support_gap == pytest.approx(expected_gap)
    assert result.rejection_gap == pytest.approx(expected_gap)
    assert result.maximum_certifiable_failure_rate == pytest.approx(0.5 - expected_gap)
    assert result.minimum_detectable_failure_rate == pytest.approx(0.5 + expected_gap)


def test_prompt_cluster_detectability_does_not_gain_from_more_rollouts() -> None:
    first = fixed_design_detectable_failure_rates(
        [1, 1],
        [0.5, 0.5],
        threshold=0.5,
        support_alpha=0.05,
        rejection_alpha=0.05,
        power=0.8,
        independence_unit="prompt_cluster",
    )
    repeated = fixed_design_detectable_failure_rates(
        [100, 100],
        [0.5, 0.5],
        threshold=0.5,
        support_alpha=0.05,
        rejection_alpha=0.05,
        power=0.8,
        independence_unit="prompt_cluster",
    )
    assert first == repeated


def test_rejection_sample_size_achieves_requested_power_minimally() -> None:
    trials = rejection_sample_size(
        true_rate=0.2,
        threshold=0.1,
        alpha=0.05,
        power=0.8,
    )
    assert (
        rejection_power(
            trials=trials,
            true_rate=0.2,
            threshold=0.1,
            alpha=0.05,
        )
        >= 0.8
    )
    assert (
        rejection_power(
            trials=trials - 1,
            true_rate=0.2,
            threshold=0.1,
            alpha=0.05,
        )
        < 0.8
    )
