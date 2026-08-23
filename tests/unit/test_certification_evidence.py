from __future__ import annotations

import pytest

from inspect_safety_stats.certification.evidence import (
    exact_rejection_p_value,
    exact_support_p_value,
    exact_threshold_bounds,
    mean_bounds,
    mean_rejection_p_value,
    mean_support_p_value,
)


def test_exact_threshold_p_values_match_boundary_probabilities() -> None:
    assert exact_support_p_value(0, 20, 0.1) == pytest.approx(0.9**20)
    assert exact_rejection_p_value(20, 20, 0.1) == pytest.approx(0.1**20)


def test_exact_p_values_are_dual_to_one_sided_bounds() -> None:
    threshold = 0.2
    alpha = 0.05
    lower, upper = exact_threshold_bounds(
        1,
        0,
        30,
        support_alpha=alpha,
        rejection_alpha=alpha,
    )
    assert (exact_support_p_value(1, 30, threshold) <= alpha) == (upper <= threshold)
    assert (exact_rejection_p_value(1, 30, threshold) <= alpha) == (lower > threshold)


def test_missing_outcomes_are_adverse_to_each_direction() -> None:
    lower_complete, upper_complete = exact_threshold_bounds(
        0, 0, 20, support_alpha=0.05, rejection_alpha=0.05
    )
    lower_missing, upper_missing = exact_threshold_bounds(
        0, 3, 20, support_alpha=0.05, rejection_alpha=0.05
    )
    assert lower_missing == pytest.approx(lower_complete)
    assert upper_missing > upper_complete


def test_mean_bound_p_values_invert_the_configured_construction() -> None:
    failures = [0, 0]
    unclassified = [0, 0]
    trials = [100, 100]
    weights = [0.5, 0.5]
    threshold = 0.1
    alpha = 0.05
    bounds = mean_bounds(
        "weighted_empirical_bernstein",
        failures,
        unclassified,
        trials,
        weights,
        support_alpha=alpha,
        rejection_alpha=alpha,
        independence_unit="rollout",
    )
    support_p = mean_support_p_value(
        "weighted_empirical_bernstein",
        failures,
        unclassified,
        trials,
        weights,
        threshold=threshold,
        independence_unit="rollout",
    )
    rejection_p = mean_rejection_p_value(
        "weighted_empirical_bernstein",
        failures,
        unclassified,
        trials,
        weights,
        threshold=threshold,
        independence_unit="rollout",
    )
    assert (support_p <= alpha) == (bounds.upper <= threshold)
    assert (rejection_p <= alpha) == (bounds.lower > threshold)
    assert rejection_p == 1.0
