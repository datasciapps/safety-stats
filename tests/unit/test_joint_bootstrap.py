from __future__ import annotations

import numpy as np
import pytest

from inspect_safety_stats.stats import (
    GeneralLinearIntervalConstraint,
    jeffreys_binomial_probabilities,
    joint_max_t_intervals,
    joint_max_t_upper_bounds,
    project_general_linear_confidence_region,
    stratified_bernoulli_draws,
)


def test_jeffreys_resampling_is_non_degenerate_at_binomial_boundaries() -> None:
    probabilities = jeffreys_binomial_probabilities([0, 8], [8, 8])
    assert 0 < probabilities[0] < 0.5
    assert 0.5 < probabilities[1] < 1

    draws = stratified_bernoulli_draws([0, 8], [8, 8], replicates=2000, seed=7)
    assert np.any(draws[:, 0] > 0)
    assert np.any(draws[:, 1] < 1)


def test_duplicate_tree_target_does_not_change_max_t_critical_value() -> None:
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(2000, 1))
    single = joint_max_t_intervals([0.0], draws, alpha=0.05)
    duplicate = joint_max_t_intervals(
        [0.0, 0.0], np.column_stack([draws[:, 0], draws[:, 0]]), alpha=0.05
    )
    assert duplicate.critical_value_upper == pytest.approx(single.critical_value_upper)


def test_one_sided_upper_max_t_is_tighter_than_two_sided_upper_endpoint() -> None:
    rng = np.random.default_rng(14)
    draws = rng.normal(size=(5000, 4))
    observed = np.array([0.1, 0.2, 0.3, 0.4])

    two_sided = joint_max_t_intervals(observed, draws, alpha=0.05)
    upper = joint_max_t_upper_bounds(observed, draws, alpha=0.05)

    assert upper.critical_value < two_sided.critical_value_upper
    assert np.all(upper.upper < two_sided.upper)
    assert upper.target_count == 4
    assert upper.replicates == 5000


def test_one_sided_upper_matches_per_direction_upper_calibration() -> None:
    rng = np.random.default_rng(21)
    draws = rng.normal(size=(3000, 3))
    observed = np.array([0.2, 0.4, 0.6])

    directional = joint_max_t_intervals(
        observed,
        draws,
        alpha=0.05,
        coverage_mode="per_direction",
    )
    upper = joint_max_t_upper_bounds(observed, draws, alpha=0.05)

    assert upper.critical_value == pytest.approx(directional.critical_value_upper)
    assert upper.upper == pytest.approx(directional.upper)


def test_joint_linear_projection_enforces_multi_arm_difference_identity() -> None:
    candidate = np.array([0.5, 0.5, 0.0, 0.0])
    baseline = np.array([0.0, 0.0, 0.5, 0.5])
    difference = candidate - baseline
    projection = project_general_linear_confidence_region(
        [
            GeneralLinearIntervalConstraint("candidate", candidate, 0.2, 0.3),
            GeneralLinearIntervalConstraint("baseline", baseline, 0.4, 0.5),
            GeneralLinearIntervalConstraint("difference", difference, -0.4, 0.0),
        ],
        {"difference": difference},
        variable_bounds=[(0.0, 1.0)] * 4,
    )
    assert projection.feasible
    assert projection.intervals["difference"] == pytest.approx((-0.3, -0.1))
