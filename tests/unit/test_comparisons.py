from __future__ import annotations

import math

import pytest

from inspect_safety_stats.stats.comparisons import (
    direct_ratio_bounds,
    direct_risk_difference_bounds,
)


def test_direct_difference_supports_unequal_arm_rollouts_and_weights() -> None:
    result = direct_risk_difference_bounds(
        candidate_failures=[1, 2],
        candidate_unclassified=[0, 0],
        candidate_trials=[10, 20],
        baseline_failures=[3, 1],
        baseline_unclassified=[0, 0],
        baseline_trials=[10, 10],
        weights=[0.5, 0.5],
        alpha_lower=0.05,
        alpha_upper=0.05,
        method="weighted_hoeffding",
    )
    assert result.completion_estimate_lower == pytest.approx(-0.1)
    assert result.completion_estimate_upper == pytest.approx(-0.1)
    assert result.lower <= -0.1 <= result.upper
    assert result.effective_sample_size > 0


def test_direct_difference_uses_adverse_arm_specific_missingness_completion() -> None:
    result = direct_risk_difference_bounds(
        candidate_failures=[0],
        candidate_unclassified=[2],
        candidate_trials=[10],
        baseline_failures=[0],
        baseline_unclassified=[3],
        baseline_trials=[10],
        weights=[1.0],
        alpha_lower=0.05,
        alpha_upper=0.05,
        method="weighted_hoeffding",
    )
    assert result.completion_estimate_lower == pytest.approx(-0.3)
    assert result.completion_estimate_upper == pytest.approx(0.2)
    assert result.lower <= -0.3
    assert result.upper >= 0.2


def test_prompt_cluster_difference_allows_cross_arm_within_prompt_dependence() -> None:
    result = direct_risk_difference_bounds(
        candidate_failures=[1, 8],
        candidate_unclassified=[0, 0],
        candidate_trials=[10, 10],
        baseline_failures=[2, 4],
        baseline_unclassified=[0, 0],
        baseline_trials=[10, 10],
        weights=[0.25, 0.75],
        alpha_lower=0.05,
        alpha_upper=0.05,
        method="weighted_empirical_bernstein",
        independence_unit="prompt_cluster",
    )
    assert result.completion_estimate_lower == pytest.approx(0.275)
    assert result.completion_estimate_upper == pytest.approx(0.275)
    assert result.independence_unit == "prompt_cluster"
    assert result.lower <= 0.275 <= result.upper


def test_weighted_kl_chernoff_risk_difference_supports_both_units() -> None:
    arguments = dict(
        candidate_failures=[1, 0],
        candidate_unclassified=[0, 0],
        candidate_trials=[20, 20],
        baseline_failures=[3, 1],
        baseline_unclassified=[0, 0],
        baseline_trials=[20, 20],
        weights=[0.5, 0.5],
        alpha_lower=0.01,
        alpha_upper=0.01,
        method="weighted_kl_chernoff",
    )

    rollout = direct_risk_difference_bounds(**arguments, independence_unit="rollout")
    cluster = direct_risk_difference_bounds(**arguments, independence_unit="prompt_cluster")

    assert rollout.lower <= -0.075 <= rollout.upper
    assert rollout.chernoff_kernel == "bernoulli_kl_equal_atomic_weights"
    assert rollout.candidate_lower is not None
    assert rollout.baseline_upper is not None
    assert cluster.lower <= -0.075 <= cluster.upper
    assert cluster.chernoff_kernel == "bernoulli_kl_equal_atomic_weights"
    assert cluster.candidate_lower is not None
    assert cluster.baseline_upper is not None


def test_two_arm_kl_difference_has_finite_sample_coverage() -> None:
    candidate_probability = 0.1
    baseline_probability = 0.3
    true_difference = candidate_probability - baseline_probability
    coverage = 0.0
    for candidate_failures in range(4):
        candidate_mass = (
            math.comb(3, candidate_failures)
            * candidate_probability**candidate_failures
            * (1 - candidate_probability) ** (3 - candidate_failures)
        )
        for baseline_failures in range(4):
            baseline_mass = (
                math.comb(3, baseline_failures)
                * baseline_probability**baseline_failures
                * (1 - baseline_probability) ** (3 - baseline_failures)
            )
            bounds = direct_risk_difference_bounds(
                candidate_failures=[candidate_failures],
                candidate_unclassified=[0],
                candidate_trials=[3],
                baseline_failures=[baseline_failures],
                baseline_unclassified=[0],
                baseline_trials=[3],
                weights=[1.0],
                alpha_lower=0.025,
                alpha_upper=0.025,
                method="weighted_kl_chernoff",
            )
            if bounds.lower <= true_difference <= bounds.upper:
                coverage += candidate_mass * baseline_mass
    assert coverage >= 0.95 - 1e-12


@pytest.mark.parametrize(
    ("method", "independence_unit"),
    [
        ("weighted_hoeffding", "rollout"),
        ("weighted_empirical_bernstein", "rollout"),
        ("weighted_kl_chernoff", "rollout"),
        ("weighted_hoeffding", "prompt_cluster"),
        ("weighted_empirical_bernstein", "prompt_cluster"),
        ("weighted_kl_chernoff", "prompt_cluster"),
    ],
)
def test_difference_bounds_are_symmetric_when_arms_are_reversed(
    method: str, independence_unit: str
) -> None:
    common = dict(
        candidate_failures=[1, 7],
        candidate_unclassified=[1, 0],
        candidate_trials=[10, 12],
        baseline_failures=[3, 2],
        baseline_unclassified=[0, 2],
        baseline_trials=[9, 11],
        weights=[0.4, 0.6],
        alpha_lower=0.03,
        alpha_upper=0.03,
        method=method,
        independence_unit=independence_unit,
    )
    forward = direct_risk_difference_bounds(**common)
    reverse = direct_risk_difference_bounds(
        candidate_failures=common["baseline_failures"],
        candidate_unclassified=common["baseline_unclassified"],
        candidate_trials=common["baseline_trials"],
        baseline_failures=common["candidate_failures"],
        baseline_unclassified=common["candidate_unclassified"],
        baseline_trials=common["candidate_trials"],
        weights=common["weights"],
        alpha_lower=common["alpha_lower"],
        alpha_upper=common["alpha_upper"],
        method=method,
        independence_unit=independence_unit,
    )
    assert forward.lower == pytest.approx(-reverse.upper)
    assert forward.upper == pytest.approx(-reverse.lower)


def test_ratio_and_relative_reduction_use_simultaneous_arm_bounds() -> None:
    ratio = direct_ratio_bounds(
        candidate_failures=[100],
        candidate_unclassified=[0],
        candidate_trials=[1000],
        baseline_failures=[200],
        baseline_unclassified=[0],
        baseline_trials=[1000],
        weights=[1.0],
        alpha_lower=0.05,
        alpha_upper=0.05,
        method="weighted_empirical_bernstein",
    )
    reduction = direct_ratio_bounds(
        candidate_failures=[100],
        candidate_unclassified=[0],
        candidate_trials=[1000],
        baseline_failures=[200],
        baseline_unclassified=[0],
        baseline_trials=[1000],
        weights=[1.0],
        alpha_lower=0.05,
        alpha_upper=0.05,
        method="weighted_empirical_bernstein",
        estimand="relative_risk_reduction",
    )
    assert ratio.estimate == pytest.approx(0.5)
    assert ratio.lower is not None and ratio.lower <= 0.5
    assert ratio.upper is not None and ratio.upper >= 0.5
    assert reduction.estimate == pytest.approx(0.5)
    assert reduction.lower == pytest.approx(1 - ratio.upper)
    assert reduction.upper == pytest.approx(1 - ratio.lower)


def test_ratio_reports_unbounded_upper_endpoint_when_baseline_can_be_zero() -> None:
    result = direct_ratio_bounds(
        candidate_failures=[1],
        candidate_unclassified=[0],
        candidate_trials=[20],
        baseline_failures=[0],
        baseline_unclassified=[0],
        baseline_trials=[20],
        weights=[1.0],
        alpha_lower=0.05,
        alpha_upper=0.05,
        method="weighted_hoeffding",
    )
    assert result.estimate is None
    assert result.upper is None
    assert result.warning is not None
