from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.stats import binom

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


def _sparse_arms(prompts: int = 300, rollouts: int = 20) -> dict[str, object]:
    """Two sparse arms shaped like the open-weight demo's harm group: about 4% and 0.5%."""

    candidate = np.zeros(prompts, dtype=int)
    candidate[:120] = 2
    baseline = np.zeros(prompts, dtype=int)
    baseline[:30] = 1
    zeros = np.zeros(prompts, dtype=int)
    trials = np.full(prompts, rollouts)
    return dict(
        candidate_failures=candidate,
        candidate_unclassified=zeros,
        candidate_trials=trials,
        baseline_failures=baseline,
        baseline_unclassified=zeros,
        baseline_trials=trials,
        weights=np.full(prompts, 1 / prompts),
        alpha_lower=0.00125,
        alpha_upper=0.00125,
    )


@pytest.mark.parametrize(
    ("independence_unit", "ratio"), [("rollout", 0.6), ("prompt_cluster", 0.6)]
)
def test_empirical_bernstein_difference_adapts_to_sparse_arms(
    independence_unit: str, ratio: float
) -> None:
    # Q = (1 + delta) / 2 mixes candidate failures near 0 with baseline safe outcomes near 1.
    # A variance proxy about 0 or 1 alone saw a large second moment and fell back to the
    # Hoeffding cap, so empirical Bernstein came out wider than Hoeffding. In prompt clusters
    # the proxy about 1/2 also needs padding for its true range, [0, 1/4], to get here.
    arms = _sparse_arms()
    bernstein = direct_risk_difference_bounds(
        **arms, method="weighted_empirical_bernstein", independence_unit=independence_unit
    )
    hoeffding = direct_risk_difference_bounds(
        **arms, method="weighted_hoeffding", independence_unit=independence_unit
    )
    assert bernstein.upper - bernstein.lower < ratio * (hoeffding.upper - hoeffding.lower)


@pytest.mark.parametrize("independence_unit", ["rollout", "prompt_cluster"])
def test_empirical_bernstein_difference_adapts_when_one_arm_fails_often(
    independence_unit: str,
) -> None:
    # 5% against 80%. Centering both arms at their failure-free end alone counts the
    # baseline's many failures; the proxies about 0 and 1 must stay available for this case.
    prompts, rollouts = 300, 20
    zeros = np.zeros(prompts, dtype=int)
    trials = np.full(prompts, rollouts)
    arms = dict(
        candidate_failures=np.ones(prompts, dtype=int),
        candidate_unclassified=zeros,
        candidate_trials=trials,
        baseline_failures=np.full(prompts, 16),
        baseline_unclassified=zeros,
        baseline_trials=trials,
        weights=np.full(prompts, 1 / prompts),
        alpha_lower=0.00125,
        alpha_upper=0.00125,
    )
    bernstein = direct_risk_difference_bounds(
        **arms, method="weighted_empirical_bernstein", independence_unit=independence_unit
    )
    hoeffding = direct_risk_difference_bounds(
        **arms, method="weighted_hoeffding", independence_unit=independence_unit
    )
    assert bernstein.upper - bernstein.lower < 0.9 * (hoeffding.upper - hoeffding.lower)


def _bernstein_half_width(variance: float, largest: float, alpha: float) -> float:
    tail = math.log(2 / alpha)
    linear = largest * tail / 3
    return linear + math.sqrt(2 * variance * tail + linear**2)


def test_empirical_bernstein_difference_matches_an_independent_computation() -> None:
    cf, cn = np.array([1, 0, 2]), np.array([40, 50, 60])
    bf, bn = np.array([0, 1, 0]), np.array([40, 50, 60])
    w = np.array([0.3, 0.3, 0.4])
    alpha = 0.05
    result = direct_risk_difference_bounds(
        candidate_failures=cf,
        candidate_unclassified=[0, 0, 0],
        candidate_trials=cn,
        baseline_failures=bf,
        baseline_unclassified=[0, 0, 0],
        baseline_trials=bn,
        weights=w,
        alpha_lower=alpha,
        alpha_upper=alpha,
        method="weighted_empirical_bernstein",
    )
    # Q's rollouts are candidate failures and baseline safe outcomes, each weighted
    # w / (2 n). Centered at 0 or 1 by arm, every squared deviation is a failure or a safe
    # outcome of that arm, and its range is the whole of [0, a^2].
    a = np.concatenate([w / 2 / cn, w / 2 / bn])
    n = np.concatenate([cn, bn])
    estimate = float(np.sum(a * np.concatenate([cf, bn - bf])))
    c2, c4, largest = float(np.sum(a**2 * n)), float(np.sum(a**4 * n)), float(np.max(a))
    squared = a[:3] ** 2, a[3:] ** 2
    candidate = {0: float(np.sum(squared[0] * cf)), 1: float(np.sum(squared[0] * (cn - cf)))}
    baseline = {0: float(np.sum(squared[1] * (bn - bf))), 1: float(np.sum(squared[1] * bf))}
    padding = math.sqrt(0.5 * c4 * math.log(8 / alpha))
    proxy = min(candidate[c] + baseline[b] for c in (0, 1) for b in (0, 1)) + padding
    assert proxy < c2 / 4  # the proxy, not the cap, decides the width
    width = _bernstein_half_width(min(c2 / 4, proxy), largest, alpha)
    assert result.lower == pytest.approx(2 * (estimate - width) - 1, abs=1e-12)
    assert result.upper == pytest.approx(2 * (estimate + width) - 1, abs=1e-12)


def test_prompt_cluster_empirical_bernstein_difference_matches_an_independent_computation() -> (
    None
):
    prompts = 400
    cf = np.zeros(prompts, dtype=int)
    cf[:40] = 1
    bf = np.zeros(prompts, dtype=int)
    bf[200:210] = 1
    trials = np.full(prompts, 10)
    w = np.full(prompts, 1 / prompts)
    alpha = 0.05
    result = direct_risk_difference_bounds(
        candidate_failures=cf,
        candidate_unclassified=np.zeros(prompts, dtype=int),
        candidate_trials=trials,
        baseline_failures=bf,
        baseline_unclassified=np.zeros(prompts, dtype=int),
        baseline_trials=trials,
        weights=w,
        alpha_lower=alpha,
        alpha_upper=alpha,
        method="weighted_empirical_bernstein",
        independence_unit="prompt_cluster",
    )
    q = (1 + cf / trials - bf / trials) / 2
    estimate = float(np.dot(w, q))
    c2, c4, largest = float(np.sum(w**2)), float(np.sum(w**4)), float(np.max(w))
    # Three proxies share half the budget. About 1/2, each squared deviation lies in
    # [0, 1/4], so its Hoeffding padding uses a sixteenth of the fourth-power sum.
    log_term = math.log(6 / alpha)
    half = float(np.dot(w**2, (q - 0.5) ** 2)) + math.sqrt(0.5 * c4 / 16 * log_term)
    low = float(np.dot(w**2, q**2)) + math.sqrt(0.5 * c4 * log_term)
    high = float(np.dot(w**2, (1 - q) ** 2)) + math.sqrt(0.5 * c4 * log_term)
    assert half < min(c2 / 4, low, high)  # the centered, range-aware proxy decides the width
    width = _bernstein_half_width(half, largest, alpha)
    assert result.lower == pytest.approx(2 * (estimate - width) - 1, abs=1e-12)
    assert result.upper == pytest.approx(2 * (estimate + width) - 1, abs=1e-12)


@pytest.mark.parametrize("independence_unit", ["rollout", "prompt_cluster"])
@pytest.mark.parametrize(
    ("candidate", "baseline"),
    [
        ((0.0, 0.0), (0.0, 0.0)),
        ((0.05, 0.3), (0.0, 0.1)),
        ((0.5, 0.2), (0.4, 0.6)),
        ((1.0, 0.0), (0.0, 1.0)),
    ],
)
def test_empirical_bernstein_difference_has_finite_sample_coverage(
    independence_unit: str, candidate: tuple[float, float], baseline: tuple[float, float]
) -> None:
    trials = (3, 4)
    weights = np.array([0.4, 0.6])
    truth = float(np.dot(weights, candidate) - np.dot(weights, baseline))
    coverage = 0.0
    counts = [range(n + 1) for n in trials]
    for c1, c2, b1, b2 in itertools.product(*counts, *counts):
        mass = float(
            binom.pmf(c1, trials[0], candidate[0])
            * binom.pmf(c2, trials[1], candidate[1])
            * binom.pmf(b1, trials[0], baseline[0])
            * binom.pmf(b2, trials[1], baseline[1])
        )
        bounds = direct_risk_difference_bounds(
            candidate_failures=[c1, c2],
            candidate_unclassified=[0, 0],
            candidate_trials=trials,
            baseline_failures=[b1, b2],
            baseline_unclassified=[0, 0],
            baseline_trials=trials,
            weights=weights,
            alpha_lower=0.025,
            alpha_upper=0.025,
            method="weighted_empirical_bernstein",
            independence_unit=independence_unit,
        )
        if bounds.lower <= truth <= bounds.upper:
            coverage += mass
    assert coverage >= 0.95 - 1e-12


@pytest.mark.parametrize("independence_unit", ["rollout", "prompt_cluster"])
def test_empirical_bernstein_difference_covers_in_the_sparse_regime(
    independence_unit: str,
) -> None:
    # Too many prompts to enumerate, so a seeded simulation: 4,000 replicates give a
    # miss-rate standard error of about 0.0034 at the nominal 0.05.
    rng = np.random.default_rng(20261031)
    prompts, rollouts, replicates = 60, 10, 4000
    candidate_p = np.array([0.0] * 40 + [0.05] * 15 + [0.3] * 5)
    baseline_p = np.array([0.0] * 50 + [0.02] * 10)
    weights = np.full(prompts, 1 / prompts)
    truth = float(np.dot(weights, candidate_p - baseline_p))
    trials = np.full(prompts, rollouts)
    zeros = np.zeros(prompts, dtype=int)
    misses = 0
    for _ in range(replicates):
        bounds = direct_risk_difference_bounds(
            candidate_failures=rng.binomial(rollouts, candidate_p),
            candidate_unclassified=zeros,
            candidate_trials=trials,
            baseline_failures=rng.binomial(rollouts, baseline_p),
            baseline_unclassified=zeros,
            baseline_trials=trials,
            weights=weights,
            alpha_lower=0.025,
            alpha_upper=0.025,
            method="weighted_empirical_bernstein",
            independence_unit=independence_unit,
        )
        misses += not bounds.lower <= truth <= bounds.upper
    assert misses / replicates <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / replicates)
