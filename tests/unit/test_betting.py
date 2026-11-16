"""Tests for the betting (mixture-of-common-bets) direct mean bounds."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.stats import binom

from inspect_safety_stats.config.models import (
    DirectMeanIntervalsSpec,
    IndependenceSelectionSpec,
)
from inspect_safety_stats.stats import (
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
)

# Experimental, so not in inspect_safety_stats.stats.__all__ until it graduates.
from inspect_safety_stats.stats.direct_mean import (
    weighted_betting_mean_bounds,
    weighted_bounded_betting_mean_bounds,
)


def _equal(count: int) -> np.ndarray:
    return np.full(count, 1.0 / count)


def _betting(values: np.ndarray, weights: np.ndarray, alpha: float = 0.05):  # type: ignore[no-untyped-def]
    return weighted_bounded_betting_mean_bounds(
        values,
        values,
        weights,
        alpha_lower=alpha,
        alpha_upper=alpha,
        independence_profile="prompt",
    )


def test_betting_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        weighted_bounded_betting_mean_bounds(
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.4],
            alpha_lower=0.05,
            alpha_upper=0.05,
            independence_profile="rollout",
        )
    with pytest.raises(ValueError, match="0 <= lower <= upper <= 1"):
        weighted_bounded_betting_mean_bounds(
            [0.0, 0.2],
            [0.0, 1.2],
            [0.5, 0.5],
            alpha_lower=0.05,
            alpha_upper=0.05,
            independence_profile="rollout",
        )
    with pytest.raises(ValueError, match="alpha"):
        weighted_bounded_betting_mean_bounds(
            [0.0, 0.2],
            [0.0, 0.2],
            [0.5, 0.5],
            alpha_lower=0.0,
            alpha_upper=0.05,
            independence_profile="rollout",
        )


def test_zero_failures_equal_weights_is_tighter_than_hoeffding_and_near_kl() -> None:
    n = 100
    zeros = np.zeros(n)
    result = _betting(zeros, _equal(n))
    hoeffding = weighted_bounded_hoeffding_mean_bounds(
        zeros, zeros, _equal(n), alpha_lower=0.05, alpha_upper=0.05, independence_profile="p"
    )
    kl = weighted_bounded_kl_chernoff_mean_bounds(
        zeros, zeros, _equal(n), alpha_lower=0.05, alpha_upper=0.05, independence_profile="p"
    )
    assert result.method == "weighted_betting"
    assert result.lower == 0.0
    assert result.completion_estimate_upper == 0.0
    assert 0.0 < result.upper < hoeffding.upper
    # The exact Bernoulli-KL inversion is optimal for binary zero-failure data;
    # the mixture bound pays only the prior mass on its most aggressive bet.
    assert result.upper <= 1.5 * kl.upper
    assert result.half_width_upper == pytest.approx(result.upper)
    assert result.effective_sample_size == pytest.approx(n)
    assert result.chernoff_kernel == "common_bet_mixture_jensen_exponent"


def test_all_failures_mirror_all_safe_by_symmetry() -> None:
    n = 40
    zeros, ones = np.zeros(n), np.ones(n)
    safe = _betting(zeros, _equal(n))
    failing = _betting(ones, _equal(n))
    assert failing.upper == 1.0
    assert failing.lower > 0.0
    # Reflecting Y -> 1 - Y swaps the two endpoints exactly.
    assert failing.lower == pytest.approx(1.0 - safe.upper, abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_betting_interval_always_contains_the_plug_in_estimate(seed: int) -> None:
    rng = np.random.default_rng(seed)
    count = int(rng.integers(2, 40))
    values = rng.uniform(0.0, 1.0, count)
    weights = rng.dirichlet(np.ones(count))
    result = _betting(values, weights, alpha=0.2)
    estimate = float(values @ weights)
    assert 0.0 <= result.lower <= estimate + 1e-12
    assert estimate - 1e-12 <= result.upper <= 1.0
    assert result.completion_estimate_lower == pytest.approx(estimate)


def test_unclassified_completion_is_monotone() -> None:
    weights = _equal(4)
    all_known = np.array([0.0, 0.0, 0.1, 0.0])
    with_unknowns_as_failures = np.array([0.0, 1.0, 0.1, 0.5])
    known = _betting(all_known, weights)
    completed = weighted_bounded_betting_mean_bounds(
        all_known,
        with_unknowns_as_failures,
        weights,
        alpha_lower=0.05,
        alpha_upper=0.05,
        independence_profile="prompt",
    )
    assert completed.lower == pytest.approx(known.lower)
    assert completed.upper > known.upper


def test_betting_is_monotone_in_alpha() -> None:
    rng = np.random.default_rng(11)
    values = rng.uniform(0.0, 0.4, 25)
    weights = _equal(25)
    strict = _betting(values, weights, alpha=0.01)
    loose = _betting(values, weights, alpha=0.10)
    assert strict.lower <= loose.lower
    assert strict.upper >= loose.upper


def test_low_variance_fractional_blocks_beat_the_range_only_engines() -> None:
    # Fifty prompts, each exactly one failure in twenty rollouts: the block
    # values are all 0.05, the regime the prompt independence profile produces.
    values = np.full(50, 0.05)
    weights = _equal(50)
    kwargs = dict(alpha_lower=0.05, alpha_upper=0.05, independence_profile="prompt")
    betting = weighted_bounded_betting_mean_bounds(values, values, weights, **kwargs)
    bernstein = weighted_bounded_empirical_bernstein_mean_bounds(values, values, weights, **kwargs)
    hoeffding = weighted_bounded_hoeffding_mean_bounds(values, values, weights, **kwargs)
    kl = weighted_bounded_kl_chernoff_mean_bounds(values, values, weights, **kwargs)
    assert betting.upper < bernstein.upper
    assert betting.upper < hoeffding.upper
    assert betting.upper <= kl.upper


@pytest.mark.parametrize(
    ("trials", "weights", "probabilities"),
    [
        ((4, 4), (0.5, 0.5), (0.05, 0.4)),
        ((4, 5), (0.3, 0.7), (0.05, 0.4)),
        ((4, 5), (0.3, 0.7), (0.2, 0.8)),
        ((4, 5), (0.3, 0.7), (0.6, 0.6)),
        ((6, 3), (0.9, 0.1), (0.0, 1.0)),
    ],
)
def test_betting_count_form_has_finite_sample_coverage(
    trials: tuple[int, int],
    weights: tuple[float, float],
    probabilities: tuple[float, float],
) -> None:
    true_mean = float(np.dot(weights, probabilities))
    coverage = 0.0
    for first in range(trials[0] + 1):
        for second in range(trials[1] + 1):
            probability_mass = float(
                binom.pmf(first, trials[0], probabilities[0])
                * binom.pmf(second, trials[1], probabilities[1])
            )
            result = weighted_betting_mean_bounds(
                failures=[first, second],
                unclassified=[0, 0],
                trials=trials,
                weights=weights,
                alpha_lower=0.025,
                alpha_upper=0.025,
            )
            assert result.method == "weighted_betting"
            if result.lower <= true_mean <= result.upper:
                coverage += probability_mass
    assert coverage >= 0.95 - 1e-12


@pytest.mark.parametrize(
    "probabilities", [(0.0, 0.0), (0.05, 0.4), (0.2, 0.8), (0.6, 0.6), (1.0, 1.0)]
)
def test_betting_prompt_cluster_unit_has_finite_sample_coverage(
    probabilities: tuple[float, float],
) -> None:
    trials = (4, 5)
    weights = (0.3, 0.7)
    true_mean = float(np.dot(weights, probabilities))
    coverage = 0.0
    for first in range(trials[0] + 1):
        for second in range(trials[1] + 1):
            probability_mass = float(
                binom.pmf(first, trials[0], probabilities[0])
                * binom.pmf(second, trials[1], probabilities[1])
            )
            result = weighted_betting_mean_bounds(
                failures=[first, second],
                unclassified=[0, 0],
                trials=trials,
                weights=weights,
                alpha_lower=0.025,
                alpha_upper=0.025,
                independence_unit="prompt_cluster",
            )
            if result.lower <= true_mean <= result.upper:
                coverage += probability_mass
    assert coverage >= 0.95 - 1e-12


def test_block_form_exact_coverage_with_fractional_blocks_and_unequal_weights() -> None:
    # Three heterogeneous blocks, each the mean of five Bernoulli rollouts,
    # with unequal weights so the Jensen exponent path is exercised.
    rollouts = 5
    weights = np.array([0.5, 0.3, 0.2])
    probabilities = np.array([0.1, 0.5, 0.9])
    true_mean = float(weights @ probabilities)
    coverage = 0.0
    for counts in itertools.product(range(rollouts + 1), repeat=3):
        probability_mass = float(
            np.prod([binom.pmf(k, rollouts, p) for k, p in zip(counts, probabilities, strict=True)])
        )
        values = np.asarray(counts, dtype=float) / rollouts
        result = weighted_bounded_betting_mean_bounds(
            values,
            values,
            weights,
            alpha_lower=0.025,
            alpha_upper=0.025,
            independence_profile="prompt",
        )
        if result.lower <= true_mean <= result.upper:
            coverage += probability_mass
    assert coverage >= 0.95 - 1e-12


def test_monte_carlo_coverage_over_many_heterogeneous_unequal_blocks() -> None:
    rng = np.random.default_rng(2026)
    replications = 1200
    blocks = 30
    alpha = 0.1
    covered = 0
    for _ in range(replications):
        means = rng.uniform(0.0, 0.3, blocks)
        weights = rng.dirichlet(np.ones(blocks))
        values = rng.binomial(8, means) / 8.0
        result = weighted_bounded_betting_mean_bounds(
            values,
            values,
            weights,
            alpha_lower=alpha,
            alpha_upper=alpha,
            independence_profile="prompt",
        )
        true_mean = float(weights @ means)
        covered += result.lower <= true_mean <= result.upper
    # Two one-sided endpoints at alpha each give a joint floor of 1 - 2 alpha.
    assert covered / replications >= 1 - 2 * alpha - 0.02


def test_lower_endpoint_is_finite_and_symmetric_for_dense_failures() -> None:
    values = np.full(50, 0.95)
    result = _betting(values, _equal(50))
    mirrored = _betting(1.0 - values, _equal(50))
    assert 0.0 < result.lower < 0.95
    assert result.lower == pytest.approx(1.0 - mirrored.upper, abs=1e-9)
    assert math.isfinite(result.half_width_lower)


def test_plan_accepts_betting_as_primary_and_reference_method() -> None:
    independence = IndependenceSelectionSpec(primary="rollout")
    primary = DirectMeanIntervalsSpec(
        enabled=True,
        method="weighted_betting",
        reference_methods=["weighted_kl_chernoff"],
        family="direct_means",
        independence=independence,
    )
    reference = DirectMeanIntervalsSpec(
        enabled=True,
        method="weighted_kl_chernoff",
        reference_methods=["weighted_betting", "weighted_hoeffding"],
        family="direct_means",
        independence=independence,
    )
    assert primary.method == "weighted_betting"
    assert "weighted_betting" in reference.reference_methods


def test_horvitz_thompson_census_with_betting_matches_the_direct_engine() -> None:
    # A census (every frame item drawn with probability one) has inflation one
    # and must reproduce the ordinary block engine exactly.
    from inspect_safety_stats.stats import InclusionDesign, horvitz_thompson_mean_bounds

    rng = np.random.default_rng(5)
    values = rng.uniform(0.0, 0.3, 12)
    weights = rng.dirichlet(np.ones(12))
    design = InclusionDesign(
        design_weights=weights,
        inclusion_probabilities=np.ones(12),
        sampled=np.ones(12, dtype=bool),
    )
    sampled = horvitz_thompson_mean_bounds(
        design, values, values, alpha_lower=0.05, alpha_upper=0.05, method="weighted_betting"
    )
    direct = _betting(values, weights)
    assert sampled.method == "weighted_betting"
    assert sampled.inflation == pytest.approx(1.0)
    assert sampled.lower == pytest.approx(direct.lower, abs=1e-9)
    assert sampled.upper == pytest.approx(direct.upper, abs=1e-9)
