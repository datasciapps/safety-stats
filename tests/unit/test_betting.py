"""Tests for the betting (mixture-of-common-bets) direct mean bounds."""

from __future__ import annotations

import itertools
import math
from typing import Literal

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import logsumexp
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
    _betting_log_capital,
    weighted_betting_mean_bounds,
    weighted_bounded_betting_mean_bounds,
)
from tests.conformance import (
    MISS_TOLERANCE,
    Unit,
    count_outcomes,
    outcome_probabilities,
    rate_grid,
    worst_misses,
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


def test_unequal_weights_can_make_hoeffding_tighter() -> None:
    # The example in statistical-methods.md and plan-reference.md: tasks of 2, 10 and 50
    # equally weighted prompts, every prompt at 0.05. The betting exponents sum to
    # 1 / max A = 6, while Hoeffding's effective sample size is 1 / sum A^2 = 14.5.
    weights = np.concatenate([np.full(size, 1 / (3 * size)) for size in (2, 10, 50)])
    values = np.full(len(weights), 0.05)
    kwargs = dict(alpha_lower=0.05, alpha_upper=0.05, independence_profile="prompt")
    betting = weighted_bounded_betting_mean_bounds(values, values, weights, **kwargs)
    hoeffding = weighted_bounded_hoeffding_mean_bounds(values, values, weights, **kwargs)
    assert betting.upper == pytest.approx(0.459, rel=0, abs=5e-4)
    assert hoeffding.upper == pytest.approx(0.371, rel=0, abs=5e-4)


# --- Exact values ---------------------------------------------------------------------------


def test_zero_failures_in_a_hundred_rollouts_give_the_exact_upper_bound() -> None:
    # The root in mu of sum_j pi_j (1 - theta_j + theta_j / (1 - mu)) ** 100 = 1 / 0.05,
    # solved independently by bisection in 60-digit decimal arithmetic:
    # 0.03343284855087082386929760976...
    result = weighted_betting_mean_bounds(
        [0], [0], [100], [1.0], alpha_lower=0.05, alpha_upper=0.05
    )
    assert result.lower == 0.0
    assert result.upper == pytest.approx(0.033432848550870824, rel=0, abs=1e-14)


def _independent_prior() -> tuple[np.ndarray, np.ndarray]:
    """The documented prior, built here rather than read from the engine.

    Sixteen Beta(1/2, 1/2) quantiles as the bet fractions, with half of the mass spread
    evenly over them and the other half on the most aggressive one.
    """

    count = 16
    fractions = np.sin(np.pi * (np.arange(count) + 0.5) / count / 2) ** 2
    masses = np.full(count, 0.5 / count)
    masses[-1] += 0.5
    return fractions, masses


def _independent_endpoint(
    values: np.ndarray, weights: np.ndarray, *, alpha: float, direction: str
) -> float:
    """One endpoint by Brent's method on the mixture capital, every block kept apart.

    The engine merges equal blocks and bisects its own log-sum-exp; this keeps each block,
    uses SciPy's ``logsumexp`` and ``log1p``, and finds the root with ``brentq``.
    """

    fractions, masses = _independent_prior()
    exponents = weights / np.max(weights)

    def log_excess(mean: float) -> float:
        ratios = values / mean if direction == "lower" else (1 - values) / (1 - mean)
        growth = [float(np.sum(exponents * np.log1p(f * (ratios - 1)))) for f in fractions]
        return float(logsumexp(growth, b=masses)) - math.log(1 / alpha)

    edge = 1e-12
    estimate = min(max(float(weights @ values), edge), 1 - edge)
    if direction == "upper":
        if log_excess(1 - edge) < 0:
            return 1.0
        return float(brentq(log_excess, estimate, 1 - edge, xtol=1e-16, rtol=1e-15))
    if log_excess(edge) < 0:
        return 0.0
    return float(brentq(log_excess, edge, estimate, xtol=1e-16, rtol=1e-15))


def _rollout_blocks(
    failures: list[int], unclassified: list[int], trials: list[int], weights: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One block per rollout, failures first, then unknowns, then safe rollouts."""

    lower: list[float] = []
    upper: list[float] = []
    block_weights: list[float] = []
    for x, u, n, w in zip(failures, unclassified, trials, weights, strict=True):
        lower += [1.0] * x + [0.0] * (n - x)
        upper += [1.0] * (x + u) + [0.0] * (n - x - u)
        block_weights += [w / n] * n
    return np.asarray(lower), np.asarray(upper), np.asarray(block_weights)


_RNG = np.random.default_rng(2061)
_FRACTIONAL = _RNG.uniform(0.0, 0.4, 9)
INDEPENDENT_DESIGNS = {
    # Rollout blocks of two weights: equal blocks merge, and the exponents differ.
    "rollouts_unequal_weights": (
        *_rollout_blocks([2, 1], [1, 0], [10, 4], [0.3, 0.7]),
        0.01,
        0.1,
    ),
    "one_heavy_block_beside_light_ones": (
        np.r_[0.0, np.ones(6), np.zeros(14)],
        np.r_[0.0, np.ones(6), np.zeros(14)],
        np.r_[0.5, np.full(20, 0.025)],
        0.05,
        0.05,
    ),
    "fractional_blocks": (
        _FRACTIONAL,
        _FRACTIONAL + _RNG.uniform(0.0, 0.2, 9),
        _RNG.dirichlet(np.ones(9)),
        0.2,
        0.02,
    ),
}


@pytest.mark.parametrize("design", INDEPENDENT_DESIGNS)
def test_bounds_match_an_independent_inversion(design: str) -> None:
    lower_values, upper_values, weights, alpha_lower, alpha_upper = INDEPENDENT_DESIGNS[design]
    result = weighted_bounded_betting_mean_bounds(
        lower_values,
        upper_values,
        weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_profile="block",
    )
    lower = _independent_endpoint(lower_values, weights, alpha=alpha_lower, direction="lower")
    upper = _independent_endpoint(upper_values, weights, alpha=alpha_upper, direction="upper")
    assert 0.0 < lower < upper < 1.0
    assert result.lower == pytest.approx(lower, rel=0, abs=1e-12)
    assert result.upper == pytest.approx(upper, rel=0, abs=1e-12)


# --- The capital is an e-value ----------------------------------------------------------------


@pytest.mark.parametrize("direction", ["lower", "upper"])
@pytest.mark.parametrize("mean", [0.01, 0.3, 0.9])
def test_the_capital_has_expectation_exactly_one_under_a_homogeneous_null(
    direction: Literal["lower", "upper"], mean: float
) -> None:
    """Equal weights and a common block mean make every factor's expectation exactly one.

    The blocks are independent with exponent one, so each fixed fraction's capital has
    expectation one, and so has the mixture, whose prior masses sum to one. A prior that
    sums to more, or a fraction picked after seeing the outcomes, pushes it above one.
    """

    blocks = 8
    expected = 0.0
    for failures in range(blocks + 1):
        values = np.r_[np.ones(failures), np.zeros(blocks - failures)]
        log_capital = _betting_log_capital(values, np.ones(blocks), mean, direction=direction)
        expected += float(binom.pmf(failures, blocks, mean)) * math.exp(log_capital)
    assert expected == pytest.approx(1.0, rel=0, abs=1e-12)


@pytest.mark.parametrize(
    ("weights", "means", "direction", "exponent_one_breaks"),
    [
        # A heavy safe block beside light failing ones, and its mirror image.
        ((0.5, *[0.05] * 10), (0.0, *[0.6] * 10), "lower", True),
        ((0.5, *[0.05] * 10), (1.0, *[0.4] * 10), "upper", True),
        ((0.4, 0.3, 0.2, 0.1), (0.05, 0.9, 0.0, 0.5), "lower", False),
        ((0.4, 0.3, 0.2, 0.1), (0.05, 0.9, 0.0, 0.5), "upper", False),
    ],
)
def test_the_capital_is_an_e_value_when_only_the_weighted_mean_is_fixed(
    weights: tuple[float, ...],
    means: tuple[float, ...],
    direction: Literal["lower", "upper"],
    exponent_one_breaks: bool,
) -> None:
    """Exponents A_g / max A keep the expected capital at most one under any such null.

    The null fixes only the weighted sum of the block means. Jensen's inequality and then
    the weighted AM-GM inequality bound the expectation by one. Exponent one does not: on a
    heavy safe block beside light failing ones the same capital has expectation 14, which
    this test also checks so that it can tell the two apart.
    """

    w, p = np.asarray(weights), np.asarray(means)
    null = float(w @ p)

    def expected_capital(exponents: np.ndarray) -> float:
        total = 0.0
        for outcome in itertools.product((0.0, 1.0), repeat=len(w)):
            values = np.asarray(outcome)
            probability = float(np.prod(np.where(values == 1.0, p, 1 - p)))
            if probability > 0:
                log_capital = _betting_log_capital(values, exponents, null, direction=direction)
                total += probability * math.exp(log_capital)
        return total

    assert expected_capital(w / np.max(w)) <= 1.0 + 1e-12
    if exponent_one_breaks:
        assert expected_capital(np.ones(len(w))) > 10.0


# --- Coverage, endpoint by endpoint -----------------------------------------------------------

# Most of the weight on one block and the rest spread over light ones. The null fixes only
# the weighted sum, so a bound that let every block grow capital at the full rate, ignoring
# the exponents A_g / max A, would read the light blocks' failures as evidence against the
# truth: with exponent one, the lower endpoint of the first design misses with probability
# about 0.25 at a heavy rate of 0 and a light rate of 0.6, against an alpha of 0.05.
HEAVY_COUNT_DESIGNS: tuple[tuple[Unit, tuple[int, ...], tuple[float, ...]], ...] = (
    ("rollout", (1, 20), (0.5, 0.5)),
    ("rollout", (2, 12), (0.6, 0.4)),
    ("prompt_cluster", (3, 3, 3), (0.6, 0.2, 0.2)),
)
COVERAGE_LEVELS = (0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)


@pytest.mark.parametrize(("unit", "trials", "weights"), HEAVY_COUNT_DESIGNS)
def test_each_endpoint_covers_exactly_when_one_block_outweighs_the_rest(
    unit: Unit, trials: tuple[int, ...], weights: tuple[float, ...]
) -> None:
    """Each endpoint misses with probability at most its own alpha, at every grid rate.

    Every outcome is enumerated, so the miss probabilities are exact. The two alphas are
    checked separately, and the unequal pair catches one direction spending the other's.
    """

    outcomes = count_outcomes(trials, "independent")
    rates = rate_grid(COVERAGE_LEVELS, len(trials))
    probabilities = outcome_probabilities(outcomes, trials, rates, "independent")
    truth = rates @ np.asarray(weights)
    for alpha_lower, alpha_upper in ((0.05, 0.05), (0.02, 0.2)):
        bounds = [
            weighted_betting_mean_bounds(
                row,
                np.zeros(len(trials), dtype=int),
                trials,
                weights,
                alpha_lower=alpha_lower,
                alpha_upper=alpha_upper,
                independence_unit=unit,
            )
            for row in outcomes
        ]
        worst = worst_misses(
            np.asarray([bound.lower for bound in bounds]),
            np.asarray([bound.upper for bound in bounds]),
            probabilities,
            truth,
            rates,
        )
        context = (unit, trials, weights, alpha_lower, alpha_upper, worst)
        assert worst.lower <= alpha_lower + MISS_TOLERANCE, context
        assert worst.upper <= alpha_upper + MISS_TOLERANCE, context


@pytest.mark.parametrize(
    ("light", "heavy_weight", "light_rate"), [(20, 0.5, 0.6), (30, 0.4, 0.5), (40, 0.6, 0.9)]
)
def test_each_endpoint_covers_a_heavy_safe_block_beside_many_failing_ones(
    light: int, heavy_weight: float, light_rate: float
) -> None:
    """A heavy block that never fails beside many light Bernoulli blocks that often do.

    The light blocks are exchangeable, so the bounds depend on their failure count alone and
    the miss probabilities are exact sums over that count. The upper endpoint is checked on
    the mirror image: a heavy block that always fails beside light blocks that rarely do.
    With exponent one the lower endpoint misses with probability 0.25, 0.10 and 1.0 here.
    """

    alpha = 0.05
    weights = np.r_[heavy_weight, np.full(light, (1 - heavy_weight) / light)]
    truth = (1 - heavy_weight) * light_rate
    lower_miss = upper_miss = 0.0
    for failures in range(light + 1):
        probability = float(binom.pmf(failures, light, light_rate))
        values = np.r_[0.0, np.ones(failures), np.zeros(light - failures)]
        if _betting(values, weights, alpha).lower > truth + MISS_TOLERANCE:
            lower_miss += probability
        if _betting(1 - values, weights, alpha).upper < 1 - truth - MISS_TOLERANCE:
            upper_miss += probability
    assert lower_miss <= alpha
    assert upper_miss <= alpha


def test_each_endpoint_covers_heterogeneous_fractional_blocks_in_simulation() -> None:
    """Prompt-level blocks of eight rollouts, unequal weights, one heavy prompt rarely failing.

    Fractional block values put the outcomes out of reach of enumeration, so this simulates
    them: 400 seeded replications per endpoint. Each endpoint's miss rate must stay below its
    alpha of 0.1 plus three standard errors of a rate at alpha, sqrt(0.1 * 0.9 / 400) = 0.015.
    The engine misses in none of them; with exponent one each endpoint misses in more than
    390.
    """

    rng = np.random.default_rng(2026)
    replications, rollouts, alpha = 400, 8, 0.1
    weights = np.r_[0.4, rng.dirichlet(np.ones(30)) * 0.6]
    rates = np.r_[0.02, rng.uniform(0.3, 0.7, 30)]
    truth = float(weights @ rates)
    lower_misses = upper_misses = 0
    for _ in range(replications):
        values = rng.binomial(rollouts, rates) / rollouts
        lower_misses += _betting(values, weights, alpha).lower > truth
        # The mirror image, rates 1 - p, tests the upper endpoint on the same geometry.
        mirrored = 1 - rng.binomial(rollouts, rates) / rollouts
        upper_misses += _betting(mirrored, weights, alpha).upper < 1 - truth
    allowance = alpha + 3 * math.sqrt(alpha * (1 - alpha) / replications)
    assert lower_misses / replications <= allowance
    assert upper_misses / replications <= allowance


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
