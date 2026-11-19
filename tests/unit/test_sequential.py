"""The anytime-valid upper bound for runs stopped early (#126)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from inspect_safety_stats.stats.direct_mean import (
    bounded_kl_chernoff_upper_bound,
    weighted_bounded_kl_chernoff_mean_bounds,
)
from inspect_safety_stats.stats.sequential import (
    _log_capital_path,
    anytime_upper_bound,
    without_replacement_upper_bound,
)


def _monte_carlo_ceiling(alpha: float, runs: int) -> float:
    """``alpha`` plus three binomial standard errors, the slack every coverage check allows."""

    return alpha + 3 * math.sqrt(alpha * (1 - alpha) / runs)


def test_the_kl_helper_is_the_engines_upper_bound() -> None:
    for weights in (np.full(40, 1 / 40), np.linspace(1, 3, 40) / np.linspace(1, 3, 40).sum()):
        values = np.linspace(0, 0.3, 40)
        engine = weighted_bounded_kl_chernoff_mean_bounds(
            values, values, weights, alpha_lower=0.05, alpha_upper=0.025, independence_profile="x"
        )
        helper, kernel = bounded_kl_chernoff_upper_bound(
            float(np.dot(weights, values)), weights, alpha=0.025
        )
        assert helper == engine.upper
        assert kernel == engine.chernoff_kernel


def test_a_complete_order_bounds_the_realized_mean_exactly() -> None:
    rng = np.random.default_rng(3)
    weights = rng.uniform(1, 4, 30)
    weights /= weights.sum()
    outcomes = rng.binomial(5, 0.2, 30) / 5
    order = rng.permutation(30)
    bound = anytime_upper_bound(outcomes[order], order, weights, alpha=0.05)
    assert bound.realized_upper == pytest.approx(float(np.dot(weights, outcomes)), abs=1e-12)
    expected, _ = bounded_kl_chernoff_upper_bound(bound.realized_upper, weights, alpha=0.025)
    assert bound.upper == expected
    assert bound.draws == bound.population == 30


def _prior_by_definition() -> tuple[list[Fraction], list[Fraction]]:
    """The betting bounds' fixed prior, from its definition rather than the library's code.

    Sixteen bet fractions sin^2(pi q / 2) at the Beta(1/2, 1/2) quantiles q = (2j - 1) / 32,
    each with prior weight 1/32, and half the mass besides on the largest.
    """

    fractions = [
        Fraction(math.sin(math.pi * (2 * j - 1) / 32 / 2) ** 2) for j in range(1, 17)
    ]
    weights = [Fraction(1, 32)] * 16
    weights[-1] += Fraction(1, 2)
    return fractions, weights


def _rejected_exactly(
    values: list[Fraction], population: int, mean: Fraction, alpha: Fraction
) -> bool:
    """Whether the capital against ``mean`` reaches ``1 / alpha`` at some prefix, in exact
    rational arithmetic (given the prior's fractions as floats), from the capital's definition:
    the product over draws of ``1 - f + f (1 - z_t) / (1 - m_t)``, with
    ``m_t = (N mean - sum_{s<t} z_s) / (N - t + 1)``, mixed over the prior.
    """

    fractions, weights = _prior_by_definition()
    capitals = [Fraction(1)] * len(fractions)
    seen = Fraction(0)
    for step, value in enumerate(values, start=1):
        conditional = (population * mean - seen) / (population - step + 1)
        if value == 1:
            ratio = Fraction(0)
        elif conditional >= 1:
            # The unseen values would all have to be one: a smaller draw rules the mean out.
            return True
        else:
            ratio = (1 - value) / (1 - conditional)
        capitals = [
            capital * (1 - fraction + fraction * ratio)
            for capital, fraction in zip(capitals, fractions, strict=True)
        ]
        if sum(w * c for w, c in zip(weights, capitals, strict=True)) >= 1 / alpha:
            return True
        seen += value
    return False


def _reference_upper(values: list[Fraction], population: int, alpha: Fraction) -> Fraction:
    """The least rejected mean by exact bisection, or the largest possible mean if none is."""

    cap = min(Fraction(1), (sum(values, Fraction(0)) + population - len(values)) / population)
    if not _rejected_exactly(values, population, cap, alpha):
        return cap
    inside, outside = Fraction(0), cap
    for _ in range(50):
        midpoint = (inside + outside) / 2
        if _rejected_exactly(values, population, midpoint, alpha):
            outside = midpoint
        else:
            inside = midpoint
    return outside


SEQUENCES = {
    # Failures (z = 1) and partial shares early in a prefix of 15 of 40.
    "mixed": ("0 0 1 0 1/2 0 0 1/4 0 0 1 0 0 0 3/4", 40, Fraction(1, 40)),
    "safe": (" ".join(["0"] * 30), 200, Fraction(1, 40)),
    # Too few draws to reject the largest mean the draws allow: the bound is that mean.
    "capped": ("1 0 0 0 0 1/3 0 0 0 0 0", 12, Fraction(1, 20)),
}


@pytest.mark.parametrize("case", list(SEQUENCES), ids=list(SEQUENCES))
def test_the_sequence_bound_before_the_end_matches_an_exact_reference(case: str) -> None:
    """Pins step 1 at t < N, where nothing else determines it, to the definition: the
    threshold 1 / alpha, the factor 1 - f of a draw of one, the prior, the running
    intersection and the cap."""

    text, population, alpha = SEQUENCES[case]
    values = [Fraction(item) for item in text.split()]
    bound = without_replacement_upper_bound(
        np.array([float(item) for item in values]), population, alpha=float(alpha)
    )
    reference = _reference_upper(values, population, alpha)
    assert bound == pytest.approx(float(reference), abs=1e-12)
    cap = min(1, (sum(values) + population - len(values)) / population)
    if case == "capped":
        assert reference == cap
    else:
        assert reference < cap
        # The bound is a rejected candidate, never the last one the bisection kept.
        log_capital = _log_capital_path(
            np.array([float(item) for item in values]), population, bound
        )
        assert np.max(log_capital) >= math.log(1 / float(alpha))


def test_the_anytime_bound_before_the_end_matches_the_reference_at_half_alpha() -> None:
    """Uneven weights, two independent rollouts per item, 12 of 30 items drawn: step 1 at
    alpha / 2 on the scaled outcomes, rescaled by N max w, and step 2 at alpha / 2 on N r
    blocks."""

    weights = np.linspace(1, 2, 30)
    weights /= weights.sum()
    order = np.array([29, 3, 17, 8, 22, 0, 11, 26, 5, 14, 19, 1])
    outcomes = np.array([0, 0.5, 0, 1, 0, 0, 0.5, 0, 0, 1, 0, 0])

    bound = anytime_upper_bound(outcomes, order, weights, alpha=0.05, independent_rollouts=2)

    scaled = [Fraction(float(item)) for item in weights[order] * outcomes / weights.max()]
    realized = float(_reference_upper(scaled, 30, Fraction(0.025))) * 30 * float(weights.max())
    assert realized < 1
    assert bound.realized_upper == pytest.approx(realized, abs=1e-12)
    expected, _ = bounded_kl_chernoff_upper_bound(
        realized, np.repeat(weights / 2, 2), alpha=0.025
    )
    assert bound.upper == pytest.approx(expected, abs=1e-12)
    assert (bound.alpha_sequence, bound.alpha_population) == (0.025, 0.025)


def test_no_draws_bound_nothing() -> None:
    bound = anytime_upper_bound([], np.array([], dtype=int), np.full(10, 0.1), alpha=0.05)
    assert bound.upper == 1.0
    assert bound.estimate is None
    assert bound.draws == 0


def test_the_bound_tightens_as_safe_draws_accumulate() -> None:
    weights = np.full(500, 1 / 500)
    uppers = [
        anytime_upper_bound(np.zeros(t), np.arange(t), weights, alpha=0.05).upper
        for t in (10, 50, 200, 500)
    ]
    assert uppers == sorted(uppers, reverse=True)
    assert uppers[-1] < 0.01


def test_counting_unknown_outcomes_as_failures_never_lowers_the_bound() -> None:
    rng = np.random.default_rng(8)
    weights = rng.uniform(1, 2, 60)
    weights /= weights.sum()
    order = rng.permutation(60)[:40]
    for _ in range(50):
        outcomes = rng.binomial(3, 0.1, 40) / 3
        raised = np.where(rng.random(40) < 0.2, 1.0, outcomes)
        assert (
            anytime_upper_bound(raised, order, weights, alpha=0.05).upper
            >= anytime_upper_bound(outcomes, order, weights, alpha=0.05).upper
        )


def test_independent_rollouts_tighten_only_the_population_step() -> None:
    weights = np.full(200, 1 / 200)
    outcomes = np.full(100, 0.1)
    items = anytime_upper_bound(outcomes, np.arange(100), weights, alpha=0.05)
    rollouts = anytime_upper_bound(
        outcomes, np.arange(100), weights, alpha=0.05, independent_rollouts=10
    )
    assert rollouts.realized_upper == items.realized_upper
    assert rollouts.upper < items.upper


def test_the_capital_is_a_fair_game_at_the_true_population_mean() -> None:
    rng = np.random.default_rng(11)
    population = rng.binomial(4, 0.3, 25) / 4
    mean = float(population.mean())
    finals = [
        math.exp(_log_capital_path(rng.permutation(population), 25, mean)[-1])
        for _ in range(20_000)
    ]
    assert np.mean(finals) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize(
    ("seed", "weights", "rates"),
    [
        # equal weights, sparse failures; unequal weights, heterogeneous rates
        (1, np.full(80, 1 / 80), np.full(80, 0.02)),
        (
            2,
            np.linspace(1, 4, 80) / np.linspace(1, 4, 80).sum(),
            np.linspace(0.0, 0.3, 80),
        ),
    ],
)
def test_the_bound_holds_at_every_prefix_at_once(
    seed: int, weights: np.ndarray, rates: np.ndarray
) -> None:
    """The bound misses theta at some prefix at most alpha of the time.

    A miss at some prefix needs either the sequence to reject the true realized mean by then
    (its capital at that mean reaching 2 / alpha) or the population step to miss theta from the
    true realized mean, so the union of the two bounds the chance of any miss.
    """

    alpha, runs, epochs = 0.1, 3000, 4
    rng = np.random.default_rng(seed)
    theta = float(np.dot(weights, rates))
    threshold = math.log(2 / alpha)
    misses = 0
    for _ in range(runs):
        outcomes = rng.binomial(epochs, rates) / epochs
        order = rng.permutation(len(weights))
        realized = float(np.dot(weights, outcomes))
        scaled = weights[order] * outcomes[order] / weights.max()
        mean = realized / (len(weights) * weights.max())
        sequence_miss = np.max(_log_capital_path(scaled, len(weights), mean)) >= threshold
        population_upper, _ = bounded_kl_chernoff_upper_bound(realized, weights, alpha=alpha / 2)
        population_miss = theta > population_upper
        misses += bool(sequence_miss or population_miss)
    assert misses / runs <= _monte_carlo_ceiling(alpha, runs)


def test_the_bound_holds_where_a_peeking_rule_stops() -> None:
    """A rule that stops at the first prefix whose estimate looks safe cannot break the bound."""

    alpha, runs, items = 0.1, 1500, 100
    weights = np.full(items, 1 / items)
    rng = np.random.default_rng(5)
    rates = rng.uniform(0, 0.12, items)
    theta = float(rates.mean())
    misses = 0
    for _ in range(runs):
        outcomes = rng.binomial(1, rates).astype(float)
        order = rng.permutation(items)
        running = np.cumsum(outcomes[order]) / np.arange(1, items + 1)
        looks_safe = np.flatnonzero((running < theta / 2) & (np.arange(1, items + 1) >= 10))
        stop = int(looks_safe[0]) + 1 if len(looks_safe) else items
        bound = anytime_upper_bound(outcomes[order][:stop], order[:stop], weights, alpha=alpha)
        misses += theta > bound.upper
    assert misses / runs <= _monte_carlo_ceiling(alpha, runs)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (([0.5], [0], [0.5, 0.6]), "sum to one"),
        (([1.5], [0], [0.5, 0.5]), r"\[0, 1\]"),
        (([0.5, 0.5], [0, 0], [0.5, 0.5]), "distinct"),
        (([0.5], [2], [0.5, 0.5]), "distinct"),
        (([0.5, 0.5], [0], [0.5, 0.5]), "aligned"),
    ],
)
def test_malformed_inputs_are_refused(arguments: tuple, message: str) -> None:
    outcomes, order, weights = arguments
    with pytest.raises(ValueError, match=message):
        anytime_upper_bound(outcomes, np.asarray(order), weights, alpha=0.05)


def test_fractional_order_indices_are_refused() -> None:
    with pytest.raises(TypeError, match="integer"):
        anytime_upper_bound([0.5], np.array([0.0]), [0.5, 0.5], alpha=0.05)


@pytest.mark.parametrize("rollouts", [0, True])
def test_independent_rollouts_must_be_a_positive_integer(rollouts: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        anytime_upper_bound(
            [0.5], np.array([0]), [0.5, 0.5], alpha=0.05, independent_rollouts=rollouts
        )


def test_more_draws_than_the_population_are_refused() -> None:
    with pytest.raises(ValueError, match="population"):
        without_replacement_upper_bound([0.1, 0.2, 0.3], 2, alpha=0.05)
