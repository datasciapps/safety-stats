"""The anytime-valid upper bound for runs stopped early (#126)."""

from __future__ import annotations

import math

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
