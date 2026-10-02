from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from inspect_safety_stats.stats.attempts import (
    attempt_atoms,
    attempt_risk_completions,
    attempt_risk_estimate,
    attempt_risk_from_rate,
)
from inspect_safety_stats.stats.direct_mean import (
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
)


@pytest.mark.parametrize(("n", "failures", "k"), [(8, 0, 3), (8, 2, 3), (10, 5, 4), (6, 6, 2)])
def test_binary_estimate_is_the_closed_form(n: int, failures: int, k: int) -> None:
    values = [1.0] * failures + [0.0] * (n - failures)
    expected = 1 - math.comb(n - failures, k) / math.comb(n, k)
    assert attempt_risk_estimate(values, k) == pytest.approx(expected)


def test_graded_estimate_averages_the_maximum_over_every_subset() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(size=7).tolist()
    for k in (1, 2, 3, 7):
        brute = np.mean([max(subset) for subset in itertools.combinations(values, k)])
        assert attempt_risk_estimate(values, k) == pytest.approx(brute)


def test_estimate_is_exactly_unbiased_for_binary_rollouts() -> None:
    n, k, rate = 6, 3, 0.3
    expectation = sum(
        math.comb(n, f) * rate**f * (1 - rate) ** (n - f)
        * attempt_risk_estimate([1.0] * f + [0.0] * (n - f), k)
        for f in range(n + 1)
    )
    assert expectation == pytest.approx(attempt_risk_from_rate(rate, k))


def test_plug_in_would_understate_the_risk() -> None:
    n, k, rate = 6, 3, 0.3
    plug_in = sum(
        math.comb(n, f) * rate**f * (1 - rate) ** (n - f) * attempt_risk_from_rate(f / n, k)
        for f in range(n + 1)
    )
    assert plug_in < attempt_risk_from_rate(rate, k) - 0.05


def test_estimate_rejects_too_few_rollouts_and_bad_inputs() -> None:
    with pytest.raises(ValueError, match="cannot estimate"):
        attempt_risk_estimate([0.0, 1.0], 3)
    with pytest.raises(ValueError, match="positive integer"):
        attempt_risk_estimate([0.0], 0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        attempt_risk_estimate([1.5, 0.0], 1)


def test_completions_bracket_the_estimate() -> None:
    lower, upper = attempt_risk_completions([0.0, 1.0, 0.0], 2, 2)
    assert lower == pytest.approx(attempt_risk_estimate([0.0, 1.0, 0.0, 0.0, 0.0], 2))
    assert upper == pytest.approx(attempt_risk_estimate([0.0, 1.0, 0.0, 1.0, 1.0], 2))
    assert lower <= upper


def test_atoms_preserve_the_weighted_mean_and_split_weight_evenly() -> None:
    atoms = attempt_atoms([0.2, 0.5], [0.3, 0.6], [0.25, 0.75], [9, 4], 2)
    assert atoms.atoms_per_prompt == (4, 2)
    assert float(np.sum(atoms.weights)) == pytest.approx(1.0)
    assert float(np.dot(atoms.weights, atoms.lower)) == pytest.approx(0.25 * 0.2 + 0.75 * 0.5)
    assert float(np.dot(atoms.weights, atoms.upper)) == pytest.approx(0.25 * 0.3 + 0.75 * 0.6)
    with pytest.raises(ValueError, match="at least k=3"):
        attempt_atoms([0.1], [0.1], [1.0], [2], 3)


@pytest.mark.parametrize(
    "engine", [weighted_bounded_kl_chernoff_mean_bounds, weighted_bounded_hoeffding_mean_bounds]
)
def test_atom_bounds_cover_the_worst_of_k_risk(engine: object) -> None:
    rng = np.random.default_rng(42)
    prompts, rollouts, k, alpha = 30, 8, 4, 0.05
    rates = rng.beta(0.4, 4.0, size=prompts)
    truth = float(np.mean([attempt_risk_from_rate(rate, k) for rate in rates]))
    weights = np.full(prompts, 1 / prompts)
    misses = 0
    replicates = 400
    for _ in range(replicates):
        estimates = [
            attempt_risk_estimate((rng.uniform(size=rollouts) < rate).astype(float).tolist(), k)
            for rate in rates
        ]
        atoms = attempt_atoms(estimates, estimates, weights, [rollouts] * prompts, k)
        bounds = engine(  # type: ignore[operator]
            atoms.lower, atoms.upper, atoms.weights,
            alpha_lower=alpha / 2, alpha_upper=alpha / 2, independence_profile="rollout",
        )
        misses += not bounds.lower <= truth <= bounds.upper
    assert misses / replicates <= alpha
