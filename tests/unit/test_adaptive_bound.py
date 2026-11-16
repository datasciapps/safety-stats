"""The anytime-valid lower bound for attackers that adapt across behaviours (#125)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from inspect_safety_stats.stats.adaptive import (
    MIXTURE_LAMBDAS,
    adaptive_lower_bound,
    log_mixture_e_value,
)
from inspect_safety_stats.stats.binomial import binomial_lower_bound


def _bisected(successes: int, runs: int, alpha: float) -> float:
    """The same bound by plain bisection over the mixture, written out with math alone."""

    def e_value(mean: float) -> float:
        return _log_e_value_reference(successes, runs, mean)

    if e_value(0.0) < math.log(1 / alpha):
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2
        if e_value(middle) >= math.log(1 / alpha):
            low = middle
        else:
            high = middle
    return low


@pytest.mark.parametrize(
    ("successes", "runs"), [(1, 1), (3, 3), (2, 20), (5, 50), (20, 50), (50, 200), (299, 300)]
)
def test_matches_a_plain_bisection(successes: int, runs: int) -> None:
    assert adaptive_lower_bound(successes, runs, alpha=0.05) == pytest.approx(
        _bisected(successes, runs, 0.05), rel=0, abs=1e-10
    )


def test_twenty_successes_in_fifty() -> None:
    # The example in #125: about 0.209, against 0.283 for fixed-horizon Clopper-Pearson.
    bound = adaptive_lower_bound(20, 50, alpha=0.05)
    assert bound == pytest.approx(0.20966, rel=0, abs=5e-5)
    assert bound < binomial_lower_bound(20, 50, alpha=0.05)


def test_the_bound_never_excludes_a_mean_the_data_do_not_reject() -> None:
    # The root is lowered by the solver's tolerance, so the e-value at the bound reaches 1/alpha
    # and the bound itself is excluded, never a mean the data still allow below it.
    for successes, runs in [(2, 20), (20, 50), (50, 200)]:
        bound = adaptive_lower_bound(successes, runs, alpha=0.05)
        assert log_mixture_e_value(successes, runs, bound) >= math.log(20)
        assert log_mixture_e_value(successes, runs, bound + 1e-9) < math.log(20)


def test_no_success_excludes_nothing() -> None:
    assert adaptive_lower_bound(0, 1, alpha=0.05) == 0.0
    assert adaptive_lower_bound(0, 1000, alpha=0.05) == 0.0


def test_rises_with_successes_and_falls_with_runs() -> None:
    by_successes = [adaptive_lower_bound(s, 100, alpha=0.05) for s in range(101)]
    assert by_successes == sorted(by_successes)
    by_runs = [adaptive_lower_bound(10, t, alpha=0.05) for t in range(10, 200)]
    assert by_runs == sorted(by_runs, reverse=True)
    assert all(0.0 <= item < 1.0 for item in by_successes)


def test_a_smaller_alpha_gives_a_lower_bound() -> None:
    assert adaptive_lower_bound(20, 50, alpha=0.01) < adaptive_lower_bound(20, 50, alpha=0.05)


def test_stays_accurate_for_tiny_means() -> None:
    # log1p(mean * expm1(lambda)) keeps the e-value finite and exact for tiny means.
    assert log_mixture_e_value(1, 10**6, 1e-15) == pytest.approx(
        _log_e_value_reference(1, 10**6, 1e-15), rel=1e-12, abs=0
    )


def _log_e_value_reference(successes: int, runs: int, mean: float) -> float:
    terms = [
        lam * successes - runs * math.log1p(mean * math.expm1(lam))
        for lam in (float(item) for item in MIXTURE_LAMBDAS)
    ]
    top = max(terms)
    return top + math.log(sum(math.exp(term - top) for term in terms) / len(terms))


@pytest.mark.parametrize(
    ("successes", "runs", "alpha", "error"),
    [
        (True, 3, 0.05, TypeError),
        (1, 3.0, 0.05, TypeError),
        (1, 0, 0.05, ValueError),
        (4, 3, 0.05, ValueError),
        (-1, 3, 0.05, ValueError),
        (1, 3, 0.0, ValueError),
        (1, 3, 1.0, ValueError),
    ],
)
def test_rejects_invalid_input(successes: object, runs: object, alpha: float, error: type) -> None:
    with pytest.raises(error):
        adaptive_lower_bound(successes, runs, alpha=alpha)  # type: ignore[arg-type]


def test_covers_an_attacker_that_learns_from_its_own_successes_uniformly_over_time() -> None:
    # The attacker's success probability grows with every success, so runs are dependent. The
    # bound is recomputed after each run; it may rise above the running mean of the conditional
    # success probabilities on at most alpha of streams. research/adaptive_attacker_coverage.py
    # runs the same check at a larger scale and with more attackers.
    alpha, horizon, streams = 0.05, 80, 4000
    table = np.zeros((horizon + 1, horizon + 1))
    for t in range(1, horizon + 1):
        for s in range(1, t + 1):
            table[t, s] = adaptive_lower_bound(s, t, alpha=alpha)
    rng = np.random.default_rng(125)
    successes = np.zeros(streams, dtype=int)
    probability_sum = np.zeros(streams)
    missed = np.zeros(streams, dtype=bool)
    for t in range(1, horizon + 1):
        probability = np.minimum(0.7, 0.1 + 0.1 * successes)
        probability_sum += probability
        successes += rng.random(streams) < probability
        missed |= table[t, successes] > probability_sum / t
    # The bound is conservative here, so the observed rate sits well below alpha; the margin
    # is several Monte Carlo standard errors (about 0.0034 at this size).
    assert missed.mean() <= alpha
    # And it is not vacuous: it excludes something on most streams by the end.
    assert np.mean(table[horizon, successes] > 0) > 0.9
