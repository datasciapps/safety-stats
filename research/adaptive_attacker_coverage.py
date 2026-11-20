"""Coverage of the adaptive-attacker lower bound, uniformly over time (#125).

An attacker that learns across behaviours has a success probability that depends on its own
history. The lower bound of ``stats/adaptive.py`` claims P(∃t: L_t > m̄_t) ≤ α, where m̄_t is
the running mean of the conditional success probabilities. This script draws attack streams
from several such attackers, recomputes the bound after every run, and counts the streams on
which it ever rises above m̄_t.

Attackers:

- ``learner``: starts at 0.05 and gains with each success, up to 0.6;
- ``discouraged``: starts at 0.6 and loses with each failure, down to 0.02;
- ``streaky``: 0.9 after a success, 0.1 after a failure;
- ``independent``: a constant 0.3, the case the binomial bounds already cover.

It also prints the bound beside the fixed-horizon Clopper-Pearson lower bound, which assumes
independent runs and is valid at one look only.

Run:
    python research/adaptive_attacker_coverage.py
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from inspect_safety_stats.stats.adaptive import adaptive_lower_bound
from inspect_safety_stats.stats.binomial import binomial_lower_bound

ALPHA = 0.05
HORIZON = 300
STREAMS = 20_000
SEED = 20261116

Policy = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def learner(successes: np.ndarray, last: np.ndarray, t: int) -> np.ndarray:
    return np.minimum(0.6, 0.05 + 0.05 * successes)


def discouraged(successes: np.ndarray, last: np.ndarray, t: int) -> np.ndarray:
    return np.maximum(0.02, 0.6 - 0.02 * (t - successes))


def streaky(successes: np.ndarray, last: np.ndarray, t: int) -> np.ndarray:
    return np.where(last, 0.9, 0.1)


def independent(successes: np.ndarray, last: np.ndarray, t: int) -> np.ndarray:
    return np.full(successes.shape, 0.3)


ATTACKERS: dict[str, Policy] = {
    "learner": learner,
    "discouraged": discouraged,
    "streaky": streaky,
    "independent": independent,
}


def bound_table(horizon: int) -> np.ndarray:
    """L[t, s] for every prefix length t ≤ horizon and success count s ≤ t."""

    table = np.zeros((horizon + 1, horizon + 1))
    for t in range(1, horizon + 1):
        for s in range(1, t + 1):
            table[t, s] = adaptive_lower_bound(s, t, alpha=ALPHA)
    return table


def miss_rate(policy: Policy, table: np.ndarray, rng: np.random.Generator) -> float:
    successes = np.zeros(STREAMS, dtype=int)
    last = np.zeros(STREAMS, dtype=bool)
    probability_sum = np.zeros(STREAMS)
    missed = np.zeros(STREAMS, dtype=bool)
    for t in range(1, HORIZON + 1):
        probability = policy(successes, last, t - 1)
        probability_sum += probability
        last = rng.random(STREAMS) < probability
        successes += last
        missed |= table[t, successes] > probability_sum / t
    return float(missed.mean())


def main() -> None:
    table = bound_table(HORIZON)
    rng = np.random.default_rng(SEED)
    print(f"one-sided alpha {ALPHA}, {STREAMS} streams, bound recomputed after each of "
          f"{HORIZON} runs")
    for name, policy in ATTACKERS.items():
        print(f"{name:12s} P(exists t: L_t > running mean) = {miss_rate(policy, table, rng):.4f}")
    print()
    print("successes/runs  adaptive  Clopper-Pearson (fixed horizon, independent runs)")
    for s, t in [(20, 50), (5, 50), (50, 200), (2, 20), (30, 300)]:
        print(f"{s:>5}/{t:<8} {table[t, s]:9.4f}  {binomial_lower_bound(s, t, alpha=ALPHA):9.4f}")


if __name__ == "__main__":
    main()
