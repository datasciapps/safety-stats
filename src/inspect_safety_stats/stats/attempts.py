"""Risk when an adversary gets ``k`` attempts at a prompt.

For a fixed prompt with per-rollout failure probability ``p``, the probability that at
least one of ``k`` independent attempts fails is ``1 - (1 - p)**k``. For graded harm in
[0, 1] the natural generalization is the expected worst outcome over ``k`` attempts,
``E[max(X_1, ..., X_k)]``, which reduces to the former for binary outcomes.

Estimation
----------
From ``n >= k`` rollouts the U-statistic with kernel ``max`` -- the average of the
maximum over all ``C(n, k)`` subsets -- is unbiased. With the rollouts sorted in
ascending order it is ``sum_i x_(i) * C(i - 1, k - 1) / C(n, k)``; for binary outcomes
with ``f`` failures that is ``1 - C(n - f, k) / C(n, k)``. The plug-in
``1 - (1 - f/n)**k`` is not used: the transform is concave, so by Jensen's inequality
the plug-in is biased downward and understates the risk.

Bounds
------
Hoeffding (1963, section 5) writes a U-statistic of order ``k`` as an average, over
permutations, of means of ``m = floor(n / k)`` independent kernel values. Because the
exponential is convex, the moment generating function of the U-statistic is at most that
of one such mean. Across independent prompts, the design-weighted sum of per-prompt
U-statistics is therefore dominated, in moment generating function, by a weighted sum of
independent [0, 1] atoms: ``m_j`` atoms of weight ``w_j / m_j`` for prompt ``j``. Every
Chernoff-type bound that uses only the weights and the observed mean -- the weighted
KL/Chernoff and Hoeffding engines -- applies to the U-statistic unchanged. The
empirical-Bernstein engine is excluded, because its variance estimate would need the
unobserved atoms.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _log_comb(n: int, r: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)


def attempt_risk_estimate(values: Sequence[float], k: int) -> float:
    """Unbiased estimate of the expected worst of ``k`` attempts from ``n >= k`` rollouts."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    data = np.sort(np.asarray(values, dtype=float))
    n = len(data)
    if n < k:
        raise ValueError(f"{n} rollouts cannot estimate the worst of {k} attempts")
    if np.any(data < 0) or np.any(data > 1):
        raise ValueError("attempt outcomes must lie in [0, 1]")
    if k == 1:
        return float(np.mean(data))
    total = _log_comb(n, k)
    estimate = 0.0
    for index in range(k, n + 1):
        value = float(data[index - 1])
        if value:
            estimate += value * math.exp(_log_comb(index - 1, k - 1) - total)
    return min(1.0, max(0.0, estimate))


def attempt_risk_completions(
    harms: Sequence[float], unclassified: int, k: int
) -> tuple[float, float]:
    """Estimates with every unclassified rollout completed as harmless, then as worst.

    The estimator is nondecreasing in each outcome, so the two completions bracket the
    estimate the complete data would have given.
    """

    lower = attempt_risk_estimate([*harms, *([0.0] * unclassified)], k)
    upper = attempt_risk_estimate([*harms, *([1.0] * unclassified)], k)
    return lower, upper


def attempt_risk_from_rate(rate: float, k: int) -> float:
    """``1 - (1 - rate)**k``: exact for binary outcomes, an upper bound for graded ones."""

    return 1.0 - (1.0 - min(max(rate, 0.0), 1.0)) ** k


@dataclass(frozen=True)
class AttemptAtoms:
    """Dominating independent atoms for a weighted sum of per-prompt U-statistics."""

    lower: np.ndarray
    upper: np.ndarray
    weights: np.ndarray
    atoms_per_prompt: tuple[int, ...]


def attempt_atoms(
    lower: Sequence[float],
    upper: Sequence[float],
    weights: Sequence[float],
    rollouts: Sequence[int],
    k: int,
) -> AttemptAtoms:
    """Expand per-prompt U-statistics into the atoms that dominate them.

    Prompt ``j`` with ``n_j`` rollouts contributes ``floor(n_j / k)`` atoms, each carrying
    the prompt's estimate and an equal share of its design weight. The weighted mean of
    the atoms equals the weighted mean of the U-statistics; only the weights matter to
    the KL/Chernoff and Hoeffding engines.
    """

    lows = np.asarray(lower, dtype=float)
    highs = np.asarray(upper, dtype=float)
    w = np.asarray(weights, dtype=float)
    counts = [int(value) for value in rollouts]
    if not len(lows) == len(highs) == len(w) == len(counts):
        raise ValueError("per-prompt inputs must be aligned")
    atoms = [count // k for count in counts]
    if any(count == 0 for count in atoms):
        short = min(counts)
        raise ValueError(
            f"every prompt needs at least k={k} rollouts for the worst-of-{k} risk; the "
            f"smallest has {short}"
        )
    return AttemptAtoms(
        lower=np.repeat(lows, atoms),
        upper=np.repeat(highs, atoms),
        weights=np.repeat(w / np.asarray(atoms, dtype=float), atoms),
        atoms_per_prompt=tuple(atoms),
    )
