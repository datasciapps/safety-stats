"""A one-sided anytime-valid upper bound on a fixed design's weighted mean (#126).

The items of a fixed design run in a random order fixed before the run, such as Inspect's
``sample_shuffle`` with a recorded seed. Item ``i`` has weight ``w_i`` and outcome ``Y_i`` in
[0, 1], its failure share over a number of rollouts fixed in advance. The estimand is the
design mean ``theta = sum_i w_i E[Y_i]``. The bound holds at every prefix of the order at once,

    P(exists t: theta > U_t) <= alpha,

so ``U_tau`` is valid at whatever time ``tau`` a stopping rule chose, however it chose it, as
long as the rule only decides when to stop and never which item runs next.

The bound has two steps, each at ``alpha / 2``:

1. **The realized design mean.** Given the outcomes, the order is a uniform permutation, so
   the prefix is a sample without replacement from the finite population of the ``N`` items.
   With ``Z_i = w_i Y_i / max w`` in [0, 1] and a candidate population mean ``mu`` of the
   ``Z_i``, the conditional mean of the next draw is
   ``m_t(mu) = (N mu - sum_{s<t} Z_s) / (N - t + 1)``, which the past determines. For each
   bet fraction ``f`` of the fixed prior of the betting bounds (``_betting_prior``), the
   capital ``prod_t (1 - f + f (1 - Z_t) / (1 - m_t(mu)))`` is then a nonnegative martingale
   with mean one when ``mu`` is the true population mean, and so is their fixed mixture.
   Ville's inequality bounds the chance that it ever reaches ``2 / alpha``. Every candidate the
   capital has reached it at by the current prefix is rejected for good (the running
   intersection), and a candidate that the observed outcomes rule out is rejected outright.
   Capital rises with ``mu``, so the rejected candidates form an upper ray and the bound is
   found by bisection.
2. **From the realized mean to theta.** The ``Y_i`` are independent across items (and, under
   the ``rollout`` profile, the rollouts within an item too), so the
   KL/Chernoff bound of the fixed-horizon engine (``bounded_kl_chernoff_upper_bound``) turns an
   upper bound on ``sum_i w_i Y_i`` into one on ``theta``. That bound rises with the realized
   mean, so step 1's bound can stand in for the unknown realized mean.

Unlike the draws with replacement of docs/sequential-design.md, this covers a permutation,
which is what Inspect runs. Derived in docs/sequential-design.md, section "Runs that optstop
stopped".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .direct_mean import _betting_prior, bounded_kl_chernoff_upper_bound

ANYTIME_KERNEL = "without_replacement_betting_mixture"
_BISECTION_STEPS = 64


@dataclass(frozen=True)
class AnytimeUpperBound:
    """A one-sided anytime-valid upper bound on a fixed design's weighted mean.

    ``draws`` items of the ``population`` were observed, in draw order. ``estimate`` is their
    weighted mean outcome, the weights renormalized over the observed items. ``realized_upper``
    bounds the design mean of the realized outcomes, ``sum_i w_i Y_i``, at ``alpha_sequence``;
    ``upper`` bounds ``theta`` at ``alpha``, with ``alpha_population`` spent on the step from
    the one to the other.
    """

    upper: float
    realized_upper: float
    estimate: float | None
    draws: int
    population: int
    alpha: float
    alpha_sequence: float
    alpha_population: float
    sequence_kernel: str
    population_kernel: str
    independent_rollouts: int


def _log_capital_path(scaled: np.ndarray, population: int, mean: float) -> np.ndarray:
    """Log mixture capital against population mean ``mean`` after each draw of ``scaled``."""

    draws = len(scaled)
    steps = np.arange(1, draws + 1, dtype=float)
    previous = np.concatenate(([0.0], np.cumsum(scaled)[:-1]))
    conditional = (population * mean - previous) / (population - steps + 1)
    room = 1.0 - conditional
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(
            scaled >= 1.0,
            0.0,
            np.where(room > 0, (1.0 - scaled) / room, math.inf),
        )
    fractions, log_prior = _betting_prior()
    with np.errstate(divide="ignore"):
        log_factors = np.log(1.0 - fractions[:, None] + fractions[:, None] * ratios[None, :])
    log_terms = log_prior[:, None] + np.cumsum(log_factors, axis=1)
    peak = np.max(log_terms, axis=0)
    finite_peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.where(
        np.isfinite(peak),
        finite_peak + np.log(np.sum(np.exp(log_terms - finite_peak), axis=0)),
        peak,
    )


def without_replacement_upper_bound(
    scaled: ArrayLike, population: int, *, alpha: float
) -> float:
    """Anytime-valid upper bound on the mean of a finite population of [0,1] values.

    ``scaled`` holds the first ``t`` values of a uniformly random order of the ``population``
    values. The bound is the running intersection of the betting sequence over the whole
    prefix, so it is the bound at ``t`` of a sequence that holds at every ``t`` at once.
    """

    z = _validated_values(scaled, "scaled values")
    if population < len(z) or population <= 0:
        raise ValueError("population must be positive and at least the number of draws")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if len(z) == 0:
        return 1.0
    # The unseen values are at most one each, so no larger mean is possible.
    cap = min(1.0, float((np.sum(z) + (population - len(z))) / population))
    threshold = math.log(1.0 / alpha)

    def rejected(mean: float) -> bool:
        return bool(np.max(_log_capital_path(z, population, mean)) >= threshold)

    if not rejected(cap):
        return cap
    # Mean 0 is never rejected: every factor is then at most one.
    inside, outside = 0.0, cap
    for _ in range(_BISECTION_STEPS):
        midpoint = (inside + outside) / 2
        if rejected(midpoint):
            outside = midpoint
        else:
            inside = midpoint
    return outside


def anytime_upper_bound(
    outcomes: ArrayLike,
    order: ArrayLike,
    weights: ArrayLike,
    *,
    alpha: float,
    independent_rollouts: int = 1,
) -> AnytimeUpperBound:
    """Anytime-valid upper bound on ``theta = sum_i w_i E[Y_i]`` after a prefix of draws.

    ``weights`` holds the weight of every item of the design, summing to one. ``order`` holds
    the indices into ``weights`` of the items observed so far, in draw order, and ``outcomes``
    their outcomes in [0, 1] in the same order. The draw order must be a uniformly random
    permutation fixed before any outcome, and only a prefix of it may be passed: an item left
    out in the middle, for example because it finished late, invalidates the bound.

    ``independent_rollouts`` is 1 when only items are independent. When each outcome is the
    failure share of ``r`` rollouts that are independent given the item (the ``rollout``
    independence profile), passing ``r`` lets step 2 count ``N r`` independent blocks of weight
    ``w_i / r``, as the fixed-horizon engine does.
    """

    y = _validated_values(outcomes, "outcomes")
    w = np.asarray(weights, dtype=float)
    index = np.asarray(order)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("weights must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not math.isclose(float(np.sum(w)), 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("weights must sum to one")
    if index.ndim != 1 or len(index) != len(y):
        raise ValueError("order and outcomes must be one-dimensional and aligned")
    if len(index) and not np.issubdtype(index.dtype, np.integer):
        raise TypeError("order must hold integer indices into weights")
    if np.any(index < 0) or np.any(index >= len(w)) or len(np.unique(index)) != len(index):
        raise ValueError("order must hold distinct indices into weights")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if isinstance(independent_rollouts, bool) or independent_rollouts < 1:
        raise ValueError("independent_rollouts must be a positive integer")

    alpha_sequence = alpha / 2
    alpha_population = alpha / 2
    population = len(w)
    maximum_weight = float(np.max(w))
    drawn = w[index] if len(index) else np.zeros(0)
    scaled = np.minimum(drawn * y / maximum_weight, 1.0)
    scaled_upper = without_replacement_upper_bound(scaled, population, alpha=alpha_sequence)
    realized_upper = min(1.0, scaled_upper * population * maximum_weight)
    upper, population_kernel = bounded_kl_chernoff_upper_bound(
        realized_upper,
        np.repeat(w / independent_rollouts, independent_rollouts),
        alpha=alpha_population,
    )
    drawn_total = float(np.sum(drawn))
    estimate = float(np.dot(drawn, y) / drawn_total) if drawn_total > 0 else None
    return AnytimeUpperBound(
        upper=upper,
        realized_upper=realized_upper,
        estimate=estimate,
        draws=len(index),
        population=population,
        alpha=alpha,
        alpha_sequence=alpha_sequence,
        alpha_population=alpha_population,
        sequence_kernel=ANYTIME_KERNEL,
        population_kernel=population_kernel,
        independent_rollouts=independent_rollouts,
    )


def _validated_values(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array
