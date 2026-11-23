"""A single-valued risk bound that needs no alpha (experimental, #127).

For a family of one-sided upper bounds U_delta, each with P(R > U_delta) <= delta for its own
fixed delta and non-increasing in delta,

    M = inf over delta in (0, 1) of [U_delta + delta]

satisfies P(M < R) <= R. If M < R, some delta < R has U_delta < R - delta < R, and since
U_R <= U_delta the level-R bound misses R, which happens with probability at most R.

The infimum is taken over a fixed logarithmic grid of delta and refined around the grid's
minimiser. Every value returned is U_delta + delta at an attained delta, so it is never below
the exact infimum: an imprecise minimiser makes the result more conservative, never less.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .binomial import binomial_upper_bound

SINGLE_VALUED_BOUND_GUARANTEE = (
    "P(M < R) <= R for this node alone: the miss probability equals the unknown risk, "
    "with no alpha and no multiplicity adjustment (experimental)"
)

# Fixed, so the result depends only on the bound family: never on the data.
_GRID = np.exp(np.linspace(math.log(1e-12), math.log(1 - 1e-9), 801))
# Tolerance for the monotonicity check, against floating-point noise in the bound families.
_MONOTONE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SingleValuedBound:
    value: float
    delta: float
    upper_at_delta: float


def single_valued_bound(upper_bound_at: Callable[[float], float]) -> SingleValuedBound:
    """M = inf_delta [U_delta + delta] for a non-increasing family of one-sided upper bounds.

    ``upper_bound_at(delta)`` must return a bound with P(R > U_delta) <= delta for each fixed
    delta in (0, 1). A family that rises with delta on the evaluation grid is rejected,
    because the guarantee P(M < R) <= R needs U_R <= U_delta for every delta < R.
    """

    uppers = np.asarray([upper_bound_at(float(delta)) for delta in _GRID], dtype=float)
    if not np.all(np.isfinite(uppers)):
        raise ValueError("the upper-bound family returned a non-finite bound")
    if np.any(np.diff(uppers) > _MONOTONE_TOLERANCE):
        raise ValueError(
            "the upper-bound family increases with delta, so the single-valued bound "
            "has no guarantee"
        )
    totals = uppers + _GRID
    best = int(np.argmin(totals))
    candidates = [(float(totals[best]), float(_GRID[best]), float(uppers[best]))]
    low = math.log(_GRID[max(best - 1, 0)])
    high = math.log(_GRID[min(best + 1, len(_GRID) - 1)])
    if high > low:
        refined = minimize_scalar(
            lambda log_delta: upper_bound_at(math.exp(log_delta)) + math.exp(log_delta),
            bounds=(low, high),
            method="bounded",
            options={"xatol": 1e-10},
        )
        delta = math.exp(float(refined.x))
        upper = float(upper_bound_at(delta))
        candidates.append((upper + delta, delta, upper))
    value, delta, upper = min(candidates)
    return SingleValuedBound(value=min(1.0, value), delta=delta, upper_at_delta=upper)


def clopper_pearson_single_valued_bound(failures: int, trials: int) -> SingleValuedBound:
    """M = min_p [p + P_p(S <= s)] for s failures in n Bernoulli trials (experimental)."""

    return single_valued_bound(
        lambda delta: binomial_upper_bound(failures, trials, alpha=delta)
    )
