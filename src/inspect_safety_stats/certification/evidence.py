from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binom

from inspect_safety_stats.stats.binomial import (
    binomial_lower_bound,
    binomial_upper_bound,
)
from inspect_safety_stats.stats.direct_mean import (
    DirectMeanBounds,
    IndependenceUnit,
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
    weighted_empirical_bernstein_mean_bounds,
    weighted_hoeffding_mean_bounds,
    weighted_kl_chernoff_mean_bounds,
)
from inspect_safety_stats.stats.independence import BoundedBlocks


def exact_support_p_value(failures: int, trials: int, threshold: float) -> float:
    """Exact lower-tail p-value for H0: p >= threshold."""

    _validate_threshold_counts(failures, trials, threshold)
    return float(binom.cdf(failures, trials, threshold))


def exact_rejection_p_value(failures: int, trials: int, threshold: float) -> float:
    """Exact upper-tail p-value for H0: p <= threshold."""

    _validate_threshold_counts(failures, trials, threshold)
    return float(binom.sf(failures - 1, trials, threshold))


def _validate_threshold_counts(failures: int, trials: int, threshold: float) -> None:
    if isinstance(failures, bool) or isinstance(trials, bool):
        raise TypeError("counts must be integers, not booleans")
    if not isinstance(failures, int) or not isinstance(trials, int):
        raise TypeError("counts must be integers")
    if trials <= 0 or failures < 0 or failures > trials:
        raise ValueError("counts must satisfy 0 <= failures <= trials and trials > 0")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")


def exact_threshold_bounds(
    failures: int,
    unclassified: int,
    trials: int,
    *,
    support_alpha: float,
    rejection_alpha: float,
) -> tuple[float, float]:
    if failures + unclassified > trials:
        raise ValueError("failures plus unclassified outcomes cannot exceed trials")
    lower = binomial_lower_bound(failures, trials, alpha=rejection_alpha)
    upper = binomial_upper_bound(failures + unclassified, trials, alpha=support_alpha)
    return lower, upper


def mean_bounds(
    method: str,
    failures: ArrayLike,
    unclassified: ArrayLike,
    trials: ArrayLike,
    weights: ArrayLike,
    *,
    support_alpha: float,
    rejection_alpha: float,
    independence_unit: IndependenceUnit,
) -> DirectMeanBounds:
    function = {
        "weighted_hoeffding": weighted_hoeffding_mean_bounds,
        "weighted_empirical_bernstein": weighted_empirical_bernstein_mean_bounds,
        "weighted_kl_chernoff": weighted_kl_chernoff_mean_bounds,
    }.get(method)
    if function is None:
        raise ValueError(f"unsupported certification mean method {method!r}")
    return function(
        failures,
        unclassified,
        trials,
        weights,
        alpha_lower=rejection_alpha,
        alpha_upper=support_alpha,
        independence_unit=independence_unit,
    )


def block_mean_bounds(
    method: str,
    blocks: BoundedBlocks,
    *,
    support_alpha: float,
    rejection_alpha: float,
    independence_profile: str,
) -> DirectMeanBounds:
    function = {
        "weighted_hoeffding": weighted_bounded_hoeffding_mean_bounds,
        "weighted_empirical_bernstein": weighted_bounded_empirical_bernstein_mean_bounds,
        "weighted_kl_chernoff": weighted_bounded_kl_chernoff_mean_bounds,
    }.get(method)
    if function is None:
        raise ValueError(f"unsupported certification mean method {method!r}")
    return function(
        blocks.lower,
        blocks.upper,
        blocks.weights,
        alpha_lower=rejection_alpha,
        alpha_upper=support_alpha,
        independence_profile=independence_profile,
    )


def block_mean_support_p_value(
    method: str,
    blocks: BoundedBlocks,
    *,
    threshold: float,
    independence_profile: str,
) -> float:
    return inverted_bound_p_value(
        lambda alpha: (
            block_mean_bounds(
                method,
                blocks,
                support_alpha=alpha,
                rejection_alpha=0.5,
                independence_profile=independence_profile,
            ).upper
            <= threshold
        )
    )


def block_mean_rejection_p_value(
    method: str,
    blocks: BoundedBlocks,
    *,
    threshold: float,
    independence_profile: str,
) -> float:
    return inverted_bound_p_value(
        lambda alpha: (
            block_mean_bounds(
                method,
                blocks,
                support_alpha=0.5,
                rejection_alpha=alpha,
                independence_profile=independence_profile,
            ).lower
            > threshold
        )
    )


def inverted_bound_p_value(bound_crosses: Callable[[float], bool]) -> float:
    """Invert a nested one-sided confidence bound into a conservative p-value."""

    lower = 1e-12
    upper = 1.0 - 1e-12
    if not bound_crosses(upper):
        return 1.0
    if bound_crosses(lower):
        return lower
    for _ in range(70):
        midpoint = (lower + upper) / 2
        if bound_crosses(midpoint):
            upper = midpoint
        else:
            lower = midpoint
    return min(1.0, max(0.0, upper))


def mean_support_p_value(
    method: str,
    failures: ArrayLike,
    unclassified: ArrayLike,
    trials: ArrayLike,
    weights: ArrayLike,
    *,
    threshold: float,
    independence_unit: IndependenceUnit,
) -> float:
    def crosses(alpha: float) -> bool:
        result = mean_bounds(
            method,
            failures,
            unclassified,
            trials,
            weights,
            support_alpha=alpha,
            rejection_alpha=0.5,
            independence_unit=independence_unit,
        )
        return result.upper <= threshold

    return inverted_bound_p_value(crosses)


def mean_rejection_p_value(
    method: str,
    failures: ArrayLike,
    unclassified: ArrayLike,
    trials: ArrayLike,
    weights: ArrayLike,
    *,
    threshold: float,
    independence_unit: IndependenceUnit,
) -> float:
    def crosses(alpha: float) -> bool:
        result = mean_bounds(
            method,
            failures,
            unclassified,
            trials,
            weights,
            support_alpha=0.5,
            rejection_alpha=alpha,
            independence_unit=independence_unit,
        )
        return result.lower > threshold

    return inverted_bound_p_value(crosses)


def adjusted_bonferroni_p_value(raw_p_value: float, normalized_weight: float) -> float:
    if not math.isfinite(raw_p_value) or not 0 <= raw_p_value <= 1:
        raise ValueError("raw p-value must be finite and in [0, 1]")
    if not math.isfinite(normalized_weight) or not 0 < normalized_weight <= 1:
        raise ValueError("normalized allocation weight must be in (0, 1]")
    return min(1.0, raw_p_value / normalized_weight)


def normalized_weights(values: list[float], *, weighted: bool) -> list[float]:
    if not values:
        return []
    if not weighted:
        return [1 / len(values)] * len(values)
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("allocation weights must be finite and positive")
    array /= np.sum(array)
    return [float(value) for value in array]
