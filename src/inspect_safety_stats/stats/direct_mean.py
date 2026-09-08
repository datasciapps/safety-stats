from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from .risk import worst_of_k_shares


@dataclass(frozen=True)
class DirectMeanBounds:
    """One-sided bounds on a fixed-design weighted mean failure risk, with their ingredients.

    ``lower`` and ``upper`` are the bounds, each at its own one-sided level ``alpha_lower`` or
    ``alpha_upper``. ``completion_estimate_lower`` and ``completion_estimate_upper`` are the
    weighted estimates the bounds start from: they differ only when some outcomes are
    unclassified, which the lower side counts as safe and the upper side as failures.
    ``half_width_*`` is the distance from each estimate to its bound before clipping to
    ``[0, 1]``. ``effective_sample_size`` is ``1 / concentration_variance``. The ``variance_*``
    fields are set by the empirical-Bernstein method and ``chernoff_kernel`` by the
    KL/Chernoff method. Derived in docs/statistical-methods.md, section 2.4.10.
    """

    lower: float
    upper: float
    completion_estimate_lower: float
    completion_estimate_upper: float
    half_width_lower: float
    half_width_upper: float
    concentration_variance: float
    effective_sample_size: float
    alpha_lower: float
    alpha_upper: float
    method: str = "weighted_hoeffding"
    independence_unit: str = "rollout"
    variance_upper_lower: float | None = None
    variance_upper_upper: float | None = None
    maximum_increment: float | None = None
    chernoff_kernel: str | None = None


IndependenceUnit = Literal["rollout", "prompt_cluster"]


def _validate_bounded_inputs(
    completion_lower: ArrayLike,
    completion_upper: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(completion_lower, dtype=float)
    upper = np.asarray(completion_upper, dtype=float)
    w = np.asarray(weights, dtype=float)
    if any(array.ndim != 1 for array in (lower, upper, w)):
        raise ValueError("bounded observations and weights must be one-dimensional")
    if len(w) == 0 or not (len(lower) == len(upper) == len(w)):
        raise ValueError("bounded observations and weights must be nonempty and aligned")
    if not all(np.all(np.isfinite(array)) for array in (lower, upper, w)):
        raise ValueError("bounded observations and weights must be finite")
    if np.any(lower < 0) or np.any(upper > 1) or np.any(lower > upper):
        raise ValueError("bounded completions must satisfy 0 <= lower <= upper <= 1")
    if np.any(w < 0) or not math.isclose(float(np.sum(w)), 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("bounded-observation weights must be nonnegative and sum to one")
    if not 0 < alpha_lower < 1 or not 0 < alpha_upper < 1:
        raise ValueError("alpha levels must be in (0, 1)")
    return lower, upper, w


def weighted_bounded_hoeffding_mean_bounds(
    completion_lower: ArrayLike,
    completion_upper: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    independence_profile: str,
) -> DirectMeanBounds:
    """Hoeffding bounds for independent, possibly heterogeneous [0,1] blocks."""

    lower_values, upper_values, w = _validate_bounded_inputs(
        completion_lower,
        completion_upper,
        weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )
    estimate_lower = float(np.dot(w, lower_values))
    estimate_upper = float(np.dot(w, upper_values))
    c2 = float(np.sum(np.square(w)))
    maximum_increment = float(np.max(w))
    lower_width = math.sqrt(0.5 * c2 * math.log(1 / alpha_lower))
    upper_width = math.sqrt(0.5 * c2 * math.log(1 / alpha_upper))
    return DirectMeanBounds(
        lower=max(0.0, estimate_lower - lower_width),
        upper=min(1.0, estimate_upper + upper_width),
        completion_estimate_lower=estimate_lower,
        completion_estimate_upper=estimate_upper,
        half_width_lower=lower_width,
        half_width_upper=upper_width,
        concentration_variance=c2,
        effective_sample_size=1 / c2,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_unit=independence_profile,
        maximum_increment=maximum_increment,
    )


def weighted_bounded_empirical_bernstein_mean_bounds(
    completion_lower: ArrayLike,
    completion_upper: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    independence_profile: str,
) -> DirectMeanBounds:
    """Variance-adaptive bounds for independent [0,1]-valued blocks."""

    lower_values, upper_values, w = _validate_bounded_inputs(
        completion_lower,
        completion_upper,
        weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )
    c2 = float(np.sum(np.square(w)))
    c4 = float(np.sum(np.power(w, 4)))
    maximum_increment = float(np.max(w))
    # A block's true value can lie anywhere between its two completions, and the variance
    # bound must hold for every such value. Q+ grows and Q- shrinks as values rise, so each
    # comes from the completion that maximises it, for both endpoints. Taking both from the
    # endpoint's own completion let the all-failure upper endpoint fall below the upper
    # endpoint of the true values.
    positive_second = float(np.dot(np.square(w), np.square(upper_values)))
    negative_second = float(np.dot(np.square(w), np.square(1 - lower_values)))

    def endpoint(values: np.ndarray, alpha: float) -> tuple[float, float, float]:
        estimate = float(np.dot(w, values))
        tail_alpha = alpha / 2
        variance_padding = math.sqrt(0.5 * c4 * math.log(4 / alpha))
        variance_upper = min(
            0.25 * c2,
            positive_second + variance_padding,
            negative_second + variance_padding,
        )
        tail_log = math.log(1 / tail_alpha)
        linear_term = maximum_increment * tail_log / 3
        width = linear_term + math.sqrt(2 * variance_upper * tail_log + linear_term**2)
        return estimate, width, variance_upper

    estimate_lower, lower_width, variance_lower = endpoint(lower_values, alpha_lower)
    estimate_upper, upper_width, variance_upper = endpoint(upper_values, alpha_upper)
    return DirectMeanBounds(
        lower=max(0.0, estimate_lower - lower_width),
        upper=min(1.0, estimate_upper + upper_width),
        completion_estimate_lower=estimate_lower,
        completion_estimate_upper=estimate_upper,
        half_width_lower=lower_width,
        half_width_upper=upper_width,
        concentration_variance=c2,
        effective_sample_size=1 / c2,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        method="weighted_empirical_bernstein",
        independence_unit=independence_profile,
        variance_upper_lower=variance_lower,
        variance_upper_upper=variance_upper,
        maximum_increment=maximum_increment,
    )


def weighted_bounded_kl_chernoff_mean_bounds(
    completion_lower: ArrayLike,
    completion_upper: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    independence_profile: str,
) -> DirectMeanBounds:
    """KL/Chernoff bounds for independent [0,1]-valued blocks."""

    lower_values, upper_values, w = _validate_bounded_inputs(
        completion_lower,
        completion_upper,
        weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )
    estimate_lower = float(np.dot(w, lower_values))
    estimate_upper = float(np.dot(w, upper_values))
    c2 = float(np.sum(np.square(w)))
    maximum_increment = float(np.max(w))
    equal_weights = bool(np.allclose(w, np.full_like(w, 1 / len(w)), rtol=1e-10, atol=1e-14))
    if equal_weights:
        kernel = "bernoulli_kl_equal_atomic_weights"
        lower_target = math.log(1 / alpha_lower) / len(w)
        upper_target = math.log(1 / alpha_upper) / len(w)
        divergence = _bernoulli_kl
    else:
        kernel = "weighted_poisson_chernoff_max_increment"
        lower_target = maximum_increment * math.log(1 / alpha_lower)
        upper_target = maximum_increment * math.log(1 / alpha_upper)
        divergence = _poisson_deviance
    lower = _invert_divergence(
        estimate_lower, lower_target, direction="lower", divergence=divergence
    )
    upper = _invert_divergence(
        estimate_upper, upper_target, direction="upper", divergence=divergence
    )
    return DirectMeanBounds(
        lower=lower,
        upper=upper,
        completion_estimate_lower=estimate_lower,
        completion_estimate_upper=estimate_upper,
        half_width_lower=estimate_lower - lower,
        half_width_upper=upper - estimate_upper,
        concentration_variance=c2,
        effective_sample_size=1 / c2,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        method="weighted_kl_chernoff",
        independence_unit=independence_profile,
        maximum_increment=maximum_increment,
        chernoff_kernel=kernel,
    )


def _validate_inputs(
    failures: ArrayLike,
    unclassified: ArrayLike,
    trials: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(failures)
    u = np.asarray(unclassified)
    n = np.asarray(trials)
    w = np.asarray(weights, dtype=float)
    if any(array.ndim != 1 for array in (x, u, n, w)):
        raise ValueError("counts and weights must be one-dimensional")
    if not (len(x) == len(u) == len(n) == len(w)) or len(w) == 0:
        raise ValueError("counts and weights must be nonempty arrays of equal length")
    if not all(np.issubdtype(array.dtype, np.integer) for array in (x, u, n)):
        raise TypeError("failures, unclassified outcomes, and trials must be integers")
    if np.any(n <= 0):
        raise ValueError("trials must be positive")
    if np.any(x < 0) or np.any(u < 0) or np.any(x + u > n):
        raise ValueError("counts must satisfy 0 <= failures + unclassified <= trials")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not math.isclose(float(np.sum(w)), 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("weights must sum to one")
    if not 0 < alpha_lower < 1 or not 0 < alpha_upper < 1:
        raise ValueError("alpha levels must be in (0, 1)")
    return x, u, n, w


def _concentration_geometry(
    trials: np.ndarray,
    weights: np.ndarray,
    independence_unit: IndependenceUnit,
) -> tuple[float, float, float]:
    n = trials.astype(float)
    if independence_unit == "rollout":
        return (
            float(np.sum(np.square(weights) / n)),
            float(np.sum(np.power(weights, 4) / np.power(n, 3))),
            float(np.max(weights / n)),
        )
    return (
        float(np.sum(np.square(weights))),
        float(np.sum(np.power(weights, 4))),
        float(np.max(weights)),
    )


def _completed_observations(
    counts: np.ndarray,
    trials: np.ndarray,
    weights: np.ndarray,
    independence_unit: IndependenceUnit,
) -> tuple[float, float, float]:
    n = trials.astype(float)
    y = counts.astype(float)
    estimate = float(np.dot(weights, y / n))
    if independence_unit == "rollout":
        positive_second = float(np.sum(np.square(weights) * y / np.square(n)))
        negative_second = float(np.sum(np.square(weights) * (n - y) / np.square(n)))
    else:
        rates = y / n
        positive_second = float(np.dot(np.square(weights), np.square(rates)))
        negative_second = float(np.dot(np.square(weights), np.square(1 - rates)))
    return estimate, positive_second, negative_second


def _variance_centerings(variance_centers: ArrayLike | None, weights: np.ndarray) -> np.ndarray:
    """Validated centerings, one row of per-prompt centers each.

    The default rows 0 and 1 are the uncentered proxies Q+ and Q-.
    """

    if variance_centers is None:
        return np.stack([np.zeros_like(weights), np.ones_like(weights)])
    shape_message = "variance_centers must have one center per prompt, in one row per centering"
    try:
        centers = np.atleast_2d(np.asarray(variance_centers, dtype=float))
    except ValueError as error:  # ragged rows
        raise ValueError(shape_message) from error
    if centers.ndim != 2 or centers.shape[1:] != weights.shape:
        raise ValueError(shape_message)
    if not np.all(np.isfinite(centers)) or np.any(centers < 0) or np.any(centers > 1):
        raise ValueError("variance_centers must lie in [0, 1]")
    return centers


def _centered_second_moment(
    failures: np.ndarray,
    unclassified: np.ndarray,
    trials: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    independence_unit: IndependenceUnit,
) -> tuple[float, float]:
    """Observed second moment about fixed centers, and its Hoeffding range term.

    For any fixed ``c``, ``Var(Y) <= E[(Y - c)^2]``, so the second moment estimates an upper
    bound on the variance of the weighted mean. It is taken at its largest over unknown
    completions: an unknown outcome contributes whichever completion lies farther from its
    center. Each term of the sum lies in ``[0, a^2 r]`` with ``r = max(c^2, (1 - c)^2)``, so the
    sum of the squared term ranges is the fourth-power sum weighted by ``r^2``. For centers 0
    and 1, ``r = 1`` and these are exactly Q+, Q- and the unweighted fourth-power sum.
    """

    n = trials.astype(float)
    x = failures.astype(float)
    u = unclassified.astype(float)
    far = np.maximum(np.square(centers), np.square(1 - centers))
    if independence_unit == "rollout":
        per_prompt = x * np.square(1 - centers) + (n - x - u) * np.square(centers) + u * far
        second = float(np.sum(np.square(weights) * per_prompt / np.square(n)))
        ranges = float(np.sum(np.power(weights, 4) / np.power(n, 3) * np.square(far)))
        return second, ranges
    low = np.square(x / n - centers)
    high = np.square((x + u) / n - centers)
    second = float(np.dot(np.square(weights), np.maximum(low, high)))
    ranges = float(np.sum(np.power(weights, 4) * np.square(far)))
    return second, ranges


def _bernoulli_kl(observed: float, mean: float) -> float:
    """Binary relative entropy ``kl(observed || mean)`` with stable endpoints."""

    if observed <= 0:
        return -math.log1p(-mean) if mean < 1 else math.inf
    if observed >= 1:
        return -math.log(mean) if mean > 0 else math.inf
    if mean <= 0 or mean >= 1:
        return math.inf
    return observed * math.log(observed / mean) + (1 - observed) * math.log(
        (1 - observed) / (1 - mean)
    )


def _poisson_deviance(observed: float, mean: float) -> float:
    """Poisson relative entropy ``observed*log(observed/mean)-observed+mean``."""

    if observed <= 0:
        return mean
    if mean <= 0:
        return math.inf
    return observed * math.log(observed / mean) - observed + mean


def _invert_divergence(
    observed: float,
    target: float,
    *,
    direction: Literal["lower", "upper"],
    divergence: Callable[[float, float], float],
) -> float:
    """Invert a convex divergence in its mean argument by monotone bisection."""

    if direction == "lower":
        if observed <= 0:
            return 0.0
        outside, inside = 0.0, observed
        for _ in range(80):
            midpoint = (outside + inside) / 2
            if divergence(observed, midpoint) > target:
                outside = midpoint
            else:
                inside = midpoint
        return (outside + inside) / 2
    if observed >= 1:
        return 1.0
    inside, outside = observed, 1.0
    for _ in range(80):
        midpoint = (inside + outside) / 2
        if divergence(observed, midpoint) > target:
            outside = midpoint
        else:
            inside = midpoint
    return (inside + outside) / 2


def weighted_kl_chernoff_mean_bounds(
    failures: ArrayLike,
    unclassified: ArrayLike,
    trials: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    independence_unit: IndependenceUnit = "rollout",
) -> DirectMeanBounds:
    """Chernoff intervals for a fixed weighted mean of failure probabilities.

    For equal atomic weights, the endpoints invert the Bernoulli KL exponent.
    This remains valid for independent, non-identically distributed Bernoulli
    rollouts and for independent bounded prompt-cluster rates. For unequal atomic
    weights, a weighted Poisson-Chernoff exponent based on the largest increment
    supplies a rigorous fallback. Unclassified outcomes are completed as safe for
    the lower endpoint and as failures for the upper endpoint.
    """

    x, u, n, w = _validate_inputs(
        failures,
        unclassified,
        trials,
        weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )
    concentration_variance, _, maximum_increment = _concentration_geometry(n, w, independence_unit)
    completion_lower, _, _ = _completed_observations(x, n, w, independence_unit)
    completion_upper, _, _ = _completed_observations(x + u, n, w, independence_unit)

    atomic_weights = w / n if independence_unit == "rollout" else w
    atomic_count = int(np.sum(n)) if independence_unit == "rollout" else len(w)
    equal_atomic_weights = bool(
        np.allclose(
            atomic_weights,
            np.full_like(atomic_weights, 1 / atomic_count),
            rtol=1e-10,
            atol=1e-14,
        )
    )
    if equal_atomic_weights:
        kernel = "bernoulli_kl_equal_atomic_weights"
        lower_target = math.log(1 / alpha_lower) / atomic_count
        upper_target = math.log(1 / alpha_upper) / atomic_count
        lower = _invert_divergence(
            completion_lower,
            lower_target,
            direction="lower",
            divergence=_bernoulli_kl,
        )
        upper = _invert_divergence(
            completion_upper,
            upper_target,
            direction="upper",
            divergence=_bernoulli_kl,
        )
    else:
        kernel = "weighted_poisson_chernoff_max_increment"
        lower_target = maximum_increment * math.log(1 / alpha_lower)
        upper_target = maximum_increment * math.log(1 / alpha_upper)
        lower = _invert_divergence(
            completion_lower,
            lower_target,
            direction="lower",
            divergence=_poisson_deviance,
        )
        upper = _invert_divergence(
            completion_upper,
            upper_target,
            direction="upper",
            divergence=_poisson_deviance,
        )

    return DirectMeanBounds(
        lower=lower,
        upper=upper,
        completion_estimate_lower=completion_lower,
        completion_estimate_upper=completion_upper,
        half_width_lower=completion_lower - lower,
        half_width_upper=upper - completion_upper,
        concentration_variance=concentration_variance,
        effective_sample_size=1 / concentration_variance,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        method="weighted_kl_chernoff",
        independence_unit=independence_unit,
        maximum_increment=maximum_increment,
        chernoff_kernel=kernel,
    )


def weighted_hoeffding_mean_bounds(
    failures: ArrayLike,
    unclassified: ArrayLike,
    trials: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    independence_unit: IndependenceUnit = "rollout",
) -> DirectMeanBounds:
    """Finite-sample bounds for a fixed weighted mean of Bernoulli probabilities.

    Each attempted rollout for prompt ``j`` contributes ``w_j / n_j``. The
    construction assumes these rollout contributions are mutually independent.
    Unclassified outcomes are completed as safe for the lower endpoint and as
    failures for the upper endpoint.
    """

    x, u, n, w = _validate_inputs(
        failures,
        unclassified,
        trials,
        weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )

    x_float = x.astype(float)
    u_float = u.astype(float)
    n_float = n.astype(float)
    completion_lower = float(np.dot(w, x_float / n_float))
    completion_upper = float(np.dot(w, (x_float + u_float) / n_float))
    concentration_variance, _, maximum_increment = _concentration_geometry(n, w, independence_unit)
    lower_width = math.sqrt(0.5 * concentration_variance * math.log(1 / alpha_lower))
    upper_width = math.sqrt(0.5 * concentration_variance * math.log(1 / alpha_upper))
    effective_sample_size = 1 / concentration_variance
    return DirectMeanBounds(
        lower=max(0.0, completion_lower - lower_width),
        upper=min(1.0, completion_upper + upper_width),
        completion_estimate_lower=completion_lower,
        completion_estimate_upper=completion_upper,
        half_width_lower=lower_width,
        half_width_upper=upper_width,
        concentration_variance=concentration_variance,
        effective_sample_size=effective_sample_size,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_unit=independence_unit,
        maximum_increment=maximum_increment,
    )


def weighted_empirical_bernstein_mean_bounds(
    failures: ArrayLike,
    unclassified: ArrayLike,
    trials: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    independence_unit: IndependenceUnit = "rollout",
    variance_centers: ArrayLike | None = None,
) -> DirectMeanBounds:
    """Variance-adaptive finite-sample bounds for a fixed weighted Bernoulli mean.

    A separate Hoeffding bound estimates an upper variance proxy from the observed
    squared outcomes and squared complements. The remaining endpoint error budget
    is used in Bernstein's inequality. This remains valid for heterogeneous prompt
    probabilities under the declared independence unit.

    ``variance_centers`` holds one or more centerings, each a row with one center in [0, 1]
    per prompt, fixed by the design and never by the data. Each centering gives a proxy, the
    second moment about its centers, and the bound uses the smallest, so half of the endpoint
    error budget is split equally across the centerings. The default, centerings 0 and 1,
    gives the second moments of the outcomes and of their complements. Callers whose
    observations sit near different ends of [0, 1] by construction, such as the two arms of a
    risk difference, pass centers per prompt: about 0 or 1 alone, their second moment is
    large even when every observation is nearly constant. A one-dimensional array is a single
    centering.
    """

    x, u, n, w = _validate_inputs(
        failures,
        unclassified,
        trials,
        weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )
    concentration_variance, _, maximum_increment = _concentration_geometry(
        n, w, independence_unit
    )
    # Unclassified outcomes can complete to any count from x to x + u, and the variance bound
    # must hold for every completion. Each proxy therefore takes every unknown at the
    # completion farther from its center, for both endpoints. Taking the proxies from each
    # endpoint's own completion let the all-failure upper endpoint fall below the upper
    # endpoint of the true labels.
    proxies = [
        _centered_second_moment(x, u, n, w, centers, independence_unit)
        for centers in _variance_centerings(variance_centers, w)
    ]

    def endpoint(counts: np.ndarray, alpha: float) -> tuple[float, float, float]:
        estimate, _, _ = _completed_observations(counts, n, w, independence_unit)
        tail_alpha = alpha / 2
        # Half the budget goes to the Bernstein tail, the other half is split across the
        # proxies. With the two default proxies this is alpha / 4 each.
        variance_component_alpha = alpha / (2 * len(proxies))
        variance_upper = 0.25 * concentration_variance
        for second, ranges in proxies:
            variance_padding = math.sqrt(0.5 * ranges * math.log(1 / variance_component_alpha))
            variance_upper = min(variance_upper, second + variance_padding)
        tail_log = math.log(1 / tail_alpha)
        linear_term = maximum_increment * tail_log / 3
        width = linear_term + math.sqrt(2 * variance_upper * tail_log + linear_term**2)
        return estimate, width, variance_upper

    completion_lower, lower_width, lower_variance = endpoint(x, alpha_lower)
    completion_upper, upper_width, upper_variance = endpoint(x + u, alpha_upper)
    return DirectMeanBounds(
        lower=max(0.0, completion_lower - lower_width),
        upper=min(1.0, completion_upper + upper_width),
        completion_estimate_lower=completion_lower,
        completion_estimate_upper=completion_upper,
        half_width_lower=lower_width,
        half_width_upper=upper_width,
        concentration_variance=concentration_variance,
        effective_sample_size=1 / concentration_variance,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        method="weighted_empirical_bernstein",
        independence_unit=independence_unit,
        variance_upper_lower=lower_variance,
        variance_upper_upper=upper_variance,
        maximum_increment=maximum_increment,
    )


def worst_of_k_kl_chernoff_bounds(
    failures: ArrayLike,
    unclassified: ArrayLike,
    trials: ArrayLike,
    weights: ArrayLike,
    *,
    k: int,
    alpha_lower: float,
    alpha_upper: float,
) -> DirectMeanBounds:
    """Experimental. KL-Chernoff bounds for the weighted chance that one of ``k`` attempts fails.

    The estimand is ``sum_i w_i (1 - (1 - p_i) ** k)`` over fixed prompts with independent
    rollouts. Prompt ``i`` contributes its unbiased U-statistic ``1 - C(n_i - x_i, k) /
    C(n_i, k)``. Hoeffding (1963, section 5) writes a U-statistic of order ``k`` on ``n``
    independent draws as an average, over permutations, of the mean of ``floor(n / k)``
    independent ``[0, 1]`` terms, one per disjoint group of ``k`` draws; by convexity its
    moment generating function is at most that mean's. The KL-Chernoff bound uses only that
    function, so prompt ``i`` enters :func:`weighted_bounded_kl_chernoff_mean_bounds` as
    ``floor(n_i / k)`` blocks at its share, each of weight ``w_i / floor(n_i / k)``. At
    ``k = 1`` this is the rollout-level :func:`weighted_kl_chernoff_mean_bounds`.

    Unclassified rollouts count as safe for the lower endpoint and as failures for the upper
    one; the share only grows with failures, so the completion is conservative. ``trials`` are
    the attempted rollouts, and every prompt needs at least ``k`` of them: for ``k > n`` no
    unbiased estimator exists, and this design cannot bound the risk.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer of at least one")
    x, u, n, w = _validate_inputs(
        failures,
        unclassified,
        trials,
        weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )
    if np.any(n < k):
        raise ValueError(
            f"every prompt needs at least {k} attempted rollouts to bound {k} attempts; "
            f"a prompt has only {int(np.min(n))}"
        )
    blocks = n // k
    return weighted_bounded_kl_chernoff_mean_bounds(
        np.repeat(worst_of_k_shares(x, n, k=k), blocks),
        np.repeat(worst_of_k_shares(x + u, n, k=k), blocks),
        np.repeat(w / blocks, blocks),
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_profile="rollout",
    )


_BETTING_GRID_SIZE = 16
BETTING_KERNEL = "common_bet_mixture_jensen_exponent"


def _betting_prior() -> tuple[np.ndarray, np.ndarray]:
    """Common bet fractions and their fixed log mixture weights.

    The fractions are Beta(1/2, 1/2) quantiles, the universal-portfolio prior.
    Half of the prior mass rides the most aggressive fraction, so the
    sparse-failure regime, where that bet is optimal, pays about ``log 2`` for
    not knowing the best bet in advance instead of ``log J``. The prior is fixed
    before any outcome is seen, which is what keeps the mixture an e-value.
    """

    quantiles = (2 * np.arange(1, _BETTING_GRID_SIZE + 1) - 1) / (2 * _BETTING_GRID_SIZE)
    fractions = np.square(np.sin(np.pi * quantiles / 2))
    log_weights = np.full(_BETTING_GRID_SIZE, math.log(0.5 / _BETTING_GRID_SIZE))
    log_weights[-1] = math.log(0.5 + 0.5 / _BETTING_GRID_SIZE)
    return fractions, log_weights


def _betting_log_capital(
    values: np.ndarray,
    exponents: np.ndarray,
    mean: float,
    *,
    direction: Literal["lower", "upper"],
) -> float:
    """Log mixture capital against the null ``sum_g A_g E[Y_g] = mean``.

    Every block is bet against with the same fraction ``theta`` of the maximal
    admissible stake, ``1 - theta + theta * Y / mean`` when betting that the
    mean exceeds ``mean`` and ``1 - theta + theta * (1 - Y) / (1 - mean)`` when
    betting that it falls short. Block ``g`` enters with exponent
    ``A_g / max A``. With ``theta < 1`` every factor is positive.
    """

    fractions, log_prior = _betting_prior()
    ratios = values / mean if direction == "lower" else (1.0 - values) / (1.0 - mean)
    factors = 1.0 - fractions[:, None] + fractions[:, None] * ratios[None, :]
    log_growth = np.sum(exponents[None, :] * np.log(factors), axis=1)
    return float(logsumexp(log_prior + log_growth))


def _invert_betting(
    values: np.ndarray,
    exponents: np.ndarray,
    *,
    alpha: float,
    direction: Literal["lower", "upper"],
) -> float:
    """Boundary of the non-rejected means ``{m : capital(m) < 1 / alpha}``.

    Capital is monotone in the candidate mean (increasing for the upper bet,
    decreasing for the lower bet), so the confidence set is an interval and
    its endpoint is found by bisection. The plug-in estimate is never rejected,
    so the endpoints always straddle it.
    """

    threshold = math.log(1.0 / alpha)
    edge = 1e-12

    def rejected(mean: float) -> bool:
        return _betting_log_capital(values, exponents, mean, direction=direction) >= threshold

    if direction == "upper":
        if not rejected(1.0 - edge):
            return 1.0
        inside, outside = 0.0, 1.0 - edge
    else:
        if not rejected(edge):
            return 0.0
        outside, inside = edge, 1.0
    for _ in range(80):
        midpoint = (inside + outside) / 2
        if rejected(midpoint):
            outside = midpoint
        else:
            inside = midpoint
    return (inside + outside) / 2


def weighted_bounded_betting_mean_bounds(
    completion_lower: ArrayLike,
    completion_upper: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    independence_profile: str,
) -> DirectMeanBounds:
    """Mixture-of-common-bets bounds for independent [0,1]-valued blocks.

    For a candidate mean ``m`` and a common bet fraction, the capital
    ``prod_g factor_g ** (A_g / max A)`` has expectation at most one whenever
    ``sum_g A_g E[Y_g] = m``: each exponent lies in (0, 1], so Jensen's
    inequality bounds every block's expected power by the power of its
    expectation, and the weighted AM-GM inequality then bounds the product by
    the weighted sum, which is exactly one under the null. A fixed mixture over
    fractions keeps that property, so the non-rejected means form a valid
    confidence set at level ``1 - alpha``.

    The bet is common to every block and fixed in advance. Choosing it block by
    block, even from earlier blocks, is invalid under heterogeneous block
    means, because the null constrains only the weighted sum. Equal weights
    give exponent one everywhere; unequal weights pay through the exponents,
    much as the KL engine pays through its largest increment. The construction
    needs no variance proxy, so it adapts to low-variance fractional blocks
    without the empirical-Bernstein padding term.
    """

    lower_values, upper_values, w = _validate_bounded_inputs(
        completion_lower,
        completion_upper,
        weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )
    estimate_lower = float(np.dot(w, lower_values))
    estimate_upper = float(np.dot(w, upper_values))
    c2 = float(np.sum(np.square(w)))
    maximum_increment = float(np.max(w))
    exponents = w / maximum_increment
    lower = _invert_betting(lower_values, exponents, alpha=alpha_lower, direction="lower")
    upper = _invert_betting(upper_values, exponents, alpha=alpha_upper, direction="upper")
    return DirectMeanBounds(
        lower=lower,
        upper=upper,
        completion_estimate_lower=estimate_lower,
        completion_estimate_upper=estimate_upper,
        half_width_lower=estimate_lower - lower,
        half_width_upper=upper - estimate_upper,
        concentration_variance=c2,
        effective_sample_size=1 / c2,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        method="weighted_betting",
        independence_unit=independence_profile,
        maximum_increment=maximum_increment,
        chernoff_kernel=BETTING_KERNEL,
    )


def weighted_betting_mean_bounds(
    failures: ArrayLike,
    unclassified: ArrayLike,
    trials: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    independence_unit: IndependenceUnit = "rollout",
) -> DirectMeanBounds:
    """Betting bounds from per-prompt counts under the two canonical partitions.

    Under ``rollout`` independence every attempted rollout is one block with
    weight ``w_j / n_j``; under ``prompt_cluster`` each prompt's completed
    failure rate is one block with weight ``w_j``. Unclassified outcomes are
    completed as safe for the lower endpoint and as failures for the upper.
    """

    x, u, n, w = _validate_inputs(
        failures,
        unclassified,
        trials,
        weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )
    if independence_unit not in {"rollout", "prompt_cluster"}:
        raise ValueError("independence_unit must be rollout or prompt_cluster")
    if independence_unit == "rollout":
        block_weights = np.repeat(w / n, n)
        lower_values = np.concatenate(
            [np.r_[np.ones(int(k)), np.zeros(int(m - k))] for k, m in zip(x, n, strict=True)]
        )
        upper_values = np.concatenate(
            [
                np.r_[np.ones(int(k + v)), np.zeros(int(m - k - v))]
                for k, v, m in zip(x, u, n, strict=True)
            ]
        )
    else:
        block_weights = w
        lower_values = x / n
        upper_values = (x + u) / n
    return weighted_bounded_betting_mean_bounds(
        lower_values,
        upper_values,
        block_weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_profile=independence_unit,
    )
