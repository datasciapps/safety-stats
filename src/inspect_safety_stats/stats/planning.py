from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import binom


@dataclass(frozen=True)
class FixedDesignDetectableRates:
    """Conservative power guarantees for a fixed weighted Bernoulli mean."""

    maximum_certifiable_failure_rate: float | None
    minimum_detectable_failure_rate: float | None
    support_gap: float | None
    rejection_gap: float | None
    target_power: float
    concentration_variance: float
    effective_sample_size: float
    independence_unit: Literal["rollout", "prompt_cluster"]


def zero_failure_sample_size(*, threshold: float, alpha: float) -> int:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    raw = math.log(alpha) / math.log1p(-threshold)
    candidate = max(1, math.ceil(raw))
    while 1.0 - alpha ** (1.0 / candidate) > threshold:
        candidate += 1
    while candidate > 1 and 1.0 - alpha ** (1.0 / (candidate - 1)) <= threshold:
        candidate -= 1
    return candidate


def _support_p_value(failures: int, trials: int, threshold: float) -> float:
    """Exact lower-tail p-value for H0: p >= threshold."""

    if isinstance(failures, bool) or isinstance(trials, bool):
        raise TypeError("counts must be integers, not booleans")
    if not isinstance(failures, int) or not isinstance(trials, int):
        raise TypeError("counts must be integers")
    if trials <= 0 or failures < 0 or failures > trials:
        raise ValueError("counts must satisfy 0 <= failures <= trials and trials > 0")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    return float(binom.cdf(failures, trials, threshold))


def maximum_certifying_failures(*, trials: int, threshold: float, alpha: float) -> int:
    """Largest failure count that rejects H0: p >= threshold at level alpha."""

    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if not 0 < threshold < 1 or not 0 < alpha < 1:
        raise ValueError("threshold and alpha must be in (0, 1)")
    quantile = int(binom.ppf(alpha, trials, threshold))
    candidates = range(max(-1, quantile - 2), min(trials, quantile + 2) + 1)
    valid = [
        failures
        for failures in candidates
        if failures >= 0 and _support_p_value(failures, trials, threshold) <= alpha
    ]
    return max(valid, default=-1)


def certification_power(*, trials: int, true_rate: float, threshold: float, alpha: float) -> float:
    if not 0 <= true_rate < threshold < 1:
        raise ValueError("true_rate must satisfy 0 <= true_rate < threshold < 1")
    cutoff = maximum_certifying_failures(trials=trials, threshold=threshold, alpha=alpha)
    return 0.0 if cutoff < 0 else float(binom.cdf(cutoff, trials, true_rate))


def minimum_rejecting_failures(*, trials: int, threshold: float, alpha: float) -> int:
    """Smallest failure count that rejects H0: p <= threshold at level alpha.

    Returns ``trials + 1`` when rejection is impossible even if every trial fails.
    """

    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if not 0 < threshold < 1 or not 0 < alpha < 1:
        raise ValueError("threshold and alpha must be in (0, 1)")
    if float(binom.sf(trials - 1, trials, threshold)) > alpha:
        return trials + 1
    lower = 1
    upper = trials
    while lower < upper:
        midpoint = (lower + upper) // 2
        if float(binom.sf(midpoint - 1, trials, threshold)) <= alpha:
            upper = midpoint
        else:
            lower = midpoint + 1
    return lower


def rejection_power(*, trials: int, true_rate: float, threshold: float, alpha: float) -> float:
    """Exact power to reject H0: p <= threshold at an unsafe true rate."""

    if not 0 < threshold < true_rate <= 1:
        raise ValueError("true_rate must satisfy 0 < threshold < true_rate <= 1")
    cutoff = minimum_rejecting_failures(trials=trials, threshold=threshold, alpha=alpha)
    return 0.0 if cutoff > trials else float(binom.sf(cutoff - 1, trials, true_rate))


def maximum_certifiable_failure_rate(
    *, trials: int, threshold: float, alpha: float, power: float
) -> float | None:
    """Largest rate below ``threshold`` supported with at least ``power``.

    This exactly inverts the lower-tail binomial certification test. ``None``
    means that the test cannot support the claim for any true rate at the
    requested horizon and alpha.
    """

    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    cutoff = maximum_certifying_failures(trials=trials, threshold=threshold, alpha=alpha)
    if cutoff < 0:
        return None

    def objective(rate: float) -> float:
        return float(binom.cdf(cutoff, trials, rate)) - power

    if objective(threshold) >= 0:
        return threshold
    return float(brentq(objective, 0.0, threshold, xtol=1e-14, rtol=1e-14))


def minimum_detectable_failure_rate(
    *, trials: int, threshold: float, alpha: float, power: float
) -> float | None:
    """Smallest rate above ``threshold`` rejected with at least ``power``.

    This exactly inverts the upper-tail binomial rejection test. ``None``
    means rejection is impossible at the requested horizon and alpha.
    """

    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    cutoff = minimum_rejecting_failures(trials=trials, threshold=threshold, alpha=alpha)
    if cutoff > trials:
        return None

    def objective(rate: float) -> float:
        return float(binom.sf(cutoff - 1, trials, rate)) - power

    if objective(threshold) >= 0:
        return threshold
    return float(brentq(objective, threshold, 1.0, xtol=1e-14, rtol=1e-14))


def fixed_design_detectable_failure_rates(
    trials: ArrayLike,
    weights: ArrayLike,
    *,
    threshold: float,
    support_alpha: float,
    rejection_alpha: float,
    power: float,
    independence_unit: Literal["rollout", "prompt_cluster"] = "rollout",
) -> FixedDesignDetectableRates:
    """Return Hoeffding-guaranteed detectable rates for a fixed weighted mean.

    The calculation combines the one-sided testing radius at ``alpha`` with a
    second one-sided concentration radius at ``1 - power``. It permits
    heterogeneous prompt probabilities. Prompt-cluster mode allows arbitrary
    dependence within each prompt batch and assumes independence across batches.
    """

    n = np.asarray(trials)
    w = np.asarray(weights, dtype=float)
    if n.ndim != 1 or w.ndim != 1 or len(n) == 0 or len(n) != len(w):
        raise ValueError("trials and weights must be nonempty one-dimensional arrays")
    if not np.issubdtype(n.dtype, np.integer) or np.any(n <= 0):
        raise ValueError("trials must contain positive integers")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not math.isclose(float(np.sum(w)), 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("weights must sum to one")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    if not 0 < support_alpha < 1 or not 0 < rejection_alpha < 1:
        raise ValueError("support_alpha and rejection_alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if independence_unit not in {"rollout", "prompt_cluster"}:
        raise ValueError("independence_unit must be rollout or prompt_cluster")

    if independence_unit == "rollout":
        concentration_variance = float(np.sum(np.square(w) / n.astype(float)))
    else:
        concentration_variance = float(np.sum(np.square(w)))
    beta = 1 - power
    power_radius = math.sqrt(0.5 * concentration_variance * math.log(1 / beta))
    support_radius = math.sqrt(0.5 * concentration_variance * math.log(1 / support_alpha))
    rejection_radius = math.sqrt(0.5 * concentration_variance * math.log(1 / rejection_alpha))
    support_gap = support_radius + power_radius
    rejection_gap = rejection_radius + power_radius
    maximum_certifiable = threshold - support_gap
    minimum_detectable = threshold + rejection_gap
    return FixedDesignDetectableRates(
        maximum_certifiable_failure_rate=(
            max(0.0, maximum_certifiable) if maximum_certifiable >= 0 else None
        ),
        minimum_detectable_failure_rate=(
            min(1.0, minimum_detectable) if minimum_detectable <= 1 else None
        ),
        support_gap=support_gap if maximum_certifiable >= 0 else None,
        rejection_gap=rejection_gap if minimum_detectable <= 1 else None,
        target_power=power,
        concentration_variance=concentration_variance,
        effective_sample_size=1 / concentration_variance,
        independence_unit=independence_unit,
    )


DirectMeanMethod = Literal[
    "weighted_hoeffding",
    "weighted_empirical_bernstein",
    "weighted_kl_chernoff",
]


@dataclass(frozen=True)
class MethodDetectableRates:
    """Detectable rates of a fixed weighted mean under the interval method analyze uses.

    ``certification_floor`` is the upper endpoint when every block is safe: a claim whose
    acceptable_upper lies below it can never PASS. ``rejection_ceiling`` is the lower endpoint
    when every block fails: a claim whose unacceptable_lower lies above it can never FAIL.
    """

    method: DirectMeanMethod
    endpoint_alpha: float
    target_power: float
    certification_floor: float
    rejection_ceiling: float
    maximum_certifiable_failure_rate: float | None
    minimum_detectable_failure_rate: float | None
    effective_sample_size: float
    blocks: int


def _planned_endpoints(
    method: DirectMeanMethod, weights: np.ndarray, estimate: float, alpha: float
) -> tuple[float, float]:
    """The interval analyze reports for blocks of the given weights and estimate.

    Hoeffding and KL-Chernoff endpoints depend on the data only through the estimate, so the
    analysis functions run on blocks that all equal it. Empirical Bernstein also reads the
    blocks' second moments; it takes the largest that any data with this estimate allows
    (a block of weight w and value v adds w^2 v^2 <= m w v, with m the largest weight).
    """

    from inspect_safety_stats.stats.direct_mean import (
        bounded_empirical_bernstein_width,
        weighted_bounded_hoeffding_mean_bounds,
        weighted_bounded_kl_chernoff_mean_bounds,
    )

    if method == "weighted_empirical_bernstein":
        maximum_increment = float(np.max(weights))
        width, _ = bounded_empirical_bernstein_width(
            concentration_variance=float(np.sum(np.square(weights))),
            fourth_power_sum=float(np.sum(np.power(weights, 4))),
            maximum_increment=maximum_increment,
            positive_second=maximum_increment * estimate,
            negative_second=maximum_increment * (1 - estimate),
            alpha=alpha,
        )
        return max(0.0, estimate - width), min(1.0, estimate + width)
    function = {
        "weighted_hoeffding": weighted_bounded_hoeffding_mean_bounds,
        "weighted_kl_chernoff": weighted_bounded_kl_chernoff_mean_bounds,
    }.get(method)
    if function is None:
        raise ValueError(f"unsupported direct mean interval method {method!r}")
    values = np.full(len(weights), estimate)
    bounds = function(
        values,
        values,
        weights,
        alpha_lower=alpha,
        alpha_upper=alpha,
        independence_profile="planned",
    )
    return bounds.lower, bounds.upper


def _last_true(predicate: Callable[[float], bool]) -> float:
    """The largest estimate in [0, 1] where ``predicate`` holds, for one true on a prefix."""

    inside, outside = 0.0, 1.0
    if predicate(outside):
        return outside
    for _ in range(100):
        midpoint = (inside + outside) / 2
        if predicate(midpoint):
            inside = midpoint
        else:
            outside = midpoint
    return inside


def _first_true(predicate: Callable[[float], bool]) -> float:
    """The smallest estimate in [0, 1] where ``predicate`` holds, for one true on a suffix."""

    outside, inside = 0.0, 1.0
    if predicate(outside):
        return outside
    for _ in range(100):
        midpoint = (outside + inside) / 2
        if predicate(midpoint):
            inside = midpoint
        else:
            outside = midpoint
    return inside


def detectable_failure_rates(
    weights: ArrayLike,
    *,
    method: DirectMeanMethod,
    alpha: float,
    power: float,
    acceptable_upper: float | None = None,
    unacceptable_lower: float | None = None,
) -> MethodDetectableRates:
    """Detectable rates of a weighted mean of independent [0,1] blocks under ``method``.

    The test side inverts the analysis interval itself: the estimates at which its upper
    endpoint is at or below ``acceptable_upper`` (PASS) or its lower endpoint at or above
    ``unacceptable_lower`` (FAIL). The power side bounds the chance that the estimate misses
    them, whatever the blocks' individual rates, by the sharpest of Hoeffding's inequality
    and the Chernoff kernel the KL-Chernoff interval uses (Bernoulli KL for equal block
    weights, Poisson deviance with the largest weight otherwise), applied to the failures
    and to the safe outcomes. Under the rollout profile
    each rollout is a block. ``None`` means no true rate reaches the target power, or no
    outcome at all crosses the threshold.
    """

    from inspect_safety_stats.stats.direct_mean import (
        _bernoulli_kl,
        _invert_divergence,
        _poisson_deviance,
    )

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("block weights must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("block weights must be finite and nonnegative")
    if not math.isclose(float(np.sum(w)), 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("block weights must sum to one")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    for name, threshold in (
        ("acceptable_upper", acceptable_upper),
        ("unacceptable_lower", unacceptable_lower),
    ):
        if threshold is not None and not 0 < threshold < 1:
            raise ValueError(f"{name} must be in (0, 1)")

    concentration_variance = float(np.sum(np.square(w)))
    log_beta = math.log(1 / (1 - power))
    hoeffding_radius = math.sqrt(0.5 * concentration_variance * log_beta)
    # The same kernel choice as the KL-Chernoff interval.
    if bool(np.allclose(w, np.full_like(w, 1 / len(w)), rtol=1e-10, atol=1e-14)):
        divergence, target = _bernoulli_kl, log_beta / len(w)
    else:
        divergence, target = _poisson_deviance, float(np.max(w)) * log_beta

    def endpoints(estimate: float) -> tuple[float, float]:
        return _planned_endpoints(method, w, estimate, alpha)

    maximum_certifiable: float | None = None
    if acceptable_upper is not None and endpoints(0.0)[1] <= acceptable_upper:
        # Every estimate up to `boundary` passes, so the claim is missed only when the
        # estimate exceeds it.
        boundary = _last_true(lambda estimate: endpoints(estimate)[1] <= acceptable_upper)
        maximum_certifiable = max(
            boundary - hoeffding_radius,
            _invert_divergence(boundary, target, direction="lower", divergence=divergence),
            # The same kernel on the safe outcomes, whose weighted mean is 1 - estimate.
            1 - _invert_divergence(1 - boundary, target, direction="upper", divergence=divergence),
        )
        if maximum_certifiable < 0:
            maximum_certifiable = None
    minimum_detectable: float | None = None
    if unacceptable_lower is not None and endpoints(1.0)[0] >= unacceptable_lower:
        boundary = _first_true(lambda estimate: endpoints(estimate)[0] >= unacceptable_lower)
        minimum_detectable = min(
            boundary + hoeffding_radius,
            _invert_divergence(boundary, target, direction="upper", divergence=divergence),
            1 - _invert_divergence(1 - boundary, target, direction="lower", divergence=divergence),
        )
        if minimum_detectable > 1:
            minimum_detectable = None
    return MethodDetectableRates(
        method=method,
        endpoint_alpha=alpha,
        target_power=power,
        certification_floor=endpoints(0.0)[1],
        rejection_ceiling=endpoints(1.0)[0],
        maximum_certifiable_failure_rate=maximum_certifiable,
        minimum_detectable_failure_rate=minimum_detectable,
        effective_sample_size=1 / concentration_variance,
        blocks=len(w),
    )


def certification_sample_size(
    *,
    true_rate: float,
    threshold: float,
    alpha: float,
    power: float,
    max_trials: int = 1_000_000,
) -> int:
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials <= 0:
        raise ValueError("max_trials must be a positive integer")
    for trials in range(1, max_trials + 1):
        if (
            certification_power(
                trials=trials,
                true_rate=true_rate,
                threshold=threshold,
                alpha=alpha,
            )
            >= power
        ):
            return trials
    raise ValueError("requested certification power exceeds max_trials search limit")


def rejection_sample_size(
    *,
    true_rate: float,
    threshold: float,
    alpha: float,
    power: float,
    max_trials: int = 1_000_000,
) -> int:
    """Smallest exact horizon attaining rejection power at an unsafe rate."""

    if not 0 < threshold < true_rate <= 1:
        raise ValueError("true_rate must satisfy 0 < threshold < true_rate <= 1")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials <= 0:
        raise ValueError("max_trials must be a positive integer")
    for trials in range(1, max_trials + 1):
        if (
            rejection_power(
                trials=trials,
                true_rate=true_rate,
                threshold=threshold,
                alpha=alpha,
            )
            >= power
        ):
            return trials
    raise ValueError("requested rejection power exceeds max_trials search limit")
