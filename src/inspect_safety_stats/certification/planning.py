from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import binom

from inspect_safety_stats.certification.evidence import exact_support_p_value


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
        if failures >= 0 and exact_support_p_value(failures, trials, threshold) <= alpha
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
