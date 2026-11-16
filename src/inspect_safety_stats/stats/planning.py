from __future__ import annotations

import math
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


ComparisonPlanningUnit = Literal["rollout", "pair", "prompt_cluster"]


@dataclass(frozen=True)
class ResolvableRiskDifferences:
    """Risk differences a fixed two-arm design resolves at a margin, with Hoeffding power.

    ``maximum_supportable_difference`` is the largest true candidate-minus-baseline
    difference at which the upper bound falls at or below ``margin`` with at least
    ``target_power``; ``minimum_detectable_difference`` is the smallest true difference at
    which the lower bound reaches ``margin`` with at least that power. ``None`` means no
    difference in ``[-1, 1]`` attains it.
    """

    margin: float
    maximum_supportable_difference: float | None
    minimum_detectable_difference: float | None
    support_gap: float | None
    rejection_gap: float | None
    target_power: float
    concentration_variance: float
    effective_sample_size: float
    block_count: int
    independence_unit: ComparisonPlanningUnit


def comparison_resolvable_differences(
    candidate_trials: ArrayLike,
    baseline_trials: ArrayLike,
    weights: ArrayLike,
    *,
    margin: float,
    alpha_lower: float,
    alpha_upper: float,
    power: float,
    independence_unit: ComparisonPlanningUnit = "rollout",
) -> ResolvableRiskDifferences:
    """Return Hoeffding-guaranteed resolvable risk differences for a two-arm design.

    The guarantee is that of the ``weighted_hoeffding`` risk difference which ``analyze``
    builds on ``Q = (1 + candidate - baseline) / 2``: with no unclassified outcomes, a true
    difference at or below ``maximum_supportable_difference`` gives an upper bound at or below
    ``margin``, and one at or above ``minimum_detectable_difference`` a lower bound at or
    above ``margin``, each with probability at least ``power`` (Hoeffding, 1963). Prompt
    probabilities may differ. The independence unit names the blocks of ``Q``:

    - ``rollout``: every rollout of either arm is independent (unpaired arms; an
      independence profile ``block_by: [rollout]``).
    - ``pair``: the candidate and baseline rollouts of a prompt with the same repeat index
      form one block, with arbitrary dependence inside it (``block_by: [prompt_id, repeat]``
      with ``comparison_coupling: joint``).
    - ``prompt_cluster``: all rollouts of both arms on a prompt form one block
      (``block_by: [prompt_id]`` with ``comparison_coupling: joint``).

    The guarantee does not extend to the KL/Chernoff or empirical-Bernstein differences,
    which are usually, not always, narrower.
    """

    nc = np.asarray(candidate_trials)
    nb = np.asarray(baseline_trials)
    w = np.asarray(weights, dtype=float)
    if any(array.ndim != 1 for array in (nc, nb, w)) or len(w) == 0:
        raise ValueError("trials and weights must be nonempty one-dimensional arrays")
    if len(nc) != len(w) or len(nb) != len(w):
        raise ValueError("candidate trials, baseline trials and weights must be aligned")
    if not np.issubdtype(nc.dtype, np.integer) or not np.issubdtype(nb.dtype, np.integer):
        raise ValueError("trials must contain positive integers")
    if np.any(nc <= 0) or np.any(nb <= 0):
        raise ValueError("trials must contain positive integers")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not math.isclose(float(np.sum(w)), 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("weights must sum to one")
    if isinstance(margin, bool) or not -1 < margin < 1:
        raise ValueError("margin must be in (-1, 1)")
    if not 0 < alpha_lower < 1 or not 0 < alpha_upper < 1:
        raise ValueError("alpha_lower and alpha_upper must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if independence_unit not in {"rollout", "pair", "prompt_cluster"}:
        raise ValueError("independence_unit must be rollout, pair or prompt_cluster")

    candidate = nc.astype(float)
    baseline = nb.astype(float)
    positive = w > 0
    # c2 is the sum of squared block masses of Q, whose blocks are bounded in [0, 1].
    if independence_unit == "rollout":
        c2 = float(np.sum(np.square(w) / 4 * (1 / candidate + 1 / baseline)))
        blocks = int(np.sum((nc + nb)[positive]))
    elif independence_unit == "pair":
        shared = np.minimum(candidate, baseline)
        longer = np.maximum(candidate, baseline)
        shared_mass = w / 2 * (1 / candidate + 1 / baseline)
        unshared_mass = w / (2 * longer)
        c2 = float(
            np.sum(shared * np.square(shared_mass) + (longer - shared) * np.square(unshared_mass))
        )
        blocks = int(np.sum(np.maximum(nc, nb)[positive]))
    else:
        c2 = float(np.sum(np.square(w)))
        blocks = int(np.sum(positive))

    # D = 2Q - 1, so every radius on Q doubles on D.
    power_radius = 2 * math.sqrt(0.5 * c2 * math.log(1 / (1 - power)))
    support_gap = 2 * math.sqrt(0.5 * c2 * math.log(1 / alpha_upper)) + power_radius
    rejection_gap = 2 * math.sqrt(0.5 * c2 * math.log(1 / alpha_lower)) + power_radius
    maximum_supportable = margin - support_gap
    minimum_detectable = margin + rejection_gap
    return ResolvableRiskDifferences(
        margin=margin,
        maximum_supportable_difference=(
            maximum_supportable if maximum_supportable >= -1 else None
        ),
        minimum_detectable_difference=minimum_detectable if minimum_detectable <= 1 else None,
        support_gap=support_gap if maximum_supportable >= -1 else None,
        rejection_gap=rejection_gap if minimum_detectable <= 1 else None,
        target_power=power,
        concentration_variance=4 * c2,
        effective_sample_size=1 / c2,
        block_count=blocks,
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
