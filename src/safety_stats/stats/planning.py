from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import betaincinv
from scipy.stats import beta as beta_distribution
from scipy.stats import binom

from safety_stats.config.methods import DirectMeanMethod


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


def expected_upper_bound(*, draws: float, rate: float, alpha: float) -> float:
    """The one-sided Clopper-Pearson upper bound ``draws`` trials at ``rate`` are expected to give.

    The expected failure count, ``rate`` times ``draws``, and ``draws`` itself are not rounded
    to whole numbers, and ``draws`` counts as at least one. The bound then falls as ``draws``
    grows, as a forecast must for a search over budgets; with rounded counts it jumps, so that
    16 trials at 3% forecast a tighter bound than 20. At whole numbers it equals
    ``binomial_upper_bound``. ``rate`` is in ``[0, 1)`` and ``alpha`` in ``(0, 1)``. For
    planning only: it describes no observed data.
    """

    trials = max(1.0, float(draws))
    failures = rate * trials
    return float(betaincinv(failures + 1.0, trials - failures, 1.0 - alpha))


def zero_failure_sample_size(*, threshold: float, alpha: float) -> int:
    """Fewest trials for which zero failures certify a failure rate at or below ``threshold``.

    Returns the smallest ``n`` whose one-sided Clopper-Pearson upper bound after zero failures
    in ``n`` trials, ``1 - alpha ** (1 / n)``, is at most ``threshold``. Both ``threshold``
    and ``alpha`` lie in ``(0, 1)``, and ``alpha`` is the one-sided error level. With
    ``alpha = 0.05`` and a small ``threshold`` it is close to the rule of three,
    ``3 / threshold``. Derived in
    docs/statistical-methods.md, sections 5.1 and 5.2.
    """

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
    if not isinstance(failures, Integral) or not isinstance(trials, Integral):
        raise TypeError("counts must be integers")
    if trials <= 0 or failures < 0 or failures > trials:
        raise ValueError("counts must satisfy 0 <= failures <= trials and trials > 0")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    return float(binom.cdf(failures, trials, threshold))


def maximum_certifying_failures(*, trials: int, threshold: float, alpha: float) -> int:
    """Largest failure count that rejects H0: p >= threshold at level alpha."""

    if isinstance(trials, bool) or not isinstance(trials, Integral) or trials <= 0:
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

    if isinstance(trials, bool) or not isinstance(trials, Integral) or trials <= 0:
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


INVERTIBLE_METHODS: Final[frozenset[DirectMeanMethod]] = frozenset(
    {"weighted_hoeffding", "weighted_empirical_bernstein", "weighted_kl_chernoff"}
)
"""The methods whose detectable rates the planner can derive by inverting the interval.

Their endpoints follow from the estimate alone, or, for empirical Bernstein, from the worst
data with that estimate. Any other method's endpoints depend on the individual block values,
so no single estimate boundary separates PASS from not PASS.
"""


@dataclass(frozen=True)
class MethodDetectableRates:
    """Detectable rates of a fixed weighted mean under the interval method analyze uses.

    ``certification_floor`` is the upper endpoint when every block is safe, and
    ``rejection_ceiling`` the lower endpoint when every block fails. They are the interval's
    own endpoints: analyze's hierarchy-region projection can lower the reported upper bound
    and raise the lower one. ``claim_planning`` reports the projected floor.
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

    from safety_stats.stats.direct_mean import (
        DIRECT_MEAN_ENGINES,
        bounded_empirical_bernstein_width,
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
    if method not in INVERTIBLE_METHODS:
        raise ValueError(
            f"the planner cannot invert the {method} interval: its endpoints depend on the "
            "individual block values, not only on their mean"
        )
    values = np.full(len(weights), estimate)
    bounds = DIRECT_MEAN_ENGINES[method].bounded(
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
    and to the safe outcomes. Under the rollout profile each rollout is a block. A rate is
    ``None`` when its threshold is not given or when no outcome at all crosses it; otherwise
    some true rate, 0 or 1 at the extreme, reaches the target power.
    """

    from safety_stats.stats.direct_mean import (
        _bounded_chernoff_kernel,
        _invert_divergence,
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
    # The KL-Chernoff interval's kernel, at level beta = 1 - power.
    _, kernel_target, divergence = _bounded_chernoff_kernel(w)
    target = kernel_target(1 - power)

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
    minimum_detectable: float | None = None
    if unacceptable_lower is not None and endpoints(1.0)[0] >= unacceptable_lower:
        boundary = _first_true(lambda estimate: endpoints(estimate)[0] >= unacceptable_lower)
        minimum_detectable = min(
            boundary + hoeffding_radius,
            _invert_divergence(boundary, target, direction="upper", divergence=divergence),
            1 - _invert_divergence(1 - boundary, target, direction="lower", divergence=divergence),
        )
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


def certification_floor(weights: ArrayLike, *, method: DirectMeanMethod, alpha: float) -> float:
    """The upper endpoint of ``method``'s interval when every block is safe.

    The data are then fully known, so this holds for every method, the ones the planner
    cannot invert included. It is the interval's own endpoint, before any hierarchy-region
    projection, which can lower it.
    """

    from safety_stats.stats.direct_mean import DIRECT_MEAN_ENGINES

    w = np.asarray(weights, dtype=float)
    zeros = np.zeros(len(w))
    return float(
        DIRECT_MEAN_ENGINES[method]
        .bounded(
            zeros,
            zeros,
            w,
            alpha_lower=alpha,
            alpha_upper=alpha,
            independence_profile="planned",
        )
        .upper
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
    builds on ``Q = (1 + candidate - baseline) / 2`` for a ``comparisons`` entry: with no
    unclassified outcomes, a true difference at or below ``maximum_supportable_difference``
    gives an upper bound at or below ``margin``, and one at or above
    ``minimum_detectable_difference`` a lower bound at or above ``margin``, each with
    probability at least ``power`` (Hoeffding, 1963). Prompt probabilities may differ. The
    independence unit names the blocks of ``Q``:

    - ``rollout``: every rollout of either arm is independent (unpaired arms; an
      independence profile ``block_by: [rollout]``).
    - ``pair``: the candidate and baseline rollouts of a prompt with the same repeat index
      form one block, with arbitrary dependence inside it (``block_by: [prompt_id, repeat]``
      with ``comparison_coupling: joint``).
    - ``prompt_cluster``: all rollouts of both arms on a prompt form one block
      (``block_by: [prompt_id]`` with ``comparison_coupling: joint``).

    The guarantee does not extend to a ``comparison_sets`` entry, whatever its method, nor
    to ``weighted_kl_chernoff``: both bound each arm at half the alpha and subtract the
    endpoints, which under ``weighted_hoeffding`` is usually wider (about 1.5 times for
    unpaired arms at equal counts and a one-sided alpha of 0.025).
    Nor does it extend to ``weighted_empirical_bernstein``. The KL/Chernoff and
    empirical-Bernstein differences are often narrower on sparse failures when prompt
    weights are even; with uneven weights, or at moderate failure rates, either can be
    wider.
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
    if isinstance(max_trials, bool) or not isinstance(max_trials, Integral) or max_trials <= 0:
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
    if isinstance(max_trials, bool) or not isinstance(max_trials, Integral) or max_trials <= 0:
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


def exact_half_width(rate: float, trials: float, alpha: float) -> float:
    """Larger one-sided Clopper-Pearson distance from ``rate`` at a real-valued count.

    Each endpoint is the beta quantile that inverts the one-sided exact binomial test at level
    ``alpha`` (Clopper and Pearson, Biometrika 26(4), 404-413, 1934; docs/statistical-methods.md
    section 4.2), with ``rate * trials`` failures, which need not be a whole number.
    """

    failures = rate * trials
    upper = (
        1.0
        if failures >= trials
        else float(beta_distribution.ppf(1.0 - alpha, failures + 1.0, trials - failures))
    )
    lower = (
        0.0
        if failures <= 0
        else float(beta_distribution.ppf(alpha, failures, trials - failures + 1.0))
    )
    return max(upper - rate, rate - lower)


def rollouts_for_half_width(
    *,
    rate: float,
    half_width: float,
    alpha: float,
    maximum: int = 10_000_000,
) -> int | None:
    """Equally weighted rollouts needed for an exact bound within ``half_width`` of ``rate``.

    The count is the smallest ``n`` at which both one-sided Clopper-Pearson endpoints
    (``exact_half_width``), each at level ``alpha``, lie within ``half_width`` of the assumed
    failure rate; at rate 0 it is the zero-failure count of docs/statistical-methods.md
    section 5. It is a planning figure: it assumes the rate stays where it is, and for a
    weighted or clustered design it applies to the effective sample size rather than the raw
    count. Returns ``None`` when more than ``maximum`` rollouts would be needed.
    """

    if not 0 <= rate <= 1:
        raise ValueError("rate must lie in [0, 1]")
    if not 0 < half_width < 1:
        raise ValueError("half_width must lie in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if exact_half_width(rate, float(maximum), alpha) > half_width:
        return None
    low, high = 1, maximum
    while low < high:
        middle = (low + high) // 2
        if exact_half_width(rate, float(middle), alpha) <= half_width:
            high = middle
        else:
            low = middle + 1
    return low


def rollout_scale_for_half_width(
    *,
    rate: float,
    observed_half_width: float,
    observed_trials: float,
    target_half_width: float,
    alpha: float,
    maximum_scale: float = 1e6,
) -> float | None:
    """Factor by which rollouts must grow to shrink an observed interval to a target width.

    The observed half-width is scaled by the ratio of Clopper-Pearson half-widths
    (``exact_half_width``) at the larger and the observed count, at ``rate``. The observed
    interval anchors the answer, whatever bound produced it, but the rate at which its width
    falls is the exact binomial bound's. For any other bound, such as a Chernoff, weighted or
    clustered one, that is an approximation: it narrows at a similar rate but not the same
    one, so the factor is a planning figure for it, not a guarantee. Returns ``1.0`` when the
    interval already meets the target and ``None`` when no factor up to ``maximum_scale``
    does.
    """

    if observed_half_width <= target_half_width:
        return 1.0
    if observed_trials <= 0 or not 0 < alpha < 1 or not 0 < target_half_width < 1:
        raise ValueError("observed_trials, alpha and target_half_width are out of range")
    rate = min(max(rate, 0.0), 1.0)
    reference = exact_half_width(rate, observed_trials, alpha)
    if reference <= 0:
        return None

    def projected(scale: float) -> float:
        return observed_half_width * exact_half_width(rate, observed_trials * scale, alpha) / (
            reference
        )

    if projected(maximum_scale) > target_half_width:
        return None
    low, high = 1.0, maximum_scale
    for _ in range(200):
        middle = math.sqrt(low * high)
        if projected(middle) <= target_half_width:
            high = middle
        else:
            low = middle
        if high / low < 1.001:
            break
    return high
