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


def exact_half_width(rate: float, trials: float, alpha: float) -> float:
    """Larger one-sided Clopper-Pearson distance from ``rate`` at a real-valued count."""

    from scipy.stats import beta as beta_distribution

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

    The count is the smallest ``n`` at which both one-sided Clopper-Pearson endpoints,
    each at level ``alpha``, lie within ``half_width`` of the assumed failure rate. It
    is a planning figure: it assumes the rate stays where it is, and for a weighted or
    clustered design it applies to the effective sample size rather than the raw count.
    Returns ``None`` when more than ``maximum`` rollouts would be needed.
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

    The observed interval anchors the answer, so it holds for whichever concentration
    bound produced it; only the rate at which width falls with more rollouts is taken
    from the exact binomial bound at ``rate``. Returns ``1.0`` when the interval already
    meets the target and ``None`` when no factor up to ``maximum_scale`` does.
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


def _planned_upper(rate: float, labels: int, alpha: float) -> float:
    """Clopper-Pearson upper bound at an anticipated rate, with a real-valued count."""

    from scipy.stats import beta as beta_distribution

    if labels <= 0:
        return 1.0
    failures = rate * labels
    if failures >= labels:
        return 1.0
    return float(beta_distribution.ppf(1.0 - alpha, failures + 1.0, labels - failures))


def _planned_lower(rate: float, labels: int, alpha: float) -> float:
    from scipy.stats import beta as beta_distribution

    if labels <= 0:
        return 0.0
    failures = rate * labels
    if failures <= 0:
        return 0.0
    return float(beta_distribution.ppf(alpha, failures, labels - failures + 1.0))


@dataclass(frozen=True)
class JudgeLabelPlan:
    """A labelling budget for validated judge correction, and what it buys.

    ``corrected_upper`` is the true-risk upper bound that the planned labels would
    give at the anticipated judge accuracies, ``rho_0 + U (rho_1 - rho_0)`` with
    every parameter at its upper endpoint. ``floor`` is the same quantity with
    unlimited labels, the best the correction can do for this judge.
    """

    judge_unsafe_labels: int
    judge_safe_labels: int
    rho_1_lower: float
    rho_1_upper: float
    rho_0_upper: float
    corrected_upper: float
    floor: float
    endpoint_alpha: float


def judge_label_plan(
    *,
    judge_risk_upper: float,
    target_upper: float,
    alpha: float,
    cohorts: int = 1,
    anticipated_rho_1: float = 1.0,
    anticipated_rho_0: float = 0.0,
    maximum_labels: int = 1_000_000,
) -> JudgeLabelPlan | None:
    """Cheapest labels per cohort for a judge-corrected upper bound below ``target_upper``.

    ``judge_risk_upper`` is the uncorrected (judge-reported) upper bound the analysis
    expects. The judge-family ``alpha`` is divided among ``cohorts`` and then across the
    four Clopper-Pearson endpoints exactly as validated calibration spends it. Labels in
    the judge-safe stratum bound ``rho_0`` (missed failures); labels in the judge-unsafe
    stratum bound ``rho_1`` (how many reported failures are real), which is what lets a
    correction fall below the judge-reported risk. Returns ``None`` when no budget up to
    ``maximum_labels`` per stratum reaches the target.
    """

    for name, value in (
        ("judge_risk_upper", judge_risk_upper),
        ("anticipated_rho_1", anticipated_rho_1),
        ("anticipated_rho_0", anticipated_rho_0),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if not 0 < target_upper < 1:
        raise ValueError("target_upper must lie in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if cohorts < 1:
        raise ValueError("cohorts must be at least 1")
    if anticipated_rho_1 <= anticipated_rho_0:
        raise ValueError("the anticipated judge must be informative: rho_1 must exceed rho_0")

    endpoint = alpha / cohorts / 4
    floor = anticipated_rho_0 + judge_risk_upper * (anticipated_rho_1 - anticipated_rho_0)
    if floor >= target_upper:
        return None

    def corrected(unsafe: int, safe: int) -> float:
        rho_1 = _planned_upper(anticipated_rho_1, unsafe, endpoint)
        rho_0 = _planned_upper(anticipated_rho_0, safe, endpoint)
        return rho_0 + judge_risk_upper * (rho_1 - rho_0)

    def informative(unsafe: int, safe: int) -> bool:
        return _planned_lower(anticipated_rho_1, unsafe, endpoint) > _planned_upper(
            anticipated_rho_0, safe, endpoint
        )

    def safe_labels_for(unsafe: int) -> int | None:
        if corrected(unsafe, maximum_labels) > target_upper or not informative(
            unsafe, maximum_labels
        ):
            return None
        low, high = 1, maximum_labels
        while low < high:
            middle = (low + high) // 2
            if corrected(unsafe, middle) <= target_upper and informative(unsafe, middle):
                high = middle
            else:
                low = middle + 1
        return low

    candidates = sorted({1, 2, 3, 5, 8, *{int(round(10 * 1.25**k)) for k in range(70)}})
    best: tuple[int, int, int] | None = None
    for unsafe in (value for value in candidates if value <= maximum_labels):
        safe = safe_labels_for(unsafe)
        if safe is None:
            continue
        total = unsafe + safe
        if best is None or total < best[0]:
            best = (total, unsafe, safe)
    if best is None:
        return None
    _, unsafe, safe = best
    return JudgeLabelPlan(
        judge_unsafe_labels=unsafe,
        judge_safe_labels=safe,
        rho_1_lower=_planned_lower(anticipated_rho_1, unsafe, endpoint),
        rho_1_upper=_planned_upper(anticipated_rho_1, unsafe, endpoint),
        rho_0_upper=_planned_upper(anticipated_rho_0, safe, endpoint),
        corrected_upper=corrected(unsafe, safe),
        floor=floor,
        endpoint_alpha=endpoint,
    )


def kl_rate_bounds(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """Weighted KL/Chernoff bounds for ``trials`` equally weighted binary outcomes.

    This is the analysis's ``weighted_kl_chernoff`` engine in its equal-atomic-weight
    form, each endpoint at level ``alpha``.
    """

    from inspect_safety_stats.stats.direct_mean import _bernoulli_kl, _invert_divergence

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("counts must satisfy 0 <= successes <= trials and trials > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    observed = successes / trials
    target = math.log(1 / alpha) / trials
    return (
        _invert_divergence(observed, target, direction="lower", divergence=_bernoulli_kl),
        _invert_divergence(observed, target, direction="upper", divergence=_bernoulli_kl),
    )


def _support(trials: int, rate: float) -> np.ndarray:
    spread = 12 * math.sqrt(max(trials * rate * (1 - rate), 1.0))
    centre = trials * rate
    low = max(0, int(math.floor(centre - spread)) - 2)
    high = min(trials, int(math.ceil(centre + spread)) + 2)
    return np.arange(low, high + 1)


def comparison_power(
    *,
    candidate_rate: float,
    baseline_rate: float,
    rollouts_per_arm: int,
    endpoint_alpha: float,
    margin: float | None = None,
) -> float:
    """Probability that the two-arm KL rectangle establishes the planned conclusion.

    The rectangle bounds the difference by ``[L_C - U_B, U_C - L_B]``, each arm endpoint
    at ``endpoint_alpha``. With a ``margin`` the conclusion is non-inferiority,
    ``U_C - L_B < margin``; without one it is a resolved difference in the true direction
    (the interval excludes zero). Each arm is ``rollouts_per_arm`` equally weighted
    rollouts at a common rate. Heterogeneous prompt rates make each arm's count less
    variable than a binomial (Hoeffding, 1956), so the power computed here is conservative
    for the declared design. The sum over both arms' counts is exact on a support that
    excludes less than 1e-20 of the probability.
    """

    for name, value in (("candidate_rate", candidate_rate), ("baseline_rate", baseline_rate)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if rollouts_per_arm < 1:
        raise ValueError("rollouts_per_arm must be positive")
    if not 0 < endpoint_alpha < 1:
        raise ValueError("endpoint_alpha must lie in (0, 1)")
    difference = candidate_rate - baseline_rate
    if margin is None and difference == 0:
        raise ValueError("equal rates leave no difference to detect; give a margin instead")
    if margin is not None and difference >= margin:
        raise ValueError(
            "non-inferiority at this margin is not true at the anticipated rates "
            f"(difference {difference:.4g} >= margin {margin:.4g})"
        )

    n = rollouts_per_arm
    candidate = _support(n, candidate_rate)
    baseline = _support(n, baseline_rate)
    candidate_mass = binom.pmf(candidate, n, candidate_rate)
    baseline_mass = binom.pmf(baseline, n, baseline_rate)
    baseline_bounds = np.asarray(
        [kl_rate_bounds(int(y), n, endpoint_alpha) for y in baseline], dtype=float
    )
    baseline_lower, baseline_upper = baseline_bounds[:, 0], baseline_bounds[:, 1]
    # Tail mass above each index, for "baseline count at or beyond y".
    above = np.concatenate([np.cumsum(baseline_mass[::-1])[::-1], [0.0]])
    below = np.concatenate([[0.0], np.cumsum(baseline_mass)])
    power = 0.0
    for x, mass in zip(candidate, candidate_mass, strict=True):
        if mass < 1e-300:
            continue
        candidate_lower, candidate_upper = kl_rate_bounds(int(x), n, endpoint_alpha)
        if margin is not None:
            # success iff L_B(y) > U_C(x) - margin; L_B is nondecreasing in y.
            first = int(np.searchsorted(baseline_lower, candidate_upper - margin, side="right"))
            power += mass * above[first]
        elif difference > 0:
            # success iff L_C(x) - U_B(y) > 0, i.e. U_B(y) < L_C(x).
            last = int(np.searchsorted(baseline_upper, candidate_lower, side="left"))
            power += mass * below[last]
        else:
            # success iff U_C(x) - L_B(y) < 0, i.e. L_B(y) > U_C(x).
            first = int(np.searchsorted(baseline_lower, candidate_upper, side="right"))
            power += mass * above[first]
    return float(min(1.0, max(0.0, power)))


@dataclass(frozen=True)
class ComparisonPlan:
    rollouts_per_arm: int
    achieved_power: float
    endpoint_alpha: float


def comparison_rollouts(
    *,
    candidate_rate: float,
    baseline_rate: float,
    power: float,
    endpoint_alpha: float,
    margin: float | None = None,
    maximum: int = 2_000_000,
) -> ComparisonPlan | None:
    """Rollouts per arm for the two-arm rectangle to reach ``power``.

    Power is not exactly monotone in the rollout count because counts are discrete;
    the search returns the smallest count found by doubling and bisection whose power
    meets the target, which is the planning figure that matters.
    """

    if not 0 < power < 1:
        raise ValueError("power must lie in (0, 1)")

    def achieved(count: int) -> float:
        return comparison_power(
            candidate_rate=candidate_rate,
            baseline_rate=baseline_rate,
            rollouts_per_arm=count,
            endpoint_alpha=endpoint_alpha,
            margin=margin,
        )

    high = 8
    while achieved(high) < power:
        high *= 2
        if high > maximum:
            return None
    low = max(1, high // 2)
    while low < high:
        middle = (low + high) // 2
        if achieved(middle) >= power:
            high = middle
        else:
            low = middle + 1
    return ComparisonPlan(
        rollouts_per_arm=high, achieved_power=achieved(high), endpoint_alpha=endpoint_alpha
    )
