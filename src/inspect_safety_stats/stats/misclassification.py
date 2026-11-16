from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

from inspect_safety_stats.stats.binomial import binomial_bounds

CalibrationBound = Literal["clopper_pearson", "confidence_sequence"]


@dataclass(frozen=True)
class JudgeCalibration:
    """Conditional accuracy of a binary judge, as a declared or estimated box.

    ``rho_1`` is ``P(truly unsafe | judge said unsafe)`` and ``rho_0`` is
    ``P(truly unsafe | judge said safe)``. Both are properties of the judge on a
    particular output distribution, so they are prevalence-dependent and do not
    transfer unchanged to a cohort with a materially different failure rate.

    ``completed_strata`` names the judge strata, ``judge_unsafe`` or ``judge_safe``, whose
    parameter no label measured and which the box therefore completes as ``[0, 1]``. Such a
    box cannot be strictly informative, so the informativeness check applies only when no
    stratum is completed.

    ``bound`` names how a validated box bounds each measured stratum: exact Clopper-Pearson
    intervals, or anytime-valid confidence sequences for labelling that may stop on results.
    """

    scenario_id: str
    rho_1_lower: float
    rho_1_upper: float
    rho_0_lower: float
    rho_0_upper: float
    source: str = "assumed"
    alpha: float | None = None
    completed_strata: tuple[str, ...] = ()
    bound: CalibrationBound = "clopper_pearson"

    def __post_init__(self) -> None:
        for name, low, high in (
            ("rho_1", self.rho_1_lower, self.rho_1_upper),
            ("rho_0", self.rho_0_lower, self.rho_0_upper),
        ):
            if not 0 <= low <= high <= 1:
                raise ValueError(f"{name} must satisfy 0 <= lower <= upper <= 1")
        unknown = set(self.completed_strata) - {"judge_unsafe", "judge_safe"}
        if unknown:
            raise ValueError(f"unknown completed judge strata: {sorted(unknown)}")
        if not self.completed_strata and self.rho_1_lower <= self.rho_0_upper:
            raise ValueError(
                "the judge must be strictly informative: rho_1 lower endpoint must exceed "
                "the rho_0 upper endpoint"
            )


@dataclass(frozen=True)
class CorrectedRisk:
    """A judge-system risk interval mapped onto the underlying true risk."""

    lower: float
    upper: float
    completion_estimate_lower: float | None
    completion_estimate_upper: float | None
    scenario_id: str
    source: str


def correct_judge_risk(
    *,
    observed_lower: float,
    observed_upper: float,
    observed_estimate: float | None,
    calibration: JudgeCalibration,
    unclassified_share: float = 0.0,
) -> CorrectedRisk:
    """Map a judge-system risk interval onto the true-risk scale.

    Let ``R`` be the design-weighted probability that the judge reports a failure.
    Conditioning on the judge verdict partitions the design exactly, so

        R* = R * rho_1 + (1 - R) * rho_0

    is the corresponding true risk. The right-hand side is multilinear in
    ``(R, rho_1, rho_0)``, so its extrema over the declared box are attained at
    the box vertices and can be enumerated exactly.

    The identity is affine in ``R`` with positive slope ``rho_1 - rho_0``. It
    therefore applies unchanged to any risk measure that is
    positively homogeneous and translation equivariant -- weighted mean, maximum,
    CVaR, and top-k mean -- but not to entropic risk or threshold measures.

    ``unclassified_share`` is the weight on rollouts with no judge verdict, which the
    observed bounds complete as judge-safe at the lower end and judge-unsafe at the
    upper. The labels say nothing about those rollouts, so they keep that completion on
    the true-risk scale. Only the classified judge-unsafe mass, which the bounds place in
    ``[observed_lower, observed_upper - unclassified_share]``, is mapped through rho:

        R* = M * rho_1 + (1 - s - M) * rho_0 + s * u,   u in [0, 1].

    For a tail measure, pass the largest share of any prompt it weights. The adjustment
    is monotone and translation equivariant, so that bounds the tail too.
    """

    if not 0 <= observed_lower <= observed_upper <= 1:
        raise ValueError("observed bounds must satisfy 0 <= lower <= upper <= 1")
    if observed_estimate is not None and not 0 <= observed_estimate <= 1:
        raise ValueError("observed estimate must be in [0, 1]")
    if not 0 <= unclassified_share <= 1:
        raise ValueError("the unclassified share must be in [0, 1]")

    share = unclassified_share
    # The classified judge-unsafe mass cannot exceed the classified share.
    mass_lower = min(observed_lower, 1 - share)
    mass_upper = max(mass_lower, observed_upper - share)
    vertices = [
        mass * rho_1 + (1 - share - mass) * rho_0 + share * unknown
        for mass, rho_1, rho_0, unknown in itertools.product(
            (mass_lower, mass_upper),
            (calibration.rho_1_lower, calibration.rho_1_upper),
            (calibration.rho_0_lower, calibration.rho_0_upper),
            (0.0, 1.0),
        )
    ]
    estimate_lower = estimate_upper = None
    if observed_estimate is not None:
        estimate_lower = (
            observed_estimate * calibration.rho_1_lower
            + (1 - observed_estimate) * calibration.rho_0_lower
        )
        estimate_upper = (
            observed_estimate * calibration.rho_1_upper
            + (1 - observed_estimate) * calibration.rho_0_upper
        )
    return CorrectedRisk(
        lower=max(0.0, min(vertices)),
        upper=min(1.0, max(vertices)),
        completion_estimate_lower=estimate_lower,
        completion_estimate_upper=estimate_upper,
        scenario_id=calibration.scenario_id,
        source=calibration.source,
    )


@dataclass(frozen=True)
class JudgeStratumCounts:
    """Human-labelled counts within one judge verdict stratum."""

    labelled: int
    truly_unsafe: int
    stratum_size: int

    def __post_init__(self) -> None:
        for name, value in (
            ("labelled", self.labelled),
            ("truly_unsafe", self.truly_unsafe),
            ("stratum_size", self.stratum_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.truly_unsafe > self.labelled:
            raise ValueError("truly_unsafe cannot exceed the labelled count")
        if self.labelled > self.stratum_size:
            raise ValueError("labelled count cannot exceed the stratum size")


def hypergeometric_confidence_sequence(
    stratum: JudgeStratumCounts, *, alpha: float
) -> tuple[float, float]:
    """An anytime-valid range for the share of truly unsafe rows in a finite stratum.

    The labelled rows are drawn one at a time without replacement, in a random order, from
    a stratum of ``N = stratum_size`` rows of which an unknown ``K`` are truly unsafe. With
    a uniform prior on ``K`` in ``{0, ..., N}``, the ratio of prior to posterior mass at the
    true ``K`` is a nonnegative martingale starting at one (the prior-posterior ratio
    martingale of Waudby-Smith and Ramdas, "Confidence sequences for sampling without
    replacement", NeurIPS 2020). By Ville's inequality it ever exceeds ``1 / alpha`` with
    probability at most ``alpha``, so the set of ``K`` whose posterior-to-prior ratio
    exceeds ``alpha`` holds the true ``K`` at every sample size simultaneously. Labelling may
    therefore stop at any time, also because of the labels seen so far.

    After ``n`` labels with ``S`` unsafe, the posterior-to-prior ratio at ``K`` is
    ``(n + 1) * P(S | N, K, n)``, with ``P`` the hypergeometric probability, because the
    uniform prior's marginal likelihood of ``S`` is ``1 / (n + 1)``. That probability is
    log-concave in ``K``, so the set is an interval. It is returned as shares of ``N``. With
    no labels it is ``[0, 1]``, and a census pins the share exactly.

    ``alpha`` bounds the probability that the range ever misses on either side, and so each
    endpoint's too.
    """

    import numpy as np
    from scipy.stats import hypergeom

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    size = stratum.stratum_size
    if stratum.labelled == 0:
        return 0.0, 1.0
    candidates = np.arange(size + 1)
    log_ratio = np.log(stratum.labelled + 1) + hypergeom.logpmf(
        stratum.truly_unsafe, size, candidates, stratum.labelled
    )
    kept = candidates[log_ratio > np.log(alpha)]
    if kept.size == 0:
        # The ratio at the maximum-likelihood K is at least one, so this cannot happen.
        raise ValueError("the confidence sequence is empty; the counts are inconsistent")
    return float(kept[0]) / size, float(kept[-1]) / size


def estimate_judge_calibration(
    *,
    scenario_id: str,
    unsafe_stratum: JudgeStratumCounts,
    safe_stratum: JudgeStratumCounts,
    alpha: float,
    bound: CalibrationBound = "clopper_pearson",
) -> JudgeCalibration:
    """Estimate rho_1 and rho_0 from human labels drawn within each judge stratum.

    Each stratum supplies an ordinary binomial proportion, so exact Clopper-Pearson
    bounds apply directly. ``alpha`` is the total error budget for the pair: it is
    split equally between the two parameters, and each parameter's budget is split
    again between its two endpoints, so a union bound over the four endpoints
    spends no more than ``alpha``.

    Validity requires the labelled rows to be a simple random sample within their
    stratum, of a size that did not depend on the labels. That is a property of the
    collection protocol, not of the data, and is recorded as a declared assumption rather
    than checked here; a plan's labelling protocol checks what the data can show.

    ``bound="confidence_sequence"`` replaces each Clopper-Pearson interval with the
    anytime-valid range of ``hypergeometric_confidence_sequence`` at the same endpoint
    budget, which stays valid when labelling stopped because of the labels seen so far.

    A stratum without labels is completed conservatively: its parameter is a
    probability, so ``[0, 1]`` contains it with certainty and spends none of ``alpha``.
    The result names that stratum in ``completed_strata`` and is not required to be
    strictly informative, which such a box cannot be. With both strata unlabelled
    nothing is measured, and the estimate is refused.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if unsafe_stratum.labelled == 0 and safe_stratum.labelled == 0:
        raise ValueError("at least one judge stratum requires a human-labelled observation")

    endpoint_alpha = alpha / 4

    def bounds(stratum: JudgeStratumCounts) -> tuple[float, float]:
        if stratum.labelled == 0:
            return 0.0, 1.0
        if bound == "confidence_sequence":
            # Two-sided at the endpoint budget, so each endpoint keeps its alpha / 4.
            return hypergeometric_confidence_sequence(stratum, alpha=endpoint_alpha)
        interval = binomial_bounds(
            stratum.truly_unsafe,
            stratum.labelled,
            alpha_lower=endpoint_alpha,
            alpha_upper=endpoint_alpha,
        )
        return interval.lower, interval.upper

    rho_1_lower, rho_1_upper = bounds(unsafe_stratum)
    rho_0_lower, rho_0_upper = bounds(safe_stratum)
    completed = tuple(
        name
        for name, stratum in (("judge_unsafe", unsafe_stratum), ("judge_safe", safe_stratum))
        if stratum.labelled == 0
    )
    if not completed and rho_1_lower <= rho_0_upper:
        raise ValueError(
            "the validation sample does not establish a strictly informative judge: the "
            f"rho_1 lower bound ({rho_1_lower:.6g}) does not exceed the rho_0 upper bound "
            f"({rho_0_upper:.6g}); more labels or a better judge are required"
        )
    return JudgeCalibration(
        scenario_id=scenario_id,
        rho_1_lower=rho_1_lower,
        rho_1_upper=rho_1_upper,
        rho_0_lower=rho_0_lower,
        rho_0_upper=rho_0_upper,
        source="validated",
        alpha=alpha,
        completed_strata=completed,
        bound=bound,
    )


def completion_warning(calibration: JudgeCalibration) -> str | None:
    """Say which strata a calibration completed as ``[0, 1]``, or ``None`` if none."""

    if not calibration.completed_strata:
        return None
    names = " and ".join(calibration.completed_strata)
    return (
        f"the {names} judge stratum has no human-labelled observation, so its error rate "
        "is completed conservatively as [0, 1]; labelling it would narrow the correction"
    )


@dataclass(frozen=True)
class DeltaCorrectedRisk:
    """Asymptotic (delta-method) interval for true risk with estimated judge accuracies."""

    estimate: float
    standard_error: float
    lower: float
    upper: float
    alpha: float
    rho_1_estimate: float
    rho_0_estimate: float


def delta_method_corrected_risk(
    *,
    observed_estimate: float,
    observed_variance: float,
    unsafe_stratum: JudgeStratumCounts,
    safe_stratum: JudgeStratumCounts,
    alpha: float,
) -> DeltaCorrectedRisk:
    """Wald interval for ``R* = R rho_1 + (1 - R) rho_0`` propagating estimation variance.

    The stratum proportions are Agresti-Coull stabilised, ``(x + 2) / (n + 4)``, and
    the variance is the first-order expansion

        (rho_1 - rho_0)^2 Var(R) + R^2 Var(rho_1) + (1 - R)^2 Var(rho_0),

    with a normal quantile at ``alpha / 2``. This is an asymptotic reference: it is
    usually much narrower than the exact vertex box and shows what the finite-sample
    guarantee costs, but it carries no coverage guarantee of its own and is reported
    as a diagnostic, never as evidence.
    """

    from scipy.stats import norm

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 <= observed_estimate <= 1:
        raise ValueError("observed estimate must be in [0, 1]")
    if not observed_variance >= 0:
        raise ValueError("observed variance must be nonnegative")
    if unsafe_stratum.labelled == 0 or safe_stratum.labelled == 0:
        raise ValueError("both judge strata require at least one labelled observation")

    def stabilised(stratum: JudgeStratumCounts) -> tuple[float, float]:
        size = stratum.labelled + 4
        proportion = (stratum.truly_unsafe + 2) / size
        return proportion, proportion * (1 - proportion) / size

    rho_1, var_1 = stabilised(unsafe_stratum)
    rho_0, var_0 = stabilised(safe_stratum)
    estimate = observed_estimate * rho_1 + (1 - observed_estimate) * rho_0
    variance = (
        (rho_1 - rho_0) ** 2 * observed_variance
        + observed_estimate**2 * var_1
        + (1 - observed_estimate) ** 2 * var_0
    )
    standard_error = variance**0.5
    z = float(norm.ppf(1 - alpha / 2))
    return DeltaCorrectedRisk(
        estimate=estimate,
        standard_error=standard_error,
        lower=max(0.0, estimate - z * standard_error),
        upper=min(1.0, estimate + z * standard_error),
        alpha=alpha,
        rho_1_estimate=rho_1,
        rho_0_estimate=rho_0,
    )


@dataclass(frozen=True)
class TruthStratumCounts:
    """Judge verdicts among human-labelled rollouts of one true class.

    Valid as a binomial sample of the judge's sensitivity (truly unsafe items) or
    false-positive rate (truly safe items) only when the labelled rollouts are a simple
    random sample of the design, or were sampled by true class -- not within judge
    strata, which would over-represent the judge's own positives.
    """

    labelled: int
    judge_positive: int

    def __post_init__(self) -> None:
        for name, value in (("labelled", self.labelled), ("judge_positive", self.judge_positive)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.judge_positive > self.labelled:
            raise ValueError("judge_positive cannot exceed the labelled count")


@dataclass(frozen=True)
class NoisyValidCertification:
    """Outcome of the finite-sample noisy-but-valid threshold test."""

    decision: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    judge_reported_lower: float
    judge_reported_upper: float
    sensitivity_lower: float
    sensitivity_upper: float
    false_positive_lower: float
    false_positive_upper: float
    critical_threshold_pass: float
    critical_threshold_fail: float
    true_risk_lower: float
    true_risk_upper: float
    identified: bool
    alpha_pass: float
    alpha_fail: float
    unidentified_reason: Literal["uninformative", "contradicted"] | None = None


def noisy_valid_certification(
    *,
    judge_reported_lower: float,
    judge_reported_upper: float,
    judge_reported_alpha: float,
    truly_unsafe: TruthStratumCounts,
    truly_safe: TruthStratumCounts,
    calibration_alpha: float,
    acceptable_upper: float,
    unacceptable_lower: float,
) -> NoisyValidCertification:
    """Certify a true-risk threshold from judge labels plus a random human-labelled sample.

    With sensitivity ``Se``, false-positive rate ``FPR`` and true risk ``pi``, the
    judge-reported rate is ``q = pi Se + (1 - pi) FPR``, increasing in ``pi`` whenever
    the judge is informative. Under ``H0: pi >= acceptable`` therefore
    ``q >= acceptable * Se + (1 - acceptable) * FPR``, and with exact lower confidence
    bounds on ``Se`` and ``FPR`` the right-hand side is at least the critical threshold
    ``c_pass = acceptable * Se_L + (1 - acceptable) * FPR_L``. PASS rejects ``H0`` when
    the judge-reported upper bound falls below ``c_pass``; a union bound over the three
    exact endpoints gives Type-I error at most ``judge_reported_alpha +
    calibration_alpha / 2``. FAIL mirrors this with the upper calibration endpoints.

    The same box also yields the identified true-risk set, so the decision and the
    interval agree by construction. When the calibration cannot establish an
    informative judge (``Se + Sp <= 1`` admitted by the box) nothing identifies ``pi``
    and the outcome is INCONCLUSIVE.
    """

    from inspect_safety_stats.stats.judges import JudgeAccuracyBox, identified_risk_set

    if not 0 <= judge_reported_lower <= judge_reported_upper <= 1:
        raise ValueError("judge-reported bounds must satisfy 0 <= lower <= upper <= 1")
    if not 0 < judge_reported_alpha < 1 or not 0 < calibration_alpha < 1:
        raise ValueError("alpha levels must be in (0, 1)")
    if not 0 <= acceptable_upper < unacceptable_lower <= 1:
        raise ValueError("acceptable_upper must be below unacceptable_lower, both in [0, 1]")
    if truly_unsafe.labelled == 0 or truly_safe.labelled == 0:
        raise ValueError("both truth strata require at least one labelled observation")

    endpoint = calibration_alpha / 4
    sensitivity = binomial_bounds(
        truly_unsafe.judge_positive,
        truly_unsafe.labelled,
        alpha_lower=endpoint,
        alpha_upper=endpoint,
    )
    false_positive = binomial_bounds(
        truly_safe.judge_positive,
        truly_safe.labelled,
        alpha_lower=endpoint,
        alpha_upper=endpoint,
    )
    c_pass = acceptable_upper * sensitivity.lower + (1 - acceptable_upper) * false_positive.lower
    c_fail = (
        unacceptable_lower * sensitivity.upper + (1 - unacceptable_lower) * false_positive.upper
    )
    box = JudgeAccuracyBox(
        sensitivity_lower=sensitivity.lower,
        sensitivity_upper=sensitivity.upper,
        specificity_lower=1 - false_positive.upper,
        specificity_upper=1 - false_positive.lower,
    )
    identified = identified_risk_set(judge_reported_lower, judge_reported_upper, box)
    decision: Literal["PASS", "FAIL", "INCONCLUSIVE"] = "INCONCLUSIVE"
    if identified.identified:
        if judge_reported_upper < c_pass:
            decision = "PASS"
        elif judge_reported_lower > c_fail:
            decision = "FAIL"
    return NoisyValidCertification(
        decision=decision,
        judge_reported_lower=judge_reported_lower,
        judge_reported_upper=judge_reported_upper,
        sensitivity_lower=sensitivity.lower,
        sensitivity_upper=sensitivity.upper,
        false_positive_lower=false_positive.lower,
        false_positive_upper=false_positive.upper,
        critical_threshold_pass=c_pass,
        critical_threshold_fail=c_fail,
        true_risk_lower=identified.lower,
        true_risk_upper=identified.upper,
        identified=identified.identified,
        alpha_pass=judge_reported_alpha + calibration_alpha / 2,
        alpha_fail=judge_reported_alpha + calibration_alpha / 2,
        unidentified_reason=identified.unidentified_reason,
    )
