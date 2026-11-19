from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from inspect_safety_stats.stats.binomial import binomial_bounds, binomial_lower_bound

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
    tail: bool = False,
) -> CorrectedRisk:
    """Map a judge-system risk interval onto the true-risk scale.

    Let ``R`` be the design-weighted probability that the judge reports a failure.
    Conditioning on the judge verdict partitions the design exactly, so

        R* = R * rho_1 + (1 - R) * rho_0

    is the corresponding true risk. The right-hand side is multilinear in
    ``(R, rho_1, rho_0)``, so its extrema over the declared box are attained at
    the box vertices and can be enumerated exactly.

    The identity is affine in ``R`` with slope ``rho_1 - rho_0``, positive for a strictly
    informative judge. It then keeps the order of prompts, so it applies unchanged to any
    risk measure that is positively homogeneous and translation equivariant -- weighted
    mean, maximum, CVaR, and top-k mean -- but not to entropic risk or threshold measures.

    A box with a completed stratum lets ``rho_1`` fall below ``rho_0``, and the map then
    reverses that order: a tail of true risk can rest on the prompts the judge flags least,
    which may have no judge-unsafe mass at all. Pass ``tail=True`` for a maximum, CVaR or
    top-k mean, and the upper end then also takes that mass as 0. A weighted mean needs no
    such vertex, since it is linear in ``R`` itself.

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
    upper = max(vertices)
    if tail and calibration.rho_1_lower < calibration.rho_0_upper:
        # No classified judge-unsafe mass, every judge-safe rollout truly unsafe at rho_0's
        # upper end, and every unclassified one unsafe as well.
        upper = max(upper, (1 - share) * calibration.rho_0_upper + share)
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
        upper=min(1.0, upper),
        completion_estimate_lower=estimate_lower,
        completion_estimate_upper=estimate_upper,
        scenario_id=calibration.scenario_id,
        source=calibration.source,
    )


def transported_sensitivity_lower(
    *, judge_flagged_unsafe: int, human_unsafe: int, alpha: float
) -> float:
    """One-sided exact lower bound on a judge's sensitivity from a published labelled set.

    ``human_unsafe`` items of the set are truly unsafe by human adjudication, and the judge
    flagged ``judge_flagged_unsafe`` of them. The bound is the Clopper-Pearson lower endpoint
    at ``alpha``; it describes the published set, and holds for another run's outputs only
    under the transport assumption the caller declares.
    """

    return binomial_lower_bound(judge_flagged_unsafe, human_unsafe, alpha=alpha)


def transport_judge_risk(
    *,
    observed_upper: float,
    observed_estimate: float | None,
    sensitivity_lower: float,
    scenario_id: str = "transported",
) -> CorrectedRisk:
    """Bound true risk from a transported sensitivity and the run's own judged rate.

    With true risk ``pi``, sensitivity ``Se`` and false-positive rate ``FPR``, the judged
    rate is ``q = pi Se + (1 - pi) FPR``. Nothing about this run's false positives is
    known except that they are among its judged failures, ``(1 - pi) FPR <= q``, so every
    judged failure may be a false positive and the lower bound is zero. The true failures
    the judge caught are at most ``q``, so ``pi <= q / Se``. With ``Se >= sensitivity_lower``
    and ``q <= observed_upper``, true risk is at most ``observed_upper / sensitivity_lower``.
    This is the Rogan-Gladen inversion ``pi = (q - 1 + Sp) / (Se + Sp - 1)`` (Rogan and
    Gladen, Am. J. Epidemiol. 107(1), 1978) at its maximising corner ``Sp = 1``: no
    specificity is declared, because a published set's specificity does not transport.

    Rollouts the judge never classified need no separate term: the observed upper bound
    already completes them as judge failures, and dividing by a sensitivity of at most one
    only enlarges that completion. The bound is monotone and positively homogeneous in the
    prompt rates, so it applies to the weighted mean and to the tail measures alike,
    provided the sensitivity bound holds on every prompt.
    """

    if not 0 <= observed_upper <= 1:
        raise ValueError("observed upper bound must be in [0, 1]")
    if observed_estimate is not None and not 0 <= observed_estimate <= 1:
        raise ValueError("observed estimate must be in [0, 1]")
    if not 0 <= sensitivity_lower <= 1:
        raise ValueError("the sensitivity lower bound must be in [0, 1]")

    def inflate(rate: float) -> float:
        # A judge that may catch nothing leaves true risk unbounded within [0, 1].
        if sensitivity_lower == 0:
            return 1.0
        return min(1.0, rate / sensitivity_lower)

    return CorrectedRisk(
        lower=0.0,
        upper=inflate(observed_upper),
        completion_estimate_lower=None if observed_estimate is None else 0.0,
        completion_estimate_upper=(
            None if observed_estimate is None else inflate(observed_estimate)
        ),
        scenario_id=scenario_id,
        source="transported",
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
    return _validated_calibration(
        scenario_id=scenario_id,
        rho_1=(rho_1_lower, rho_1_upper),
        rho_0=(rho_0_lower, rho_0_upper),
        alpha=alpha,
        completed=completed,
        bound=bound,
    )


def _validated_calibration(
    *,
    scenario_id: str,
    rho_1: tuple[float, float],
    rho_0: tuple[float, float],
    alpha: float,
    completed: tuple[str, ...],
    bound: CalibrationBound = "clopper_pearson",
) -> JudgeCalibration:
    """The measured box, or the refusal of one that does not establish an informative judge."""

    if not completed and rho_1[0] <= rho_0[1]:
        raise ValueError(
            "the validation sample does not establish a strictly informative judge: the "
            f"rho_1 lower bound ({rho_1[0]:.6g}) does not exceed the rho_0 upper bound "
            f"({rho_0[1]:.6g}); more labels or a better judge are required"
        )
    return JudgeCalibration(
        scenario_id=scenario_id,
        rho_1_lower=rho_1[0],
        rho_1_upper=rho_1[1],
        rho_0_lower=rho_0[0],
        rho_0_upper=rho_0[1],
        source="validated",
        alpha=alpha,
        completed_strata=completed,
        bound=bound,
    )


def hypergeometric_count_bounds(
    *, truly_unsafe: int, labelled: int, stratum_size: int, alpha_lower: float, alpha_upper: float
) -> tuple[int, int]:
    """Exact bounds on the number of truly unsafe rollouts in a finite stratum.

    The ``labelled`` rows are a simple random sample without replacement of the
    ``stratum_size`` rows, so ``truly_unsafe`` is hypergeometric given the unknown count
    ``K``. The lower bound is the smallest ``K`` whose upper tail ``P(X >= truly_unsafe)``
    exceeds ``alpha_lower``, the upper bound the largest ``K`` whose lower tail
    ``P(X <= truly_unsafe)`` exceeds ``alpha_upper``; both tails are monotone in ``K``, so
    each bound is one-sided at its level. A census gives the count exactly, and a stratum
    without labels is completed as ``[0, stratum_size]``, with certainty.
    """

    import numpy as np
    from scipy.stats import hypergeom

    counts = JudgeStratumCounts(
        labelled=labelled, truly_unsafe=truly_unsafe, stratum_size=stratum_size
    )
    if counts.labelled == 0:
        return 0, counts.stratum_size
    if counts.labelled == counts.stratum_size:
        return counts.truly_unsafe, counts.truly_unsafe
    for name, level in (("alpha_lower", alpha_lower), ("alpha_upper", alpha_upper)):
        if not 0 < level < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    candidates = np.arange(counts.stratum_size + 1)
    upper_tail = hypergeom.sf(truly_unsafe - 1, stratum_size, candidates, labelled)
    lower_tail = hypergeom.cdf(truly_unsafe, stratum_size, candidates, labelled)
    # Neither set below is empty: the upper tail is 1 at the largest count the labels admit,
    # stratum_size - (labelled - truly_unsafe), and the lower tail at the smallest,
    # truly_unsafe.
    lower = int(candidates[upper_tail > alpha_lower].min())
    upper = int(candidates[lower_tail > alpha_upper].max())
    return lower, upper


@dataclass(frozen=True)
class FineStratumBounds:
    """One declared fine stratum within a judge stratum, its labels and its count bounds.

    ``alpha`` is the level each one-sided count bound spends; it is zero for a census,
    which is exact, and for a stratum without labels, completed as ``[0, stratum_size]``.
    """

    judge_stratum: str
    stratum: str
    counts: JudgeStratumCounts
    unsafe_lower: int
    unsafe_upper: int
    alpha: float


@dataclass(frozen=True)
class StratifiedJudgeCalibration:
    """A validated calibration from labels drawn by a stratified design without replacement."""

    calibration: JudgeCalibration
    fine_strata: tuple[FineStratumBounds, ...]

    @property
    def unlabelled_fine_strata(self) -> tuple[FineStratumBounds, ...]:
        """The fine strata completed as ``[0, stratum_size]`` because none was labelled."""

        return tuple(item for item in self.fine_strata if item.counts.labelled == 0)


def estimate_stratified_judge_calibration(
    *,
    scenario_id: str,
    unsafe_strata: Mapping[str, JudgeStratumCounts],
    safe_strata: Mapping[str, JudgeStratumCounts],
    alpha: float,
) -> StratifiedJudgeCalibration:
    """Estimate rho_1 and rho_0 from labels in declared fine strata of each judge stratum.

    Each fine stratum ``h`` of a judge stratum of size ``N`` holds ``N_h`` rollouts, of which
    ``n_h`` were labelled as a simple random sample without replacement whose size was fixed
    independently of the labels. Its truly unsafe count ``K_h`` then has exact hypergeometric
    bounds, and the judge stratum's parameter is ``sum_h K_h / N``: each fine stratum is
    weighted back by its own size, whatever share of it was labelled.

    ``alpha`` is split as in :func:`estimate_judge_calibration`: a quarter per endpoint of
    the pair. An endpoint's quarter is split equally over the fine strata of its judge
    stratum that are labelled but not censused, so a union bound over every count bound
    spends no more than ``alpha``. A census is exact and an unlabelled fine stratum is
    completed as ``[0, N_h]``; neither spends any of it. A judge stratum without any label
    is completed as ``[0, 1]`` and named in ``completed_strata``; with both unlabelled the
    estimate is refused.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    endpoint_alpha = alpha / 4
    fine: list[FineStratumBounds] = []

    def bounds(judge_stratum: str, strata: Mapping[str, JudgeStratumCounts]) -> tuple[float, float]:
        total = sum(item.stratum_size for item in strata.values())
        if total == 0:
            raise ValueError(f"the {judge_stratum} judge stratum holds no rollout")
        sampled = sum(
            1 for item in strata.values() if 0 < item.labelled < item.stratum_size
        )
        level = endpoint_alpha / sampled if sampled else 0.0
        lower = upper = 0
        for name in sorted(strata):
            item = strata[name]
            if item.stratum_size == 0:
                raise ValueError(f"fine stratum {name!r} of {judge_stratum} holds no rollout")
            spent = level if 0 < item.labelled < item.stratum_size else 0.0
            low, high = hypergeometric_count_bounds(
                truly_unsafe=item.truly_unsafe,
                labelled=item.labelled,
                stratum_size=item.stratum_size,
                # A census or an unlabelled stratum returns before any level is read.
                alpha_lower=spent,
                alpha_upper=spent,
            )
            fine.append(FineStratumBounds(judge_stratum, name, item, low, high, spent))
            lower += low
            upper += high
        return lower / total, upper / total

    labelled = {
        name: sum(item.labelled for item in strata.values())
        for name, strata in (("judge_unsafe", unsafe_strata), ("judge_safe", safe_strata))
    }
    if not any(labelled.values()):
        raise ValueError("at least one judge stratum requires a human-labelled observation")
    rho_1 = bounds("judge_unsafe", unsafe_strata)
    rho_0 = bounds("judge_safe", safe_strata)
    completed = tuple(name for name, count in labelled.items() if count == 0)
    calibration = _validated_calibration(
        scenario_id=scenario_id, rho_1=rho_1, rho_0=rho_0, alpha=alpha, completed=completed
    )
    return StratifiedJudgeCalibration(calibration=calibration, fine_strata=tuple(fine))


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
