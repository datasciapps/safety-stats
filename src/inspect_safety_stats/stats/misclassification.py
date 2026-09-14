from __future__ import annotations

import itertools
from dataclasses import dataclass

from inspect_safety_stats.stats.binomial import binomial_bounds


@dataclass(frozen=True)
class JudgeCalibration:
    """Conditional accuracy of a binary judge, as a declared or estimated box.

    ``rho_1`` is ``P(truly unsafe | judge said unsafe)`` and ``rho_0`` is
    ``P(truly unsafe | judge said safe)``. Both are properties of the judge on a
    particular output distribution, so they are prevalence-dependent and do not
    transfer unchanged to a cohort with a materially different failure rate.
    """

    scenario_id: str
    rho_1_lower: float
    rho_1_upper: float
    rho_0_lower: float
    rho_0_upper: float
    source: str = "assumed"
    alpha: float | None = None

    def __post_init__(self) -> None:
        for name, low, high in (
            ("rho_1", self.rho_1_lower, self.rho_1_upper),
            ("rho_0", self.rho_0_lower, self.rho_0_upper),
        ):
            if not 0 <= low <= high <= 1:
                raise ValueError(f"{name} must satisfy 0 <= lower <= upper <= 1")
        if self.rho_1_lower <= self.rho_0_upper:
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
    """

    if not 0 <= observed_lower <= observed_upper <= 1:
        raise ValueError("observed bounds must satisfy 0 <= lower <= upper <= 1")
    if observed_estimate is not None and not 0 <= observed_estimate <= 1:
        raise ValueError("observed estimate must be in [0, 1]")

    vertices = [
        risk * rho_1 + (1 - risk) * rho_0
        for risk, rho_1, rho_0 in itertools.product(
            (observed_lower, observed_upper),
            (calibration.rho_1_lower, calibration.rho_1_upper),
            (calibration.rho_0_lower, calibration.rho_0_upper),
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


def estimate_judge_calibration(
    *,
    scenario_id: str,
    unsafe_stratum: JudgeStratumCounts,
    safe_stratum: JudgeStratumCounts,
    alpha: float,
) -> JudgeCalibration:
    """Estimate rho_1 and rho_0 from human labels drawn within each judge stratum.

    Each stratum supplies an ordinary binomial proportion, so exact Clopper-Pearson
    bounds apply directly. ``alpha`` is the total error budget for the pair: it is
    split equally between the two parameters, and each parameter's budget is split
    again between its two endpoints, so a union bound over the four endpoints
    spends no more than ``alpha``.

    Validity requires the labelled rows to be a simple random sample within their
    stratum. That is a property of the collection protocol, not of the data, and is
    recorded as a declared assumption rather than checked here.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if unsafe_stratum.labelled == 0 or safe_stratum.labelled == 0:
        raise ValueError("both judge strata require at least one human-labelled observation")

    endpoint_alpha = alpha / 4
    rho_1 = binomial_bounds(
        unsafe_stratum.truly_unsafe,
        unsafe_stratum.labelled,
        alpha_lower=endpoint_alpha,
        alpha_upper=endpoint_alpha,
    )
    rho_0 = binomial_bounds(
        safe_stratum.truly_unsafe,
        safe_stratum.labelled,
        alpha_lower=endpoint_alpha,
        alpha_upper=endpoint_alpha,
    )
    if rho_1.lower <= rho_0.upper:
        raise ValueError(
            "the validation sample does not establish a strictly informative judge: the "
            f"rho_1 lower bound ({rho_1.lower:.6g}) does not exceed the rho_0 upper bound "
            f"({rho_0.upper:.6g}); more labels or a better judge are required"
        )
    return JudgeCalibration(
        scenario_id=scenario_id,
        rho_1_lower=rho_1.lower,
        rho_1_upper=rho_1.upper,
        rho_0_lower=rho_0.lower,
        rho_0_upper=rho_0.upper,
        source="validated",
        alpha=alpha,
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
