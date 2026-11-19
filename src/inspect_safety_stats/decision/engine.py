from __future__ import annotations

from inspect_safety_stats.schema.reports import (
    AdaptiveClaimResult,
    ClaimResult,
    Decision,
    VerdictRange,
)
from inspect_safety_stats.stats.misclassification import NoisyValidCertification

# The thresholds a plan allows a claim (ClaimSpec): acceptable_upper < unacceptable_lower,
# both in [-1, 1].
THRESHOLD_MIN = -1.0
THRESHOLD_MAX = 1.0


def claim_verdict_range(
    *,
    lower: float | None,
    upper: float | None,
    eligible: bool,
) -> VerdictRange:
    """Invert ``decide_claim``: the thresholds at which these bounds give PASS or FAIL.

    PASS needs ``upper <= acceptable_upper`` and FAIL ``lower >= unacceptable_lower``, so the
    boundaries are the bounds themselves. A boundary that leaves no valid threshold pair, such
    as an upper bound of 1, gives no range.
    """

    if not eligible or lower is None or upper is None:
        return VerdictRange()
    return VerdictRange(
        pass_acceptable_upper=upper if upper < THRESHOLD_MAX else None,
        fail_unacceptable_lower=lower if lower > THRESHOLD_MIN else None,
    )


def noisy_valid_verdict_range(result: NoisyValidCertification) -> VerdictRange:
    """Invert the noisy-but-valid test: the thresholds at which its outcome is PASS or FAIL.

    The test passes when the judge-reported upper bound is below
    ``c_pass = a * Se_L + (1 - a) * FPR_L``, which rises in ``a`` for an informative judge, so
    it passes for every ``acceptable_upper`` strictly above
    ``(judge_upper - FPR_L) / (Se_L - FPR_L)``. FAIL mirrors this with the upper calibration
    endpoints. None of this depends on the thresholds except the critical values, so an
    unidentified judge gives no range.

    The test takes thresholds in [0, 1] only, and the range speaks only of those. Its FAIL
    boundary reads as a half-line, but below it the test still needs ``acceptable_upper >= 0``.
    """

    if not result.identified:
        return VerdictRange()
    # For an identified judge both boundaries lie in [0, 1] algebraically, but identification
    # reads the false-positive rate through specificity, 1 - FPR, so rounding can leave a
    # boundary a few ulps outside. It is then clipped to the end of [0, 1], inclusive.
    pass_value: float | None = None
    pass_strict = False
    pass_slope = result.sensitivity_lower - result.false_positive_lower
    if pass_slope > 0:
        boundary = (result.judge_reported_upper - result.false_positive_lower) / pass_slope
        if boundary < 0:
            pass_value = 0.0
        elif boundary < 1:
            pass_value, pass_strict = boundary, True
    fail_value: float | None = None
    fail_strict = False
    fail_slope = result.sensitivity_upper - result.false_positive_upper
    if fail_slope > 0:
        boundary = (result.judge_reported_lower - result.false_positive_upper) / fail_slope
        if boundary > 1:
            fail_value = 1.0
        elif boundary > 0:
            fail_value, fail_strict = boundary, True
    return VerdictRange(
        pass_acceptable_upper=pass_value,
        pass_strict=pass_strict,
        fail_unacceptable_lower=fail_value,
        fail_strict=fail_strict,
    )


def decide_claim(
    *,
    lower: float | None,
    upper: float | None,
    acceptable_upper: float,
    unacceptable_lower: float,
    eligible: bool,
) -> tuple[Decision, list[str]]:
    if not eligible or lower is None or upper is None:
        return "INCONCLUSIVE", ["No complete certifying bound is available."]
    if upper <= acceptable_upper:
        return "PASS", [f"Upper bound {upper:.12g} is at or below {acceptable_upper:.12g}."]
    if lower >= unacceptable_lower:
        return "FAIL", [f"Lower bound {lower:.12g} is at or above {unacceptable_lower:.12g}."]
    return "INCONCLUSIVE", ["Bounds do not establish either the acceptable or unacceptable region."]


def mandatory_statuses(
    claims: list[ClaimResult], adaptive_claims: list[AdaptiveClaimResult]
) -> list[Decision]:
    """The statuses ``roll_up`` takes: every mandatory claim's, threshold and adaptive alike."""

    return [claim.status for claim in claims if claim.mandatory] + [
        claim.status for claim in adaptive_claims if claim.mandatory
    ]


def roll_up(
    statuses: list[Decision],
    *,
    integrity_valid: bool,
    zero_tolerance_failure: bool,
    confirmatory: bool,
) -> Decision:
    if zero_tolerance_failure:
        return "FAIL"
    if not integrity_valid or not confirmatory:
        return "INCONCLUSIVE"
    if not statuses:
        return "INCONCLUSIVE"
    if "FAIL" in statuses:
        return "FAIL"
    if "INCONCLUSIVE" in statuses:
        return "INCONCLUSIVE"
    return "PASS"
