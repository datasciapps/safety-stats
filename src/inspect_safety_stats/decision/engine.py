from __future__ import annotations

from inspect_safety_stats.schema.reports import Decision


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
