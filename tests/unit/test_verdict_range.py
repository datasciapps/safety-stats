"""Verdict ranges: the thresholds at which a claim would PASS or FAIL, read off its bounds.

Each range is checked against the decision it inverts, over a grid of the threshold pairs a
plan allows, so the range and the verdict cannot drift apart.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from inspect_safety_stats.decision.engine import (
    claim_verdict_range,
    decide_claim,
    noisy_valid_verdict_range,
)
from inspect_safety_stats.report.writer import _number, _verdict_range
from inspect_safety_stats.schema.reports import Decision, VerdictRange
from inspect_safety_stats.stats.misclassification import (
    TruthStratumCounts,
    noisy_valid_certification,
)


def _predicted(verdict_range: VerdictRange, acceptable: float, unacceptable: float) -> Decision:
    def beyond(value: float, boundary: float | None, strict: bool, above: bool) -> bool:
        if boundary is None:
            return False
        if above:
            return value > boundary if strict else value >= boundary
        return value < boundary if strict else value <= boundary

    passes = beyond(
        acceptable, verdict_range.pass_acceptable_upper, verdict_range.pass_strict, True
    )
    fails = beyond(
        unacceptable, verdict_range.fail_unacceptable_lower, verdict_range.fail_strict, False
    )
    assert not (passes and fails), "a valid threshold pair cannot both pass and fail"
    return "PASS" if passes else "FAIL" if fails else "INCONCLUSIVE"


def _pairs(low: float, high: float, extra: tuple[float, ...] = ()) -> list[tuple[float, float]]:
    grid = sorted({*np.round(np.linspace(low, high, 41), 12).tolist(), *extra})
    return [(a, u) for a, u in itertools.combinations(grid, 2)]


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(0.0, 0.01), (0.03, 0.05), (0.005, 0.04), (0.2, 0.2), (-0.3, 0.1), (0.0, 1.0), (-1.0, -0.5)],
)
def test_interval_claim_range_agrees_with_the_decision_at_every_threshold(
    lower: float, upper: float
) -> None:
    verdict_range = claim_verdict_range(lower=lower, upper=upper, eligible=True)
    # The bounds themselves are on the grid, where PASS and FAIL are attained.
    for acceptable, unacceptable in _pairs(-1.0, 1.0, (lower, upper)):
        decided, _ = decide_claim(
            lower=lower,
            upper=upper,
            acceptable_upper=acceptable,
            unacceptable_lower=unacceptable,
            eligible=True,
        )
        assert _predicted(verdict_range, acceptable, unacceptable) == decided, (
            acceptable,
            unacceptable,
        )


def test_interval_claim_range_is_the_bounds() -> None:
    assert claim_verdict_range(lower=0.02, upper=0.07, eligible=True) == VerdictRange(
        pass_acceptable_upper=0.07, fail_unacceptable_lower=0.02
    )


def test_bounds_at_the_ends_of_the_threshold_range_leave_no_verdict() -> None:
    # acceptable_upper < unacceptable_lower <= 1, so an upper bound of 1 never passes.
    assert claim_verdict_range(lower=-1.0, upper=1.0, eligible=True) == VerdictRange()


@pytest.mark.parametrize(
    ("lower", "upper", "eligible"), [(0.0, 0.01, False), (None, 0.01, True), (0.0, None, True)]
)
def test_an_undecidable_claim_is_inconclusive_at_every_threshold(
    lower: float | None, upper: float | None, eligible: bool
) -> None:
    assert claim_verdict_range(lower=lower, upper=upper, eligible=eligible) == VerdictRange()


@pytest.mark.parametrize(
    ("q_lower", "q_upper", "unsafe", "safe"),
    [
        (0.02, 0.06, (40, 36), (200, 4)),
        (0.30, 0.45, (40, 36), (200, 4)),
        (0.0, 0.01, (40, 36), (200, 4)),
        (0.85, 0.95, (40, 36), (200, 4)),
        (0.10, 0.40, (10, 10), (30, 0)),
        (0.10, 0.20, (10, 4), (10, 5)),
    ],
)
def test_noisy_valid_range_agrees_with_the_test_at_every_threshold(
    q_lower: float, q_upper: float, unsafe: tuple[int, int], safe: tuple[int, int]
) -> None:
    def certify(acceptable: float, unacceptable: float):  # type: ignore[no-untyped-def]
        return noisy_valid_certification(
            judge_reported_lower=q_lower,
            judge_reported_upper=q_upper,
            judge_reported_alpha=0.025,
            truly_unsafe=TruthStratumCounts(*unsafe),
            truly_safe=TruthStratumCounts(*safe),
            calibration_alpha=0.05,
            acceptable_upper=acceptable,
            unacceptable_lower=unacceptable,
        )

    verdict_range = noisy_valid_verdict_range(certify(0.1, 0.3))
    boundaries = [
        value
        for value in (verdict_range.pass_acceptable_upper, verdict_range.fail_unacceptable_lower)
        if value is not None
    ]
    checked = refused = 0
    unit = tuple(np.round(np.linspace(0.0, 1.0, 41), 12).tolist())
    # Every pair a plan allows. The test takes only pairs in [0, 1], and the range speaks only
    # of those: elsewhere the test does not run, and the claim is INCONCLUSIVE with no range.
    for acceptable, unacceptable in _pairs(-1.0, 1.0, unit):
        # A boundary is a quotient, so its own grid point is left to rounding.
        if any(abs(value - boundary) < 1e-9 for value in (acceptable, unacceptable)
               for boundary in boundaries):
            continue
        try:
            decided = certify(acceptable, unacceptable).decision
        except ValueError:
            assert acceptable < 0, (acceptable, unacceptable)
            refused += 1
            continue
        assert _predicted(verdict_range, acceptable, unacceptable) == decided, (
            acceptable,
            unacceptable,
        )
        checked += 1
    assert checked > 700
    assert refused > 0


def test_noisy_valid_range_inverts_the_critical_thresholds() -> None:
    result = noisy_valid_certification(
        judge_reported_lower=0.30,
        judge_reported_upper=0.45,
        judge_reported_alpha=0.025,
        truly_unsafe=TruthStratumCounts(40, 36),
        truly_safe=TruthStratumCounts(200, 4),
        calibration_alpha=0.05,
        acceptable_upper=0.1,
        unacceptable_lower=0.3,
    )
    verdict_range = noisy_valid_verdict_range(result)
    pass_at = verdict_range.pass_acceptable_upper
    fail_at = verdict_range.fail_unacceptable_lower
    assert pass_at is not None and fail_at is not None
    assert verdict_range.pass_strict and verdict_range.fail_strict
    # At its boundary each critical value equals the judge-reported endpoint it is tested on.
    assert pass_at * result.sensitivity_lower + (
        1 - pass_at
    ) * result.false_positive_lower == pytest.approx(0.45, rel=1e-12, abs=0)
    assert fail_at * result.sensitivity_upper + (
        1 - fail_at
    ) * result.false_positive_upper == pytest.approx(0.30, rel=1e-12, abs=0)


def test_an_unidentified_judge_is_inconclusive_at_every_threshold() -> None:
    result = noisy_valid_certification(
        judge_reported_lower=0.10,
        judge_reported_upper=0.20,
        judge_reported_alpha=0.025,
        truly_unsafe=TruthStratumCounts(10, 4),
        truly_safe=TruthStratumCounts(10, 5),
        calibration_alpha=0.05,
        acceptable_upper=0.1,
        unacceptable_lower=0.3,
    )
    assert not result.identified
    assert noisy_valid_verdict_range(result) == VerdictRange()


@pytest.mark.parametrize(
    ("verdict_range", "shown"),
    [
        (None, "—"),
        (VerdictRange(), "INCONCLUSIVE at every threshold"),
        (
            VerdictRange(pass_acceptable_upper=0.01234, fail_unacceptable_lower=0.00567),
            "PASS at acceptable ≥ 0.013; FAIL at unacceptable ≤ 0.005",
        ),
        (
            VerdictRange(
                pass_acceptable_upper=0.25,
                pass_strict=True,
                fail_unacceptable_lower=0.1,
                fail_strict=True,
            ),
            "PASS at acceptable > 0.250; FAIL at unacceptable < 0.100",
        ),
        # A strict boundary that rounding moves is not attained, so it reads as inclusive.
        (
            VerdictRange(pass_acceptable_upper=0.2501, pass_strict=True),
            "PASS at acceptable ≥ 0.251",
        ),
        (VerdictRange(fail_unacceptable_lower=-0.0004), "FAIL at unacceptable ≤ -0.001"),
        (VerdictRange(pass_acceptable_upper=1e-17), "PASS at acceptable ≥ 0.001"),
    ],
)
def test_the_markdown_range_rounds_each_boundary_so_that_it_holds(
    verdict_range: VerdictRange | None, shown: str
) -> None:
    assert _verdict_range(verdict_range, 3) == shown


@pytest.mark.parametrize("places", [7, 10, 15])
def test_the_markdown_range_is_fixed_point_at_every_precision(places: int) -> None:
    # A zero-failure claim has a lower bound of 0, which str() of a Decimal wrote as 0E-7.
    shown = _verdict_range(
        VerdictRange(pass_acceptable_upper=3e-17, fail_unacceptable_lower=0.0), places
    )
    assert shown == (
        f"PASS at acceptable ≥ {_number(10.0**-places, places)}; "
        f"FAIL at unacceptable ≤ {_number(0.0, places)}"
    )
