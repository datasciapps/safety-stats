from __future__ import annotations

import itertools

import pytest

from inspect_safety_stats.stats.judge_stress import (
    stress_verdict,
    true_risk_bounds,
    verdict_lines,
)
from inspect_safety_stats.stats.judges import JudgeAccuracyBox, identified_risk_set


def point(sensitivity: float, specificity: float) -> JudgeAccuracyBox:
    return JudgeAccuracyBox(sensitivity, sensitivity, specificity, specificity)


def test_a_thin_pass_needs_a_nearly_perfect_judge() -> None:
    # Upper bound 4.98% against a 5% limit: at perfect specificity the judge must catch
    # 99.6% of failures for the PASS to hold on true risk.
    pass_line, fail_line = verdict_lines(
        judge_reported_lower=0.03,
        judge_reported_upper=0.0498,
        acceptable_upper=0.05,
        unacceptable_lower=0.10,
    )
    assert pass_line.sensitivity_at_perfect_specificity == pytest.approx(0.996, abs=1e-12)
    # Above 1: with a perfect sensitivity the PASS holds at every specificity.
    assert pass_line.specificity_at_perfect_sensitivity == pytest.approx(
        0.9502 / 0.95, abs=1e-12
    )
    assert fail_line.sensitivity_at_perfect_specificity == pytest.approx(0.3, abs=1e-12)


def test_a_judge_that_misses_one_failure_in_six_breaks_the_thin_pass() -> None:
    stress = stress_verdict(
        judge_reported_lower=0.03,
        judge_reported_upper=0.0498,
        acceptable_upper=0.05,
        unacceptable_lower=0.10,
        box=JudgeAccuracyBox(0.80, 0.87, 0.98, 1.0),
    )
    assert stress.stressed is not None
    assert stress.stressed.verdict == "INCONCLUSIVE"
    assert stress.stressed.true_risk_upper == pytest.approx(0.0498 / 0.80, abs=1e-12)


def test_a_perfect_judge_leaves_the_bounds_unchanged() -> None:
    lower, upper, identified = true_risk_bounds(
        judge_reported_lower=0.01, judge_reported_upper=0.04, box=point(1.0, 1.0)
    )
    assert lower == pytest.approx(0.01, rel=1e-12, abs=0)
    assert upper == pytest.approx(0.04, rel=1e-12, abs=0)
    assert identified


def test_an_uninformative_box_identifies_nothing() -> None:
    stress = stress_verdict(
        judge_reported_lower=0.0,
        judge_reported_upper=0.01,
        acceptable_upper=0.05,
        unacceptable_lower=0.10,
        box=JudgeAccuracyBox(0.4, 0.9, 0.6, 0.99),
    )
    assert stress.stressed is not None
    assert stress.stressed.identified is False
    assert stress.stressed.verdict == "INCONCLUSIVE"
    assert (stress.stressed.true_risk_lower, stress.stressed.true_risk_upper) == (0.0, 1.0)


def test_the_box_bounds_are_the_inversion_the_panel_correction_uses() -> None:
    boxes = [point(0.5, 1.0), point(0.95, 0.8), JudgeAccuracyBox(0.6, 0.9, 0.85, 0.99)]
    for box, (low, high) in itertools.product(
        boxes, [(0.0, 0.05), (0.2455, 0.466), (0.4, 0.55), (0.9, 1.0)]
    ):
        lower, upper, identified = true_risk_bounds(
            judge_reported_lower=low, judge_reported_upper=high, box=box
        )
        expected = identified_risk_set(low, high, box)
        assert (lower, upper, identified) == (expected.lower, expected.upper, expected.identified)


def test_a_tipping_line_never_points_the_wrong_way() -> None:
    # A maximum row with L = 0.2455 and U = 0.466 against a = 0.66: the PASS needs
    # Se >= U / a = 0.706 at perfect specificity, so the point box Se = 0.5 does not pass.
    pass_line, fail_line = verdict_lines(
        judge_reported_lower=0.2455,
        judge_reported_upper=0.466,
        acceptable_upper=0.66,
        unacceptable_lower=0.95,
    )
    assert pass_line.sensitivity_at_perfect_specificity == pytest.approx(
        0.466 / 0.66, rel=1e-12, abs=0
    )
    assert fail_line.specificity_at_perfect_sensitivity == pytest.approx(
        (1 - 0.2455) / 0.05, rel=1e-12, abs=0
    )
    stress = stress_verdict(
        judge_reported_lower=0.2455,
        judge_reported_upper=0.466,
        acceptable_upper=0.66,
        unacceptable_lower=0.95,
        box=point(0.5, 1.0),
    )
    assert stress.stressed is not None and stress.stressed.verdict == "INCONCLUSIVE"


def test_an_unacceptable_lower_of_one_has_no_specificity_tipping_point() -> None:
    # FAIL at perfect sensitivity needs L >= 1 whatever the specificity: all of the edge or
    # none of it.
    _, fail_line = verdict_lines(
        judge_reported_lower=0.6,
        judge_reported_upper=0.9,
        acceptable_upper=0.5,
        unacceptable_lower=1.0,
    )
    assert fail_line.specificity_at_perfect_sensitivity is None
    for sp in (0.2, 0.6, 1.0):
        stress = stress_verdict(
            judge_reported_lower=0.6,
            judge_reported_upper=0.9,
            acceptable_upper=0.5,
            unacceptable_lower=1.0,
            box=point(1.0, sp),
        )
        assert stress.stressed is not None and stress.stressed.verdict != "FAIL"


def test_a_zero_limit_has_no_sensitivity_tipping_point() -> None:
    pass_line, _ = verdict_lines(
        judge_reported_lower=0.0,
        judge_reported_upper=0.0,
        acceptable_upper=0.0,
        unacceptable_lower=0.1,
    )
    assert pass_line.sensitivity_at_perfect_specificity is None


def test_the_stressed_verdict_agrees_with_the_tipping_lines() -> None:
    # Every point of the (Se, Sp) grid: PASS on true risk exactly where the PASS line says,
    # FAIL exactly where the FAIL line says.
    acceptable, unacceptable = 0.05, 0.10
    grid = [index / 40 for index in range(41)]
    for low, high in ((0.02, 0.045), (0.11, 0.16), (0.04, 0.12), (0.2455, 0.466)):
        for se, sp in itertools.product(grid, grid):
            if se + sp <= 1:
                continue
            fpr = 1 - sp
            pass_margin = acceptable * se + (1 - acceptable) * fpr - high
            fail_margin = low - unacceptable * se - (1 - unacceptable) * fpr
            if min(abs(pass_margin), abs(fail_margin)) < 1e-9:
                continue
            stress = stress_verdict(
                judge_reported_lower=low,
                judge_reported_upper=high,
                acceptable_upper=acceptable,
                unacceptable_lower=unacceptable,
                box=point(se, sp),
            )
            assert stress.stressed is not None
            expected = "PASS" if pass_margin > 0 else "FAIL" if fail_margin > 0 else "INCONCLUSIVE"
            if stress.stressed.identified:
                assert stress.stressed.verdict == expected, (low, high, se, sp)


def test_invalid_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="judge-reported bounds"):
        verdict_lines(
            judge_reported_lower=0.2,
            judge_reported_upper=0.1,
            acceptable_upper=0.05,
            unacceptable_lower=0.1,
        )
    with pytest.raises(ValueError, match="acceptable_upper"):
        stress_verdict(
            judge_reported_lower=0.0,
            judge_reported_upper=0.1,
            acceptable_upper=0.1,
            unacceptable_lower=0.1,
            box=None,
        )
