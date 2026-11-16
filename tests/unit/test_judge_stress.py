from __future__ import annotations

import itertools

import pytest

from inspect_safety_stats.stats.judge_stress import (
    stress_verdict,
    true_risk_bounds,
    verdict_lines,
)
from inspect_safety_stats.stats.judges import JudgeAccuracyBox


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


def test_unclassified_rollouts_stay_failures_at_the_upper_bound() -> None:
    # 2% of the weight is unclassified and completed as unsafe: only the classified 2.5%
    # above it is inverted, so the PASS needs Se >= 0.025 / 0.03 at perfect specificity.
    pass_line, _ = verdict_lines(
        judge_reported_lower=0.01,
        judge_reported_upper=0.045,
        acceptable_upper=0.05,
        unacceptable_lower=0.10,
        unclassified_share=0.02,
    )
    assert pass_line.sensitivity_at_perfect_specificity == pytest.approx(0.025 / 0.03, abs=1e-12)
    _, upper, _ = true_risk_bounds(
        judge_reported_lower=0.01,
        judge_reported_upper=0.045,
        box=point(0.5, 1.0),
        unclassified_share=0.02,
    )
    assert upper == pytest.approx(0.02 + 0.025 / 0.5, abs=1e-12)


def test_a_zero_limit_has_no_sensitivity_tipping_point() -> None:
    pass_line, _ = verdict_lines(
        judge_reported_lower=0.0,
        judge_reported_upper=0.0,
        acceptable_upper=0.0,
        unacceptable_lower=0.1,
    )
    assert pass_line.sensitivity_at_perfect_specificity is None


@pytest.mark.parametrize("share", [0.0, 0.01])
def test_the_stressed_verdict_agrees_with_the_tipping_lines(share: float) -> None:
    # Every point of the (Se, Sp) grid: PASS on true risk exactly where the PASS line says,
    # FAIL exactly where the FAIL line says.
    acceptable, unacceptable = 0.05, 0.10
    grid = [index / 40 for index in range(41)]
    for low, high in ((0.02, 0.045), (0.11, 0.16), (0.04, 0.12)):
        for se, sp in itertools.product(grid, grid):
            if se + sp <= 1:
                continue
            fpr = 1 - sp
            pass_margin = (acceptable - share) * se + (1 - acceptable) * fpr - (high - share)
            fail_margin = low - unacceptable * se - (1 - share - unacceptable) * fpr
            if min(abs(pass_margin), abs(fail_margin)) < 1e-9:
                continue
            stress = stress_verdict(
                judge_reported_lower=low,
                judge_reported_upper=high,
                acceptable_upper=acceptable,
                unacceptable_lower=unacceptable,
                box=point(se, sp),
                unclassified_share=share,
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
