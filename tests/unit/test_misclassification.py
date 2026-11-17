import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inspect_safety_stats.stats import JudgeCalibration, correct_judge_risk
from inspect_safety_stats.stats.misclassification import (
    CorrectedRisk,
    JudgeStratumCounts,
    completion_warning,
    estimate_judge_calibration,
)


def _calibration(**kwargs: float) -> JudgeCalibration:
    defaults = dict(rho_1_lower=0.85, rho_1_upper=1.0, rho_0_lower=0.0, rho_0_upper=0.02)
    defaults.update(kwargs)
    return JudgeCalibration(scenario_id="s", **defaults)  # type: ignore[arg-type]


def test_perfect_judge_is_the_identity() -> None:
    result = correct_judge_risk(
        observed_lower=0.11,
        observed_upper=0.20,
        observed_estimate=0.15,
        calibration=JudgeCalibration("perfect", 1.0, 1.0, 0.0, 0.0),
    )
    assert result.lower == pytest.approx(0.11)
    assert result.upper == pytest.approx(0.20)
    assert result.completion_estimate_lower == pytest.approx(0.15)
    assert result.completion_estimate_upper == pytest.approx(0.15)


def test_correction_widens_and_lowers_under_false_positives() -> None:
    observed_lower, observed_upper = 0.040, 0.060
    result = correct_judge_risk(
        observed_lower=observed_lower,
        observed_upper=observed_upper,
        observed_estimate=0.05,
        calibration=_calibration(rho_1_lower=0.80, rho_0_upper=0.02),
    )
    # A judge that over-reports pulls the true-risk band down at the bottom.
    assert result.lower < observed_lower
    # And uncertainty about rho widens the band overall.
    assert result.upper - result.lower > observed_upper - observed_lower


def test_bounds_are_monotone_in_the_declared_accuracies() -> None:
    tight = correct_judge_risk(
        observed_lower=0.10,
        observed_upper=0.20,
        observed_estimate=0.15,
        calibration=_calibration(rho_0_upper=0.02),
    )
    loose = correct_judge_risk(
        observed_lower=0.10,
        observed_upper=0.20,
        observed_estimate=0.15,
        calibration=_calibration(rho_0_upper=0.10),
    )
    assert loose.upper > tight.upper
    assert loose.lower == pytest.approx(tight.lower)


def test_anti_informative_and_unordered_calibrations_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly informative"):
        JudgeCalibration("bad", 0.02, 1.0, 0.0, 0.05)
    with pytest.raises(ValueError, match="rho_1 must satisfy"):
        JudgeCalibration("bad", 0.9, 0.5, 0.0, 0.02)
    with pytest.raises(ValueError, match="observed bounds"):
        correct_judge_risk(
            observed_lower=0.5,
            observed_upper=0.2,
            observed_estimate=None,
            calibration=_calibration(),
        )


@settings(max_examples=200, deadline=None)
@given(
    observed_lower=st.floats(0.0, 1.0),
    width=st.floats(0.0, 1.0),
    rho_1_lower=st.floats(0.2, 1.0),
    rho_0_upper=st.floats(0.0, 0.19),
)
def test_vertex_enumeration_matches_a_dense_grid(
    observed_lower: float, width: float, rho_1_lower: float, rho_0_upper: float
) -> None:
    """R* is multilinear, so its box extrema must sit at the box corners."""
    observed_upper = min(1.0, observed_lower + width)
    calibration = JudgeCalibration("g", rho_1_lower, 1.0, 0.0, rho_0_upper)
    result = correct_judge_risk(
        observed_lower=observed_lower,
        observed_upper=observed_upper,
        observed_estimate=None,
        calibration=calibration,
    )
    steps = 9
    grid = []
    for i in range(steps):
        risk = observed_lower + (observed_upper - observed_lower) * i / (steps - 1)
        for j in range(steps):
            rho_1 = rho_1_lower + (1.0 - rho_1_lower) * j / (steps - 1)
            for k in range(steps):
                rho_0 = rho_0_upper * k / (steps - 1)
                grid.append(risk * rho_1 + (1 - risk) * rho_0)
    assert result.lower <= min(grid) + 1e-12
    assert result.upper >= max(grid) - 1e-12


def test_estimated_calibration_uses_exact_binomial_bounds() -> None:
    calibration = estimate_judge_calibration(
        scenario_id="validated",
        unsafe_stratum=JudgeStratumCounts(labelled=100, truly_unsafe=92, stratum_size=321),
        safe_stratum=JudgeStratumCounts(labelled=600, truly_unsafe=0, stratum_size=2879),
        alpha=0.05,
    )
    assert calibration.source == "validated"
    # Zero observed misses gives the exact one-sided rule 1 - alpha^(1/n) at alpha/4.
    assert calibration.rho_0_lower == 0.0
    assert calibration.rho_0_upper == pytest.approx(1 - (0.05 / 4) ** (1 / 600))
    assert calibration.rho_1_lower < 0.92 < calibration.rho_1_upper


def test_more_labels_tighten_the_estimated_calibration() -> None:
    def upper(labelled: int) -> float:
        return estimate_judge_calibration(
            scenario_id="v",
            unsafe_stratum=JudgeStratumCounts(labelled=100, truly_unsafe=92, stratum_size=321),
            safe_stratum=JudgeStratumCounts(labelled=labelled, truly_unsafe=0, stratum_size=2879),
            alpha=0.05,
        ).rho_0_upper

    assert upper(600) < upper(150) < upper(60)


def test_estimation_refuses_an_uninformative_validation_sample() -> None:
    with pytest.raises(ValueError, match="strictly informative"):
        estimate_judge_calibration(
            scenario_id="v",
            unsafe_stratum=JudgeStratumCounts(labelled=10, truly_unsafe=3, stratum_size=50),
            safe_stratum=JudgeStratumCounts(labelled=10, truly_unsafe=4, stratum_size=50),
            alpha=0.05,
        )
    with pytest.raises(ValueError, match="at least one judge stratum requires"):
        estimate_judge_calibration(
            scenario_id="v",
            unsafe_stratum=JudgeStratumCounts(labelled=0, truly_unsafe=0, stratum_size=50),
            safe_stratum=JudgeStratumCounts(labelled=0, truly_unsafe=0, stratum_size=50),
            alpha=0.05,
        )


@pytest.mark.parametrize("unlabelled", ["judge_unsafe", "judge_safe"])
def test_an_unlabelled_stratum_is_completed_as_the_unit_interval(unlabelled: str) -> None:
    """L49: a stratum without labels is bounded by [0, 1] instead of refusing."""

    empty = JudgeStratumCounts(labelled=0, truly_unsafe=0, stratum_size=7)
    unsafe = JudgeStratumCounts(labelled=40, truly_unsafe=38, stratum_size=60)
    safe = JudgeStratumCounts(labelled=300, truly_unsafe=0, stratum_size=900)
    calibration = estimate_judge_calibration(
        scenario_id="v",
        unsafe_stratum=empty if unlabelled == "judge_unsafe" else unsafe,
        safe_stratum=empty if unlabelled == "judge_safe" else safe,
        alpha=0.05,
    )
    assert calibration.completed_strata == (unlabelled,)
    assert calibration.alpha == 0.05
    completed = (
        (calibration.rho_1_lower, calibration.rho_1_upper)
        if unlabelled == "judge_unsafe"
        else (calibration.rho_0_lower, calibration.rho_0_upper)
    )
    assert completed == (0.0, 1.0)
    # The labelled stratum keeps its exact bound at the same endpoint budget.
    if unlabelled == "judge_unsafe":
        exact = 1 - (0.05 / 4) ** (1 / 300)
        assert calibration.rho_0_upper == pytest.approx(exact, rel=1e-12, abs=0)
    else:
        assert 0 < calibration.rho_1_lower < 0.95 < calibration.rho_1_upper < 1
    warning = completion_warning(calibration)
    assert warning is not None and unlabelled in warning and "[0, 1]" in warning


def test_a_completed_unsafe_stratum_bounds_true_risk_by_the_judge_safe_error() -> None:
    """With rho_1 in [0, 1], true risk lies in [(1 - U) rho_0_L, U + (1 - U) rho_0_U]."""

    calibration = estimate_judge_calibration(
        scenario_id="v",
        unsafe_stratum=JudgeStratumCounts(labelled=0, truly_unsafe=0, stratum_size=5),
        safe_stratum=JudgeStratumCounts(labelled=500, truly_unsafe=0, stratum_size=995),
        alpha=0.05,
    )
    result = correct_judge_risk(
        observed_lower=0.002,
        observed_upper=0.011,
        observed_estimate=0.005,
        calibration=calibration,
    )
    assert result.lower == 0.0
    expected = 0.011 + (1 - 0.011) * calibration.rho_0_upper
    assert result.upper == pytest.approx(expected, rel=1e-12, abs=0)


def test_a_tail_measure_reaches_rho_0_when_a_completed_box_reverses_the_judge() -> None:
    """rho_1 below rho_0 reverses the order of prompts, so a tail can rest on an unflagged one.

    Two prompts with judge-reported risk 0.7 and 0: the maximum is 0.7, but with rho_1 = 0.9
    and rho_0 = 1 their true risks are 0.97 and 1, so the true maximum is 1.
    """

    calibration = JudgeCalibration(
        "v", 0.2, 0.9, 0.0, 1.0, source="validated", completed_strata=("judge_safe",)
    )

    def corrected(*, tail: bool, share: float = 0.0) -> CorrectedRisk:
        return correct_judge_risk(
            observed_lower=0.6,
            observed_upper=0.8,
            observed_estimate=0.7,
            calibration=calibration,
            unclassified_share=share,
            tail=tail,
        )

    assert corrected(tail=True).upper == 1.0
    assert corrected(tail=True, share=0.1).upper == 1.0
    assert corrected(tail=True).lower == corrected(tail=False).lower
    # The mean's judge-unsafe mass is at least 0.6, and at most 0.9 of it is truly unsafe.
    assert corrected(tail=False).upper == pytest.approx(0.6 * 0.9 + 0.4, rel=0, abs=1e-12)


@pytest.mark.parametrize("share", [0.0, 0.05])
def test_tail_changes_nothing_for_a_strictly_informative_judge(share: float) -> None:
    def corrected(*, tail: bool) -> CorrectedRisk:
        return correct_judge_risk(
            observed_lower=0.10,
            observed_upper=0.20,
            observed_estimate=0.15,
            calibration=_calibration(),
            unclassified_share=share,
            tail=tail,
        )

    assert corrected(tail=True) == corrected(tail=False)


def test_calibrations_reject_an_unknown_completed_stratum() -> None:
    with pytest.raises(ValueError, match="unknown completed judge strata"):
        JudgeCalibration("g", 0.0, 1.0, 0.0, 0.01, completed_strata=("judge_maybe",))


def test_a_declared_box_without_completion_must_still_be_informative() -> None:
    with pytest.raises(ValueError, match="strictly informative"):
        JudgeCalibration("g", 0.0, 1.0, 0.0, 0.01)


def test_stratum_counts_reject_impossible_totals() -> None:
    with pytest.raises(ValueError, match="cannot exceed the labelled count"):
        JudgeStratumCounts(labelled=5, truly_unsafe=6, stratum_size=10)
    with pytest.raises(ValueError, match="cannot exceed the stratum size"):
        JudgeStratumCounts(labelled=20, truly_unsafe=1, stratum_size=10)


def test_unclassified_rollouts_keep_their_worst_case_on_the_true_scale() -> None:
    """#97: 3,200 rollouts, 1,000 of them unclassified.

    200 are judge-unsafe, of which 20 are truly unsafe, and 2,000 are judge-safe, none truly
    unsafe. The base bounds complete the unclassified rollouts as judge failures at the upper
    end. Mapping that completion through rho_1 used to give an upper bound of 0.064, below
    the 0.319 risk the unclassified rollouts allow on their own.
    """
    calibration = estimate_judge_calibration(
        scenario_id="validated",
        unsafe_stratum=JudgeStratumCounts(labelled=200, truly_unsafe=20, stratum_size=200),
        safe_stratum=JudgeStratumCounts(labelled=2000, truly_unsafe=0, stratum_size=2000),
        alpha=0.05,
    )
    worst_case = (20 + 1000) / 3200
    best_case = 20 / 3200
    result = correct_judge_risk(
        observed_lower=0.053,
        observed_upper=0.395,
        observed_estimate=200 / 2200,
        calibration=calibration,
        unclassified_share=1000 / 3200,
    )
    assert result.upper >= worst_case
    assert result.lower <= best_case
    # Only the classified mass is mapped through rho.
    assert result.upper == pytest.approx(
        (0.395 - 1000 / 3200) * calibration.rho_1_upper
        + (1 - 0.395) * calibration.rho_0_upper
        + 1000 / 3200
    )


def test_a_zero_unclassified_share_is_the_plain_correction() -> None:
    plain = correct_judge_risk(
        observed_lower=0.1, observed_upper=0.3, observed_estimate=0.2, calibration=_calibration()
    )
    explicit = correct_judge_risk(
        observed_lower=0.1,
        observed_upper=0.3,
        observed_estimate=0.2,
        calibration=_calibration(),
        unclassified_share=0.0,
    )
    assert explicit == plain


def test_an_unclassified_share_outside_the_unit_interval_is_rejected() -> None:
    for share in (-0.1, 1.1):
        with pytest.raises(ValueError, match="unclassified share"):
            correct_judge_risk(
                observed_lower=0.1,
                observed_upper=0.3,
                observed_estimate=None,
                calibration=_calibration(),
                unclassified_share=share,
            )


@settings(max_examples=200, deadline=None)
@given(
    classified_lower=st.floats(0.0, 1.0),
    width=st.floats(0.0, 1.0),
    share=st.floats(0.0, 1.0),
    rho_1_lower=st.floats(0.2, 1.0),
    rho_0_upper=st.floats(0.0, 0.19),
)
def test_the_corrected_interval_covers_every_completion_and_calibration(
    classified_lower: float, width: float, share: float, rho_1_lower: float, rho_0_upper: float
) -> None:
    """True risk is rho-mapped classified mass plus unclassified mass of any truth."""
    # Bounds whose completions are consistent: the classified judge-unsafe mass lies in
    # [lower, upper - share] and cannot exceed the classified share 1 - share.
    mass_lower = classified_lower * (1 - share)
    mass_upper = min(1 - share, mass_lower + width)
    observed_lower, observed_upper = mass_lower, mass_upper + share
    calibration = JudgeCalibration("g", rho_1_lower, 1.0, 0.0, rho_0_upper)
    result = correct_judge_risk(
        observed_lower=observed_lower,
        observed_upper=observed_upper,
        observed_estimate=None,
        calibration=calibration,
        unclassified_share=share,
    )
    steps = 7
    grid = []
    for i in range(steps):
        mass = mass_lower + (mass_upper - mass_lower) * i / (steps - 1)
        for j in range(steps):
            rho_1 = rho_1_lower + (1.0 - rho_1_lower) * j / (steps - 1)
            for k in range(steps):
                rho_0 = rho_0_upper * k / (steps - 1)
                for unknown in (0.0, 0.5, 1.0):
                    grid.append(mass * rho_1 + (1 - share - mass) * rho_0 + share * unknown)
    assert result.lower <= min(grid) + 1e-12
    assert result.upper >= max(grid) - 1e-12
