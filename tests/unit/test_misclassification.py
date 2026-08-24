import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inspect_safety_stats.stats import JudgeCalibration, correct_judge_risk


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
