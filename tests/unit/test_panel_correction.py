"""Tests for multi-judge partial identification over declared accuracy boxes."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from inspect_safety_stats.stats import binomial_bounds
from inspect_safety_stats.stats.judges import (
    JudgeAccuracyBox,
    composite_accuracy_box,
    identified_risk_set,
)
from inspect_safety_stats.stats.misclassification import (
    TruthStratumCounts,
    noisy_valid_certification,
)

UNIT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def _box(se_lo: float, se_hi: float, sp_lo: float, sp_hi: float) -> JudgeAccuracyBox:
    return JudgeAccuracyBox(
        sensitivity_lower=se_lo,
        sensitivity_upper=se_hi,
        specificity_lower=sp_lo,
        specificity_upper=sp_hi,
    )


def test_accuracy_box_validates_ranges() -> None:
    with pytest.raises(ValueError, match="sensitivity"):
        _box(0.9, 0.8, 0.9, 0.95)
    with pytest.raises(ValueError, match="specificity"):
        _box(0.8, 0.9, 1.1, 1.2)


def test_independent_any_composite_matches_the_product_formula() -> None:
    a, b = _box(0.8, 0.8, 0.95, 0.95), _box(0.6, 0.6, 0.9, 0.9)
    derived = composite_accuracy_box([a, b], ["x", "y"], "any")
    # P(any flags | unsafe) = 1 - 0.2 * 0.4; P(none flags | safe) = 0.95 * 0.9.
    assert derived.sensitivity_lower == pytest.approx(0.92)
    assert derived.sensitivity_upper == pytest.approx(0.92)
    assert derived.specificity_lower == pytest.approx(0.855)
    assert derived.specificity_upper == pytest.approx(0.855)


def test_dependent_any_composite_uses_frechet_bounds_within_the_block() -> None:
    a, b = _box(0.8, 0.8, 0.95, 0.95), _box(0.6, 0.6, 0.9, 0.9)
    derived = composite_accuracy_box([a, b], ["x", "x"], "any")
    # Arbitrary dependence: sensitivity in [max, min(1, sum)], specificity in
    # [max(0, sum - 1), min].
    assert derived.sensitivity_lower == pytest.approx(0.8)
    assert derived.sensitivity_upper == pytest.approx(1.0)
    assert derived.specificity_lower == pytest.approx(0.85)
    assert derived.specificity_upper == pytest.approx(0.9)


def test_all_composite_formulas() -> None:
    a, b = _box(0.8, 0.8, 0.95, 0.95), _box(0.6, 0.6, 0.9, 0.9)
    independent = composite_accuracy_box([a, b], ["x", "y"], "all")
    assert independent.sensitivity_lower == pytest.approx(0.48)
    assert independent.specificity_lower == pytest.approx(1 - 0.05 * 0.1)
    dependent = composite_accuracy_box([a, b], ["x", "x"], "all")
    assert dependent.sensitivity_lower == pytest.approx(0.4)
    assert dependent.sensitivity_upper == pytest.approx(0.6)
    assert dependent.specificity_lower == pytest.approx(0.95)
    assert dependent.specificity_upper == pytest.approx(1.0)


def test_derived_box_is_monotone_in_the_declared_boxes() -> None:
    narrow = [_box(0.8, 0.85, 0.95, 0.97), _box(0.6, 0.7, 0.9, 0.95)]
    wide = [_box(0.7, 0.9, 0.9, 0.99), _box(0.5, 0.8, 0.85, 0.99)]
    for rule in ("any", "all"):
        for blocks in (["x", "y"], ["x", "x"]):
            n = composite_accuracy_box(narrow, blocks, rule)
            w = composite_accuracy_box(wide, blocks, rule)
            assert (
                w.sensitivity_lower
                <= n.sensitivity_lower
                <= n.sensitivity_upper
                <= w.sensitivity_upper
            )
            assert (
                w.specificity_lower
                <= n.specificity_lower
                <= n.specificity_upper
                <= w.specificity_upper
            )


def test_majority_composites_have_no_closed_form_and_are_refused() -> None:
    with pytest.raises(ValueError, match="majority"):
        composite_accuracy_box([_box(0.8, 0.9, 0.9, 0.95)] * 3, ["x", "y", "z"], "majority")
    with pytest.raises(ValueError, match="aligned"):
        composite_accuracy_box([_box(0.8, 0.9, 0.9, 0.95)] * 2, ["x"], "any")


def test_identified_set_of_a_perfect_judge_is_the_identity() -> None:
    result = identified_risk_set(0.04, 0.06, _box(1.0, 1.0, 1.0, 1.0))
    assert result.identified is True
    assert result.lower == pytest.approx(0.04)
    assert result.upper == pytest.approx(0.06)


def test_identified_set_inverts_rogan_gladen_at_a_point_box() -> None:
    result = identified_risk_set(0.05, 0.05, _box(0.9, 0.9, 0.98, 0.98))
    # (q - (1 - Sp)) / (Se + Sp - 1) = (0.05 - 0.02) / 0.88.
    assert result.lower == pytest.approx(0.03 / 0.88)
    assert result.upper == pytest.approx(0.03 / 0.88)


def test_identified_set_is_the_unit_interval_when_the_box_is_uninformative() -> None:
    result = identified_risk_set(0.05, 0.10, _box(0.4, 0.9, 0.5, 0.98))
    assert result.identified is False
    assert (result.lower, result.upper) == (0.0, 1.0)


def test_identified_set_widens_with_the_box_and_clips_to_the_unit_interval() -> None:
    narrow = identified_risk_set(0.05, 0.10, _box(0.85, 0.9, 0.96, 0.98))
    wide = identified_risk_set(0.05, 0.10, _box(0.7, 0.95, 0.9, 0.99))
    assert wide.lower <= narrow.lower <= narrow.upper <= wide.upper
    # A judge-reported interval reaching below the false-positive floor clips at zero.
    clipped = identified_risk_set(0.05, 0.12, _box(0.9, 0.9, 0.9, 0.9))
    assert clipped.lower == 0.0
    assert clipped.upper == pytest.approx(0.02 / 0.8)
    assert clipped.identified is True


def test_identified_set_is_unidentified_when_the_data_contradict_the_box() -> None:
    """A judge-reported rate below the box's false-positive floor fits no true risk at all.

    Clipping each end separately returned [0, -0.1125] marked as identified, and
    noisy-but-valid claims took their bounds from it.
    """
    below = identified_risk_set(0.0, 0.01, _box(0.9, 0.9, 0.9, 0.9))
    assert below.identified is False
    assert below.unidentified_reason == "contradicted"
    assert (below.lower, below.upper) == (0.0, 1.0)
    # A rate above the box's sensitivity ceiling contradicts it from the other side.
    above = identified_risk_set(0.95, 0.99, _box(0.8, 0.9, 0.9, 0.95))
    assert above.identified is False
    assert above.unidentified_reason == "contradicted"
    uninformative = identified_risk_set(0.05, 0.10, _box(0.4, 0.9, 0.5, 0.98))
    assert uninformative.unidentified_reason == "uninformative"


# The case clipping each end separately turned into [0, -0.1125].
@example(observed=[0.0, 0.01], sensitivity=[0.9, 0.9], specificity=[0.9, 0.9])
@given(
    observed=st.lists(UNIT, min_size=2, max_size=2).map(sorted),
    sensitivity=st.lists(UNIT, min_size=2, max_size=2).map(sorted),
    specificity=st.lists(UNIT, min_size=2, max_size=2).map(sorted),
)
def test_identified_set_is_never_inverted(
    observed: list[float], sensitivity: list[float], specificity: list[float]
) -> None:
    box = _box(sensitivity[0], sensitivity[1], specificity[0], specificity[1])
    result = identified_risk_set(observed[0], observed[1], box)
    assert 0.0 <= result.lower <= result.upper <= 1.0
    if not result.identified:
        assert (result.lower, result.upper) == (0.0, 1.0)


def test_noisy_valid_cannot_pass_when_the_judge_rate_contradicts_the_calibration() -> None:
    """No flags in 1,000 rollouts is impossible for a judge whose labels show a 10% false
    positive rate. The test used to PASS here on an inverted true-risk interval."""
    reported = binomial_bounds(0, 1000, alpha_lower=0.025, alpha_upper=0.025)
    result = noisy_valid_certification(
        judge_reported_lower=reported.lower,
        judge_reported_upper=reported.upper,
        judge_reported_alpha=0.025,
        truly_unsafe=TruthStratumCounts(labelled=50, judge_positive=45),
        truly_safe=TruthStratumCounts(labelled=200, judge_positive=20),
        calibration_alpha=0.05,
        acceptable_upper=0.10,
        unacceptable_lower=0.50,
    )
    assert result.identified is False
    assert result.decision == "INCONCLUSIVE"
    assert result.true_risk_lower <= result.true_risk_upper


def test_monte_carlo_coverage_under_maximal_within_block_dependence() -> None:
    rng = np.random.default_rng(6)
    prevalence, items, alpha = 0.10, 400, 0.05
    boxes = [_box(0.8, 0.9, 0.95, 0.99), _box(0.8, 0.9, 0.95, 0.99)]
    derived = composite_accuracy_box(boxes, ["x", "x"], "any")
    covered = 0
    replications = 400
    for _ in range(replications):
        unsafe = rng.random(items) < prevalence
        # Judge A has true accuracy inside its box; judge B is a perfect copy of A,
        # the most dependent panel the shared block allows.
        flag = np.where(unsafe, rng.random(items) < 0.85, rng.random(items) < 0.03)
        composite = flag  # any(A, A) == A
        bounds = binomial_bounds(int(composite.sum()), items, alpha_lower=alpha, alpha_upper=alpha)
        result = identified_risk_set(bounds.lower, bounds.upper, derived)
        covered += result.lower <= prevalence <= result.upper
    assert covered / replications >= 1 - 2 * alpha - 0.03
