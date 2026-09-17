"""Tests for multi-judge partial identification over declared accuracy boxes."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_safety_stats.stats import binomial_bounds
from inspect_safety_stats.stats.judges import (
    JudgeAccuracyBox,
    composite_accuracy_box,
    identified_risk_set,
)


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
    # A judge-reported rate below the false-positive floor clips at zero.
    clipped = identified_risk_set(0.0, 0.01, _box(0.9, 0.9, 0.9, 0.9))
    assert clipped.lower == 0.0
    assert clipped.identified is True


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
