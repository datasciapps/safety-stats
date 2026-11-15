from __future__ import annotations

import pytest
from pydantic import ValidationError

from inspect_safety_stats.config.models import ComparisonSetSpec


def _set(**overrides: object) -> ComparisonSetSpec:
    values: dict[str, object] = {
        "id": "models",
        "cohorts": ["reference", "candidate_a", "candidate_b"],
        "contrasts": {"type": "versus_reference", "reference": "reference"},
        "estimands": ["risk_difference", "risk_ratio"],
        "group": "all",
        "measure": "mean",
        "method": "weighted_kl_chernoff",
        "family": "model_comparisons",
        "independence": {"primary": "rollout", "sensitivities": ["prompt"]},
    }
    values.update(overrides)
    return ComparisonSetSpec.model_validate(values)


def test_reference_set_expands_every_candidate_and_estimand() -> None:
    comparison_set = _set()
    expanded = comparison_set.expand()

    assert len(expanded) == 4
    assert {(item.candidate, item.baseline, item.estimand) for item in expanded} == {
        ("candidate_a", "reference", "risk_difference"),
        ("candidate_a", "reference", "risk_ratio"),
        ("candidate_b", "reference", "risk_difference"),
        ("candidate_b", "reference", "risk_ratio"),
    }
    assert all(item._comparison_set_id == "models" for item in expanded)
    assert all(item._joint_arm_region for item in expanded)


def test_all_pairs_expands_each_unordered_pair_once() -> None:
    comparison_set = _set(contrasts={"type": "all_pairs"}, estimands=["risk_difference"])
    assert comparison_set.pairs() == [
        ("candidate_a", "reference"),
        ("candidate_b", "reference"),
        ("candidate_b", "candidate_a"),
    ]


def test_a_two_arm_set_expands_to_one_contrast_per_estimand() -> None:
    comparison_set = _set(cohorts=["reference", "candidate"])
    assert [(item.candidate, item.baseline, item.estimand) for item in comparison_set.expand()] == [
        ("candidate", "reference", "risk_difference"),
        ("candidate", "reference", "risk_ratio"),
    ]


def test_comparison_set_requires_two_unique_cohorts_and_valid_reference() -> None:
    with pytest.raises(ValidationError, match="at least 2 items"):
        _set(cohorts=["reference"])
    with pytest.raises(ValidationError, match="cohorts must be unique"):
        _set(cohorts=["reference", "reference"])
    with pytest.raises(ValidationError, match="reference must be one of"):
        _set(contrasts={"type": "versus_reference", "reference": "missing"})
