"""Experimental outcome mapping sensitivities: alternative readings of the same score."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.config.models import AnalysisPlan

PRIMARY = {"failure_values": ["COMPLY", "PARTIAL"], "safe_values": ["REFUSE"]}
LENIENT = {"failure_values": ["COMPLY"], "safe_values": ["PARTIAL", "REFUSE"]}

BASE: dict[str, Any] = {
    "schema_version": "3.0",
    "analysis": {"id": "mappings", "title": "Outcome mappings", "intent": "exploratory"},
    "inputs": {
        "design_manifest": "design.csv",
        "score": {"name": "verdict", "mapping": PRIMARY},
    },
    "cohorts": {"candidate": {"match": {"model": "m"}}},
    "groups": [{"id": "all"}],
    "outputs": {
        "prompt_bounds": {"enabled": True, "family": "displayed"},
        "risk_measures": [{"id": "mean", "type": "weighted_mean"}],
    },
    "multiplicity": {
        "families": [
            {
                "id": "displayed",
                "purpose": "simultaneous_intervals",
                "method": "bonferroni",
                "alpha": 0.05,
            }
        ]
    },
    "claims": [],
}


def _plan(sensitivities: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    plan = copy.deepcopy(BASE)
    if sensitivities is not None:
        plan["inputs"]["score"]["mapping_sensitivities"] = sensitivities
    return plan


def test_a_plan_without_sensitivities_leaves_them_out_and_keeps_its_hash() -> None:
    plan = AnalysisPlan.model_validate(_plan())
    assert "mapping_sensitivities" not in plan.model_dump(mode="json")["inputs"]["score"]
    assert canonical_plan_hash(plan) == canonical_plan_hash(
        AnalysisPlan.model_validate(_plan([]))
    )


def test_declared_sensitivities_are_kept_in_the_plan_and_its_hash() -> None:
    declared = [{"id": "partial_safe", "title": "Partial refusal is safe", "mapping": LENIENT}]
    plan = AnalysisPlan.model_validate(_plan(declared))
    dumped = plan.model_dump(mode="json", by_alias=True)
    assert dumped["inputs"]["score"]["mapping_sensitivities"][0]["id"] == "partial_safe"
    assert AnalysisPlan.model_validate(dumped) == plan
    assert canonical_plan_hash(plan) != canonical_plan_hash(AnalysisPlan.model_validate(_plan()))


@pytest.mark.parametrize(
    ("sensitivities", "message"),
    [
        (
            [{"id": "a", "mapping": LENIENT}, {"id": "a", "mapping": LENIENT}],
            "identifiers must be unique",
        ),
        ([{"id": "primary", "mapping": LENIENT}], "names the primary mapping"),
        ([{"id": "same", "mapping": PRIMARY}], "repeats the primary mapping"),
        ([{"id": "has space", "mapping": LENIENT}], "id"),
        ([{"id": "a", "mapping": {"failure_values": ["COMPLY"]}}], "requires both"),
    ],
)
def test_invalid_sensitivities_are_rejected(
    sensitivities: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        AnalysisPlan.model_validate(_plan(sensitivities))


def test_a_judge_score_cannot_declare_sensitivities() -> None:
    plan = _plan()
    plan["inputs"]["judges"] = [
        {
            "id": "j",
            "score": {
                "name": "j",
                "mapping": PRIMARY,
                "mapping_sensitivities": [{"id": "a", "mapping": LENIENT}],
            },
        }
    ]
    with pytest.raises(ValidationError, match="applies to inputs.score only"):
        AnalysisPlan.model_validate(plan)
