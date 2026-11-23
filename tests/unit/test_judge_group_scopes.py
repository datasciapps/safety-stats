"""Plan rules for scoping judge calibration, declared accuracies and panel diagnostics to groups.

A scope names declared groups, is never empty, and is left out of the resolved plan while
absent, so the plans written before it keep their hash and their report bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from safety_stats.config.loader import canonical_plan_hash
from safety_stats.config.models import AnalysisPlan
from safety_stats.generative.schema import load_generative_plan
from safety_stats.validate.plan import validate_plan_references
from tests.generative_fixtures import SEEDS, base_plan, materialize

BODY = """
analysis: {id: scoped, title: Scoped, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
  judge_calibration:
    mode: assumed
    primary: declared
    scenarios:
      - {id: declared, rho_1: {lower: 0.9, upper: 1.0}, rho_0: {lower: 0.0, upper: 0.1}}
  judges:
    - id: first
      score: {name: first, mapping: {failure_values: [FAIL], safe_values: [PASS]}}
      accuracy:
        sensitivity: {lower: 0.8, upper: 0.9}
        specificity: {lower: 0.9, upper: 0.95}
    - id: second
      score: {name: second, mapping: {failure_values: [FAIL], safe_values: [PASS]}}
cohorts:
  candidate: {match: {model: model/candidate}}
groups:
  - {id: harm, filter: {task_id: harm}}
  - {id: refusal, filter: {task_id: refusal}}
outputs: {prompt_bounds: {enabled: false}}
multiplicity: {families: []}
claims: []
"""


def _plan(**scopes: Any) -> AnalysisPlan:
    data = yaml.safe_load(BODY)
    inputs = data["inputs"]
    if "calibration" in scopes:
        inputs["judge_calibration"]["groups"] = scopes["calibration"]
    if "accuracy" in scopes:
        inputs["judges"][0]["accuracy"]["groups"] = scopes["accuracy"]
    if "panel" in scopes:
        inputs["panel_groups"] = scopes["panel"]
    return AnalysisPlan.model_validate(data)


def test_absent_scopes_are_left_out_of_the_resolved_plan() -> None:
    inputs = _plan().model_dump(mode="json")["inputs"]
    assert "panel_groups" not in inputs
    assert "groups" not in inputs["judge_calibration"]
    assert "groups" not in inputs["judges"][0]["accuracy"]
    # So the hash is that of a plan written before the scopes existed...
    unscoped = canonical_plan_hash(_plan())
    # ...and a scope, once set, is part of what the hash covers.
    for scope in ("calibration", "accuracy", "panel"):
        scoped = _plan(**{scope: ["harm"]})
        assert canonical_plan_hash(scoped) != unscoped


def test_set_scopes_are_recorded() -> None:
    inputs = _plan(calibration=["harm"], accuracy=["harm"], panel=["harm", "refusal"]).model_dump(
        mode="json"
    )["inputs"]
    assert inputs["judge_calibration"]["groups"] == ["harm"]
    assert inputs["judges"][0]["accuracy"]["groups"] == ["harm"]
    assert inputs["panel_groups"] == ["harm", "refusal"]


@pytest.mark.parametrize("scope", ["calibration", "accuracy", "panel"])
def test_a_scope_is_never_empty(scope: str) -> None:
    with pytest.raises(ValidationError, match="groups cannot be empty"):
        _plan(**{scope: []})


@pytest.mark.parametrize("scope", ["calibration", "accuracy", "panel"])
def test_a_scope_names_each_group_once(scope: str) -> None:
    with pytest.raises(ValidationError, match="groups must be unique"):
        _plan(**{scope: ["harm", "harm"]})


def test_panel_groups_need_declared_judges() -> None:
    data = yaml.safe_load(BODY)
    data["inputs"]["judges"] = []
    data["inputs"]["panel_groups"] = ["harm"]
    with pytest.raises(ValidationError, match="panel_groups requires declared judges"):
        AnalysisPlan.model_validate(data)


def test_a_scope_naming_an_undeclared_group_is_a_plan_reference_error() -> None:
    plan = _plan(calibration=["harm", "cyber"], accuracy=["typo"], panel=["refusal"])
    unknown = [
        (item.scope, item.message)
        for item in validate_plan_references(plan)
        if item.code == "PLAN_UNKNOWN_REFERENCE"
    ]
    assert unknown == [
        (
            "plan/inputs/judge_calibration",
            "inputs.judge_calibration references unknown groups: cyber",
        ),
        (
            "plan/inputs/judges/first/accuracy",
            "inputs.judges.first.accuracy references unknown groups: typo",
        ),
    ]


def test_a_transported_calibration_takes_no_group_scope() -> None:
    data = yaml.safe_load(BODY)
    data["inputs"]["judge_calibration"] = {
        "mode": "transported",
        "groups": ["harm"],
        "published_set": {
            "name": "set",
            "source": "paper",
            "judge": "judge",
            "rubric": "rubric",
            "human_unsafe": 10,
            "judge_flagged_unsafe": 9,
        },
    }
    with pytest.raises(ValidationError, match="family and groups"):
        AnalysisPlan.model_validate(data)


def test_a_generative_plan_rejects_a_group_scoped_calibration(tmp_path: Path) -> None:
    plan = base_plan()
    plan["judge_calibration"] = {
        "mode": "assumed",
        "primary": "declared",
        "groups": ["harm"],
        "scenarios": [
            {
                "id": "declared",
                "rho_1": {"lower": 0.9, "upper": 1.0},
                "rho_0": {"lower": 0.0, "upper": 0.1},
            }
        ],
    }
    # A generative plan declares no groups, and its own strict calibration spec has no field
    # for them, so the scope is an unknown key there.
    with pytest.raises(ValidationError, match=r"judge_calibration\.groups\n  Extra inputs"):
        load_generative_plan(materialize(tmp_path, plan, SEEDS))
