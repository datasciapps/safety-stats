"""Plan rules for an assumed judge-calibration scenario that gives cohorts boxes of their own.

``cohorts`` maps a declared cohort to its own (rho_1, rho_0) box, so a judge whose accuracy
depends on the model under test can be stress-tested without human labels. The field is left
out of the resolved plan while absent, so plans written before it keep their hash.
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
analysis: {id: boxes, title: Boxes, intent: exploratory}
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
cohorts:
  candidate: {match: {model: model/candidate}}
  baseline: {match: {model: model/baseline}}
groups:
  - {id: harm, filter: {task_id: harm}}
outputs: {prompt_bounds: {enabled: false}}
multiplicity: {families: []}
claims: []
"""

CANDIDATE_BOX = {"rho_1": {"lower": 0.7, "upper": 0.9}, "rho_0": {"lower": 0.0, "upper": 0.05}}


def _plan(cohorts: Any = None) -> AnalysisPlan:
    data = yaml.safe_load(BODY)
    if cohorts is not None:
        data["inputs"]["judge_calibration"]["scenarios"][0]["cohorts"] = cohorts
    return AnalysisPlan.model_validate(data)


def test_absent_cohort_boxes_are_left_out_of_the_resolved_plan() -> None:
    scenario = _plan().model_dump(mode="json")["inputs"]["judge_calibration"]["scenarios"][0]
    assert "cohorts" not in scenario
    # So the hash is that of a plan written before the field existed...
    plain = canonical_plan_hash(_plan())
    # ...and a cohort box, once set, is part of what the hash covers.
    assert canonical_plan_hash(_plan({"candidate": CANDIDATE_BOX})) != plain


def test_a_cohort_takes_its_own_box_and_every_other_cohort_the_default() -> None:
    plan = _plan({"candidate": CANDIDATE_BOX})
    scenario = plan.inputs.judge_calibration.scenarios[0]  # type: ignore[union-attr]
    rho_1, rho_0 = scenario.box("candidate")
    assert (rho_1.lower, rho_1.upper, rho_0.lower, rho_0.upper) == (0.7, 0.9, 0.0, 0.05)
    rho_1, rho_0 = scenario.box("baseline")
    assert (rho_1.lower, rho_1.upper, rho_0.lower, rho_0.upper) == (0.9, 1.0, 0.0, 0.1)
    recorded = plan.model_dump(mode="json")["inputs"]["judge_calibration"]["scenarios"][0]
    assert recorded["cohorts"] == {"candidate": CANDIDATE_BOX}


def test_cohort_boxes_are_never_empty() -> None:
    with pytest.raises(ValidationError, match="cohorts cannot be empty"):
        _plan({})


def test_each_cohort_box_must_be_strictly_informative() -> None:
    box = {"rho_1": {"lower": 0.5, "upper": 1.0}, "rho_0": {"lower": 0.0, "upper": 0.5}}
    with pytest.raises(ValidationError, match="not strictly informative for cohort 'candidate'"):
        _plan({"candidate": box})


def test_a_cohort_box_takes_only_rho_1_and_rho_0() -> None:
    with pytest.raises(ValidationError) as raised:
        _plan({"candidate": {**CANDIDATE_BOX, "id": "other"}})
    # The box refuses the key itself. A scenario without a `cohorts` field would also raise
    # "Extra inputs", for `cohorts` as a whole, so only the location shows which.
    assert [(error["type"], error["loc"]) for error in raised.value.errors()] == [
        (
            "extra_forbidden",
            ("inputs", "judge_calibration", "scenarios", 0, "cohorts", "candidate", "id"),
        )
    ]


def test_a_cohort_box_naming_an_undeclared_cohort_is_a_plan_reference_error() -> None:
    plan = _plan({"candidate": CANDIDATE_BOX, "typo": CANDIDATE_BOX})
    unknown = [
        (item.scope, item.message)
        for item in validate_plan_references(plan)
        if item.code == "PLAN_UNKNOWN_REFERENCE"
    ]
    assert unknown == [
        (
            "plan/inputs/judge_calibration",
            "inputs.judge_calibration scenario 'declared' references unknown cohorts: typo",
        )
    ]


def test_a_generative_plan_rejects_cohort_boxes(tmp_path: Path) -> None:
    plan = base_plan()
    plan["judge_calibration"] = {
        "mode": "assumed",
        "primary": "declared",
        "scenarios": [
            {
                "id": "declared",
                "rho_1": {"lower": 0.9, "upper": 1.0},
                "rho_0": {"lower": 0.0, "upper": 0.1},
                "cohorts": {"candidate": CANDIDATE_BOX},
            }
        ],
    }
    # The generative correction reads one box per scenario, so a per-cohort box it would
    # ignore is refused rather than dropped.
    with pytest.raises(ValidationError, match="per-cohort judge calibration boxes"):
        load_generative_plan(materialize(tmp_path, plan, SEEDS))
