"""Reports list the registry's assumptions their results rest on, with the settings they took."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from inspect_safety_stats import analyze
from inspect_safety_stats.analysis.assumptions import APPLIES, assumptions_used, plan_values
from inspect_safety_stats.config.loader import load_plan
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.registry import ASSUMPTIONS

ROOT = Path(__file__).resolve().parents[2]
SIMPLEQA = ROOT / "examples" / "real_simpleqa"
XSTEST = ROOT / "examples" / "real_xstest"


def _plan() -> AnalysisPlan:
    return load_plan(XSTEST / "safety-plan.yaml")


def _with(plan: AnalysisPlan, section: str, **changes: Any) -> AnalysisPlan:
    return plan.model_copy(update={section: getattr(plan, section).model_copy(update=changes)})


def test_every_registry_assumption_says_when_it_applies() -> None:
    assert set(APPLIES) == set(ASSUMPTIONS)


@pytest.mark.parametrize("entry", list(ASSUMPTIONS.values()), ids=lambda entry: entry.id)
def test_every_plan_field_of_the_registry_resolves_on_a_plan(entry: Any) -> None:
    if entry.plan_field is None:
        return
    for path, _ in plan_values(_plan(), entry.plan_field):
        assert path.split(".")[0] == entry.plan_field.split(".")[0]


def test_a_default_setting_has_the_value_the_registry_states() -> None:
    used = assumptions_used(_plan())
    checked = 0
    for assumption in used:
        stated = ASSUMPTIONS[assumption["id"]].default
        for setting in assumption["settings"]:
            if setting["default"] and stated is not None and setting["value"] is not None:
                assert setting["value"] == yaml.safe_load(stated), setting["path"]
                checked += 1
    assert checked >= 2


def test_written_and_default_settings_are_told_apart() -> None:
    settings = {
        setting["path"]: setting
        for assumption in assumptions_used(_plan())
        for setting in assumption["settings"]
    }
    assert settings["inputs.score.token_limit"] == {
        "path": "inputs.score.token_limit",
        "value": "score",
        "default": True,
    }
    assert settings["design.require_expected_rollouts"]["default"] is False


def test_assumptions_apply_only_where_the_plan_makes_them() -> None:
    plan = _plan()
    ids = {assumption["id"] for assumption in assumptions_used(plan)}
    # The example is exploratory and compares nothing.
    assert plan.analysis.intent == "exploratory" and not plan.comparisons
    assert "plan_fixed_in_advance" not in ids
    assert "arm_independence" not in ids
    assert {"fixed_design", "scorer_validity", "rollout_independence"} <= ids

    confirmatory = _with(plan, "analysis", intent="confirmatory")
    assert APPLIES["plan_fixed_in_advance"](confirmatory)
    sampled = _with(plan, "design", type="sampled")
    assert not APPLIES["fixed_design"](sampled)


def test_a_mapping_pattern_names_every_key() -> None:
    plan = _plan()
    paths = [path for path, _ in plan_values(plan, "cohorts.<id>.require_uniform")]
    assert paths == [f"cohorts.{cohort}.require_uniform" for cohort in plan.cohorts]


def test_the_report_states_the_methods_it_used_and_the_assumptions_it_rests_on(
    tmp_path: Path,
) -> None:
    report = analyze(
        plan=SIMPLEQA / "safety-plan.yaml", logs=str(SIMPLEQA / "simpleqa-8-real.eval")
    )
    output = report.write(tmp_path / "report")
    data = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in data["assumptions_used"]] == [
        item.id for item in report.assumptions_used
    ]
    markdown = (output / "report.md").read_text(encoding="utf-8")

    measures = markdown.split("## Fixed-design risk measures", 1)[1].split("\n| Cohort", 1)[0]
    # The fixed list of methods is gone (#64): every method used is named, with its assumptions.
    assert "may use direct weighted Hoeffding" not in measures
    for risk in report.group_risks:
        for interval in risk.intervals:
            assert f"`{interval.method}`" in measures
            for assumption in interval.assumptions:
                assert assumption in measures

    limitations = markdown.split("## Limitations", 1)[1].split("\n## ", 1)[0]
    for assumption in report.assumptions_used:
        assert f"**{assumption.id}.** {assumption.statement}" in limitations
    assert "`inputs.score.token_limit`: `\"score\"` (default)" in limitations
    assert "Outputs marked diagnostic cannot independently produce a safety pass." in limitations
