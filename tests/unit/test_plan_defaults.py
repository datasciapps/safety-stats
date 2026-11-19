"""analyze records every default it applied: the plan fields the author left out."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from inspect_safety_stats import analyze
from inspect_safety_stats.config.loader import canonical_plan_value, plan_defaults
from inspect_safety_stats.config.models import AnalysisPlan

ROOT = Path(__file__).resolve().parents[2]
SIMPLEQA = ROOT / "examples" / "real_simpleqa"

PLAN: dict[str, Any] = {
    "schema_version": "3.0",
    "analysis": {"id": "defaults", "title": "Defaults", "intent": "exploratory"},
    "inputs": {
        "design_manifest": "design.csv",
        "score": {
            "name": "harmful",
            "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
        },
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


def _defaults(plan: dict[str, Any]) -> dict[str, Any]:
    entries = plan_defaults(AnalysisPlan.model_validate(plan))
    return {entry["path"]: entry["value"] for entry in entries}


def test_every_field_left_out_is_listed_with_the_value_it_got() -> None:
    defaults = _defaults(PLAN)
    plan = AnalysisPlan.model_validate(PLAN)
    assert defaults["inputs.score.token_limit"] == plan.inputs.score.token_limit
    # The plan leaves design out whole, and schema 3.0 gives it the neutral rollout key.
    assert defaults["design.rollout_key"] == ["run_key", "item_id", "repeat"]
    # Paths name list items by position and mapping entries by key.
    assert defaults["outputs.risk_measures[0].diagnostic_only"] is False
    assert defaults["cohorts.candidate.require_uniform"] == []
    # A field the plan wrote is not a default, and no path is both written and defaulted.
    assert "outputs.prompt_bounds.enabled" not in defaults
    assert not {"analysis", "inputs", "claims"} & set(defaults)


def test_a_default_written_out_is_no_longer_a_default() -> None:
    written = copy.deepcopy(PLAN)
    written["inputs"]["score"]["token_limit"] = _defaults(PLAN)["inputs.score.token_limit"]
    assert "inputs.score.token_limit" in _defaults(PLAN)
    assert "inputs.score.token_limit" not in _defaults(written)


def test_a_section_left_out_whole_has_an_entry_for_each_field() -> None:
    defaults = _defaults(PLAN)
    assert "report" not in defaults
    # report.claims_export is experimental: the section's dump would leave it out.
    assert defaults["report.claims_export"] is False
    assert defaults["report.include_decisions"] is False


def test_a_field_absent_while_none_is_not_a_default() -> None:
    # Schema 3.0 has no sample_join: it is absent, not defaulted.
    assert "inputs.sample_join" not in _defaults(PLAN)
    legacy = copy.deepcopy(PLAN)
    legacy["schema_version"] = "2.0"
    legacy["design"] = {"rollout_key": ["eval_id", "id", "epoch"]}
    legacy["cohorts"] = {"candidate": {"match": {"model": "m"}}}
    assert _defaults(legacy)["inputs.sample_join.manifest_field"] == "prompt_id"


def test_defaults_and_the_hashed_plan_cover_every_top_level_field() -> None:
    plan = AnalysisPlan.model_validate(PLAN)
    written = set(canonical_plan_value(plan))
    defaulted = {entry["path"].split(".")[0].split("[")[0] for entry in plan_defaults(plan)}
    assert {"design", "policy", "report"} <= defaulted - written
    assert written | defaulted == set(AnalysisPlan.model_fields)


def _set(target: Any, path: str, value: Any) -> None:
    steps: list[str | int] = []
    for part in path.split("."):
        name, *indices = part.split("[")
        steps.append(name)
        steps.extend(int(index.rstrip("]")) for index in indices)
    for step in steps[:-1]:
        target = target[step] if isinstance(step, int) else target.setdefault(step, {})
    target[steps[-1]] = value


def test_what_the_plan_wrote_and_its_defaults_make_the_plan_that_ran() -> None:
    plan = AnalysisPlan.model_validate(PLAN)
    rebuilt = canonical_plan_value(plan)
    for entry in plan_defaults(plan):
        _set(rebuilt, entry["path"], entry["value"])
    written_out = AnalysisPlan.model_validate(rebuilt)
    assert written_out == plan
    # Written out in full, the plan has no defaults left.
    assert plan_defaults(written_out) == []


def test_experimental_defaults_are_listed_although_the_resolved_plan_leaves_them_out() -> None:
    # Their models leave them out of dumps while off, to keep older outputs' bytes; the
    # analysis still applied them.
    defaults = _defaults(PLAN)
    assert defaults["inputs.score.token_limit"] == "score"
    assert "token_limit" not in AnalysisPlan.model_validate(PLAN).model_dump()["inputs"]["score"]


def test_analyze_records_the_defaults_in_report_json_and_report_md(tmp_path: Path) -> None:
    report = analyze(
        plan=SIMPLEQA / "safety-plan.yaml", logs=str(SIMPLEQA / "simpleqa-8-real.eval")
    )
    recorded = {entry.path: entry.value for entry in report.plan_defaults}
    assert recorded["inputs.score.token_limit"] == "score"
    assert recorded["outputs.judge_claims"] is False

    report.write(tmp_path / "report")
    data = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    expected = [entry.model_dump(mode="json") for entry in report.plan_defaults]
    assert data["plan_defaults"] == expected

    markdown = (tmp_path / "report" / "report-full.md").read_text(encoding="utf-8")
    section = markdown.split("## Defaults applied", 1)[1].split("\n## ", 1)[0]
    assert '- `inputs.score.token_limit`: `"score"`' in section
    # Defaults that leave a feature off are counted, not listed.
    assert "outputs.judge_claims" not in section
    off = sum(entry.value in (None, False, [], {}) for entry in report.plan_defaults)
    assert f"{off} further defaults leave" in section
