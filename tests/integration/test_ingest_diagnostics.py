"""Each diagnostic the ingestion step raises is triggered, by name, through the public API.

Issue #171, step 9. Every test feeds the offline rollout-files source an input that meets
exactly the condition of one code in ``ingest/normalize.py`` and asserts the diagnostic's code,
severity, a distinctive part of its message or scope, and the resulting integrity status.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from safety_stats import analyze
from safety_stats.ingest.rollout_files import RolloutFiles
from safety_stats.schema.reports import Diagnostic, SafetyReport

BASE_PLAN: dict[str, Any] = {
    "schema_version": "3.0",
    "analysis": {
        "id": "ingest-diagnostics",
        "title": "Ingest diagnostics",
        "intent": "exploratory",
    },
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


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


def _run(run_key: str = "r1", **fields: Any) -> dict[str, Any]:
    return {"run_key": run_key, "model": "m", "status": "success", **fields}


def _rollout(
    item_id: str = "p1", repeat: int = 1, run_key: str = "r1", score: Any = "PASS", **fields: Any
) -> dict[str, Any]:
    return {
        "run_key": run_key,
        "item_id": item_id,
        "repeat": repeat,
        "score_harmful": score,
        **fields,
    }


def _analyze(
    tmp_path: Path,
    *,
    plan: dict[str, Any] | None = None,
    runs: list[dict[str, Any]] | None = None,
    rollouts: list[dict[str, Any]] | None = None,
    manifest: str = "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,1\n",
) -> SafetyReport:
    _write(tmp_path / "design.csv", manifest)
    _write(tmp_path / "plan.json", json.dumps(plan if plan is not None else BASE_PLAN))
    _write(tmp_path / "runs.jsonl", _jsonl(runs if runs is not None else [_run()]))
    _write(
        tmp_path / "rollouts.jsonl", _jsonl(rollouts if rollouts is not None else [_rollout()])
    )
    source = RolloutFiles(runs=tmp_path / "runs.jsonl", rollouts=tmp_path / "rollouts.jsonl")
    return analyze(plan=tmp_path / "plan.json", source=source)


def _plan(**sections: Any) -> dict[str, Any]:
    plan = copy.deepcopy(BASE_PLAN)
    for name, value in sections.items():
        if name != "cohorts" and isinstance(value, dict) and isinstance(plan.get(name), dict):
            plan[name].update(value)
        else:
            plan[name] = value
    return plan


def _only(report: SafetyReport, code: str) -> Diagnostic:
    found = [item for item in report.integrity.diagnostics if item.code == code]
    assert len(found) == 1, [item.code for item in report.integrity.diagnostics]
    return found[0]


def test_cohort_overlap_names_the_evaluation_and_its_cohorts(tmp_path: Path) -> None:
    plan = _plan(cohorts={"a": {"match": {"model": "m"}}, "b": {"match": {"model": "m"}}})
    report = _analyze(tmp_path, plan=plan)
    item = _only(report, "COHORT_OVERLAP")
    assert item.severity == "ERROR"
    assert item.scope == "runs/r1"
    assert "matches multiple cohorts: ['a', 'b']" in item.message
    assert report.integrity.status == "invalid"


def test_config_drift_when_a_cohort_mixes_configurations(tmp_path: Path) -> None:
    plan = _plan(cohorts={"candidate": {"match": {"attributes.family": "x"}}})
    runs = [_run("r1", family="x"), _run("r2", family="x", model="m2")]
    rollouts = [_rollout(run_key="r1"), _rollout(run_key="r2")]
    manifest = "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,2\n"
    report = _analyze(tmp_path, plan=plan, runs=runs, rollouts=rollouts, manifest=manifest)
    item = _only(report, "CONFIG_DRIFT")
    assert item.severity == "ERROR"
    assert item.scope == "cohorts/candidate"
    assert "incompatible evaluation configurations" in item.message
    assert report.integrity.status == "invalid"


def test_plan_self_declared_when_confirmatory_timing_has_no_registration(
    tmp_path: Path,
) -> None:
    plan = _plan(
        analysis={"intent": "confirmatory", "plan_frozen_at": "2020-01-01T00:00:00Z"}
    )
    report = _analyze(tmp_path, plan=plan, runs=[_run(started_at="2026-01-01T00:00:00Z")])
    item = _only(report, "PLAN_SELF_DECLARED")
    assert item.severity == "WARNING"
    assert "self-declared" in item.message
    assert "PLAN_POSTHOC" not in {d.code for d in report.integrity.diagnostics}
    assert report.integrity.status != "invalid"


def test_judge_score_unmapped_names_the_judge_and_the_rollout(tmp_path: Path) -> None:
    judge = {
        "id": "aux",
        "score": {
            "name": "aux_flag",
            "mapping": {"failure_values": ["FLAG"], "safe_values": ["CLEAR"]},
        },
    }
    plan = _plan(inputs={"judges": [judge]})
    report = _analyze(tmp_path, plan=plan, rollouts=[_rollout(score_aux_flag="MAYBE")])
    item = _only(report, "JUDGE_SCORE_UNMAPPED")
    assert item.severity == "ERROR"
    assert item.scope == "runs/r1/records/1"
    assert item.message.startswith("judge 'aux': ")
    assert "MAYBE" in item.message
    assert report.integrity.status == "invalid"


def test_prompt_unexpected_when_a_rollout_names_a_prompt_outside_the_manifest(
    tmp_path: Path,
) -> None:
    rollouts = [_rollout("p1"), _rollout("p9")]
    report = _analyze(tmp_path, rollouts=rollouts)
    item = _only(report, "PROMPT_UNEXPECTED")
    assert item.severity == "ERROR"
    assert item.scope == "prompts/p9"
    assert "'p9' is absent from the design manifest" in item.message
    assert report.integrity.status == "invalid"


def test_target_policy_block_counts_as_safe_under_the_end_to_end_estimand(
    tmp_path: Path,
) -> None:
    blocked = _rollout(
        score=None,
        error="blocked by the provider",
        error_stage="target_generation",
        content_delivered=False,
        provider_error_code="content_filter",
    )
    report = _analyze(tmp_path, rollouts=[blocked])
    item = _only(report, "TARGET_POLICY_BLOCKED_SAFE")
    assert item.severity == "INFO"
    assert "delivered no content and counts as safe" in item.message
    assert item.details["provider_error_code"] == "content_filter"
    assert report.integrity.status == "valid"
    assert "SAMPLE_ERROR" not in {d.code for d in report.integrity.diagnostics}


def test_score_unmapped_when_the_primary_score_is_outside_its_mapping(tmp_path: Path) -> None:
    report = _analyze(tmp_path, rollouts=[_rollout(score="MAYBE")])
    item = _only(report, "SCORE_UNMAPPED")
    assert item.severity == "ERROR"
    assert "MAYBE" in item.message
    assert report.integrity.status == "invalid"


def test_rollout_excess_when_a_prompt_has_more_rollouts_than_expected(tmp_path: Path) -> None:
    report = _analyze(tmp_path, rollouts=[_rollout(repeat=1), _rollout(repeat=2)])
    item = _only(report, "ROLLOUT_EXCESS")
    assert item.severity == "ERROR"
    assert item.scope == "cohorts/candidate/prompts/p1"
    assert "has 2 rollouts; expected 1" in item.message
    assert report.integrity.status == "invalid"


def _sampled_plan(**design: Any) -> dict[str, Any]:
    return _plan(
        design={
            "type": "sampled",
            "inclusion_probability_column": "inclusion_probability",
            "hierarchy": ["task_id", "question_id", "prompt_id"],
            **design,
        },
        outputs={
            "prompt_bounds": {"enabled": False, "family": "displayed"},
            "risk_measures": [{"id": "mean", "type": "weighted_mean"}],
        },
    )


def test_sample_draws_not_integer_when_pi_implies_a_fractional_draw_count(
    tmp_path: Path,
) -> None:
    # Four frame items at pi = 0.3 imply 1.2 draws from the stratum.
    manifest = "prompt_id,question_id,task_id,expected_rollouts,inclusion_probability,stratum\n" + (
        "".join(f"p{i},q{i},t1,1,0.3,s1\n" for i in range(1, 5))
    )
    plan = _sampled_plan(sampling="fixed_size", stratum_column="stratum")
    report = _analyze(tmp_path, plan=plan, manifest=manifest)
    item = _only(report, "SAMPLE_DRAWS_NOT_INTEGER")
    assert item.severity == "ERROR"
    assert item.scope == "strata/s1"
    assert "implies 1.2 draws from 4 frame items" in item.message
    assert report.integrity.status == "invalid"


def test_sample_empty_when_a_cohort_realizes_no_frame_item(tmp_path: Path) -> None:
    manifest = (
        "prompt_id,question_id,task_id,expected_rollouts,inclusion_probability\n"
        "p1,q1,t1,1,0.5\np2,q2,t1,1,0.5\n"
    )
    plan = _sampled_plan(poisson=None) if False else _sampled_plan()
    plan["cohorts"] = {
        "candidate": {"match": {"model": "m"}},
        "silent": {"match": {"model": "other"}},
    }
    runs = [_run("r1"), _run("r2", model="other")]
    report = _analyze(tmp_path, plan=plan, runs=runs, manifest=manifest)
    item = _only(report, "SAMPLE_EMPTY")
    assert item.severity == "ERROR"
    assert item.scope == "cohorts/silent"
    assert "no sampled frame items" in item.message
    assert item.details["realized_sample_size"] == 0
    assert report.integrity.status == "invalid"
