"""A safeguard that a source gives no data for is reported, not skipped."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from inspect_safety_stats import validate
from inspect_safety_stats.ingest.rollout_files import RolloutFiles

PLAN = """
schema_version: "2.0"
analysis:
  id: degradation
  title: Degradation diagnostics
  intent: INTENT
  plan_frozen_at: 2026-01-01T00:00:00Z
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
design:
  rollout_key: [run_key, item_id, repeat]
cohorts:
  candidate:
    match: {team: red}
    require_uniform: REQUIRED
groups:
  - id: all
outputs:
  prompt_bounds:
    enabled: true
    family: displayed
  risk_measures:
    - {id: mean, type: weighted_mean}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
claims: []
"""


def _jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _codes(
    tmp_path: Path,
    run: dict[str, object],
    *,
    intent: str = "confirmatory",
    required: str = "[]",
    schema: str = "2.0",
) -> dict[str, dict[str, object]]:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,2\n", encoding="utf-8"
    )
    plan = tmp_path / "plan.yaml"
    text = PLAN.replace("INTENT", intent).replace("REQUIRED", required)
    if schema == "3.0":
        text = text.replace('"2.0"', '"3.0"').replace("{team: red}", "{attributes.team: red}")
    plan.write_text(text, encoding="utf-8")
    runs = _jsonl(tmp_path / "runs.jsonl", [{"run_key": "r1", "team": "red", **run}])
    rollouts = _jsonl(
        tmp_path / "rollouts.jsonl",
        [
            {"run_key": "r1", "item_id": "p1", "repeat": repeat, "score_harmful": "PASS"}
            for repeat in (1, 2)
        ],
    )
    summary = validate(plan=plan, source=RolloutFiles(runs=runs, rollouts=rollouts))
    return {item.code: {"severity": item.severity, **item.details} for item in summary.diagnostics}


COMPLETE = {"status": "success", "started_at": "2026-02-01T00:00:00Z", "model": "m"}


def test_a_run_with_every_safeguard_observed_raises_none(tmp_path: Path) -> None:
    codes = _codes(tmp_path, COMPLETE)
    assert not {"RUN_STATUS_UNOBSERVED", "START_TIME_UNOBSERVED", "CONFIG_UNOBSERVED"} & set(
        codes
    )


def test_a_run_without_a_status_is_assumed_complete_and_reported(tmp_path: Path) -> None:
    codes = _codes(tmp_path, {**COMPLETE, "status": None})
    assert codes["RUN_STATUS_UNOBSERVED"] == {"severity": "WARNING", "integrity_degraded": True}
    assert "LOG_INCOMPLETE" not in codes


def test_a_confirmatory_plan_reports_runs_without_a_start_time(tmp_path: Path) -> None:
    codes = _codes(tmp_path, {**COMPLETE, "started_at": None})
    assert codes["START_TIME_UNOBSERVED"] == {
        "severity": "WARNING",
        "integrity_degraded": True,
        "evaluations": ["r1"],
    }
    assert "PLAN_POSTHOC" not in codes
    # An exploratory plan makes no timing claim to check.
    assert "START_TIME_UNOBSERVED" not in _codes(
        tmp_path, {**COMPLETE, "started_at": None}, intent="exploratory"
    )


def test_a_run_that_reports_no_configuration_is_reported(tmp_path: Path) -> None:
    codes = _codes(tmp_path, {**COMPLETE, "model": None})
    assert codes["CONFIG_UNOBSERVED"] == {
        "severity": "WARNING",
        "integrity_degraded": True,
        "evaluations": ["r1"],
    }
    # A field the cohort requires uniform is configuration too.
    assert "CONFIG_UNOBSERVED" not in _codes(
        tmp_path, {**COMPLETE, "model": None, "seed": 3}, required="[seed]"
    )


@pytest.mark.parametrize("field", ["model", "run_id"])
def test_a_schema_3_uniform_field_a_run_does_not_give_is_reported(
    tmp_path: Path, field: str
) -> None:
    """A run without the field read as None, which every such run shares: uniform, unchecked."""
    codes = _codes(tmp_path, {**COMPLETE, field: None}, required=f"[{field}]", schema="3.0")
    assert codes["PROVENANCE_UNOBSERVED"]["severity"] == "WARNING"
    given = _codes(tmp_path, {**COMPLETE, field: "x"}, required=f"[{field}]", schema="3.0")
    assert "PROVENANCE_UNOBSERVED" not in given


@pytest.mark.parametrize(
    ("column", "code"),
    [("status", "RUN_STATUS_UNOBSERVED"), ("created", "START_TIME_UNOBSERVED")],
)
def test_inspect_logs_without_a_status_or_start_time_are_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, column: str, code: str
) -> None:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,1\n", encoding="utf-8"
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        PLAN.replace("INTENT", "confirmatory")
        .replace("REQUIRED", "[]")
        .replace("[run_key, item_id, repeat]", "[eval_id, id, epoch]"),
        encoding="utf-8",
    )
    evals = pd.DataFrame(
        [
            {
                "eval_id": "e1",
                "log": "/not/a/local/log.eval",
                "created": "2026-02-01T00:00:00Z",
                "status": "success",
                "model": "m",
                "team": "red",
            }
        ]
    ).drop(columns=[column])
    samples = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "eval_id": "e1",
                "id": "p1",
                "epoch": 1,
                "score_harmful": "PASS",
                "error": None,
                "retries": 0,
            }
        ]
    )
    monkeypatch.setattr("inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: evals)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )
    summary = validate(plan=plan, logs="ignored")
    assert code in {item.code for item in summary.diagnostics}
    assert summary.status == "degraded"
