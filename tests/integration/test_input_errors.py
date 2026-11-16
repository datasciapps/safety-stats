"""Bad input is an INGEST_FAILED error in both validate and analyze, and analyze still reports.

A missing manifest or an unreadable rollouts file used to make ``analyze`` raise and write no
report, while ``validate`` reported the same input as ``INGEST_FAILED``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inspect_safety_stats import analyze, validate
from inspect_safety_stats.cli import app
from inspect_safety_stats.ingest.rollout_files import RolloutFiles
from tests.cli_output import plain

PLAN = """
schema_version: "3.0"
analysis: {id: input-errors, title: Input errors, intent: confirmatory,
           plan_frozen_at: "2020-01-01T00:00:00Z"}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
cohorts:
  candidate: {match: {model: m}}
groups:
  - id: all
outputs:
  prompt_bounds: {enabled: true, family: displayed}
  risk_measures:
    - {id: mean, type: weighted_mean}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: claims, purpose: all_mandatory, method: bonferroni, alpha: 0.05}
claims:
  - {id: low, title: Mean failure rate is low, cohort: candidate, group: all, measure: mean,
     family: claims, acceptable_upper: 0.1, unacceptable_lower: 0.3}
"""


def _inputs(tmp_path: Path, *, manifest: bool = True) -> tuple[Path, RolloutFiles]:
    if manifest:
        (tmp_path / "design.csv").write_text(
            "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,2\n",
            encoding="utf-8",
            newline="\n",
        )
    plan = tmp_path / "plan.yaml"
    plan.write_text(PLAN, encoding="utf-8", newline="\n")
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        json.dumps({"run_key": "r1", "model": "m", "status": "success"}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        "".join(
            json.dumps(
                {"run_key": "r1", "item_id": "p1", "repeat": repeat, "score_harmful": "PASS"}
            )
            + "\n"
            for repeat in (1, 2)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return plan, RolloutFiles(runs=runs, rollouts=rollouts)


def test_analyze_reports_a_missing_manifest_instead_of_raising(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path, manifest=False)
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "invalid"
    assert [item.code for item in report.integrity.diagnostics] == ["INGEST_FAILED"]
    assert "design.csv" in report.integrity.diagnostics[0].message
    assert report.decision == "INCONCLUSIVE"
    assert [(claim.claim_id, claim.status) for claim in report.claims] == [("low", "INCONCLUSIVE")]
    assert report.claims[0].alpha == pytest.approx(0.05, abs=0)
    assert report.prompt_estimates == []
    assert report.group_risks == []
    assert report.design["prompt_count"] is None
    written = report.write(tmp_path / "report")
    stored = json.loads((written / "report.json").read_text(encoding="utf-8"))
    assert stored["integrity"]["diagnostics"][0]["code"] == "INGEST_FAILED"
    assert "INGEST_FAILED" in (written / "report.md").read_text(encoding="utf-8")


def test_validate_and_analyze_report_bad_input_the_same_way(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path)
    source.rollouts.write_text("{not json\n", encoding="utf-8", newline="\n")
    summary = validate(plan=plan, source=source)
    report = analyze(plan=plan, source=source)
    assert summary.status == report.integrity.status == "invalid"
    assert summary.diagnostics == report.integrity.diagnostics
    assert [item.code for item in summary.diagnostics] == ["INGEST_FAILED"]


def test_the_cli_writes_the_report_of_unreadable_input_and_exits_3(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path, manifest=False)
    output = tmp_path / "report"
    arguments = [
        "analyze",
        "--plan",
        str(plan),
        "--runs",
        str(source.runs),
        "--rollouts",
        str(source.rollouts),
        "--output",
        str(output),
    ]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 3, plain(result.output)
    assert "Integrity: invalid" in plain(result.output)
    assert "Analysis failed" not in plain(result.output)
    assert (output / "report.json").exists()
    allowed = CliRunner().invoke(
        app, [*arguments, "--overwrite", "--allow-invalid", "--decision-exit-codes"]
    )
    assert allowed.exit_code == 2, plain(allowed.output)


def test_logs_and_a_source_together_are_refused_by_validate_too(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path)
    with pytest.raises(ValueError, match="either logs or source"):
        validate(plan=plan, logs="x.eval", source=source)
