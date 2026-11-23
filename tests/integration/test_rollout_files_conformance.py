"""The neutral rollout files and the Inspect logs they transcribe give the same analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from safety_stats import analyze, validate
from safety_stats.cli import app
from safety_stats.ingest.rollout_files import RolloutFiles
from tests.cli_output import plain

PLAN = """
schema_version: "3.0"
analysis:
  id: conformance
  title: Neutral and Inspect sources agree
  intent: exploratory
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
    missing: conservative_bounds
    errors: end_to_end
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  require_complete_manifest: true
  require_expected_rollouts: true
  independence_profiles:
    rollout:
      block_by: [rollout]
      comparison_coupling: independent
cohorts:
  baseline:
    match: {model: model/baseline}
    require_uniform: [attributes.task_version]
  candidate:
    match: {model: model/candidate}
    require_uniform: [attributes.task_version]
groups:
  - id: all
outputs:
  prompt_bounds:
    enabled: true
    simultaneous: true
    family: displayed
    alpha_allocation: equal_bonferroni
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    family: direct_means
    groups: [all]
    simultaneous: true
    coverage_mode: two_sided
    independence:
      primary: rollout
  risk_measures:
    - {id: mean, type: weighted_mean}
    - {id: worst, type: maximum}
comparisons:
  - id: candidate_vs_baseline
    group: all
    measure: mean
    baseline: baseline
    candidate: candidate
    estimand: risk_difference
    method: weighted_kl_chernoff
    family: comparisons
    independence:
      primary: rollout
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: comparisons, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - id: candidate_safe
    title: Candidate risk is acceptable
    cohort: candidate
    group: all
    measure: mean
    family: gate
    acceptable_upper: 0.4
    unacceptable_lower: 0.7
    mandatory: true
"""

EPOCHS = 12


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    evals = pd.DataFrame(
        [
            {
                "eval_id": f"eval-{name}",
                "run_id": "run-1",
                "log": f"/not/a/local/{name}.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "conformance",
                "task_version": "1",
                "model": f"model/{name}",
            }
            for name in ("baseline", "candidate")
        ]
    )
    rows = []
    for name, failing in (("baseline", 5), ("candidate", 2)):
        for prompt_id in ("p1", "p2", "p3"):
            for epoch in range(1, EPOCHS + 1):
                score: str | None = "FAIL" if epoch <= failing else "PASS"
                if prompt_id == "p2" and epoch == EPOCHS:
                    score = None
                rows.append(
                    {
                        "sample_id": f"{name}-{prompt_id}-{epoch}",
                        "eval_id": f"eval-{name}",
                        "log": f"/not/a/local/{name}.eval",
                        "id": prompt_id,
                        "epoch": epoch,
                        "score_harmful": score,
                        "error": "boom" if prompt_id == "p3" and epoch == EPOCHS else None,
                        "retries": 0,
                    }
                )
    return evals, pd.DataFrame(rows)


def _write_neutral(tmp_path: Path, evals: pd.DataFrame, samples: pd.DataFrame) -> RolloutFiles:
    runs = tmp_path / "runs.jsonl"
    with runs.open("w", encoding="utf-8", newline="\n") as stream:
        for _, row in evals.iterrows():
            record = {
                "run_key": row["eval_id"],
                "model": row["model"],
                "status": row["status"],
                "started_at": row["created"],
                "run_id": row["run_id"],
                "task_version": row["task_version"],
            }
            stream.write(json.dumps(record) + "\n")
    rollouts = tmp_path / "rollouts.jsonl"
    with rollouts.open("w", encoding="utf-8", newline="\n") as stream:
        # Written in reverse, so that the adapter must restore the collection order.
        for _, row in samples.iloc[::-1].iterrows():
            record = {
                "run_key": row["eval_id"],
                "item_id": row["id"],
                "repeat": int(row["epoch"]),
                "rollout_id": row["sample_id"],
                "scores": {"harmful": row["score_harmful"]},
                "error": row["error"],
                "retries": int(row["retries"]),
            }
            stream.write(json.dumps(record) + "\n")
    return RolloutFiles(runs=runs, rollouts=rollouts)


def _inputs(tmp_path: Path) -> tuple[Path, RolloutFiles]:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\n"
        + "".join(f"p{i},q{i},t1,{EPOCHS}\n" for i in (1, 2, 3)),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(PLAN, encoding="utf-8")
    return plan, _write_neutral(tmp_path, *_frames())


def _assert_same(left: Any, right: Any, where: str = "report") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        assert sorted(left) == sorted(right), where
        for key in left:
            _assert_same(left[key], right[key], f"{where}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), where
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _assert_same(a, b, f"{where}[{index}]")
    elif isinstance(left, float) and isinstance(right, float) and math.isfinite(left):
        # Rollout blocks are ordered by observation id, which differs between the sources,
        # so float sums may differ in their last bits.
        assert left == pytest.approx(right, rel=1e-12, abs=1e-12), where
    else:
        assert left == right, where


def test_neutral_files_and_inspect_logs_give_the_same_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, neutral = _inputs(tmp_path)
    evals, samples = _frames()
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )
    # The transcripts are read and say nothing about the errors' stage, as the rollout files
    # say nothing about it either.
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.read_eval_log",
        lambda path: SimpleNamespace(samples=[]),
    )
    # One schema 3.0 plan serves both sources.
    from_inspect = analyze(plan=plan, logs="ignored").model_dump(mode="json")
    from_files = analyze(plan=plan, source=neutral).model_dump(mode="json")

    # What legitimately differs: the files read and their format, and the run fields each
    # adapter counts as configuration.
    assert from_inspect["provenance"]["source_format"] == {
        "format": "inspect_logs",
        "inspect_version_constraint": ">=0.3.259,<0.4",
    }
    assert from_files["provenance"]["source_format"] == {"format": "rollout_files"}
    for report in (from_inspect, from_files):
        del report["provenance"]["sources"]
        del report["provenance"]["source_format"]
        for cohort in report["cohorts"]:
            del cohort["configuration_fingerprints"]
    assert {item["code"] for item in from_files["integrity"]["diagnostics"]} >= {
        "SCORE_MISSING_BOUNDED",
        "SAMPLE_ERROR_UNCLASSIFIED",
    }
    _assert_same(from_inspect, from_files)


def test_validate_reads_a_neutral_source(tmp_path: Path) -> None:
    neutral_plan, neutral = _inputs(tmp_path)
    summary = validate(plan=neutral_plan, source=neutral)
    assert summary.status == "degraded"


def test_logs_and_a_source_together_are_refused(tmp_path: Path) -> None:
    neutral_plan, neutral = _inputs(tmp_path)
    with pytest.raises(ValueError, match="either logs or source"):
        analyze(plan=neutral_plan, logs="x.eval", source=neutral)


def test_the_cli_analyses_rollout_files(tmp_path: Path) -> None:
    neutral_plan, neutral = _inputs(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--plan",
            str(neutral_plan),
            "--runs",
            str(neutral.runs),
            "--rollouts",
            str(neutral.rollouts),
            "--output",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, plain(result.output)
    assert "Integrity: degraded" in plain(result.output)
    assert (tmp_path / "report" / "report.json").exists()


@pytest.mark.parametrize(
    "arguments",
    [["--runs", "RUNS"], ["--rollouts", "ROLLOUTS"], ["--logs", "RUNS", "--runs", "RUNS"]],
)
def test_the_cli_refuses_an_incomplete_or_doubled_source(
    tmp_path: Path, arguments: list[str]
) -> None:
    neutral_plan, neutral = _inputs(tmp_path)
    paths = {"RUNS": str(neutral.runs), "ROLLOUTS": str(neutral.rollouts)}
    for command in ("analyze", "validate"):
        result = CliRunner().invoke(
            app,
            [command, "--plan", str(neutral_plan), *[paths.get(a, a) for a in arguments]],
        )
        assert result.exit_code == 2, plain(result.output)
