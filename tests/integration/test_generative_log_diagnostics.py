"""Diagnostics the generative pipeline raises while reading logs and calibrating the judge.

Each test feeds the pipeline offline input that meets exactly the condition of one code:
GENERATIVE_COHORT_AMBIGUOUS, GENERATIVE_LOG_INCOMPLETE, GENERATIVE_SCORE_UNMAPPED and the
generative site of JUDGE_VALIDATION_STRATUM_EMPTY. Inspect's reader is replaced by frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from inspect_safety_stats.generative.ledger import read_rewrites
from inspect_safety_stats.generative.logs import read_evaluated_samples
from inspect_safety_stats.generative.runner import analyze_run, materialize_run
from inspect_safety_stats.generative.schema import GenerativePlan, load_generative_plan
from tests.generative_fixtures import SEEDS_WITH_TEXT, base_plan

MODEL_A = "provider/model-a"
MODEL_B = "provider/model-b"


def write_plan(tmp_path: Path, cohorts: dict[str, Any], **extra: Any) -> Path:
    plan = base_plan()
    plan["sampling"] = {"method": "direct", "total_rewrites": 200}
    plan["target"]["cohorts"] = cohorts
    plan.update(extra)
    (tmp_path / "seeds.csv").write_text(SEEDS_WITH_TEXT, encoding="utf-8", newline="\n")
    (tmp_path / "template.txt").write_text("Rewrite: {seed}\n", encoding="utf-8", newline="\n")
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def stub_frames(
    monkeypatch: pytest.MonkeyPatch, evals: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> None:
    frame = pd.DataFrame(evals)
    rows = pd.DataFrame(samples)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False, **_: rows,
    )


def sample_rows(eval_id: str, scores: list[Any]) -> list[dict[str, Any]]:
    return [
        {"eval_id": eval_id, "id": f"p{index}", "epoch": 1, "score_harmful": score}
        for index, score in enumerate(scores)
    ]


def load(tmp_path: Path, cohorts: dict[str, Any]) -> GenerativePlan:
    return load_generative_plan(write_plan(tmp_path, cohorts))


def test_two_cohorts_naming_one_model_are_ambiguous(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = load(tmp_path, {"first": {"model": MODEL_A}, "second": {"model": MODEL_A}})
    stub_frames(
        monkeypatch,
        [{"eval_id": "eval-a", "model": MODEL_A, "status": "success"}],
        sample_rows("eval-a", ["PASS", "FAIL"]),
    )

    samples, diagnostics = read_evaluated_samples(plan, "logs")

    assert samples == []
    assert [item.code for item in diagnostics] == ["GENERATIVE_COHORT_AMBIGUOUS"]
    item = diagnostics[0]
    assert item.severity == "ERROR"
    assert item.scope == "evals/eval-a"
    assert "declared by cohorts ['first', 'second']" in item.message


def test_a_run_that_did_not_succeed_is_flagged_but_still_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = load(tmp_path, {"baseline": {"model": MODEL_A}})
    stub_frames(
        monkeypatch,
        [{"eval_id": "eval-a", "model": MODEL_A, "status": "cancelled"}],
        sample_rows("eval-a", ["PASS", "FAIL", "PASS"]),
    )

    samples, diagnostics = read_evaluated_samples(plan, "logs")

    assert [item.code for item in diagnostics] == ["GENERATIVE_LOG_INCOMPLETE"]
    item = diagnostics[0]
    assert item.severity == "ERROR"
    assert item.scope == "evals/eval-a"
    assert "status 'cancelled'" in item.message
    # The partial run's samples are still returned, so the analysis can report on them.
    assert [sample.failure for sample in samples] == [False, True, False]
    assert {sample.cohort_id for sample in samples} == {"baseline"}


def test_a_successful_run_raises_no_incomplete_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = load(tmp_path, {"baseline": {"model": MODEL_A}})
    stub_frames(
        monkeypatch,
        [{"eval_id": "eval-a", "model": MODEL_A, "status": "success"}],
        sample_rows("eval-a", ["PASS", "FAIL"]),
    )

    samples, diagnostics = read_evaluated_samples(plan, "logs")

    assert diagnostics == []
    assert len(samples) == 2


def test_scores_outside_the_mapping_are_counted_and_left_unclassified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = load(tmp_path, {"baseline": {"model": MODEL_A}})
    stub_frames(
        monkeypatch,
        [{"eval_id": "eval-a", "model": MODEL_A, "status": "success"}],
        sample_rows("eval-a", ["PASS", "FAIL", "MAYBE", "REFUSED"]),
    )

    samples, diagnostics = read_evaluated_samples(plan, "logs")

    assert [item.code for item in diagnostics] == ["GENERATIVE_SCORE_UNMAPPED"]
    item = diagnostics[0]
    assert item.severity == "WARNING"
    assert item.details == {"count": 2}
    assert "2 samples carry a score outside the declared mapping" in item.message
    # Kept, with no verdict, rather than dropped.
    assert [sample.failure for sample in samples] == [False, True, None, None]


def test_an_empty_judge_stratum_gives_no_calibration_and_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The judge flagged nothing, so the judge-unsafe stratum has no row to label."""

    plan = write_plan(
        tmp_path,
        {"baseline": {"model": MODEL_A}, "candidate": {"model": MODEL_B}},
        judge_calibration={"mode": "validated", "validation_manifest": "labels.csv"},
    )
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    prompt_ids = [
        str(record.prompt_id)
        for record in read_rewrites(artifacts / "rewrites.csv")
        if record.is_evaluable()
    ]
    evals = [
        {"eval_id": "eval-a", "model": MODEL_A, "status": "success"},
        {"eval_id": "eval-b", "model": MODEL_B, "status": "success"},
    ]
    samples = [
        {"eval_id": eval_id, "id": prompt_id, "epoch": 1, "score_harmful": "PASS"}
        for eval_id in ("eval-a", "eval-b")
        for prompt_id in prompt_ids
    ]
    stub_frames(monkeypatch, evals, samples)
    labels = ["observation_id,human_label", *(f"{pid},safe" for pid in prompt_ids[:50])]
    (tmp_path / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")

    item = next(i for i in report.diagnostics if i.code == "JUDGE_VALIDATION_STRATUM_EMPTY")
    assert item.severity == "WARNING"
    assert item.scope == "cohorts/candidate"
    assert "the judge_unsafe stratum is empty, so rho cannot be estimated" in item.message
    assert not report.corrected
    assert report.evaluation["judge_calibration"] == {"mode": "validated", "available": False}
