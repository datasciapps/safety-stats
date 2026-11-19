"""Errors about a plan that does not fit the data say what the data holds.

Such errors used to name only what the plan asked for, so a user had to open the logs to see
what the plan should say.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from inspect_safety_stats import analyze
from inspect_safety_stats.ingest.rollout_files import RolloutFiles

CATEGORICAL = {"failure_values": ["FAIL"], "safe_values": ["PASS"]}


def _inputs(
    tmp_path: Path,
    scores: dict[str, list[Any]],
    *,
    score_name: str = "harmful",
    mapping: dict[str, Any] | None = None,
    cohort_model: str = "m",
    score_column: str | None = "score_harmful",
    judges: list[dict[str, Any]] | None = None,
) -> tuple[Path, RolloutFiles]:
    """One run of model ``m`` whose rollouts score each prompt with the values given."""
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\n"
        + "".join(
            f"{prompt},q-{prompt},t1,{len(values)}\n" for prompt, values in scores.items()
        ),
        encoding="utf-8",
        newline="\n",
    )
    inputs: dict[str, Any] = {
        "design_manifest": "design.csv",
        "score": {"name": score_name, "mapping": mapping or CATEGORICAL},
    }
    if judges is not None:
        inputs["judges"] = judges
    plan = {
        "schema_version": "3.0",
        "analysis": {"id": "source-values", "title": "Source values", "intent": "exploratory"},
        "inputs": inputs,
        "cohorts": {"candidate": {"match": {"model": cohort_model}}},
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
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8", newline="\n")
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
                {"run_key": "r1", "item_id": prompt, "repeat": repeat}
                | ({score_column: value} if score_column else {})
            )
            + "\n"
            for prompt, values in scores.items()
            for repeat, value in enumerate(values, start=1)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return plan_path, RolloutFiles(runs=runs, rollouts=rollouts)


def test_an_absent_score_names_the_scores_the_logs_have(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path, {"p1": ["PASS"]}, score_name="harm")

    report = analyze(plan=plan, source=source)

    (missing,) = [
        item
        for item in report.integrity.diagnostics
        if item.code == "SCORE_MISSING" and item.scope is None
    ]
    assert missing.message == (
        "score column 'score_harm' is absent from selected logs; inputs.score.name must name a "
        "score they have: 'harmful'"
    )


def test_an_absent_score_says_when_the_logs_have_no_scores(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path, {"p1": ["PASS"]}, score_column=None)

    report = analyze(plan=plan, source=source)

    (missing,) = [
        item
        for item in report.integrity.diagnostics
        if item.code == "SCORE_MISSING" and item.scope is None
    ]
    assert missing.message == (
        "score column 'score_harmful' is absent from selected logs, which have no scores"
    )


def test_an_empty_cohort_names_the_values_the_evaluations_have(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path, {"p1": ["PASS"]}, cohort_model="openai/gpt-4o")

    report = analyze(plan=plan, source=source)

    (empty,) = [
        item
        for item in report.integrity.diagnostics
        if item.code == "COHORT_EMPTY" and item.scope == "cohorts/candidate"
    ]
    assert empty.message == (
        "cohort 'candidate' contains no evaluations: it requires model = 'openai/gpt-4o', and "
        "the selected evaluations have model 'm' in 1 evaluation"
    )


def test_a_cohort_emptied_by_an_overlap_says_so(tmp_path: Path) -> None:
    """Its evaluations have the values it requires; they were put in no cohort instead."""
    plan, source = _inputs(tmp_path, {"p1": ["PASS"]})
    raw = yaml.safe_load(plan.read_text(encoding="utf-8"))
    raw["cohorts"]["also"] = {"match": {"model": "m"}}
    plan.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8", newline="\n")

    report = analyze(plan=plan, source=source)

    empty = {
        item.scope: item.message
        for item in report.integrity.diagnostics
        if item.code == "COHORT_EMPTY" and item.scope
    }
    assert empty["cohorts/candidate"] == (
        "cohort 'candidate' contains no evaluations: the 1 evaluation(s) it matches also match "
        "another cohort (COHORT_OVERLAP)"
    )
    assert "COHORT_OVERLAP" in {item.code for item in report.integrity.diagnostics}


def test_an_absent_judge_score_names_the_scores_the_logs_have(tmp_path: Path) -> None:
    judges = [{"id": "absent", "score": {"name": "other", "mapping": CATEGORICAL}}]
    plan, source = _inputs(tmp_path, {"p1": ["PASS"]}, judges=judges)

    report = analyze(plan=plan, source=source)

    (missing,) = [
        item for item in report.integrity.diagnostics if item.code == "JUDGE_SCORE_MISSING"
    ]
    assert missing.message == (
        "judge 'absent' score column 'score_other' is absent from selected logs; "
        "inputs.judges.absent.score.name must name a score they have: 'harmful'"
    )
