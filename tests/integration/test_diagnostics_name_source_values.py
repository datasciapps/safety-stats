"""Errors about a plan that does not fit the data say what the data holds.

Such errors used to name only what the plan asked for, so a user had to open the logs to see
what the plan should say.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from safety_stats import analyze, validate
from safety_stats.ingest.rollout_files import RolloutFiles

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


def test_unmapped_values_are_listed_once_ahead_of_each_rollouts_error(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path, {"p1": ["C", "C"], "p2": ["PASS", "PASS"]})

    report = analyze(plan=plan, source=source)

    diagnostics = report.integrity.diagnostics
    codes = [item.code for item in diagnostics]
    assert codes.count("SCORE_MAPPING_MISMATCH") == 1
    summary = diagnostics[codes.index("SCORE_MAPPING_MISMATCH")]
    assert summary.severity == "ERROR"
    assert summary.scope == "plan/inputs/score/mapping"
    assert summary.message == (
        "2 rollout(s) have a score value that inputs.score.mapping lists in neither "
        "failure_values ['FAIL'] nor safe_values ['PASS']: 'C' (str) in 2 rollouts. Add each "
        "value the scorer gives to one of the two lists; matching is type-sensitive."
    )
    assert summary.details == {"rollouts": 2, "distinct_values": 1}
    # Each rollout keeps its own error, which the summary comes ahead of.
    rollout_errors = [item for item in diagnostics if item.code == "SCORE_UNMAPPED"]
    assert [item.message for item in rollout_errors] == ["unmapped score value 'C' (str)"] * 2
    assert codes.index("SCORE_MAPPING_MISMATCH") < codes.index("SCORE_UNMAPPED")


def test_the_summary_withholds_no_bounds_that_the_rollouts_errors_do_not(
    tmp_path: Path,
) -> None:
    plan, source = _inputs(tmp_path, {"p1": ["C", "C"], "p2": ["PASS", "PASS"]})

    report = analyze(plan=plan, source=source)

    estimates = {item.prompt_id: item for item in report.prompt_estimates}
    assert estimates["p1"].lower is None and estimates["p1"].upper is None
    assert estimates["p2"].lower is not None and estimates["p2"].upper is not None


def test_values_are_listed_most_frequent_first_by_type_and_at_most_ten(tmp_path: Path) -> None:
    values: list[Any] = [1, 1, "1", *(f"v{index:02d}" for index in range(10))]
    plan, source = _inputs(tmp_path, {"p1": values})

    report = analyze(plan=plan, source=source)

    (summary,) = [
        item for item in report.integrity.diagnostics if item.code == "SCORE_MAPPING_MISMATCH"
    ]
    assert summary.message.startswith(
        "13 rollout(s) have a score value that inputs.score.mapping lists in neither "
        "failure_values ['FAIL'] nor safe_values ['PASS']: 1 (int) in 2 rollouts, "
        "'1' (str) in 1 rollout, 'v00' (str) in 1 rollout, "
    )
    assert "'v07' (str) in 1 rollout, and 2 more. Add each value" in summary.message
    assert "'v08'" not in summary.message
    assert summary.details == {"rollouts": 13, "distinct_values": 12}


def test_a_numeric_mapping_names_the_nonnumeric_values(tmp_path: Path) -> None:
    mapping = {"failure_if": {"operator": "greater_than_or_equal", "value": 0.5}}
    plan, source = _inputs(tmp_path, {"p1": ["high", "high", 0.9]}, mapping=mapping)

    report = analyze(plan=plan, source=source)

    (summary,) = [
        item for item in report.integrity.diagnostics if item.code == "SCORE_MAPPING_MISMATCH"
    ]
    assert summary.message == (
        "2 rollout(s) have a nonnumeric score, which inputs.score.mapping.failure_if cannot "
        "compare: 'high' (str) in 2 rollouts. Map categorical scores with failure_values and "
        "safe_values."
    )


def test_a_mapping_that_lists_a_value_twice_gets_no_list_of_values(tmp_path: Path) -> None:
    """Every value is unmapped then because the plan is wrong, whatever the logs hold."""
    mapping = {"failure_values": ["X"], "safe_values": ["X"]}
    plan, source = _inputs(tmp_path, {"p1": ["X", "X"]}, mapping=mapping)

    report = analyze(plan=plan, source=source)

    codes = [item.code for item in report.integrity.diagnostics]
    assert "SCORE_MAPPING_MISMATCH" not in codes
    assert codes.count("SCORE_UNMAPPED") == 2


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


def test_a_judges_unmapped_values_are_listed_once_ahead_of_its_rollout_errors(
    tmp_path: Path,
) -> None:
    judges = [{"id": "strict", "score": {"name": "harmful", "mapping": CATEGORICAL}}]
    mapping = {"failure_values": ["C"], "safe_values": ["PASS"]}
    plan, source = _inputs(tmp_path, {"p1": ["C", "PASS"]}, mapping=mapping, judges=judges)

    report = analyze(plan=plan, source=source)

    codes = [item.code for item in report.integrity.diagnostics]
    assert codes.count("JUDGE_SCORE_MAPPING_MISMATCH") == 1
    mismatch = report.integrity.diagnostics[codes.index("JUDGE_SCORE_MAPPING_MISMATCH")]
    assert mismatch.scope == "plan/inputs/judges/strict/score/mapping"
    assert mismatch.message == (
        "1 rollout(s) have a score value that inputs.judges.strict.score.mapping lists in "
        "neither failure_values ['FAIL'] nor safe_values ['PASS']: 'C' (str) in 1 rollout. "
        "Add each value the scorer gives to one of the two lists; matching is type-sensitive."
    )
    assert codes.index("JUDGE_SCORE_MAPPING_MISMATCH") < codes.index("JUDGE_SCORE_UNMAPPED")


def test_validate_reports_the_same_messages_as_analyze(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path, {"p1": ["C", "I"]}, cohort_model="m")

    summary = validate(plan=plan, source=source)
    report = analyze(plan=plan, source=source)

    assert summary.diagnostics == report.integrity.diagnostics
    assert "SCORE_MAPPING_MISMATCH" in {item.code for item in summary.diagnostics}
