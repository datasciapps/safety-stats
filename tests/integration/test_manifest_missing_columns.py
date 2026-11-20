"""A missing manifest column's message names the plan settings that need it.

A plan names ``question_id`` and ``task_id`` through its hierarchy, its weights, its groups and
its independence profiles, and through defaults it may not spell out. A manifest that lacks one
is refused, and the message says which settings name it, so the plan or the manifest can be
fixed at the right place. Adapted from #173.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from inspect_safety_stats import analyze
from inspect_safety_stats.ingest.rollout_files import RolloutFiles

PLAN = """
schema_version: "3.0"
analysis: {id: flat-manifest, title: Flat manifest, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
design:
@DESIGN@
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

FLAT = "  hierarchy: [prompt_id]\n  weights: {strategy: nested_equal, hierarchy: [prompt_id]}"
DEFAULT = "  type: fixed"
FLAT_MANIFEST = "prompt_id,expected_rollouts\np1,2\np2,2\n"
FULL_MANIFEST = "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,2\np2,q2,t1,2\n"


def _inputs(
    tmp_path: Path, design: str, manifest: str, profiles: str = ""
) -> tuple[Path, RolloutFiles]:
    (tmp_path / "design.csv").write_text(manifest, encoding="utf-8", newline="\n")
    text = PLAN.replace("@DESIGN@", design + profiles)
    plan = tmp_path / "plan.yaml"
    plan.write_text(text, encoding="utf-8", newline="\n")
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
                {"run_key": "r1", "item_id": prompt, "repeat": repeat, "score_harmful": "PASS"}
            )
            + "\n"
            for prompt in ("p1", "p2")
            for repeat in (1, 2)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return plan, RolloutFiles(runs=runs, rollouts=rollouts)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_a_manifest_with_the_identifiers_still_carries_them(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path, DEFAULT, FULL_MANIFEST)
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "valid", report.integrity.diagnostics
    assert [(item.question_id, item.task_id) for item in report.prompt_estimates] == [
        ("q1", "t1"),
        ("q2", "t1"),
    ]
    written = report.write(tmp_path / "report")
    assert {row["question_id"] for row in _rows(written / "observations.csv")} == {"q1", "q2"}
    assert list(_rows(written / "design.resolved.csv")[0])[:4] == [
        "group_id",
        "task_id",
        "question_id",
        "prompt_id",
    ]


@pytest.mark.parametrize(
    ("design", "named"),
    [
        (
            DEFAULT,
            "(design.hierarchy names question_id, task_id; "
            "design.weights.hierarchy names question_id, task_id)",
        ),
        ("  hierarchy: [prompt_id]", "(design.weights.hierarchy names question_id, task_id)"),
        (FLAT, None),
    ],
)
def test_a_missing_column_names_the_settings_that_need_it(
    tmp_path: Path, design: str, named: str | None
) -> None:
    plan, source = _inputs(tmp_path, design, FLAT_MANIFEST)
    report = analyze(plan=plan, source=source)
    messages = [item.message for item in report.integrity.diagnostics]
    if named is None:
        assert report.integrity.status == "valid", messages
        return
    assert report.integrity.status == "invalid"
    assert [item.code for item in report.integrity.diagnostics] == ["INGEST_FAILED"]
    assert f"design manifest is missing required columns: question_id, task_id {named}" in (
        messages[0]
    )


def test_a_missing_column_in_a_group_s_own_weights_is_named(tmp_path: Path) -> None:
    plan, source = _inputs(tmp_path, FLAT, FLAT_MANIFEST)
    text = plan.read_text(encoding="utf-8").replace(
        "  - id: all\n",
        "  - id: all\n    weights: {strategy: nested_equal, hierarchy: [task_id, prompt_id]}\n",
    )
    plan.write_text(text, encoding="utf-8", newline="\n")
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "invalid"
    assert (
        "missing required columns: task_id (groups[all].weights.hierarchy names task_id)"
        in report.integrity.diagnostics[0].message
    )


def test_blocking_by_an_identifier_the_manifest_lacks_names_the_profile(tmp_path: Path) -> None:
    profiles = "\n  independence_profiles:\n    by_question: {block_by: [question_id]}"
    plan, source = _inputs(tmp_path, FLAT, FLAT_MANIFEST, profiles)
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "invalid"
    assert [item.code for item in report.integrity.diagnostics] == ["INGEST_FAILED"]
    assert (
        "missing required columns: question_id "
        "(design.independence_profiles.by_question.block_by names question_id)"
        in report.integrity.diagnostics[0].message
    )
    # The same profile is fine once the manifest has the column.
    plan, source = _inputs(
        tmp_path,
        "  hierarchy: [question_id, prompt_id]\n"
        "  weights: {strategy: nested_equal, hierarchy: [question_id, prompt_id]}",
        "prompt_id,question_id,expected_rollouts\np1,q1,2\np2,q2,2\n",
        profiles,
    )
    assert analyze(plan=plan, source=source).integrity.status == "valid"
