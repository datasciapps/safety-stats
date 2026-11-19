"""The report says when the plan leaves unclassified rollouts out of the bounds.

report-full.md says that missingness-robust bounds cover both completions of unclassified
outcomes, the judge-error stress says that an unclassified rollout counts as judge-unsafe at the
upper bound, and report.html says the unclassified outcomes are "Included in robust bounds".
Under the default score policies each holds. Under ``inputs.score.missing: exclude`` or
``inputs.score.errors: exclude`` those rollouts are left out of the bounds, which cover only the
classified rollouts: with 20 PASS and 20 rollouts without a score, the claim's upper bound is
0.168 instead of 0.662. The report says so then, and only then, so that no other report changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inspect_safety_stats import SafetyReport, analyze
from inspect_safety_stats.ingest.rollout_files import RolloutFiles

_PLAN = """
schema_version: "2.0"
analysis:
  id: exclusion
  title: Exclusion
  intent: confirmatory
  plan_frozen_at: 2026-01-01T00:00:00Z
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
    missing: {missing}
    errors: {errors}
  judges:
    - id: primary_judge
      score: {{name: harmful, mapping: {{failure_values: [FAIL], safe_values: [PASS]}}}}
      accuracy:
        sensitivity: {{lower: 0.9, upper: 0.9}}
        specificity: {{lower: 0.95, upper: 0.95}}
design:
  rollout_key: [run_key, item_id, repeat]
cohorts:
  candidate:
    match: {{team: red}}
groups:
  - id: all
outputs:
  prompt_bounds:
    enabled: true
    family: displayed
  risk_measures:
    - {{id: mean, type: weighted_mean}}
multiplicity:
  families:
    - {{id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: gate, purpose: all_mandatory, method: bonferroni, alpha: 0.05}}
claims:
  - id: mean_ok
    title: Mean failure risk is acceptable
    cohort: candidate
    group: all
    measure: mean
    family: gate
    acceptable_upper: 0.2
    unacceptable_lower: 0.5
    mandatory: true
report:
  include_decisions: true
  judge_error_stress: true
"""

PREAMBLE = (
    "Where outcomes are unclassified, reported missingness-robust bounds cover both the "
    "all-safe and all-failure completions."
)
STRESS = (
    "an unclassified rollout counts as judge-unsafe at the upper bound and judge-safe at the "
    "lower, which is conservative on both sides."
)
NOTE = 'unclassified ? "Included in robust bounds" : "None"'
NOTE_EXCLUDED = (
    'unclassified ? "Included in robust bounds, except those the score policy excludes" : "None"'
)


def _write(
    tmp_path: Path, *, missing: str, errors: str, unknown: str
) -> tuple[SafetyReport, str, str]:
    """20 PASS rollouts of one prompt and 20 without a score, as missing or errored ones."""

    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\np0,q0,t1,40\n",
        encoding="utf-8",
        newline="\n",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(_PLAN.format(missing=missing, errors=errors), encoding="utf-8", newline="\n")
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        json.dumps(
            {
                "run_key": "r1",
                "team": "red",
                "status": "success",
                "started_at": "2026-02-01T00:00:00Z",
                "model": "m",
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    records: list[dict[str, object]] = []
    for index in range(40):
        record: dict[str, object] = {"run_key": "r1", "item_id": "p0", "repeat": index + 1}
        if index < 20:
            record["score_harmful"] = "PASS"
        elif unknown == "error":
            record["error"] = "RuntimeError: the harness failed"
        records.append(record)
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8", newline="\n"
    )
    report = analyze(plan=plan, source=RolloutFiles(runs=runs, rollouts=rollouts))
    output = tmp_path / "report"
    report.write(output)
    return (
        report,
        (output / "report-full.md").read_text(encoding="utf-8"),
        (output / "report.html").read_text(encoding="utf-8"),
    )


@pytest.mark.parametrize(
    ("missing", "errors", "unknown", "excluded"),
    [
        ("exclude", "end_to_end", "missing", "`inputs.score.missing: exclude`, though, a "
         "rollout without a score"),
        ("conservative_bounds", "exclude", "error", "`inputs.score.errors: exclude`, though, a "
         "rollout with a sample error"),
        ("exclude", "exclude", "missing", "`inputs.score.missing: exclude` and "
         "`inputs.score.errors: exclude`, though, a rollout without a score or with a sample "
         "error"),
    ],
    ids=["missing-exclude", "errors-exclude", "both-exclude"],
)
def test_the_report_says_that_excluded_rollouts_are_left_out_of_the_bounds(
    tmp_path: Path, missing: str, errors: str, unknown: str, excluded: str
) -> None:
    report, full, page = _write(tmp_path, missing=missing, errors=errors, unknown=unknown)

    (estimate,) = report.prompt_estimates
    assert (estimate.n_unclassified, estimate.missingness_robust) == (20, False)
    assert report.integrity.status == "valid"
    sentence = (
        f" Under {excluded} is left out of the bounds, which then cover only the classified "
        "rollouts."
    )
    assert PREAMBLE + sentence in full
    stress = full.split("## Judge-error stress (experimental)", 1)[1]
    assert STRESS + sentence + " Where the primary" in stress
    assert NOTE not in page
    assert page.count(NOTE_EXCLUDED) == 1


@pytest.mark.parametrize(
    ("missing", "errors", "unknown"),
    [
        ("conservative_bounds", "end_to_end", "missing"),
        ("conservative_bounds", "end_to_end", "error"),
        ("conservative_bounds", "conservative_bounds", "error"),
    ],
)
def test_bounds_that_cover_every_completion_keep_their_wording(
    tmp_path: Path, missing: str, errors: str, unknown: str
) -> None:
    report, full, page = _write(tmp_path, missing=missing, errors=errors, unknown=unknown)

    (estimate,) = report.prompt_estimates
    assert (estimate.n_unclassified, estimate.missingness_robust) == (20, True)
    assert PREAMBLE + "\n" in full
    assert STRESS + " Where the primary" in full
    assert "left out of the bounds" not in full
    assert page.count(NOTE) == 1
    assert NOTE_EXCLUDED not in page
