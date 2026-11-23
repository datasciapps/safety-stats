"""A rollout that an ``exclude`` policy leaves out of the bounds is reported, not left silent.

``inputs.score.missing: exclude`` and ``inputs.score.errors: exclude`` leave an unscored or
errored rollout out of the bounds, which then rest on its outcome being like the classified
rollouts'. The inputs cannot confirm that, so each such rollout gets a warning and integrity is
degraded, as under the conservative default; the bounds and the claim stay as the plan asks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safety_stats import analyze
from safety_stats.ingest.rollout_files import RolloutFiles

PLAN = """
schema_version: "2.0"
analysis:
  id: excluded
  title: Excluded rollouts
  intent: confirmatory
  plan_frozen_at: 2026-01-01T00:00:00Z
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
    POLICY: exclude
design:
  rollout_key: [run_key, item_id, repeat]
cohorts:
  candidate:
    match: {team: red}
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
    - {id: gate, purpose: all_mandatory, method: bonferroni, alpha: 0.05}
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
"""


def _jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8", newline="\n"
    )
    return path


@pytest.mark.parametrize(
    ("policy", "code", "errored"),
    [("missing", "SCORE_MISSING_EXCLUDED", False), ("errors", "SAMPLE_ERROR_EXCLUDED", True)],
)
def test_each_excluded_rollout_is_reported_and_degrades_integrity(
    tmp_path: Path, policy: str, code: str, errored: bool
) -> None:
    # The review's case: one prompt of 40 rollouts, 20 scored PASS and 20 without a score.
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\np0,q0,t1,40\n",
        encoding="utf-8",
        newline="\n",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(PLAN.replace("POLICY", policy), encoding="utf-8", newline="\n")
    runs = _jsonl(
        tmp_path / "runs.jsonl",
        [
            {
                "run_key": "r1",
                "team": "red",
                "status": "success",
                "started_at": "2026-02-01T00:00:00Z",
                "model": "m",
            }
        ],
    )
    records: list[dict[str, object]] = []
    for repeat in range(1, 41):
        record: dict[str, object] = {"run_key": "r1", "item_id": "p0", "repeat": repeat}
        if repeat <= 20:
            record["score_harmful"] = "PASS"
        elif errored:
            record["error"] = "RuntimeError: harness failed"
        records.append(record)
    rollouts = _jsonl(tmp_path / "rollouts.jsonl", records)

    report = analyze(plan=plan, source=RolloutFiles(runs=runs, rollouts=rollouts))

    found = [item for item in report.integrity.diagnostics if item.code == code]
    assert len(found) == 20
    assert {item.severity for item in found} == {"WARNING"}
    # Before, integrity was valid and no diagnostic named the 20 rollouts left out.
    assert report.integrity.status == "degraded"
    # The bounds still cover the 20 classified rollouts alone, as the plan asks, so the claim
    # passes on them; the warning says what that rests on.
    (claim,) = report.claims
    assert claim.upper is not None and claim.upper < 0.2
    assert claim.status == "PASS"
