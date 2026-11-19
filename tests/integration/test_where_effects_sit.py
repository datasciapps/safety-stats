"""Through analyze: a hierarchy node whose prompt has no classified rollout is still placed.

A node has no estimate when one of its prompts has no classified rollout, but under
``conservative_bounds`` or ``exclude`` it still has a missingness-robust interval. When that
interval is wholly above a claim's limit, "Where effects sit" lists the node; it used to look
only at estimates, and said that no node was above the limit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inspect_safety_stats import analyze
from inspect_safety_stats.ingest.rollout_files import RolloutFiles

PLAN = """
schema_version: "3.0"
analysis:
  id: where-effects-sit
  title: A node without an estimate
  intent: exploratory
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
    missing: {missing}
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
  model:
    match: {{model: model/a}}
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
    hierarchy_nodes: true
  risk_measures:
    - {{id: mean, type: weighted_mean}}
multiplicity:
  families:
    - {{id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}}
claims:
  - id: model_safe
    title: Model risk below 5%
    cohort: model
    group: all
    measure: mean
    family: gate
    acceptable_upper: 0.05
    unacceptable_lower: 0.2
    mandatory: true
report:
  include_decisions: true
  decimal_places: 4
"""

ROLLOUTS = 20
# Task t1 holds question q1, whose prompt p1 has no scored rollout and whose prompt p2 fails
# 19 times in 20; task t2 never fails.
DESIGN = {"p1": ("q1", "t1"), "p2": ("q1", "t1"), "p3": ("q2", "t2"), "p4": ("q2", "t2")}


def _score(prompt_id: str, repeat: int) -> str | None:
    if prompt_id == "p1":
        return None
    if prompt_id == "p2" and repeat <= 19:
        return "FAIL"
    return "PASS"


def _inputs(directory: Path, missing: str) -> tuple[Path, RolloutFiles]:
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\n"
        + "".join(
            f"{prompt_id},{question},{task},{ROLLOUTS}\n"
            for prompt_id, (question, task) in DESIGN.items()
        ),
        encoding="utf-8",
        newline="\n",
    )
    runs = directory / "runs.jsonl"
    runs.write_text(
        json.dumps({"run_key": "run-a", "model": "model/a", "status": "success"}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    rollouts = directory / "rollouts.jsonl"
    with rollouts.open("w", encoding="utf-8", newline="\n") as stream:
        for prompt_id in DESIGN:
            for repeat in range(1, ROLLOUTS + 1):
                record = {
                    "run_key": "run-a",
                    "item_id": prompt_id,
                    "repeat": repeat,
                    "rollout_id": f"{prompt_id}-{repeat}",
                    "scores": {"harmful": _score(prompt_id, repeat)},
                }
                stream.write(json.dumps(record) + "\n")
    plan = directory / "plan.yaml"
    plan.write_text(PLAN.format(missing=missing), encoding="utf-8", newline="\n")
    return plan, RolloutFiles(runs=runs, rollouts=rollouts)


@pytest.mark.parametrize("missing", ["conservative_bounds", "exclude"])
def test_a_node_whose_interval_is_above_the_limit_is_listed_without_an_estimate(
    tmp_path: Path, missing: str
) -> None:
    plan, source = _inputs(tmp_path, missing)
    report = analyze(plan=plan, source=source)
    t1 = next(
        node
        for node in report.hierarchy_risks
        if node.level == "task_id" and node.path.get("task_id") == "t1"
    )
    # The premise: no estimate, and an interval wholly above the claim's limit.
    assert t1.estimate is None
    assert t1.lower is not None and t1.lower > 0.05
    full = (report.write(tmp_path / "report") / "report-full.md").read_text(encoding="utf-8")
    section = full[full.index("## Where effects sit") : full.index("## Data integrity")]
    assert "No hierarchy node's interval or estimate is above" not in section
    rows = [line for line in section.splitlines() if line.startswith("| Model risk below 5%")]
    labels = {line.split(" | ")[2] for line in rows}
    assert {"task_id: t1", "question_id: t1 → q1"} <= labels
    for line in rows:
        if "t1" in line.split(" | ")[2]:
            assert line.split(" | ")[5] == "—"
            assert line.endswith("| interval above the limit |")
