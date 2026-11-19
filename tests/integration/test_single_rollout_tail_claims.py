"""A claim on a tail or threshold measure over one rollout per prompt degrades integrity.

docs/statistical-methods.md (section 2.4.9) calls the measures nonlinear in each prompt's
failure probability misleading or vacuous at one rollout per prompt, where every per-prompt rate
is 0 or 1: cvar, maximum, top_k_mean, entropic, threshold_exceedance and threshold_excess.
``validate`` reports it, so that such a plan is no longer silently valid, and ``analyze``
reports it the same way. The claim is flagged when at least half its prompts have one rollout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inspect_safety_stats import analyze, validate
from inspect_safety_stats.ingest.rollout_files import RolloutFiles

CODE = "TAIL_MEASURE_SINGLE_ROLLOUT"

PLAN = """
schema_version: "3.0"
analysis: {id: single, title: Single rollout, intent: exploratory}
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
    - {id: tail, type: cvar, beta: 0.5}
    - {id: worst, type: maximum}
    - {id: entropic, type: entropic, lambda: 10}
    - {id: top2, type: top_k_mean, k: 2}
    - {id: exceed, type: threshold_exceedance, threshold: 0.5}
    - {id: excess, type: threshold_excess, threshold: 0.5}
    - {id: attempts, type: worst_of_k, k: 2}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: claims, purpose: all_mandatory, method: bonferroni, alpha: 0.05}
claims:
  - id: low
    title: Low
    cohort: candidate
    group: all
    measure: MEASURE
    family: claims
    acceptable_upper: 0.9
    unacceptable_lower: 0.95
"""


def _source(tmp_path: Path, measure: str, rollouts: dict[str, int]) -> tuple[Path, RolloutFiles]:
    """Write a plan claiming on ``measure``, with ``rollouts[prompt]`` rollouts of each prompt."""

    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\n"
        + "".join(
            f"{prompt},q{index},t1,{count}\n"
            for index, (prompt, count) in enumerate(rollouts.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(PLAN.replace("MEASURE", measure), encoding="utf-8", newline="\n")
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        json.dumps({"run_key": "r1", "model": "m", "status": "success"}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    records = [
        {
            "run_key": "r1",
            "item_id": prompt,
            "repeat": repeat,
            "score_harmful": "FAIL" if (index, repeat) == (0, 1) else "PASS",
        }
        for index, (prompt, count) in enumerate(rollouts.items())
        for repeat in range(1, count + 1)
    ]
    path = tmp_path / "rollouts.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8", newline="\n"
    )
    return plan, RolloutFiles(runs=runs, rollouts=path)


SINGLE = {"p1": 1, "p2": 1, "p3": 1}


@pytest.mark.parametrize(
    ("measure", "kind"),
    [
        ("tail", "cvar"),
        ("worst", "maximum"),
        ("entropic", "entropic"),
        # At one rollout an equal-weight CVaR: the cvar claim's estimate and interval.
        ("top2", "top_k_mean"),
        # Each prompt that failed its one trial counts as above the threshold.
        ("exceed", "threshold_exceedance"),
        ("excess", "threshold_excess"),
    ],
)
def test_validate_warns_on_a_tail_claim_over_single_rollouts(
    tmp_path: Path, measure: str, kind: str
) -> None:
    plan, source = _source(tmp_path, measure, SINGLE)
    summary = validate(plan=plan, source=source)

    (warning,) = [item for item in summary.diagnostics if item.code == CODE]
    assert warning.severity == "WARNING"
    assert warning.scope == "plan/claims/low"
    assert warning.details == {
        "integrity_degraded": True,
        "measure": measure,
        "measure_type": kind,
        "cohort": "candidate",
        "group": "all",
        "prompts_with_rollouts": 3,
        "single_rollout_prompts": 3,
        "median_rollouts_per_prompt": 1,
    }
    # No longer silently valid.
    assert summary.status == "degraded"


def test_analyze_reports_it_as_validate_does(tmp_path: Path) -> None:
    plan, source = _source(tmp_path, "tail", SINGLE)
    report = analyze(plan=plan, source=source)

    assert [item.code for item in report.integrity.diagnostics if item.code == CODE] == [CODE]
    assert report.integrity.status == "degraded"


@pytest.mark.parametrize(
    "measure",
    [
        "mean",
        # No estimate below k rollouts per prompt, and an interval from the prompt bounds.
        "attempts",
    ],
)
def test_a_claim_on_the_weighted_mean_or_worst_of_k_is_not_flagged(
    tmp_path: Path, measure: str
) -> None:
    plan, source = _source(tmp_path, measure, SINGLE)
    summary = validate(plan=plan, source=source)

    assert CODE not in {item.code for item in summary.diagnostics}
    assert summary.status == "valid"


@pytest.mark.parametrize(
    ("rollouts", "flagged"),
    [
        # Repeated rollouts throughout.
        ({"p1": 2, "p2": 2, "p3": 2}, False),
        # The median prompt has repeated rollouts: one single prompt does not flag the claim.
        ({"p1": 1, "p2": 3, "p3": 3}, False),
        # The median prompt has one rollout.
        ({"p1": 1, "p2": 1, "p3": 4}, True),
        # Half the prompts have one rollout: the lower median, 1, flags the claim.
        ({"p1": 1, "p2": 1, "p3": 2, "p4": 2}, True),
        # Fewer than half do.
        ({"p1": 1, "p2": 2, "p3": 2, "p4": 2}, False),
    ],
)
def test_the_median_prompt_decides(
    tmp_path: Path, rollouts: dict[str, int], flagged: bool
) -> None:
    plan, source = _source(tmp_path, "worst", rollouts)
    summary = validate(plan=plan, source=source)

    assert (CODE in {item.code for item in summary.diagnostics}) is flagged
    assert summary.status == ("degraded" if flagged else "valid")
