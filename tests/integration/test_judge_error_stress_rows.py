"""Judge-error stress of tail rows and of rows with unclassified rollouts, end to end.

Each case runs ``analyze`` on synthetic logs and checks the stress record against the
inversion of the claim's own bounds, ``(U - FPR) / (Se - FPR)`` and ``(L - FPR) / (Se - FPR)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from inspect_safety_stats import SafetyReport, analyze
from inspect_safety_stats.stats.judges import JudgeAccuracyBox, identified_risk_set

_PLAN = """
schema_version: "2.0"
analysis:
  id: stress-rows
  title: Stress rows
  intent: confirmatory
  plan_frozen_at: 2025-01-01T00:00:00Z
inputs:
  design_manifest: design.csv
  score:
    name: {score}
    mapping:
      failure_values: {failure_values}
      safe_values: {safe_values}
    missing: {missing}
    errors: block
{judges}design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  rollout_key: [eval_id, id, epoch]
  weights:
    strategy: nested_equal
    hierarchy: [task_id, question_id, prompt_id]
  independence_profiles:
    rollout:
      block_by: [rollout]
      comparison_coupling: independent
cohorts:
  candidate:
    match:
      model: model/candidate
groups:
  - id: cyber
    filter: {{risk_domain: cyber}}
outputs:
  prompt_bounds:
    enabled: true
    simultaneous: true
    family: displayed
    alpha_allocation: equal_bonferroni
  risk_measures:
    - {measure}
multiplicity:
  families:
    - {{id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}}
claims:
  - id: row
    title: The row is acceptable
    cohort: candidate
    group: cyber
    measure: row
    family: gate
    acceptable_upper: {acceptable}
    unacceptable_lower: {unacceptable}
report:
  include_decisions: true
  judge_error_stress: true
"""


def _judge(sensitivity: float, specificity: float) -> str:
    return (
        "  judges:\n"
        "    - id: primary_judge\n"
        "      score: {name: harmful, mapping: {failure_values: [FAIL], safe_values: [PASS]}}\n"
        f"      accuracy: {{sensitivity: {{lower: {sensitivity}, upper: {sensitivity}}}, "
        f"specificity: {{lower: {specificity}, upper: {specificity}}}}}\n"
    )


def _scores(fails: int, missing: int, total: int = 100) -> list[str | None]:
    """``fails`` judge FAILs, then ``missing`` rollouts without a score, then PASSes."""
    return ["FAIL"] * fails + [None] * missing + ["PASS"] * (total - fails - missing)


def _analyze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prompts: dict[str, list[str | None]],
    measure: str,
    acceptable: float,
    unacceptable: float,
    judges: str,
    missing: str = "conservative_bounds",
    extra_scores: dict[str, dict[str, list[str | None]]] | None = None,
    score: str = "harmful",
    failure_values: str = '["FAIL"]',
    safe_values: str = '["PASS"]',
) -> SafetyReport:
    epochs = max(len(scores) for scores in prompts.values())
    evals = pd.DataFrame(
        [
            {
                "eval_id": "eval-1",
                "run_id": "run-1",
                "log": "/not/a/local/log.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "fixed_safety",
                "task_version": "1",
                "model": "model/candidate",
                "model_generate_config": {"temperature": 1},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "fixed",
                "epochs": epochs,
            }
        ]
    )
    rows: list[dict[str, Any]] = []
    for prompt_id, scores in prompts.items():
        for epoch, value in enumerate(scores, start=1):
            row: dict[str, Any] = {
                "sample_id": f"{prompt_id}-{epoch}",
                "eval_id": "eval-1",
                "log": "/not/a/local/log.eval",
                "id": prompt_id,
                "epoch": epoch,
                "score_harmful": value,
                "error": None,
                "retries": 0,
            }
            for name, by_prompt in (extra_scores or {}).items():
                row[f"score_{name}"] = by_prompt[prompt_id][epoch - 1]
            rows.append(row)
    samples = pd.DataFrame(rows)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,risk_domain,expected_rollouts\n"
        + "".join(
            f"{prompt_id},q{prompt_id},t1,cyber,{len(scores)}\n"
            for prompt_id, scores in prompts.items()
        ),
        encoding="utf-8",
        newline="\n",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        _PLAN.format(
            score=score,
            failure_values=failure_values,
            safe_values=safe_values,
            missing=missing,
            judges=judges,
            measure=measure,
            acceptable=acceptable,
            unacceptable=unacceptable,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return analyze(plan=plan, logs="ignored")


def _row(report: SafetyReport) -> Any:
    (claim,) = report.claims
    assert claim.judge_error_stress is not None and claim.judge_error_stress.applicable
    return claim, claim.judge_error_stress


def test_a_maximum_row_inverts_the_upper_bound_of_its_riskiest_prompt_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # p1 carries the maximum and has no unclassified rollout; p2's 5% unclassified share used
    # to be taken out of p1's inversion, which put the true-risk upper bound under the 0.35
    # limit and passed the claim at a sensitivity of 0.5.
    report = _analyze(
        tmp_path,
        monkeypatch,
        prompts={"p1": _scores(10, 0), "p2": _scores(0, 5)},
        measure="{id: row, type: maximum}",
        acceptable=0.35,
        unacceptable=0.95,
        judges=_judge(0.5, 1.0),
    )
    claim, stress = _row(report)
    assert claim.status == "PASS"
    assert stress.true_risk_upper == pytest.approx(claim.upper / 0.5, rel=1e-12, abs=0)
    assert stress.true_risk_upper > 0.35
    assert (stress.stressed_status, stress.robust) == ("INCONCLUSIVE", False)
    assert stress.pass_sensitivity_at_perfect_specificity == pytest.approx(
        claim.upper / 0.35, rel=1e-12, abs=0
    )


def test_a_cvar_row_is_not_passed_by_its_largest_unclassified_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Four prompts judged safe, one with 30% unclassified. A judge that catches one failure
    # in ten leaves true risk unbounded below 1; the share used to report 0.30 and a PASS.
    report = _analyze(
        tmp_path,
        monkeypatch,
        prompts={
            "p1": _scores(0, 30),
            "p2": _scores(0, 0),
            "p3": _scores(0, 0),
            "p4": _scores(0, 0),
        },
        measure="{id: row, type: cvar, beta: 0.5}",
        acceptable=0.35,
        unacceptable=0.95,
        judges=_judge(0.1, 1.0),
    )
    claim, stress = _row(report)
    assert claim.status == "PASS"
    assert stress.true_risk_upper == pytest.approx(min(1.0, claim.upper / 0.1), abs=0)
    assert (stress.stressed_status, stress.robust) == ("INCONCLUSIVE", False)
    # The tipping line is never negative: PASS needs Se >= U / a.
    assert stress.pass_sensitivity_at_perfect_specificity == pytest.approx(
        claim.upper / 0.35, rel=1e-12, abs=0
    )


def test_a_top_k_row_inverts_its_bounds_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _analyze(
        tmp_path,
        monkeypatch,
        prompts={"p1": _scores(12, 0), "p2": _scores(8, 10), "p3": _scores(0, 20)},
        measure="{id: row, type: top_k_mean, k: 2}",
        acceptable=0.40,
        unacceptable=0.95,
        judges=_judge(0.7, 0.95),
    )
    claim, stress = _row(report)
    fpr, gap = 0.05, 0.7 - 0.05
    assert stress.true_risk_upper == pytest.approx(
        min(1.0, (claim.upper - fpr) / gap), rel=1e-12, abs=0
    )
    assert stress.true_risk_lower == pytest.approx(
        max(0.0, (claim.lower - fpr) / gap), rel=1e-12, abs=1e-15
    )


def test_rollouts_dropped_under_exclude_take_no_share_of_the_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The prompt bounds drop p2's five unscored rollouts, so nothing of the mean is
    # unclassified; a share of 0.025 used to lower the true-risk upper bound below 0.22.
    report = _analyze(
        tmp_path,
        monkeypatch,
        prompts={"p1": _scores(10, 0), "p2": _scores(0, 5)},
        measure="{id: row, type: weighted_mean}",
        acceptable=0.22,
        unacceptable=0.95,
        judges=_judge(0.5, 1.0),
        missing="exclude",
    )
    claim, stress = _row(report)
    assert claim.status == "PASS"
    assert stress.true_risk_upper == pytest.approx(claim.upper / 0.5, rel=1e-12, abs=0)
    assert (stress.stressed_status, stress.robust) == ("INCONCLUSIVE", False)


def test_a_maximum_rows_fail_is_stressed_on_its_own_lower_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # p1 holds the lower bound and is fully classified; p2's 30% unclassified share used to
    # shrink the false-alarm term of p1's inversion and keep the FAIL.
    report = _analyze(
        tmp_path,
        monkeypatch,
        prompts={"p1": _scores(60, 0), "p2": _scores(0, 30)},
        measure="{id: row, type: maximum}",
        acceptable=0.10,
        unacceptable=0.42,
        judges=_judge(0.95, 0.8),
    )
    claim, stress = _row(report)
    assert claim.status == "FAIL"
    assert stress.true_risk_lower == pytest.approx(
        (claim.lower - 0.2) / 0.75, rel=1e-12, abs=0
    )
    assert stress.true_risk_lower < 0.42
    assert (stress.stressed_status, stress.robust) == ("INCONCLUSIVE", False)


def test_the_tipping_line_and_the_box_agree_on_a_tail_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The box is the point Se = 0.5 on the perfect-specificity edge, so the PASS line's value
    # there decides the box's verdict. They used to disagree when p2's unclassified share
    # exceeded what the row's bounds left for it.
    report = _analyze(
        tmp_path,
        monkeypatch,
        prompts={"p1": _scores(35, 0), "p2": _scores(0, 30)},
        measure="{id: row, type: maximum}",
        acceptable=0.66,
        unacceptable=0.95,
        judges=_judge(0.5, 1.0),
    )
    claim, stress = _row(report)
    assert claim.status == "PASS"
    line = stress.pass_sensitivity_at_perfect_specificity
    assert line is not None and line >= 0
    assert (stress.stressed_status == "PASS") == (0.5 >= line)
    assert stress.stressed_status == "INCONCLUSIVE"


_COMPOSITE = (
    "  judges:\n"
    "    - id: judge_a\n"
    "      score: {name: harmful_a, mapping: {failure_values: [FAIL], safe_values: [PASS]}}\n"
    "      block: openai\n"
    "      accuracy: {sensitivity: {lower: 0.8, upper: 0.95}, "
    "specificity: {lower: 0.9, upper: 0.99}}\n"
    "    - id: judge_b\n"
    "      score: {name: harmful_b, mapping: {failure_values: [FAIL], safe_values: [PASS]}}\n"
    "      block: openai\n"
    "      accuracy: {sensitivity: {lower: 0.7, upper: 0.9}, "
    "specificity: {lower: 0.92, upper: 0.99}}\n"
    "  composites:\n"
    "    - {id: any_judge, rule: any, judges: [judge_a, judge_b]}\n"
)


@pytest.mark.parametrize("measure", ["weighted_mean", "maximum"])
def test_a_composites_stress_is_its_panel_derived_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measure: str
) -> None:
    # Judge b is silent on p2's first 20 rollouts, so the composite leaves them unclassified.
    a = {"p1": _scores(30, 0), "p2": _scores(5, 0)}
    b = {"p1": _scores(25, 0), "p2": [None] * 20 + _scores(0, 0, 80)}
    report = _analyze(
        tmp_path,
        monkeypatch,
        prompts={"p1": _scores(0, 0), "p2": _scores(0, 0)},
        measure=f"{{id: row, type: {measure}}}",
        acceptable=0.5,
        unacceptable=0.95,
        judges=_COMPOSITE,
        extra_scores={"harmful_a": a, "harmful_b": b},
        score="any_judge",
        failure_values="[true]",
        safe_values="[false]",
    )
    claim, stress = _row(report)
    assert stress.accuracy_source == "composite:any_judge"
    (row,) = [item for item in report.group_risks if item.measure_id == "row"]
    derived = next(
        item for item in row.intervals if item.interval_id == "judge_corrected_panel_derived"
    )
    assert (stress.true_risk_lower, stress.true_risk_upper) == (derived.lower, derived.upper)
    box = JudgeAccuracyBox(
        stress.sensitivity_lower,
        stress.sensitivity_upper,
        stress.specificity_lower,
        stress.specificity_upper,
    )
    expected = identified_risk_set(claim.lower, claim.upper, box)
    assert (stress.true_risk_lower, stress.true_risk_upper) == (expected.lower, expected.upper)
