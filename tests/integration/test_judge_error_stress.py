"""Judge-error stress of threshold claims, end to end through ``analyze``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.stats.judge_stress import stress_verdict
from inspect_safety_stats.stats.judges import JudgeAccuracyBox

_PLAN = """
schema_version: "2.0"
analysis:
  id: stress
  title: Stress analysis
  intent: confirmatory
  plan_frozen_at: 2025-01-01T00:00:00Z
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
    missing: block
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
  direct_mean_intervals:
    enabled: true
    method: weighted_hoeffding
    family: direct_means
    groups: [cyber]
    simultaneous: true
    coverage_mode: two_sided
    independence:
      primary: rollout
  risk_measures:
    - {{id: mean, type: weighted_mean}}
    - {{id: worst_of_2, type: worst_of_k, k: 2}}
multiplicity:
  families:
    - {{id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}}
claims:
  - id: mean_safe
    title: Mean risk is acceptable
    cohort: candidate
    group: cyber
    measure: mean
    family: gate
    acceptable_upper: 0.30
    unacceptable_lower: 0.60
  - id: worst_of_2_safe
    title: Worst of two is acceptable
    cohort: candidate
    group: cyber
    measure: worst_of_2
    family: gate
    acceptable_upper: 0.50
    unacceptable_lower: 0.90
report:
  include_decisions: {include_decisions}
{stress}"""

_JUDGES = (
    "  judges:\n"
    "    - id: primary_judge\n"
    "      score: {name: harmful, mapping: {failure_values: [FAIL], safe_values: [PASS]}}\n"
    "      accuracy: {sensitivity: {lower: 0.8, upper: 0.95}, "
    "specificity: {lower: 0.95, upper: 0.99}}\n"
)


def _write(
    tmp_path: Path,
    *,
    stress: str = "  judge_error_stress: true\n",
    judges: str = _JUDGES,
    include_decisions: bool = True,
) -> Path:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,risk_domain,expected_rollouts\n"
        "p1,q1,t1,cyber,20\np2,q2,t1,cyber,20\n",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        _PLAN.format(
            judges=judges,
            stress=stress,
            include_decisions=str(include_decisions).lower(),
        ),
        encoding="utf-8",
    )
    return plan


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
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
                "epochs": 20,
            }
        ]
    )
    rows = [
        {
            "sample_id": f"{prompt_id}-{epoch}",
            "eval_id": "eval-1",
            "log": "/not/a/local/log.eval",
            "id": prompt_id,
            "epoch": epoch,
            "score_harmful": "FAIL" if prompt_id == "p1" and epoch <= 2 else "PASS",
            "error": None,
            "retries": 0,
        }
        for prompt_id in ("p1", "p2")
        for epoch in range(1, 21)
    ]
    return evals, pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _offline_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    evals, samples = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )


def test_a_claim_reports_its_tipping_lines_and_the_verdict_under_the_declared_box(
    tmp_path: Path,
) -> None:
    report = analyze(plan=_write(tmp_path), logs="ignored")

    claim = next(item for item in report.claims if item.claim_id == "mean_safe")
    stress = claim.judge_error_stress
    assert stress is not None and stress.applicable
    assert claim.lower is not None and claim.upper is not None
    expected = stress_verdict(
        judge_reported_lower=claim.lower,
        judge_reported_upper=claim.upper,
        acceptable_upper=0.30,
        unacceptable_lower=0.60,
        box=JudgeAccuracyBox(0.8, 0.95, 0.95, 0.99),
    )
    assert expected.stressed is not None
    assert stress.pass_sensitivity_at_perfect_specificity == pytest.approx(
        claim.upper / 0.30, rel=1e-12, abs=0
    )
    assert stress.fail_sensitivity_at_perfect_specificity == pytest.approx(
        claim.lower / 0.60, rel=1e-12, abs=0
    )
    assert stress.accuracy_source == "judge:primary_judge"
    assert stress.stressed_status == expected.stressed.verdict
    assert stress.true_risk_upper == pytest.approx(
        expected.stressed.true_risk_upper, rel=1e-12, abs=0
    )
    assert stress.robust == (expected.stressed.verdict == claim.status)
    # Upper bound 0.265 against 0.30 needs Se >= 0.88 at perfect specificity; the declared
    # box admits 0.80, so the PASS does not survive it.
    assert claim.status == "PASS"
    assert stress.pass_sensitivity_at_perfect_specificity == pytest.approx(0.8824, abs=1e-4)
    assert (stress.stressed_status, stress.robust) == ("INCONCLUSIVE", False)
    # Diagnostic only: the claim is still decided on the judge-reported bounds.
    assert claim.method != "derived_panel_misclassification_correction"
    assert all("judge-error" not in reason.lower() for reason in claim.reasons)

    other = next(item for item in report.claims if item.claim_id == "worst_of_2_safe")
    assert other.judge_error_stress is not None
    assert not other.judge_error_stress.applicable
    assert "worst_of_k" in str(other.judge_error_stress.reason)

    output = report.write(tmp_path / "report")
    markdown = (output / "report-full.md").read_text(encoding="utf-8")
    assert "## Judge-error stress (experimental)" in markdown
    assert "Declared accuracy of `judge:primary_judge`" in markdown
    assert "Worst of two is acceptable: a worst_of_k measure" in markdown
    stored = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert {item["claim_id"]: "judge_error_stress" in item for item in stored["claims"]} == {
        "mean_safe": True,
        "worst_of_2_safe": True,
    }


def test_without_a_declared_accuracy_only_the_tipping_lines_are_reported(tmp_path: Path) -> None:
    report = analyze(plan=_write(tmp_path, judges=""), logs="ignored")

    claim = next(item for item in report.claims if item.claim_id == "mean_safe")
    stress = claim.judge_error_stress
    assert stress is not None and stress.applicable
    assert stress.pass_sensitivity_at_perfect_specificity is not None
    assert (stress.accuracy_source, stress.stressed_status, stress.robust) == (None, None, None)


def test_a_report_without_decisions_says_where_the_stress_is(tmp_path: Path) -> None:
    report = analyze(plan=_write(tmp_path, include_decisions=False), logs="ignored")

    markdown = (report.write(tmp_path / "report") / "report-full.md").read_text(encoding="utf-8")
    assert "## Judge-error stress (experimental)" in markdown
    assert "only in `report.json`" in markdown


def test_a_plan_that_does_not_opt_in_writes_what_it_wrote_before(tmp_path: Path) -> None:
    (tmp_path / "off").mkdir()
    (tmp_path / "explicit").mkdir()
    off = _write(tmp_path / "off", stress="")
    explicit = _write(tmp_path / "explicit", stress="  judge_error_stress: false\n")
    assert canonical_plan_hash(load_plan(off)) == canonical_plan_hash(load_plan(explicit))

    output = analyze(plan=off, logs="ignored").write(tmp_path / "report")
    # The report's record of the defaults applied lists the option, and nothing else does.
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert {"path": "report.judge_error_stress", "value": False} in report.pop("plan_defaults")
    assert "judge_error_stress" not in json.dumps(report)
    for name in ("report.md", "report-full.md", "plan.resolved.yaml"):
        assert "judge_error_stress" not in (output / name).read_text(encoding="utf-8")
    assert "Judge-error stress" not in (output / "report-full.md").read_text(encoding="utf-8")


def test_an_exploratory_claim_is_not_stressed_against_its_forced_inconclusive(
    tmp_path: Path,
) -> None:
    plan = _write(tmp_path)
    plan.write_text(
        plan.read_text(encoding="utf-8").replace(
            "intent: confirmatory", "intent: exploratory"
        ),
        encoding="utf-8",
    )
    report = analyze(plan=plan, logs="ignored")

    claim = next(item for item in report.claims if item.claim_id == "mean_safe")
    # The same bounds give PASS in a confirmatory analysis; here the claim is INCONCLUSIVE,
    # which no judge accuracy would change, so there is nothing to stress.
    assert (claim.status, claim.descriptive_status) == ("INCONCLUSIVE", "PASS")
    stress = claim.judge_error_stress
    assert stress is not None and not stress.applicable
    assert "exploratory" in str(stress.reason)
    assert stress.robust is None


@pytest.mark.parametrize(("scope", "applied"), [("[cyber]", True), ("[first]", False)])
def test_a_box_declared_for_other_groups_is_not_applied_to_the_claim(
    tmp_path: Path, scope: str, applied: bool
) -> None:
    judges = _JUDGES.replace("upper: 0.99}}\n", f"upper: 0.99}}, groups: {scope}}}\n")
    plan = _write(tmp_path, judges=judges)
    plan.write_text(
        plan.read_text(encoding="utf-8").replace(
            "groups:\n  - id: cyber\n",
            "groups:\n  - id: first\n    filter: {question_id: q1}\n  - id: cyber\n",
            1,
        ),
        encoding="utf-8",
    )
    report = analyze(plan=plan, logs="ignored")

    claim = next(item for item in report.claims if item.claim_id == "mean_safe")
    stress = claim.judge_error_stress
    assert stress is not None and stress.applicable
    # The tipping lines need no box, so they are reported either way.
    assert stress.pass_sensitivity_at_perfect_specificity is not None
    if applied:
        assert stress.accuracy_source == "judge:primary_judge"
        assert stress.stressed_status is not None
    else:
        assert (stress.accuracy_source, stress.stressed_status, stress.robust) == (
            None,
            None,
            None,
        )
