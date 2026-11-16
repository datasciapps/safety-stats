"""A validated judge correction certifies only the estimand its labels measure (#95).

The labels are drawn within judge strata over every classified rollout of a cohort, so they
estimate the judge's accuracy with each of those rollouts weighted equally. The identity
R* = R rho_1 + (1 - R) rho_0 needs the accuracy weighted the way the row weights its rollouts.

In this design one hard prompt carries half the weight, and nine easy prompts share the other
half. The judge misses 40% of the hard prompt's failures, and the easy prompts' false positives
dominate the pooled labels. So the pooled rho_1 is 36/75 = 0.48 where the design-weighted one
is about 0.84, and the correction understates true risk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.schema.reports import ComparisonResult, GroupRiskResult, SafetyReport

EPOCHS = 200
PROMPTS = [("h1", "qh", "tH", "hard")] + [(f"e{i}", f"qe{i}", "tE", "easy") for i in range(1, 10)]


def _truly_unsafe(prompt_id: str, epoch: int) -> bool:
    # True failure rate 0.30 on the hard prompt and 0 on the easy ones.
    return prompt_id == "h1" and epoch <= 60


def _judge_unsafe(prompt_id: str, epoch: int) -> bool:
    if prompt_id == "h1":
        # Sensitivity 0.6, and three false positives among the 140 safe rollouts.
        return epoch <= 36 or 61 <= epoch <= 63
    # A false-positive rate of 0.02.
    return epoch <= 4


PLAN = """
schema_version: "2.0"
analysis: {id: scope, title: Judge correction scope, intent: confirmatory,
           plan_frozen_at: 2025-01-01T00:00:00Z}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
    missing: block
    errors: block
@JUDGE@
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  rollout_key: [eval_id, id, epoch]
  require_complete_manifest: true
  require_expected_rollouts: true
  weights: {strategy: @WEIGHTS@, hierarchy: [task_id, question_id, prompt_id]}
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
cohorts:
  candidate: {match: {model: model/candidate}, require_uniform: [task_version]}
@BASELINE@
groups:
  - id: all
    filter: {suite: s}
    weights: {strategy: @WEIGHTS@, hierarchy: [task_id, question_id, prompt_id]}
  - id: hard
    filter: {risk_domain: hard}
    weights: {strategy: @WEIGHTS@, hierarchy: [task_id, question_id, prompt_id]}
outputs:
  prompt_bounds: {enabled: true, simultaneous: true, family: displayed,
                  alpha_allocation: equal_bonferroni}
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    family: direct_means
    groups: [all, hard]
    simultaneous: true
    coverage_mode: two_sided
    independence: {primary: rollout}
  risk_measures:
    - {id: mean, type: weighted_mean}
    - {id: worst, type: maximum}
@COMPARISONS@
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: model_comparisons, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: judge_error, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
@CLAIMS@
policy: {zero_tolerance: {enabled: true, manifest_field: zero_tolerance}}
report: {include_observations: true, decimal_places: 6}
"""

CORRECTED_CLAIMS = """
  - {id: mean_all, title: All, cohort: candidate, group: all, measure: mean, family: gate,
     acceptable_upper: 0.12, unacceptable_lower: 0.40}
  - {id: mean_hard, title: Hard, cohort: candidate, group: hard, measure: mean, family: gate,
     acceptable_upper: 0.25, unacceptable_lower: 0.60}
  - {id: worst_all, title: Worst, cohort: candidate, group: all, measure: worst, family: gate,
     acceptable_upper: 0.25, unacceptable_lower: 0.60}
""".strip("\n")

# A second cohort halves each cohort's share of the judge budget, so it is added only where a
# comparison needs it.
BASELINE = "  baseline: {match: {model: model/baseline}, require_uniform: [task_version]}"
COMPARISON = """
comparisons:
  - id: hard_candidate_vs_baseline
    candidate: candidate
    baseline: baseline
    group: hard
    measure: mean
    estimand: risk_difference
    method: weighted_kl_chernoff
    family: model_comparisons
    independence: {primary: rollout}
""".strip("\n")

NOISY_VALID_CLAIM = """
  - {id: mean_all, title: All, cohort: candidate, group: all, measure: mean, family: gate,
     acceptable_upper: 0.12, unacceptable_lower: 0.40, method: noisy_valid_test}
""".strip("\n")


def _frames(cohorts: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    evals = []
    samples = []
    for cohort in cohorts:
        eval_id = f"eval-{cohort}"
        evals.append(
            {
                "eval_id": eval_id,
                "run_id": "run-1",
                "log": "/not/a/local/log.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "fixed_safety",
                "task_version": "1",
                "model": f"model/{cohort}",
                "model_generate_config": {"temperature": 1},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "fixed",
                "epochs": EPOCHS,
            }
        )
        for prompt_id, *_ in PROMPTS:
            for epoch in range(1, EPOCHS + 1):
                samples.append(
                    {
                        "sample_id": f"{cohort}-{prompt_id}-{epoch}",
                        "eval_id": eval_id,
                        "log": "/not/a/local/log.eval",
                        "id": prompt_id,
                        "epoch": epoch,
                        "score_harmful": "FAIL" if _judge_unsafe(prompt_id, epoch) else "PASS",
                        "error": None,
                        "retries": 0,
                    }
                )
    return pd.DataFrame(evals), pd.DataFrame(samples)


def _write_plan(
    directory: Path,
    *,
    weights: str,
    claims: str,
    sampling: str | None,
    calibrated: bool,
    compare: bool,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,risk_domain,suite,expected_rollouts,zero_tolerance\n"
        + "".join(f"{p},{q},{t},{d},s,{EPOCHS},false\n" for p, q, t, d in PROMPTS),
        encoding="utf-8",
    )
    judge = (
        "  judge_calibration:\n"
        "    mode: validated\n"
        "    validation_manifest: labels.csv\n"
        "    family: judge_error\n" + (f"    sampling: {sampling}\n" if sampling else "")
        if calibrated
        else ""
    )
    text = (
        PLAN.replace("@JUDGE@\n", judge)
        .replace("@BASELINE@\n", BASELINE + "\n" if compare else "")
        .replace("@COMPARISONS@\n", COMPARISON + "\n" if compare else "")
        .replace("@WEIGHTS@", weights)
        .replace("@CLAIMS@", claims)
        .strip()
    )
    plan = directory / "plan.yaml"
    plan.write_text(text + "\n", encoding="utf-8")
    return plan


def _analyze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    weights: str = "nested_equal",
    claims: str = CORRECTED_CLAIMS,
    sampling: str | None = None,
    compare: bool = False,
) -> SafetyReport:
    """Label every classified rollout with its truth, then analyse with the labels."""

    eval_frame, sample_frame = _frames(("candidate", "baseline") if compare else ("candidate",))
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    unlabelled = _write_plan(
        tmp_path / "unlabelled",
        weights=weights,
        claims=claims,
        sampling=None,
        calibrated=False,
        compare=compare,
    )
    baseline = analyze(plan=load_plan(unlabelled), logs="ignored")
    rows = ["observation_id,human_label"]
    for observation in baseline._observations:
        if observation.failure is None:
            continue
        epoch = int(observation.rollout_id.rsplit("-", 1)[1])
        truth = _truly_unsafe(observation.prompt_id, epoch)
        rows.append(f"{observation.observation_id},{'unsafe' if truth else 'safe'}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    plan = _write_plan(
        tmp_path,
        weights=weights,
        claims=claims,
        sampling=sampling,
        calibrated=True,
        compare=compare,
    )
    return analyze(plan=load_plan(plan), logs="ignored")


def _risk(report: SafetyReport, group: str, measure: str) -> GroupRiskResult:
    return next(
        row
        for row in report.group_risks
        if row.cohort_id == "candidate" and row.group_id == group and row.measure_id == measure
    )


def _corrected(row: GroupRiskResult | ComparisonResult) -> object:
    return next(item for item in row.intervals if item.interval_id == "judge_corrected_validated")


def test_a_pooled_calibration_never_certifies_a_row_it_does_not_measure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _analyze(tmp_path, monkeypatch)

    # The pooled correction understates every true risk: all 0.15, hard 0.30, worst 0.30.
    # Each corrected upper bound clears its claim's threshold, so these claims used to PASS.
    mean_all = _risk(report, "all", "mean")
    corrected = _corrected(mean_all)
    assert corrected.upper is not None and corrected.upper < 0.12 < 0.15
    for group, measure, threshold in [("hard", "mean", 0.25), ("all", "worst", 0.25)]:
        interval = _corrected(_risk(report, group, measure))
        assert interval.upper is not None and interval.upper < threshold < 0.30

    # Now each is a sensitivity: no coverage, no decision, and a reason.
    gaps = {
        "mean_all": "9 times the weight of others",
        "mean_hard": "covers only part of the cohort's labelled rollouts",
        "worst_all": "a maximum measure concentrates on the riskiest prompts",
    }
    for (group, measure), claim_id in [
        (("all", "mean"), "mean_all"),
        (("hard", "mean"), "mean_hard"),
        (("all", "worst"), "worst_all"),
    ]:
        interval = _corrected(_risk(report, group, measure))
        assert interval.eligibility == "diagnostic"
        assert interval.role == "sensitivity"
        assert interval.varied_assumption == "judge_error"
        assert interval.simultaneous_coverage is None
        assert interval.upper_one_sided_coverage is None
        (warning,) = interval.warnings
        assert gaps[claim_id] in warning
        claim = next(item for item in report.claims if item.claim_id == claim_id)
        assert claim.status == "INCONCLUSIVE"
        assert warning in claim.reasons
    assert report.decision != "PASS"
    unavailable = next(
        item for item in report.integrity.diagnostics if item.code == "JUDGE_CORRECTION_UNAVAILABLE"
    )
    assert unavailable.details["claim_ids"] == ["mean_all", "mean_hard", "worst_all"]

    # The delta-method reference beside it is diagnostic anyway, and now says why too.
    delta = next(
        item for item in mean_all.intervals if item.interval_id == "judge_corrected_validated_delta"
    )
    assert delta.warnings == corrected.warnings

    # The report states the restriction.
    markdown = (report.write(tmp_path / "report") / "report.md").read_text(encoding="utf-8")
    assert "so they certify only a weighted mean that does the same" in markdown


def test_a_comparison_over_part_of_a_cohort_is_a_judge_sensitivity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _analyze(tmp_path, monkeypatch, compare=True)

    comparison = next(item for item in report.comparisons if item.level is None)
    corrected = _corrected(comparison)
    assert corrected.eligibility == "diagnostic"
    assert corrected.role == "sensitivity"
    assert corrected.varied_assumption == "judge_error"
    assert corrected.simultaneous_coverage is None
    (warning,) = corrected.warnings
    # Each arm is corrected with its own cohort's labels, so each arm is checked.
    assert "candidate cohort 'candidate': the row covers only part" in warning
    assert "baseline cohort 'baseline': the row covers only part" in warning


def test_a_self_weighting_whole_cohort_mean_is_still_certified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Equal prompt weights and equal rollout counts give every rollout the same weight.
    report = _analyze(tmp_path, monkeypatch, weights="equal_prompt")

    mean_all = _risk(report, "all", "mean")
    corrected = _corrected(mean_all)
    assert corrected.eligibility == "certifying"
    assert corrected.role == "reference"
    assert corrected.simultaneous_coverage is not None
    assert corrected.warnings == []
    # With every rollout weighted equally the pooled labels do measure this row's
    # accuracy, and the corrected interval covers the true risk, 0.30 / 10.
    assert corrected.lower is not None and corrected.upper is not None
    assert corrected.lower <= 0.03 <= corrected.upper
    claim = next(item for item in report.claims if item.claim_id == "mean_all")
    assert claim.method == "validated_judge_misclassification_correction"

    # Part of the cohort, or a tail measure, is still only a sensitivity.
    assert _corrected(_risk(report, "hard", "mean")).eligibility == "diagnostic"
    assert _corrected(_risk(report, "all", "worst")).eligibility == "diagnostic"


def test_noisy_valid_test_refuses_a_row_its_labels_do_not_measure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _analyze(tmp_path, monkeypatch, claims=NOISY_VALID_CLAIM, sampling="simple_random")

    (claim,) = report.claims
    assert claim.method == "noisy_valid_certification_test"
    assert claim.status == "INCONCLUSIVE"
    (reason,) = claim.reasons
    assert reason.startswith("The noisy-but-valid test cannot certify this row: the row gives")
