"""`plan fixed-design --plan` plans every claim the way `analyze` decides it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from inspect_safety_stats import analyze
from inspect_safety_stats.claim_planning import PlannedClaim, UnplannedClaim, plan_claims
from inspect_safety_stats.cli import app
from inspect_safety_stats.config.loader import load_plan
from inspect_safety_stats.stats.planning import fixed_design_detectable_failure_rates
from tests.cli_output import plain

runner = CliRunner()

MANIFEST = (
    "prompt_id,question_id,task_id,risk_domain,expected_rollouts\n"
    "p1,q1,t1,cyber,20\n"
    "p2,q1,t1,cyber,10\n"
    "p3,q2,t1,cyber,20\n"
    "p4,q3,t2,bio,30\n"
)

PLAN = """
schema_version: "2.0"
analysis:
  id: planning
  title: Planning
  intent: confirmatory
  plan_frozen_at: 2025-01-01T00:00:00Z
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
    missing: block
    errors: block
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  rollout_key: [eval_id, id, epoch]
  weights: {strategy: nested_equal, hierarchy: [task_id, question_id, prompt_id]}
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
    question: {block_by: [question_id], comparison_coupling: joint}
cohorts:
  candidate:
    match: {model: model/candidate}
groups:
  - id: all
  - id: cyber
    filter: {risk_domain: cyber}
    weights: {strategy: equal_prompt}
  - id: bio
    filter: {risk_domain: bio}
outputs:
  prompt_bounds: {enabled: true, simultaneous: true, family: displayed}
  direct_mean_intervals:
    enabled: true
    method: weighted_hoeffding
    family: direct_means
    groups: [all, cyber]
    hierarchy_nodes: true
    independence: {primary: PRIMARY}
  risk_measures:
    - {id: mean, type: weighted_mean}
    - {id: worst, type: maximum}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.1}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - {id: all_mean, title: All, cohort: candidate, group: all, measure: mean, family: gate,
     acceptable_upper: 0.3, unacceptable_lower: 0.6}
  - {id: cyber_mean, title: Cyber, cohort: candidate, group: cyber, measure: mean,
     family: gate, acceptable_upper: 0.2, unacceptable_lower: 0.5}
  - {id: bio_mean, title: Bio, cohort: candidate, group: bio, measure: mean, family: gate,
     acceptable_upper: 0.3, unacceptable_lower: 0.6}
  - {id: all_worst, title: Worst, cohort: candidate, group: all, measure: worst, family: gate,
     acceptable_upper: 0.3, unacceptable_lower: 0.6}
"""


def _write(tmp_path: Path, *, primary: str = "rollout", plan: str = PLAN) -> Path:
    (tmp_path / "design.csv").write_text(MANIFEST, encoding="utf-8")
    path = tmp_path / "plan.yaml"
    path.write_text(plan.replace("PRIMARY", primary).strip() + "\n", encoding="utf-8")
    return path


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
                "epochs": 30,
            }
        ]
    )
    counts = {"p1": 20, "p2": 10, "p3": 20, "p4": 30}
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"{prompt_id}-{epoch}",
                "eval_id": "eval-1",
                "log": "/not/a/local/log.eval",
                "id": prompt_id,
                "epoch": epoch,
                "score_harmful": "FAIL" if epoch == 1 else "PASS",
                "error": None,
                "retries": 0,
            }
            for prompt_id, count in counts.items()
            for epoch in range(1, count + 1)
        ]
    )
    return evals, samples


@pytest.mark.parametrize("primary", ["rollout", "question"])
def test_planned_claims_use_the_alpha_and_design_analyze_uses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, primary: str
) -> None:
    """Endpoint alpha and effective sample size match the intervals analyze decides on."""
    plan_path = _write(tmp_path, primary=primary)
    evals, samples = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )
    report = analyze(plan=plan_path, logs="ignored")
    planned = {
        item.claim_id: item
        for item in plan_claims(load_plan(plan_path), power=0.9)
        if isinstance(item, PlannedClaim)
    }
    assert set(planned) == {"all_mean", "cyber_mean"}
    for item in planned.values():
        risk = next(
            result
            for result in report.group_risks
            if result.group_id == item.group and result.measure_id == "mean"
        )
        direct = next(interval for interval in risk.intervals if interval.role == "primary")
        assert item.alpha.endpoint_alpha == pytest.approx(direct.alpha_lower, rel=1e-12, abs=0)
        assert item.support is not None
        assert item.support.effective_sample_size == pytest.approx(
            direct.effective_sample_size, rel=1e-12, abs=0
        )
    # Two selected groups; nodes: all has 2 tasks and 3 questions, cyber 1 task and 2 questions.
    assert planned["all_mean"].alpha.target_count == 2 + 5 + 3
    assert planned["all_mean"].alpha.endpoint_alpha == pytest.approx(0.05 / 10, rel=1e-12, abs=0)


def test_direct_worst_of_k_intervals_count_in_the_direct_mean_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A direct worst_of_k interval on each selected group takes a share of the family."""
    with_worst = PLAN.replace(
        "    - {id: worst, type: maximum}\n",
        "    - {id: worst, type: maximum}\n"
        "    - {id: worst2, type: worst_of_k, k: 2, direct_interval: true}\n",
    ).replace(
        "claims:\n",
        "claims:\n"
        "  - {id: all_worst2, title: Worst of 2, cohort: candidate, group: all, measure: worst2,\n"
        "     family: gate, acceptable_upper: 0.3, unacceptable_lower: 0.6}\n",
    )
    assert with_worst.count("worst2") == 3
    plan_path = _write(tmp_path, plan=with_worst)
    evals, samples = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )
    report = analyze(plan=plan_path, logs="ignored")
    claims = {item.claim_id: item for item in plan_claims(load_plan(plan_path), power=0.9)}
    planned = claims["all_mean"]
    assert isinstance(planned, PlannedClaim)
    risk = next(
        result
        for result in report.group_risks
        if result.group_id == "all" and result.measure_id == "mean"
    )
    direct = next(interval for interval in risk.intervals if interval.role == "primary")
    # Two selected groups, each with a mean and a worst-of-2 interval, and eight nodes.
    assert planned.alpha.target_count == 2 * 2 + 8
    assert planned.alpha.endpoint_alpha == pytest.approx(direct.alpha_lower, rel=1e-12, abs=0)
    unplanned = claims["all_worst2"]
    assert isinstance(unplanned, UnplannedClaim)
    assert "worst_of_k with a direct interval" in unplanned.reason


def test_each_side_is_planned_at_its_own_threshold(tmp_path: Path) -> None:
    plan_path = _write(tmp_path)
    item = next(
        claim
        for claim in plan_claims(load_plan(plan_path), power=0.8)
        if isinstance(claim, PlannedClaim) and claim.claim_id == "cyber_mean"
    )
    assert item.prompts == 3
    assert item.total_rollouts == 50
    for threshold, rates in ((0.2, item.support), (0.5, item.rejection)):
        expected = fixed_design_detectable_failure_rates(
            [20, 10, 20],
            [1 / 3, 1 / 3, 1 / 3],
            threshold=threshold,
            support_alpha=item.alpha.endpoint_alpha,
            rejection_alpha=item.alpha.endpoint_alpha,
            power=0.8,
        )
        assert rates == expected


def test_rollouts_override_the_manifest_counts(tmp_path: Path) -> None:
    plan_path = _write(tmp_path)
    claims = plan_claims(load_plan(plan_path), power=0.9, rollouts=100)
    item = next(claim for claim in claims if isinstance(claim, PlannedClaim))
    assert item.total_rollouts == 100 * item.prompts


def test_claims_decided_otherwise_are_reported_with_the_reason(tmp_path: Path) -> None:
    plan_path = _write(tmp_path)
    unplanned = {
        item.claim_id: item.reason
        for item in plan_claims(load_plan(plan_path), power=0.9)
        if isinstance(item, UnplannedClaim)
    }
    assert set(unplanned) == {"bio_mean", "all_worst"}
    assert "group 'bio' has no direct-mean interval" in unplanned["bio_mean"]
    assert "'worst' is maximum" in unplanned["all_worst"]


def test_blocks_must_be_manifest_columns(tmp_path: Path) -> None:
    plan_path = _write(
        tmp_path,
        primary="judge_run",
        plan=PLAN.replace(
            "    question: {block_by",
            "    judge_run: {block_by: [eval_id], comparison_coupling: joint}\n"
            "    question: {block_by",
        ),
    )
    with pytest.raises(ValueError, match="absent from the design manifest"):
        plan_claims(load_plan(plan_path), power=0.9)


def test_cli_plans_every_claim_from_the_plan(tmp_path: Path) -> None:
    plan_path = _write(tmp_path)
    result = runner.invoke(app, ["plan", "fixed-design", "--plan", str(plan_path)])
    output = plain(result.output)
    assert result.exit_code == 0, output
    assert "Claims: 4" in output
    assert "Claim cyber_mean: cohort candidate, group cyber, measure mean" in output
    assert "Independence profile: rollout" in output
    assert "Bonferroni share of 0.05 across 10 targets and 1 cohorts" in output
    assert (
        "Interval method: rates guaranteed for weighted_hoeffding; analyze decides on "
        "weighted_hoeffding"
    ) in output
    assert "PASS at acceptable_upper 0.2: maximum certifiable failure rate" in output
    assert "FAIL at unacceptable_lower 0.5: minimum detectable failure rate" in output
    assert "Claim all_worst: not planned:" in output


def test_cli_refuses_options_the_plan_supplies(tmp_path: Path) -> None:
    plan_path = _write(tmp_path)
    for option, value in (
        ("--threshold", "0.5"),
        ("--alpha", "0.05"),
        ("--hierarchy", "task_id,prompt_id"),
        ("--family-size", "1"),
    ):
        result = runner.invoke(
            app, ["plan", "fixed-design", "--plan", str(plan_path), option, value]
        )
        assert result.exit_code != 0
        assert f"{option} cannot be combined with --plan" in plain(result.output)


def test_cli_without_a_plan_still_needs_manifest_and_threshold(tmp_path: Path) -> None:
    (tmp_path / "design.csv").write_text(MANIFEST, encoding="utf-8")
    result = runner.invoke(
        app, ["plan", "fixed-design", "--manifest", str(tmp_path / "design.csv")]
    )
    assert result.exit_code != 0
    assert "give --manifest and --threshold, or --plan" in plain(result.output)


def test_cli_fails_when_no_claim_can_be_planned(tmp_path: Path) -> None:
    disabled = PLAN.replace(
        "direct_mean_intervals:\n    enabled: true", "direct_mean_intervals:\n    enabled: false"
    )
    assert disabled != PLAN
    plan_path = _write(tmp_path, plan=disabled)
    result = runner.invoke(app, ["plan", "fixed-design", "--plan", str(plan_path)])
    assert result.exit_code == 1
    assert "outputs.direct_mean_intervals is disabled" in plain(result.output)
    assert "No claim of this plan could be planned." in plain(result.output)


def test_sampled_designs_are_refused(tmp_path: Path) -> None:
    plan_path = _write(
        tmp_path,
        plan=PLAN.replace(
            "  type: fixed\n",
            "  type: sampled\n  inclusion_probability_column: pi\n",
        ),
    )
    with pytest.raises(ValueError, match="sampled design"):
        plan_claims(load_plan(plan_path), power=0.9)
