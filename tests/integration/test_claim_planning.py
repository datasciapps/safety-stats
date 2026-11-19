"""`plan fixed-design --plan` plans every claim the way `analyze` decides it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from inspect_safety_stats import analyze
from inspect_safety_stats.claim_planning import (
    PlannedClaim,
    UnplannedClaim,
    plan_claims,
    plan_design,
)
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


def _frames(*, all_safe: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                "score_harmful": "FAIL" if epoch == 1 and not all_safe else "PASS",
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
        assert item.planned.effective_sample_size == pytest.approx(
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
    for threshold, rates in ((0.2, item.hoeffding_support), (0.5, item.hoeffding_rejection)):
        expected = fixed_design_detectable_failure_rates(
            [20, 10, 20],
            [1 / 3, 1 / 3, 1 / 3],
            threshold=threshold,
            support_alpha=item.alpha.endpoint_alpha,
            rejection_alpha=item.alpha.endpoint_alpha,
            power=0.8,
        )
        assert rates == expected


def _patch_logs(monkeypatch: pytest.MonkeyPatch, *, all_safe: bool = False) -> None:
    evals, samples = _frames(all_safe=all_safe)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )


def _parity_plan(calibration: str, method: str) -> str:
    reference = "weighted_empirical_bernstein" if method == "weighted_hoeffding" else (
        "weighted_hoeffding"
    )
    plan = PLAN.replace(
        "    method: weighted_hoeffding\n",
        f"    method: {method}\n    reference_methods: [{reference}]\n",
    ).replace(
        "    hierarchy_nodes: true\n    independence: {primary: PRIMARY}\n",
        f"    hierarchy_nodes: true\n    hierarchy_calibration: {calibration}\n"
        "    independence: {primary: rollout, sensitivities: [question]}\n",
    )
    assert plan.count(calibration) == 1 and plan.count("sensitivities") == 1
    return plan


@pytest.mark.parametrize("calibration", ["flat_bonferroni", "frontier_sidak"])
@pytest.mark.parametrize(
    "method",
    [
        "weighted_hoeffding",
        "weighted_empirical_bernstein",
        "weighted_kl_chernoff",
        "weighted_betting",
    ],
)
def test_planned_intervals_match_analyze_for_every_profile_and_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calibration: str, method: str
) -> None:
    """#9: for each profile and node, the planning alpha equals the alpha analyze assigns.

    Blocks, effective sample size and calibration match too. On logs where every rollout is
    safe the planned certification floor is the upper bound analyze reports, after its
    hierarchy-region projection, and the raw floor the interval's own bound before it.
    """
    plan_path = _write(tmp_path, plan=_parity_plan(calibration, method))
    _patch_logs(monkeypatch, all_safe=True)
    report = analyze(plan=plan_path, logs="ignored")
    reported = {}
    for result in [
        *(risk for risk in report.group_risks if risk.measure_id == "mean"),
        *report.hierarchy_risks,
    ]:
        level = getattr(result, "level", None)
        path = tuple(getattr(result, "path", {}).values())
        for interval in result.intervals:
            if interval.method == method and interval.role in {"primary", "sensitivity"}:
                profile = interval.independence_profile
                reported[(profile, result.cohort_id, result.group_id, level, path)] = interval
    design = plan_design(load_plan(plan_path), power=0.9)
    planned = {
        (
            item.independence_profile,
            item.cohort,
            item.group,
            item.level,
            tuple(item.path.values()),
        ): item
        for item in design.intervals
    }
    assert design.method == method
    # Two groups (all, cyber) with 2 + 3 and 1 + 2 nodes, under two profiles.
    assert len(planned) == 2 * (2 + 5 + 3)
    assert set(planned) == set(reported)
    # The floor is the bound analyze reports, after its hierarchy-region projection.
    assert {key: item.certification_floor for key, item in planned.items()} == pytest.approx(
        {key: reported[key].upper for key in planned}, rel=1e-9, abs=1e-15
    )
    for key, item in planned.items():
        interval = reported[key]
        assert item.alpha.endpoint_alpha == pytest.approx(interval.alpha_lower, rel=1e-12, abs=0)
        assert item.alpha.calibration == interval.calibration_method
        assert item.alpha.frontier_id == interval.calibration_frontier
        assert item.independent_blocks == interval.independent_block_count
        assert item.effective_sample_size == pytest.approx(
            interval.effective_sample_size, rel=1e-12, abs=0
        )
        unprojected = interval.raw_upper if interval.raw_upper is not None else interval.upper
        assert item.raw_certification_floor == pytest.approx(unprojected, rel=1e-9, abs=1e-15)
        assert item.resolves == (
            None if item.acceptable_upper is None else interval.upper <= item.acceptable_upper
        )
    if calibration == "frontier_sidak" or method in {"weighted_kl_chernoff", "weighted_betting"}:
        # The projection lowers some of these floors, so the parity covers it.
        assert any(
            item.certification_floor < item.raw_certification_floor - 1e-6
            for item in planned.values()
        )
    if calibration == "frontier_sidak":
        assert {item.alpha.calibration for item in planned.values()} == {"frontier_sidak"}
    decided = {claim.claim_id: claim for claim in report.claims}
    for claim in design.claims:
        if not isinstance(claim, PlannedClaim):
            continue
        result = decided[claim.claim_id]
        assert result.upper is not None
        assert claim.certification_floor == pytest.approx(result.upper, rel=1e-9, abs=1e-15)
        assert (result.status == "PASS") == (claim.certification_floor <= claim.acceptable_upper)


def test_a_floor_the_projection_lowers_below_the_threshold_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Questions as blocks leave the root interval at 1 when every rollout is safe.

    analyze projects it onto the hierarchy region, to 0.96, and PASSes a claim at 0.97. The
    planner reports that floor, and says that its rates are planned before the projection.
    """
    plan = (
        _parity_plan("flat_bonferroni", "weighted_kl_chernoff")
        .replace(
            "primary: rollout, sensitivities: [question]",
            "primary: question, sensitivities: [rollout]",
        )
        .replace(
            "group: all, measure: mean, family: gate,\n"
            "     acceptable_upper: 0.3, unacceptable_lower: 0.6}",
            "group: all, measure: mean, family: gate,\n"
            "     acceptable_upper: 0.97, unacceptable_lower: 0.99}",
        )
    )
    assert plan.count("primary: question") == 1 and plan.count("acceptable_upper: 0.97") == 1
    plan_path = _write(tmp_path, plan=plan)
    _patch_logs(monkeypatch, all_safe=True)
    result = next(claim for claim in analyze(plan=plan_path, logs="ignored").claims)
    assert result.claim_id == "all_mean" and result.status == "PASS"
    item = next(
        claim
        for claim in plan_claims(load_plan(plan_path), power=0.9)
        if isinstance(claim, PlannedClaim) and claim.claim_id == "all_mean"
    )
    assert item.raw_certification_floor == 1.0
    assert item.certification_floor == pytest.approx(result.upper, rel=1e-9, abs=0)
    assert item.certification_floor < 0.97
    # Planned on the raw interval, whose floor is above the threshold.
    assert item.planned.maximum_certifiable_failure_rate is None
    output = plain(runner.invoke(app, ["plan", "fixed-design", "--plan", str(plan_path)]).output)
    assert (
        f"Certification floor: {item.certification_floor:.8g}, the upper bound analyze reports "
        "when every rollout is safe (1 before the hierarchy-region projection; the rates above "
        "are planned before it, and the projection can only raise their power)"
    ) in output


def test_planned_claims_are_decided_on_the_plan_method(tmp_path: Path) -> None:
    plan_path = _write(tmp_path, plan=_parity_plan("flat_bonferroni", "weighted_kl_chernoff"))
    item = next(
        claim
        for claim in plan_claims(load_plan(plan_path), power=0.9)
        if isinstance(claim, PlannedClaim) and claim.claim_id == "all_mean"
    )
    assert item.planned.method == "weighted_kl_chernoff"
    assert [reference.method for reference in item.references] == ["weighted_hoeffding"]
    # The rollouts carry unequal weights (nested_equal over 20, 10, 20 and 30 rollouts). At a
    # low failure rate the KL-Chernoff interval certifies where Hoeffding cannot; at a high one
    # its Poisson kernel is wider than Hoeffding's. The planner reports both sides as they are
    # rather than assume the analysis method is the sharper one.
    support, rejection = item.hoeffding_support, item.hoeffding_rejection
    assert support is not None and rejection is not None
    assert support.maximum_certifiable_failure_rate is None
    assert item.planned.maximum_certifiable_failure_rate is not None
    assert rejection.minimum_detectable_failure_rate is not None
    assert item.planned.minimum_detectable_failure_rate is not None
    assert item.planned.minimum_detectable_failure_rate > rejection.minimum_detectable_failure_rate


def test_claims_under_a_method_the_planner_cannot_invert_are_not_planned(tmp_path: Path) -> None:
    """weighted_betting: no rates, but the claims say why and the floors are still exact."""
    plan_path = _write(tmp_path, plan=_parity_plan("flat_bonferroni", "weighted_betting"))
    design = plan_design(load_plan(plan_path), power=0.9)
    reasons = {
        item.claim_id: item.reason for item in design.claims if isinstance(item, UnplannedClaim)
    }
    assert "cannot invert" in reasons["all_mean"]
    assert "cannot invert" in reasons["cyber_mean"]
    assert design.intervals and all(0 < item.certification_floor < 1 for item in design.intervals)
    result = runner.invoke(app, ["plan", "fixed-design", "--plan", str(plan_path)])
    output = plain(result.output)
    assert result.exit_code == 1, output
    assert "Claim all_mean: not planned: decided on the weighted_betting interval" in output
    assert "No claim of this plan could be planned." in output


def test_a_reference_method_the_planner_cannot_invert_is_named(tmp_path: Path) -> None:
    plan = _parity_plan("flat_bonferroni", "weighted_kl_chernoff").replace(
        "reference_methods: [weighted_hoeffding]",
        "reference_methods: [weighted_hoeffding, weighted_betting]",
    )
    assert plan.count("weighted_betting") == 1
    plan_path = _write(tmp_path, plan=plan)
    item = next(
        claim
        for claim in plan_claims(load_plan(plan_path), power=0.9)
        if isinstance(claim, PlannedClaim) and claim.claim_id == "all_mean"
    )
    assert [reference.method for reference in item.references] == ["weighted_hoeffding"]
    assert item.unplanned_references == ("weighted_betting",)
    result = runner.invoke(app, ["plan", "fixed-design", "--plan", str(plan_path)])
    output = plain(result.output)
    assert result.exit_code == 0, output
    assert (
        "Reference method weighted_betting: not planned; the planner cannot invert its interval"
    ) in output


def test_a_level_the_declared_rollouts_cannot_resolve_is_reported(tmp_path: Path) -> None:
    """Like the generative budget table: cyber's questions are too thin for its threshold."""
    plan_path = _write(tmp_path, plan=_parity_plan("flat_bonferroni", "weighted_kl_chernoff"))
    design = plan_design(load_plan(plan_path), power=0.9)
    by_target = {
        (item.independence_profile, item.group, item.level, tuple(item.path.values())): item
        for item in design.intervals
    }
    root = by_target[("rollout", "cyber", None, ())]
    question = by_target[("rollout", "cyber", "question_id", ("t1", "q2"))]
    assert root.acceptable_upper == question.acceptable_upper == 0.2
    assert root.resolves is True
    assert question.resolves is False
    # bio has no direct-mean interval, so it has no row; a group without a claim would have
    # no threshold to resolve against.
    assert not any(item.group == "bio" for item in design.intervals)
    result = runner.invoke(app, ["plan", "fixed-design", "--plan", str(plan_path)])
    output = plain(result.output)
    assert result.exit_code == 0, output
    assert "Levels the declared rollouts cannot resolve:" in output
    assert "level question_id of group cyber in cohort candidate under profile rollout" in output


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
    with pytest.raises(ValueError, match="the logs supply and the design manifest does not"):
        plan_claims(load_plan(plan_path), power=0.9)


@pytest.mark.parametrize("field", ["run_id", "model", "eval_id", "repeat"])
def test_a_manifest_column_does_not_stand_in_for_a_log_field(tmp_path: Path, field: str) -> None:
    """analyze blocks by the rollouts' own field, so a manifest column of that name is unused.

    Planning by the column would give every rollout the planner's one synthetic value instead:
    a single block, whatever the column says.
    """
    plan_path = _write(
        tmp_path,
        primary="by_log_field",
        plan=PLAN.replace(
            "    question: {block_by",
            f"    by_log_field: {{block_by: [{field}], comparison_coupling: joint}}\n"
            "    question: {block_by",
        ),
    )
    header, *rows = MANIFEST.strip().split("\n")
    lines = [f"{header},{field}", *(f"{row},{index}" for index, row in enumerate(rows))]
    (tmp_path / "design.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"blocks by '{field}', which analyze reads from the"):
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
        "Interval method: weighted_hoeffding, as outputs.direct_mean_intervals declares"
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
