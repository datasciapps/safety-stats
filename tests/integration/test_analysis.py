from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan, render_visualization, write_visualization
from inspect_safety_stats.api import _alpha_by_prompt, _prompt_blocked
from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.config.models import CohortSpec, PromptBoundsSpec, RiskMeasureSpec
from inspect_safety_stats.ingest.inspect_logs import ErrorContext, NormalizedData
from inspect_safety_stats.ingest.manifest import load_manifest
from inspect_safety_stats.schema.reports import Diagnostic, RiskIntervalResult, SafetyReport


def _write_inputs(tmp_path: Path) -> Path:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,risk_domain,expected_rollouts,zero_tolerance\n"
        "p1,q1,t1,cyber,20,false\n"
        "p2,q2,t1,cyber,20,false\n",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        """
schema_version: "2.0"
analysis:
  id: integration
  title: Integration safety analysis
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
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  rollout_key: [eval_id, id, epoch]
  require_complete_manifest: true
  require_expected_rollouts: true
  weights:
    strategy: nested_equal
    hierarchy: [task_id, question_id, prompt_id]
  independence_profiles:
    rollout:
      block_by: [rollout]
      comparison_coupling: independent
    prompt:
      block_by: [prompt_id]
      comparison_coupling: joint
cohorts:
  candidate:
    match:
      model: model/candidate
    require_uniform: [task_version, model_generate_config]
groups:
  - id: cyber
    filter: {risk_domain: cyber}
    weights:
      strategy: nested_equal
      hierarchy: [task_id, question_id, prompt_id]
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
    - {id: mean, type: weighted_mean}
    - {id: cvar, type: cvar, beta: 0.5}
    - {id: entropic, type: entropic, lambda: 10}
    - {id: worst, type: maximum}
    - {id: worst_of_2, type: worst_of_k, k: 2}
    - {id: excess, type: threshold_excess, threshold: 0.3}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - id: mean_safe
    title: Mean fixed-design risk is acceptable
    cohort: candidate
    group: cyber
    measure: mean
    family: gate
    acceptable_upper: 0.30
    unacceptable_lower: 0.60
    mandatory: true
policy:
  zero_tolerance: {enabled: true, manifest_field: zero_tolerance}
report:
  include_observations: true
  decimal_places: 6
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return plan


def _frames(*, missing_score: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    rows = []
    for prompt_id in ("p1", "p2"):
        for epoch in range(1, 21):
            rows.append(
                {
                    "sample_id": f"{prompt_id}-{epoch}",
                    "eval_id": "eval-1",
                    "log": "/not/a/local/log.eval",
                    "id": prompt_id,
                    "epoch": epoch,
                    "score_harmful": None
                    if missing_score and prompt_id == "p1" and epoch == 1
                    else "PASS",
                    "error": None,
                    "retries": 0,
                }
            )
    return evals, pd.DataFrame(rows)


def test_end_to_end_pass_and_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_path = _write_inputs(tmp_path)
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan_path, logs="ignored")
    assert report.decision == "PASS"
    assert report.integrity.status == "valid"
    assert len(report.prompt_estimates) == 2
    assert {result.measure_id for result in report.group_risks} == {
        "mean",
        "cvar",
        "entropic",
        "worst",
        "worst_of_2",
        "excess",
    }
    assert report.presentation == {"include_decisions": False}
    risk = next(item for item in report.group_risks if item.measure_id == "mean")
    assert risk.coverage_mode == "two_sided"
    assert risk.family_alpha == pytest.approx(0.05)
    assert risk.directional_family_alpha == pytest.approx(0.025)
    assert risk.simultaneous_coverage == pytest.approx(0.95)
    assert risk.lower_one_sided_coverage == pytest.approx(0.975)
    assert risk.upper_one_sided_coverage == pytest.approx(0.975)
    assert risk.primary_interval_id == "direct_weighted_hoeffding"
    assert risk.interval_method == "weighted_hoeffding"
    assert len(risk.intervals) == 2
    direct = next(item for item in risk.intervals if item.method == "weighted_hoeffding")
    propagated = next(
        item for item in risk.intervals if item.method == "propagated_exact_prompt_bounds"
    )
    assert direct.role == "primary"
    assert direct.alpha_lower == pytest.approx(0.025)
    assert direct.simultaneous_coverage == pytest.approx(0.95)
    assert direct.effective_sample_size == pytest.approx(40)
    assert propagated.role == "reference"
    output = report.write(tmp_path / "report")
    assert (output / "report.json").exists()
    assert (output / "report.md").exists()
    assert (output / "report.html").exists()
    assert (output / "observations.csv").exists()
    assert (output / "comparisons.csv").exists()
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "fixed evaluation set" in markdown
    assert "**Decision:**" not in markdown
    assert "## Mandatory claims" not in markdown
    assert "## Policy events" not in markdown
    assert "95.0% joint; 97.5% each direction" in markdown
    assert "finite-sample confidence bounds" in markdown
    assert "## Interval-method comparison" in markdown
    assert "weighted_hoeffding" in markdown
    html = (output / "report.html").read_text(encoding="utf-8")
    assert '"presentation":{"include_decisions":false}' in html
    second = report.write(tmp_path / "report-copy")
    assert (output / "report.json").read_bytes() == (second / "report.json").read_bytes()
    assert (output / "report.html").read_bytes() == (second / "report.html").read_bytes()
    assert "report.html" in (output / "checksums.json").read_text(encoding="utf-8")
    assert "task_id,question_id,prompt_id" in (output / "design.resolved.csv").read_text(
        encoding="utf-8"
    )
    standalone = write_visualization(output / "report.json", tmp_path / "standalone.html")
    assert standalone.exists()
    assert "__INSPECT_SAFETY_STATS_DATA__" not in standalone.read_text(encoding="utf-8")
    report._resolved_plan["report"]["include_observations"] = False
    report.write(output, overwrite=True)
    assert not (output / "observations.csv").exists()
    assert "observations.csv" not in (output / "checksums.json").read_text(encoding="utf-8")


def test_text_artifacts_have_the_same_bytes_on_every_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Platform newline translation would make report bytes, and so checksums, differ by OS."""
    plan_path = _write_inputs(tmp_path)
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    output = analyze(plan=plan_path, logs="ignored").write(tmp_path / "report")
    standalone = write_visualization(output, tmp_path / "standalone.html")
    for path in (
        output / "report.json",
        output / "report.md",
        output / "report.html",
        output / "plan.resolved.yaml",
        output / "checksums.json",
        standalone,
    ):
        assert b"\r" not in path.read_bytes(), path.name
    # CSV sidecars keep the csv module's CRLF row terminator, which is the same everywhere.
    for path in sorted(output.glob("*.csv")):
        data = path.read_bytes()
        if data:
            assert data.split(b"\n", 1)[0].endswith(b"\r"), path.name


def test_empirical_hierarchy_intervals_and_independence_sensitivity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = load_plan(_write_inputs(tmp_path))
    spec = plan.outputs.direct_mean_intervals
    spec.method = "weighted_empirical_bernstein"
    spec.reference_methods = ["weighted_hoeffding"]
    spec.hierarchy_nodes = True
    assert spec.independence is not None
    spec.independence.sensitivities = ["prompt"]
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=plan, logs="ignored")

    assert {(row.level, tuple(row.path.values())) for row in report.hierarchy_risks} == {
        ("task_id", ("t1",)),
        ("question_id", ("t1", "q1")),
        ("question_id", ("t1", "q2")),
    }
    # One group root plus the three hierarchy nodes share each directional
    # Bonferroni family budget: 0.05 / 2 / 4.
    expected_endpoint_alpha = 0.00625
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    assert mean.interval_method == "weighted_empirical_bernstein"
    assert mean.intervals[0].alpha_upper == pytest.approx(expected_endpoint_alpha)
    assert len(mean.intervals) == 5  # Direct primary/reference, sensitivities, and propagation.
    for node in report.hierarchy_risks:
        assert node.interval_method == "weighted_empirical_bernstein"
        assert len(node.intervals) == 4  # Two methods x two independence profiles.
        assert {interval.role for interval in node.intervals} == {
            "primary",
            "reference",
            "sensitivity",
        }
        assert all(
            interval.alpha_upper == pytest.approx(expected_endpoint_alpha)
            for interval in node.intervals
        )
        sensitivity = [interval for interval in node.intervals if interval.role == "sensitivity"]
        assert {interval.independence_profile for interval in sensitivity} == {"prompt"}

    output = report.write(tmp_path / "hierarchy-report")
    hierarchy_csv = (output / "hierarchy-risks.csv").read_text(encoding="utf-8")
    assert "weighted_empirical_bernstein" in hierarchy_csv
    assert "prompt" in hierarchy_csv
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "Configuration risk and comparison hierarchy" in html
    assert "renderHierarchyRisks" in html
    assert "tree-axis" in html
    assert "Raw interval before hierarchy projection" in html


def test_weighted_kl_intervals_are_reported_at_root_and_hierarchy_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = load_plan(_write_inputs(tmp_path))
    spec = plan.outputs.direct_mean_intervals
    spec.method = "weighted_kl_chernoff"
    spec.reference_methods = ["weighted_empirical_bernstein"]
    spec.hierarchy_nodes = True
    assert spec.independence is not None
    spec.independence.sensitivities = ["prompt"]
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=plan, logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    assert mean.interval_method == "weighted_kl_chernoff"
    assert mean.primary_interval_id == "direct_weighted_kl_chernoff"
    primary = next(interval for interval in mean.intervals if interval.role == "primary")
    assert primary.chernoff_kernel is not None
    assert primary.projection_method == "hierarchy_linear_program"
    assert primary.raw_lower is not None
    assert primary.raw_upper is not None
    assert primary.lower >= primary.raw_lower
    assert primary.upper <= primary.raw_upper
    assert any("Chernoff" in assumption for assumption in primary.assumptions)
    for node in report.hierarchy_risks:
        assert node.interval_method == "weighted_kl_chernoff"
        assert node.primary_interval_id == "direct_weighted_kl_chernoff"
        assert any(
            interval.method == "weighted_kl_chernoff" and interval.independence_profile == "prompt"
            for interval in node.intervals
        )
        node_primary = next(interval for interval in node.intervals if interval.role == "primary")
        assert node_primary.projection_method == "hierarchy_linear_program"


def test_projected_intervals_report_the_simultaneous_floor_in_each_direction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A projected endpoint covers only on the joint event.

    Its optimum depends on the raw endpoints of both directions, so a raw lower endpoint that
    misses can pull a projected upper endpoint below the truth. In each direction only the
    95% simultaneous floor holds, not the 97.5% of a raw endpoint.
    """
    plan = load_plan(_write_inputs(tmp_path))
    spec = plan.outputs.direct_mean_intervals
    spec.method = "weighted_kl_chernoff"
    spec.reference_methods = ["weighted_empirical_bernstein"]
    spec.hierarchy_nodes = True
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=plan, logs="ignored")

    rows = [*report.group_risks, *report.hierarchy_risks]
    intervals = [interval for row in rows for interval in row.intervals]
    projected = [interval for interval in intervals if interval.projection_method is not None]
    assert projected
    for interval in projected:
        assert interval.simultaneous_coverage == pytest.approx(0.95)
        assert interval.lower_one_sided_coverage == interval.simultaneous_coverage
        assert interval.upper_one_sided_coverage == interval.simultaneous_coverage
    for row in rows:
        if row.projection_method is not None:
            assert row.lower_one_sided_coverage == row.simultaneous_coverage
            assert row.upper_one_sided_coverage == row.simultaneous_coverage
    # Endpoints that were not projected keep their one-sided level.
    raw = [
        interval
        for interval in intervals
        if interval.projection_method is None and interval.upper_one_sided_coverage is not None
    ]
    assert raw
    assert all(interval.upper_one_sided_coverage == pytest.approx(0.975) for interval in raw)


def test_frontier_sidak_calibrates_disjoint_hierarchy_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = load_plan(_write_inputs(tmp_path))
    spec = plan.outputs.direct_mean_intervals
    spec.method = "weighted_kl_chernoff"
    spec.hierarchy_nodes = True
    spec.hierarchy_calibration = "frontier_sidak"
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=plan, logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    root = next(interval for interval in mean.intervals if interval.role == "primary")
    task = next(
        interval
        for node in report.hierarchy_risks
        if node.level == "task_id"
        for interval in node.intervals
        if interval.role == "primary"
    )
    questions = [
        next(interval for interval in node.intervals if interval.role == "primary")
        for node in report.hierarchy_risks
        if node.level == "question_id"
    ]

    # Root and the single task are the same statistic, so they are aliases in one
    # frontier. The two disjoint question nodes form the second frontier.
    assert root.calibration_method == "frontier_sidak"
    assert root.calibration_frontier == task.calibration_frontier
    assert root.alpha_upper == pytest.approx(0.0125)
    expected_question_alpha = 1 - (1 - 0.0125) ** 0.5
    assert {interval.calibration_frontier_size for interval in questions} == {2}
    assert all(
        interval.alpha_upper == pytest.approx(expected_question_alpha) for interval in questions
    )
    assert all("frontier_sidak" in str(interval.coverage_scope) for interval in [root, *questions])
    output = report.write(tmp_path / "frontier-report")
    hierarchy_csv = (output / "hierarchy-risks.csv").read_text(encoding="utf-8")
    assert "calibration_frontier_alpha" in hierarchy_csv
    assert "frontier_sidak" in hierarchy_csv
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "Frontier Šidák" in html


def test_an_all_failure_group_bound_stays_a_probability_under_judge_correction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nine weights of 1/9 sum to 1.0000000000000002, and the judge correction raised on it."""
    plan_path = _write_inputs(tmp_path)
    prompts = [f"p{index}" for index in range(1, 10)]
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,risk_domain,expected_rollouts,zero_tolerance\n"
        + "".join(f"{prompt},q{prompt},t1,cyber,20,false\n" for prompt in prompts),
        encoding="utf-8",
    )
    plan_text = plan_path.read_text(encoding="utf-8").replace(
        "design:\n",
        "  judge_calibration:\n"
        "    mode: assumed\n"
        "    primary: declared\n"
        "    scenarios:\n"
        "      - id: declared\n"
        "        rho_1: {lower: 0.9, upper: 1.0}\n"
        "        rho_0: {lower: 0.0, upper: 0.01}\n"
        "design:\n",
        1,
    )
    plan_path.write_text(plan_text, encoding="utf-8")
    plan = load_plan(plan_path)
    # The propagated envelope is then the mean's primary interval, and it was not clipped.
    plan.outputs.direct_mean_intervals.enabled = False
    eval_frame, sample_frame = _frames()
    template = sample_frame[sample_frame["id"] == "p1"]
    sample_frame = pd.concat(
        [
            template.assign(id=prompt, sample_id=template["sample_id"].str.replace("p1", prompt))
            for prompt in prompts
        ],
        ignore_index=True,
    ).assign(score_harmful="FAIL")
    _patch_frames(monkeypatch, (eval_frame, sample_frame))

    report = analyze(plan=plan, logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    assert mean.upper == 1.0
    assert any(item.interval_id == "judge_corrected_declared" for item in mean.intervals)


def test_every_measure_is_classified_for_clipping_to_the_unit_interval() -> None:
    """A new measure must say whether it is a probability, and so whether it is clipped."""
    from typing import get_args

    from inspect_safety_stats.api import _PROBABILITY_MEASURES
    from inspect_safety_stats.config.models import RiskMeasureSpec

    types = set(get_args(RiskMeasureSpec.model_fields["type"].annotation))
    assert types - _PROBABILITY_MEASURES == {"threshold_excess"}
    assert _PROBABILITY_MEASURES <= types


def test_judge_calibration_emits_assumed_correction_sensitivities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "design:\n",
        "  judge_calibration:\n"
        "    mode: assumed\n"
        "    primary: declared\n"
        "    scenarios:\n"
        "      - id: declared\n"
        "        rho_1: {lower: 0.9, upper: 1.0}\n"
        "        rho_0: {lower: 0.0, upper: 0.01}\n"
        "      - id: pessimistic\n"
        "        rho_1: {lower: 0.7, upper: 1.0}\n"
        "        rho_0: {lower: 0.0, upper: 0.05}\n"
        "design:\n",
        1,
    )
    plan_path.write_text(plan_text, encoding="utf-8")
    plan = load_plan(plan_path)
    plan.outputs.direct_mean_intervals.hierarchy_nodes = True
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=plan, logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    corrected = {
        interval.interval_id: interval
        for interval in mean.intervals
        if interval.interval_id.startswith("judge_corrected_")
    }
    assert set(corrected) == {"judge_corrected_declared", "judge_corrected_pessimistic"}

    declared = corrected["judge_corrected_declared"]
    # An assumption can never certify.
    assert declared.eligibility == "diagnostic"
    assert declared.role == "sensitivity"
    # The uncorrected bounds are retained for comparison, and the primary is untouched.
    assert declared.raw_lower == pytest.approx(mean.lower)
    assert declared.raw_upper == pytest.approx(mean.upper)
    assert mean.primary_interval_id != declared.interval_id
    # A looser assumption box must produce a superset true-risk band.
    pessimistic = corrected["judge_corrected_pessimistic"]
    assert pessimistic.upper > declared.upper
    assert pessimistic.lower <= declared.lower
    assert pessimistic.upper - pessimistic.lower > declared.upper - declared.lower
    assert any("declared assumptions, not measurements" in a for a in declared.assumptions)

    # Hierarchy nodes are corrected too; entropic and threshold measures are not.
    assert report.hierarchy_risks
    for node in report.hierarchy_risks:
        assert any(i.interval_id == "judge_corrected_declared" for i in node.intervals)
    entropic = next(row for row in report.group_risks if row.measure_id == "entropic")
    assert not any(i.interval_id.startswith("judge_corrected_") for i in entropic.intervals)

    markdown = (report.write(tmp_path / "judge-report") / "report.md").read_text(encoding="utf-8")
    assert "## Judge misclassification sensitivity" in markdown
    assert "assumes the judge never errs" in markdown


def test_validated_judge_calibration_produces_certifying_corrections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "design:\n",
        "  judge_calibration:\n"
        "    mode: validated\n"
        "    validation_manifest: labels.csv\n"
        "    family: judge_error\n"
        "design:\n",
        1,
    )
    plan_text = plan_text.replace(
        "    - {id: displayed,",
        "    - {id: judge_error, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: displayed,",
        1,
    )
    plan_path.write_text(plan_text, encoding="utf-8")

    # The shared fixture reports no failures, so both judge strata must be populated
    # before a calibration can be estimated at all.
    eval_frame, sample_frame = _frames()
    sample_frame = sample_frame.copy()
    failing = (sample_frame["id"] == "p1") & (sample_frame["epoch"] <= 10)
    sample_frame.loc[failing, "score_harmful"] = "FAIL"
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    # Label every classified observation, agreeing with the judge throughout, so the
    # calibration is as sharp as this fixture allows.
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_plan = load_plan(_write_inputs(baseline_dir))
    baseline = analyze(plan=baseline_plan, logs="ignored")
    rows = ["observation_id,human_label"]
    for observation in baseline._observations:
        if observation.failure is None:
            continue
        rows.append(f"{observation.observation_id},{'unsafe' if observation.failure else 'safe'}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    plan = load_plan(plan_path)
    plan.outputs.direct_mean_intervals.hierarchy_nodes = True
    report = analyze(plan=plan, logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    corrected = next(
        interval
        for interval in mean.intervals
        if interval.interval_id == "judge_corrected_validated"
    )
    # Measured accuracies produce evidence, not a sensitivity.
    assert corrected.eligibility == "certifying"
    assert corrected.role == "reference"
    assert corrected.method == "validated_judge_misclassification_correction"
    assert corrected.calibration_method == "validated_judge_calibration"
    # Coverage is the union bound over the base interval and the calibration budget.
    base = next(item for item in mean.intervals if item.role == "primary")
    assert base.family_alpha is not None
    assert corrected.simultaneous_coverage == pytest.approx(1 - base.family_alpha - 0.05)
    assert corrected.raw_lower == pytest.approx(mean.lower)

    # A node spanning the whole cohort is the same estimand, so it stays certified. A node
    # inside it weights rollouts differently from the labels, so it is only a sensitivity.
    nodes = {
        tuple(row.path.values()): row
        for row in report.hierarchy_risks
        if row.cohort_id == "candidate"
    }
    for path, eligibility in [(("t1",), "certifying"), (("t1", "q1"), "diagnostic")]:
        node_correction = next(
            item
            for item in nodes[path].intervals
            if item.interval_id == "judge_corrected_validated"
        )
        assert node_correction.eligibility == eligibility

    # Provenance records the labels behind each cohort's calibration.
    summary = report.provenance["judge_calibration"]
    assert summary["mode"] == "validated"
    assert summary["sampling"] == "simple_random_within_judge_stratum"
    strata = summary["cohorts"]["candidate"]
    assert strata["judge_safe"]["labelled"] == strata["judge_safe"]["stratum_size"]

    # Threshold claims are decided on true risk, and say so.
    codes = {item.code for item in report.integrity.diagnostics}
    assert "JUDGE_CORRECTED_CLAIMS" in codes
    for result in report.claims:
        assert result.method == "validated_judge_misclassification_correction"
        assert "interval:judge_corrected_validated" in result.evidence_refs
        assert result.lower == pytest.approx(corrected.lower)
        assert result.upper == pytest.approx(corrected.upper)
        # The error budget is the union of the claim family and the calibration.
        assert result.alpha == pytest.approx(0.05 + 0.05)
        # A calibration box gives a range of corrected estimates, not a single one.
        assert result.estimate is None
        assert any("validated judge correction" in reason for reason in result.reasons)

    markdown = (report.write(tmp_path / "validated-report") / "report.md").read_text(
        encoding="utf-8"
    )
    assert "## Judge misclassification correction" in markdown
    assert "Human labels behind each cohort's calibration" in markdown


def _analyze_with_judge_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: Callable[[bool | None], str | None],
) -> SafetyReport:
    """Analyse p1 failing in 10 of its 20 epochs, with a validated judge calibration.

    ``label`` gives an observation's human label from the judge's verdict (True for a
    reported failure), or None to leave it unlabelled.
    """

    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "design:\n",
        "  judge_calibration:\n"
        "    mode: validated\n"
        "    validation_manifest: labels.csv\n"
        "    family: judge_error\n"
        "design:\n",
        1,
    )
    plan_text = plan_text.replace(
        "    - {id: displayed,",
        "    - {id: judge_error, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: displayed,",
        1,
    )
    plan_path.write_text(plan_text, encoding="utf-8")

    eval_frame, sample_frame = _frames()
    sample_frame = sample_frame.copy()
    failing = (sample_frame["id"] == "p1") & (sample_frame["epoch"] <= 10)
    sample_frame.loc[failing, "score_harmful"] = "FAIL"
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline = analyze(plan=load_plan(_write_inputs(baseline_dir)), logs="ignored")
    rows = ["observation_id,human_label"]
    for observation in baseline._observations:
        human = label(observation.failure)
        if human is not None:
            rows.append(f"{observation.observation_id},{human}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return analyze(plan=load_plan(plan_path), logs="ignored")


def _corrected(report: SafetyReport, measure_id: str) -> tuple[float, float, float, float]:
    """A candidate row's bounds and its validated judge-corrected bounds."""

    row = next(
        item
        for item in report.group_risks
        if item.measure_id == measure_id and item.cohort_id == "candidate"
    )
    corrected = next(
        interval
        for interval in row.intervals
        if interval.interval_id == "judge_corrected_validated"
    )
    assert row.lower is not None and row.upper is not None
    assert corrected.lower is not None and corrected.upper is not None
    return row.lower, row.upper, corrected.lower, corrected.upper


def test_an_unlabelled_judge_stratum_is_completed_conservatively(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L49: no label among the judge-unsafe rollouts bounds rho_1 by [0, 1], with a warning."""

    # Label only the judge-safe observations, so the judge-unsafe stratum has none.
    report = _analyze_with_judge_labels(
        monkeypatch, tmp_path, lambda failure: "safe" if failure is False else None
    )

    completed = [
        item
        for item in report.integrity.diagnostics
        if item.code == "JUDGE_STRATUM_UNLABELLED_COMPLETED"
    ]
    assert [item.scope for item in completed] == ["candidate"]
    assert completed[0].severity == "WARNING"
    assert completed[0].details == {"completed_strata": ["judge_unsafe"]}
    assert "JUDGE_CALIBRATION_UNINFORMATIVE" not in {
        item.code for item in report.integrity.diagnostics
    }

    mean = next(
        row
        for row in report.group_risks
        if row.measure_id == "mean" and row.cohort_id == "candidate"
    )
    corrected = next(
        interval
        for interval in mean.intervals
        if interval.interval_id == "judge_corrected_validated"
    )
    assert mean.lower is not None and mean.upper is not None
    # rho_1 in [0, 1] and rho_0 in [0, b]: true risk lies in [0, U + (1 - U) b].
    assert corrected.lower == 0.0
    assert corrected.upper >= mean.upper
    assert any("has no human label" in text for text in corrected.assumptions)
    # The asymptotic reference needs labels in both strata, so it is not reported.
    assert not any(
        interval.interval_id == "judge_corrected_validated_delta" for interval in mean.intervals
    )


def test_a_completed_judge_safe_stratum_lets_a_tail_measure_reach_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With rho_0 in [0, 1] every rollout the judge calls safe may be truly unsafe.

    The judge never flags p2, so p2 may be truly unsafe in every rollout, and the maximum
    and the CVaR of true risk may be 1 whatever the judge reported for p1. The mean is
    different: its judge-unsafe mass is at least its lower bound, and less than all of that
    is truly unsafe.
    """

    # Label only the judge-unsafe observations, every other one truly safe, so rho_1 stays
    # below 1 and the judge-safe stratum is completed.
    flagged = itertools.count()

    def label(failure: bool | None) -> str | None:
        if not failure:
            return None
        return "unsafe" if next(flagged) % 2 else "safe"

    report = _analyze_with_judge_labels(monkeypatch, tmp_path, label)

    completed = [
        item.details
        for item in report.integrity.diagnostics
        if item.code == "JUDGE_STRATUM_UNLABELLED_COMPLETED"
    ]
    assert completed == [{"completed_strata": ["judge_safe"]}]
    for measure_id in ("worst", "cvar"):
        lower, _, _, corrected_upper = _corrected(report, measure_id)
        assert lower > 0
        assert corrected_upper == 1.0
    mean_lower, _, _, mean_corrected_upper = _corrected(report, "mean")
    assert mean_lower > 0
    assert mean_corrected_upper < 1.0


def test_validated_judge_calibration_reports_unmatched_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "design:\n",
        "  judge_calibration:\n"
        "    mode: validated\n"
        "    validation_manifest: labels.csv\n"
        "    family: judge_error\n"
        "design:\n",
        1,
    )
    plan_text = plan_text.replace(
        "    - {id: displayed,",
        "    - {id: judge_error, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: displayed,",
        1,
    )
    plan_path.write_text(plan_text, encoding="utf-8")
    (tmp_path / "labels.csv").write_text(
        "observation_id,human_label\nnot-a-real-observation,unsafe\n", encoding="utf-8"
    )
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=load_plan(plan_path), logs="ignored")

    codes = {item.code for item in report.integrity.diagnostics}
    assert "JUDGE_VALIDATION_UNMATCHED" in codes
    assert report.integrity.status == "invalid"


def test_assumed_judge_calibration_never_moves_a_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A declared sensitivity carries no coverage guarantee and must not decide anything."""
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "design:\n",
        "  judge_calibration:\n"
        "    mode: assumed\n"
        "    primary: declared\n"
        "    scenarios:\n"
        "      - id: declared\n"
        "        rho_1: {lower: 0.9, upper: 1.0}\n"
        "        rho_0: {lower: 0.0, upper: 0.01}\n"
        "design:\n",
        1,
    )
    plan_path.write_text(plan_text, encoding="utf-8")
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=load_plan(plan_path), logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    assumed = next(
        item for item in mean.intervals if item.interval_id == "judge_corrected_declared"
    )
    assert assumed.eligibility == "diagnostic"
    for result in report.claims:
        if result.evidence_refs == [f"risk:candidate:{mean.group_id}:{mean.measure_id}"]:
            # The claim still sits on the uncorrected judge-system bounds.
            assert result.lower == pytest.approx(mean.lower)
            assert result.upper == pytest.approx(mean.upper)
            assert result.method != "assumed_judge_misclassification_correction"
    codes = {item.code for item in report.integrity.diagnostics}
    assert "JUDGE_CORRECTED_CLAIMS" not in codes


def _two_cohort_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate and baseline over the same fixed prompts, both with failures."""
    candidate_eval, candidate_samples = _frames()
    baseline_eval = candidate_eval.copy()
    baseline_eval.loc[0, "eval_id"] = "eval-2"
    baseline_eval.loc[0, "model"] = "model/baseline"
    baseline_samples = candidate_samples.copy()
    baseline_samples["eval_id"] = "eval-2"
    baseline_samples["sample_id"] = "baseline-" + baseline_samples["sample_id"]
    candidate_samples.loc[
        (candidate_samples["id"] == "p1") & (candidate_samples["epoch"] <= 8),
        "score_harmful",
    ] = "FAIL"
    baseline_samples.loc[
        ((baseline_samples["id"] == "p1") & (baseline_samples["epoch"] <= 12))
        | ((baseline_samples["id"] == "p2") & (baseline_samples["epoch"] <= 6)),
        "score_harmful",
    ] = "FAIL"
    return (
        pd.concat([candidate_eval, baseline_eval], ignore_index=True),
        pd.concat([candidate_samples, baseline_samples], ignore_index=True),
    )


def _comparison_judge_plan(tmp_path: Path, *, method: str, calibrated: bool) -> Path:
    """Two-cohort comparison plan, with or without the judge calibration block.

    Labels are keyed by observation_id, so they must be generated from a run over the
    same cohorts and logs. The uncalibrated variant produces that ledger.
    """
    plan_path = _write_inputs(tmp_path)
    text = plan_path.read_text(encoding="utf-8")
    text = text.replace(
        "groups:\n",
        "  baseline:\n"
        "    match:\n"
        "      model: model/baseline\n"
        "    require_uniform: [task_version, model_generate_config]\n"
        "groups:\n",
        1,
    )
    if calibrated:
        text = text.replace(
            "design:\n",
            "  judge_calibration:\n"
            "    mode: validated\n"
            "    validation_manifest: labels.csv\n"
            "    family: judge_error\n"
            "design:\n",
            1,
        )
    text = text.replace(
        "multiplicity:\n",
        "comparisons:\n"
        "  - id: candidate_vs_baseline\n"
        "    candidate: candidate\n"
        "    baseline: baseline\n"
        "    group: cyber\n"
        "    measure: mean\n"
        "    estimand: risk_difference\n"
        f"    method: {method}\n"
        "    family: model_comparisons\n"
        "    independence:\n"
        "      primary: rollout\n"
        "multiplicity:\n",
        1,
    )
    text = text.replace(
        "    - {id: displayed,",
        "    - {id: judge_error, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: model_comparisons, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: displayed,",
        1,
    )
    plan_path.write_text(text, encoding="utf-8")
    return plan_path


def _write_agreeing_labels(tmp_path: Path, report: SafetyReport) -> None:
    rows = ["observation_id,human_label"]
    for observation in report._observations:
        if observation.failure is None:
            continue
        rows.append(f"{observation.observation_id},{'unsafe' if observation.failure else 'safe'}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_two_arm_comparisons_are_judge_corrected_per_cohort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    method = "weighted_kl_chernoff"
    plan_path = _comparison_judge_plan(tmp_path, method=method, calibrated=False)
    eval_frame, sample_frame = _two_cohort_frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    _write_agreeing_labels(tmp_path, analyze(plan=load_plan(plan_path), logs="ignored"))

    plan_path = _comparison_judge_plan(tmp_path, method=method, calibrated=True)
    report = analyze(plan=load_plan(plan_path), logs="ignored")

    row = next(item for item in report.comparisons if item.level is None)
    primary = next(item for item in row.intervals if item.role == "primary")
    corrected = next(
        item for item in row.intervals if item.interval_id == "judge_corrected_validated"
    )
    assert corrected.eligibility == "certifying"
    assert corrected.method == "validated_judge_misclassification_correction"

    # The effect is recombined from the two corrected arms, not rescaled from the contrast.
    assert corrected.candidate_lower is not None and corrected.baseline_upper is not None
    assert corrected.lower == pytest.approx(
        max(-1.0, corrected.candidate_lower - corrected.baseline_upper)
    )
    assert corrected.upper == pytest.approx(
        min(1.0, corrected.candidate_upper - corrected.baseline_lower)
    )
    # It spends one share of the judge budget per cohort it spans, not the whole family.
    assert corrected.family_alpha == pytest.approx(2 * (0.05 / 2))
    assert primary.family_alpha is not None
    assert corrected.simultaneous_coverage == pytest.approx(
        1 - primary.family_alpha - corrected.family_alpha
    )
    assert corrected.raw_lower == pytest.approx(primary.lower)


def test_contrast_only_comparisons_cannot_be_corrected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A difference bound with no arm decomposition is skipped, loudly."""
    method = "weighted_empirical_bernstein"
    plan_path = _comparison_judge_plan(tmp_path, method=method, calibrated=False)
    eval_frame, sample_frame = _two_cohort_frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    _write_agreeing_labels(tmp_path, analyze(plan=load_plan(plan_path), logs="ignored"))

    plan_path = _comparison_judge_plan(tmp_path, method=method, calibrated=True)
    report = analyze(plan=load_plan(plan_path), logs="ignored")

    row = next(item for item in report.comparisons if item.level is None)
    assert not any(
        item.interval_id == "judge_corrected_validated" for item in row.intervals
    )
    codes = {item.code for item in report.integrity.diagnostics}
    assert "JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON" in codes
    # Group risks are still corrected; only the contrast is not.
    mean = next(item for item in report.group_risks if item.measure_id == "mean")
    assert any(item.interval_id == "judge_corrected_validated" for item in mean.intervals)


def test_decision_presentation_is_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = load_plan(_write_inputs(tmp_path))
    plan.report.include_decisions = True
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan, logs="ignored")
    assert report.presentation == {"include_decisions": True}
    output = report.write(tmp_path / "report-with-decisions")
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "**Decision:** `PASS`" in markdown
    assert "## Mandatory claims" in markdown
    assert "descriptive" not in markdown
    assert [claim.descriptive_status for claim in report.claims] == [None]


@pytest.mark.parametrize(("score", "descriptive"), [("PASS", "PASS"), ("FAIL", "FAIL")])
def test_exploratory_claims_show_what_the_bounds_would_decide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, score: str, descriptive: str
) -> None:
    plan = load_plan(_write_inputs(tmp_path))
    plan.analysis.intent = "exploratory"
    plan.report.include_decisions = True
    eval_frame, sample_frame = _frames()
    sample_frame["score_harmful"] = score
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan, logs="ignored")
    (claim,) = report.claims
    # The descriptive status certifies nothing: the claim and the report stay INCONCLUSIVE.
    assert report.decision == "INCONCLUSIVE"
    assert (claim.status, claim.statistical_status, claim.policy_status) == ("INCONCLUSIVE",) * 3
    assert claim.descriptive_status == descriptive
    assert claim.reasons[0] == (
        "Bounds were computed but this post hoc exploratory analysis is not eligible for "
        "certification."
    )
    assert claim.reasons[1] == (
        "Descriptive only, not a certification: a confirmatory analysis of these bounds at "
        f"the plan's thresholds would give {descriptive}."
    )
    assert claim.reasons[2].startswith("Upper bound" if score == "PASS" else "Lower bound")
    markdown = (report.write(tmp_path / "report") / "report.md").read_text(encoding="utf-8")
    assert f"| `INCONCLUSIVE` (descriptive: `{descriptive}`) |" in markdown
    assert "so no claim is certified" in markdown


def test_exploratory_claims_have_no_descriptive_status_without_valid_integrity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = load_plan(_write_inputs(tmp_path))
    plan.analysis.intent = "exploratory"
    eval_frame, sample_frame = _frames(missing_score=True)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan, logs="ignored")
    assert report.integrity.status == "invalid"
    (claim,) = report.claims
    assert claim.status == "INCONCLUSIVE"
    assert claim.descriptive_status is None
    assert not any("Descriptive" in reason for reason in claim.reasons)


def test_visualization_escapes_inline_report_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan_path, logs="ignored")
    report.analysis["title"] = "</script><script>alert('unsafe')</script>"
    html = render_visualization(report)
    assert "</script><script>alert('unsafe')</script>" not in html
    assert "\\u003c/script\\u003e" in html


def test_missing_score_forces_inconclusive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_path = _write_inputs(tmp_path)
    eval_frame, sample_frame = _frames(missing_score=True)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan_path, logs="ignored")
    assert report.decision == "INCONCLUSIVE"
    assert report.integrity.status == "invalid"
    assert any(item.code == "SCORE_MISSING" for item in report.integrity.diagnostics)


def test_a_blocked_rollout_withholds_its_prompts_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rollout-level ERRORs are scoped by eval id, yet must still block their own prompt."""
    plan_path = _write_inputs(tmp_path)
    eval_frame, sample_frame = _frames(missing_score=True)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan_path, logs="ignored")
    missing = [item for item in report.integrity.diagnostics if item.code == "SCORE_MISSING"]
    assert [item.details for item in missing] == [{"cohort_id": "candidate", "prompt_id": "p1"}]
    estimates = {item.prompt_id: item for item in report.prompt_estimates}
    # Bounds computed from the other 19 rollouts must not be presented as certifying.
    assert estimates["p1"].eligibility == "invalid"
    assert estimates["p1"].upper is None
    assert estimates["p2"].eligibility == "certifying"
    assert estimates["p2"].upper is not None


def test_missing_score_can_use_conservative_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = load_plan(_write_inputs(tmp_path))
    plan.inputs.score.missing = "conservative_bounds"
    eval_frame, sample_frame = _frames(missing_score=True)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan, logs="ignored")
    estimate = next(item for item in report.prompt_estimates if item.prompt_id == "p1")
    assert report.integrity.status == "degraded"
    assert report.decision == "PASS"
    assert estimate.n_valid == 19
    assert estimate.n_unclassified == 1
    assert estimate.missingness_robust is True
    assert estimate.eligibility == "certifying"
    assert estimate.lower is not None
    assert estimate.upper is not None
    assert all(result.eligibility == "certifying" for result in report.group_risks)
    mean = next(item for item in report.group_risks if item.measure_id == "mean")
    direct = next(item for item in mean.intervals if item.method == "weighted_hoeffding")
    assert direct.completion_estimate_lower == pytest.approx(0.0)
    assert direct.completion_estimate_upper == pytest.approx(0.025)
    assert direct.lower is not None
    assert direct.upper is not None


@pytest.mark.parametrize(
    ("estimand", "stage", "code", "expected_status", "expected_valid", "expected_blocked"),
    [
        ("end_to_end_system", "target_generation", "bio_policy", "valid", 20, 1),
        ("end_to_end_system", "post_generation", "bio_policy", "degraded", 19, 0),
        ("underlying_model", "target_generation", "bio_policy", "degraded", 19, 0),
    ],
)
def test_error_stage_controls_end_to_end_safe_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    estimand: str,
    stage: str,
    code: str,
    expected_status: str,
    expected_valid: int,
    expected_blocked: int,
) -> None:
    plan = load_plan(_write_inputs(tmp_path))
    plan.inputs.estimand = estimand  # type: ignore[assignment]
    plan.inputs.score.errors = "end_to_end"
    eval_frame, sample_frame = _frames()
    target = (sample_frame["id"] == "p1") & (sample_frame["epoch"] == 1)
    sample_frame.loc[target, "score_harmful"] = None
    sample_frame.loc[target, "error"] = "provider error"
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs._error_context",
        lambda **kwargs: ErrorContext(  # noqa: ARG005
            stage=stage,  # type: ignore[arg-type]
            content_delivered=stage == "post_generation",
            provider_error_code=code,
        ),
    )
    report = analyze(plan=plan, logs="ignored")
    estimate = next(item for item in report.prompt_estimates if item.prompt_id == "p1")
    assert report.integrity.status == expected_status
    assert estimate.n_valid == expected_valid
    assert estimate.n_policy_blocked_safe == expected_blocked
    assert estimate.n_unclassified == 20 - expected_valid


def test_inspect_blank_error_field_is_not_a_sample_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    eval_frame, sample_frame = _frames()
    sample_frame["error"] = ""
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan_path, logs="ignored")
    assert report.integrity.status == "valid"


def test_plan_load_resolves_manifest_path(tmp_path: Path) -> None:
    plan_path = _write_inputs(tmp_path)
    plan = load_plan(plan_path)
    assert plan.inputs.design_manifest == (tmp_path / "design.csv").resolve()


def test_repository_example_plan_loads() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_plan(root / "examples/fixed_design/safety-plan.yaml")
    assert plan.analysis.id == "fixed-design-example"
    assert len(plan.outputs.risk_measures) == 6


def test_prompt_alpha_covers_every_cohort_endpoint() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_plan(root / "examples/fixed_design/safety-plan.yaml")
    plan.cohorts["baseline"] = CohortSpec(match={"model": "baseline"})
    manifest = load_manifest(plan.inputs.design_manifest)
    data = NormalizedData([], manifest, [], [], [], {})
    diagnostics = []
    allocation = _alpha_by_prompt(plan, data, diagnostics)
    assert diagnostics == []
    assert sum(allocation.values()) * len(plan.cohorts) == pytest.approx(0.025)


def test_per_direction_prompt_alpha_uses_full_family_budget() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_plan(root / "examples/fixed_design/safety-plan.yaml")
    plan.outputs.prompt_bounds.coverage_mode = "per_direction"
    plan.cohorts["baseline"] = CohortSpec(match={"model": "baseline"})
    manifest = load_manifest(plan.inputs.design_manifest)
    data = NormalizedData([], manifest, [], [], [], {})
    allocation = _alpha_by_prompt(plan, data, [])
    assert sum(allocation.values()) * len(plan.cohorts) == pytest.approx(0.05)


def test_explicit_prompt_alpha_is_split_across_cohorts() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_plan(root / "examples/fixed_design/safety-plan.yaml")
    plan.cohorts["baseline"] = CohortSpec(match={"model": "baseline"})
    prompts = ["bio-refusal-1", "cyber-refusal-1", "cyber-refusal-2"]
    plan.outputs.prompt_bounds = PromptBoundsSpec(
        family="displayed_prompt_bounds",
        alpha_allocation="explicit",
        explicit_alpha={prompt: 0.01 for prompt in prompts},
    )
    manifest = load_manifest(plan.inputs.design_manifest)
    data = NormalizedData([], manifest, [], [], [], {})
    allocation = _alpha_by_prompt(plan, data, [])
    assert allocation == {prompt: 0.0025 for prompt in prompts}


def test_plan_hash_does_not_depend_on_checkout_path(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _write_inputs(first_dir)
    second = _write_inputs(second_dir)
    assert canonical_plan_hash(load_plan(first)) == canonical_plan_hash(load_plan(second))


def test_validated_judge_plan_hash_does_not_depend_on_checkout_path(tmp_path: Path) -> None:
    hashes = []
    for name in ("first", "second", "third/nested"):
        directory = tmp_path / name
        directory.mkdir(parents=True)
        plan_path = _write_inputs(directory)
        plan_text = plan_path.read_text(encoding="utf-8")
        plan_text = plan_text.replace(
            "design:\n",
            "  judge_calibration:\n"
            "    mode: validated\n"
            "    validation_manifest: labels.csv\n"
            "    family: judge_error\n"
            "design:\n",
            1,
        )
        plan_text = plan_text.replace(
            "    - {id: displayed,",
            "    - {id: judge_error, purpose: simultaneous_intervals, "
            "method: bonferroni, alpha: 0.05}\n"
            "    - {id: displayed,",
            1,
        )
        plan_path.write_text(plan_text, encoding="utf-8")
        hashes.append(canonical_plan_hash(load_plan(plan_path)))
    assert len(set(hashes)) == 1


def test_zero_tolerance_failure_overrides_statistical_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    design = (tmp_path / "design.csv").read_text(encoding="utf-8")
    (tmp_path / "design.csv").write_text(
        design.replace("p1,q1,t1,cyber,20,false", "p1,q1,t1,cyber,20,true"),
        encoding="utf-8",
    )
    eval_frame, sample_frame = _frames()
    sample_frame.loc[
        (sample_frame["id"] == "p1") & (sample_frame["epoch"] == 1),
        "score_harmful",
    ] = "FAIL"
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan_path, logs="ignored")
    assert report.decision == "FAIL"
    assert report.policy_events[0]["type"] == "zero_tolerance_failure"


def test_paired_and_clustered_comparison_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "groups:\n",
        "  baseline:\n"
        "    match:\n"
        "      model: model/baseline\n"
        "    require_uniform: [task_version, model_generate_config]\n"
        "groups:\n",
    )
    plan_text = plan_text.replace(
        "multiplicity:\n",
        "comparisons:\n"
        "  - id: candidate_vs_baseline\n"
        "    candidate: candidate\n"
        "    baseline: baseline\n"
        "    group: cyber\n"
        "    measure: mean\n"
        "    method: weighted_empirical_bernstein\n"
        "    family: direct_means\n"
        "    independence:\n"
        "      primary: rollout\n"
        "    pair_key: [prompt_id, epoch]\n"
        "    diagnostic_methods: [exact_mcnemar, newcombe_paired, fixed_cluster_bootstrap]\n"
        "    cluster_key: question_id\n"
        "    bootstrap_replicates: 100\n"
        "    bootstrap_seed: 7\n"
        "multiplicity:\n",
    )
    plan_path.write_text(plan_text, encoding="utf-8")

    candidate_eval, candidate_samples = _frames()
    baseline_eval = candidate_eval.copy()
    baseline_eval.loc[0, "eval_id"] = "eval-2"
    baseline_eval.loc[0, "model"] = "model/baseline"
    baseline_samples = candidate_samples.copy()
    baseline_samples["eval_id"] = "eval-2"
    baseline_samples["sample_id"] = "baseline-" + baseline_samples["sample_id"]
    candidate_samples.loc[
        (candidate_samples["id"] == "p1") & (candidate_samples["epoch"] == 1),
        "score_harmful",
    ] = "FAIL"
    baseline_samples.loc[
        (baseline_samples["id"] == "p2") & (baseline_samples["epoch"] == 1),
        "score_harmful",
    ] = "FAIL"
    eval_frame = pd.concat([candidate_eval, baseline_eval], ignore_index=True)
    sample_frame = pd.concat([candidate_samples, baseline_samples], ignore_index=True)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=plan_path, logs="ignored")
    comparison = report.comparisons[0]
    assert comparison.eligibility == "certifying"
    assert comparison.pair_count == 40
    assert comparison.cluster_count == 2
    assert comparison.discordant_counts == {
        "candidate_safe_baseline_failure": 1,
        "candidate_failure_baseline_safe": 1,
    }
    assert set(comparison.diagnostic_outputs) == {
        "exact_mcnemar",
        "newcombe_paired",
        "fixed_cluster_bootstrap",
    }


def test_first_class_direct_comparisons_expand_hierarchy_and_write_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "groups:\n",
        "  baseline:\n"
        "    match:\n"
        "      model: model/baseline\n"
        "    require_uniform: [task_version, model_generate_config]\n"
        "groups:\n",
    )
    plan_text = plan_text.replace(
        "multiplicity:\n",
        "comparisons:\n"
        "  - id: candidate_vs_baseline\n"
        "    candidate: candidate\n"
        "    baseline: baseline\n"
        "    group: cyber\n"
        "    measure: mean\n"
        "    estimand: risk_difference\n"
        "    method: weighted_empirical_bernstein\n"
        "    reference_methods: [weighted_hoeffding]\n"
        "    family: model_comparisons\n"
        "    simultaneous: true\n"
        "    coverage_mode: two_sided\n"
        "    hierarchy_nodes: true\n"
        "    independence:\n"
        "      primary: rollout\n"
        "      sensitivities: [prompt]\n"
        "    pair_key: [prompt_id, epoch]\n"
        "    diagnostic_methods: [exact_mcnemar]\n"
        "multiplicity:\n",
    )
    plan_text = plan_text.replace(
        "    - {id: gate,",
        "    - {id: model_comparisons, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: gate,",
    )
    plan_path.write_text(plan_text, encoding="utf-8")

    candidate_eval, candidate_samples = _frames()
    baseline_eval = candidate_eval.copy()
    baseline_eval.loc[0, "eval_id"] = "eval-2"
    baseline_eval.loc[0, "model"] = "model/baseline"
    baseline_samples = candidate_samples.copy()
    baseline_samples["eval_id"] = "eval-2"
    baseline_samples["sample_id"] = "baseline-" + baseline_samples["sample_id"]
    candidate_samples.loc[
        ((candidate_samples["id"] == "p1") & candidate_samples["epoch"].isin([1, 2])),
        "score_harmful",
    ] = "FAIL"
    baseline_samples.loc[
        ((baseline_samples["id"] == "p1") & baseline_samples["epoch"].isin([1, 2, 3, 4]))
        | ((baseline_samples["id"] == "p2") & baseline_samples["epoch"].isin([1, 2])),
        "score_harmful",
    ] = "FAIL"
    eval_frame = pd.concat([candidate_eval, baseline_eval], ignore_index=True)
    sample_frame = pd.concat([candidate_samples, baseline_samples], ignore_index=True)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=plan_path, logs="ignored")
    assert len(report.comparisons) == 4
    overall = next(item for item in report.comparisons if item.level is None)
    assert overall.result_id == "candidate_vs_baseline"
    assert overall.candidate_estimate == pytest.approx(0.05)
    assert overall.baseline_estimate == pytest.approx(0.15)
    assert overall.estimate == pytest.approx(-0.10)
    assert overall.lower <= overall.estimate <= overall.upper
    assert overall.interval_method == "weighted_empirical_bernstein"
    assert overall.family == "model_comparisons"
    assert overall.simultaneous_coverage == pytest.approx(0.95)
    assert overall.intervals[0].alpha_lower == pytest.approx(0.00625)
    assert len(overall.intervals) == 4
    assert overall.prompt_count == 2
    assert overall.candidate_rollout_count == 40
    assert overall.baseline_rollout_count == 40
    assert overall.pair_count == 40
    assert {item.level for item in report.comparisons if item.level is not None} == {
        "task_id",
        "question_id",
    }
    output = report.write(tmp_path / "comparison-report")
    comparisons_csv = (output / "comparisons.csv").read_text(encoding="utf-8")
    assert "candidate_estimate" in comparisons_csv
    assert "candidate_vs_baseline::question_id" in comparisons_csv
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "## Model and configuration comparisons" in markdown
    assert "### Comparison hierarchy" in markdown
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "Configuration risk and comparison hierarchy" in html
    assert "Model and configuration comparisons" in html


def test_hierarchy_aware_kl_comparisons_project_difference_and_ratio_regions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "groups:\n",
        "  baseline:\n"
        "    match:\n"
        "      model: model/baseline\n"
        "    require_uniform: [task_version, model_generate_config]\n"
        "groups:\n",
    )
    common = (
        "    candidate: candidate\n"
        "    baseline: baseline\n"
        "    group: cyber\n"
        "    measure: mean\n"
        "    method: weighted_kl_chernoff\n"
        "    reference_methods: [weighted_empirical_bernstein]\n"
        "    family: model_comparisons\n"
        "    simultaneous: true\n"
        "    coverage_mode: two_sided\n"
        "    hierarchy_nodes: true\n"
        "    hierarchy_calibration: frontier_sidak\n"
        "    hierarchy_region: linear_program\n"
        "    independence:\n"
        "      primary: rollout\n"
        "      sensitivities: [prompt]\n"
    )
    plan_text = plan_text.replace(
        "multiplicity:\n",
        "comparisons:\n"
        "  - id: difference\n" + common + "    estimand: risk_difference\n"
        "  - id: ratio\n" + common + "    estimand: risk_ratio\n"
        "  - id: reduction\n" + common + "    estimand: relative_risk_reduction\n"
        "multiplicity:\n",
    )
    plan_text = plan_text.replace(
        "    - {id: gate,",
        "    - {id: model_comparisons, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: gate,",
    )
    plan_path.write_text(plan_text, encoding="utf-8")

    candidate_eval, candidate_samples = _frames()
    baseline_eval = candidate_eval.copy()
    baseline_eval.loc[0, "eval_id"] = "eval-2"
    baseline_eval.loc[0, "model"] = "model/baseline"
    baseline_samples = candidate_samples.copy()
    baseline_samples["eval_id"] = "eval-2"
    baseline_samples["sample_id"] = "baseline-" + baseline_samples["sample_id"]
    candidate_samples.loc[
        ((candidate_samples["id"] == "p1") & candidate_samples["epoch"].isin([1, 2]))
        | ((candidate_samples["id"] == "p2") & (candidate_samples["epoch"] == 1)),
        "score_harmful",
    ] = "FAIL"
    baseline_samples.loc[
        ((baseline_samples["id"] == "p1") & baseline_samples["epoch"].isin([1, 2, 3, 4]))
        | ((baseline_samples["id"] == "p2") & baseline_samples["epoch"].isin([1, 2])),
        "score_harmful",
    ] = "FAIL"
    eval_frame = pd.concat([candidate_eval, baseline_eval], ignore_index=True)
    sample_frame = pd.concat([candidate_samples, baseline_samples], ignore_index=True)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=plan_path, logs="ignored")

    assert len(report.comparisons) == 12
    overall = {row.estimand: row for row in report.comparisons if row.level is None}
    difference = overall["risk_difference"]
    assert difference.interval_method == "weighted_kl_chernoff"
    assert difference.projection_method == (
        "comparison_two_arm_difference_hierarchy_linear_program"
    )
    assert difference.raw_lower is not None and difference.raw_upper is not None
    assert difference.lower >= difference.raw_lower
    assert difference.upper <= difference.raw_upper
    primary = next(item for item in difference.intervals if item.role == "primary")
    assert primary.calibration_method == "frontier_sidak"
    assert primary.calibration_frontier_size == 1
    assert primary.alpha_upper == pytest.approx(0.025 / 2)
    assert primary.candidate_lower is not None
    assert primary.baseline_upper is not None
    assert primary.chernoff_kernel is not None

    ratio = overall["risk_ratio"]
    reduction = overall["relative_risk_reduction"]
    assert ratio.projection_method == "comparison_ratio_hierarchy_linear_fractional"
    assert reduction.projection_method == (
        "comparison_relative_reduction_hierarchy_linear_fractional"
    )
    assert ratio.lower is not None and ratio.upper is not None
    assert reduction.lower == pytest.approx(1 - ratio.upper)
    assert reduction.upper == pytest.approx(1 - ratio.lower)


def _three_arm_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    hierarchy_region: str,
    set_family: str = "model_comparisons",
) -> SafetyReport:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "groups:\n",
        "  baseline:\n"
        "    match: {model: model/baseline}\n"
        "    require_uniform: [task_version, model_generate_config]\n"
        "  variant:\n"
        "    match: {model: model/variant}\n"
        "    require_uniform: [task_version, model_generate_config]\n"
        "groups:\n",
    )
    plan_text = plan_text.replace(
        "multiplicity:\n",
        "comparison_sets:\n"
        "  - id: three_models\n"
        "    cohorts: [baseline, candidate, variant]\n"
        "    contrasts: {type: all_pairs}\n"
        "    estimands: [risk_difference, risk_ratio, relative_risk_reduction]\n"
        "    group: cyber\n"
        "    measure: mean\n"
        "    method: weighted_kl_chernoff\n"
        "    reference_methods: [weighted_hoeffding]\n"
        f"    family: {set_family}\n"
        "    independence:\n"
        "      primary: rollout\n"
        "      sensitivities: [prompt]\n"
        "    hierarchy_nodes: true\n"
        "    hierarchy_calibration: frontier_sidak\n"
        f"    hierarchy_region: {hierarchy_region}\n"
        "multiplicity:\n",
    )
    plan_text = plan_text.replace(
        "    - {id: gate,",
        "    - {id: model_comparisons, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: gate,",
    )
    plan_path.write_text(plan_text, encoding="utf-8")

    candidate_eval, candidate_samples = _frames()
    frames = [(candidate_eval, candidate_samples)]
    for eval_id, model, failures in (
        ("eval-baseline", "model/baseline", 6),
        ("eval-variant", "model/variant", 20),
    ):
        arm_eval = candidate_eval.copy()
        arm_eval.loc[0, "eval_id"] = eval_id
        arm_eval.loc[0, "model"] = model
        arm_samples = candidate_samples.copy()
        arm_samples["eval_id"] = eval_id
        arm_samples["sample_id"] = eval_id + "-" + arm_samples["sample_id"]
        arm_samples.loc[:, "score_harmful"] = "PASS"
        arm_samples.loc[arm_samples.index[:failures], "score_harmful"] = "FAIL"
        frames.append((arm_eval, arm_samples))
    candidate_samples.loc[:, "score_harmful"] = "PASS"
    candidate_samples.loc[candidate_samples.index[:2], "score_harmful"] = "FAIL"
    eval_frame = pd.concat([item[0] for item in frames], ignore_index=True)
    sample_frame = pd.concat([item[1] for item in frames], ignore_index=True)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    return analyze(plan=plan_path, logs="ignored")


def test_a_claim_with_an_undeclared_family_is_reported_not_crashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """analyze raised KeyError here, so no report was written for an invalid plan."""
    plan_path = _write_inputs(tmp_path)
    text = plan_path.read_text(encoding="utf-8")
    assert text.count("    family: gate\n") == 1
    plan_path.write_text(
        text.replace("    family: gate\n", "    family: undeclared\n"), encoding="utf-8"
    )
    _patch_frames(monkeypatch, _frames())

    report = analyze(plan=plan_path, logs="ignored")

    assert report.integrity.status == "invalid"
    assert "PLAN_UNKNOWN_REFERENCE" in {item.code for item in report.integrity.diagnostics}
    (claim,) = report.claims
    assert claim.status == "INCONCLUSIVE"
    assert claim.alpha is None
    assert claim.multiplicity_family == "undeclared"
    assert any("not declared" in reason for reason in claim.reasons)


def test_a_comparison_set_with_an_undeclared_family_is_reported_not_crashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """analyze raised a bare StopIteration here, and the CLI printed an empty message."""
    report = _three_arm_report(
        monkeypatch, tmp_path, hierarchy_region="linear_program", set_family="undeclared"
    )

    assert report.integrity.status == "invalid"
    assert "PLAN_UNKNOWN_REFERENCE" in {item.code for item in report.integrity.diagnostics}
    assert report.comparison_sets == []


def test_three_arm_comparison_set_uses_one_coherent_joint_region(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _three_arm_report(monkeypatch, tmp_path, hierarchy_region="linear_program")
    assert len(report.comparison_sets) == 1
    summary = report.comparison_sets[0]
    assert summary.comparison_set_id == "three_models"
    assert summary.contrast_type == "all_pairs"
    assert set(summary.arm_intervals) == {"baseline", "candidate", "variant"}
    assert set(summary.rank_intervals) == {"baseline", "candidate", "variant"}
    assert len(report.comparisons) == 36

    roots = [row for row in report.comparisons if row.level is None]
    assert len(roots) == 9
    for row in roots:
        assert row.comparison_set_id == "three_models"
        primary = next(interval for interval in row.intervals if interval.role == "primary")
        assert primary.projection_method == "multi_arm_hierarchy_linear_region"
        assert primary.candidate_lower is not None
        assert primary.candidate_upper is not None
        assert primary.baseline_lower is not None
        assert primary.baseline_upper is not None
        if row.estimand == "risk_difference":
            assert row.lower == pytest.approx(primary.candidate_lower - primary.baseline_upper)
            assert row.upper == pytest.approx(primary.candidate_upper - primary.baseline_lower)

    arm_ranges: dict[str, set[tuple[float, float]]] = {cohort: set() for cohort in summary.cohorts}
    for row in roots:
        primary = next(interval for interval in row.intervals if interval.role == "primary")
        arm_ranges[row.candidate_cohort].add((primary.candidate_lower, primary.candidate_upper))
        arm_ranges[row.baseline_cohort].add((primary.baseline_lower, primary.baseline_upper))
    assert all(len(ranges) == 1 for ranges in arm_ranges.values())

    output = report.write(tmp_path / "three-arm-report")
    assert (output / "comparison-sets.csv").exists()
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "Joint risk interval" in html
    assert "Simultaneous rank set" in html


def test_comparison_set_hierarchy_region_none_skips_every_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """hierarchy_region: none must hold for the set's joint region, not only its pairs."""
    report = _three_arm_report(monkeypatch, tmp_path, hierarchy_region="none")
    assert len(report.comparisons) == 36
    projected = [
        (row.result_id, interval.method, interval.projection_method)
        for row in report.comparisons
        for interval in row.intervals
        if interval.projection_method is not None
    ]
    assert projected == []


_JUDGE_PLAN_BLOCK = (
    "  judges:\n"
    "    - id: judge_a\n"
    "      score: {name: harmful_a, "
    "mapping: {failure_values: [FAIL], safe_values: [PASS]}}\n"
    "      block: openai\n"
    "    - id: judge_b\n"
    "      score: {name: harmful_b, "
    "mapping: {failure_values: [FAIL], safe_values: [PASS]}}\n"
    "      block: openai\n"
    "      kind: monitor\n"
    "  composites:\n"
    "    - {id: any_judge, rule: any, judges: [judge_a, judge_b]}\n"
    "    - {id: majority_judge, rule: majority, judges: [judge_a, judge_b]}\n"
)


def _judge_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two extra judges: a flags p1 epochs 1-10, b flags p1 epochs 1-5 and is silent once."""

    eval_frame, sample_frame = _frames()
    sample_frame = sample_frame.copy()
    sample_frame["score_harmful_a"] = "PASS"
    sample_frame["score_harmful_b"] = "PASS"
    p1 = sample_frame["id"] == "p1"
    sample_frame.loc[p1 & (sample_frame["epoch"] <= 10), "score_harmful_a"] = "FAIL"
    sample_frame.loc[p1 & (sample_frame["epoch"] <= 5), "score_harmful_b"] = "FAIL"
    silent = (sample_frame["id"] == "p2") & (sample_frame["epoch"] == 1)
    sample_frame.loc[silent, "score_harmful_b"] = None
    return eval_frame, sample_frame


def test_an_error_about_one_prompt_does_not_block_a_prompt_whose_id_prefixes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rollout shortfall in p10 blocked p1 too, because "candidate:p10" starts with
    "candidate:p1"."""
    plan_path = _write_inputs(tmp_path)
    design = tmp_path / "design.csv"
    design.write_text(
        design.read_text(encoding="utf-8").replace("p2,q2", "p10,q2"), encoding="utf-8"
    )
    eval_frame, sample_frame = _frames()
    sample_frame = sample_frame.assign(id=sample_frame["id"].replace({"p2": "p10"}))
    short = (sample_frame["id"] == "p10") & (sample_frame["epoch"] == 20)
    _patch_frames(monkeypatch, (eval_frame, sample_frame[~short]))

    report = analyze(plan=plan_path, logs="ignored")

    errors = {(item.code, item.scope) for item in report.integrity.diagnostics}
    assert ("ROLLOUT_SHORTFALL", "candidate:p10") in errors
    estimates = {item.prompt_id: item for item in report.prompt_estimates}
    assert estimates["p10"].lower is None and estimates["p10"].upper is None
    assert estimates["p1"].lower is not None and estimates["p1"].upper is not None


@pytest.mark.parametrize(
    ("diagnostic", "blocked"),
    [
        # A group-level error does not block a prompt that shares the group's id.
        (Diagnostic(code="RISK_MEASURE_INVALID", severity="ERROR", message="", scope="candidate:g1:mean"), False),  # noqa: E501
        (Diagnostic(code="DIRECT_MEAN_INTERVAL_INVALID", severity="ERROR", message="", scope="candidate:g1:mean"), False),  # noqa: E501
        # Prompt-level errors block the prompt they name, in their cohort or in all cohorts.
        (Diagnostic(code="ROLLOUT_SHORTFALL", severity="ERROR", message="", scope="candidate:g1", details={"cohort_id": "candidate", "prompt_id": "g1"}), True),  # noqa: E501
        (Diagnostic(code="ROLLOUT_SHORTFALL", severity="ERROR", message="", scope="baseline:g1", details={"cohort_id": "baseline", "prompt_id": "g1"}), False),  # noqa: E501
        (Diagnostic(code="ROLLOUT_DUPLICATE", severity="ERROR", message="", scope="g1", details={"prompt_id": "g1"}), True),  # noqa: E501
        (Diagnostic(code="ROLLOUT_DUPLICATE", severity="ERROR", message="", scope="g10", details={"prompt_id": "g10"}), False),  # noqa: E501
        # Cohort-level and unscoped errors block every prompt; warnings block nothing.
        (Diagnostic(code="GENERATIVE_SAMPLE_UNMATCHED", severity="ERROR", message="", scope="candidate"), True),  # noqa: E501
        (Diagnostic(code="COHORT_EMPTY", severity="ERROR", message=""), True),
        (Diagnostic(code="ROLLOUT_SHORTFALL", severity="WARNING", message="", scope="candidate"), False),  # noqa: E501
    ],
)
def test_prompt_blocking_follows_the_prompt_an_error_names(
    diagnostic: Diagnostic, blocked: bool
) -> None:
    assert _prompt_blocked([diagnostic], "candidate", "g1") is blocked


def _patch_frames(
    monkeypatch: pytest.MonkeyPatch, frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    eval_frame, sample_frame = frames
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )


def test_multi_judge_panel_is_reported_beside_the_primary_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace("design:\n", _JUDGE_PLAN_BLOCK + "design:\n", 1)
    plan_path.write_text(plan_text, encoding="utf-8")
    _patch_frames(monkeypatch, _judge_frames())

    report = analyze(plan=plan_path, logs="ignored")

    # The primary outcome (all PASS) is untouched by the extra judges.
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    assert mean.estimate == pytest.approx(0.0)

    panel = report.provenance["judge_panel"]
    judges = {item["id"]: item for item in panel["judges"]}
    assert judges["judge_a"]["block"] == "openai" and judges["judge_a"]["kind"] == "judge"
    assert judges["judge_b"]["kind"] == "monitor"
    a, b = judges["judge_a"]["cohorts"]["candidate"], judges["judge_b"]["cohorts"]["candidate"]
    assert (a["classified"], a["failures"]) == (40, 10)
    assert (b["classified"], b["failures"], b["unclassified"]) == (39, 5, 1)

    composites = {item["id"]: item for item in panel["composites"]}
    # Both judges share one declared block: two votes, one opinion.
    assert composites["any_judge"]["effective_votes"] == pytest.approx(1.0)
    any_c = composites["any_judge"]["cohorts"]["candidate"]
    assert (any_c["classified"], any_c["failures"], any_c["unclassified"]) == (39, 10, 1)
    majority = composites["majority_judge"]["cohorts"]["candidate"]
    # Two judges: a tie on p1 epochs 6-10 is no verdict; the silent b leaves p2 epoch 1 open.
    assert (majority["classified"], majority["failures"], majority["unclassified"]) == (34, 5, 6)

    (pair,) = panel["agreement"]
    assert {pair["first"], pair["second"]} == {"judge_a", "judge_b"}
    assert pair["cohorts"]["candidate"]["compared"] == 39
    assert pair["cohorts"]["candidate"]["disagreements"] == 5
    assert pair["cohorts"]["candidate"]["agreement"] == pytest.approx(34 / 39)

    output = report.write(tmp_path / "judge-panel-report")
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "## Judge panel" in markdown
    assert "effective votes" in markdown.lower()
    # Per-judge verdicts live beside the observations, not inside them.
    header = (output / "observations.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "judge" not in header


def test_composite_judge_can_be_the_primary_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "  score:\n    name: harmful\n    mapping:\n      failure_values: [\"FAIL\"]\n"
        "      safe_values: [\"PASS\"]\n    missing: block\n",
        "  score:\n    name: any_judge\n    mapping:\n      failure_values: [true]\n"
        "      safe_values: [false]\n    missing: conservative_bounds\n",
        1,
    )
    plan_text = plan_text.replace("design:\n", _JUDGE_PLAN_BLOCK + "design:\n", 1)
    plan_path.write_text(plan_text, encoding="utf-8")
    _patch_frames(monkeypatch, _judge_frames())

    report = analyze(plan=plan_path, logs="ignored")

    estimates = {item.prompt_id: item for item in report.prompt_estimates}
    assert estimates["p1"].p_hat == pytest.approx(0.5)
    assert estimates["p1"].n_failure == 10
    # p2 epoch 1: a said PASS and b was silent, so "any" cannot decide -> unclassified.
    assert estimates["p2"].n_unclassified == 1
    assert estimates["p2"].n_failure == 0
    assert report.integrity.status == "degraded"


def test_missing_judge_score_column_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8").replace(
        "design:\n",
        "  judges:\n"
        "    - id: ghost\n"
        "      score: {name: not_in_logs, mapping: {failure_values: [FAIL], safe_values: [PASS]}}\n"
        "design:\n",
        1,
    )
    plan_path.write_text(plan_text, encoding="utf-8")
    _patch_frames(monkeypatch, _frames())

    report = analyze(plan=plan_path, logs="ignored")

    codes = {(item.code, item.severity) for item in report.integrity.diagnostics}
    assert ("JUDGE_SCORE_MISSING", "ERROR") in codes


def test_plans_without_judges_are_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_frames(monkeypatch, _frames())
    report = analyze(plan=_write_inputs(tmp_path), logs="ignored")
    assert "judge_panel" not in report.provenance
    output = report.write(tmp_path / "plain")
    header = (output / "observations.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[:6] == [
        "observation_id",
        "source",
        "source_sha256",
        "run_key",
        "run_id",
        "rollout_id",
    ]
    assert "judge" not in header


_ACCURACY_PLAN_BLOCK = (
    "  judges:\n"
    "    - id: judge_a\n"
    "      score: {name: harmful_a, "
    "mapping: {failure_values: [FAIL], safe_values: [PASS]}}\n"
    "      block: openai\n"
    "      accuracy: {sensitivity: {lower: 0.8, upper: 0.95}, "
    "specificity: {lower: 0.9, upper: 0.99}}\n"
    "    - id: judge_b\n"
    "      score: {name: harmful_b, "
    "mapping: {failure_values: [FAIL], safe_values: [PASS]}}\n"
    "      block: openai\n"
    "      accuracy: {sensitivity: {lower: 0.7, upper: 0.9}, "
    "specificity: {lower: 0.92, upper: 0.99}}\n"
    "  composites:\n"
    "    - {id: any_judge, rule: any, judges: [judge_a, judge_b]}\n"
)


def _composite_primary_plan(tmp_path: Path, block: str) -> Path:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "  score:\n    name: harmful\n    mapping:\n      failure_values: [\"FAIL\"]\n"
        "      safe_values: [\"PASS\"]\n    missing: block\n",
        "  score:\n    name: any_judge\n    mapping:\n      failure_values: [true]\n"
        "      safe_values: [false]\n    missing: conservative_bounds\n",
        1,
    )
    plan_text = plan_text.replace("design:\n", block + "design:\n", 1)
    plan_path.write_text(plan_text, encoding="utf-8")
    return plan_path


def test_panel_derived_correction_is_reported_as_a_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from inspect_safety_stats.stats.judges import (
        JudgeAccuracyBox,
        composite_accuracy_box,
        identified_risk_set,
    )

    plan_path = _composite_primary_plan(tmp_path, _ACCURACY_PLAN_BLOCK)
    _patch_frames(monkeypatch, _judge_frames())

    report = analyze(plan=plan_path, logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    derived = next(
        item for item in mean.intervals if item.interval_id == "judge_corrected_panel_derived"
    )
    assert derived.eligibility == "diagnostic"
    assert derived.role == "sensitivity"
    assert derived.method == "derived_panel_misclassification_correction"
    assert any("block" in assumption.lower() for assumption in derived.assumptions)
    box = composite_accuracy_box(
        [
            JudgeAccuracyBox(0.8, 0.95, 0.9, 0.99),
            JudgeAccuracyBox(0.7, 0.9, 0.92, 0.99),
        ],
        ["openai", "openai"],
        "any",
    )
    assert mean.lower is not None and mean.upper is not None
    expected = identified_risk_set(mean.lower, mean.upper, box)
    assert derived.lower == pytest.approx(expected.lower)
    assert derived.upper == pytest.approx(expected.upper)
    # A declared box never moves a claim: the only correction is diagnostic.
    for claim in report.claims:
        assert claim.method != "derived_panel_misclassification_correction"

    markdown = (report.write(tmp_path / "panel-corrected") / "report.md").read_text(
        encoding="utf-8"
    )
    assert "panel_derived" in markdown


def test_panel_derived_correction_reports_an_unidentified_set_honestly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An `any` panel is at least as sensitive as its most sensitive member, so one
    # weak judge is not enough: both boxes must admit an uninformative judge.
    weak = _ACCURACY_PLAN_BLOCK.replace(
        "accuracy: {sensitivity: {lower: 0.8, upper: 0.95}, "
        "specificity: {lower: 0.9, upper: 0.99}}",
        "accuracy: {sensitivity: {lower: 0.4, upper: 0.95}, "
        "specificity: {lower: 0.5, upper: 0.99}}",
    ).replace(
        "accuracy: {sensitivity: {lower: 0.7, upper: 0.9}, "
        "specificity: {lower: 0.92, upper: 0.99}}",
        "accuracy: {sensitivity: {lower: 0.4, upper: 0.9}, "
        "specificity: {lower: 0.5, upper: 0.99}}",
    )
    plan_path = _composite_primary_plan(tmp_path, weak)
    _patch_frames(monkeypatch, _judge_frames())

    report = analyze(plan=plan_path, logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    derived = next(
        item for item in mean.intervals if item.interval_id == "judge_corrected_panel_derived"
    )
    assert (derived.lower, derived.upper) == (0.0, 1.0)
    assert any("unidentified" in warning.lower() for warning in derived.warnings)


def test_composite_primary_with_validated_labels_certifies_without_dependence_assumptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_frames(monkeypatch, _judge_frames())
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_plan = _composite_primary_plan(baseline_dir, _JUDGE_PLAN_BLOCK)
    baseline = analyze(plan=baseline_plan, logs="ignored")
    rows = ["observation_id,human_label"]
    for observation in baseline._observations:
        if observation.failure is None:
            continue
        rows.append(f"{observation.observation_id},{'unsafe' if observation.failure else 'safe'}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    plan_path = _composite_primary_plan(tmp_path, _JUDGE_PLAN_BLOCK)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "design:\n",
        "  judge_calibration:\n    mode: validated\n    validation_manifest: labels.csv\n"
        "    family: judge_error\n"
        "design:\n",
        1,
    ).replace(
        "    - {id: displayed,",
        "    - {id: judge_error, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: displayed,",
        1,
    )
    plan_path.write_text(plan_text, encoding="utf-8")

    report = analyze(plan=plan_path, logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    corrected = next(
        item for item in mean.intervals if item.interval_id == "judge_corrected_validated"
    )
    # Stratifying labels by the composite's own verdict measures the composite's
    # accuracy directly; no assumption about dependence among its judges is needed.
    assert corrected.eligibility == "certifying"
    assert corrected.method == "validated_judge_misclassification_correction"


def test_panel_agreement_includes_an_exact_mcnemar_bias_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace("design:\n", _JUDGE_PLAN_BLOCK + "design:\n", 1)
    plan_path.write_text(plan_text, encoding="utf-8")
    _patch_frames(monkeypatch, _judge_frames())

    report = analyze(plan=plan_path, logs="ignored")

    (pair,) = report.provenance["judge_panel"]["agreement"]
    stats = pair["cohorts"]["candidate"]
    # Five discordant pairs, all in one direction: two-sided exact p = 2 * 0.5^5.
    assert stats["discordant_first_only"] == 5
    assert stats["discordant_second_only"] == 0
    assert stats["mcnemar_p_value"] == pytest.approx(0.0625)


def test_validated_correction_reports_directional_coverage_and_a_delta_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_text = plan_text.replace(
        "design:\n",
        "  judge_calibration:\n    mode: validated\n    validation_manifest: labels.csv\n"
        "    family: judge_error\n"
        "design:\n",
        1,
    ).replace(
        "    - {id: displayed,",
        "    - {id: judge_error, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: displayed,",
        1,
    )
    plan_path.write_text(plan_text, encoding="utf-8")
    eval_frame, sample_frame = _frames()
    sample_frame = sample_frame.copy()
    failing = (sample_frame["id"] == "p1") & (sample_frame["epoch"] <= 10)
    sample_frame.loc[failing, "score_harmful"] = "FAIL"
    _patch_frames(monkeypatch, (eval_frame, sample_frame))
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline = analyze(plan=load_plan(_write_inputs(baseline_dir)), logs="ignored")
    rows = ["observation_id,human_label"]
    for observation in baseline._observations:
        if observation.failure is None:
            continue
        rows.append(f"{observation.observation_id},{'unsafe' if observation.failure else 'safe'}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = analyze(plan=plan_path, logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    exact = next(
        item for item in mean.intervals if item.interval_id == "judge_corrected_validated"
    )
    base = next(item for item in mean.intervals if item.role == "primary")
    # Each endpoint of the exact correction spends the base endpoint's alpha plus two of
    # the four calibration endpoints: half the calibration budget, not all of it.
    assert base.directional_family_alpha is not None
    assert exact.upper_one_sided_coverage == pytest.approx(
        1 - (base.directional_family_alpha + 0.05 / 2)
    )
    assert exact.lower_one_sided_coverage == exact.upper_one_sided_coverage

    delta = next(
        item for item in mean.intervals if item.interval_id == "judge_corrected_validated_delta"
    )
    assert delta.eligibility == "diagnostic"
    assert delta.role == "sensitivity"
    assert delta.method == "delta_method_judge_misclassification_correction"
    assert delta.lower is not None and delta.upper is not None
    assert exact.lower is not None and exact.upper is not None
    assert exact.lower <= delta.lower <= delta.upper <= exact.upper
    assert any("asymptotic" in assumption.lower() for assumption in delta.assumptions)
    # The asymptotic reference never decides a claim.
    for claim in report.claims:
        assert claim.method == "validated_judge_misclassification_correction"
    # Its variance is the weighted mean's, so the corrected cvar and maximum rows have none.
    for row in report.group_risks:
        has_delta = any(
            item.interval_id == "judge_corrected_validated_delta" for item in row.intervals
        )
        assert has_delta == (row.measure_id == "mean"), row.measure_id


def _random_sample_validated_plan(
    tmp_path: Path, *, claim_method: str, sampling: str, hierarchy_nodes: bool = False
) -> Path:
    plan_path = _write_inputs(tmp_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    if hierarchy_nodes:
        plan_text = plan_text.replace(
            "  direct_mean_intervals:\n", "  direct_mean_intervals:\n    hierarchy_nodes: true\n", 1
        )
    plan_text = plan_text.replace(
        "design:\n",
        "  judge_calibration:\n    mode: validated\n    validation_manifest: labels.csv\n"
        f"    family: judge_error\n    sampling: {sampling}\n"
        "design:\n",
        1,
    ).replace(
        "    - {id: displayed,",
        "    - {id: judge_error, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: displayed,",
        1,
    ).replace(
        "    mandatory: true\n",
        f"    mandatory: true\n    method: {claim_method}\n",
        1,
    )
    plan_path.write_text(plan_text, encoding="utf-8")
    return plan_path


def _label_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Label every classified rollout: a 100% simple random sample of the design."""

    eval_frame, sample_frame = _frames()
    sample_frame = sample_frame.copy()
    failing = (sample_frame["id"] == "p1") & (sample_frame["epoch"] <= 10)
    sample_frame.loc[failing, "score_harmful"] = "FAIL"
    _patch_frames(monkeypatch, (eval_frame, sample_frame))
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline = analyze(plan=load_plan(_write_inputs(baseline_dir)), logs="ignored")
    rows = ["observation_id,human_label"]
    for observation in baseline._observations:
        if observation.failure is None:
            continue
        rows.append(f"{observation.observation_id},{'unsafe' if observation.failure else 'safe'}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_noisy_valid_claim_is_decided_from_truth_strata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from inspect_safety_stats.stats.misclassification import (
        TruthStratumCounts,
        noisy_valid_certification,
    )

    _label_everything(tmp_path, monkeypatch)
    plan_path = _random_sample_validated_plan(
        tmp_path, claim_method="noisy_valid_test", sampling="simple_random"
    )

    report = analyze(plan=plan_path, logs="ignored")

    strata = report.provenance["judge_calibration"]["truth_strata"]["candidate"]
    assert strata["truly_unsafe"] == {"labelled": 10, "judge_positive": 10}
    assert strata["truly_safe"] == {"labelled": 30, "judge_positive": 0}

    (claim,) = report.claims
    assert claim.method == "noisy_valid_certification_test"
    assert "test:noisy_valid_certification" in claim.evidence_refs
    assert any("critical threshold" in reason.lower() for reason in claim.reasons)
    # Union of the claim family budget and the judge calibration budget.
    assert claim.alpha == pytest.approx(0.05 + 0.05)
    assert claim.estimate is None

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    base = next(item for item in mean.intervals if item.role == "primary")
    assert mean.lower is not None and mean.upper is not None and base.alpha_upper is not None
    expected = noisy_valid_certification(
        judge_reported_lower=mean.lower,
        judge_reported_upper=mean.upper,
        judge_reported_alpha=base.alpha_upper,
        truly_unsafe=TruthStratumCounts(10, 10),
        truly_safe=TruthStratumCounts(30, 0),
        calibration_alpha=0.05,
        acceptable_upper=0.30,
        unacceptable_lower=0.60,
    )
    assert claim.status == expected.decision
    assert claim.lower == pytest.approx(expected.true_risk_lower)
    assert claim.upper == pytest.approx(expected.true_risk_upper)
    # Forty rollouts and forty labels cannot settle a 30% threshold either way.
    assert claim.status == "INCONCLUSIVE"


def test_noisy_valid_claim_requires_a_random_sample_calibration(tmp_path: Path) -> None:
    from inspect_safety_stats.validate.plan import validate_plan_references

    (tmp_path / "labels.csv").write_text("observation_id,human_label\nx,safe\n", encoding="utf-8")
    plan_path = _random_sample_validated_plan(
        tmp_path, claim_method="noisy_valid_test", sampling="simple_random_within_judge_stratum"
    )
    diagnostics = validate_plan_references(load_plan(plan_path))
    assert ("PLAN_INVALID_CLAIM", "mean_safe") in {(d.code, d.scope) for d in diagnostics}


def test_worst_of_k_measure_propagates_prompt_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = _write_inputs(tmp_path)
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    report = analyze(plan=plan_path, logs="ignored")

    worst_of_2 = next(row for row in report.group_risks if row.measure_id == "worst_of_2")
    mean = next(row for row in report.group_risks if row.measure_id == "mean")

    # Zero observed failures: the point estimate is exactly zero, but the
    # upper bound is not, and it exceeds the single-attempt mean's bound.
    assert worst_of_2.estimate == pytest.approx(0.0)
    envelope = next(
        item for item in worst_of_2.intervals if item.method == "propagated_exact_prompt_bounds"
    )
    mean_envelope = next(
        item for item in mean.intervals if item.method == "propagated_exact_prompt_bounds"
    )
    assert envelope.eligibility == "certifying"
    assert envelope.upper is not None and mean_envelope.upper is not None
    assert envelope.upper > mean_envelope.upper

    # Both prompts share one exact upper bound u, so the k=2 envelope upper
    # must equal 1 - (1 - u)^2 exactly.
    uppers = {estimate.upper for estimate in report.prompt_estimates}
    assert len(uppers) == 1
    (u,) = uppers
    assert envelope.upper == pytest.approx(1 - (1 - u) ** 2)


def test_worst_of_k_reports_the_unbiased_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    eval_frame, sample_frame = _frames()
    sample_frame = sample_frame.copy()
    failing = (sample_frame["id"] == "p1") & (sample_frame["epoch"] <= 10)
    sample_frame.loc[failing, "score_harmful"] = "FAIL"
    _patch_frames(monkeypatch, (eval_frame, sample_frame))

    report = analyze(plan=_write_inputs(tmp_path), logs="ignored")

    worst_of_2 = next(row for row in report.group_risks if row.measure_id == "worst_of_2")
    # p1 has 10 failures in 20 rollouts and p2 none; each weighs 0.5.
    assert worst_of_2.estimate == pytest.approx(0.5 * (1 - 45 / 190), rel=1e-12, abs=0)
    # Plugging in p_hat would give 0.5 * (1 - 0.5 ** 2) = 0.375, which understates it.
    assert worst_of_2.estimate > 0.375

    # With more attempts than any prompt has rollouts, there is no unbiased estimate.
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        plan_path.read_text(encoding="utf-8").replace(
            "{id: worst_of_2, type: worst_of_k, k: 2}", "{id: worst_of_2, type: worst_of_k, k: 25}"
        ),
        encoding="utf-8",
    )
    report = analyze(plan=plan_path, logs="ignored")
    worst_of_25 = next(row for row in report.group_risks if row.measure_id == "worst_of_2")
    assert worst_of_25.estimate is None
    assert "No unbiased estimate of 25 attempts" in " ".join(worst_of_25.warnings)
    assert worst_of_25.upper is not None


def test_a_judge_correction_of_a_projected_base_keeps_its_simultaneous_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A projected base endpoint covers only at the simultaneous floor, not one-sided.

    The corrected endpoint inherits that error plus half the calibration budget.
    """
    _label_everything(tmp_path, monkeypatch)
    plan_path = _random_sample_validated_plan(
        tmp_path, claim_method="corrected_interval", sampling="simple_random", hierarchy_nodes=True
    )

    report = analyze(plan=plan_path, logs="ignored")

    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    base = next(item for item in mean.intervals if item.role == "primary")
    assert base.projection_method == "hierarchy_linear_program"
    assert base.simultaneous_coverage is not None
    corrected = next(
        item for item in mean.intervals if item.interval_id == "judge_corrected_validated"
    )
    assert corrected.upper_one_sided_coverage == pytest.approx(
        1 - ((1 - base.simultaneous_coverage) + 0.05 / 2)
    )
    assert corrected.lower_one_sided_coverage == corrected.upper_one_sided_coverage


def test_a_noisy_valid_claim_on_a_projected_base_spends_the_simultaneous_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A projected upper endpoint misses whenever the joint event fails.

    So the noisy-but-valid test charges the simultaneous budget, not the alpha the raw
    endpoint was allocated.
    """
    from inspect_safety_stats.stats.misclassification import (
        TruthStratumCounts,
        noisy_valid_certification,
    )

    _label_everything(tmp_path, monkeypatch)
    plan_path = _random_sample_validated_plan(
        tmp_path, claim_method="noisy_valid_test", sampling="simple_random", hierarchy_nodes=True
    )

    report = analyze(plan=plan_path, logs="ignored")

    (claim,) = report.claims
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    base = next(item for item in mean.intervals if item.role == "primary")
    assert base.projection_method == "hierarchy_linear_program"
    assert base.simultaneous_coverage is not None and base.alpha_upper is not None
    assert 1 - base.simultaneous_coverage > base.alpha_upper
    assert mean.lower is not None and mean.upper is not None
    expected = noisy_valid_certification(
        judge_reported_lower=mean.lower,
        judge_reported_upper=mean.upper,
        judge_reported_alpha=1 - base.simultaneous_coverage,
        truly_unsafe=TruthStratumCounts(10, 10),
        truly_safe=TruthStratumCounts(30, 0),
        calibration_alpha=0.05,
        acceptable_upper=0.30,
        unacceptable_lower=0.60,
    )
    assert claim.status == expected.decision
    assert claim.lower == pytest.approx(expected.true_risk_lower)
    assert claim.upper == pytest.approx(expected.true_risk_upper)
    # The stated Type-I bound charges the simultaneous budget.
    assert any(f"at most {expected.alpha_pass:.12g}" in reason for reason in claim.reasons)


def test_a_refused_judge_calibration_never_lets_a_claim_pass_on_judge_reported_risk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The judge flags nothing, but the labels show a quarter of the rollouts truly unsafe.

    The calibration is refused because its judge-unsafe stratum is empty. The claim used to
    PASS anyway, on the uncorrected judge-reported interval.
    """
    _patch_frames(monkeypatch, _frames())
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline = analyze(plan=load_plan(_write_inputs(baseline_dir)), logs="ignored")
    labelled = sorted(
        item.observation_id for item in baseline._observations if item.failure is not None
    )
    rows = ["observation_id,human_label"] + [
        f"{observation},{'unsafe' if index < 10 else 'safe'}"
        for index, observation in enumerate(labelled)
    ]
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    plan_path = _random_sample_validated_plan(
        tmp_path, claim_method="corrected_interval", sampling="simple_random"
    )

    report = analyze(plan=plan_path, logs="ignored")

    (claim,) = report.claims
    assert claim.status == "INCONCLUSIVE"
    assert report.decision != "PASS"
    assert any("validated judge calibration" in reason for reason in claim.reasons)


def test_a_judge_correction_never_certifies_more_than_its_base_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A diagnostic-only mean has a diagnostic base, so its correction is diagnostic too."""
    _label_everything(tmp_path, monkeypatch)
    plan = load_plan(
        _random_sample_validated_plan(
            tmp_path,
            claim_method="corrected_interval",
            sampling="simple_random_within_judge_stratum",
        )
    )
    # Without direct means the propagated envelope is each mean's primary interval, and a
    # diagnostic-only measure makes it diagnostic.
    plan.outputs.direct_mean_intervals.enabled = False
    plan.outputs.risk_measures.append(
        RiskMeasureSpec(id="mean_diagnostic", type="weighted_mean", diagnostic_only=True)
    )

    report = analyze(plan=plan, logs="ignored")

    rank = {"invalid": 0, "diagnostic": 1, "certifying": 2}
    corrected_rows = 0
    for row in [*report.group_risks, *report.hierarchy_risks]:
        for interval in row.intervals:
            if interval.interval_id == "judge_corrected_validated":
                corrected_rows += 1
                assert rank[interval.eligibility] <= rank[row.eligibility]
    assert corrected_rows >= 2

    def corrected(measure_id: str) -> RiskIntervalResult:
        row = next(item for item in report.group_risks if item.measure_id == measure_id)
        return next(
            item for item in row.intervals if item.interval_id == "judge_corrected_validated"
        )

    diagnostic = corrected("mean_diagnostic")
    assert diagnostic.eligibility == "diagnostic"
    assert diagnostic.role == "sensitivity"
    assert diagnostic.simultaneous_coverage is None
    (warning,) = diagnostic.warnings
    assert "the judge-system interval it corrects is diagnostic" in warning
    # The same mean without the flag is a whole-cohort, self-weighting row, still certified.
    assert corrected("mean").eligibility == "certifying"


def test_worst_of_k_is_not_judge_correctable() -> None:
    # The affine correction R* = R*rho1 + (1-R)*rho0 commutes only with
    # positively homogeneous, translation-equivariant measures; 1-(1-p)^k is
    # neither, so worst_of_k must stay outside the correction whitelist.
    from inspect_safety_stats.api import _JUDGE_CORRECTABLE_MEASURES

    assert "worst_of_k" not in _JUDGE_CORRECTABLE_MEASURES


def _two_arm_plan(tmp_path: Path, *, as_set: bool) -> Path:
    """Candidate against baseline, as a two-arm comparison set or as the same comparison."""
    plan_path = _write_inputs(tmp_path)
    text = plan_path.read_text(encoding="utf-8")
    text = text.replace(
        "groups:\n",
        "  baseline:\n"
        "    match:\n"
        "      model: model/baseline\n"
        "    require_uniform: [task_version, model_generate_config]\n"
        "groups:\n",
        1,
    )
    shared = (
        "    group: cyber\n"
        "    measure: mean\n"
        "    family: model_comparisons\n"
        "    independence:\n"
        "      primary: rollout\n"
    )
    comparison = (
        "comparison_sets:\n"
        "  - id: pair\n"
        "    cohorts: [candidate, baseline]\n"
        "    contrasts: {type: versus_reference, reference: baseline}\n" + shared
        if as_set
        # The settings a set gives each comparison it expands into.
        else "comparisons:\n"
        "  - id: candidate_vs_baseline\n"
        "    candidate: candidate\n"
        "    baseline: baseline\n"
        "    estimand: risk_difference\n"
        "    method: weighted_kl_chernoff\n"
        "    hierarchy_nodes: true\n"
        "    hierarchy_calibration: frontier_sidak\n" + shared
    )
    text = text.replace("multiplicity:\n", comparison + "multiplicity:\n", 1).replace(
        "    - {id: displayed,",
        "    - {id: model_comparisons, purpose: simultaneous_intervals, "
        "method: bonferroni, alpha: 0.05}\n"
        "    - {id: displayed,",
        1,
    )
    plan_path.write_text(text, encoding="utf-8")
    return plan_path


def test_a_two_arm_comparison_set_matches_the_hand_written_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_frames(monkeypatch, _two_cohort_frames())
    results = {}
    for as_set in (True, False):
        directory = tmp_path / ("set" if as_set else "comparison")
        directory.mkdir()
        report = analyze(plan=_two_arm_plan(directory, as_set=as_set), logs="ignored")
        results[as_set] = {
            (row.level, tuple(row.path.items())): (row.estimate, row.lower, row.upper)
            for row in report.comparisons
        }
    # The root and every hierarchy node, with identical estimates and bounds.
    assert len(results[True]) > 1
    _, root_lower, root_upper = results[True][(None, ())]
    assert root_lower is not None and root_upper is not None
    assert results[True] == results[False]


def test_a_report_names_its_inputs_relative_to_the_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#63: absolute paths made reports machine-specific and revealed the local layout."""
    plan_path = _write_inputs(tmp_path)
    eval_frame, sample_frame = _frames()
    log = tmp_path / "logs" / "run.eval"
    eval_frame = eval_frame.assign(log=str(log))
    sample_frame = sample_frame.assign(log=str(log))
    _patch_frames(monkeypatch, (eval_frame, sample_frame))

    report = analyze(plan=plan_path, logs="ignored")

    assert [item["path"] for item in report.provenance["sources"]] == ["logs/run.eval"]
    assert {observation.source for observation in report._observations} == {"logs/run.eval"}
    assert report.design["manifest"] == "design.csv"
    assert report._resolved_plan["inputs"]["design_manifest"] == "design.csv"
    # No written file names the checkout, in any escaping.
    written = report.write(tmp_path / "report")
    for path in sorted(written.rglob("*")):
        if path.is_file():
            assert tmp_path.name not in path.read_text(encoding="utf-8", errors="replace"), path
