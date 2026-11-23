"""Every interval and estimation diagnostic code is named in a test that triggers it (#171).

The codes are the ones ``analyze`` raises while it estimates and bounds risk: the alpha
allocation, the judge validation labels and calibration, the panel correction, the hierarchy
and comparison intervals, their simultaneous regions, and paired comparisons.

Most run ``analyze`` end to end on offline rollout files and need nothing but the input. The
codes that guard a numeric engine (the simultaneous regions, the interval and calibration
engines) cannot be reached by input that validates: the sample proportions always satisfy the
raw intervals, and the engines accept every well-formed block. Their tests monkeypatch the
engine to fail and check that ``analyze`` reports the failure and degrades as designed; each
says so in its docstring.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from safety_stats import analyze
from safety_stats.analysis import (
    comparison_intervals,
    comparison_regions,
    comparisons,
    regions,
    risks,
)
from safety_stats.ingest.rollout_files import RolloutFiles
from safety_stats.schema.reports import Diagnostic, RiskIntervalResult, SafetyReport
from safety_stats.stats.hierarchy_region import HierarchyRegionProjection

PLAN = """
schema_version: "3.0"
analysis: {id: interval-codes, title: Interval codes, intent: confirmatory,
           plan_frozen_at: "2020-01-01T00:00:00Z"}
inputs:
  design_manifest: design.csv
  score:
    name: @SCORE@
    mapping: {failure_values: [@FAIL@], safe_values: [@SAFE@]}
    missing: conservative_bounds
    errors: block
@INPUTS@
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  require_complete_manifest: true
  require_expected_rollouts: true
  weights: {strategy: nested_equal, hierarchy: [task_id, question_id, prompt_id]}
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
cohorts:
  candidate: {match: {model: m/candidate}}
@BASELINES@
groups:
  - id: all
outputs:
  prompt_bounds: {enabled: true, simultaneous: true, family: displayed,
                  alpha_allocation: @ALLOCATION@}
@DIRECT@
  risk_measures:
    - {id: mean, type: weighted_mean}
@COMPARISONS@
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: direct, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: compare, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: judge_error, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: claims, purpose: all_mandatory, method: bonferroni, alpha: 0.05}
claims:
  - {id: low, title: Mean failure rate is low, cohort: candidate, group: all, measure: mean,
     family: claims, acceptable_upper: 0.1, unacceptable_lower: 0.3}
"""

DIRECT_NODES = """  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    family: direct
    groups: [all]
    independence: {primary: rollout}
    hierarchy_nodes: true
"""

VALIDATED_JUDGE = """  judge_calibration:
    mode: validated
    validation_manifest: labels.csv
    family: judge_error
"""

PANEL = """  judges:
    - id: judge_a
      score: {name: harmful_a, mapping: {failure_values: [FAIL], safe_values: [PASS]}}
      block: openai
      accuracy: {sensitivity: {lower: 0.8, upper: 0.95}, specificity: {lower: 0.9, upper: 0.99}}
    - id: judge_b
      score: {name: harmful_b, mapping: {failure_values: [FAIL], safe_values: [PASS]}}
      block: openai
      accuracy: {sensitivity: {lower: 0.7, upper: 0.9}, specificity: {lower: 0.92, upper: 0.99}}
    - id: judge_c
      score: {name: harmful_c, mapping: {failure_values: [FAIL], safe_values: [PASS]}}
      block: other
      accuracy: {sensitivity: {lower: 0.7, upper: 0.9}, specificity: {lower: 0.92, upper: 0.99}}
  composites:
    - {id: panel, rule: @RULE@, judges: [judge_a, judge_b, judge_c]}
"""

TWO_PROMPTS = (("p1", "q1"), ("p2", "q2"))
NOISE = {"PLAN_SELF_DECLARED", "START_TIME_UNOBSERVED"}


def _first_repeat_fails(arm: str, prompt: str, repeat: int) -> str:
    return "FAIL" if repeat == 1 else "PASS"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _study(
    tmp_path: Path,
    *,
    inputs: str = "",
    arms: tuple[str, ...] = ("candidate",),
    allocation: str = "equal_bonferroni",
    direct: str = "",
    comparisons: str = "",
    prompts: tuple[tuple[str, str], ...] = TWO_PROMPTS,
    repeats: int = 4,
    score: Callable[[str, str, int], str] = _first_repeat_fails,
    primary_score: tuple[str, str, str] = ("harmful", '"FAIL"', '"PASS"'),
    weight_column: list[str] | None = None,
    baseline_repeats: tuple[int, ...] | None = None,
    extra_scores: dict[str, str] | None = None,
) -> tuple[Path, RolloutFiles]:
    """Write a plan, manifest, runs and rollouts: every prompt is run ``repeats`` times per arm."""

    name, fail, safe = primary_score
    text = PLAN.replace("@SCORE@", name).replace("@FAIL@", fail).replace("@SAFE@", safe)
    text = text.replace("@INPUTS@", inputs).replace("@ALLOCATION@", allocation)
    text = text.replace("@DIRECT@", direct).replace("@COMPARISONS@", comparisons)
    baselines = "".join(
        f"  {arm}: {{match: {{model: m/{arm}}}}}\n" for arm in arms if arm != "candidate"
    )
    text = text.replace("@BASELINES@\n", baselines)
    if weight_column is not None:
        text = text.replace(
            "weights: {strategy: nested_equal, hierarchy: [task_id, question_id, prompt_id]}",
            "weights: {strategy: explicit, column: w}",
        )
    _write(tmp_path / "plan.yaml", text)
    header = "prompt_id,question_id,task_id,expected_rollouts" + (",w" if weight_column else "")
    rows = [
        f"{prompt},{question},t1,{repeats}" + (f",{weight_column[i]}" if weight_column else "")
        for i, (prompt, question) in enumerate(prompts)
    ]
    _write(tmp_path / "design.csv", "\n".join([header, *rows]) + "\n")
    runs = [{"run_key": arm, "model": f"m/{arm}", "status": "success"} for arm in arms]
    _write(tmp_path / "runs.jsonl", "".join(json.dumps(run) + "\n" for run in runs))
    records: list[dict[str, Any]] = []
    for arm in arms:
        numbers = (
            baseline_repeats
            if arm == "baseline" and baseline_repeats is not None
            else tuple(range(1, repeats + 1))
        )
        for prompt, _ in prompts:
            for number in numbers:
                record: dict[str, Any] = {
                    "run_key": arm,
                    "item_id": prompt,
                    "repeat": number,
                    f"score_{name}": score(arm, prompt, number),
                }
                for extra, value in (extra_scores or {}).items():
                    record[f"score_{extra}"] = value if number != 1 else "FAIL"
                records.append(record)
    _write(tmp_path / "rollouts.jsonl", "".join(json.dumps(record) + "\n" for record in records))
    source = RolloutFiles(runs=tmp_path / "runs.jsonl", rollouts=tmp_path / "rollouts.jsonl")
    return tmp_path / "plan.yaml", source


def _run(tmp_path: Path, **kwargs: Any) -> SafetyReport:
    plan, source = _study(tmp_path, **kwargs)
    return analyze(plan=plan, source=source)


def _raised(report: SafetyReport, code: str) -> list[Diagnostic]:
    return [item for item in report.integrity.diagnostics if item.code == code]


def _comparison(
    *,
    estimand: str = "risk_difference",
    method: str = "weighted_kl_chernoff",
    extra: str = "",
) -> str:
    return f"""comparisons:
  - id: cvb
    candidate: candidate
    baseline: baseline
    group: all
    measure: mean
    estimand: {estimand}
    method: {method}
    family: compare
    independence: {{primary: rollout}}
{extra}"""


def _intervals(report: SafetyReport, *, measure: str = "mean") -> list[RiskIntervalResult]:
    (row,) = [item for item in report.group_risks if item.measure_id == measure]
    return row.intervals


# --- ALPHA_ALLOCATION_INVALID -------------------------------------------------------------


@pytest.mark.parametrize(
    "allocation",
    [
        pytest.param("explicit, explicit_alpha: {p1: 0.01}", id="missing-a-prompt"),
        pytest.param("explicit, explicit_alpha: {p1: 0.04, p2: 0.04}", id="sum-above-the-family"),
    ],
)
def test_an_explicit_alpha_that_does_not_fit_the_family_is_an_error(
    tmp_path: Path, allocation: str
) -> None:
    report = _run(tmp_path, allocation=allocation)

    (item,) = _raised(report, "ALPHA_ALLOCATION_INVALID")
    assert item.severity == "ERROR"
    assert item.scope == "plan/multiplicity/families/displayed"
    assert "explicit_alpha must contain every manifest prompt exactly once" in item.message
    assert report.integrity.status == "invalid"
    # No allocation, so no prompt carries a bound.
    assert report.prompt_estimates
    assert all(row.lower is None and row.upper is None for row in report.prompt_estimates)


def test_weighted_bonferroni_needs_every_prompt_weight_positive(tmp_path: Path) -> None:
    report = _run(tmp_path, allocation="weighted_bonferroni", weight_column=["0", "1"])

    (item,) = _raised(report, "ALPHA_ALLOCATION_INVALID")
    assert item.severity == "ERROR"
    assert item.scope == "plan/multiplicity/families/displayed"
    assert item.message == "all alpha weights must be positive"
    assert report.integrity.status == "invalid"
    assert all(row.lower is None and row.upper is None for row in report.prompt_estimates)


def test_a_valid_alpha_allocation_raises_nothing(tmp_path: Path) -> None:
    report = _run(tmp_path, allocation="explicit, explicit_alpha: {p1: 0.02, p2: 0.02}")

    assert _raised(report, "ALPHA_ALLOCATION_INVALID") == []
    assert all(row.lower is not None and row.upper is not None for row in report.prompt_estimates)


# --- JUDGE_VALIDATION_INVALID and JUDGE_VALIDATION_STRATUM_EMPTY ---------------------------


def _agreeing_labels(tmp_path: Path, **kwargs: Any) -> str:
    """Human labels that agree with the judge on every classified rollout of the same study."""

    reference = tmp_path / "reference"
    reference.mkdir()
    baseline = _run(reference, **kwargs)
    rows = ["observation_id,human_label"]
    for observation in baseline._observations:
        if observation.failure is not None:
            label = "unsafe" if observation.failure else "safe"
            rows.append(f"{observation.observation_id},{label}")
    return "\n".join(rows) + "\n"


@pytest.mark.parametrize(
    ("labels", "fragment"),
    [
        pytest.param("foo,bar\n1,2\n", "missing required columns", id="wrong-columns"),
        pytest.param(
            "observation_id,human_label\nx,maybe\n", "invalid human_label 'maybe'", id="bad-label"
        ),
        pytest.param(None, "No such file", id="no-file"),
    ],
)
def test_an_unreadable_validation_manifest_is_an_error(
    tmp_path: Path, labels: str | None, fragment: str
) -> None:
    if labels is not None:
        _write(tmp_path / "labels.csv", labels)

    report = _run(tmp_path, inputs=VALIDATED_JUDGE)

    (item,) = _raised(report, "JUDGE_VALIDATION_INVALID")
    assert item.severity == "ERROR"
    assert item.scope == "plan/inputs/judge_calibration"
    assert fragment in item.message
    assert report.integrity.status == "invalid"
    # No calibration, so the claim has no corrected interval to rest on.
    assert [item.interval_id for item in _intervals(report)] == ["propagated_prompt_envelope"]
    assert "JUDGE_CORRECTION_UNAVAILABLE" in {
        item.code for item in report.integrity.diagnostics
    }


@pytest.mark.parametrize("verdict", ["PASS", "FAIL"])
def test_a_judge_that_never_flips_leaves_a_stratum_empty(tmp_path: Path, verdict: str) -> None:
    """With every rollout judged one way, the other judge stratum has no observation."""

    def always(arm: str, prompt: str, repeat: int) -> str:
        return verdict

    _write(tmp_path / "labels.csv", _agreeing_labels(tmp_path, score=always))

    report = _run(tmp_path, inputs=VALIDATED_JUDGE, score=always)

    (item,) = _raised(report, "JUDGE_VALIDATION_STRATUM_EMPTY")
    assert item.severity == "WARNING"
    assert item.scope == "cohorts/candidate"
    assert "has no observations in one judge stratum" in item.message
    assert "no judge correction is reported" in item.message
    assert report.integrity.status == "degraded"
    assert _raised(report, "JUDGE_VALIDATION_INVALID") == []
    assert "judge_corrected_validated" not in {item.interval_id for item in _intervals(report)}


def test_both_judge_strata_populated_raise_no_empty_stratum(tmp_path: Path) -> None:
    _write(tmp_path / "labels.csv", _agreeing_labels(tmp_path))

    report = _run(tmp_path, inputs=VALIDATED_JUDGE)

    assert _raised(report, "JUDGE_VALIDATION_STRATUM_EMPTY") == []


# --- PANEL_CORRECTION_UNSUPPORTED_RULE -----------------------------------------------------


def _panel_report(tmp_path: Path, rule: str) -> SafetyReport:
    return _run(
        tmp_path,
        inputs=PANEL.replace("@RULE@", rule),
        primary_score=("panel", "true", "false"),
        extra_scores={"harmful_a": "PASS", "harmful_b": "PASS", "harmful_c": "PASS"},
    )


def test_a_majority_panel_gets_no_derived_correction_and_says_why(tmp_path: Path) -> None:
    report = _panel_report(tmp_path, "majority")

    (item,) = _raised(report, "PANEL_CORRECTION_UNSUPPORTED_RULE")
    assert item.severity == "INFO"
    assert item.scope == "plan/inputs/composites/panel"
    assert "majority composite has no closed-form accuracy box" in item.message
    assert report.integrity.status == "degraded"
    assert "judge_corrected_panel_derived" not in {item.interval_id for item in _intervals(report)}


def test_an_any_panel_gets_the_derived_correction(tmp_path: Path) -> None:
    report = _panel_report(tmp_path, "any")

    assert _raised(report, "PANEL_CORRECTION_UNSUPPORTED_RULE") == []
    assert "judge_corrected_panel_derived" in {item.interval_id for item in _intervals(report)}


# --- PAIR_BROKEN ---------------------------------------------------------------------------

PAIRED = "    pair_key: [prompt_id, repeat]\n    diagnostic_methods: [exact_mcnemar]\n"


def test_a_pairing_that_matches_every_rollout_raises_nothing(tmp_path: Path) -> None:
    report = _run(
        tmp_path, arms=("candidate", "baseline"), comparisons=_comparison(extra=PAIRED)
    )

    assert _raised(report, "PAIR_BROKEN") == []
    (comparison,) = report.comparisons
    assert comparison.pair_count == 8


def test_a_rollout_without_a_partner_breaks_the_pairing(tmp_path: Path) -> None:
    """The baseline's repeats 4 and 5 differ from the candidate's, so two pairs have no partner."""

    report = _run(
        tmp_path,
        arms=("candidate", "baseline"),
        comparisons=_comparison(extra=PAIRED),
        baseline_repeats=(1, 2, 3, 5),
    )

    (item,) = _raised(report, "PAIR_BROKEN")
    assert item.severity == "ERROR"
    assert item.scope == "comparisons/cvb"
    assert "broken pairing: 2 candidate-only and 2 baseline-only pairs" in item.message
    assert report.integrity.status == "invalid"
    (comparison,) = report.comparisons
    assert comparison.pair_count is None
    assert comparison.discordant_counts == {}
    assert comparison.diagnostic_outputs == {}
    assert item.message in comparison.warnings


def test_a_duplicate_pair_key_breaks_the_pairing(tmp_path: Path) -> None:
    """Pairing on the prompt alone gives four rollouts per key in each arm."""

    report = _run(
        tmp_path,
        arms=("candidate", "baseline"),
        comparisons=_comparison(extra=PAIRED.replace("[prompt_id, repeat]", "[prompt_id]")),
    )

    (item,) = _raised(report, "PAIR_BROKEN")
    assert item.severity == "ERROR"
    assert "duplicate pair key in candidate: ('p1',)" in item.message
    assert report.integrity.status == "invalid"


# --- The handling path of engine failures --------------------------------------------------


def _infeasible(*_: Any, **__: Any) -> HierarchyRegionProjection:
    return HierarchyRegionProjection(
        feasible=False, intervals={}, constraint_count=6, message="simulated empty region"
    )


def test_a_node_interval_engine_failure_is_reported_for_that_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Handling path: no input makes the engine raise, so a single-prompt node's call fails.

    The engine accepts every well-formed block, so the node intervals are made to raise
    ``ValueError`` for single-prompt nodes. Node rows then carry the message and no interval,
    while the group row, which the engine still serves, keeps its own.
    """

    real = risks._direct_intervals

    def failing(**kwargs: Any) -> list[RiskIntervalResult]:
        if len(kwargs["prompt_ids"]) == 1:
            raise ValueError("simulated engine failure")
        return real(**kwargs)

    monkeypatch.setattr(risks, "_direct_intervals", failing)

    report = _run(tmp_path, direct=DIRECT_NODES)

    raised = _raised(report, "HIERARCHY_MEAN_INTERVAL_INVALID")
    # One per single-prompt node: the two questions of the group.
    node = "cohorts/candidate/groups/all/levels/question_id/nodes/task_id=t1%2Fquestion_id="
    assert [item.scope for item in raised] == [f"{node}q1", f"{node}q2"]
    assert all(item.severity == "ERROR" for item in raised)
    assert all(item.message == "simulated engine failure" for item in raised)
    assert report.integrity.status == "invalid"
    singles = [row for row in report.hierarchy_risks if row.prompt_count == 1]
    assert singles
    assert all(row.intervals == [] and row.lower is None for row in singles)
    assert all("simulated engine failure" in row.warnings for row in singles)
    assert _raised(report, "DIRECT_MEAN_INTERVAL_INVALID") == []
    assert [item.role for item in _intervals(report)].count("primary") == 1


def test_an_empty_hierarchy_region_keeps_the_raw_node_intervals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Handling path: the sample proportions always satisfy the raw node intervals.

    The raw simultaneous intervals all contain the observed proportions, so the region is never
    empty for real input. The projection is made to report an empty region.
    """

    monkeypatch.setattr(regions, "project_hierarchy_confidence_region", _infeasible)

    report = _run(tmp_path, direct=DIRECT_NODES)

    (item,) = _raised(report, "HIERARCHY_CONFIDENCE_REGION_EMPTY")
    assert item.severity == "WARNING"
    assert item.scope == "cohorts/candidate/methods/weighted_kl_chernoff/units/rollout"
    assert item.details == {"solver_message": "simulated empty region", "constraint_count": 6}
    assert "no common parameter vector" in item.message
    assert report.integrity.status == "degraded"
    # The intervals stay as the raw bounds: nothing was projected.
    primary = next(row for row in _intervals(report) if row.role == "primary")
    assert primary.projection_method is None
    assert primary.raw_lower is None


def test_an_empty_comparison_region_keeps_the_raw_comparison_intervals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Handling path: the raw comparison intervals always contain the observed difference.

    The two-arm difference projection is made to report an empty region.
    """

    monkeypatch.setattr(
        comparison_regions, "project_two_arm_difference_hierarchy_confidence_region", _infeasible
    )
    nodes = "    hierarchy_nodes: true\n    hierarchy_calibration: frontier_sidak\n"

    report = _run(
        tmp_path, arms=("candidate", "baseline"), comparisons=_comparison(extra=nodes)
    )

    (item,) = _raised(report, "COMPARISON_HIERARCHY_CONFIDENCE_REGION_EMPTY")
    assert item.severity == "WARNING"
    assert item.scope == "comparisons/cvb/methods/weighted_kl_chernoff/units/rollout"
    assert item.details == {"solver_message": "simulated empty region", "constraint_count": 6}
    assert "no common parameter vector under the declared hierarchy" in item.message
    assert report.integrity.status == "degraded"
    assert report.comparisons
    assert all(
        row.projection_method is None and row.raw_lower is None for row in report.comparisons
    )
    assert all(row.lower is not None and row.upper is not None for row in report.comparisons)


def test_an_empty_multi_arm_region_leaves_the_set_unprojected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Handling path: each arm's raw intervals contain the arm's observed proportions.

    The arm projection is made to report an empty region, which no valid input yields.
    """

    monkeypatch.setattr(comparison_regions, "project_hierarchy_confidence_region", _infeasible)
    comparison_set = """comparison_sets:
  - id: arms
    cohorts: [baseline, candidate, variant]
    contrasts: {type: all_pairs}
    estimands: [risk_difference]
    group: all
    measure: mean
    method: weighted_kl_chernoff
    family: compare
    independence: {primary: rollout}
    hierarchy_nodes: true
    hierarchy_calibration: frontier_sidak
"""

    def score(arm: str, prompt: str, repeat: int) -> str:
        return "FAIL" if repeat <= {"candidate": 1, "baseline": 2, "variant": 4}[arm] else "PASS"

    report = _run(
        tmp_path,
        arms=("candidate", "baseline", "variant"),
        comparisons=comparison_set,
        score=score,
    )

    raised = _raised(report, "MULTI_ARM_CONFIDENCE_REGION_EMPTY")
    assert raised
    assert all(item.severity == "WARNING" for item in raised)
    assert all(item.scope.startswith("comparison_sets/arms/") for item in raised)
    assert all(item.scope.endswith("/profiles/rollout") for item in raised)
    fragment = "no common parameter vector under the declared hierarchy"
    assert all(fragment in item.message for item in raised)
    assert report.integrity.status == "degraded"
    roots = [row for row in report.comparisons if row.level is None]
    assert len(roots) == 3
    assert "multi_arm_hierarchy_linear_region" not in {
        row.projection_method for row in report.comparisons
    }
    assert all(row.lower is not None for row in roots)


def test_a_comparison_calibration_failure_is_reported_for_its_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Handling path: the calibration refuses frontiers that overlap or are empty.

    The planner builds disjoint, non-empty frontiers from any manifest, so the calibration is
    made to raise ``ValueError``. The comparison is then left without a calibrated interval.
    """

    def failing(*_: Any, **__: Any) -> dict[str, Any]:
        raise ValueError("frontier 'f' targets 'a' and 'b' are not disjoint")

    monkeypatch.setattr(comparison_intervals, "calibrate_independent_frontiers", failing)

    report = _run(tmp_path, arms=("candidate", "baseline"), comparisons=_comparison())

    (item,) = _raised(report, "COMPARISON_CALIBRATION_INVALID")
    assert item.severity == "ERROR"
    assert item.scope == "families/compare/profiles/rollout"
    assert "are not disjoint" in item.message
    assert report.integrity.status == "invalid"
    (comparison,) = report.comparisons
    assert comparison.intervals == []
    assert comparison.lower is None and comparison.upper is None
    assert any("Calibration is unavailable" in warning for warning in comparison.warnings)


def test_a_comparison_interval_engine_failure_is_reported_for_that_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Handling path: the interval engine accepts every pair of well-formed blocks.

    The difference engine is made to raise ``ValueError``; the comparison keeps no interval.
    """

    def failing(*_: Any, **__: Any) -> Any:
        raise ValueError("simulated engine failure")

    monkeypatch.setattr(comparisons, "_difference_from_blocks", failing)

    report = _run(
        tmp_path,
        arms=("candidate", "baseline"),
        comparisons=_comparison(method="weighted_hoeffding"),
    )

    (item,) = _raised(report, "COMPARISON_INTERVAL_INVALID")
    assert item.severity == "ERROR"
    assert item.scope == "comparisons/cvb/methods/weighted_hoeffding/profiles/rollout"
    assert item.message == "simulated engine failure"
    assert report.integrity.status == "invalid"
    (comparison,) = report.comparisons
    assert comparison.intervals == []
    assert comparison.lower is None and comparison.upper is None
    assert "simulated engine failure" in comparison.warnings
