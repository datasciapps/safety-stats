"""Lang-Reiczigel limits on the rows a validated calibration corrects, end to end.

One cohort, three prompts: p1 and p2 are cyber prompts under one task, p3 a bio prompt under
another. Nested-equal weights give p1 and p2 a quarter of the cohort's weight each and p3
half, and the prompts need not have the same number of rollouts, so a row can weight its
rollouts unevenly. The labels are a simple random sample of the classified rollouts: unless
a test says otherwise, all of them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.schema.reports import (
    GroupRiskResult,
    HierarchyRiskResult,
    RiskIntervalResult,
    SafetyReport,
)
from inspect_safety_stats.stats.misclassification import (
    TruthStratumCounts,
    lang_reiczigel_corrected_risk,
)

LIMITS = "judge_corrected_validated_lang_reiczigel"
EXACT = "judge_corrected_validated"

# (prompt, question, task, risk domain)
PROMPTS = [("p1", "q1", "t1", "cyber"), ("p2", "q2", "t1", "cyber"), ("p3", "q3", "t2", "bio")]

Verdict = Callable[[str, int], bool]

PLAN = """
schema_version: "2.0"
analysis: {id: lang_reiczigel, title: Lang-Reiczigel rows, intent: confirmatory,
           plan_frozen_at: 2025-01-01T00:00:00Z}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
    missing: conservative_bounds
    errors: block
@JUDGE@
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  rollout_key: [eval_id, id, epoch]
  require_complete_manifest: true
  require_expected_rollouts: true
  weights: {strategy: nested_equal, hierarchy: [task_id, question_id, prompt_id]}
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
cohorts:
  candidate: {match: {model: model/candidate}, require_uniform: [task_version]}
groups:
  - id: all
    filter: {suite: s}
    weights: {strategy: nested_equal, hierarchy: [task_id, question_id, prompt_id]}
  - id: cyber
    filter: {risk_domain: cyber}
    weights: {strategy: nested_equal, hierarchy: [task_id, question_id, prompt_id]}
outputs:
  prompt_bounds: {enabled: true, simultaneous: true, family: displayed,
                  alpha_allocation: equal_bonferroni}
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    family: direct_means
    groups: [all, cyber]
    simultaneous: true
    coverage_mode: two_sided
    independence: {primary: rollout}
    hierarchy_nodes: true
  risk_measures:
    - {id: mean, type: weighted_mean}
    - {id: worst, type: maximum}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: judge_error, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - {id: mean_all, title: All, cohort: candidate, group: all, measure: mean, family: gate,
     acceptable_upper: 0.30, unacceptable_lower: 0.60}
  - {id: mean_cyber, title: Cyber, cohort: candidate, group: cyber, measure: mean,
     family: gate, acceptable_upper: 0.30, unacceptable_lower: 0.60}
report: {include_observations: true, decimal_places: 6}
"""


def _write_plan(directory: Path, rollouts: dict[str, int], judge: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,risk_domain,suite,expected_rollouts\n"
        + "".join(f"{p},{q},{t},{d},s,{rollouts[p]}\n" for p, q, t, d in PROMPTS),
        encoding="utf-8",
    )
    plan = directory / "plan.yaml"
    plan.write_text(PLAN.replace("@JUDGE@\n", judge).strip() + "\n", encoding="utf-8")
    return plan


def _analyze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rollouts: dict[str, int],
    judge: Verdict,
    truth: Verdict,
    label: Verdict = lambda prompt_id, epoch: True,
    missing: Verdict = lambda prompt_id, epoch: False,
    judge_groups: str | None = None,
) -> SafetyReport:
    """Label the classified rollouts ``label`` picks with ``truth``, then analyse.

    ``judge`` gives the judge's verdict, true for a reported failure, and ``missing`` the
    rollouts it left unscored.
    """

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
                "epochs": max(rollouts.values()),
            }
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"{prompt_id}-{epoch}",
                "eval_id": "eval-1",
                "log": "/not/a/local/log.eval",
                "id": prompt_id,
                "epoch": epoch,
                "score_harmful": None
                if missing(prompt_id, epoch)
                else ("FAIL" if judge(prompt_id, epoch) else "PASS"),
                "error": None,
                "retries": 0,
            }
            for prompt_id, *_ in PROMPTS
            for epoch in range(1, rollouts[prompt_id] + 1)
        ]
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )
    unlabelled = analyze(
        plan=load_plan(_write_plan(tmp_path / "unlabelled", rollouts, "")), logs="ignored"
    )
    rows = ["observation_id,human_label"]
    for observation in unlabelled._observations:
        epoch = int(observation.rollout_id.rsplit("-", 1)[1])
        if observation.failure is None or not label(observation.prompt_id, epoch):
            continue
        unsafe = truth(observation.prompt_id, epoch)
        rows.append(f"{observation.observation_id},{'unsafe' if unsafe else 'safe'}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    judge_block = (
        "  judge_calibration:\n"
        "    mode: validated\n"
        "    validation_manifest: labels.csv\n"
        "    family: judge_error\n"
        "    sampling: simple_random\n"
        + (f"    groups: {judge_groups}\n" if judge_groups else "")
    )
    return analyze(plan=load_plan(_write_plan(tmp_path, rollouts, judge_block)), logs="ignored")


def _row(report: SafetyReport, group: str, measure: str = "mean") -> GroupRiskResult:
    return next(
        row for row in report.group_risks if row.group_id == group and row.measure_id == measure
    )


def _interval(row: GroupRiskResult, interval_id: str) -> RiskIntervalResult:
    return next(item for item in row.intervals if item.interval_id == interval_id)


def _has_limits(row: GroupRiskResult | HierarchyRiskResult) -> bool:
    return any(item.interval_id == LIMITS for item in row.intervals)


# Uneven rows: p1 and p2 have 20 rollouts each, p3 has 10. The judge reports p1's first ten
# rollouts and p3's first two as failures. It misses p2's first rollout, which is truly unsafe,
# and p3's second rollout is a false positive.
UNEVEN = {"p1": 20, "p2": 20, "p3": 10}


def _uneven_judge(prompt_id: str, epoch: int) -> bool:
    return (prompt_id == "p1" and epoch <= 10) or (prompt_id == "p3" and epoch <= 2)


def _uneven_truth(prompt_id: str, epoch: int) -> bool:
    return (prompt_id == "p1" and epoch <= 10) or (prompt_id in ("p2", "p3") and epoch == 1)


def test_the_limits_use_the_effective_size_of_an_unevenly_weighted_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _analyze(
        tmp_path, monkeypatch, rollouts=UNEVEN, judge=_uneven_judge, truth=_uneven_truth
    )

    # Twelve truly unsafe rollouts, eleven reported, and one false positive among 38 safe ones.
    strata = report.provenance["judge_calibration"]["truth_strata"]["candidate"]
    assert strata == {
        "truly_unsafe": {"labelled": 12, "judge_positive": 11},
        "truly_safe": {"labelled": 38, "judge_positive": 1},
    }
    unsafe, safe = TruthStratumCounts(12, 11), TruthStratumCounts(38, 1)
    # Every weighted-mean group row of the cohort carries limits from the cohort's labels, each
    # over its own judge-reported rate and effective size (sum w)^2 / sum(w^2 / n):
    # all: weights 1/4, 1/4, 1/2 over 20, 20, 10 rollouts, so 1 / (1/320 + 1/320 + 1/40) = 32
    #      of 50 classified, and rate 1/4 * 10/20 + 1/4 * 0 + 1/2 * 2/10 = 0.225;
    # cyber: weights 1/2, 1/2 over 20 rollouts each, so 40, and rate 1/2 * 10/20 = 0.25.
    for group, size, rate in [("all", 32, 0.225), ("cyber", 40, 0.25)]:
        mean = _row(report, group)
        assert mean.estimate == pytest.approx(rate, rel=1e-12, abs=0)
        limits = _interval(mean, LIMITS)
        assert limits.effective_sample_size == pytest.approx(size, rel=1e-12, abs=0)
        assert any(f"over an effective {size} rollouts" in item for item in limits.assumptions)
        expected = lang_reiczigel_corrected_risk(
            apparent_rate=rate,
            apparent_size=size,
            truly_unsafe=unsafe,
            truly_safe=safe,
            alpha=0.05,
        )
        assert limits.lower == pytest.approx(expected.lower, rel=1e-12, abs=0)
        assert limits.upper == pytest.approx(expected.upper, rel=1e-12, abs=0)
        assert limits.completion_estimate_lower == pytest.approx(expected.estimate, abs=1e-15)
        assert limits.family_alpha == 0.05
        assert limits.eligibility == "diagnostic"
        # Neither row weights the cohort's rollouts as the labels do, so neither exact box
        # certifies, and the limits carry the same warning.
        exact = _interval(mean, EXACT)
        assert exact.eligibility == "diagnostic"
        (warning,) = exact.warnings
        assert warning.startswith("Sensitivity analysis, not a certified bound")
        assert limits.warnings == exact.warnings
    assert "times the weight of others" in _interval(_row(report, "all"), LIMITS).warnings[0]
    assert "covers only part" in _interval(_row(report, "cyber"), LIMITS).warnings[0]

    # No tail measure, hierarchy node or claim gets them, though each node has its exact box.
    for row in report.group_risks:
        assert _has_limits(row) == (row.measure_id == "mean"), (row.group_id, row.measure_id)
    assert report.hierarchy_risks
    for node in report.hierarchy_risks:
        assert any(item.interval_id == EXACT for item in node.intervals)
        assert not _has_limits(node)
    for claim in report.claims:
        assert f"interval:{LIMITS}" not in claim.evidence_refs


def test_a_calibration_scoped_to_a_group_gives_its_row_limits_from_its_own_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _analyze(
        tmp_path,
        monkeypatch,
        rollouts=UNEVEN,
        judge=_uneven_judge,
        truth=_uneven_truth,
        judge_groups="[cyber]",
    )

    # The cyber prompts' labels alone: p3's truly unsafe rollout and false positive drop out.
    strata = report.provenance["judge_calibration"]["truth_strata"]["candidate"]["cyber"]
    assert strata == {
        "truly_unsafe": {"labelled": 11, "judge_positive": 10},
        "truly_safe": {"labelled": 29, "judge_positive": 0},
    }
    mean = _row(report, "cyber")
    limits = _interval(mean, LIMITS)
    expected = lang_reiczigel_corrected_risk(
        apparent_rate=0.25,
        apparent_size=40,
        truly_unsafe=TruthStratumCounts(11, 10),
        truly_safe=TruthStratumCounts(29, 0),
        alpha=0.05,
    )
    assert limits.lower == pytest.approx(expected.lower, rel=1e-12, abs=0)
    assert limits.upper == pytest.approx(expected.upper, rel=1e-12, abs=0)
    # One calibration takes the whole judge family alpha.
    assert limits.family_alpha == 0.05
    # The row is the whole group, every rollout weighing the same, so nothing is uncertified.
    assert _interval(mean, EXACT).eligibility == "certifying"
    assert limits.warnings == []
    # The group outside the scope has no calibration, so neither limits nor a reason for none.
    assert not _has_limits(_row(report, "all"))
    assert "JUDGE_LANG_REICZIGEL_UNAVAILABLE" not in {
        item.code for item in report.integrity.diagnostics
    }


def test_each_scoped_group_spends_its_share_of_the_judge_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _analyze(
        tmp_path,
        monkeypatch,
        rollouts=UNEVEN,
        judge=_uneven_judge,
        truth=_uneven_truth,
        judge_groups="[all, cyber]",
    )

    # Two calibrations share the judge family's 0.05, each from its own group's labels.
    for group, size, rate, unsafe, safe in [
        ("all", 32, 0.225, TruthStratumCounts(12, 11), TruthStratumCounts(38, 1)),
        ("cyber", 40, 0.25, TruthStratumCounts(11, 10), TruthStratumCounts(29, 0)),
    ]:
        limits = _interval(_row(report, group), LIMITS)
        assert limits.family_alpha == pytest.approx(0.025, rel=1e-12, abs=0)
        expected = lang_reiczigel_corrected_risk(
            apparent_rate=rate,
            apparent_size=size,
            truly_unsafe=unsafe,
            truly_safe=safe,
            alpha=0.025,
        )
        assert limits.lower == pytest.approx(expected.lower, rel=1e-12, abs=0)
        assert limits.upper == pytest.approx(expected.upper, rel=1e-12, abs=0)


# Many reported failures on the bio prompt, a fifth of them false positives, and few on the
# cyber prompts: 100 rollouts each. The cohort's labels put the false-positive rate near 7%,
# far above the 1% the judge reports on the cyber group.
WIDE = {"p1": 100, "p2": 100, "p3": 100}


def _bio_heavy_truth(prompt_id: str, epoch: int) -> bool:
    return (prompt_id == "p1" and epoch <= 2) or (prompt_id == "p3" and epoch <= 20)


def test_limits_clipped_to_zero_say_the_data_contradict_the_estimated_accuracies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _analyze(
        tmp_path,
        monkeypatch,
        rollouts=WIDE,
        judge=lambda prompt_id, epoch: (
            (prompt_id == "p1" and epoch <= 2) or (prompt_id == "p3" and epoch <= 40)
        ),
        truth=_bio_heavy_truth,
    )

    assert _row(report, "cyber").estimate == pytest.approx(0.01, rel=1e-12, abs=0)
    cyber = _interval(_row(report, "cyber"), LIMITS)
    assert cyber.lower == 0 and cyber.upper == 0
    assert any("clipped to the single point 0" in item for item in cyber.warnings)
    assert any("not evidence of zero risk" in item for item in cyber.warnings)
    # The whole cohort's rate, 0.205, is far above the false-positive rate, so its limits are
    # not clipped.
    assert _row(report, "all").estimate == pytest.approx(0.205, rel=1e-12, abs=0)
    whole = _interval(_row(report, "all"), LIMITS)
    assert whole.upper is not None and whole.upper > 0
    assert not any("clipped" in item for item in whole.warnings)


def test_limits_from_an_insensitive_judge_say_the_method_assumes_better_accuracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The judge reports ten of the 22 truly unsafe rollouts: sensitivity 0.4545, specificity 1.
    report = _analyze(
        tmp_path,
        monkeypatch,
        rollouts=WIDE,
        judge=lambda prompt_id, epoch: (
            (prompt_id == "p1" and epoch <= 2) or (prompt_id == "p3" and epoch <= 8)
        ),
        truth=_bio_heavy_truth,
    )

    for group in ("all", "cyber"):
        limits = _interval(_row(report, group), LIMITS)
        warning = next(item for item in limits.warnings if "below 0.5" in item)
        assert warning.startswith("Estimated sensitivity 0.454545 or specificity 1 is below 0.5")


def test_limits_over_unclassified_rollouts_cover_the_classified_ones_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The judge leaves p2's last rollout unscored, a twentieth of p2's quarter of the weight.
    report = _analyze(
        tmp_path,
        monkeypatch,
        rollouts=UNEVEN,
        judge=_uneven_judge,
        truth=_uneven_truth,
        missing=lambda prompt_id, epoch: prompt_id == "p2" and epoch == 20,
    )

    mean = _row(report, "all")
    limits = _interval(mean, LIMITS)
    # The rate and the effective size count p2's 19 classified rollouts, and the labels the
    # 49 classified rollouts, p2's truly unsafe first one among them.
    assert mean.estimate == pytest.approx(0.225, rel=1e-12, abs=0)
    size = 1 / (0.25**2 / 20 + 0.25**2 / 19 + 0.5**2 / 10)
    assert limits.effective_sample_size == pytest.approx(size, rel=1e-12, abs=0)
    expected = lang_reiczigel_corrected_risk(
        apparent_rate=0.225,
        apparent_size=size,
        truly_unsafe=TruthStratumCounts(12, 11),
        truly_safe=TruthStratumCounts(37, 1),
        alpha=0.05,
    )
    assert limits.lower == pytest.approx(expected.lower, rel=1e-12, abs=0)
    assert limits.upper == pytest.approx(expected.upper, rel=1e-12, abs=0)
    gap, unclassified = limits.warnings
    assert gap == _interval(mean, EXACT).warnings[0]
    assert unclassified == (
        "It covers the classified rollouts only: 0.0125 of the row's weight rests on "
        "unclassified rollouts, which the exact box keeps at their worst case."
    )


@pytest.mark.parametrize(
    ("label", "missing", "reason"),
    [
        # Labels from judge-safe rollouts only: the exact box completes the unlabelled
        # stratum, but the truly unsafe labels show a sensitivity of zero.
        pytest.param(
            lambda prompt_id, epoch: not _uneven_judge(prompt_id, epoch),
            lambda prompt_id, epoch: False,
            "the labels do not establish an informative judge",
            id="uninformative-judge",
        ),
        # The judge scores none of p2's rollouts, so neither row has a judge-reported rate.
        pytest.param(
            lambda prompt_id, epoch: True,
            lambda prompt_id, epoch: prompt_id == "p2",
            "the row has no judge-reported estimate",
            id="unclassified-prompt",
        ),
    ],
)
def test_each_reason_for_missing_limits_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: Verdict,
    missing: Verdict,
    reason: str,
) -> None:
    report = _analyze(
        tmp_path,
        monkeypatch,
        rollouts=UNEVEN,
        judge=_uneven_judge,
        truth=_uneven_truth,
        label=label,
        missing=missing,
    )

    unavailable = [
        item
        for item in report.integrity.diagnostics
        if item.code == "JUDGE_LANG_REICZIGEL_UNAVAILABLE"
    ]
    assert {item.details["group_id"] for item in unavailable} == {"all", "cyber"}
    for item in unavailable:
        assert item.severity == "INFO"
        assert reason in item.message
    for group in ("all", "cyber"):
        mean = _row(report, group)
        assert not _has_limits(mean)
        # The exact correction is unaffected.
        assert any(item.interval_id == EXACT for item in mean.intervals)
