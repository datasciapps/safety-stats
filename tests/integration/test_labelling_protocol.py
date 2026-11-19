"""A plan's labelling protocol: the queue, its order and the stopping rule (roadmap F41, L50).

One prompt, 400 rollouts of one cohort: 80 judge-unsafe, of which 60 are truly unsafe, and
320 judge-safe, of which 5 are truly unsafe. The queue alternates the two judge strata, each
in a fixed shuffled order, and puts a row of an `oversample` tier after every fourth
calibration row. The labels are the queue's first calibration rows unless a test breaks that.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.schema.reports import RiskIntervalResult, SafetyReport
from inspect_safety_stats.stats import misclassification
from inspect_safety_stats.stats.misclassification import JudgeStratumCounts

ROLLOUTS = 400
JUDGE_UNSAFE = 80
JUDGE = """  judge_calibration:
    mode: validated
    validation_manifest: labels.csv
    family: judge_error
"""

PLAN = """
schema_version: "2.0"
analysis: {id: protocol, title: Labelling protocol, intent: confirmatory,
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
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
cohorts:
  candidate: {match: {model: model/candidate}, require_uniform: [task_version]}
groups:
  - id: all
    filter: {suite: s}
outputs:
  prompt_bounds: {enabled: true, simultaneous: true, family: displayed,
                  alpha_allocation: equal_bonferroni}
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    family: direct_means
    groups: [all]
    simultaneous: true
    coverage_mode: two_sided
    independence: {primary: rollout}
  risk_measures:
    - {id: mean, type: weighted_mean}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: judge_error, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - {id: mean_all, title: All, cohort: candidate, group: all, measure: mean, family: gate,
     acceptable_upper: 0.25, unacceptable_lower: 0.50}
policy: {zero_tolerance: {enabled: true, manifest_field: zero_tolerance}}
report: {include_observations: true, decimal_places: 6}
"""


def _truly_unsafe(epoch: int) -> bool:
    return epoch <= 60 or JUDGE_UNSAFE < epoch <= JUDGE_UNSAFE + 5


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> None:
    evals = pd.DataFrame(
        [
            {
                "eval_id": "eval-candidate",
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
                "epochs": ROLLOUTS,
            }
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"candidate-{epoch}",
                "eval_id": "eval-candidate",
                "log": "/not/a/local/log.eval",
                "id": "p1",
                "epoch": epoch,
                "score_harmful": "FAIL" if epoch <= JUDGE_UNSAFE else "PASS",
                "error": None,
                "retries": 0,
            }
            for epoch in range(1, ROLLOUTS + 1)
        ]
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, **_: samples,
    )


def _observation_ids(directory: Path) -> dict[int, str]:
    """Each epoch's observation identifier, as an analysis without a protocol reports it."""

    plan = _write_plan(directory / "probe", None)
    report = analyze(plan=load_plan(plan), logs="ignored")
    return {
        int(item.rollout_id.rsplit("-", 1)[1]): item.observation_id
        for item in report._observations
    }


def _queue(ids: dict[int, str]) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    unsafe = [ids[e] for e in rng.permutation(np.arange(1, JUDGE_UNSAFE + 1))]
    safe = [ids[e] for e in rng.permutation(np.arange(JUDGE_UNSAFE + 1, ROLLOUTS + 1))]
    rows: list[dict[str, str]] = []
    # Calibration rows take each stratum's order from the front, oversample rows from the
    # back, so the tiers never meet.
    for slot in range(60):
        rows.append({"observation_id": (unsafe if slot % 2 else safe)[slot // 2],
                     "tier": "calibration"})
        if slot % 4 == 3:
            rows.append({"observation_id": safe[-1 - slot // 4], "tier": "oversample"})
    return pd.DataFrame(rows)


def _write_plan(directory: Path, protocol: str | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite,expected_rollouts,zero_tolerance\n"
        f"p1,q1,t1,s,{ROLLOUTS},false\n",
        encoding="utf-8",
    )
    # Without a protocol the plan has no calibration either: it only finds the identifiers.
    text = PLAN.replace("@JUDGE@\n", "" if protocol is None else JUDGE + protocol)
    plan = directory / "plan.yaml"
    plan.write_text(text.strip() + "\n", encoding="utf-8")
    return plan


def _labels(directory: Path, ids: dict[int, str], observation_ids: list[str]) -> None:
    truth = {value: _truly_unsafe(epoch) for epoch, value in ids.items()}
    rows = ["observation_id,human_label"] + [
        f"{key},{'unsafe' if truth[key] else 'safe'}" for key in observation_ids
    ]
    (directory / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _protocol(stopping: str, **extra: object) -> str:
    lines = [
        "    protocol:",
        "      queue: queue.csv",
        "      tier: calibration",
        "      seed: 41",
        f"      stopping: {stopping}",
        *(f"      {key}: {value}" for key, value in extra.items()),
    ]
    return "\n".join(lines) + "\n"


def _run(
    tmp_path: Path, protocol: str, take: int = 40, edit: str = "none"
) -> tuple[SafetyReport, pd.DataFrame]:
    ids = _observation_ids(tmp_path)
    queue = _queue(ids)
    queue.to_csv(tmp_path / "queue.csv", index=False, lineterminator="\n")
    calibration = list(queue.loc[queue["tier"] == "calibration", "observation_id"])
    labelled = calibration[:take]
    if edit == "skip":
        labelled = calibration[: take - 2] + calibration[take - 1 : take + 1]
    elif edit == "oversample":
        labelled = labelled + [queue.loc[queue["tier"] == "oversample", "observation_id"].iloc[0]]
    _labels(tmp_path, ids, labelled)
    plan = _write_plan(tmp_path, protocol)
    return analyze(plan=load_plan(plan), logs="ignored"), queue


def _corrected(report: SafetyReport) -> RiskIntervalResult | None:
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    return next(
        (item for item in mean.intervals if item.interval_id == "judge_corrected_validated"),
        None,
    )


def _violations(report: SafetyReport) -> list[str]:
    return [
        item.message
        for item in report.integrity.diagnostics
        if item.code == "JUDGE_LABELLING_PROTOCOL_VIOLATED"
    ]


@pytest.mark.usefixtures("logs")
def test_a_labelled_prefix_is_recorded_and_keeps_clopper_pearson(tmp_path: Path) -> None:
    report, _ = _run(tmp_path, _protocol("label_independent"))

    assert _violations(report) == []
    record = report.provenance["judge_calibration"]["protocol"]
    queue_bytes = (tmp_path / "queue.csv").read_bytes()
    assert record == {
        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "queue_rows": 60,
        "labelled_prefix": 40,
        "tier": "calibration",
        "seed": 41,
        "stopping": "label_independent",
        "size": None,
        "bound": "clopper_pearson",
    }
    corrected = _corrected(report)
    assert corrected is not None and corrected.eligibility == "certifying"
    assert any("exact Clopper-Pearson" in text for text in corrected.assumptions)


@pytest.mark.usefixtures("logs")
def test_a_skipped_queue_row_stops_the_calibration(tmp_path: Path) -> None:
    report, queue = _run(tmp_path, _protocol("label_independent"), edit="skip")

    (message,) = _violations(report)
    skipped = queue.loc[queue["tier"] == "calibration", "observation_id"].iloc[38]
    assert f"row 39 ({skipped!r}) has no label, but row 41 does" in message
    assert _corrected(report) is None
    assert "protocol" not in report.provenance.get("judge_calibration", {})


@pytest.mark.usefixtures("logs")
def test_a_label_outside_the_declared_tier_stops_the_calibration(tmp_path: Path) -> None:
    report, _ = _run(tmp_path, _protocol("label_independent"), edit="oversample")

    (message,) = _violations(report)
    assert "1 labelled observations are not in the labelling queue tier 'calibration'" in (
        message
    )
    assert _corrected(report) is None


@pytest.mark.usefixtures("logs")
@pytest.mark.parametrize(("size", "accepted"), [(40, True), (50, False)])
def test_a_fixed_size_must_be_the_number_labelled(
    tmp_path: Path, size: int, accepted: bool
) -> None:
    report, _ = _run(tmp_path, _protocol("fixed_size", size=size))

    if accepted:
        assert _violations(report) == []
        assert _corrected(report) is not None
    else:
        assert _violations(report) == [
            "the labelling protocol fixes 50 labels in advance, but the validation manifest "
            "has 40"
        ]
        assert _corrected(report) is None


@pytest.mark.usefixtures("logs")
def test_data_dependent_stopping_bounds_each_stratum_with_the_confidence_sequence(
    tmp_path: Path,
) -> None:
    report, _ = _run(tmp_path, _protocol("data_dependent"), take=60)

    summary = report.provenance["judge_calibration"]
    assert summary["protocol"]["bound"] == "confidence_sequence"
    strata = summary["cohorts"]["candidate"]
    endpoint_alpha = summary["per_cohort_alpha"] / 4
    expected = [
        misclassification.judge_accuracy_confidence_sequence(
            JudgeStratumCounts(
                strata[name]["labelled"],
                strata[name]["truly_unsafe"],
                strata[name]["stratum_size"],
            ),
            alpha=endpoint_alpha,
        )
        for name in ("judge_unsafe", "judge_safe")
    ]
    corrected = _corrected(report)
    assert corrected is not None and corrected.eligibility == "certifying"
    text = "\n".join(corrected.assumptions)
    rho_1, rho_0 = expected
    assert f"[{rho_1[0]:.6g}, {rho_1[1]:.6g}]" in text
    assert f"[{rho_0[0]:.6g}, {rho_0[1]:.6g}]" in text
    assert "anytime-valid binomial confidence sequence" in text
    assert "labelling may have stopped because of the labels seen" in text
    # The sequence bounds the judge's accuracy, which the stratum's rollouts only sample, not
    # the share of them that is truly unsafe.
    assert "even a census leaves the probability uncertain" in text
    assert "hypergeometric" not in text
    assert "Clopper-Pearson" not in text


@pytest.mark.usefixtures("logs")
def test_the_queue_path_is_plan_relative_in_the_report_and_the_hash(tmp_path: Path) -> None:
    report, _ = _run(tmp_path / "a", _protocol("label_independent"))

    assert report._resolved_plan["inputs"]["judge_calibration"]["protocol"]["queue"] == "queue.csv"
    plan = tmp_path / "a" / "plan.yaml"
    moved = tmp_path / "b" / "c"
    moved.mkdir(parents=True)
    (moved / "plan.yaml").write_bytes(plan.read_bytes())
    assert canonical_plan_hash(load_plan(plan)) == canonical_plan_hash(
        load_plan(moved / "plan.yaml")
    )
