"""The experimental stratified, finite-population validated calibration (L48).

One prompt, 1,000 rollouts. The judge calls epochs 1-100 unsafe and 101-1000 safe. A second
judge disagrees on epochs 1-20 and 101-130, which form the "disagree" fine strata; the rest
are "agree". The truly unsafe rollouts are epochs 1-8, 21-100 and 101-110. The labelling
design labels every disagreement and samples the agreements: epochs 21-40 and 131-230.

Some tests add a second prompt, "p2", of another suite and so outside the group "all". The
judge calls its epochs 1-300 unsafe, epochs 1-250 are truly unsafe, all of them are "agree",
and every tenth is labelled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.stats import binom

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.config.models import JudgeCalibrationSpec
from inspect_safety_stats.generative.schema import (
    JudgeCalibrationSpec as GenerativeJudgeCalibrationSpec,
)
from inspect_safety_stats.schema.reports import RiskIntervalResult, SafetyReport
from inspect_safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    estimate_stratified_judge_calibration,
)

ROLLOUTS = 1000


def _judge_unsafe(epoch: int) -> bool:
    return epoch <= 100


def _stratum(epoch: int) -> str:
    return "disagree" if epoch <= 20 or 101 <= epoch <= 130 else "agree"


def _truly_unsafe(epoch: int) -> bool:
    return epoch <= 8 or 21 <= epoch <= 110


def _labelled(epoch: int) -> bool:
    return _stratum(epoch) == "disagree" or 21 <= epoch <= 40 or 131 <= epoch <= 230


OUTSIDE = "p2"
"""A second prompt, outside the group "all"."""


PLAN = """
schema_version: "2.0"
analysis: {id: stratified, title: Stratified labels, intent: confirmatory,
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

STRATIFIED = (
    "  judge_calibration:\n"
    "    mode: validated\n"
    "    validation_manifest: labels.csv\n"
    "    sampling: stratified_without_replacement\n"
    "    validation_strata: strata.csv\n"
    "    family: judge_error\n"
)


def _frames(*, outside: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                "score_harmful": "FAIL" if _judge_unsafe(epoch) else "PASS",
                "error": None,
                "retries": 0,
            }
            for epoch in range(1, ROLLOUTS + 1)
        ]
        + [
            {
                "sample_id": f"candidate-{OUTSIDE}-{epoch}",
                "eval_id": "eval-candidate",
                "log": "/not/a/local/log.eval",
                "id": OUTSIDE,
                "epoch": epoch,
                "score_harmful": "FAIL" if epoch <= 300 else "PASS",
                "error": None,
                "retries": 0,
            }
            for epoch in (range(1, ROLLOUTS + 1) if outside else ())
        ]
    )
    return evals, samples


def _write_plan(directory: Path, judge: str, *, outside: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite,expected_rollouts,zero_tolerance\n"
        f"p1,q1,t1,s,{ROLLOUTS},false\n"
        + (f"{OUTSIDE},q2,t1,t,{ROLLOUTS},false\n" if outside else ""),
        encoding="utf-8",
    )
    plan = directory / "plan.yaml"
    plan.write_text(PLAN.replace("@JUDGE@\n", judge).strip() + "\n", encoding="utf-8")
    return plan


@pytest.fixture
def observations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[int, str]:
    """Each epoch's observation identifier, from an analysis without labels."""

    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    baseline = analyze(plan=load_plan(_write_plan(tmp_path / "unlabelled", "")), logs="ignored")
    return {
        int(item.rollout_id.rsplit("-", 1)[1]): item.observation_id
        for item in baseline._observations
    }


def _analyze(
    directory: Path,
    observations: dict[int, str],
    *,
    labelled: set[int] | None = None,
    strata: dict[int, str] | None = None,
) -> SafetyReport:
    labelled = labelled if labelled is not None else {e for e in observations if _labelled(e)}
    strata = strata if strata is not None else {e: _stratum(e) for e in observations}
    plan = _write_plan(directory, STRATIFIED)
    (directory / "labels.csv").write_text(
        "observation_id,human_label\n"
        + "".join(
            f"{observations[e]},{'unsafe' if _truly_unsafe(e) else 'safe'}\n"
            for e in sorted(labelled)
        ),
        encoding="utf-8",
    )
    (directory / "strata.csv").write_text(
        "observation_id,stratum\n"
        + "".join(f"{observations[e]},{stratum}\n" for e, stratum in sorted(strata.items())),
        encoding="utf-8",
    )
    return analyze(plan=load_plan(plan), logs="ignored")


def _corrected(report: SafetyReport) -> RiskIntervalResult | None:
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    return next(
        (item for item in mean.intervals if item.interval_id.startswith("judge_corrected_")),
        None,
    )


def _codes(report: SafetyReport) -> dict[str, str]:
    return {item.code: item.severity for item in report.integrity.diagnostics}


def test_stratified_labels_certify_a_correction_weighted_by_fine_stratum(
    tmp_path: Path, observations: dict[int, str]
) -> None:
    report = _analyze(tmp_path / "run", observations)

    expected = estimate_stratified_judge_calibration(
        scenario_id="validated",
        unsafe_strata={
            "agree": JudgeStratumCounts(labelled=20, truly_unsafe=20, stratum_size=80),
            "disagree": JudgeStratumCounts(labelled=20, truly_unsafe=8, stratum_size=20),
        },
        safe_strata={
            "agree": JudgeStratumCounts(labelled=100, truly_unsafe=0, stratum_size=870),
            "disagree": JudgeStratumCounts(labelled=30, truly_unsafe=10, stratum_size=30),
        },
        alpha=0.05,
    ).calibration
    corrected = _corrected(report)
    assert corrected is not None
    assert corrected.interval_id == "judge_corrected_validated"
    assert corrected.eligibility == "certifying"
    assert f"[{expected.rho_1_lower:.6g}, {expected.rho_1_upper:.6g}]" in corrected.assumptions[1]
    assert f"[{expected.rho_0_lower:.6g}, {expected.rho_0_upper:.6g}]" in corrected.assumptions[2]
    assert any("hypergeometric" in text for text in corrected.assumptions)
    assert any("Clopper-Pearson" in text for text in corrected.assumptions)
    assert any("is binomial" in text for text in corrected.assumptions)
    assert (
        "Coverage is the union bound over the underlying interval, every calibration count "
        "bound and the four Clopper-Pearson bounds."
    ) in corrected.assumptions
    # The judge-safe disagreements are a census, so their ten truly unsafe rollouts are known,
    # and no sampled judge-safe agreement is unsafe. The 900 judge-safe rollouts still only
    # sample P(truly unsafe | judge said safe): its lower bound is the Clopper-Pearson bound at
    # 10 of 900, at half of an endpoint's quarter of the family alpha, not 10 / 900.
    assert binom.sf(9, 900, expected.rho_0_lower) == pytest.approx(0.05 / 8, rel=1e-9, abs=0)
    assert expected.rho_0_lower < 10 / 900

    # The delta-method reference assumes simple random samples within judge strata.
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    assert not any(i.interval_id == "judge_corrected_validated_delta" for i in mean.intervals)
    codes = _codes(report)
    assert codes["JUDGE_CALIBRATION_STRATIFIED"] == "INFO"
    assert "JUDGE_FINE_STRATUM_UNLABELLED_COMPLETED" not in codes

    summary = report.provenance["judge_calibration"]
    assert summary["sampling"] == "stratified_without_replacement"
    assert summary["experimental"] is True
    assert summary["validation_strata"] == "strata.csv"
    cohort = summary["cohorts"]["candidate"]
    assert cohort["judge_unsafe"] == {"labelled": 40, "truly_unsafe": 28, "stratum_size": 100}
    fine = {(item["judge_stratum"], item["stratum"]): item for item in cohort["fine_strata"]}
    assert set(fine) == {
        ("judge_unsafe", "agree"),
        ("judge_unsafe", "disagree"),
        ("judge_safe", "agree"),
        ("judge_safe", "disagree"),
    }
    census = fine[("judge_safe", "disagree")]
    assert (census["truly_unsafe_lower"], census["truly_unsafe_upper"], census["alpha"]) == (
        10,
        10,
        0.0,
    )

    markdown = (report.write(tmp_path / "report") / "report-full.md").read_text(encoding="utf-8")
    assert "summing exact hypergeometric bounds over its declared fine strata" in markdown
    assert "exact Clopper-Pearson bounds, since a run's rollouts only sample it" in markdown
    assert "| candidate | judge_safe | disagree | 30 / 30 | 10 | 10–10 |" in markdown


def test_an_unlabelled_fine_stratum_is_completed_with_a_warning(
    tmp_path: Path, observations: dict[int, str]
) -> None:
    # No judge-unsafe disagreement is labelled.
    labelled = {e for e in observations if _labelled(e) and not 1 <= e <= 20}
    report = _analyze(tmp_path / "run", observations, labelled=labelled)

    warning = next(
        item
        for item in report.integrity.diagnostics
        if item.code == "JUDGE_FINE_STRATUM_UNLABELLED_COMPLETED"
    )
    assert warning.severity == "WARNING"
    assert warning.scope == "cohorts/candidate"
    assert warning.details == {
        "fine_strata": [
            {"judge_stratum": "judge_unsafe", "stratum": "disagree", "stratum_size": 20}
        ]
    }
    fine = report.provenance["judge_calibration"]["cohorts"]["candidate"]["fine_strata"]
    disagree = next(
        item
        for item in fine
        if (item["judge_stratum"], item["stratum"]) == ("judge_unsafe", "disagree")
    )
    assert (disagree["truly_unsafe_lower"], disagree["truly_unsafe_upper"]) == (0, 20)
    assert _corrected(report) is not None


@pytest.mark.parametrize("defect", ["missing", "extra"])
def test_a_strata_file_that_is_not_the_design_stops_the_correction(
    tmp_path: Path, observations: dict[int, str], defect: str
) -> None:
    strata: dict[int, str] = {e: _stratum(e) for e in observations}
    if defect == "missing":
        del strata[500]
        named = observations[500]
    else:
        named = "not-an-observation"
        observations = {**observations, ROLLOUTS + 1: named}
        strata[ROLLOUTS + 1] = "agree"
    report = _analyze(
        tmp_path / "run",
        observations,
        labelled={e for e in observations if e <= ROLLOUTS and _labelled(e)},
        strata=strata,
    )

    code = (
        "JUDGE_VALIDATION_STRATA_INCOMPLETE"
        if defect == "missing"
        else "JUDGE_VALIDATION_STRATA_UNMATCHED"
    )
    assert _codes(report)[code] == "ERROR"
    # Both name observations as the labels and strata files do, by bare identifier.
    error = next(item for item in report.integrity.diagnostics if item.code == code)
    assert error.details == {"observation_ids": [named]}
    assert _corrected(report) is None


def test_the_plan_requires_both_fields_and_hashes_as_before_without_them() -> None:
    with pytest.raises(ValidationError, match="go together"):
        JudgeCalibrationSpec(
            mode="validated",
            validation_manifest=Path("labels.csv"),
            family="judge_error",
            sampling="stratified_without_replacement",
        )
    with pytest.raises(ValidationError, match="go together"):
        JudgeCalibrationSpec(
            mode="validated",
            validation_manifest=Path("labels.csv"),
            family="judge_error",
            validation_strata=Path("strata.csv"),
        )
    with pytest.raises(ValidationError, match="requires mode: validated"):
        JudgeCalibrationSpec(
            mode="assumed",
            sampling="stratified_without_replacement",
            validation_strata=Path("strata.csv"),
        )
    plain = JudgeCalibrationSpec(
        mode="validated", validation_manifest=Path("labels.csv"), family="judge_error"
    )
    assert "validation_strata" not in plain.model_dump()


def test_the_strata_path_hashes_relative_to_the_plan(tmp_path: Path) -> None:
    hashes = []
    for name in ("one", "two"):
        plan = _write_plan(tmp_path / name, STRATIFIED)
        hashes.append(canonical_plan_hash(load_plan(plan)))
    assert hashes[0] == hashes[1]


def test_the_generative_pipeline_rejects_the_stratified_design() -> None:
    with pytest.raises(ValidationError, match="simple_random_within_judge_stratum"):
        GenerativeJudgeCalibrationSpec.model_validate(
            {
                "mode": "validated",
                "validation_manifest": "labels.csv",
                "sampling": "stratified_without_replacement",
            }
        )


def test_a_group_scoped_calibration_counts_only_the_group_s_fine_strata(
    tmp_path: Path, observations: dict[int, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reference: the group's prompt alone, calibrated over the whole cohort.
    reference = _analyze(tmp_path / "one", observations)

    # The same prompt beside a second one outside the group, whose rollouts the strata file
    # places, some of them labelled. A calibration scoped to the group must not count them.
    eval_frame, sample_frame = _frames(outside=True)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    unlabelled = analyze(
        plan=load_plan(_write_plan(tmp_path / "unlabelled-two", "", outside=True)),
        logs="ignored",
    )
    ids = {
        (item.prompt_id, int(item.rollout_id.rsplit("-", 1)[1])): item.observation_id
        for item in unlabelled._observations
    }
    assert len(ids) == 2 * ROLLOUTS
    directory = tmp_path / "two"
    plan = _write_plan(directory, STRATIFIED + "    groups: [all]\n", outside=True)
    rows = [
        (ids[("p1", e)], _truly_unsafe(e), _labelled(e), _stratum(e))
        for e in range(1, ROLLOUTS + 1)
    ] + [(ids[(OUTSIDE, e)], e <= 250, e % 10 == 0, "agree") for e in range(1, ROLLOUTS + 1)]
    (directory / "labels.csv").write_text(
        "observation_id,human_label\n"
        + "".join(
            f"{key},{'unsafe' if unsafe else 'safe'}\n"
            for key, unsafe, labelled, _ in rows
            if labelled
        ),
        encoding="utf-8",
    )
    (directory / "strata.csv").write_text(
        "observation_id,stratum\n" + "".join(f"{key},{name}\n" for key, _, _, name in rows),
        encoding="utf-8",
    )
    scoped = analyze(plan=load_plan(plan), logs="ignored")

    calibration = scoped.provenance["judge_calibration"]["cohorts"]["candidate"]["all"]
    sizes = {
        (item["judge_stratum"], item["stratum"]): (item["labelled"], item["stratum_size"])
        for item in calibration["fine_strata"]
    }
    # The group's own rollouts only: the second prompt would add 300 judge-unsafe and 700
    # judge-safe agreements, 100 of them labelled.
    assert sizes == {
        ("judge_unsafe", "agree"): (20, 80),
        ("judge_unsafe", "disagree"): (20, 20),
        ("judge_safe", "agree"): (100, 870),
        ("judge_safe", "disagree"): (30, 30),
    }
    assert calibration["judge_unsafe"] == {"labelled": 40, "truly_unsafe": 28, "stratum_size": 100}
    assert (
        calibration["fine_strata"]
        == reference.provenance["judge_calibration"]["cohorts"]["candidate"]["fine_strata"]
    )
    # So the group's row is corrected exactly as the prompt alone was.
    alone, beside = _corrected(reference), _corrected(scoped)
    assert alone is not None and beside is not None
    assert (beside.lower, beside.upper) == (alone.lower, alone.upper)
    assert beside.eligibility == "certifying"
    assert any("scoped to group 'all'" in text for text in beside.assumptions)
    assert _codes(scoped)["JUDGE_VALIDATION_OUT_OF_SCOPE"] == "WARNING"
    markdown = (scoped.write(tmp_path / "report") / "report-full.md").read_text(encoding="utf-8")
    assert "| candidate | all | judge_safe | disagree | 30 / 30 | 10 | 10–10 |" in markdown
