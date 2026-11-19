"""claims.json (experimental, #11), end to end through ``analyze`` and ``write``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.cli import app
from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.report.artifacts import sha256
from inspect_safety_stats.schema.claims import ClaimsExport
from tests.claims_export import round_trip_differences
from tests.cli_output import plain

_PLAN = """
schema_version: "2.0"
analysis:
  id: claims-export
  title: Claims export
  intent: {intent}
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
{judge}design:
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
{cohorts}groups:
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
{comparisons}multiplicity:
  families:
    - {{id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}}
{families}claims:
  - id: mean_safe
    title: Mean risk is acceptable
    cohort: candidate
    group: cyber
    measure: mean
    family: gate
    acceptable_upper: 0.30
    unacceptable_lower: 0.60
{mean_method}  - id: worst_of_2_safe
    title: Worst of two is acceptable
    cohort: candidate
    group: cyber
    measure: worst_of_2
    family: gate
    acceptable_upper: 0.50
    unacceptable_lower: 0.90
{claims}report:
  include_decisions: {include_decisions}
{export}"""


def _write(
    tmp_path: Path,
    *,
    export: str = "  claims_export: true\n",
    include_decisions: bool = True,
    intent: str = "confirmatory",
    judge: str = "",
    families: str = "",
    mean_method: str = "",
    cohorts: str = "",
    comparisons: str = "",
    claims: str = "",
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,risk_domain,expected_rollouts\n"
        "p1,q1,t1,cyber,20\np2,q2,t1,cyber,20\n",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        _PLAN.format(
            export=export,
            include_decisions=str(include_decisions).lower(),
            intent=intent,
            judge=judge,
            families=families,
            mean_method=mean_method,
            cohorts=cohorts,
            comparisons=comparisons,
            claims=claims,
        ),
        encoding="utf-8",
    )
    return plan


_JUDGE = (
    "  judge_calibration:\n"
    "    mode: validated\n"
    "    validation_manifest: labels.csv\n"
    "    family: judge_error\n"
    "    sampling: {sampling}\n"
)
_JUDGE_FAMILY = (
    "    - {id: judge_error, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}\n"
)


def _calibrated(tmp_path: Path, *, sampling: str, mean_method: str = "") -> Path:
    """A plan with a validated judge calibration whose labels agree with every verdict."""

    unlabelled = analyze(plan=_write(tmp_path / "unlabelled"), logs="ignored")
    rows = ["observation_id,human_label"]
    rows.extend(
        f"{item.observation_id},{'unsafe' if item.failure else 'safe'}"
        for item in unlabelled._observations
        if item.failure is not None
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return _write(
        tmp_path,
        judge=_JUDGE.format(sampling=sampling),
        families=_JUDGE_FAMILY,
        mean_method=mean_method,
    )


def _located(report: dict[str, Any], claim_id: str) -> tuple[str, dict[str, Any]]:
    """The pointer to the group risk a claim of report.json is about, found by its estimand."""

    (record,) = [
        item for item in report["presentation"]["claim_evidence"] if item["claim_id"] == claim_id
    ]
    (index,) = [
        index
        for index, risk in enumerate(report["group_risks"])
        if (risk["cohort_id"], risk["group_id"], risk["measure_id"])
        == (record["cohort"], record["group"], record["measure"])
    ]
    return f"/group_risks/{index}", report["group_risks"][index]


def _patch_logs(monkeypatch: pytest.MonkeyPatch, failures: dict[str, int]) -> None:
    """Twenty rollouts of p1 and p2 per model; the first ``failures[model]`` of p1 fail."""

    evals = pd.DataFrame(
        [
            {
                "eval_id": f"eval-{number}",
                "run_id": f"run-{number}",
                "log": "/not/a/local/log.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "fixed_safety",
                "task_version": "1",
                "model": f"model/{model}",
                "model_generate_config": {"temperature": 1},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "fixed",
                "epochs": 20,
            }
            for number, model in enumerate(failures, start=1)
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"{model}-{prompt_id}-{epoch}",
                "eval_id": f"eval-{number}",
                "log": "/not/a/local/log.eval",
                "id": prompt_id,
                "epoch": epoch,
                "score_harmful": "FAIL" if prompt_id == "p1" and epoch <= failing else "PASS",
                "error": None,
                "retries": 0,
            }
            for number, (model, failing) in enumerate(failures.items(), start=1)
            for prompt_id in ("p1", "p2")
            for epoch in range(1, 21)
        ]
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )


@pytest.fixture(autouse=True)
def _offline_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_logs(monkeypatch, {"candidate": 2})


@pytest.mark.parametrize("intent", ["confirmatory", "exploratory"])
def test_every_exported_value_is_the_report_json_value(tmp_path: Path, intent: str) -> None:
    report = analyze(plan=_write(tmp_path, intent=intent), logs="ignored")
    output = report.write(tmp_path / "report")

    assert round_trip_differences(output) == []
    exported = ClaimsExport.model_validate_json(
        (output / "claims.json").read_text(encoding="utf-8")
    )
    assert [record.claim_id for record in exported.claims] == ["mean_safe", "worst_of_2_safe"]
    mean = exported.claims[0]
    # The claim is decided on the group risk's primary interval, and points at both.
    assert mean.sources.evidence is not None
    assert mean.sources.evidence.startswith("/group_risks/")
    assert mean.sources.evidence_interval == f"{mean.sources.evidence}/intervals/0"
    assert mean.evidence.interval_id == report.group_risks[0].primary_interval_id
    assert mean.evidence.independence_profile == "rollout"
    assert mean.evidence.assumptions
    assert mean.evidence.test is None
    assert not mean.judge_corrected
    if intent == "exploratory":
        assert mean.status == "INCONCLUSIVE"
        assert mean.descriptive_status == "PASS"
    else:
        assert mean.status == "PASS"


def test_claims_json_is_in_checksums(tmp_path: Path) -> None:
    output = analyze(plan=_write(tmp_path), logs="ignored").write(tmp_path / "report")

    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["claims.json"] == sha256(output / "claims.json")


def test_the_same_analysis_writes_the_same_bytes(tmp_path: Path) -> None:
    plan = _write(tmp_path)
    first = analyze(plan=plan, logs="ignored").write(tmp_path / "first")
    second = analyze(plan=plan, logs="ignored").write(tmp_path / "second")

    assert (first / "claims.json").read_bytes() == (second / "claims.json").read_bytes()


def test_a_plan_that_does_not_opt_in_writes_what_it_wrote_before(tmp_path: Path) -> None:
    off = _write(tmp_path / "off", export="")
    explicit = _write(tmp_path / "explicit", export="  claims_export: false\n")
    assert canonical_plan_hash(load_plan(off)) == canonical_plan_hash(load_plan(explicit))

    output = analyze(plan=off, logs="ignored").write(tmp_path / "report")
    assert not (output / "claims.json").exists()
    # The report's record of the defaults applied lists the option, and nothing else does.
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert {"path": "report.claims_export", "value": False} in report.pop("plan_defaults")
    texts = {
        "report.json": json.dumps(report),
        **{
            name: (output / name).read_text(encoding="utf-8")
            for name in ("plan.resolved.yaml", "checksums.json")
        },
    }
    for name, text in texts.items():
        assert "claims" + "_export" not in text, name
        assert "claims.json" not in text, name


def test_rewriting_without_the_export_removes_a_stale_claims_json(tmp_path: Path) -> None:
    output = tmp_path / "report"
    analyze(plan=_write(tmp_path / "on"), logs="ignored").write(output)
    assert (output / "claims.json").exists()

    analyze(plan=_write(tmp_path / "off", export=""), logs="ignored").write(output, overwrite=True)
    assert not (output / "claims.json").exists()


def test_the_export_is_refused_without_decisions(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="claims_export needs report.include_decisions"):
        load_plan(_write(tmp_path, include_decisions=False))


def test_the_schema_command_prints_the_claims_schema() -> None:
    result = CliRunner().invoke(app, ["schema", "claims"])

    assert result.exit_code == 0, plain(result.output)
    assert json.loads(result.output) == ClaimsExport.model_json_schema()


def test_a_noisy_valid_claim_exports_the_interval_its_test_was_applied_to(
    tmp_path: Path,
) -> None:
    plan = _calibrated(
        tmp_path, sampling="simple_random", mean_method="    method: noisy_valid_test\n"
    )
    output = analyze(plan=plan, logs="ignored").write(tmp_path / "report")

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    exported = json.loads((output / "claims.json").read_text(encoding="utf-8"))
    (claim,) = [item for item in report["claims"] if item["claim_id"] == "mean_safe"]
    (record,) = [item for item in exported["claims"] if item["claim_id"] == "mean_safe"]
    # The analysis names the validated correction too, but decides the claim by the test,
    # which it applies to the group risk's primary interval: the judge-reported one.
    assert claim["method"] == "noisy_valid_certification_test"
    assert claim["evidence_refs"] == [
        "risk:candidate:cyber:mean",
        "interval:judge_corrected_validated",
        "test:noisy_valid_certification",
    ]
    pointer, risk = _located(report, "mean_safe")
    assert risk["primary_interval_id"] == "direct_weighted_hoeffding"
    (position,) = [
        index
        for index, item in enumerate(risk["intervals"])
        if item["interval_id"] == "direct_weighted_hoeffding"
    ]
    assert record["evidence"]["interval_id"] == "direct_weighted_hoeffding"
    assert record["evidence"]["interval_method"] == "weighted_hoeffding"
    assert record["sources"]["evidence"] == pointer
    assert record["sources"]["evidence_interval"] == f"{pointer}/intervals/{position}"
    assert record["evidence"]["test"] == "noisy_valid_certification"
    # The bound is the test's, which ran.
    assert record["bound"]["method"] == "noisy_valid_certification_test"
    assert record["bound"]["upper"] is not None
    assert (record["bound"]["lower"], record["bound"]["upper"]) == (claim["lower"], claim["upper"])
    assert not record["judge_corrected"]
    assert round_trip_differences(output) == []


def test_a_claim_the_validated_calibration_cannot_correct_is_flagged(tmp_path: Path) -> None:
    plan = _calibrated(tmp_path, sampling="simple_random_within_judge_stratum")
    report = analyze(plan=plan, logs="ignored")
    output = report.write(tmp_path / "report")

    exported = json.loads((output / "claims.json").read_text(encoding="utf-8"))
    records = {item["claim_id"]: item for item in exported["claims"]}
    claims = {item.claim_id: item for item in report.claims}
    # worst_of_k has no judge correction, so under a validated calibration its claim is
    # INCONCLUSIVE, though report.json still gives it the uncorrected judge-reported bound.
    worst = records["worst_of_2_safe"]
    assert worst["judge_correction_unavailable"] is True
    assert worst["judge_corrected"] is False
    assert worst["status"] == "INCONCLUSIVE"
    assert worst["bound"]["upper"] is not None
    assert worst["bound"]["upper"] == claims["worst_of_2_safe"].upper
    assert worst["reasons"] == claims["worst_of_2_safe"].reasons
    assert worst["reasons"][0].startswith(
        "The plan declares a validated judge calibration, but this claim's evidence has no "
        "certifying correction"
    )
    # The mean is decided on its validated correction.
    mean = records["mean_safe"]
    assert mean["judge_corrected"] is True
    assert mean["judge_correction_unavailable"] is False
    assert mean["reasons"] == claims["mean_safe"].reasons
    assert round_trip_differences(output) == []


_BASELINE = "  baseline:\n    match:\n      model: model/baseline\n"
_COMPARISON = """  - id: {id}
    candidate: candidate
    baseline: baseline
    group: cyber
    measure: mean
    estimand: risk_difference
    method: weighted_hoeffding
    family: model_comparisons
    hierarchy_nodes: {nodes}
    independence:
      primary: rollout
"""
_COMPARISONS = (
    "comparisons:\n"
    + _COMPARISON.format(id="difference", nodes="false")
    + _COMPARISON.format(id="node_difference", nodes="true")
)
_COMPARISON_FAMILY = (
    "    - {id: model_comparisons, purpose: simultaneous_intervals, method: bonferroni, "
    "alpha: 0.05}\n"
)
_COMPARISON_CLAIM = """  - id: not_worse
    title: No worse than the baseline
    comparison: node_difference
    family: gate
    acceptable_upper: 0.10
    unacceptable_lower: 0.40
"""


def _compared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claims: str) -> Path:
    """A report on a candidate and a baseline, written with claims.json."""

    _patch_logs(monkeypatch, {"candidate": 2, "baseline": 12})
    plan = _write(
        tmp_path,
        cohorts=_BASELINE,
        comparisons=_COMPARISONS,
        families=_COMPARISON_FAMILY,
        claims=claims,
    )
    return analyze(plan=plan, logs="ignored").write(tmp_path / "report")


def test_a_comparison_claim_exports_its_overall_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _compared(tmp_path, monkeypatch, _COMPARISON_CLAIM)

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    exported = json.loads((output / "claims.json").read_text(encoding="utf-8"))
    (record,) = [item for item in exported["claims"] if item["claim_id"] == "not_worse"]
    # The claim's comparison is the second one declared, and its hierarchy nodes share its id.
    rows = [(item["comparison_id"], item["level"]) for item in report["comparisons"]]
    assert rows.count(("node_difference", None)) == 1
    assert len([row for row in rows if row[0] == "node_difference"]) > 1
    index = rows.index(("node_difference", None))
    assert index > 0
    comparison = report["comparisons"][index]
    assert comparison["primary_interval_id"] == "direct_weighted_hoeffding"
    (position,) = [
        position
        for position, item in enumerate(comparison["intervals"])
        if item["interval_id"] == "direct_weighted_hoeffding"
    ]
    assert record["sources"]["evidence"] == f"/comparisons/{index}"
    assert record["sources"]["evidence_interval"] == f"/comparisons/{index}/intervals/{position}"
    assert record["evidence"]["interval_id"] == "direct_weighted_hoeffding"
    assert record["evidence"]["test"] is None
    assert record["estimand"]["comparison"] == "node_difference"
    assert record["estimand"]["cohort"] is None
    assert (record["bound"]["lower"], record["bound"]["upper"]) == (
        comparison["lower"],
        comparison["upper"],
    )
    assert record["bound"]["upper"] is not None
    assert round_trip_differences(output) == []


def test_a_claim_without_a_result_interval_or_family_exports_nulls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claims = (
        _COMPARISON_CLAIM
        + """  - id: elsewhere
    title: A group the plan does not declare
    cohort: candidate
    group: nowhere
    measure: mean
    family: gate
    acceptable_upper: 0.10
    unacceptable_lower: 0.40
  - id: unbudgeted
    title: A family the plan does not declare
    cohort: candidate
    group: cyber
    measure: mean
    family: undeclared
    acceptable_upper: 0.10
    unacceptable_lower: 0.40
"""
    )
    output = _compared(tmp_path, monkeypatch, claims)

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    exported = json.loads((output / "claims.json").read_text(encoding="utf-8"))
    records = {item["claim_id"]: item for item in exported["claims"]}
    claims_by_id = {item["claim_id"]: item for item in report["claims"]}
    # Both references are plan errors, so integrity is invalid and no claim is decided.
    assert report["integrity"]["status"] == "invalid"

    # No result for the undeclared group: every evidence field is null.
    elsewhere = records["elsewhere"]
    assert not [
        risk for risk in report["group_risks"] if risk["group_id"] == "nowhere"
    ]
    (estimand,) = [
        index
        for index, item in enumerate(report["presentation"]["claim_evidence"])
        if item["claim_id"] == "elsewhere"
    ]
    assert elsewhere["sources"]["estimand"] == f"/presentation/claim_evidence/{estimand}"
    assert elsewhere["sources"]["evidence"] is None
    assert elsewhere["sources"]["evidence_interval"] is None
    assert elsewhere["evidence"] == {
        "refs": ["risk:candidate:nowhere:mean"],
        "test": None,
        "eligibility": None,
        "interval_id": None,
        "interval_method": None,
        "coverage_mode": None,
        "coverage_scope": None,
        "family_alpha": None,
        "independence_profile": None,
        "assumptions": [],
    }
    assert elsewhere["bound"]["method"] == "unavailable"

    # The comparison is reported without intervals, so it has a result but no interval.
    not_worse = records["not_worse"]
    (index,) = [
        index
        for index, item in enumerate(report["comparisons"])
        if (item["comparison_id"], item["level"]) == ("node_difference", None)
    ]
    assert report["comparisons"][index]["primary_interval_id"] is None
    assert not_worse["sources"]["evidence"] == f"/comparisons/{index}"
    assert not_worse["sources"]["evidence_interval"] is None
    assert not_worse["evidence"]["interval_id"] is None
    assert not_worse["evidence"]["assumptions"] == []

    # Without a declared family the claim has no error budget.
    unbudgeted = records["unbudgeted"]
    assert unbudgeted["bound"]["alpha"] is None
    assert unbudgeted["bound"]["multiplicity_family"] == "undeclared"
    assert unbudgeted["status"] == "INCONCLUSIVE"
    assert unbudgeted["reasons"] == claims_by_id["unbudgeted"]["reasons"]
    assert any("is not declared" in reason for reason in unbudgeted["reasons"])
    assert round_trip_differences(output) == []
