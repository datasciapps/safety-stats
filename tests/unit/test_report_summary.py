"""report.json has a summary level: what to read first, each entry pointing at its record."""

from __future__ import annotations

import json
from typing import Any

from inspect_safety_stats.schema.reports import (
    AdaptiveClaimResult,
    ClaimResult,
    ComparisonResult,
    Eligibility,
    GroupRiskResult,
    IntegritySummary,
    JudgeCalibrationPresentation,
    ReportPresentation,
    RiskIntervalResult,
    SafetyReport,
)
from inspect_safety_stats.schema.scopes import plan_scope
from inspect_safety_stats.schema.summary import judge_interval


def _resolve(document: Any, pointer: str) -> Any:
    value = document
    for token in pointer.lstrip("/").split("/"):
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _risk(cohort: str, method: str) -> GroupRiskResult:
    return GroupRiskResult(
        cohort_id=cohort,
        group_id="all",
        measure_id=method.removeprefix("finite_design_"),
        estimate=0.1,
        lower=0.05,
        upper=0.2,
        method=method,
        prompt_count=1,
        valid_prompt_count=1,
        resolved_weight=1.0,
        unresolved_weight=0.0,
        eligibility="certifying",
    )


def _comparison(level: str | None) -> ComparisonResult:
    return ComparisonResult(
        comparison_id="on_vs_off",
        candidate_cohort="on",
        baseline_cohort="off",
        group_id="all",
        measure_id="weighted_mean",
        estimand="risk_difference",
        level=level,
        path={"domain": "cyber"} if level else {},
        estimate=0.02,
        lower=-0.01,
        upper=0.05,
        method="weighted_kl_chernoff",
        eligibility="certifying",
    )


def _report(*, include_decisions: bool) -> SafetyReport:
    diagnostics = [
        {"code": "PLAN_SELF_DECLARED", "severity": "WARNING", "message": "m"},
        {"code": "PLAN_SELF_DECLARED", "severity": "INFO", "message": "m"},
        {
            "code": "JUDGE_CORRECTION_UNAVAILABLE",
            "severity": "WARNING",
            "message": "m",
            "scope": plan_scope("claims", "safe"),
        },
        {"code": "ALPHA_ALLOCATION_INVALID", "severity": "ERROR", "message": "m"},
        {"code": "PLAN_SELF_DECLARED", "severity": "WARNING", "message": "m2"},
    ]
    return SafetyReport.model_validate(
        SafetyReport(
            package_version="0",
            analysis={"id": "a"},
            provenance={},
            scope={},
            presentation=ReportPresentation(include_decisions=include_decisions),
            integrity=IntegritySummary(status="degraded"),
            design={},
            cohorts=[],
            prompt_estimates=[],
            group_risks=[
                _risk("off", "finite_design_worst"),
                _risk("off", "finite_design_weighted_mean"),
                _risk("on", "finite_design_weighted_mean"),
            ],
            comparisons=[_comparison("domain"), _comparison(None)],
            multiplicity=[],
            claims=[
                ClaimResult(
                    claim_id="safe",
                    title="Safe",
                    status="FAIL",
                    statistical_status="FAIL",
                    policy_status="FAIL",
                    acceptable_upper=0.05,
                    unacceptable_lower=0.1,
                    alpha=0.05,
                    multiplicity_family="claims",
                    method="m",
                    mandatory=True,
                )
            ],
            adaptive_claims=[
                AdaptiveClaimResult(
                    claim_id="adaptive",
                    title="Adaptive",
                    status="INCONCLUSIVE",
                    cohort_id="on",
                    group_id="all",
                    order_field="run",
                    unacceptable_lower=0.1,
                    alpha=0.05,
                    multiplicity_family="claims",
                    method="m",
                    mandatory=False,
                )
            ],
            policy_events=[],
            decision="FAIL",
        )
        .model_dump(mode="json")
        | {"integrity": {"status": "degraded", "diagnostics": diagnostics}}
    )


def test_the_summary_counts_diagnostics_and_lists_the_headline_estimates() -> None:
    document = _report(include_decisions=False).model_dump(mode="json")
    summary = document["summary"]

    assert (summary["integrity"], summary["error_count"], summary["warning_count"]) == (
        "degraded",
        1,
        3,
    )
    assert summary["info_count"] == 1
    # The weighted-mean rates and the overall comparison, not the tail measure or the breakdown.
    assert [item["item"] for item in summary["headline"]] == [
        "/group_risks/1",
        "/group_risks/2",
        "/comparisons/1",
    ]
    for item in summary["headline"]:
        record = _resolve(document, item["item"])
        assert (item["estimate"], item["lower"], item["upper"]) == (
            record["estimate"],
            record["lower"],
            record["upper"],
        )


def test_the_summary_groups_errors_then_warnings_by_code_with_pointers() -> None:
    document = _report(include_decisions=False).model_dump(mode="json")
    warnings = document["summary"]["warnings"]

    assert [(item["code"], item["severity"], item["count"]) for item in warnings] == [
        ("ALPHA_ALLOCATION_INVALID", "ERROR", 1),
        ("PLAN_SELF_DECLARED", "WARNING", 2),
        ("JUDGE_CORRECTION_UNAVAILABLE", "WARNING", 1),
    ]
    self_declared = warnings[1]
    assert [item["diagnostic"] for item in self_declared["occurrences"]] == [
        "/integrity/diagnostics/0",
        "/integrity/diagnostics/4",
    ]
    assert self_declared["assumption"] and self_declared["consequence"]
    (occurrence,) = warnings[2]["occurrences"]
    assert occurrence == {
        "diagnostic": "/integrity/diagnostics/2",
        "scope": "plan/claims/safe",
        "item": "/claims/0",
    }
    for warning in warnings:
        for occurrence in warning["occurrences"]:
            assert _resolve(document, occurrence["diagnostic"])["code"] == warning["code"]


def test_hidden_decisions_stay_out_of_the_summary_but_not_out_of_the_report() -> None:
    document = _report(include_decisions=False).model_dump(mode="json")

    assert document["summary"]["decisions_shown"] is False
    assert "decision" not in document["summary"]
    assert "claims" not in document["summary"]
    assert document["decision"] == "FAIL"


def test_shown_decisions_are_in_the_summary_with_pointers_to_each_claim() -> None:
    document = _report(include_decisions=True).model_dump(mode="json")
    summary = document["summary"]

    assert summary["decision"] == "FAIL"
    assert [(item["claim_id"], item["status"], item["item"]) for item in summary["claims"]] == [
        ("safe", "FAIL", "/claims/0"),
        ("adaptive", "INCONCLUSIVE", "/adaptive_claims/0"),
    ]


def test_a_saved_report_reads_back_to_the_same_model_and_summary() -> None:
    report = _report(include_decisions=True)
    text = report.model_dump_json()

    again = SafetyReport.model_validate_json(text)

    assert again == report
    assert json.loads(again.model_dump_json())["summary"] == json.loads(text)["summary"]


def _interval(interval_id: str, calibration: str, eligibility: Eligibility) -> RiskIntervalResult:
    return RiskIntervalResult(
        interval_id=interval_id,
        lower=0.07,
        upper=0.3,
        method=f"{calibration}_judge_misclassification_correction",
        role="reference" if eligibility == "certifying" else "sensitivity",
        simultaneous_coverage=0.9 if eligibility == "certifying" else None,
        eligibility=eligibility,
    )


def test_the_true_risk_is_the_validated_correction_or_the_primary_scenarios() -> None:
    validated = _interval("judge_corrected_validated", "validated", "certifying")
    delta = _interval("judge_corrected_validated_delta", "validated", "diagnostic")
    assert judge_interval([delta, validated], None) is validated

    # Under an assumed calibration only the primary scenario's, even beside a scenario that
    # happens to be named "validated".
    primary = _interval("judge_corrected_declared", "assumed", "diagnostic")
    named = _interval("judge_corrected_validated", "assumed", "diagnostic")
    assert judge_interval([named, primary], "declared") is primary
    assert judge_interval([named], "declared") is None

    # A transported or panel-derived correction is a sensitivity, never the headline's.
    sensitivities = [
        _interval("judge_corrected_transported", "transported", "diagnostic"),
        _interval("judge_corrected_panel_derived", "panel", "diagnostic"),
    ]
    assert judge_interval(sensitivities, None) is None
    assert judge_interval([], None) is None


def test_a_headline_with_a_judge_correction_gives_its_true_risk_interval() -> None:
    report = _report(include_decisions=False)
    report.presentation.judge_calibration = JudgeCalibrationPresentation(
        mode="assumed", primary="declared"
    )
    other = _interval("judge_corrected_optimistic", "assumed", "diagnostic")
    primary = _interval("judge_corrected_declared", "assumed", "diagnostic")
    report.group_risks[2].intervals.extend([other, primary])
    report.comparisons[1].intervals.append(primary)
    document = report.model_dump(mode="json")
    headline = document["summary"]["headline"]

    # A headline that no correction applies to leaves the field out, as before it existed.
    assert "judge_corrected" not in headline[0]
    for item, pointer in [
        (headline[1], "/group_risks/2/intervals/1"),
        (headline[2], "/comparisons/1/intervals/0"),
    ]:
        assert item["judge_corrected"] == {
            "calibration": "assumed",
            "lower": 0.07,
            "upper": 0.3,
            "eligibility": "diagnostic",
            "simultaneous_coverage": None,
            "item": pointer,
        }
        assert _resolve(document, pointer)["interval_id"] == "judge_corrected_declared"
    again = SafetyReport.model_validate_json(json.dumps(document))
    assert json.loads(again.model_dump_json())["summary"] == document["summary"]
