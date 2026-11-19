"""A diagnostic points at the record of its item in report.json, found from its scope."""

from __future__ import annotations

import json
from typing import Any

import pytest

from inspect_safety_stats.report.artifacts import json_text
from inspect_safety_stats.schema.reports import (
    STORED_REPORT,
    ClaimResult,
    ComparisonResult,
    Diagnostic,
    GroupRiskResult,
    IntegritySummary,
    PromptEstimate,
    SafetyReport,
)
from inspect_safety_stats.schema.scopes import (
    cohort_scope,
    node_id,
    plan_scope,
    prompt_scope,
    scope_path,
)

NODE = scope_path("comparisons", "on_vs_off", "levels", "domain", "nodes", "domain=cyber")


def _comparison(level: str | None) -> ComparisonResult:
    return ComparisonResult(
        comparison_id="on_vs_off",
        candidate_cohort="on",
        baseline_cohort="off",
        group_id="all",
        measure_id="mean",
        estimand="risk_difference",
        level=level,
        path={"domain": "cyber"} if level else {},
        method="weighted_kl_chernoff",
        eligibility="certifying",
    )


def _claim(claim_id: str) -> ClaimResult:
    return ClaimResult(
        claim_id=claim_id,
        title=claim_id,
        status="PASS",
        statistical_status="PASS",
        policy_status="PASS",
        acceptable_upper=0.05,
        unacceptable_lower=0.1,
        alpha=0.05,
        multiplicity_family="claims",
        method="weighted_kl_chernoff",
        mandatory=True,
    )


def _prompt(cohort: str, prompt: str) -> PromptEstimate:
    return PromptEstimate(
        cohort_id=cohort,
        prompt_id=prompt,
        question_id=prompt,
        task_id="t",
        n_total=1,
        n_valid=1,
        n_failure=0,
        n_safe=1,
        n_error=0,
    )


def _risk(cohort: str, measure: str) -> GroupRiskResult:
    return GroupRiskResult(
        cohort_id=cohort,
        group_id="all",
        measure_id=measure,
        method="finite_design_weighted_mean",
        prompt_count=1,
        valid_prompt_count=1,
        resolved_weight=1.0,
        unresolved_weight=0.0,
        eligibility="certifying",
    )


def _report(scopes: list[str | None], **context: Any) -> SafetyReport:
    document = SafetyReport(
        package_version="0",
        analysis={"id": "a"},
        provenance={},
        scope={},
        integrity=IntegritySummary(status="valid"),
        design={"prompts": [{"prompt_id": "p0"}, {"prompt_id": "a/b"}]},
        cohorts=[{"cohort_id": "off"}, {"cohort_id": "on"}],
        prompt_estimates=[_prompt("off", "p0"), _prompt("on", "p0"), _prompt("on", "a/b")],
        group_risks=[_risk("off", "mean"), _risk("on", "mean"), _risk("on", "worst")],
        comparisons=[_comparison("domain"), _comparison(None)],
        multiplicity=[{"id": "displayed"}, {"id": "claims"}],
        claims=[_claim("c0"), _claim("mean.safe")],
        policy_events=[],
        decision="PASS",
    ).model_dump(mode="json")
    document["integrity"]["diagnostics"] = [
        {"code": "PLAN_SELF_DECLARED", "severity": "INFO", "message": "m", "scope": scope}
        for scope in scopes
    ]
    return SafetyReport.model_validate(document, context=context)


@pytest.mark.parametrize(
    ("scope", "item"),
    [
        (plan_scope("claims", "mean.safe"), "/claims/1"),
        (plan_scope("claims"), "/claims"),
        (plan_scope("comparisons", "on_vs_off"), "/comparisons/1"),
        (scope_path("comparisons", "on_vs_off", "methods", "m", "units", "prompt"), "/comparisons/1"),  # noqa: E501
        (scope_path("comparisons"), "/comparisons"),
        # A hierarchy node's result, not the overall one.
        (NODE, "/comparisons/0"),
        (f"{NODE}/methods/m/profiles/rollout", "/comparisons/0"),
        (plan_scope("multiplicity", "families", "claims"), "/multiplicity/1"),
        (scope_path("families", "displayed"), "/multiplicity/0"),
        (cohort_scope("on"), "/cohorts/1"),
        (prompt_scope("on", "a/b"), "/prompt_estimates/2"),
        (cohort_scope("on", "groups", "all", "measures", "worst"), "/group_risks/2"),
        (scope_path("prompts", "a/b"), "/design/prompts/1"),
    ],
)
def test_a_diagnostic_points_at_the_record_its_scope_names(scope: str, item: str) -> None:
    (diagnostic,) = _report([scope]).integrity.diagnostics

    assert diagnostic.item == item


@pytest.mark.parametrize(
    "scope",
    [
        None,
        plan_scope("outputs", "risk_measures"),  # a plan setting: no record in report.json
        plan_scope("claims", "missing"),
        prompt_scope("off", "a/b"),  # the prompt exists, but not in this cohort
        cohort_scope("on", "strata", "s1"),  # not the cohort: never an enclosing record
        scope_path("runs", "r1"),
        "decision_policy/versions",  # the report has no decision policy
        # A node the comparison has no result for, and a level: never the overall result.
        scope_path("comparisons", "on_vs_off", "levels", "domain", "nodes", "domain=bio"),
        scope_path("comparisons", "on_vs_off", "levels", "domain"),
    ],
)
def test_a_diagnostic_whose_item_the_report_does_not_hold_points_nowhere(
    scope: str | None,
) -> None:
    (diagnostic,) = _report([scope]).integrity.diagnostics

    assert diagnostic.item is None


def test_a_free_text_scope_of_an_old_report_points_nowhere() -> None:
    (diagnostic,) = _report(["candidate:p1"], **{STORED_REPORT: True}).integrity.diagnostics

    assert diagnostic.item is None


def test_a_stored_item_is_recomputed_on_read_back() -> None:
    document = _report([prompt_scope("on", "p0")]).model_dump(mode="json")
    assert document["integrity"]["diagnostics"][0]["item"] == "/prompt_estimates/1"
    document["integrity"]["diagnostics"][0]["item"] = "/claims/0"

    report = SafetyReport.model_validate_json(json.dumps(document))

    assert report.integrity.diagnostics[0].item == "/prompt_estimates/1"


def test_a_node_pointer_survives_the_sorted_keys_of_report_json() -> None:
    """A node's id lists its levels root first; report.json writes its path's keys sorted."""

    path = {"task_id": "t1", "question_id": "q1"}
    scope = scope_path(
        "comparisons", "on_vs_off", "levels", "question_id", "nodes", node_id(path)
    )
    document = _report([scope]).model_dump(mode="json")
    document["design"]["hierarchy"] = ["task_id", "question_id", "prompt_id"]
    document["comparisons"][0] |= {"level": "question_id", "path": path}
    report = SafetyReport.model_validate(document)
    assert report.integrity.diagnostics[0].item == "/comparisons/0"

    text = json_text(json.loads(report.model_dump_json()))
    assert list(json.loads(text)["comparisons"][0]["path"]) == ["question_id", "task_id"]
    again = SafetyReport.model_validate_json(text)

    assert again.integrity.diagnostics[0].item == "/comparisons/0"


def test_a_diagnostic_outside_a_report_has_no_item() -> None:
    assert Diagnostic(code="PLAN_SELF_DECLARED", severity="INFO", message="m").item is None
