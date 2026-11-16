"""The report's first page says where in the hierarchy an effect or a limit exceedance sits.

Headline rows average over the hierarchy, so an effect concentrated in one category is visible
only by reading every hierarchy row. The "Where effects sit" section lists those nodes.
"""

from __future__ import annotations

from typing import Any

from inspect_safety_stats.report.writer import _markdown
from inspect_safety_stats.schema.reports import (
    ClaimEvidenceRecord,
    ClaimResult,
    ComparisonResult,
    HierarchyRiskResult,
    IntegritySummary,
    ReportPresentation,
    SafetyReport,
)


def _comparison(
    path: dict[str, str] | None,
    estimate: float,
    lower: float,
    upper: float,
    estimand: str = "risk_difference",
) -> ComparisonResult:
    return ComparisonResult(
        comparison_id="thinking",
        candidate_cohort="on",
        baseline_cohort="off",
        group_id="all",
        measure_id="mean",
        estimand=estimand,
        level="domain" if path else None,
        path=path or {},
        candidate_estimate=0.2,
        baseline_estimate=0.1,
        estimate=estimate,
        lower=lower,
        upper=upper,
        method="weighted_kl_chernoff",
        prompt_count=10,
        resolved_weight=0.25 if path else 1.0,
        eligibility="certifying",
    )


def _node(domain: str, estimate: float, lower: float, upper: float) -> HierarchyRiskResult:
    return HierarchyRiskResult(
        cohort_id="on",
        group_id="all",
        level="domain",
        path={"domain": domain},
        estimate=estimate,
        lower=lower,
        upper=upper,
        design_weight=0.25,
        prompt_count=10,
        valid_prompt_count=10,
        eligibility="certifying",
    )


def _claim(claim_id: str, title: str) -> ClaimResult:
    return ClaimResult(
        claim_id=claim_id,
        title=title,
        status="PASS",
        statistical_status="PASS",
        policy_status="PASS",
        estimate=0.032,
        lower=0.02,
        upper=0.045,
        acceptable_upper=0.05,
        unacceptable_lower=0.1,
        alpha=0.05,
        multiplicity_family="claims",
        method="weighted_kl_chernoff",
        mandatory=True,
    )


def _report(
    *,
    comparisons: list[ComparisonResult],
    hierarchy: list[HierarchyRiskResult],
    claims: list[ClaimResult],
    claim_specs: list[dict[str, Any]],
    measure_type: str = "weighted_mean",
    include_decisions: bool = True,
) -> SafetyReport:
    report = SafetyReport(
        package_version="0",
        analysis={"id": "a"},
        provenance={},
        scope={"disclaimer": "Fixed design."},
        presentation=ReportPresentation(
            include_decisions=include_decisions,
            decimal_places=3,
            claim_evidence=[
                ClaimEvidenceRecord(
                    claim_id=spec["id"],
                    comparison=spec.get("comparison"),
                    cohort=spec.get("cohort"),
                    group=spec.get("group"),
                    measure=spec.get("measure"),
                    measure_type=measure_type if spec.get("measure") == "mean" else None,
                )
                for spec in claim_specs
            ],
        ),
        integrity=IntegritySummary(status="valid"),
        design={},
        cohorts=[],
        prompt_estimates=[],
        group_risks=[],
        hierarchy_risks=hierarchy,
        comparisons=comparisons,
        multiplicity=[],
        claims=claims,
        policy_events=[],
        decision="PASS",
    )
    return report


def _section(markdown: str) -> str:
    start = markdown.index("## Where effects sit")
    return markdown[start : markdown.index("## Data integrity", start)]


AGGREGATE = {"id": "level", "cohort": "on", "group": "all", "measure": "mean"}
MISINFORMATION_EFFECT = ({"domain": "misinformation"}, 0.139, 0.061, 0.217)


def test_a_concentrated_effect_is_listed_beside_the_overall_effect() -> None:
    report = _report(
        comparisons=[
            _comparison(None, 0.028, -0.01, 0.06),
            _comparison(*MISINFORMATION_EFFECT),
            _comparison({"domain": "cybercrime"}, 0.022, -0.03, 0.07),
        ],
        hierarchy=[],
        claims=[],
        claim_specs=[],
    )
    section = _section(_markdown(report))
    assert (
        "| thinking | domain: misinformation | 0.250 | 10 | 0.200 | 0.100 | 0.139 | "
        "0.061–0.217 | 0.028 |"
    ) in section
    assert "cybercrime" not in section


def test_no_node_excluding_no_difference_is_said_plainly() -> None:
    report = _report(
        comparisons=[_comparison({"domain": "cybercrime"}, 0.022, -0.03, 0.07)],
        hierarchy=[],
        claims=[],
        claim_specs=[],
    )
    section = _section(_markdown(report))
    assert "No hierarchy node's comparison interval excludes no difference." in section


def test_a_risk_ratio_node_is_judged_against_one() -> None:
    report = _report(
        comparisons=[
            _comparison({"domain": "a"}, 1.5, 0.8, 2.0, estimand="risk_ratio"),
            _comparison({"domain": "b"}, 0.5, 0.2, 0.9, estimand="risk_ratio"),
        ],
        hierarchy=[],
        claims=[],
        claim_specs=[],
    )
    section = _section(_markdown(report))
    assert "domain: b" in section
    assert "domain: a" not in section


def test_nodes_above_a_claims_limit_say_whether_the_interval_or_the_estimate_is_above() -> None:
    report = _report(
        comparisons=[],
        hierarchy=[
            _node("misinformation", 0.152, 0.116, 0.194),
            _node("cybercrime", 0.074, 0.049, 0.116),
            _node("fraud", 0.02, 0.005, 0.06),
        ],
        claims=[_claim("level", "Harmful compliance below 5%")],
        claim_specs=[AGGREGATE],
    )
    section = _section(_markdown(report))
    assert (
        "| Harmful compliance below 5% | on / all / mean | domain: misinformation | 0.250 | 10 | "
        "0.152 | 0.116–0.194 | 0.050 | 0.032 | interval above the limit |"
    ) in section
    assert (
        "| domain: cybercrime | 0.250 | 10 | 0.074 | 0.049–0.116 | 0.050 | 0.032 | "
        "estimate above the limit |"
    ) in section
    assert "fraud" not in section
    assert "comparison interval" not in section


def test_no_node_above_a_limit_is_said_plainly() -> None:
    report = _report(
        comparisons=[],
        hierarchy=[_node("fraud", 0.02, 0.005, 0.06)],
        claims=[_claim("level", "Harmful compliance below 5%")],
        claim_specs=[AGGREGATE],
    )
    section = _section(_markdown(report))
    assert "No hierarchy node's estimate is above a claim's acceptable limit." in section


def test_a_claim_on_another_measure_does_not_reach_hierarchy_nodes() -> None:
    report = _report(
        comparisons=[_comparison(*MISINFORMATION_EFFECT)],
        hierarchy=[_node("misinformation", 0.152, 0.116, 0.194)],
        claims=[_claim("level", "Worst prompt below 5%")],
        claim_specs=[AGGREGATE],
        measure_type="maximum",
    )
    section = _section(_markdown(report))
    assert "acceptable limit" not in section.lower()


def test_a_comparison_claim_reaches_its_comparisons_hierarchy_rows() -> None:
    report = _report(
        comparisons=[
            _comparison(None, 0.028, -0.01, 0.06),
            _comparison(*MISINFORMATION_EFFECT),
        ],
        hierarchy=[],
        claims=[_claim("increase", "Increase below 5 points")],
        claim_specs=[{"id": "increase", "comparison": "thinking"}],
    )
    section = _section(_markdown(report))
    assert (
        "| Increase below 5 points | comparison thinking | domain: misinformation | 0.250 | 10 | "
        "0.139 | 0.061–0.217 | 0.050 | 0.032 | interval above the limit |"
    ) in section


def test_hidden_decisions_hide_the_limit_table() -> None:
    report = _report(
        comparisons=[_comparison(*MISINFORMATION_EFFECT)],
        hierarchy=[_node("misinformation", 0.152, 0.116, 0.194)],
        claims=[_claim("level", "Harmful compliance below 5%")],
        claim_specs=[AGGREGATE],
        include_decisions=False,
    )
    section = _section(_markdown(report))
    assert "| thinking | domain: misinformation |" in section
    assert "acceptable limit" not in section.lower()


def test_hierarchy_risks_that_no_shown_claim_reaches_give_no_section() -> None:
    hierarchy = [_node("misinformation", 0.152, 0.116, 0.194)]
    hidden = _report(
        comparisons=[],
        hierarchy=hierarchy,
        claims=[_claim("level", "Harmful compliance below 5%")],
        claim_specs=[AGGREGATE],
        include_decisions=False,
    )
    unclaimed = _report(comparisons=[], hierarchy=hierarchy, claims=[], claim_specs=[])
    assert "## Where effects sit" not in _markdown(hidden)
    assert "## Where effects sit" not in _markdown(unclaimed)


def test_a_report_without_hierarchy_rows_has_no_section() -> None:
    report = _report(
        comparisons=[_comparison(None, 0.028, 0.01, 0.06)],
        hierarchy=[],
        claims=[_claim("level", "Harmful compliance below 5%")],
        claim_specs=[AGGREGATE],
    )
    assert "## Where effects sit" not in _markdown(report)
