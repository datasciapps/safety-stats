"""report-full.md says where in the hierarchy an effect or a limit exceedance sits.

Headline rows average over the hierarchy, so an effect concentrated in one category is visible
only by reading every hierarchy row. The "Where effects sit" section lists those nodes. It
decides on intervals, and on estimates where a node has one, as printed; a node it cannot
place is counted rather than left out.
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
    JudgeCalibrationPresentation,
    ReportPresentation,
    RiskIntervalResult,
    SafetyReport,
)

SIMULTANEOUS = "simultaneous_rollout_fixed_design_mean_family_hierarchy_region"
POINTWISE = "pointwise_rollout_fixed_design_mean_family_hierarchy_region"


def _comparison(
    path: dict[str, str] | None,
    estimate: float | None,
    lower: float | None,
    upper: float | None,
    estimand: str = "risk_difference",
    *,
    comparison_id: str = "thinking",
    coverage_scope: str = SIMULTANEOUS,
) -> ComparisonResult:
    return ComparisonResult(
        comparison_id=comparison_id,
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
        coverage_scope=coverage_scope,
        prompt_count=10,
        resolved_weight=0.25 if path else 1.0,
        eligibility="certifying" if lower is not None else "invalid",
    )


def _node(
    domain: str,
    estimate: float | None,
    lower: float | None,
    upper: float | None,
    *,
    cohort: str = "on",
    group: str = "all",
    coverage_scope: str = SIMULTANEOUS,
    corrected: str | None = None,
) -> HierarchyRiskResult:
    """A node row; ``corrected`` gives it a judge-corrected interval of that eligibility."""
    intervals = []
    if corrected is not None:
        diagnostic = corrected == "diagnostic"
        intervals.append(
            RiskIntervalResult(
                interval_id="judge_corrected_validated",
                lower=0.01,
                upper=0.6,
                method="validated_judge_misclassification_correction",
                role="sensitivity" if diagnostic else "reference",
                varied_assumption="judge_error" if diagnostic else None,
                eligibility="diagnostic" if diagnostic else "certifying",
            )
        )
    return HierarchyRiskResult(
        cohort_id=cohort,
        group_id=group,
        level="domain",
        path={"domain": domain},
        estimate=estimate,
        lower=lower,
        upper=upper,
        intervals=intervals,
        coverage_scope=coverage_scope,
        design_weight=0.25,
        prompt_count=10,
        valid_prompt_count=10 if estimate is not None else 9,
        eligibility="certifying" if lower is not None else "invalid",
    )


def _claim(claim_id: str, title: str, *, refs: list[str] | None = None) -> ClaimResult:
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
        evidence_refs=refs or [],
    )


def _report(
    *,
    comparisons: list[ComparisonResult],
    hierarchy: list[HierarchyRiskResult],
    claims: list[ClaimResult],
    claim_specs: list[dict[str, Any]],
    measure_type: str = "weighted_mean",
    include_decisions: bool = True,
    judge_mode: Any = None,
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
            judge_calibration=(
                JudgeCalibrationPresentation(mode=judge_mode) if judge_mode else None
            ),
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


def _limit_part(section: str) -> str:
    """The part of the section about claims' limits, after the comparison effects."""
    return section[section.index("| Claim | Evidence |") :]


AGGREGATE = {"id": "level", "cohort": "on", "group": "all", "measure": "mean"}
LEVEL_CLAIM = "Harmful compliance below 5%"
MISINFORMATION_EFFECT = ({"domain": "misinformation"}, 0.139, 0.061, 0.217)


def _limit_report(
    hierarchy: list[HierarchyRiskResult],
    *,
    refs: list[str] | None = None,
    judge_mode: Any = None,
) -> SafetyReport:
    return _report(
        comparisons=[],
        hierarchy=hierarchy,
        claims=[_claim("level", LEVEL_CLAIM, refs=refs)],
        claim_specs=[AGGREGATE],
        judge_mode=judge_mode,
    )


def _effects_report(comparisons: list[ComparisonResult]) -> SafetyReport:
    return _report(comparisons=comparisons, hierarchy=[], claims=[], claim_specs=[])


def test_a_concentrated_effect_is_listed_beside_the_overall_effect() -> None:
    report = _effects_report(
        [
            _comparison(None, 0.028, -0.01, 0.06),
            _comparison(*MISINFORMATION_EFFECT),
            _comparison({"domain": "cybercrime"}, 0.022, -0.03, 0.07),
        ]
    )
    section = _section(_markdown(report))
    assert (
        "| thinking | domain: misinformation | 0.250 | 10 | 0.200 | 0.100 | 0.139 | "
        "0.061–0.217 | 0.028 |"
    ) in section
    assert "cybercrime" not in section
    assert "excludes zero difference (a ratio of 1)" in section
    assert "excludes no difference" not in section


def test_each_node_is_shown_beside_its_own_comparisons_overall_effect() -> None:
    report = _effects_report(
        [
            _comparison(None, 0.028, -0.01, 0.06),
            _comparison(*MISINFORMATION_EFFECT),
            _comparison(None, -0.044, -0.09, 0.01, comparison_id="tools"),
            _comparison({"domain": "fraud"}, -0.12, -0.2, -0.04, comparison_id="tools"),
        ]
    )
    section = _section(_markdown(report))
    assert "| thinking | domain: misinformation | 0.250 | 10 |" in section
    assert "| 0.139 | 0.061–0.217 | 0.028 |" in section
    assert (
        "| tools | domain: fraud | 0.250 | 10 | 0.200 | 0.100 | -0.120 | -0.200–-0.040 | "
        "-0.044 |"
    ) in section


def test_no_node_excluding_zero_difference_is_said_plainly() -> None:
    report = _effects_report([_comparison({"domain": "cybercrime"}, 0.022, -0.03, 0.07)])
    section = _section(_markdown(report))
    assert (
        "No hierarchy node's comparison interval excludes zero difference (a ratio of 1)."
        in section
    )
    assert "no interval" not in section


def test_a_risk_ratio_node_is_judged_against_one() -> None:
    report = _effects_report(
        [
            _comparison({"domain": "a"}, 1.5, 0.8, 2.0, estimand="risk_ratio"),
            _comparison({"domain": "b"}, 0.5, 0.2, 0.9, estimand="risk_ratio"),
        ]
    )
    section = _section(_markdown(report))
    assert "domain: b" in section
    assert "domain: a" not in section


def test_a_relative_risk_reduction_node_is_judged_against_zero() -> None:
    # Below one but not above zero: judged against a ratio's 1, it would be listed.
    report = _effects_report(
        [
            _comparison({"domain": "a"}, 0.15, -0.2, 0.5, estimand="relative_risk_reduction"),
            _comparison({"domain": "b"}, 0.3, 0.1, 0.5, estimand="relative_risk_reduction"),
        ]
    )
    section = _section(_markdown(report))
    assert "domain: b" in section
    assert "domain: a" not in section


def test_an_interval_that_ends_at_the_null_does_not_exclude_it() -> None:
    report = _effects_report(
        [
            _comparison({"domain": "a"}, 0.1, 0.0, 0.2),
            _comparison({"domain": "b"}, -0.1, -0.2, 0.0),
            _comparison({"domain": "c"}, 1.2, 1.0, 1.5, estimand="risk_ratio"),
        ]
    )
    section = _section(_markdown(report))
    assert "No hierarchy node's comparison interval excludes zero difference" in section


def test_an_effect_within_rounding_of_zero_is_not_listed() -> None:
    # At three decimal places 0.0004 prints as 0.000, which on the page does not exclude zero.
    report = _effects_report([_comparison({"domain": "a"}, 0.1, 0.0004, 0.2)])
    section = _section(_markdown(report))
    assert "domain: a" not in section
    assert "No hierarchy node's comparison interval excludes zero difference" in section


def test_comparison_nodes_without_an_interval_are_counted() -> None:
    report = _effects_report(
        [
            _comparison({"domain": "a"}, None, None, None),
            _comparison({"domain": "b"}, None, None, None),
        ]
    )
    section = _section(_markdown(report))
    assert (
        "2 comparison node rows have no interval, so the report cannot say whether their "
        "effect differs from zero (a ratio of 1)."
    ) in section



def test_nodes_above_a_claims_limit_say_whether_the_interval_or_the_estimate_is_above() -> None:
    report = _limit_report(
        [
            _node("misinformation", 0.152, 0.116, 0.194),
            _node("cybercrime", 0.074, 0.049, 0.116),
            _node("fraud", 0.02, 0.005, 0.06),
        ]
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


def test_a_node_without_an_estimate_is_listed_when_its_interval_is_above_the_limit() -> None:
    # A node has no estimate when one of its prompts has no classified rollout; its
    # missingness-robust interval still places it above the limit.
    report = _limit_report([_node("misinformation", None, 0.2347, 0.99996)])
    section = _section(_markdown(report))
    assert (
        "| domain: misinformation | 0.250 | 10 | — | 0.235–1.000 | 0.050 | 0.032 | "
        "interval above the limit |"
    ) in section
    assert "No hierarchy node" not in section


def test_nodes_that_cannot_be_placed_are_counted() -> None:
    report = _limit_report(
        [
            _node("misinformation", None, 0.02, 0.4),
            _node("cybercrime", None, None, None),
            # Wholly below the limit: placed, though it has no estimate.
            _node("fraud", None, 0.001, 0.04),
        ]
    )
    section = _section(_markdown(report))
    assert "No hierarchy node's interval or estimate is above a claim's acceptable limit." in (
        section
    )
    assert (
        "Not placed for “Harmful compliance below 5%”: 1 node row with no estimate and an "
        "interval that reaches above the limit; 1 node row with neither an estimate nor an "
        "interval. Whether these nodes are above the limit is unknown."
    ) in section


def test_a_node_at_the_limit_is_not_above_it() -> None:
    report = _limit_report([_node("misinformation", 0.05, 0.05, 0.3)])
    section = _section(_markdown(report))
    assert "No hierarchy node's interval or estimate is above" in section


def test_a_node_within_rounding_of_the_limit_is_not_called_above_it() -> None:
    # 0.0502 and 0.0504 print as 0.050 at three places, as the limit does.
    report = _limit_report(
        [
            _node("misinformation", 0.0504, 0.0502, 0.3),
            _node("cybercrime", 0.074, 0.0502, 0.3),
        ]
    )
    section = _section(_markdown(report))
    assert "misinformation" not in section
    assert (
        "| domain: cybercrime | 0.250 | 10 | 0.074 | 0.050–0.300 | 0.050 | 0.032 | "
        "estimate above the limit |"
    ) in section



def test_only_the_claims_cohort_and_group_reach_its_limit() -> None:
    report = _limit_report(
        [
            _node("misinformation", 0.152, 0.116, 0.194),
            _node("cybercrime", 0.3, 0.2, 0.4, cohort="off"),
            _node("fraud", 0.3, 0.2, 0.4, group="other"),
        ]
    )
    section = _section(_markdown(report))
    assert "misinformation" in section
    assert "cybercrime" not in section
    assert "fraud" not in section


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


def test_a_comparison_claim_reaches_only_its_own_comparison() -> None:
    report = _report(
        comparisons=[
            _comparison(None, 0.028, -0.01, 0.06),
            _comparison(*MISINFORMATION_EFFECT),
            _comparison(None, 0.05, 0.01, 0.09, comparison_id="tools"),
            _comparison({"domain": "fraud"}, 0.2, 0.1, 0.3, comparison_id="tools"),
        ],
        hierarchy=[],
        claims=[_claim("increase", "Increase below 5 points")],
        claim_specs=[{"id": "increase", "comparison": "thinking"}],
    )
    limits = _limit_part(_section(_markdown(report)))
    assert "domain: misinformation" in limits
    assert "fraud" not in limits








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


def test_hidden_decisions_hide_the_limit_table() -> None:
    report = _report(
        comparisons=[_comparison(*MISINFORMATION_EFFECT)],
        hierarchy=[_node("misinformation", 0.152, 0.116, 0.194)],
        claims=[_claim("level", LEVEL_CLAIM)],
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
        claims=[_claim("level", LEVEL_CLAIM)],
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
        claims=[_claim("level", LEVEL_CLAIM)],
        claim_specs=[AGGREGATE],
    )
    assert "## Where effects sit" not in _markdown(report)
