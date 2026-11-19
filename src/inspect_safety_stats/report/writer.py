from __future__ import annotations

import json
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

import yaml

from inspect_safety_stats.config.canonicalize import canonical_value
from inspect_safety_stats.config.models import PRIMARY_MAPPING_ID
from inspect_safety_stats.report.artifacts import (
    csv_text,
    json_text,
    write_checksums,
    write_csv,
    write_csv_text,
    write_json,
    write_text,
)
from inspect_safety_stats.report.claims import CLAIMS_JSON, claims_export
from inspect_safety_stats.report.index import index_files
from inspect_safety_stats.schema.reports import (
    ClaimResult,
    ComparisonResult,
    HierarchyRiskResult,
    RiskIntervalResult,
    SafetyReport,
    VerdictRange,
    mapping_results,
)

MAPPING_SENSITIVITY_CSV = "outcome-mapping-sensitivity.csv"


def _escape(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: float | None, places: int) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"


def _verdict_range(verdict_range: VerdictRange | None, places: int) -> str:
    """State a claim's verdict range, with each boundary rounded so that the statement holds.

    The PASS boundary rounds up and the FAIL boundary down. A boundary that rounding moves is
    no longer attained, so it reads as inclusive.
    """

    if verdict_range is None:
        return "—"
    step = Decimal(1).scaleb(-places)
    parts = []
    for verdict, value, strict, name, rounding, symbols in (
        (
            "PASS",
            verdict_range.pass_acceptable_upper,
            verdict_range.pass_strict,
            "acceptable",
            ROUND_CEILING,
            (">", "≥"),
        ),
        (
            "FAIL",
            verdict_range.fail_unacceptable_lower,
            verdict_range.fail_strict,
            "unacceptable",
            ROUND_FLOOR,
            ("<", "≤"),
        ),
    ):
        if value is None:
            continue
        exact = Decimal(repr(value))
        shown = exact.quantize(step, rounding=rounding)
        symbol = symbols[0] if strict and shown == exact else symbols[1]
        parts.append(f"{verdict} at {name} {symbol} {shown}")
    return "; ".join(parts) if parts else "INCONCLUSIVE at every threshold"


def _percent(value: float | None, places: int = 1) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.{places}f}%"


def _bound_status(eligibility: str) -> str:
    return {
        "certifying": "coverage-guaranteed",
        "diagnostic": "diagnostic",
        "invalid": "unavailable",
    }.get(eligibility, eligibility)


def _role_label(interval: RiskIntervalResult) -> str:
    """The interval's role, with what a sensitivity interval varies, so that an independence
    sensitivity cannot be read as a judge-error one."""
    if interval.varied_assumption is None:
        return interval.role
    return f"{interval.role} ({interval.varied_assumption.replace('_', ' ')})"


def _coverage_label(risk: Any) -> str:
    if risk.simultaneous_coverage is None:
        return "—"
    directional_values = [
        value
        for value in (risk.lower_one_sided_coverage, risk.upper_one_sided_coverage)
        if value is not None
    ]
    directional = min(directional_values) if directional_values else None
    return f"{_percent(risk.simultaneous_coverage)} joint; {_percent(directional)} each direction"


def _profile_verdict_lines(claims: list[Any], places: int) -> list[str]:
    """The claims' verdicts under each independence profile, for claims with sensitivities."""
    profiled = [claim for claim in claims if len(claim.profile_verdicts) > 1]
    if not profiled:
        return []
    lines = [
        "",
        "### Verdicts by independence profile",
        "",
        "A claim's status is its verdict under the primary profile. Each sensitivity profile "
        "decides the same thresholds on its own interval; it shows how far the verdict rests "
        "on the primary profile's independence assumption and cannot replace it.",
        "",
        "| Claim | Profile | Role | Status | Lower | Upper | Independent blocks "
        "| Effective sample size |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for claim in profiled:
        for verdict in claim.profile_verdicts:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(claim.title),
                        f"`{_escape(verdict.independence_profile)}`",
                        verdict.role,
                        f"`{verdict.status}`",
                        _number(verdict.lower, places),
                        _number(verdict.upper, places),
                        "—"
                        if verdict.independent_block_count is None
                        else str(verdict.independent_block_count),
                        _number(verdict.effective_sample_size, 1),
                    ]
                )
                + " |"
            )
    return lines


def _judge_error_stress_lines(
    report: SafetyReport, places: int, include_decisions: bool
) -> list[str]:
    """The judge-error stress section, present only when the plan opted in."""

    stressed = [
        (claim, claim.judge_error_stress)
        for claim in report.claims
        if claim.judge_error_stress is not None
    ]
    if not stressed:
        return []
    lines = ["", "## Judge-error stress (experimental)", ""]
    if not include_decisions:
        return [
            *lines,
            "This report omits decisions (`report.include_decisions` is false), so the judge "
            "accuracy at which each verdict would change is only in `report.json`, under each "
            "claim's `judge_error_stress`.",
        ]
    lines.extend(
        [
            "Each claim below was decided on the risk the **judge reports**. With sensitivity "
            "Se and specificity Sp applied unchanged to every rollout, the verdict holds for "
            "**true** risk only on one side of a line in (Se, Sp). The columns give where that "
            "line meets the edges of perfect specificity and perfect sensitivity; a value above "
            "1 or below 0 means the condition holds along the whole edge or nowhere on it. "
            "Unclassified rollouts keep their conservative completion. Where the primary "
            "outcome's judge declares accuracy ranges, the verdict on true risk over that box "
            "is shown too. Declared ranges are assumptions: nothing here moves a claim.",
            "",
            (
                "| Claim | Status | Judge-reported bounds | PASS: Se at least (Sp = 1) | "
                "PASS: Sp at most (Se = 1) | FAIL: Se at most (Sp = 1) | "
                "FAIL: Sp at least (Se = 1) | True risk over declared accuracy | "
                "Verdict there | Robust |"
            ),
            "|---|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    reasons: list[str] = []
    for claim, stress in stressed:
        if not stress.applicable:
            reasons.append(f"- {_escape(claim.title)}: {_escape(stress.reason)}.")
            cells = ["not applicable", *["—"] * 7]
        else:
            cells = [
                f"{_number(stress.judge_reported_lower, places)}–"
                f"{_number(stress.judge_reported_upper, places)}",
                _number(stress.pass_sensitivity_at_perfect_specificity, places),
                _number(stress.pass_specificity_at_perfect_sensitivity, places),
                _number(stress.fail_sensitivity_at_perfect_specificity, places),
                _number(stress.fail_specificity_at_perfect_sensitivity, places),
                (
                    f"{_number(stress.true_risk_lower, places)}–"
                    f"{_number(stress.true_risk_upper, places)}"
                    + ("" if stress.identified else " (not identified)")
                    if stress.stressed_status is not None
                    else "no declared accuracy"
                ),
                f"`{stress.stressed_status}`" if stress.stressed_status else "—",
                "—" if stress.robust is None else "yes" if stress.robust else "**no**",
            ]
        lines.append(
            "| " + " | ".join([_escape(claim.title), f"`{claim.status}`", *cells]) + " |"
        )
    sources = sorted(
        {
            (
                stress.accuracy_source,
                stress.sensitivity_lower,
                stress.sensitivity_upper,
                stress.specificity_lower,
                stress.specificity_upper,
            )
            for _, stress in stressed
            if stress.accuracy_source is not None
        },
        key=lambda item: str(item[0]),
    )
    for source, se_low, se_high, sp_low, sp_high in sources:
        lines.extend(
            [
                "",
                f"Declared accuracy of `{_escape(source)}`: sensitivity in "
                f"[{_number(se_low, places)}, {_number(se_high, places)}], specificity in "
                f"[{_number(sp_low, places)}, {_number(sp_high, places)}].",
            ]
        )
    if reasons:
        lines.extend(["", "Not stressed:", "", *reasons])
    return lines


def _claim_status(claim: Any) -> str:
    if claim.descriptive_status is None:
        return f"`{claim.status}`"
    return f"`{claim.status}` (descriptive: `{claim.descriptive_status}`)"


_MAPPING_KIND_ORDER = {"group_risk": 0, "comparison": 1, "claim": 2}


def _mapping_sensitivity_rows(report: SafetyReport) -> list[dict[str, Any]]:
    """Every displayed result under each outcome mapping, the primary one first.

    Rows of one result are adjacent, in the primary mapping's then the plan's declared order.
    """

    readings = [(PRIMARY_MAPPING_ID, mapping_results(report))]
    readings.extend((item.id, item.results) for item in report.outcome_mapping_sensitivities)
    rows: list[tuple[tuple[int, str], int, dict[str, Any]]] = []
    for position, (mapping_id, results) in enumerate(readings):
        for result in results:
            rows.append(
                (
                    (_MAPPING_KIND_ORDER[result.kind], result.result_id),
                    position,
                    {
                        "kind": result.kind,
                        "result_id": result.result_id,
                        "cohort_id": result.cohort_id,
                        "group_id": result.group_id,
                        "measure_id": result.measure_id,
                        "mapping_id": mapping_id,
                        "estimate": result.estimate,
                        "lower": result.lower,
                        "upper": result.upper,
                        "status": result.status,
                    },
                )
            )
    return [row for _, _, row in sorted(rows, key=lambda item: (item[0], item[1]))]


def _mapping_sensitivity_markdown(
    report: SafetyReport, places: int, include_decisions: bool
) -> list[str]:
    lines = [
        "",
        "## Outcome mapping sensitivity (experimental)",
        "",
        (
            "The analysis is repeated under each declared alternative outcome mapping. Each "
            "mapping's intervals carry the coverage the plan declares for that mapping alone; "
            "they are not simultaneous across mappings. Only the primary mapping enters claims "
            "and the decision."
        ),
        "",
        "| Mapping | Title | Failure values | Safe values | Integrity |"
        + (" Decision |" if include_decisions else ""),
        "|---|---|---|---|---|" + ("---|" if include_decisions else ""),
    ]
    primary = report.presentation.primary_outcome_mapping or {}
    mappings: list[tuple[str, str | None, dict[str, Any], str, str]] = [
        (PRIMARY_MAPPING_ID, None, primary, report.integrity.status, report.decision)
    ]
    mappings.extend(
        (item.id, item.title, item.mapping, item.integrity_status, item.decision)
        for item in report.outcome_mapping_sensitivities
    )
    for mapping_id, title, mapping, integrity, decision in mappings:
        if mapping.get("failure_if") is not None:
            rule = mapping["failure_if"]
            failure = f"{rule.get('operator')} {rule.get('value')}"
            safe = "otherwise"
        else:
            failure = ", ".join(str(value) for value in mapping.get("failure_values") or [])
            safe = ", ".join(str(value) for value in mapping.get("safe_values") or [])
        cells = [
            _escape(mapping_id),
            _escape(title),
            _escape(failure),
            _escape(safe),
            f"`{integrity}`",
        ]
        if include_decisions:
            descriptive = "" if mapping_id == PRIMARY_MAPPING_ID else " (descriptive)"
            cells.append(f"`{decision}`{descriptive}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "| Result | Mapping | Estimate | Lower | Upper |"
            + (" Claim status |" if include_decisions else ""),
            "|---|---|---:|---:|---:|" + ("---|" if include_decisions else ""),
        ]
    )
    for row in _mapping_sensitivity_rows(report):
        if row["kind"] == "claim" and not include_decisions:
            continue
        cells = [
            _escape(f"{row['kind']} {row['result_id']}"),
            _escape(row["mapping_id"]),
            _number(row["estimate"], places),
            _number(row["lower"], places),
            _number(row["upper"], places),
        ]
        if include_decisions:
            cells.append(_escape(row["status"]) if row["status"] is not None else "")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Every row is in `outcome-mapping-sensitivity.csv`; a sensitivity's diagnostics are "
        "counted in `report.json`."
    )
    return lines


def _recorded(values: list[Any] | None) -> str:
    if values is None:
        return "not recorded"
    return "; ".join(
        f"`{_escape(json.dumps(value, sort_keys=True, separators=(',', ':')))}`"
        for value in values
    )


def _panel_group_lines(panel: dict[str, Any]) -> list[str]:
    """The judge panel's counts and agreement per group, for the groups the plan lists."""

    groups = panel.get("groups")
    if not groups:
        return []
    lines = [
        "",
        "### Judge panel by group",
        "",
        "The same counts and agreement over each listed group's rollouts only, so that a "
        "cohort holding several benchmarks is not summarised as one: "
        + ", ".join(f"`{_escape(item)}`" for item in groups)
        + ".",
        "",
        "| Judge | Group | Cohort | Classified | Failures | Unclassified |",
        "|---|---|---|---:|---:|---:|",
    ]
    for judge in panel["judges"]:
        for group_id in groups:
            for cohort_id, stats in sorted(judge["groups"][group_id].items()):
                lines.append(
                    f"| {_escape(judge['id'])} | {_escape(group_id)} | {_escape(cohort_id)} | "
                    f"{stats['classified']} | {stats['failures']} | {stats['unclassified']} |"
                )
    if panel["composites"]:
        lines.extend(
            [
                "",
                "| Composite | Group | Cohort | Classified | Failures | Unclassified |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for composite in panel["composites"]:
            for group_id in groups:
                for cohort_id, stats in sorted(composite["groups"][group_id].items()):
                    lines.append(
                        f"| {_escape(composite['id'])} | {_escape(group_id)} | "
                        f"{_escape(cohort_id)} | {stats['classified']} | {stats['failures']} | "
                        f"{stats['unclassified']} |"
                    )
    if panel["agreement"]:
        lines.extend(
            [
                "",
                "| Judges | Group | Cohort | Compared | Disagreements | Agreement | "
                "Cohen's kappa |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for pair in panel["agreement"]:
            for group_id in groups:
                for cohort_id, stats in sorted(pair["groups"][group_id].items()):
                    agreement, kappa = stats["agreement"], stats["kappa"]
                    lines.append(
                        f"| {_escape(pair['first'])} / {_escape(pair['second'])} | "
                        f"{_escape(group_id)} | {_escape(cohort_id)} | {stats['compared']} | "
                        f"{stats['disagreements']} | "
                        f"{'—' if agreement is None else f'{agreement:.4f}'} | "
                        f"{'—' if kappa is None else f'{kappa:.4f}'} |"
                    )
    return lines


def _judge_claim_lines(
    panel: dict[str, Any], places: int, include_decisions: bool
) -> list[str]:
    """The judge panel's per-judge claim rows and recorded configurations, when the plan
    asked for them with ``outputs.judge_claims``."""

    judges = [judge for judge in panel["judges"] if "claims" in judge]
    if not judges:
        return []
    status_header, status_rule = ("Status | ", "---|") if include_decisions else ("", "")
    lines = [
        "",
        "### Claims by judge (experimental)",
        "",
        "Each judge's rows are the claims this plan gives with that judge's verdicts as its "
        "primary outcome, over the same rollouts, without the panel or the judge calibration. "
        "They compare judge configurations. The report's decision rests on the primary outcome "
        "alone, and these rows are not adjusted for their number.",
        "",
        f"| Judge | Integrity | Claim | {status_header}Estimate | Lower | Upper | Method |",
        f"|---|---|---|{status_rule}---:|---:|---:|---|",
    ]
    for judge in judges:
        for claim in judge["claims"]:
            status = f"`{claim['status']}` | " if include_decisions else ""
            lines.append(
                f"| {_escape(judge['id'])} | `{judge['integrity']}` | "
                f"{_escape(claim['claim_id'])} | {status}"
                f"{_number(claim['estimate'], places)} | {_number(claim['lower'], places)} | "
                f"{_number(claim['upper'], places)} | {_escape(claim['method'])} |"
            )
    lines.extend(
        [
            "",
            "The scorer configuration and model roles each cohort's runs record, by judge: the "
            "options of the scorer named like the judge's score.",
            "",
            "| Judge | Cohort | Scorer | Model roles |",
            "|---|---|---|---|",
        ]
    )
    for judge in judges:
        for cohort_id, recorded in sorted(judge["configuration"].items()):
            lines.append(
                f"| {_escape(judge['id'])} | {_escape(cohort_id)} | "
                f"{_recorded(recorded['scorer'])} | {_recorded(recorded['model_roles'])} |"
            )
    return lines


_NULL_EFFECT = {"risk_difference": 0.0, "relative_risk_reduction": 0.0, "risk_ratio": 1.0}


def _excludes_null(comparison: ComparisonResult) -> bool:
    null = _NULL_EFFECT.get(comparison.estimand)
    if null is None or comparison.lower is None or comparison.upper is None:
        return False
    return comparison.lower > null or comparison.upper < null


def _path_text(report: SafetyReport, path: dict[str, str]) -> str:
    """A node's path, outermost level first.

    The order comes from the design's hierarchy, not from the dictionary: report.json stores
    its keys sorted, so a report read back from it has them in another order.
    """

    hierarchy = [str(level) for level in report.design.get("hierarchy") or []]
    levels = [level for level in hierarchy if level in path]
    levels += [level for level in path if level not in levels]
    return " → ".join(path[level] for level in levels)


def _node_label(report: SafetyReport, level: str | None, path: dict[str, str]) -> str:
    return f"{_escape(level)}: {_escape(_path_text(report, path))}"


def _localisation_lines(
    report: SafetyReport, places: int, include_decisions: bool
) -> list[str]:
    """The hierarchy nodes where an effect or a limit exceedance sits, for the first page.

    Headline rows average over the hierarchy, so an effect concentrated in one node can look
    small there. This lists the nodes a reader would otherwise find only by reading every
    hierarchy row. The section appears only when there is something to look for: hierarchy
    comparison rows, or (with decisions shown) a claim whose evidence has hierarchy rows.
    """
    node_comparisons = [item for item in report.comparisons if item.level is not None]
    claim_nodes = _claim_nodes(report) if include_decisions else []
    if not (node_comparisons or claim_nodes):
        return []
    looked_for = []
    if node_comparisons:
        looked_for.append("whose comparison interval excludes no difference")
    if claim_nodes:
        looked_for.append("whose estimate is above a claim's acceptable limit")
    lines = [
        "",
        "## Where effects sit",
        "",
        (
            "Headline rows average over the hierarchy, so an effect concentrated in one node "
            "can look small there. This section lists the hierarchy nodes "
            + ", and the nodes ".join(looked_for)
            + ". Their intervals are those of the hierarchy rows below; they are not "
            "separate claims."
        ),
    ]
    if node_comparisons:
        lines.append("")
        lines.extend(_effect_lines(report, node_comparisons, places))
    if claim_nodes:
        lines.append("")
        lines.extend(_exceedance_lines(report, claim_nodes, places))
    return lines


def _effect_lines(
    report: SafetyReport, node_comparisons: list[ComparisonResult], places: int
) -> list[str]:
    effects = [item for item in node_comparisons if _excludes_null(item)]
    if not effects:
        return ["No hierarchy node's comparison interval excludes no difference."]
    overall = {item.comparison_id: item for item in report.comparisons if item.level is None}
    lines = [
        (
            "| Comparison | Node | Design weight | Prompts | Candidate risk | "
            "Baseline risk | Effect | Confidence interval | Overall effect |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for item in effects:
        headline = overall.get(item.comparison_id)
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(item.comparison_id),
                    _node_label(report, item.level, item.path),
                    _number(item.resolved_weight, places),
                    str(item.prompt_count),
                    _number(item.candidate_estimate, places),
                    _number(item.baseline_estimate, places),
                    _number(item.estimate, places),
                    f"{_number(item.lower, places)}–{_number(item.upper, places)}",
                    _number(headline.estimate if headline else None, places),
                ]
            )
            + " |"
        )
    return lines


def _exceedance_lines(
    report: SafetyReport,
    claim_nodes: list[tuple[ClaimResult, str, list[HierarchyRiskResult | ComparisonResult]]],
    places: int,
) -> list[str]:
    rows = [
        (claim, evidence, node)
        for claim, evidence, nodes in claim_nodes
        for node in nodes
        if node.estimate is not None and node.estimate > claim.acceptable_upper
    ]
    if not rows:
        return ["No hierarchy node's estimate is above a claim's acceptable limit."]
    lines = [
        (
            "| Claim | Evidence | Node | Design weight | Prompts | Estimate | "
            "Confidence interval | Acceptable limit | Claim estimate | Position |"
        ),
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for claim, evidence, node in rows:
        above = node.lower is not None and node.lower > claim.acceptable_upper
        weight = (
            node.design_weight if isinstance(node, HierarchyRiskResult) else node.resolved_weight
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(claim.title),
                    _escape(evidence),
                    _node_label(report, node.level, node.path),
                    _number(weight, places),
                    str(node.prompt_count),
                    _number(node.estimate, places),
                    f"{_number(node.lower, places)}–{_number(node.upper, places)}",
                    _number(claim.acceptable_upper, places),
                    _number(claim.estimate, places),
                    "interval above the limit" if above else "estimate above the limit",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Node rows are judge-reported risks; any judge correction applies only to the claim.",
        ]
    )
    return lines


def _claim_nodes(
    report: SafetyReport,
) -> list[tuple[ClaimResult, str, list[HierarchyRiskResult | ComparisonResult]]]:
    """Each claim whose evidence has hierarchy rows, with a label for it and those rows.

    An aggregate claim reaches hierarchy rows only through a weighted-mean measure, the one
    estimand the hierarchy rows report.
    """
    specs = {item.claim_id: item for item in report.presentation.claim_evidence or []}
    result: list[tuple[ClaimResult, str, list[HierarchyRiskResult | ComparisonResult]]] = []
    for claim in report.claims:
        spec = specs.get(claim.claim_id)
        if spec is None:
            continue
        nodes: list[HierarchyRiskResult | ComparisonResult]
        if spec.comparison:
            evidence = f"comparison {spec.comparison}"
            nodes = [
                item
                for item in report.comparisons
                if item.level is not None and item.comparison_id == spec.comparison
            ]
        elif spec.measure_type == "weighted_mean":
            evidence = f"{spec.cohort} / {spec.group} / {spec.measure}"
            nodes = [
                item
                for item in report.hierarchy_risks
                if item.cohort_id == spec.cohort and item.group_id == spec.group
            ]
        else:
            continue
        if nodes:
            result.append((claim, evidence, nodes))
    return result


def _decision_policy_lines(report: SafetyReport, places: int) -> list[str]:
    """The decision policy's versions and the thresholds each claim was decided at."""

    record = report.decision_policy
    if record is None:
        return []
    if record.applied:
        summary = (
            "Claim thresholds come from a decision policy kept apart from the plan "
            "(experimental). Thresholds do not enter the bounds, so a policy set after the run "
            "leaves the plan's hash and the verdicts' certification as they are."
        )
    else:
        summary = (
            "A decision policy was given but not applied, because it failed a check (see the "
            "`DECISION_POLICY_*` diagnostics). The claims keep the plan's thresholds."
        )
    lines = [
        "",
        "## Decision policy",
        "",
        summary,
        "",
        f"**Inference plan SHA-256:** `{_escape(record.inference_plan_sha256)}`  ",
        f"**Policy SHA-256:** `{_escape(record.policy_sha256)}`",
        "",
        "| Version | Set at | After first evaluation | Claims | Reason |",
        "|---|---|---|---|---|",
    ]
    for version in record.versions:
        after = {True: "yes", False: "no", None: "unknown"}[version.set_after_first_evaluation]
        lines.append(
            f"| `{_escape(version.id)}` | {_escape(version.set_at)} | {after} | "
            + ", ".join(f"`{_escape(claim)}`" for claim in version.thresholds)
            + f" | {_escape(version.reason) if version.reason else '—'} |"
        )
    lines.extend(
        [
            "",
            "| Claim | Acceptable | Unacceptable | Set by | Plan's acceptable | "
            "Plan's unacceptable |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for claim in record.claims:
        lines.append(
            f"| `{_escape(claim.claim_id)}` | {_number(claim.acceptable_upper, places)} | "
            f"{_number(claim.unacceptable_lower, places)} | `{_escape(claim.set_by)}` | "
            f"{_number(claim.plan_acceptable_upper, places)} | "
            f"{_number(claim.plan_unacceptable_lower, places)} |"
        )
    return lines


def _adaptive_claim_lines(report: SafetyReport, places: int, include_decisions: bool) -> list[str]:
    """The FAIL-only claims on attackers that adapt across behaviours (experimental, #125)."""

    if not report.adaptive_claims:
        return []
    lines = [
        "",
        "## Adaptive-attacker claims (experimental)",
        "",
        (
            "Each bound is an anytime-valid lower bound on the attacker's running mean of "
            "conditional success probabilities over its runs, in the declared attack order. It "
            "needs no independence between runs, and describes this attacker at this budget "
            "only: it is no evidence about other attackers or budgets, so these claims can only "
            "FAIL."
        ),
        "",
    ]
    header = ["Claim", "Mandatory", "Runs", "Successes", "Lower", "Unacceptable", "Alpha"]
    if include_decisions:
        header.insert(1, "Status")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * (len(header) - 5) + "---:|" * 5)
    for claim in report.adaptive_claims:
        cells = [
            _escape(claim.title),
            "yes" if claim.mandatory else "no",
            "—" if claim.runs is None else str(claim.runs),
            "—" if claim.successes is None else str(claim.successes),
            _number(claim.lower, places),
            _number(claim.unacceptable_lower, places),
            _number(claim.alpha, places),
        ]
        if include_decisions:
            cells.insert(1, _claim_status(claim))
        lines.append("| " + " | ".join(cells) + " |")
    if not include_decisions:
        # The reasons state the verdict, which the plan asked the report to leave out.
        return lines
    for claim in report.adaptive_claims:
        lines.extend(["", f"`{_escape(claim.claim_id)}`:", ""])
        lines.extend(f"- {_escape(reason)}" for reason in claim.reasons)
    return lines


def _markdown(report: SafetyReport) -> str:
    places = report.presentation.decimal_places
    if places is None:
        places = 6
    include_decisions = report.presentation.include_decisions
    lines = [
        f"# {_escape(report.analysis.get('title', report.analysis.get('id', 'Safety report')))}",
        "",
        f"**Analysis ID:** `{_escape(report.analysis.get('id'))}`  ",
        f"**Plan SHA-256:** `{_escape(report.analysis.get('plan_sha256'))}`  ",
        f"**Integrity:** `{report.integrity.status}`",
        "",
        "## Scope",
        "",
        str(report.scope["disclaimer"]),
        "",
        f"**Safety estimand:** `{_escape(report.scope.get('estimand'))}`",
    ]
    establish = report.scope.get("what_results_establish")
    if establish:
        # Reports written before the link existed have none.
        lines.extend(["", f"What these results establish, and what they do not: <{establish}>"])
    if include_decisions:
        lines.insert(2, f"**Decision:** `{report.decision}`  ")
        lines.extend(
            [
                "",
                "## Mandatory claims",
                "",
                (
                    "| Claim | Status | Estimate | Lower | Upper | Acceptable | Unacceptable | "
                    "Verdict at other thresholds |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        mandatory = [claim for claim in report.claims if claim.mandatory]
        if mandatory:
            for claim in mandatory:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _escape(claim.title),
                            _claim_status(claim),
                            _number(claim.estimate, places),
                            _number(claim.lower, places),
                            _number(claim.upper, places),
                            _number(claim.acceptable_upper, places),
                            _number(claim.unacceptable_lower, places),
                            _verdict_range(claim.verdict_range, places),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("| _No mandatory claims_ | `INCONCLUSIVE` | — | — | — | — | — | — |")
        if any(claim.descriptive_status is not None for claim in mandatory):
            lines.extend(
                [
                    "",
                    (
                        "This analysis is exploratory, so no claim is certified. A descriptive "
                        "status is what a confirmatory analysis of the same bounds would give at "
                        "the plan's thresholds, and the verdict at other thresholds is what it "
                        "would give at those. Both are shown for reading the bounds and certify "
                        "nothing."
                    ),
                ]
            )
        lines.extend(_profile_verdict_lines(mandatory, places))
    lines.extend(_adaptive_claim_lines(report, places, include_decisions))
    lines.extend(_decision_policy_lines(report, places))
    lines.extend(_judge_error_stress_lines(report, places, include_decisions))
    lines.extend(_localisation_lines(report, places, include_decisions))
    lines.extend(["", "## Data integrity", ""])
    if report.integrity.diagnostics:
        lines.extend(
            [
                "| Severity | Code | Scope | Message |",
                "|---|---|---|---|",
            ]
        )
        for diagnostic in report.integrity.diagnostics:
            lines.append(
                f"| {diagnostic.severity} | `{_escape(diagnostic.code)}` | "
                f"{_escape(diagnostic.scope)} | {_escape(diagnostic.message)} |"
            )
    else:
        lines.append("No integrity diagnostics were emitted.")
    lines.extend(
        [
            "",
            "## Fixed-design risk measures",
            "",
            (
                "Ranges are finite-sample confidence bounds, not standard errors or bootstrap "
                "intervals. Coverage is the conservative floor stated for each method. "
                "Where outcomes are unclassified, reported missingness-robust bounds cover "
                "both the all-safe and all-failure completions."
            ),
            *_interval_methods_markdown(report),
            "",
            (
                "| Cohort | Group | Measure | Parameters | Estimate | Lower bound | Upper bound | "
                "Coverage floor | Bound status | Uncertainty method |"
            ),
            "|---|---|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for risk in report.group_risks:
        parameters = ", ".join(f"{key}={value}" for key, value in risk.parameters.items()) or "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(risk.cohort_id),
                    _escape(risk.group_id),
                    _escape(risk.measure_id),
                    _escape(parameters),
                    _number(risk.estimate, places),
                    _number(risk.lower, places),
                    _number(risk.upper, places),
                    _coverage_label(risk),
                    _bound_status(risk.eligibility),
                    _escape(risk.interval_method or "unavailable"),
                ]
            )
            + " |"
        )
    alternatives = [
        (risk, interval)
        for risk in report.group_risks
        for interval in risk.intervals
        if len(risk.intervals) > 1
    ]
    if alternatives:
        lines.extend(
            [
                "",
                "## Interval-method comparison",
                "",
                (
                    "The primary interval is used in the main risk table. Reference and "
                    "sensitivity intervals show prespecified alternative constructions; their "
                    "coverage statements apply construction-by-construction."
                ),
                "",
                (
                    "| Cohort | Group | Measure | Role | Method | Independence profile | "
                    "Lower bound | Upper bound | Completion estimate range | Coverage floor | "
                    "Scope | Bound status |"
                ),
                "|---|---|---|---|---|---|---:|---:|---|---|---|---|",
            ]
        )
        for risk, interval in alternatives:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(risk.cohort_id),
                        _escape(risk.group_id),
                        _escape(risk.measure_id),
                        _escape(_role_label(interval)),
                        _escape(interval.method),
                        _escape(interval.independence_profile),
                        _number(interval.lower, places),
                        _number(interval.upper, places),
                        (
                            f"{_number(interval.completion_estimate_lower, places)}–"
                            f"{_number(interval.completion_estimate_upper, places)}"
                            if interval.completion_estimate_lower is not None
                            else "—"
                        ),
                        _coverage_label(interval),
                        _escape(interval.coverage_scope),
                        _bound_status(interval.eligibility),
                    ]
                )
                + " |"
            )
    panel = report.provenance.get("judge_panel")
    if panel:
        lines.extend(
            [
                "",
                "## Judge panel",
                "",
                "Several judges were declared for every rollout. Judges sharing a declared block "
                "are one opinion, not independent votes, and **effective votes** is the Kish count "
                "over blocks. Agreement figures are descriptive diagnostics: judges that share a "
                "blind spot agree perfectly and are still both wrong.",
                "",
                "| Judge | Kind | Block | Score | Cohort | Classified | Failures | Unclassified |",
                "|---|---|---|---|---|---:|---:|---:|",
            ]
        )
        for judge in panel["judges"]:
            for cohort_id, stats in sorted(judge["cohorts"].items()):
                lines.append(
                    f"| {_escape(judge['id'])} | {judge['kind']} | {_escape(judge['block'])} | "
                    f"{_escape(judge['score'])} | {_escape(cohort_id)} | {stats['classified']} | "
                    f"{stats['failures']} | {stats['unclassified']} |"
                )
        if panel["composites"]:
            lines.extend(
                [
                    "",
                    "| Composite | Rule | Judges | Effective votes | Cohort | Classified | "
                    "Failures | Unclassified |",
                    "|---|---|---|---:|---|---:|---:|---:|",
                ]
            )
            for composite in panel["composites"]:
                for cohort_id, stats in sorted(composite["cohorts"].items()):
                    lines.append(
                        f"| {_escape(composite['id'])} | {composite['rule']} | "
                        f"{_escape(', '.join(composite['judges']))} | "
                        f"{composite['effective_votes']:.3g} | {_escape(cohort_id)} | "
                        f"{stats['classified']} | {stats['failures']} | {stats['unclassified']} |"
                    )
        if panel["agreement"]:
            lines.extend(
                [
                    "",
                    "| Judges | Cohort | Compared | Disagreements | Agreement | Cohen's kappa |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for pair in panel["agreement"]:
                for cohort_id, stats in sorted(pair["cohorts"].items()):
                    agreement = stats["agreement"]
                    kappa = stats["kappa"]
                    lines.append(
                        f"| {_escape(pair['first'])} / {_escape(pair['second'])} | "
                        f"{_escape(cohort_id)} | {stats['compared']} | {stats['disagreements']} | "
                        f"{'—' if agreement is None else f'{agreement:.4f}'} | "
                        f"{'—' if kappa is None else f'{kappa:.4f}'} |"
                    )
        lines.extend(_panel_group_lines(panel))
        lines.extend(_judge_claim_lines(panel, places, include_decisions))
    judge_rows = [
        (risk, interval)
        for risk in report.group_risks
        for interval in risk.intervals
        if interval.interval_id.startswith("judge_corrected_")
    ]
    if judge_rows:
        # Corrected rows can come without a plan calibration (a judge panel's derived
        # correction); they read as the assumed mode with no declared scenarios.
        calibration = report.presentation.judge_calibration
        mode = calibration.mode if calibration else None
        primary_scenario = (calibration.primary if calibration else None) or ""
        validated = mode == "validated"
        transported = mode == "transported"
        stratified = (
            validated
            and calibration is not None
            and calibration.sampling == "stratified_within_judge_stratum"
        )
        preamble = (
            (
                "Experimental. Only the judge's sensitivity is taken from a published "
                "human-labelled set, on the assumption that the same judge and rubric are at "
                "least as sensitive on this run's outputs as on that set's, which came from "
                "other generators. The set's false-positive rate is not transported: this "
                "run's false positives are bounded only by its own judged failures, so the "
                "lower bound is zero and only the upper bound is informative. The rows below "
                "carry no coverage guarantee and cannot support a safety claim."
            )
            if transported
            else (
                "Experimental. These accuracies bound each judge stratum's truly unsafe count "
                "by summing exact hypergeometric bounds over its declared fine strata, each "
                "weighted by its size, from human labels drawn without replacement within "
                "each fine stratum. They then bound the conditional accuracy at those counts "
                "with exact Clopper-Pearson bounds, since a run's rollouts only sample it. The "
                "fine strata weight every classified rollout of a cohort equally, so they certify "
                "only a weighted mean that does the same, over the whole cohort in a fixed "
                "design. Those rows are coverage-guaranteed at the union of the underlying "
                "interval's budget and the calibration budget; every other corrected row is a "
                "sensitivity analysis, marked diagnostic below. Validity rests on each fine "
                "stratum's labelled rows being a simple random sample of it, of a size fixed "
                "independently of the labels, which is a property of the collection protocol."
            )
            if stratified
            else (
                "These accuracies are exact Clopper-Pearson bounds from human labels drawn "
                "within each judge stratum. The labels weight every classified rollout of a "
                "cohort equally, so they certify only a weighted mean that does the same, over "
                "the whole cohort in a fixed design. Those rows are coverage-guaranteed at the "
                "union of the underlying interval's budget and the calibration budget; every "
                "other corrected row is a sensitivity analysis, marked diagnostic below. "
                "Validity rests on the labelled rows being a simple random sample within "
                "their stratum, which is a property of the collection protocol."
            )
            if validated
            else (
                "These accuracies are assumptions, not measurements. The rows below carry no "
                "coverage guarantee and cannot support a safety claim. They exist to make an "
                "otherwise invisible dependency explicit: an uncorrected report implicitly "
                "assumes the judge never errs."
            )
        )
        scoped = calibration.groups if calibration else None
        if scoped:
            preamble += (
                " The calibration is scoped to the groups "
                + ", ".join(f"`{_escape(item)}`" for item in scoped)
                + (
                    ": each cohort has one calibration per group, from the labels on that "
                    "group's rollouts"
                    if validated
                    else ": the declared accuracies apply to those groups' rows only"
                )
                + ", and rows of other groups are not corrected."
            )
        lines.extend(
            [
                "",
                (
                    "## Judge misclassification correction"
                    if validated
                    else "## Judge misclassification sensitivity"
                ),
                "",
                (
                    "Every other interval in this report estimates the risk that the **judge "
                    "reports** a failure. This section maps those intervals onto the underlying "
                    "**true** failure risk using "
                    + (
                        "a judge sensitivity transported from a published validation set."
                        if transported
                        else ("measured" if validated else "declared")
                        + " conditional judge accuracies."
                    )
                ),
                "",
                preamble,
                "",
                (
                    "| Cohort | Group | Measure | Scenario | Judge-system bounds | "
                    "True-risk bounds | Corrected estimate range | Bound status |"
                ),
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for risk, interval in judge_rows:
            scenario = interval.interval_id.removeprefix("judge_corrected_")
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(risk.cohort_id),
                        _escape(risk.group_id),
                        _escape(risk.measure_id),
                        f"`{_escape(scenario)}`"
                        + (" (primary)" if scenario == primary_scenario else ""),
                        f"{_number(interval.raw_lower, places)}–"
                        f"{_number(interval.raw_upper, places)}",
                        f"{_number(interval.lower, places)}–{_number(interval.upper, places)}",
                        (
                            f"{_number(interval.completion_estimate_lower, places)}–"
                            f"{_number(interval.completion_estimate_upper, places)}"
                            if interval.completion_estimate_lower is not None
                            else "—"
                        ),
                        _bound_status(interval.eligibility),
                    ]
                )
                + " |"
            )
        if validated:
            cohorts = (report.provenance.get("judge_calibration") or {}).get("cohorts") or {}
            if cohorts and scoped:
                lines.extend(
                    [
                        "",
                        "Human labels behind each cohort's calibration for each group:",
                        "",
                        (
                            "| Cohort | Group | Judge-unsafe labelled / stratum | Truly unsafe | "
                            "Judge-safe labelled / stratum | Truly unsafe |"
                        ),
                        "|---|---|---|---:|---|---:|",
                    ]
                )
                for cohort_id, by_group in cohorts.items():
                    for group_id, strata in by_group.items():
                        unsafe, safe = strata["judge_unsafe"], strata["judge_safe"]
                        lines.append(
                            "| "
                            + " | ".join(
                                [
                                    _escape(cohort_id),
                                    _escape(group_id),
                                    f"{unsafe['labelled']} / {unsafe['stratum_size']}",
                                    str(unsafe["truly_unsafe"]),
                                    f"{safe['labelled']} / {safe['stratum_size']}",
                                    str(safe["truly_unsafe"]),
                                ]
                            )
                            + " |"
                        )
            elif cohorts:
                lines.extend(
                    [
                        "",
                        "Human labels behind each cohort's calibration:",
                        "",
                        (
                            "| Cohort | Judge-unsafe labelled / stratum | Truly unsafe | "
                            "Judge-safe labelled / stratum | Truly unsafe |"
                        ),
                        "|---|---|---:|---|---:|",
                    ]
                )
                for cohort_id, strata in cohorts.items():
                    unsafe, safe = strata["judge_unsafe"], strata["judge_safe"]
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                _escape(cohort_id),
                                f"{unsafe['labelled']} / {unsafe['stratum_size']}",
                                str(unsafe["truly_unsafe"]),
                                f"{safe['labelled']} / {safe['stratum_size']}",
                                str(safe["truly_unsafe"]),
                            ]
                        )
                        + " |"
                    )
            # A scoped calibration nests its summaries one level deeper, by group.
            scoped_summaries = [
                (cohort_id, group_id, strata)
                for cohort_id, by_group in cohorts.items()
                for group_id, strata in (by_group.items() if scoped else [(None, by_group)])
            ]
            fine_rows = [
                (cohort_id, group_id, item)
                for cohort_id, group_id, strata in scoped_summaries
                for item in strata.get("fine_strata") or []
            ]
            if fine_rows:
                lines.extend(
                    [
                        "",
                        "Declared fine strata, with exact bounds on each one's truly unsafe "
                        "count (a census knows its count; an unlabelled stratum spans all its "
                        "rows). The accuracies are Clopper-Pearson bounds at the summed counts:",
                        "",
                        "| Cohort | "
                        + ("Group | " if scoped else "")
                        + "Judge stratum | Fine stratum | Labelled / size | Truly unsafe | "
                        "Truly unsafe count bounds |",
                        "|---|" + ("---|" if scoped else "") + "---|---|---|---:|---|",
                    ]
                )
                for cohort_id, group_id, item in fine_rows:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                _escape(cohort_id),
                                *([_escape(group_id)] if scoped else []),
                                _escape(item["judge_stratum"]),
                                _escape(item["stratum"]),
                                f"{item['labelled']} / {item['stratum_size']}",
                                str(item["truly_unsafe"]),
                                f"{item['truly_unsafe_lower']}–{item['truly_unsafe_upper']}",
                            ]
                        )
                        + " |"
                    )
        elif transported:
            summary = report.provenance.get("judge_calibration") or {}
            published = summary.get("published_set") or {}
            if published:
                lines.extend(
                    [
                        "",
                        "Published validation set behind the transported sensitivity:",
                        "",
                        f"- Set: {_escape(published.get('name'))} "
                        f"({_escape(published.get('source'))})",
                        f"- Judge: {_escape(published.get('judge'))}; rubric: "
                        f"{_escape(published.get('rubric'))}",
                        f"- Judge flagged {published.get('judge_flagged_unsafe')} of "
                        f"{published.get('human_unsafe')} human-unsafe items; sensitivity "
                        f"at least {_number(summary.get('sensitivity_lower'), places)} "
                        "(one-sided Clopper-Pearson, alpha "
                        f"{published.get('sensitivity_alpha')})",
                    ]
                )
        else:
            scenarios = calibration.scenarios if calibration else []
            if scenarios:
                lines.extend(["", "Declared scenarios:", ""])
                for item in scenarios:
                    lines.append(
                        f"- `{_escape(item.id)}`: "
                        f"P(truly unsafe | judge said unsafe) in "
                        f"[{item.rho_1.lower}, {item.rho_1.upper}]; "
                        f"P(truly unsafe | judge said safe) in "
                        f"[{item.rho_0.lower}, {item.rho_0.upper}]"
                        + ("" if item.cohorts is None else ", for every cohort not listed below")
                    )
                    for cohort_id, box in (item.cohorts or {}).items():
                        lines.append(
                            f"  - cohort `{_escape(cohort_id)}`: "
                            f"P(truly unsafe | judge said unsafe) in "
                            f"[{box.rho_1.lower}, {box.rho_1.upper}]; "
                            f"P(truly unsafe | judge said safe) in "
                            f"[{box.rho_0.lower}, {box.rho_0.upper}]"
                        )
    if report.hierarchy_risks:
        lines.extend(
            [
                "",
                "## Hierarchy marginal risks",
                "",
                (
                    "Node estimates use the declared fixed-design weights normalized within each "
                    "node. When enabled, displayed endpoints are projections of one common "
                    "hierarchy-constrained confidence region. Coverage guarantees apply "
                    "construction-by-construction."
                ),
                "",
                (
                    "| Cohort | Group | Level | Node path | Design weight | Prompts | Estimate | "
                    "Primary CI | Prompt-cluster sensitivity | Method | Calibration |"
                ),
                "|---|---|---|---|---:|---:|---:|---|---|---|---|",
            ]
        )
        for node in report.hierarchy_risks:
            primary = next(
                (interval for interval in node.intervals if interval.role == "primary"),
                None,
            )
            cluster = next(
                (
                    interval
                    for interval in node.intervals
                    if interval.role == "sensitivity" and interval.method == node.interval_method
                ),
                None,
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(node.cohort_id),
                        _escape(node.group_id),
                        _escape(node.level),
                        _escape(_path_text(report, node.path)),
                        _number(node.design_weight, places),
                        str(node.prompt_count),
                        _number(node.estimate, places),
                        f"{_number(node.lower, places)}–{_number(node.upper, places)}",
                        (
                            f"{_number(cluster.lower, places)}–{_number(cluster.upper, places)}"
                            if cluster is not None
                            else "—"
                        ),
                        _escape(node.interval_method),
                        (
                            f"frontier_sidak(k={primary.calibration_frontier_size})"
                            if primary is not None
                            and primary.calibration_method == "frontier_sidak"
                            else _escape(
                                primary.calibration_method if primary is not None else None
                            )
                        ),
                    ]
                )
                + " |"
            )
    single_valued = [
        (risk.cohort_id, risk.group_id, "—", interval)
        for risk in report.group_risks
        for interval in risk.intervals
        if interval.single_valued_bound is not None
    ] + [
        (node.cohort_id, node.group_id, _path_text(report, node.path), interval)
        for node in report.hierarchy_risks
        for interval in node.intervals
        if interval.single_valued_bound is not None
    ]
    if single_valued:
        lines.extend(
            [
                "",
                "## Single-valued bounds (experimental)",
                "",
                (
                    "M is the smallest U + δ over one-sided upper bounds U at level δ for each "
                    "node alone. It needs no alpha: P(M < R) ≤ R, so its miss probability "
                    "equals the unknown risk R. It is not simultaneous over nodes and is not a "
                    "gate. It is meant as one input per component of a system model, which "
                    "stays conservative only if the model is monotone in it."
                ),
                "",
                "| Cohort | Group | Node path | Method | Independence profile | M | δ at M |",
                "|---|---|---|---|---|---:|---:|",
            ]
        )
        for cohort_id, group_id, path, interval in single_valued:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(cohort_id),
                        _escape(group_id),
                        _escape(path),
                        _escape(interval.method),
                        _escape(interval.independence_profile),
                        _number(interval.single_valued_bound, places),
                        _number(interval.single_valued_bound_delta, places),
                    ]
                )
                + " |"
            )
    # The column shows only when the plan's score.token_limit looks for such answers.
    token_limits = any(item.n_token_limit is not None for item in report.prompt_estimates)
    lines.extend(
        [
            "",
            "## Prompt-level evidence",
            "",
            "The complete normalized table is in `observations.csv`.",
            "",
            "| Cohort | Prompt | Failures | Valid | Unclassified | "
            + ("Token limit | " if token_limits else "")
            + "Estimate | Lower | Upper |",
            "|---|---|---:|---:|---:|" + ("---:|" if token_limits else "") + "---:|---:|---:|",
        ]
    )
    for estimate in report.prompt_estimates:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(estimate.cohort_id),
                    _escape(estimate.prompt_id),
                    str(estimate.n_failure),
                    str(estimate.n_valid),
                    str(estimate.n_unclassified),
                    *([str(estimate.n_token_limit or 0)] if token_limits else []),
                    _number(estimate.p_hat, places),
                    _number(estimate.lower, places),
                    _number(estimate.upper, places),
                ]
            )
            + " |"
        )
    if report.comparisons:
        overall_comparisons = [item for item in report.comparisons if item.level is None]
        hierarchy_comparisons = [item for item in report.comparisons if item.level is not None]
        lines.extend(
            [
                "",
                "## Model and configuration comparisons",
                "",
                (
                    "Effects use the same fixed prompts and declared weights in both arms. "
                    "Negative risk differences mean the candidate has lower failure risk. "
                    "Direct intervals quantify rollout uncertainty and do not treat prompts "
                    "as a sampled population."
                ),
            ]
        )
        for comparison_set in report.comparison_sets:
            lines.extend(
                [
                    "",
                    f"### Comparison set: {_escape(comparison_set.comparison_set_id)}",
                    "",
                    (
                        f"{_escape(comparison_set.contrast_type)}; "
                        f"{100 * comparison_set.simultaneous_coverage:.1f}% simultaneous "
                        f"coverage; primary independence profile "
                        f"`{_escape(comparison_set.independence_profile)}`."
                    ),
                    "",
                    "| Model / configuration | Risk estimate | Joint risk interval | Rank set |",
                    "|---|---:|---|---|",
                ]
            )
            for cohort in comparison_set.cohorts:
                arm_interval = comparison_set.arm_intervals.get(cohort)
                rank = comparison_set.rank_intervals.get(cohort)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _escape(cohort),
                            _number(comparison_set.arm_estimates.get(cohort), places),
                            (
                                f"{_number(arm_interval[0], places)}–"
                                f"{_number(arm_interval[1], places)}"
                                if arm_interval
                                else "—"
                            ),
                            f"{rank[0]}–{rank[1]}" if rank else "—",
                        ]
                    )
                    + " |"
                )
        lines.extend(
            [
                "",
                "### Requested contrasts" if report.comparison_sets else "",
                "" if report.comparison_sets else "",
                (
                    "| Comparison | Candidate | Baseline | Group | Measure | Estimand | "
                    "Candidate risk | Baseline risk | Effect | Confidence interval | "
                    "Prompts | Arm rollouts | Coverage floor | Method | Bound status |"
                ),
                "|---|---|---|---|---|---|---:|---:|---:|---|---:|---|---|---|---|",
            ]
        )
        for comparison in overall_comparisons:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(comparison.comparison_id),
                        _escape(comparison.candidate_cohort),
                        _escape(comparison.baseline_cohort),
                        _escape(comparison.group_id),
                        _escape(comparison.measure_id),
                        _escape(comparison.estimand),
                        _number(comparison.candidate_estimate, places),
                        _number(comparison.baseline_estimate, places),
                        _number(comparison.estimate, places),
                        f"{_number(comparison.lower, places)}–{_number(comparison.upper, places)}",
                        str(comparison.prompt_count),
                        (
                            f"{comparison.candidate_rollout_count} / "
                            f"{comparison.baseline_rollout_count}"
                        ),
                        _coverage_label(comparison),
                        _escape(comparison.interval_method or comparison.method),
                        _bound_status(comparison.eligibility),
                    ]
                )
                + " |"
            )
        if hierarchy_comparisons:
            lines.extend(
                [
                    "",
                    "### Comparison hierarchy",
                    "",
                    (
                        "| Comparison | Level | Node path | Design weight | Prompts | "
                        "Candidate risk | Baseline risk | Effect | Confidence interval | Method |"
                    ),
                    "|---|---|---|---:|---:|---:|---:|---:|---|---|",
                ]
            )
            for comparison in hierarchy_comparisons:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _escape(comparison.comparison_id),
                            _escape(comparison.level),
                            _escape(_path_text(report, comparison.path)),
                            _number(comparison.resolved_weight, places),
                            str(comparison.prompt_count),
                            _number(comparison.candidate_estimate, places),
                            _number(comparison.baseline_estimate, places),
                            _number(comparison.estimate, places),
                            (
                                f"{_number(comparison.lower, places)}–"
                                f"{_number(comparison.upper, places)}"
                            ),
                            _escape(comparison.interval_method or comparison.method),
                        ]
                    )
                    + " |"
                )
        alternative_comparisons = [
            (comparison, interval)
            for comparison in report.comparisons
            for interval in comparison.intervals
            if interval.role != "primary"
        ]
        if alternative_comparisons:
            lines.extend(
                [
                    "",
                    "### Comparison interval sensitivity",
                    "",
                    (
                        "| Result | Role | Method | Independence profile | Lower | Upper | "
                        "Coverage scope | Bound status |"
                    ),
                    "|---|---|---|---|---:|---:|---|---|",
                ]
            )
            for comparison, interval in alternative_comparisons:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _escape(comparison.result_id),
                            _escape(_role_label(interval)),
                            _escape(interval.method),
                            _escape(interval.independence_profile),
                            _number(interval.lower, places),
                            _number(interval.upper, places),
                            _escape(interval.coverage_scope),
                            _bound_status(interval.eligibility),
                        ]
                    )
                    + " |"
                )
    if report.outcome_mapping_sensitivities:
        lines.extend(_mapping_sensitivity_markdown(report, places, include_decisions))
    if include_decisions and report.policy_events:
        lines.extend(["", "## Policy events", ""])
        for event in report.policy_events:
            lines.append(
                f"- `{_escape(event.get('type'))}` on prompt `{_escape(event.get('prompt_id'))}` "
                f"in cohort `{_escape(event.get('cohort_id'))}`."
            )
    lines.extend(
        [
            "",
            "## Multiplicity",
            "",
        ]
    )
    for family in report.multiplicity:
        lines.append(
            f"- `{_escape(family['id'])}`: {_escape(family['method'])}, "
            f"alpha={_escape(family['alpha'])}, purpose={_escape(family['purpose'])}."
        )
    lines.extend(_defaults_markdown(report))
    lines.extend(_limitations_markdown(report))
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"Generated by `inspect-safety-stats` {report.package_version}.",
            "",
        ]
    )
    for source in report.provenance.get("sources", []):
        lines.append(
            f"- `{_escape(source.get('path'))}` — SHA-256 `{_escape(source.get('sha256'))}`"
        )
    return "\n".join(lines) + "\n"


def _interval_methods_markdown(report: SafetyReport) -> list[str]:
    """The methods and independence profiles of the risk intervals, with their assumptions.

    Read from the intervals themselves (#64), in the order they first appear among the group
    and hierarchy risks, so the text names what this report used and nothing else.
    """

    used: dict[tuple[str, str | None], list[str]] = {}
    interval_lists = [risk.intervals for risk in report.group_risks]
    interval_lists.extend(node.intervals for node in report.hierarchy_risks)
    for intervals in interval_lists:
        for interval in intervals:
            key = (interval.method, interval.independence_profile)
            used.setdefault(key, interval.assumptions)
    if not used:
        return []
    profiles = report.design.get("independence_profiles") or {}
    lines = ["", "The intervals rest on these methods and assumptions:", ""]
    for (method, profile), assumptions in used.items():
        heading = f"- `{_escape(method)}`"
        if profile is not None:
            block_by = (profiles.get(profile) or {}).get("block_by") or []
            blocks = ", ".join(f"`{_escape(field)}`" for field in block_by)
            heading += f" under independence profile `{_escape(profile)}`"
            heading += f", which blocks by {blocks}" if blocks else ""
        lines.append(heading + (":" if assumptions else ", which records no assumptions."))
        lines.extend(f"  - {_escape(assumption)}" for assumption in assumptions)
    return lines


def _limitations_markdown(report: SafetyReport) -> list[str]:
    """The "Limitations" section: the assumptions this analysis used, as report.json lists them.

    Each assumption comes with the plan settings it took, marked as defaults or as written in
    the plan, and with what follows if it is wrong.
    """

    lines = ["", "## Limitations", ""]
    if report.assumptions_used:
        lines.extend(
            [
                "The results rest on these assumptions. Each names the plan fields that set it, "
                "with the value this analysis used and whether the plan wrote it or left it to "
                "its default.",
                "",
            ]
        )
    for assumption in report.assumptions_used:
        settings = "; ".join(
            f"`{_escape(setting.path)}`: `{_escape(json.dumps(setting.value, sort_keys=True))}` "
            f"({'default' if setting.default else 'written in the plan'})"
            for setting in assumption.settings
        )
        lines.append(
            f"- **{_escape(assumption.id)}.** {_escape(assumption.statement)} "
            + (f"Set by {settings}. " if settings else "No plan field sets it. ")
            + f"*If it is wrong:* {_escape(assumption.if_wrong)}"
        )
    lines.append("- Outputs marked diagnostic cannot independently produce a safety pass.")
    return lines


def _sets_a_value(value: Any) -> bool:
    """Whether a default sets something, rather than leaving a feature off or a field unused."""

    return value not in (None, False, [], {})


def _defaults_markdown(report: SafetyReport) -> list[str]:
    """The "Defaults applied" section: the plan fields the analysis filled with defaults.

    It lists the defaults that set a value and counts the rest, which leave an optional
    feature off or a parameter unused (None, false or empty); report.json lists them all.
    """

    if not report.plan_defaults:
        return []
    setting = [entry for entry in report.plan_defaults if _sets_a_value(entry.value)]
    lines = [
        "",
        "## Defaults applied",
        "",
        "The plan left these fields out, and the analysis used their defaults. The plan hash "
        "covers only what the plan wrote, so a release that changes one of these defaults "
        "changes the analysis but not the hash; writing a field into the plan fixes its value.",
        "",
    ]
    lines.extend(
        # Sorted keys, as report.json stores them, so a report re-rendered from it is the same.
        f"- `{_escape(entry.path)}`: `{_escape(json.dumps(entry.value, sort_keys=True))}`"
        for entry in setting
    )
    rest = len(report.plan_defaults) - len(setting)
    if rest:
        lines.extend(
            [
                "",
                f"{rest} further {'default leaves' if rest == 1 else 'defaults leave'} an "
                "optional feature off or a parameter unused (None, false or empty). "
                "`plan_defaults` in report.json lists every default.",
            ]
        )
    return lines


def _design_rows(report: SafetyReport) -> list[dict[str, Any]]:
    """The rows of design.resolved.csv, with their columns in the design's order.

    report.json stores the rows with sorted keys, so the order comes from the hierarchy:
    the group, the hierarchy levels, the task, question and prompt, then the weight.
    """

    hierarchy = [str(item) for item in report.design.get("hierarchy") or []]
    columns = [
        "group_id",
        *dict.fromkeys([*hierarchy, "task_id", "question_id", "prompt_id"]),
        "weight",
    ]
    return [
        {column: row[column] for column in columns if column in row}
        for row in report.design.get("prompts") or []
    ]


def _hierarchy_rows(report: SafetyReport) -> list[dict[str, Any]]:
    return [
        {
            "cohort_id": node.cohort_id,
            "group_id": node.group_id,
            "level": node.level,
            "path": node.path,
            "design_weight": node.design_weight,
            "prompt_count": node.prompt_count,
            "estimate": node.estimate,
            "lower": node.lower,
            "upper": node.upper,
            "raw_lower": node.raw_lower,
            "raw_upper": node.raw_upper,
            "projection_method": node.projection_method,
            "interval_method": node.interval_method,
            "calibration_method": (
                node.intervals[0].calibration_method if node.intervals else None
            ),
            "calibration_frontier": (
                node.intervals[0].calibration_frontier if node.intervals else None
            ),
            "calibration_frontier_alpha": (
                node.intervals[0].calibration_frontier_alpha if node.intervals else None
            ),
            "calibration_frontier_size": (
                node.intervals[0].calibration_frontier_size if node.intervals else None
            ),
            "endpoint_alpha": (node.intervals[0].alpha_upper if node.intervals else None),
            "eligibility": node.eligibility,
            "intervals": [item.model_dump(mode="json") for item in node.intervals],
        }
        for node in report.hierarchy_risks
    ]


def _rendered(report: SafetyReport) -> dict[str, str]:
    """The report's renderings, by file name: functions of the report model alone.

    A report read back from its report.json renders to the same text (L38). plan.resolved.yaml
    and observations.csv are not among them: they copy the analysis input.
    """

    from inspect_safety_stats.report.summary import summary_markdown
    from inspect_safety_stats.report.visualization import render_visualization

    return {
        "report.md": summary_markdown(report),
        "report-full.md": _markdown(report),
        "report.html": render_visualization(report),
        "design.resolved.csv": csv_text(_design_rows(report)),
        "hierarchy-risks.csv": csv_text(_hierarchy_rows(report)),
        "comparisons.csv": csv_text(
            [item.model_dump(mode="json") for item in report.comparisons]
        ),
        "comparison-sets.csv": csv_text(
            [item.model_dump(mode="json") for item in report.comparison_sets]
        ),
        **(
            {MAPPING_SENSITIVITY_CSV: csv_text(_mapping_sensitivity_rows(report))}
            if report.outcome_mapping_sensitivities
            else {}
        ),
        **(
            {
                CLAIMS_JSON: json_text(
                    canonical_value(claims_export(report).model_dump(mode="json"))
                )
            }
            if report.presentation.claims_export
            else {}
        ),
    }


def write_report(report: SafetyReport, output: str | Path, *, overwrite: bool = False) -> Path:
    directory = Path(output).resolve()
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"report output exists and is not a directory: {directory}")
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"report directory already exists and is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "report.json", canonical_value(report.model_dump(mode="json")))
    rendered = _rendered(report)
    for name in ("report.md", "report-full.md", "report.html"):
        write_text(directory / name, rendered[name])
    write_text(
        directory / "plan.resolved.yaml",
        yaml.safe_dump(canonical_value(report._resolved_plan), sort_keys=True, allow_unicode=True),
    )
    for name in (
        "design.resolved.csv",
        "hierarchy-risks.csv",
        "comparisons.csv",
        "comparison-sets.csv",
    ):
        write_csv_text(directory / name, rendered[name])
    include_observations = bool(
        report._resolved_plan.get("report", {}).get("include_observations", True)
    )
    if include_observations:
        write_csv(
            directory / "observations.csv",
            [item.model_dump(mode="json") for item in report._observations],
        )
    elif (directory / "observations.csv").exists():
        (directory / "observations.csv").unlink()
    generated_names = [
        "report.json",
        "report.md",
        "report-full.md",
        "report.html",
        "plan.resolved.yaml",
        "design.resolved.csv",
        "hierarchy-risks.csv",
        "comparisons.csv",
        "comparison-sets.csv",
    ]
    if include_observations:
        generated_names.append("observations.csv")
    if report.outcome_mapping_sensitivities:
        write_csv_text(directory / MAPPING_SENSITIVITY_CSV, rendered[MAPPING_SENSITIVITY_CSV])
        generated_names.append(MAPPING_SENSITIVITY_CSV)
    elif (directory / MAPPING_SENSITIVITY_CSV).exists():
        (directory / MAPPING_SENSITIVITY_CSV).unlink()
    if report.presentation.claims_export:
        write_text(directory / CLAIMS_JSON, rendered[CLAIMS_JSON])
        generated_names.append(CLAIMS_JSON)
    elif (directory / CLAIMS_JSON).exists():
        (directory / CLAIMS_JSON).unlink()
    # ORDER MATTERS: the index lists every file written above, so it is written last, just
    # before the checksums that cover it.
    for name, text in index_files(report, generated_names).items():
        write_text(directory / name, text)
        generated_names.append(name)
    write_checksums(directory, generated_names)
    return directory
