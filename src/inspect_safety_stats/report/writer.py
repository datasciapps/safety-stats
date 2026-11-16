from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from inspect_safety_stats.config.canonicalize import canonical_value
from inspect_safety_stats.schema.reports import RiskIntervalResult, SafetyReport


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(canonical_value(value), ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else canonical_value(value)
                    for key, value in row.items()
                }
            )


def _escape(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: float | None, places: int) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"


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


def _recorded(values: list[Any] | None) -> str:
    if values is None:
        return "not recorded"
    return "; ".join(
        f"`{_escape(json.dumps(value, sort_keys=True, separators=(',', ':')))}`"
        for value in values
    )


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


def _markdown(report: SafetyReport) -> str:
    places = int(report._resolved_plan.get("report", {}).get("decimal_places", 6))
    include_decisions = bool(report.presentation.get("include_decisions", True))
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
    if include_decisions:
        lines.insert(2, f"**Decision:** `{report.decision}`  ")
        lines.extend(
            [
                "",
                "## Mandatory claims",
                "",
                "| Claim | Status | Estimate | Lower | Upper | Acceptable | Unacceptable |",
                "|---|---|---:|---:|---:|---:|---:|",
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
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("| _No mandatory claims_ | `INCONCLUSIVE` | — | — | — | — | — |")
        if any(claim.descriptive_status is not None for claim in mandatory):
            lines.extend(
                [
                    "",
                    (
                        "This analysis is exploratory, so no claim is certified. A descriptive "
                        "status is what a confirmatory analysis of the same bounds would give at "
                        "the plan's thresholds. It is shown for reading the bounds and certifies "
                        "nothing."
                    ),
                ]
            )
    lines.extend(_judge_error_stress_lines(report, places, include_decisions))
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
                "intervals. Weighted means may use direct weighted Hoeffding or variance-adaptive "
                "empirical-Bernstein intervals. Rollout-level intervals assume independent "
                "attempted rollouts; prompt-cluster sensitivities allow arbitrary dependence "
                "within a prompt. Other "
                "measures use propagated simultaneous exact prompt bounds, which do not require "
                "independence between prompts. Coverage is the conservative floor stated for "
                "each method. "
                "Where outcomes are unclassified, reported missingness-robust bounds cover "
                "both the all-safe and all-failure completions."
            ),
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
        lines.extend(_judge_claim_lines(panel, places, include_decisions))
    judge_rows = [
        (risk, interval)
        for risk in report.group_risks
        for interval in risk.intervals
        if interval.interval_id.startswith("judge_corrected_")
    ]
    if judge_rows:
        calibration = report._resolved_plan.get("inputs", {}).get("judge_calibration") or {}
        primary_scenario = str(calibration.get("primary") or "")
        validated = str(calibration.get("mode")) == "validated"
        transported = str(calibration.get("mode")) == "transported"
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
        scoped = calibration.get("groups")
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
            scenarios = calibration.get("scenarios") or []
            if scenarios:
                lines.extend(["", "Declared scenarios:", ""])
                for item in scenarios:
                    rho_1, rho_0 = item.get("rho_1", {}), item.get("rho_0", {})
                    lines.append(
                        f"- `{_escape(item.get('id'))}`: "
                        f"P(truly unsafe | judge said unsafe) in "
                        f"[{rho_1.get('lower')}, {rho_1.get('upper')}]; "
                        f"P(truly unsafe | judge said safe) in "
                        f"[{rho_0.get('lower')}, {rho_0.get('upper')}]"
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
                        _escape(" → ".join(node.path.values())),
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
        (node.cohort_id, node.group_id, " → ".join(node.path.values()), interval)
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
    lines.extend(
        [
            "",
            "## Prompt-level evidence",
            "",
            "The complete normalized table is in `observations.csv`.",
            "",
            "| Cohort | Prompt | Failures | Valid | Unclassified | Estimate | Lower | Upper |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
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
                            _escape(" → ".join(comparison.path.values())),
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
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "- Prompt, question, task, and group membership are fixed "
                "creator-declared design elements."
            ),
            (
                "- Statistical bounds quantify rollout randomness only under the "
                "declared stability and independence assumptions."
            ),
            (
                "- Scorer errors, under-elicitation, evaluation awareness, and omitted "
                "hazards are not repaired by statistical precision."
            ),
            (
                "- Outputs marked diagnostic cannot independently produce a safety pass."
            ),
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


def write_report(report: SafetyReport, output: str | Path, *, overwrite: bool = False) -> Path:
    from inspect_safety_stats.report.visualization import render_visualization

    directory = Path(output).resolve()
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"report output exists and is not a directory: {directory}")
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"report directory already exists and is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    report_json = directory / "report.json"
    report_json.write_text(
        json.dumps(
            canonical_value(report.model_dump(mode="json")),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (directory / "report.md").write_text(
        _markdown(report), encoding="utf-8", newline="\n"
    )
    (directory / "report.html").write_text(
        render_visualization(report), encoding="utf-8", newline="\n"
    )
    (directory / "plan.resolved.yaml").write_text(
        yaml.safe_dump(canonical_value(report._resolved_plan), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    _write_csv(directory / "design.resolved.csv", report._resolved_design)
    _write_csv(
        directory / "hierarchy-risks.csv",
        [
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
        ],
    )
    _write_csv(
        directory / "comparisons.csv",
        [item.model_dump(mode="json") for item in report.comparisons],
    )
    _write_csv(
        directory / "comparison-sets.csv",
        [item.model_dump(mode="json") for item in report.comparison_sets],
    )
    include_observations = bool(
        report._resolved_plan.get("report", {}).get("include_observations", True)
    )
    if include_observations:
        _write_csv(
            directory / "observations.csv",
            [item.model_dump(mode="json") for item in report._observations],
        )
    elif (directory / "observations.csv").exists():
        (directory / "observations.csv").unlink()
    generated_names = [
        "report.json",
        "report.md",
        "report.html",
        "plan.resolved.yaml",
        "design.resolved.csv",
        "hierarchy-risks.csv",
        "comparisons.csv",
        "comparison-sets.csv",
    ]
    if include_observations:
        generated_names.append("observations.csv")
    checksums = {name: _sha256(directory / name) for name in sorted(generated_names)}
    (directory / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return directory
