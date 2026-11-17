"""The one-page summary written as ``report.md``.

The full report (every interval construction, sensitivity, prompt-level bound and
multiplicity family) is ``report-full.md``, with ``report.html`` and the CSVs. This page
answers the questions a reader brings: how often does each model fail, does one fail more
than another, where, which of those numbers are too imprecise to act on, what the analysis
warned about, and which assumptions the results rest on.

An interval is flagged as imprecise when either endpoint lies further than the plan's
``report.precision_target`` from the estimate. Each flag carries a planning figure for the
rollouts that would resolve it at the observed rate, so an uninformative interval is reported
as a budget question rather than silently.

Like every rendering, the page is a function of the report model alone, so that a report read
back from its ``report.json`` renders to the same text (L38). The registry supplies what each
warning's code means.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import median

from inspect_safety_stats.config.models import DEFAULT_PRECISION_TARGET
from inspect_safety_stats.registry import CODES
from inspect_safety_stats.report.writer import _claim_status, _escape
from inspect_safety_stats.schema.reports import (
    ComparisonResult,
    Diagnostic,
    GroupRiskResult,
    HierarchyRiskResult,
    RiskIntervalResult,
    SafetyReport,
)
from inspect_safety_stats.schema.summary import is_headline_risk
from inspect_safety_stats.stats.planning import (
    rollout_scale_for_half_width,
    rollouts_for_half_width,
)

_MAX_BREAKDOWN_ROWS = 40
_MAX_WARNING_SCOPES = 3
_JUDGE_PREFIX = "judge_corrected_"
_MEASURE_PREFIX = "finite_design_"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%"


def _signed_points(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:+.1f} pts"


def _interval(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "unavailable"
    return f"[{_rate(lower)}, {_rate(upper)}]"


def _signed_interval(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "unavailable"
    return f"[{_signed_points(lower)}, {_signed_points(upper)}]"


def _ratio_interval(lower: float | None, upper: float | None) -> str:
    left = "—" if lower is None else f"{lower:.2f}"
    right = "∞" if upper is None else f"{upper:.2f}"
    return f"[{left}, {right}]"


def _half_width(estimate: float | None, lower: float | None, upper: float | None) -> float | None:
    if estimate is None or lower is None or upper is None:
        return None
    return max(upper - estimate, estimate - lower)


def _imprecise(
    estimate: float | None, lower: float | None, upper: float | None, target: float
) -> bool:
    width = _half_width(estimate, lower, upper)
    return width is None or width > target + 1e-12


def _primary(intervals: Iterable[RiskIntervalResult]) -> RiskIntervalResult | None:
    return next((item for item in intervals if item.role == "primary"), None)


def _sensitivity(
    intervals: Iterable[RiskIntervalResult], method: str | None
) -> RiskIntervalResult | None:
    return next(
        (
            item
            for item in intervals
            if item.role == "sensitivity"
            and item.method == method
            and not item.interval_id.startswith(_JUDGE_PREFIX)
        ),
        None,
    )


def _judge_interval(
    intervals: Iterable[RiskIntervalResult], primary_scenario: str | None
) -> RiskIntervalResult | None:
    corrected = [item for item in intervals if item.interval_id.startswith(_JUDGE_PREFIX)]
    validated = next(
        (item for item in corrected if item.interval_id == f"{_JUDGE_PREFIX}validated"), None
    )
    if validated is not None:
        return validated
    if primary_scenario:
        return next(
            (
                item
                for item in corrected
                if item.interval_id == f"{_JUDGE_PREFIX}{primary_scenario}"
            ),
            None,
        )
    return None


def _measure_type(risk: GroupRiskResult) -> str | None:
    """The plan's type of the measure, which the result's method names."""

    if risk.method.startswith(_MEASURE_PREFIX):
        return risk.method[len(_MEASURE_PREFIX) :]
    return None


def precision_target(report: SafetyReport) -> float:
    """The report's ``report.precision_target``: as the plan set it, else the default."""

    target = report.presentation.precision_target
    return DEFAULT_PRECISION_TARGET if target is None else target


class _Budget:
    """Rollouts per prompt, used to turn a flag into a budget."""

    def __init__(self, report: SafetyReport) -> None:
        counts = [item.n_valid for item in report.prompt_estimates if item.n_valid > 0]
        self.per_prompt = int(median(counts)) if counts else 0

    def advice(
        self,
        rate: float | None,
        half_width: float | None,
        interval: RiskIntervalResult | None,
        target: float,
    ) -> str:
        """Rollouts needed to bring ``half_width`` down to ``target`` at this rate."""

        if rate is None or half_width is None or interval is None:
            return "no interval is available"
        alpha = interval.alpha_upper or interval.alpha_lower
        effective = interval.effective_sample_size
        if alpha is None or not effective or effective <= 0:
            return "add rollouts or prompts"
        scale = rollout_scale_for_half_width(
            rate=rate,
            observed_half_width=half_width,
            observed_trials=effective,
            target_half_width=target,
            alpha=alpha,
        )
        if scale is None:
            return "not reachable by adding rollouts at this rate"
        per_prompt = (
            f", about {math.ceil(self.per_prompt * scale)} per prompt" if self.per_prompt else ""
        )
        return f"needs about {scale:.1f}× the current rollouts{per_prompt}"


def _root_section(
    report: SafetyReport,
    target: float,
    budget: _Budget,
    primary_scenario: str | None,
    flags: list[str],
) -> list[str]:
    risks = [risk for risk in report.group_risks if is_headline_risk(risk)]
    if not risks:
        return []
    with_judge = any(
        _judge_interval(risk.intervals, primary_scenario) is not None for risk in risks
    )
    header = "| Model | Group | Failure rate | Interval | Prompt-level sensitivity |"
    rule = "|---|---|---:|---|---|"
    if with_judge:
        header += " True risk (judge-corrected) |"
        rule += "---|"
    lines = ["## Failure risk", "", header, rule]
    for risk in risks:
        sensitivity = _sensitivity(risk.intervals, risk.interval_method)
        flagged = _imprecise(risk.estimate, risk.lower, risk.upper, target)
        cells = [
            _escape(risk.cohort_id),
            _escape(risk.group_id),
            _rate(risk.estimate),
            _interval(risk.lower, risk.upper) + (" †" if flagged else ""),
            (
                _interval(sensitivity.lower, sensitivity.upper)
                if sensitivity is not None
                else "—"
            ),
        ]
        if with_judge:
            judged = _judge_interval(risk.intervals, primary_scenario)
            cells.append(
                "—"
                if judged is None
                else _interval(judged.lower, judged.upper)
                + ("" if judged.eligibility == "certifying" else " (assumed)")
            )
        lines.append("| " + " | ".join(cells) + " |")
        if flagged:
            flags.append(
                f"{_escape(risk.cohort_id)} · {_escape(risk.group_id)}: {_rate(risk.estimate)} "
                f"{_interval(risk.lower, risk.upper)}: "
                + budget.advice(
                    risk.estimate,
                    _half_width(risk.estimate, risk.lower, risk.upper),
                    _primary(risk.intervals),
                    target,
                )
            )
    coverage = next(
        (risk.simultaneous_coverage for risk in risks if risk.simultaneous_coverage), None
    )
    note = (
        "Intervals bound the rollout uncertainty on these fixed prompts"
        + (f" with {_rate(coverage)} simultaneous coverage" if coverage else "")
        + ". The prompt-level sensitivity allows arbitrary dependence among the rollouts of "
        "one prompt."
    )
    if with_judge:
        note += (
            " The judge-corrected column bounds the true failure risk; it is marked "
            "assumed when it rests on declared rather than measured judge accuracy."
        )
    lines.extend(["", note])
    return lines


def _comparison_rows(
    report: SafetyReport,
) -> dict[tuple[str, str, str, str], dict[str, ComparisonResult]]:
    rows: dict[tuple[str, str, str, str], dict[str, ComparisonResult]] = {}
    for item in report.comparisons:
        if item.level is not None:
            continue
        key = (item.candidate_cohort, item.baseline_cohort, item.group_id, item.measure_id)
        rows.setdefault(key, {})[item.estimand] = item
    return rows


def _comparison_section(
    report: SafetyReport, target: float, budget: _Budget, flags: list[str]
) -> list[str]:
    rows = _comparison_rows(report)
    if not rows:
        return []
    lines = [
        "## Comparisons",
        "",
        "| Candidate vs baseline | Group | Difference | Interval | Ratio | Interval | Reading |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for (candidate, baseline, group, _measure), estimands in rows.items():
        difference = estimands.get("risk_difference")
        ratio = estimands.get("risk_ratio")
        reading = "—"
        if difference is not None and difference.lower is not None and difference.upper is not None:
            if difference.upper < 0:
                reading = f"{_escape(candidate)} fails less"
            elif difference.lower > 0:
                reading = f"{_escape(candidate)} fails more"
            else:
                reading = "no difference resolved"
            if _imprecise(difference.estimate, difference.lower, difference.upper, target):
                reading += " †"
                arm_rate = max(
                    value
                    for value in (difference.candidate_estimate, difference.baseline_estimate, 0.0)
                    if value is not None
                )
                flags.append(
                    f"{_escape(candidate)} − {_escape(baseline)} · {_escape(group)}: "
                    f"{_signed_points(difference.estimate)} "
                    f"{_signed_interval(difference.lower, difference.upper)}: "
                    + budget.advice(
                        arm_rate,
                        _half_width(difference.estimate, difference.lower, difference.upper),
                        _primary(difference.intervals),
                        target,
                    )
                )
        cells = [
            f"{_escape(candidate)} vs {_escape(baseline)}",
            _escape(group),
            _signed_points(difference.estimate) if difference is not None else "—",
            (
                _signed_interval(difference.lower, difference.upper)
                if difference is not None
                else "—"
            ),
            f"{ratio.estimate:.2f}" if ratio is not None and ratio.estimate is not None else "—",
            _ratio_interval(ratio.lower, ratio.upper) if ratio is not None else "—",
            reading,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Differences are in percentage points, candidate minus baseline; a ratio above "
            "1 means the candidate fails more often.",
        ]
    )
    return lines


def _breakdown_section(
    report: SafetyReport, target: float, budget: _Budget, flags: list[str]
) -> list[str]:
    nodes = report.hierarchy_risks
    if not nodes:
        return []
    group = "all" if any(node.group_id == "all" for node in nodes) else nodes[0].group_id
    hierarchy = [
        str(level) for level in report.design.get("hierarchy") or [] if level != "prompt_id"
    ]
    levels_present = [node.level for node in nodes if node.group_id == group]
    level = next((item for item in hierarchy if item in levels_present), None)
    if level is None and levels_present:
        level = levels_present[0]
    if level is None:
        return []
    selected = [node for node in nodes if node.group_id == group and node.level == level]
    cohorts = list(dict.fromkeys(node.cohort_id for node in selected))
    paths = list(dict.fromkeys(" → ".join(node.path.values()) for node in selected))
    index: dict[tuple[str, str], HierarchyRiskResult] = {
        (node.cohort_id, " → ".join(node.path.values())): node for node in selected
    }
    pairs = sorted(
        {
            (item.candidate_cohort, item.baseline_cohort)
            for item in report.comparisons
            if item.level == level
            and item.group_id == group
            and item.estimand == "risk_difference"
        }
    )
    pair = pairs[0] if len(pairs) == 1 else None
    differences: dict[str, ComparisonResult] = {}
    if pair is not None:
        for item in report.comparisons:
            if (
                item.level == level
                and item.group_id == group
                and item.estimand == "risk_difference"
                and (item.candidate_cohort, item.baseline_cohort) == pair
            ):
                differences[" → ".join(item.path.values())] = item
    header = f"| {_escape(level)} | Prompts | " + " | ".join(_escape(c) for c in cohorts) + " |"
    rule = "|---|---:|" + "---|" * len(cohorts)
    if pair is not None:
        header += f" {_escape(pair[0])} − {_escape(pair[1])} |"
        rule += "---|"
    lines = [f"## By {_escape(level)}", "", header, rule]
    flagged_cells = 0
    for path in paths[:_MAX_BREAKDOWN_ROWS]:
        first = next(index[(cohort, path)] for cohort in cohorts if (cohort, path) in index)
        cells = [_escape(path), str(first.prompt_count)]
        for cohort in cohorts:
            node = index.get((cohort, path))
            if node is None:
                cells.append("—")
                continue
            flagged = _imprecise(node.estimate, node.lower, node.upper, target)
            flagged_cells += flagged
            cells.append(
                f"{_rate(node.estimate)} {_interval(node.lower, node.upper)}"
                + (" †" if flagged else "")
            )
        if pair is not None:
            contrast = differences.get(path)
            cells.append(
                "—"
                if contrast is None
                else f"{_signed_points(contrast.estimate)} "
                f"{_signed_interval(contrast.lower, contrast.upper)}"
                + (
                    " †"
                    if _imprecise(contrast.estimate, contrast.lower, contrast.upper, target)
                    else ""
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    if len(paths) > _MAX_BREAKDOWN_ROWS:
        lines.extend(
            ["", f"{len(paths) - _MAX_BREAKDOWN_ROWS} further rows are in `hierarchy-risks.csv`."]
        )
    if flagged_cells:
        widest = max(
            selected,
            key=lambda node: _half_width(node.estimate, node.lower, node.upper) or 1.0,
        )
        flags.append(
            f"{flagged_cells} of {len(selected)} {_escape(level)} intervals exceed "
            f"±{_rate(target)}; the widest ({_escape(widest.cohort_id)} · "
            f"{_escape(' → '.join(widest.path.values()))}: "
            f"{_rate(widest.estimate)} {_interval(widest.lower, widest.upper)}) "
            + budget.advice(
                widest.estimate,
                _half_width(widest.estimate, widest.lower, widest.upper),
                _primary(widest.intervals),
                target,
            )
        )
    return lines


def _other_measures_section(
    report: SafetyReport, target: float, budget: _Budget, flags: list[str]
) -> list[str]:
    risks = [
        risk for risk in report.group_risks if _measure_type(risk) not in {None, "weighted_mean"}
    ]
    if not risks:
        return []
    lines = [
        "## Tail and threshold measures",
        "",
        "| Model | Group | Measure | Value | Interval |",
        "|---|---|---|---:|---|",
    ]
    flagged = 0
    for risk in risks:
        imprecise = _imprecise(risk.estimate, risk.lower, risk.upper, target)
        flagged += imprecise
        cells = [
            _escape(risk.cohort_id),
            _escape(risk.group_id),
            _escape(risk.measure_id),
            f"{risk.estimate:.3f}" if risk.estimate is not None else "—",
            (
                f"[{risk.lower:.3f}, {risk.upper:.3f}]"
                if risk.lower is not None and risk.upper is not None
                else "unavailable"
            )
            + (" †" if imprecise else ""),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "These are built from simultaneous per-prompt bounds, so their precision is set "
            "by the rollouts on each prompt rather than the total.",
        ]
    )
    if flagged:
        prompt_alpha = next(
            (item.alpha_upper for item in report.prompt_estimates if item.alpha_upper), None
        )
        needed = (
            rollouts_for_half_width(rate=0.0, half_width=target, alpha=prompt_alpha)
            if prompt_alpha
            else None
        )
        flags.append(
            f"{flagged} tail or threshold intervals exceed ±{target:.2f}"
            + (
                f"; per-prompt bounds that tight need about {needed} rollouts on every "
                f"prompt (currently {budget.per_prompt})"
                if needed
                else ""
            )
        )
    return lines


def _claims_section(report: SafetyReport) -> list[str]:
    if not report.presentation.include_decisions:
        return []
    lines = [
        "## Claims",
        "",
        f"Overall decision: **{report.decision}**",
        "",
        "| Claim | Status | Lower | Upper | Acceptable below | Unacceptable above |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for claim in report.claims:
        cells = [
            _escape(claim.title),
            _claim_status(claim),
            f"{claim.lower:.4f}" if claim.lower is not None else "—",
            f"{claim.upper:.4f}" if claim.upper is not None else "—",
            f"{claim.acceptable_upper:.4f}",
            f"{claim.unacceptable_lower:.4f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if not report.claims:
        lines.append("| _No claims declared_ | — | — | — | — | — |")
    return lines


def _warnings_section(diagnostics: list[Diagnostic]) -> list[str]:
    """One row per code among the errors and warnings: what it assumes and what it does.

    The assumption and the consequence are the registry's, so the row says what the code means
    in every analysis; the messages, which say what was found in this one, are in
    ``report-full.md``.
    """

    shown = [item for item in diagnostics if item.severity in {"ERROR", "WARNING"}]
    if not shown:
        return []
    by_code: dict[tuple[str, str], list[Diagnostic]] = {}
    for item in sorted(shown, key=lambda item: item.severity != "ERROR"):
        by_code.setdefault((item.severity, item.code), []).append(item)
    lines = [
        "## Warnings",
        "",
        "| Code | Where | Assumption | Effect |",
        "|---|---|---|---|",
    ]
    for (severity, code), items in by_code.items():
        entry = CODES.get(code)
        scopes = list(dict.fromkeys(item.scope for item in items if item.scope))
        where = ", ".join(f"`{_escape(scope)}`" for scope in scopes[:_MAX_WARNING_SCOPES])
        if len(scopes) > _MAX_WARNING_SCOPES:
            where += f" and {len(scopes) - _MAX_WARNING_SCOPES} more"
        effect = items[0].effect or (entry.effect if entry is not None else None)
        consequence = entry.consequence if entry is not None else ""
        cells = [
            f"`{_escape(code)}` ({severity.lower()}"
            + (f", {len(items)}×" if len(items) > 1 else "")
            + ")",
            where or "—",
            _escape(entry.assumption) if entry is not None else "—",
            (f"**{effect}**. " if effect else "") + _escape(consequence),
        ]
        lines.append("| " + " | ".join(cells).rstrip() + " |")
    lines.extend(
        [
            "",
            "Each code's message for this analysis is under Data integrity in "
            "`report-full.md`; `docs/diagnostics.md` gives the remedy for each.",
        ]
    )
    return lines


def _assumptions_section(report: SafetyReport) -> list[str]:
    """The assumptions the results rest on, with the plan fields left to their defaults."""

    if not report.assumptions_used:
        return []
    lines = ["## Assumptions used", ""]
    for assumption in report.assumptions_used:
        defaults = [setting.path for setting in assumption.settings if setting.default]
        lines.append(
            f"- **{_escape(assumption.id)}.** {_escape(assumption.statement)}"
            + (
                " Left to the default: "
                + ", ".join(f"`{_escape(path)}`" for path in defaults)
                + "."
                if defaults
                else ""
            )
        )
    lines.extend(
        [
            "",
            "What follows if one is wrong, and the value of each setting, is under Limitations "
            "in `report-full.md`.",
        ]
    )
    return lines


def summary_markdown(report: SafetyReport) -> str:
    target = precision_target(report)
    calibration = report.presentation.judge_calibration
    primary_scenario = calibration.primary if calibration is not None else None
    budget = _Budget(report)
    prompts = len({item.prompt_id for item in report.prompt_estimates})
    cohorts = sorted({item.cohort_id for item in report.prompt_estimates})
    diagnostics = report.integrity.diagnostics
    errors = sum(item.severity == "ERROR" for item in diagnostics)
    warnings = sum(item.severity == "WARNING" for item in diagnostics)
    counts = [_count(errors, "error")] if errors else []
    if warnings:
        counts.append(_count(warnings, "warning"))

    lines = [
        f"# {_escape(report.analysis.get('title') or report.analysis.get('id'))}",
        "",
        f"{_count(prompts, 'fixed prompt')} · "
        f"{_count(budget.per_prompt, 'rollout')} per prompt (median) · "
        f"{_count(len(cohorts), 'model')} · data integrity "
        f"**{report.integrity.status}**" + (f" ({', '.join(counts)})" if counts else ""),
        "",
    ]
    flags: list[str] = []
    for section in (
        _warnings_section(diagnostics),
        _claims_section(report),
        _root_section(report, target, budget, primary_scenario, flags),
        _comparison_section(report, target, budget, flags),
        _breakdown_section(report, target, budget, flags),
        _other_measures_section(report, target, budget, flags),
    ):
        if section:
            lines.extend([*section, ""])
    if flags:
        lines.extend(
            [
                "## Precision",
                "",
                f"† marks an interval with an endpoint more than ±{_rate(target)} from its "
                "estimate (`report.precision_target`). The budgets below assume the observed "
                "rate holds; they are planning figures, not guarantees.",
                "",
                *[f"- {flag}" for flag in flags],
                "",
            ]
        )
    assumptions = _assumptions_section(report)
    if assumptions:
        lines.extend([*assumptions, ""])
    lines.extend(
        [
            "## Scope",
            "",
            str(report.scope.get("disclaimer", "")),
            "",
        ]
    )
    establish = report.scope.get("what_results_establish")
    if establish:
        # Reports written before the link existed have none.
        lines.extend([f"What these results establish, and what they do not: <{establish}>", ""])
    lines.extend(
        [
            "Every interval, sensitivity, prompt-level bound and diagnostic message is in "
            "`report-full.md`, the interactive `report.html` and the CSV files beside this "
            f"page. Plan SHA-256 `{_escape(report.analysis.get('plan_sha256'))}`; "
            f"inspect-safety-stats {_escape(report.package_version)}.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
