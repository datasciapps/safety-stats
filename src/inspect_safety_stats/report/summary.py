"""The one-page summary written as ``report.md``.

The full report (every interval construction, sensitivity, prompt-level bound and
multiplicity family) is ``report-full.md``, with ``report.html`` and the CSVs. This page
answers the questions a reader brings: what was decided and why, how often does each model
fail, does one fail more than another, where, which of those numbers are too imprecise to act
on, what the analysis warned about, and which assumptions the results rest on.

No bound reads tighter than the analysis gives: each upper bound is rounded up and each lower
bound down at the decimals shown (``report.rounding``). Rates and differences show a tenth of a
percentage point. The claims show ``report.decimal_places``, as ``report-full.md`` does, and
each threshold exactly, so that a bound shows at or below its threshold exactly when it is.

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
from collections.abc import Callable, Iterable
from statistics import median
from typing import Any

from inspect_safety_stats.config.models import DEFAULT_PRECISION_TARGET
from inspect_safety_stats.registry import CODES
from inspect_safety_stats.report.index import DIAGNOSTICS_URL
from inspect_safety_stats.report.rounding import bound_text, decimals, exact, threshold_text
from inspect_safety_stats.report.writer import _claim_status, _escape
from inspect_safety_stats.schema.reports import (
    ComparisonResult,
    Diagnostic,
    GroupRiskResult,
    HierarchyRiskResult,
    RiskIntervalResult,
    SafetyReport,
)
from inspect_safety_stats.schema.summary import JUDGE_PREFIX, is_headline_risk, judge_interval
from inspect_safety_stats.stats.planning import (
    rollout_scale_for_half_width,
    rollouts_for_half_width,
)

# The section that gives each error and warning code's assumption and effect; analyze points
# to it by name.
WARNINGS_SECTION = "Warnings"
_MAX_BREAKDOWN_ROWS = 40
# Decimals shown: of a percentage point for rates and differences, of a ratio, and of a tail
# or threshold measure. Every bound is rounded outward at these (``report.rounding``).
_RATE_PLACES = 1
_RATIO_PLACES = 2
_MEASURE_PLACES = 3
# report-full.md's precision, which the claims share, for a report that does not record it.
_DEFAULT_DECIMAL_PLACES = 6
_MAX_WARNING_SCOPES = 3
_MAX_POLICY_PROMPTS = 5
# The effects of the info codes the summary names (registry.Effect).
_NOTED_EFFECTS = ("adjusted", "withheld")
_MEASURE_PREFIX = "finite_design_"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def rate_text(value: float | None) -> str:
    """A failure rate as a percentage, to the nearest tenth of a point."""

    if value is None:
        return "—"
    return f"{100 * value:.1f}%"


def _signed_points(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:+.1f} pts"


def rate_interval_text(lower: float | None, upper: float | None) -> str:
    """An interval of rates in percent, rounded outward to a tenth of a point."""

    if lower is None or upper is None:
        return "unavailable"
    left = bound_text(lower, _RATE_PLACES, upper=False, percent=True)
    right = bound_text(upper, _RATE_PLACES, upper=True, percent=True)
    return f"[{left}%, {right}%]"


def _signed_interval(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "unavailable"
    left = bound_text(lower, _RATE_PLACES, upper=False, percent=True, signed=True)
    right = bound_text(upper, _RATE_PLACES, upper=True, percent=True, signed=True)
    return f"[{left} pts, {right} pts]"


def _ratio_interval(lower: float | None, upper: float | None) -> str:
    left = bound_text(lower, _RATIO_PLACES, upper=False)
    right = (
        "∞"
        if upper is None or not math.isfinite(upper)
        else bound_text(upper, _RATIO_PLACES, upper=True)
    )
    return f"[{left}, {right}]"


def _measure_interval(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "unavailable"
    left = bound_text(lower, _MEASURE_PLACES, upper=False)
    right = bound_text(upper, _MEASURE_PLACES, upper=True)
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


def _prompt_profiles(report: SafetyReport) -> set[str]:
    """The independence profiles whose blocks are the prompts: ``block_by: [prompt_id]``."""

    profiles = report.design.get("independence_profiles") or {}
    return {
        str(profile_id)
        for profile_id, profile in profiles.items()
        if isinstance(profile, dict) and list(profile.get("block_by") or []) == ["prompt_id"]
    }


def _sensitivity(
    intervals: Iterable[RiskIntervalResult], method: str | None, profiles: set[str]
) -> RiskIntervalResult | None:
    """The prompt-level sensitivity: from a profile in ``profiles``, by the primary's method.

    Other sensitivities, such as a coarser profile's or a judge scenario's, are left to
    report-full.md, whatever their place among the intervals.
    """

    return next(
        (
            item
            for item in intervals
            if item.role == "sensitivity"
            and item.varied_assumption != "judge_error"
            and item.independence_profile in profiles
            and item.method == method
            and not item.interval_id.startswith(JUDGE_PREFIX)
        ),
        None,
    )


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
    """Rollouts per prompt, used to turn a flag into a budget.

    ``per_prompt`` is the median of the prompts' valid rollouts, which for an even number of
    prompts can lie halfway between two counts; it is kept as it is, not truncated.
    """

    def __init__(self, report: SafetyReport) -> None:
        counts = [item.n_valid for item in report.prompt_estimates if item.n_valid > 0]
        self.per_prompt: float = median(counts) if counts else 0

    @property
    def per_prompt_text(self) -> str:
        """The median rollouts per prompt: a whole number, or one ending in .5."""

        whole = int(self.per_prompt)
        return str(whole) if whole == self.per_prompt else f"{self.per_prompt:.1f}"

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
        judge_interval(risk.intervals, primary_scenario) is not None for risk in risks
    )
    header = "| Model | Group | Failure rate | Interval | Prompt-level sensitivity |"
    rule = "|---|---|---:|---|---|"
    if with_judge:
        header += " True risk (judge-corrected) |"
        rule += "---|"
    lines = ["## Failure risk", "", header, rule]
    profiles = _prompt_profiles(report)
    for risk in risks:
        sensitivity = _sensitivity(risk.intervals, risk.interval_method, profiles)
        flagged = _imprecise(risk.estimate, risk.lower, risk.upper, target)
        cells = [
            _escape(risk.cohort_id),
            _escape(risk.group_id),
            rate_text(risk.estimate),
            rate_interval_text(risk.lower, risk.upper) + (" †" if flagged else ""),
            (
                rate_interval_text(sensitivity.lower, sensitivity.upper)
                if sensitivity is not None
                else "—"
            ),
        ]
        if with_judge:
            judged = judge_interval(risk.intervals, primary_scenario)
            cells.append(
                "—"
                if judged is None
                else rate_interval_text(judged.lower, judged.upper)
                + ("" if judged.eligibility == "certifying" else " (assumed)")
            )
        lines.append("| " + " | ".join(cells) + " |")
        if flagged:
            flags.append(
                f"{_escape(risk.cohort_id)} · {_escape(risk.group_id)}: "
                f"{rate_text(risk.estimate)} {rate_interval_text(risk.lower, risk.upper)}: "
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
    # Rounded down, like a lower bound, so that the page never claims more coverage.
    coverage_text = (
        f" with {bound_text(coverage, _RATE_PLACES, upper=False, percent=True)}% simultaneous "
        "coverage"
        if coverage
        else ""
    )
    if report.scope.get("design") == "sampled":
        # A sampled design's intervals are about the frame its items were drawn from, and it
        # has no sensitivity profiles: each drawn item is one observation, its failure rate.
        note = (
            f"Intervals bound the risk over the declared item frame{coverage_text}, allowing "
            "for both the draw of items from the frame and the rollouts of the drawn items. "
            "Each drawn item enters as one observation, so the rollouts of one item may "
            "depend on each other arbitrarily."
        )
    else:
        note = (
            f"Intervals bound the rollout uncertainty on these fixed prompts{coverage_text}. "
            + _prompt_level_note(report, risks, profiles)
        )
    if with_judge:
        note += (
            " The judge-corrected column bounds the true failure risk; it is marked "
            "assumed when it rests on declared rather than measured judge accuracy."
        )
    lines.extend(["", note])
    return lines


def _prompt_level_note(
    report: SafetyReport, risks: list[GroupRiskResult], profiles: set[str]
) -> str:
    """What allows dependence among one prompt's rollouts: the primary, a sensitivity or none."""

    primaries = {
        primary.independence_profile
        for primary in (_primary(risk.intervals) for risk in risks)
        if primary is not None
    }
    if primaries and primaries <= profiles:
        return (
            "Their primary independence profile blocks the rollouts by prompt, so they already "
            "allow arbitrary dependence among the rollouts of one prompt."
        )
    if any(
        _sensitivity(risk.intervals, risk.interval_method, profiles) is not None
        for risk in risks
    ):
        return (
            "The prompt-level sensitivity, from the profile with `block_by: [prompt_id]`, "
            "allows arbitrary dependence among the rollouts of one prompt."
        )
    return (
        "No profile with `block_by: [prompt_id]` is a sensitivity of these intervals, so "
        "none allows dependence among the rollouts of one prompt."
    )


def _comparison_rows(
    report: SafetyReport,
) -> list[tuple[tuple[str, str, str, str], dict[str, ComparisonResult]]]:
    """The root comparisons, a row per cohorts, group and measure, with its estimands.

    A difference and a ratio of the same cohorts, group and measure share a row. A second
    comparison of an estimand already in the row, such as the same difference under another
    independence profile, starts a row of its own rather than replace the first.
    """

    rows: list[tuple[tuple[str, str, str, str], dict[str, ComparisonResult]]] = []
    for item in report.comparisons:
        if item.level is not None:
            continue
        key = (item.candidate_cohort, item.baseline_cohort, item.group_id, item.measure_id)
        row = next(
            (
                estimands
                for row_key, estimands in rows
                if row_key == key and item.estimand not in estimands
            ),
            None,
        )
        if row is None:
            row = {}
            rows.append((key, row))
        row[item.estimand] = item
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
    labels = [(candidate, baseline, group) for (candidate, baseline, group, _), _ in rows]
    for (candidate, baseline, group, _measure), estimands in rows:
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
                named = (
                    f" (`{_escape(difference.comparison_id)}`)"
                    if labels.count((candidate, baseline, group)) > 1
                    else ""
                )
                flags.append(
                    f"{_escape(candidate)} − {_escape(baseline)} · {_escape(group)}{named}: "
                    f"{_signed_points(difference.estimate)} "
                    f"{_signed_interval(difference.lower, difference.upper)}: "
                    + budget.advice(
                        arm_rate,
                        _half_width(difference.estimate, difference.lower, difference.upper),
                        _primary(difference.intervals),
                        target,
                    )
                )
        label = f"{_escape(candidate)} vs {_escape(baseline)}"
        if labels.count((candidate, baseline, group)) > 1:
            # Rows that would read alike name their comparisons.
            label += " (" + ", ".join(
                f"`{_escape(item.comparison_id)}`" for item in estimands.values()
            ) + ")"
        cells = [
            label,
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
    for path in paths[:_MAX_BREAKDOWN_ROWS]:
        first = next(index[(cohort, path)] for cohort in cohorts if (cohort, path) in index)
        cells = [_escape(path), str(first.prompt_count)]
        for cohort in cohorts:
            node = index.get((cohort, path))
            if node is None:
                cells.append("—")
                continue
            flagged = _imprecise(node.estimate, node.lower, node.upper, target)
            cells.append(
                f"{rate_text(node.estimate)} {rate_interval_text(node.lower, node.upper)}"
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
    flags.extend(_breakdown_flag(selected, level, target, budget))
    return lines


def _breakdown_flag(
    nodes: list[HierarchyRiskResult], level: str, target: float, budget: _Budget
) -> list[str]:
    """The flag for a breakdown: how many of all its nodes are too wide, and the widest.

    It counts every node of the level, also those beyond the rows shown. A node without an
    interval is counted as unavailable, not as the widest.
    """

    widths = [
        (node, _half_width(node.estimate, node.lower, node.upper)) for node in nodes
    ]
    wide = [(node, width) for node, width in widths if width is not None and width > target + 1e-12]
    unavailable = sum(width is None for _, width in widths)
    if not wide and not unavailable:
        return []
    text = f"{len(wide)} of {len(nodes)} {_escape(level)} intervals exceed ±{rate_text(target)}"
    if unavailable:
        text += f", and {unavailable} {'is' if unavailable == 1 else 'are'} unavailable"
    if wide:
        widest, width = max(wide, key=lambda item: item[1])
        text += (
            f"; the widest ({_escape(widest.cohort_id)} · "
            f"{_escape(' → '.join(widest.path.values()))}: "
            f"{rate_text(widest.estimate)} {rate_interval_text(widest.lower, widest.upper)}) "
            + budget.advice(widest.estimate, width, _primary(widest.intervals), target)
        )
    return [text]


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
            f"{risk.estimate:.{_MEASURE_PLACES}f}" if risk.estimate is not None else "—",
            _measure_interval(risk.lower, risk.upper) + (" †" if imprecise else ""),
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
        # The count at rate 0 is the fewest any prompt needs: a prompt that fails needs more.
        # It is computed here, from the plan's precision target, and is not in report.json.
        flags.append(
            f"{flagged} tail or threshold intervals exceed ±{exact(target)}"
            + (
                f"; per-prompt bounds that tight need about {needed} rollouts on every "
                "prompt, the count for a prompt with no failures and the fewest any prompt "
                f"needs (currently {budget.per_prompt_text} per prompt, median)"
                if needed
                else ""
            )
        )
    return lines


def _claim_places(report: SafetyReport) -> int:
    """The decimals of the claims: report-full.md's, or more where a threshold needs them.

    Each threshold is shown exactly and each bound rounded outward at the same decimals, so a
    bound shows at or below its threshold exactly when it is at or below it.
    """

    places = report.presentation.decimal_places
    if places is None:
        places = _DEFAULT_DECIMAL_PLACES
    thresholds = [
        value
        for claim in report.claims
        for value in (claim.acceptable_upper, claim.unacceptable_lower)
    ] + [claim.unacceptable_lower for claim in report.adaptive_claims]
    return max([places, *(decimals(value) for value in thresholds)])


def _zero_tolerance_events(report: SafetyReport) -> list[dict[str, Any]]:
    return [
        event for event in report.policy_events if event.get("type") == "zero_tolerance_failure"
    ]


def _titles(titles: list[str]) -> str:
    shown = ", ".join(f"*{_escape(title)}*" for title in titles[:_MAX_WARNING_SCOPES])
    if len(titles) > _MAX_WARNING_SCOPES:
        shown += f" and {len(titles) - _MAX_WARNING_SCOPES} more"
    return shown


def decision_reason(report: SafetyReport) -> str:
    """Why the overall decision is what it is, as a clause after "because".

    The reason is found by the rules ``decision.engine.roll_up`` applies, in its order: a
    zero-tolerance failure fails the analysis; invalid integrity or an exploratory intent
    leaves it inconclusive, as does having no mandatory claim; then a mandatory claim that
    fails, or one that is inconclusive, decides it. A decision policy that leaves an overall
    PASS uncertified makes it INCONCLUSIVE (``analysis.policy``). Should the rules found here
    give another decision than the report's, the clause says so and points to
    ``report-full.md`` rather than give a reason that does not hold.
    """

    statuses = [(claim.title, claim.status) for claim in report.claims if claim.mandatory]
    statuses += [(claim.title, claim.status) for claim in report.adaptive_claims if claim.mandatory]
    events = _zero_tolerance_events(report)
    if events:
        derived = "FAIL"
        reason = (
            f"{_count(len(events), 'rollout')} failed on a zero-tolerance prompt, which fails "
            "the analysis whatever the claims' statuses"
        )
    elif report.integrity.status == "invalid":
        derived, reason = "INCONCLUSIVE", "data integrity is invalid, so no claim is certified"
    elif report.analysis.get("intent") != "confirmatory":
        derived = "INCONCLUSIVE"
        reason = "the analysis is exploratory, so no claim is certified"
    elif not statuses:
        derived, reason = "INCONCLUSIVE", "no claim is mandatory"
    elif any(status == "FAIL" for _, status in statuses):
        derived = "FAIL"
        failed = [title for title, status in statuses if status == "FAIL"]
        reason = f"mandatory {_plural(len(failed), 'claim')} {_titles(failed)} failed"
    elif any(status == "INCONCLUSIVE" for _, status in statuses):
        derived = "INCONCLUSIVE"
        open_ = [title for title, status in statuses if status == "INCONCLUSIVE"]
        reason = (
            f"mandatory {_plural(len(open_), 'claim')} {_titles(open_)} "
            + ("is" if len(open_) == 1 else "are")
            + " inconclusive"
        )
    else:
        derived, reason = "PASS", "every mandatory claim passed"
    if derived == report.decision:
        return reason
    uncertified = any(
        item.code == "DECISION_POLICY_OVERALL_UNCERTIFIED"
        for item in report.integrity.diagnostics
    )
    if derived == "PASS" and report.decision == "INCONCLUSIVE" and uncertified:
        return (
            "every mandatory claim passed, but a decision policy moved a mandatory claim's "
            "threshold and their bounds rest on more than one coverage event, so the overall "
            "PASS is not certified (`DECISION_POLICY_OVERALL_UNCERTIFIED`)"
        )
    return "of a rule the claims below do not show; see `report-full.md`"


def _plural(number: int, noun: str) -> str:
    return noun if number == 1 else f"{noun}s"


def _zero_tolerance_line(report: SafetyReport) -> str:
    """The prompts a zero-tolerance failure was found on, with the cohorts that failed them."""

    cohorts: dict[str, list[str]] = {}
    for event in _zero_tolerance_events(report):
        prompt = str(event.get("prompt_id"))
        cohort = str(event.get("cohort_id"))
        if cohort not in cohorts.setdefault(prompt, []):
            cohorts[prompt].append(cohort)
    if not cohorts:
        return ""
    prompts = list(cohorts)
    shown = ", ".join(
        f"`{_escape(prompt)}` ({', '.join(_escape(cohort) for cohort in cohorts[prompt])})"
        for prompt in prompts[:_MAX_POLICY_PROMPTS]
    )
    if len(prompts) > _MAX_POLICY_PROMPTS:
        shown += f" and {len(prompts) - _MAX_POLICY_PROMPTS} more"
    return (
        f"**Zero-tolerance failures** on {_count(len(prompts), 'prompt')}, with the models that "
        f"failed them: {shown}. Each failing rollout is under Policy events in "
        "`report-full.md`."
    )


def _claims_section(report: SafetyReport) -> list[str]:
    if not report.presentation.include_decisions:
        return []
    places = _claim_places(report)
    lines = [
        "## Claims",
        "",
        f"Overall decision: **{report.decision}**, because {decision_reason(report)}.",
        "",
        (
            "| Claim | Mandatory | Status | Lower | Upper | Acceptable below | "
            "Unacceptable above |"
        ),
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for claim in report.claims:
        cells = [
            _escape(claim.title),
            "yes" if claim.mandatory else "no",
            _claim_status(claim),
            bound_text(claim.lower, places, upper=False),
            bound_text(claim.upper, places, upper=True),
            threshold_text(claim.acceptable_upper, places),
            threshold_text(claim.unacceptable_lower, places),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    for adaptive in report.adaptive_claims:
        cells = [
            f"{_escape(adaptive.title)} (adaptive attacker, FAIL only)",
            "yes" if adaptive.mandatory else "no",
            _claim_status(adaptive),
            bound_text(adaptive.lower, places, upper=False),
            "—",
            "—",
            threshold_text(adaptive.unacceptable_lower, places),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if not report.claims and not report.adaptive_claims:
        lines.append("| _No claims declared_ | — | — | — | — | — | — |")
    notes = []
    if any(claim.descriptive_status is not None for claim in report.claims) or any(
        claim.descriptive_status is not None for claim in report.adaptive_claims
    ):
        notes.append(
            "A descriptive status is what a confirmatory analysis of the same bounds would give "
            "at the plan's thresholds. It is shown for reading the bounds and certifies nothing."
        )
    if report.adaptive_claims:
        notes.append(
            "An adaptive-attacker claim (experimental) bounds from below how often this "
            "attacker succeeds at this budget, with no independence between its runs. It is no "
            "evidence about other attackers, so it can fail but never pass, and a mandatory one "
            "keeps the decision from PASS. Its runs, successes and reasons are under "
            "Adaptive-attacker claims in `report-full.md`."
        )
    events = _zero_tolerance_line(report)
    if events:
        notes.append(events)
    policy = decision_policy_note(report)
    if policy:
        notes.append(policy)
    for note in notes:
        lines.extend(["", note])
    return lines


def decision_policy_note(
    report: SafetyReport,
    code: Callable[[str], str] = lambda text: f"`{_escape(text)}`",
    *,
    pointer: bool = True,
) -> str:
    """One paragraph on the decision policy the thresholds come from, or "" without one.

    It says which thresholds the policy set, and which versions state a time after the first
    evaluation started. The summary lists errors and warnings only, so without it the latter
    would be left to an ``INFO`` diagnostic in ``report-full.md``. ``code`` formats an
    identifier or a file name; ``pointer`` adds where the versions are listed.
    """

    record = report.decision_policy
    if record is None:
        return ""
    if not record.applied:
        return (
            "A decision policy (experimental) was given but not applied, because it failed a "
            "check; the claims keep the plan's thresholds."
        )
    claims = [item.claim_id for item in record.claims if item.set_by != "plan"]
    later = [item.id for item in record.versions if item.set_after_first_evaluation]
    text = (
        "Thresholds of "
        + (", ".join(code(item) for item in claims) or "no claim")
        + " come from a decision policy (experimental), not the plan; the others are the "
        "plan's."
    )
    if later:
        text += (
            " Version(s) "
            + ", ".join(code(item) for item in later)
            + " state a time after the first evaluation started, so those thresholds may "
            "describe the bounds rather than a requirement fixed in advance."
        )
    if pointer:
        text += " The versions and their times are under Decision policy in "
        text += code("report-full.md") + "."
    return text


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
        f"## {WARNINGS_SECTION}",
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
            f"`report-full.md`; [the diagnostics reference]({DIAGNOSTICS_URL}) gives the "
            "remedy for each.",
        ]
    )
    return lines


def _notes_section(diagnostics: list[Diagnostic]) -> list[str]:
    """One line naming the info codes that adjust or withhold a result.

    The warnings block lists errors and warnings only. An info code that changes how a result
    is computed, such as ``TOKEN_LIMIT_REACHED`` counting an answer cut off at the token limit
    as the plan says, or that leaves one out, is named here, so that the reader of this page
    knows a number on it was affected. Info codes that only record a fact are left to
    ``report-full.md``.
    """

    noted: dict[tuple[str, str], int] = {}
    for item in diagnostics:
        if item.severity != "INFO":
            continue
        entry = CODES.get(item.code)
        effect = item.effect or (entry.effect if entry is not None else None)
        if effect in _NOTED_EFFECTS:
            noted[(item.code, str(effect))] = noted.get((item.code, str(effect)), 0) + 1
    if not noted:
        return []
    names = ", ".join(
        f"`{_escape(code)}` ({effect}" + (f", {count}×" if count > 1 else "") + ")"
        for (code, effect), count in noted.items()
    )
    return [
        "## Notes",
        "",
        f"Info diagnostics that change a result: {names}. An adjusted result is computed "
        "differently, for example with a wider bound or another method, and reported; a "
        "withheld one is not reported. Each code's message is under Data integrity in "
        "`report-full.md`.",
    ]


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
    # A sampled design's manifest is its frame: most of its items were never drawn.
    sampled = report.scope.get("design") == "sampled"
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
        f"{_count(prompts, 'frame item' if sampled else 'fixed prompt')} · "
        f"{budget.per_prompt_text} rollout{'' if budget.per_prompt == 1 else 's'} per prompt "
        "(median) · "
        f"{_count(len(cohorts), 'model')} · data integrity "
        f"**{report.integrity.status}**" + (f" ({', '.join(counts)})" if counts else ""),
        "",
    ]
    flags: list[str] = []
    for section in (
        _warnings_section(diagnostics),
        _notes_section(diagnostics),
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
                f"† marks an interval with an endpoint more than ±{rate_text(target)} from its "
                "estimate (`report.precision_target`). The budgets below assume the observed "
                "rate holds, unless they say otherwise; they are planning figures, not "
                "guarantees.",
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
