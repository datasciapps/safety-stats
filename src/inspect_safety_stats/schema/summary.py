"""The summary level of ``report.json``: what to read first, with pointers to the rest.

``summary`` holds the integrity status and counts, the headline estimates (each model's
weighted-mean failure rate and each overall comparison), the errors and warnings grouped by
code with a pointer to each occurrence and to the item it concerns, and the ids of the
assumptions the results rest on. With ``report.include_decisions`` it also holds the overall
decision and each claim's status; without it they are left out, as in every rendering, while
``/decision`` and ``/claims`` still hold them (#58).

A headline's ``lower`` and ``upper`` bound the rate the scorer reports. Where a judge correction
applies, the headline also holds ``judge_corrected``: the interval that bounds the true failure
risk, picked by :func:`judge_interval`, the rule ``report.md`` and ``analyze``'s output use too.

Nothing in it is computed anew: each value is copied from the report, so the summary is a
function of the rest of ``report.json``. The report writes it and drops it when it is read
back, which keeps a saved report and the model it reads back to the same.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer

from inspect_safety_stats.registry import CODES, Effect, SourceKind
from inspect_safety_stats.schema.reports import Decision, Eligibility, ResultModel

if TYPE_CHECKING:
    from inspect_safety_stats.schema.reports import (
        Diagnostic,
        GroupRiskResult,
        RiskIntervalResult,
        SafetyReport,
    )

HEADLINE_METHOD = "finite_design_weighted_mean"
JUDGE_PREFIX = "judge_corrected_"
VALIDATED_CORRECTION = "validated_judge_misclassification_correction"
SHOWN_SEVERITIES = ("ERROR", "WARNING")


def is_headline_risk(risk: GroupRiskResult) -> bool:
    """Whether a group risk is a headline: a weighted-mean failure rate."""

    return risk.method == HEADLINE_METHOD


def judge_interval(
    intervals: Iterable[RiskIntervalResult], primary_scenario: str | None
) -> RiskIntervalResult | None:
    """The interval of a row that bounds its true failure risk; None when no correction applies.

    It is the one rule for every rendering that shows the true risk beside the judge-reported
    one: ``report.md``, the summary's headline (``index.json`` and ``index.md``) and
    ``analyze``'s output. Under an assumed calibration it is the primary scenario's correction
    (``judge_corrected_<primary>``), and otherwise the validated correction
    (``judge_corrected_validated``). ``primary_scenario`` is the plan's
    ``inputs.judge_calibration.primary``, which only an assumed calibration has. A transported
    or panel-derived correction is never picked: each is a sensitivity, kept in ``report.json``.

    The interval picked may be a sensitivity: only a ``certifying`` one, which only a
    validated calibration gives, bounds the true risk with the coverage it states.
    """

    wanted = f"{JUDGE_PREFIX}{primary_scenario or 'validated'}"
    return next((item for item in intervals if item.interval_id == wanted), None)


def judge_calibration_of(interval: RiskIntervalResult) -> Literal["validated", "assumed"]:
    """How a judge-corrected interval took the judge's accuracy: measured, or declared."""

    return "validated" if interval.method == VALIDATED_CORRECTION else "assumed"


class SummaryJudgeCorrection(ResultModel):
    """The interval that bounds a headline's true failure risk (``judge_interval``)."""

    # How the judge's accuracy entered: measured from labels, or declared by the plan.
    calibration: Literal["validated", "assumed"]
    lower: float | None = None
    upper: float | None = None
    # Only a certifying interval supports a claim about true risk.
    eligibility: Eligibility
    # The coverage the interval holds jointly with its family; None for a sensitivity.
    simultaneous_coverage: float | None = None
    # The interval's record: /group_risks/<i>/intervals/<j> or /comparisons/<i>/intervals/<j>.
    item: str


class SummaryEstimate(ResultModel):
    """A headline estimate and where its full record is."""

    kind: Literal["group_risk", "comparison"]
    # The cohort of a group risk; the candidate of a comparison.
    cohort: str
    baseline: str | None = None
    group: str
    measure: str
    # A comparison's estimand, such as risk_difference.
    estimand: str | None = None
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    eligibility: Eligibility
    item: str
    # Where a judge correction applies: the interval of the true failure risk, which lower
    # and upper, the judge-reported rate's bounds, are not. Left out when none applies.
    judge_corrected: SummaryJudgeCorrection | None = None

    @model_serializer(mode="wrap")
    def leave_out_an_absent_correction(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        # A report without a judge correction keeps the bytes it had before the field.
        data: dict[str, object] = handler(self)
        if self.judge_corrected is None:
            data.pop("judge_corrected", None)
        return data


class SummaryOccurrence(ResultModel):
    """One diagnostic of a code: its own record and the record of the item it concerns."""

    diagnostic: str
    scope: str | None = None
    item: str | None = None


class SummaryWarning(ResultModel):
    """The errors or warnings of one code, with what the registry says the code means."""

    code: str
    severity: Literal["ERROR", "WARNING"]
    count: int
    source: SourceKind | None = None
    effect: Effect | None = None
    assumption: str | None = None
    consequence: str | None = None
    occurrences: list[SummaryOccurrence] = Field(default_factory=list)


class SummaryClaim(ResultModel):
    claim_id: str
    title: str
    status: Decision
    mandatory: bool
    item: str


class ReportSummary(ResultModel):
    integrity: Literal["valid", "degraded", "invalid"]
    error_count: int
    warning_count: int
    info_count: int
    decisions_shown: bool
    # Present only when decisions_shown.
    decision: Decision | None = None
    claims: list[SummaryClaim] | None = None
    headline: list[SummaryEstimate] = Field(default_factory=list)
    warnings: list[SummaryWarning] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def leave_out_hidden_decisions(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if not self.decisions_shown:
            data.pop("decision", None)
            data.pop("claims", None)
        return data


def _judge_corrected(
    item: str, intervals: list[RiskIntervalResult], primary_scenario: str | None
) -> SummaryJudgeCorrection | None:
    """The true-risk interval of the row at ``item``, by ``judge_interval``."""

    interval = judge_interval(intervals, primary_scenario)
    if interval is None:
        return None
    position = next(index for index, other in enumerate(intervals) if other is interval)
    return SummaryJudgeCorrection(
        calibration=judge_calibration_of(interval),
        lower=interval.lower,
        upper=interval.upper,
        eligibility=interval.eligibility,
        simultaneous_coverage=interval.simultaneous_coverage,
        item=f"{item}/intervals/{position}",
    )


def _warnings(diagnostics: list[Diagnostic]) -> list[SummaryWarning]:
    grouped: dict[tuple[str, str], list[tuple[int, Diagnostic]]] = {}
    for index, item in enumerate(diagnostics):
        if item.severity in SHOWN_SEVERITIES:
            grouped.setdefault((item.severity, item.code), []).append((index, item))
    warnings = []
    # Errors first, then warnings; within each, the order the analysis raised them.
    for severity in SHOWN_SEVERITIES:
        for (shown, code), items in grouped.items():
            if shown != severity:
                continue
            entry = CODES.get(code)
            warnings.append(
                SummaryWarning(
                    code=code,
                    severity=severity,  # type: ignore[arg-type]
                    count=len(items),
                    source=items[0][1].source,
                    effect=items[0][1].effect,
                    assumption=entry.assumption if entry is not None else None,
                    consequence=entry.consequence if entry is not None else None,
                    occurrences=[
                        SummaryOccurrence(
                            diagnostic=f"/integrity/diagnostics/{index}",
                            scope=item.scope,
                            item=item.item,
                        )
                        for index, item in items
                    ],
                )
            )
    return warnings


def report_summary(report: SafetyReport) -> ReportSummary:
    """The summary level of ``report``, copied from its other fields."""

    diagnostics = report.integrity.diagnostics
    shown = report.presentation.include_decisions
    calibration = report.presentation.judge_calibration
    primary_scenario = calibration.primary if calibration is not None else None
    headline = [
        SummaryEstimate(
            kind="group_risk",
            cohort=risk.cohort_id,
            group=risk.group_id,
            measure=risk.measure_id,
            estimate=risk.estimate,
            lower=risk.lower,
            upper=risk.upper,
            eligibility=risk.eligibility,
            item=f"/group_risks/{index}",
            judge_corrected=_judge_corrected(
                f"/group_risks/{index}", risk.intervals, primary_scenario
            ),
        )
        for index, risk in enumerate(report.group_risks)
        if is_headline_risk(risk)
    ]
    headline.extend(
        SummaryEstimate(
            kind="comparison",
            cohort=comparison.candidate_cohort,
            baseline=comparison.baseline_cohort,
            group=comparison.group_id,
            measure=comparison.measure_id,
            estimand=comparison.estimand,
            estimate=comparison.estimate,
            lower=comparison.lower,
            upper=comparison.upper,
            eligibility=comparison.eligibility,
            item=f"/comparisons/{index}",
            judge_corrected=_judge_corrected(
                f"/comparisons/{index}", comparison.intervals, primary_scenario
            ),
        )
        for index, comparison in enumerate(report.comparisons)
        if comparison.level is None
    )
    return ReportSummary(
        integrity=report.integrity.status,
        error_count=sum(item.severity == "ERROR" for item in diagnostics),
        warning_count=sum(item.severity == "WARNING" for item in diagnostics),
        info_count=sum(item.severity == "INFO" for item in diagnostics),
        decisions_shown=shown,
        decision=report.decision if shown else None,
        claims=(
            [
                SummaryClaim(
                    claim_id=claim.claim_id,
                    title=claim.title,
                    status=claim.status,
                    mandatory=claim.mandatory,
                    item=f"/claims/{index}",
                )
                for index, claim in enumerate(report.claims)
            ]
            + [
                SummaryClaim(
                    claim_id=claim.claim_id,
                    title=claim.title,
                    status=claim.status,
                    mandatory=claim.mandatory,
                    item=f"/adaptive_claims/{index}",
                )
                for index, claim in enumerate(report.adaptive_claims)
            ]
            if shown
            else None
        ),
        headline=headline,
        warnings=_warnings(diagnostics),
        assumptions=[item.id for item in report.assumptions_used],
    )
