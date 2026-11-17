"""The summary level of ``report.json``: what to read first, with pointers to the rest.

``summary`` holds the integrity status and counts, the headline estimates (each model's
weighted-mean failure rate and each overall comparison), the errors and warnings grouped by
code with a pointer to each occurrence and to the item it concerns, and the ids of the
assumptions the results rest on. With ``report.include_decisions`` it also holds the overall
decision and each claim's status; without it they are left out, as in every rendering, while
``/decision`` and ``/claims`` still hold them (#58).

Nothing in it is computed anew: each value is copied from the report, so the summary is a
function of the rest of ``report.json``. The report writes it and drops it when it is read
back, which keeps a saved report and the model it reads back to the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer

from inspect_safety_stats.registry import CODES, Effect, SourceKind
from inspect_safety_stats.schema.reports import Decision, Eligibility, ResultModel

if TYPE_CHECKING:
    from inspect_safety_stats.schema.reports import Diagnostic, GroupRiskResult, SafetyReport

HEADLINE_METHOD = "finite_design_weighted_mean"
SHOWN_SEVERITIES = ("ERROR", "WARNING")


def is_headline_risk(risk: GroupRiskResult) -> bool:
    """Whether a group risk is a headline: a weighted-mean failure rate."""

    return risk.method == HEADLINE_METHOD


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
