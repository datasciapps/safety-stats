"""Read Inspect logs for a generative run.

The fixed-design ingestion joins samples against a prompt manifest and resolves groups
and independence profiles from an ``AnalysisPlan``. A generative run has none of those:
the sample identifier written at materialization is the whole join key, and the cohort is
determined by which model produced the log. This module does that much and no more.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inspect_ai.analysis import evals_df, samples_df

from inspect_safety_stats.config.models import OutcomeMappingSpec
from inspect_safety_stats.generative.join import EvaluatedSample
from inspect_safety_stats.generative.schema import GenerativePlan
from inspect_safety_stats.ingest.outcomes import is_missing, map_outcome
from inspect_safety_stats.schema.reports import Diagnostic

LogInput = str | Path | list[str] | list[Path]


def _mapping(plan: GenerativePlan) -> OutcomeMappingSpec:
    return OutcomeMappingSpec(
        failure_values=list(plan.target.score.mapping.failure_values),
        safe_values=list(plan.target.score.mapping.safe_values),
    )


def _native(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _cohorts_by_eval(
    plan: GenerativePlan, logs: LogInput, diagnostics: list[Diagnostic]
) -> dict[str, str]:
    evaluations = evals_df(logs)
    if evaluations.empty:
        raise ValueError("no Inspect evaluation logs were found")
    by_model: dict[str, list[str]] = {}
    for cohort_id, cohort in plan.target.cohorts.items():
        by_model.setdefault(cohort.model, []).append(cohort_id)

    resolved: dict[str, str] = {}
    for _, row in evaluations.iterrows():
        eval_id = str(row["eval_id"])
        model = str(row.get("model", ""))
        matches = by_model.get(model, [])
        if not matches:
            diagnostics.append(
                Diagnostic(
                    code="GENERATIVE_EVAL_UNMATCHED",
                    severity="WARNING",
                    message=(
                        f"evaluation {eval_id!r} ran model {model!r}, which no declared "
                        "cohort names; its samples are ignored"
                    ),
                    scope=eval_id,
                )
            )
            continue
        if len(matches) > 1:
            diagnostics.append(
                Diagnostic(
                    code="GENERATIVE_COHORT_AMBIGUOUS",
                    severity="ERROR",
                    message=f"model {model!r} is declared by cohorts {sorted(matches)}",
                    scope=eval_id,
                )
            )
            continue
        if str(row.get("status", "success")) != "success":
            diagnostics.append(
                Diagnostic(
                    code="GENERATIVE_LOG_INCOMPLETE",
                    severity="ERROR",
                    message=f"evaluation {eval_id!r} has status {row.get('status')!r}",
                    scope=eval_id,
                )
            )
        resolved[eval_id] = matches[0]
    return resolved


def read_evaluated_samples(
    plan: GenerativePlan, logs: LogInput, *, parallel: bool | int = False
) -> tuple[list[EvaluatedSample], list[Diagnostic]]:
    """Normalize Inspect logs into the join's input.

    A sample whose score is missing or unmappable is returned with ``failure=None`` rather
    than dropped, so the conservative completion inside the interval carries it.
    """

    diagnostics: list[Diagnostic] = []
    cohorts = _cohorts_by_eval(plan, logs, diagnostics)
    if not cohorts:
        return [], diagnostics

    frame = samples_df(logs, parallel=parallel)
    if frame.empty:
        raise ValueError("selected Inspect logs contain no sample records")
    column = f"score_{plan.target.score.name}"
    if column not in frame.columns:
        diagnostics.append(
            Diagnostic(
                code="GENERATIVE_SCORE_ABSENT",
                severity="ERROR",
                message=f"score column {column!r} is absent from the selected logs",
            )
        )
        frame[column] = None

    mapping = _mapping(plan)
    unmapped = 0
    samples: list[EvaluatedSample] = []
    for _, row in frame.iterrows():
        eval_id = str(row.get("eval_id", ""))
        cohort_id = cohorts.get(eval_id)
        if cohort_id is None:
            continue
        raw = _native(row.get(column))
        failure: bool | None = None
        if not is_missing(raw):
            try:
                failure = map_outcome(raw, mapping)
            except ValueError:
                unmapped += 1
        samples.append(
            EvaluatedSample(
                sample_id=str(row.get("id", "")),
                cohort_id=cohort_id,
                epoch=int(row.get("epoch", 1) or 1),
                failure=failure,
            )
        )
    if unmapped:
        diagnostics.append(
            Diagnostic(
                code="GENERATIVE_SCORE_UNMAPPED",
                severity="WARNING",
                message=(
                    f"{unmapped} samples carry a score outside the declared mapping and are "
                    "treated as unclassified"
                ),
                details={"count": unmapped},
            )
        )
    return samples, diagnostics
