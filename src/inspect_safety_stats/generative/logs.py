"""Read Inspect logs for a generative run, through the Inspect reader's records.

The fixed-design ingestion joins samples against a prompt manifest and resolves groups
and independence profiles from an ``AnalysisPlan``. A generative run has none of those:
the sample identifier written at materialization is the whole join key, and the cohort is
determined by which model produced the log. This module does that much and no more.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inspect_safety_stats.config.models import OutcomeMappingSpec
from inspect_safety_stats.generative.join import EvaluatedSample
from inspect_safety_stats.generative.schema import GenerativePlan
from inspect_safety_stats.ingest.inspect_logs import (
    read_inspect_rollouts,
    read_inspect_runs,
)
from inspect_safety_stats.ingest.outcomes import is_missing, map_outcome
from inspect_safety_stats.ingest.records import RunRecord
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.schema.scopes import scope_path

LogInput = str | Path | list[str] | list[Path]


def _mapping(plan: GenerativePlan) -> OutcomeMappingSpec:
    return OutcomeMappingSpec(
        failure_values=list(plan.target.score.mapping.failure_values),
        safe_values=list(plan.target.score.mapping.safe_values),
    )


def _native(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _cohorts_by_eval(
    plan: GenerativePlan, runs: list[RunRecord], diagnostics: list[Diagnostic]
) -> dict[str, str]:
    by_model: dict[str, list[str]] = {}
    for cohort_id, cohort in plan.target.cohorts.items():
        by_model.setdefault(cohort.model, []).append(cohort_id)

    resolved: dict[str, str] = {}
    for run in runs:
        eval_id = run.run_key
        model = run.model if run.model is not None else ""
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
                    scope=scope_path("evals", eval_id),
                )
            )
            continue
        if len(matches) > 1:
            diagnostics.append(
                Diagnostic(
                    code="GENERATIVE_COHORT_AMBIGUOUS",
                    severity="ERROR",
                    message=f"model {model!r} is declared by cohorts {sorted(matches)}",
                    scope=scope_path("evals", eval_id),
                )
            )
            continue
        # A run without a status counts as complete here, as it always has.
        if run.status is not None and str(run.status) != "success":
            diagnostics.append(
                Diagnostic(
                    code="GENERATIVE_LOG_INCOMPLETE",
                    severity="ERROR",
                    message=f"evaluation {eval_id!r} has status {run.status!r}",
                    scope=scope_path("evals", eval_id),
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
    # ORDER MATTERS: the samples are read only once some evaluation matched a cohort, so
    # logs of other models alone give a warning and no samples, never an error.
    runs = read_inspect_runs(logs)
    cohorts = _cohorts_by_eval(plan, runs, diagnostics)
    if not cohorts:
        return [], diagnostics

    collected = read_inspect_rollouts(logs, runs, parallel=parallel)
    score_name = plan.target.score.name
    if score_name not in collected.score_names:
        column = f"score_{score_name}"
        diagnostics.append(
            Diagnostic(
                code="GENERATIVE_SCORE_ABSENT",
                severity="ERROR",
                message=f"score column {column!r} is absent from the selected logs",
            )
        )

    mapping = _mapping(plan)
    unmapped = 0
    samples: list[EvaluatedSample] = []
    for rollout in collected.rollouts:
        cohort_id = cohorts.get(rollout.run_key)
        if cohort_id is None:
            continue
        raw = _native(rollout.scores.get(score_name))
        failure: bool | None = None
        if not is_missing(raw):
            try:
                failure = map_outcome(raw, mapping)
            except ValueError:
                unmapped += 1
        samples.append(
            EvaluatedSample(
                sample_id=rollout.item_id,
                cohort_id=cohort_id,
                epoch=int(rollout.repeat or 1),
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
