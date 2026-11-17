from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from inspect_ai.analysis import evals_df, samples_df
from inspect_ai.log import EvalLog, read_eval_log

from inspect_safety_stats.config.canonicalize import sha256_value
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.manifest import (
    ResolvedGroup,
    load_inclusion_probabilities,
    load_manifest,
    resolve_groups,
)
from inspect_safety_stats.ingest.outcomes import is_missing, map_outcome
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.stats.independence import BlockIndex
from inspect_safety_stats.stats.judges import compose_verdicts

LogInput = str | Path | EvalLog | Sequence[str | Path] | Sequence[EvalLog] | None


@dataclass
class NormalizedData:
    observations: list[Observation]
    manifest: pd.DataFrame
    groups: list[ResolvedGroup]
    diagnostics: list[Diagnostic]
    cohorts: list[dict[str, Any]]
    source_hashes: dict[str, str | None]
    judge_verdicts: dict[str, dict[str, bool | None]] = dataclass_field(default_factory=dict)
    # Built by the analysis on first use, from these observations and this manifest.
    block_index: BlockIndex | None = dataclass_field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True)
class ErrorContext:
    stage: Literal["target_generation", "post_generation", "unknown"]
    content_delivered: bool | None
    provider_error_code: str | None


_POLICY_BLOCK_CODES = {
    "bio_policy",
    "content_filter",
    "content_policy_violation",
    "guardrail_blocked",
    "policy_violation",
    "safety_filter",
    "safety_violation",
}


def _provided_eval_logs(logs: LogInput) -> dict[str, EvalLog]:
    if isinstance(logs, EvalLog):
        candidates = [logs]
    elif isinstance(logs, Sequence) and not isinstance(logs, (str, bytes, Path)):
        candidates = [item for item in logs if isinstance(item, EvalLog)]
    else:
        candidates = []
    return {str(item.eval.eval_id): item for item in candidates}


def _error_code(*values: Any) -> str | None:
    text = "\n".join(str(value) for value in values if value is not None).lower()
    for pattern in (
        r"['\"]code['\"]\s*:\s*['\"]([^'\"]+)",
        r"\bcode[=:]\s*([a-z0-9_-]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return next((code for code in sorted(_POLICY_BLOCK_CODES) if code in text), None)


def _error_context(
    *,
    log_path: str,
    eval_id: str,
    sample_id: str,
    provided: dict[str, EvalLog],
    cache: dict[str, dict[str, ErrorContext]],
) -> ErrorContext:
    cache_key = eval_id if eval_id in provided else log_path
    if cache_key not in cache:
        contexts: dict[str, ErrorContext] = {}
        try:
            log = provided.get(eval_id) or read_eval_log(log_path)
            for sample in log.samples or []:
                if sample.error is None:
                    continue
                completion = sample.output.completion if sample.output is not None else ""
                content_delivered = bool(completion and completion.strip())
                model_events = [
                    event
                    for event in (sample.events or [])
                    if getattr(event, "event", None) == "model"
                ]
                first_error = next(
                    (
                        index
                        for index, event in enumerate(model_events)
                        if getattr(event, "error", None) is not None
                    ),
                    None,
                )
                prior_success = bool(
                    first_error is not None
                    and any(
                        getattr(event, "error", None) is None
                        for event in model_events[:first_error]
                    )
                )
                stage: Literal["target_generation", "post_generation", "unknown"]
                if content_delivered or prior_success:
                    stage = "post_generation"
                elif first_error is not None:
                    stage = "target_generation"
                else:
                    stage = "unknown"
                event_errors = [getattr(event, "error", None) for event in model_events]
                contexts[str(sample.uuid)] = ErrorContext(
                    stage=stage,
                    content_delivered=content_delivered,
                    provider_error_code=_error_code(sample.error, *event_errors),
                )
        except Exception:
            contexts = {}
        cache[cache_key] = contexts
    return cache[cache_key].get(
        sample_id,
        ErrorContext(stage="unknown", content_delivered=None, provider_error_code=None),
    )


def _equal(left: Any, right: Any) -> bool:
    if is_missing(left) or is_missing(right):
        return is_missing(left) and is_missing(right)
    if isinstance(left, (dict, list)) or isinstance(right, (dict, list)):
        return sha256_value(left) == sha256_value(right)
    return bool(left == right)


def _native(value: Any) -> Any:
    if is_missing(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    return value


def _blocked_prompt(cohort_id: str, prompt_id: str) -> dict[str, Any]:
    """Name the cohort and prompt whose bounds a rollout-level ERROR must withhold.

    Rollout-level diagnostics are scoped by eval id, which is not a cohort, so the analysis
    cannot recover the blocked prompt from the scope string alone.
    """
    return {"cohort_id": cohort_id, "prompt_id": prompt_id}


def _hash_file(path: str) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _match_cohorts(row: pd.Series, plan: AnalysisPlan) -> list[str]:
    matched: list[str] = []
    for cohort_id, cohort in plan.cohorts.items():
        if all(
            field in row and _equal(row[field], expected)
            for field, expected in cohort.match.items()
        ):
            matched.append(cohort_id)
    return matched


def _configuration(row: pd.Series, required: list[str]) -> dict[str, Any]:
    standard = [
        "task_name",
        "task_version",
        "model",
        "model_generate_config",
        "solver",
        "solver_args",
        "sandbox_type",
        "sandbox_config",
        "dataset_name",
        "epochs",
    ]
    return {
        field: _native(row[field]) for field in dict.fromkeys(standard + required) if field in row
    }


def _parse_time(value: Any) -> datetime | None:
    if is_missing(value):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    result = parsed.to_pydatetime()
    return result if isinstance(result, datetime) else None


def ingest_inspect_logs(
    plan: AnalysisPlan,
    logs: LogInput,
    *,
    parallel: bool | int = False,
) -> NormalizedData:
    diagnostics: list[Diagnostic] = []
    manifest = load_manifest(plan.inputs.design_manifest)
    groups = resolve_groups(plan, manifest)
    evaluations = evals_df(logs)
    samples = samples_df(logs, parallel=parallel)
    if evaluations.empty:
        raise ValueError("no Inspect evaluation logs were found")
    if samples.empty:
        raise ValueError("selected Inspect logs contain no sample records")

    eval_cohorts: dict[str, str] = {}
    eval_rows: dict[str, pd.Series] = {}
    cohort_summaries: list[dict[str, Any]] = []
    for _, row in evaluations.iterrows():
        eval_id = str(row["eval_id"])
        matches = _match_cohorts(row, plan)
        if len(matches) > 1:
            diagnostics.append(
                Diagnostic(
                    code="COHORT_OVERLAP",
                    severity="ERROR",
                    message=f"evaluation {eval_id!r} matches multiple cohorts: {matches}",
                    scope=eval_id,
                )
            )
        elif len(matches) == 1:
            eval_cohorts[eval_id] = matches[0]
            eval_rows[eval_id] = row
            if str(row.get("status", "success")) != "success":
                diagnostics.append(
                    Diagnostic(
                        code="LOG_INCOMPLETE",
                        severity="ERROR",
                        message=f"evaluation {eval_id!r} has status {row.get('status')!r}",
                        scope=eval_id,
                    )
                )

    if not eval_cohorts:
        diagnostics.append(
            Diagnostic(
                code="COHORT_EMPTY",
                severity="ERROR",
                message="no selected evaluation matched any declared cohort",
            )
        )

    for cohort_id, cohort in plan.cohorts.items():
        rows = [eval_rows[eid] for eid, cid in eval_cohorts.items() if cid == cohort_id]
        if not rows:
            diagnostics.append(
                Diagnostic(
                    code="COHORT_EMPTY",
                    severity="ERROR",
                    message=f"cohort {cohort_id!r} contains no evaluations",
                    scope=cohort_id,
                )
            )
            continue
        missing_fields = [
            field for field in cohort.require_uniform if not all(field in row for row in rows)
        ]
        for field in missing_fields:
            diagnostics.append(
                Diagnostic(
                    code="PROVENANCE_UNOBSERVED",
                    severity="WARNING",
                    message=f"cohort {cohort_id!r} cannot verify uniform field {field!r}",
                    scope=cohort_id,
                )
            )
        configs = [_configuration(row, cohort.require_uniform) for row in rows]
        fingerprints = {sha256_value(config) for config in configs}
        if len(fingerprints) > 1:
            diagnostics.append(
                Diagnostic(
                    code="CONFIG_DRIFT",
                    severity="ERROR",
                    message=f"cohort {cohort_id!r} contains incompatible evaluation configurations",
                    scope=cohort_id,
                )
            )
        cohort_summaries.append(
            {
                "cohort_id": cohort_id,
                "eval_ids": sorted(str(row["eval_id"]) for row in rows),
                "evaluation_count": len(rows),
                "configuration_fingerprints": sorted(fingerprints),
                "models": sorted({str(row.get("model")) for row in rows}),
            }
        )

    if plan.analysis.intent == "confirmatory" and plan.analysis.plan_frozen_at is not None:
        creation_times = [_parse_time(row.get("created")) for row in eval_rows.values()]
        observed = [time for time in creation_times if time is not None]
        frozen = plan.analysis.plan_frozen_at
        if frozen.tzinfo is None:
            frozen = frozen.replace(tzinfo=timezone.utc)
        if observed and frozen > min(observed):
            diagnostics.append(
                Diagnostic(
                    code="PLAN_POSTHOC",
                    severity="ERROR",
                    message=(
                        "analysis plan freeze time is later than the earliest selected evaluation"
                    ),
                )
            )
        elif plan.analysis.registration is None:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_SELF_DECLARED",
                    severity="WARNING",
                    message=(
                        "confirmatory timing is self-declared; no external plan "
                        "registration was supplied"
                    ),
                )
            )

    selected = samples.loc[samples["eval_id"].astype(str).isin(eval_cohorts)].copy()
    judge_series: dict[str, dict[Any, bool | None]] = {}
    for judge in plan.inputs.judges:
        column = f"score_{judge.score.name}"
        if column not in selected:
            diagnostics.append(
                Diagnostic(
                    code="JUDGE_SCORE_MISSING",
                    severity="ERROR",
                    message=(
                        f"judge {judge.id!r} score column {column!r} is absent from selected logs"
                    ),
                    scope=f"inputs.judges.{judge.id}",
                )
            )
            absent: dict[Any, bool | None] = {label: None for label in selected.index}
            judge_series[judge.id] = absent
            continue
        verdicts: list[bool | None] = []
        for label, raw in selected[column].items():
            value = _native(raw)
            if is_missing(value):
                verdicts.append(None)
                continue
            try:
                verdicts.append(map_outcome(value, judge.score.mapping))
            except ValueError as exc:
                diagnostics.append(
                    Diagnostic(
                        code="JUDGE_SCORE_UNMAPPED",
                        severity="ERROR",
                        message=f"judge {judge.id!r}: {exc}",
                        scope=f"inputs.judges.{judge.id}:{label}",
                    )
                )
                verdicts.append(None)
        judge_series[judge.id] = dict(zip(selected.index, verdicts, strict=True))
    # Composites become ordinary score columns, so a composite may serve as the
    # primary outcome and every downstream construction runs unchanged.
    for composite in plan.inputs.composites:
        selected[f"score_{composite.id}"] = pd.Series(
            [
                compose_verdicts(
                    {judge_id: judge_series[judge_id][label] for judge_id in composite.judges},
                    composite.rule,
                )
                for label in selected.index
            ],
            index=selected.index,
            dtype=object,
        )
    score_column = f"score_{plan.inputs.score.name}"
    if score_column not in selected:
        diagnostics.append(
            Diagnostic(
                code="SCORE_MISSING",
                severity="ERROR",
                message=f"score column {score_column!r} is absent from selected logs",
            )
        )
        selected[score_column] = None
    selected["_prompt_id"] = selected["id"].astype(str)
    manifest_by_prompt = manifest.set_index("prompt_id", drop=False)
    unexpected = sorted(set(selected["_prompt_id"]) - set(manifest["prompt_id"]))
    for prompt_id in unexpected[:100]:
        diagnostics.append(
            Diagnostic(
                code="PROMPT_UNEXPECTED",
                severity="ERROR",
                message=f"observed prompt {prompt_id!r} is absent from the design manifest",
                scope=prompt_id,
                details={"prompt_id": prompt_id},
            )
        )

    selected["_cohort_id"] = selected["eval_id"].astype(str).map(eval_cohorts)
    selected = selected.sort_values(["eval_id", "id", "epoch"], kind="stable")
    rollout_keys: set[tuple[Any, ...]] = set()
    observations: list[Observation] = []
    judge_verdicts: dict[str, dict[str, bool | None]] = {}
    source_hashes: dict[str, str | None] = {}
    provided_logs = _provided_eval_logs(logs)
    error_context_cache: dict[str, dict[str, ErrorContext]] = {}
    for collection_order, (label, row) in enumerate(selected.iterrows(), start=1):
        eval_id = str(row["eval_id"])
        prompt_id = str(row["_prompt_id"])
        if prompt_id not in manifest_by_prompt.index:
            continue
        eval_row = eval_rows[eval_id]
        cohort_id = str(row["_cohort_id"])
        key_values: list[Any] = []
        for field in plan.design.rollout_key:
            if field == "id":
                key_values.append(str(row["id"]))
            elif field in row:
                key_values.append(_native(row[field]))
            elif field in eval_row:
                key_values.append(_native(eval_row[field]))
            else:
                key_values.append(None)
        rollout_key = tuple(key_values)
        if rollout_key in rollout_keys:
            diagnostics.append(
                Diagnostic(
                    code="ROLLOUT_DUPLICATE",
                    severity="ERROR",
                    message=f"duplicate rollout key {rollout_key!r}",
                    scope=prompt_id,
                    details={"prompt_id": prompt_id},
                )
            )
        rollout_keys.add(rollout_key)
        raw_error = _native(row.get("error"))
        if isinstance(raw_error, str) and not raw_error.strip():
            raw_error = None
        raw_score = _native(row.get(score_column))
        failure: bool | None = None
        outcome_status: Literal[
            "classified_failure",
            "classified_safe",
            "target_policy_blocked_safe",
            "unclassified",
        ] = "unclassified"
        error_stage: Literal["target_generation", "post_generation", "unknown"] | None = None
        content_delivered: bool | None = None
        provider_error_code: str | None = None
        if raw_error is not None:
            log_path = str(row.get("log", eval_row.get("log", "")))
            context = _error_context(
                log_path=log_path,
                eval_id=eval_id,
                sample_id=str(row["sample_id"]),
                provided=provided_logs,
                cache=error_context_cache,
            )
            error_stage = context.stage
            content_delivered = context.content_delivered
            provider_error_code = context.provider_error_code
            if plan.inputs.score.errors == "count_as_failure":
                failure = True
                outcome_status = "classified_failure"
            elif plan.inputs.score.errors == "block":
                diagnostics.append(
                    Diagnostic(
                        code="SAMPLE_ERROR",
                        severity="ERROR",
                        message="terminal sample error blocks certification",
                        scope=f"{eval_id}:{prompt_id}:{row.get('epoch')}",
                        details=_blocked_prompt(cohort_id, prompt_id),
                    )
                )
            elif (
                plan.inputs.score.errors == "end_to_end"
                and plan.inputs.estimand == "end_to_end_system"
                and context.stage == "target_generation"
                and context.content_delivered is False
                and context.provider_error_code in _POLICY_BLOCK_CODES
            ):
                failure = False
                outcome_status = "target_policy_blocked_safe"
                diagnostics.append(
                    Diagnostic(
                        code="TARGET_POLICY_BLOCKED_SAFE",
                        severity="INFO",
                        message=(
                            "explicit target-side policy block delivered no content and "
                            "counts as safe for the end-to-end system estimand"
                        ),
                        scope=f"{eval_id}:{prompt_id}:{row.get('epoch')}",
                        details={
                            "error_stage": context.stage,
                            "provider_error_code": context.provider_error_code,
                        },
                    )
                )
            elif plan.inputs.score.errors in {"end_to_end", "conservative_bounds"}:
                diagnostics.append(
                    Diagnostic(
                        code="SAMPLE_ERROR_UNCLASSIFIED",
                        severity="WARNING",
                        message=(
                            "sample error is unclassified and is covered by conservative "
                            "missing-outcome bounds"
                        ),
                        scope=f"{eval_id}:{prompt_id}:{row.get('epoch')}",
                        details={
                            "integrity_degraded": True,
                            "error_stage": context.stage,
                            "content_delivered": context.content_delivered,
                            "provider_error_code": context.provider_error_code,
                        },
                    )
                )
        elif is_missing(raw_score):
            if plan.inputs.score.missing == "count_as_failure":
                failure = True
                outcome_status = "classified_failure"
            elif plan.inputs.score.missing == "block":
                diagnostics.append(
                    Diagnostic(
                        code="SCORE_MISSING",
                        severity="ERROR",
                        message="selected score is missing",
                        scope=f"{eval_id}:{prompt_id}:{row.get('epoch')}",
                        details=_blocked_prompt(cohort_id, prompt_id),
                    )
                )
            elif plan.inputs.score.missing == "conservative_bounds":
                diagnostics.append(
                    Diagnostic(
                        code="SCORE_MISSING_BOUNDED",
                        severity="WARNING",
                        message=(
                            "selected score is missing and is covered by conservative "
                            "missing-outcome bounds"
                        ),
                        scope=f"{eval_id}:{prompt_id}:{row.get('epoch')}",
                        details={"integrity_degraded": True},
                    )
                )
        else:
            try:
                failure = map_outcome(raw_score, plan.inputs.score.mapping)
                outcome_status = "classified_failure" if failure else "classified_safe"
            except ValueError as exc:
                diagnostics.append(
                    Diagnostic(
                        code="SCORE_UNMAPPED",
                        severity="ERROR",
                        message=str(exc),
                        scope=f"{eval_id}:{prompt_id}:{row.get('epoch')}",
                        details=_blocked_prompt(cohort_id, prompt_id),
                    )
                )
        log_path = str(row.get("log", eval_row.get("log", "")))
        if log_path not in source_hashes:
            source_hashes[log_path] = _hash_file(log_path)
        config = _configuration(eval_row, plan.cohorts[cohort_id].require_uniform)
        design_row = manifest_by_prompt.loc[prompt_id]
        identity = {
            "eval_id": eval_id,
            "sample_id": str(row["sample_id"]),
            "inspect_id": str(row["id"]),
            "epoch": int(row["epoch"]),
            "score": plan.inputs.score.name,
        }
        observation_id = sha256_value(identity)
        if judge_series:
            judge_verdicts[observation_id] = {
                judge_id: series[label] for judge_id, series in judge_series.items()
            }
        observations.append(
            Observation(
                observation_id=observation_id,
                source_log=log_path,
                source_log_sha256=source_hashes[log_path],
                eval_id=eval_id,
                run_id=_native(eval_row.get("run_id")),
                sample_id=str(row["sample_id"]),
                inspect_id=str(row["id"]),
                epoch=int(row["epoch"]),
                prompt_id=prompt_id,
                question_id=str(design_row["question_id"]),
                task_id=str(design_row["task_id"]),
                cohort_id=cohort_id,
                score_name=plan.inputs.score.name,
                score_raw=raw_score,
                failure=failure,
                outcome_status=outcome_status,
                error=raw_error,
                error_stage=error_stage,
                content_delivered=content_delivered,
                provider_error_code=provider_error_code,
                retries=int(_native(row.get("retries")) or 0),
                model=str(eval_row.get("model", "unknown")),
                configuration_fingerprint=sha256_value(config),
                collection_order=collection_order,
            )
        )

    observed_counts: dict[tuple[str, str], int] = {}
    for observation in observations:
        key = (observation.cohort_id, observation.prompt_id)
        observed_counts[key] = observed_counts.get(key, 0) + 1
    sampled_design = plan.design.type == "sampled"
    for cohort_id in plan.cohorts:
        for _, item in manifest.iterrows():
            prompt_id = str(item["prompt_id"])
            count = observed_counts.get((cohort_id, prompt_id), 0)
            # Under a sampled design the manifest is the frame, not the realized sample,
            # so a frame item with no observations was simply not drawn.
            if count == 0 and sampled_design:
                continue
            if plan.design.require_complete_manifest and count == 0:
                diagnostics.append(
                    Diagnostic(
                        code="PROMPT_MISSING",
                        severity="ERROR",
                        message=f"prompt {prompt_id!r} has no observations in cohort {cohort_id!r}",
                        scope=f"{cohort_id}:{prompt_id}",
                        details={"cohort_id": cohort_id, "prompt_id": prompt_id},
                    )
                )
            if plan.design.require_expected_rollouts and "expected_rollouts" in manifest:
                expected = int(item["expected_rollouts"])
                if count < expected:
                    diagnostics.append(
                        Diagnostic(
                            code="ROLLOUT_SHORTFALL",
                            severity="ERROR",
                            message=(
                                f"prompt {prompt_id!r} in cohort {cohort_id!r} has "
                                f"{count} rollouts; "
                                f"expected {expected}"
                            ),
                            scope=f"{cohort_id}:{prompt_id}",
                            details={"cohort_id": cohort_id, "prompt_id": prompt_id},
                        )
                    )
                elif count > expected:
                    diagnostics.append(
                        Diagnostic(
                            code="ROLLOUT_EXCESS",
                            severity="ERROR",
                            message=(
                                f"prompt {prompt_id!r} in cohort {cohort_id!r} has "
                                f"{count} rollouts; "
                                f"expected {expected}"
                            ),
                            scope=f"{cohort_id}:{prompt_id}",
                            details={"cohort_id": cohort_id, "prompt_id": prompt_id},
                        )
                    )
            elif plan.design.require_expected_rollouts:
                diagnostics.append(
                    Diagnostic(
                        code="PLAN_EXPECTED_ROLLOUTS_MISSING",
                        severity="ERROR",
                        message=(
                            "design.require_expected_rollouts is on, but the design manifest "
                            "has no expected_rollouts column. Add it, with the number of "
                            "rollouts the evaluation ran for each prompt, or set "
                            "design.require_expected_rollouts: false"
                        ),
                    )
                )
                break

    if sampled_design:
        column = str(plan.design.inclusion_probability_column)
        probabilities = load_inclusion_probabilities(manifest, column)
        prompt_ids = manifest["prompt_id"].astype(str).tolist()
        expected_size = float(np.sum(probabilities))
        deviation = math.sqrt(float(np.sum(probabilities * (1 - probabilities))))
        strata = (
            manifest[str(plan.design.stratum_column)].astype(str).tolist()
            if plan.design.stratum_column
            else ["_all"] * len(prompt_ids)
        )
        if plan.design.sampling == "fixed_size":
            # A fixed-size design must realize exactly its planned draws. Inferring the
            # plan from pi requires pi to be constant inside a stratum and to imply an
            # integer draw count.
            planned: dict[str, int] = {}
            for stratum in sorted(set(strata)):
                members = [i for i, value in enumerate(strata) if value == stratum]
                stratum_probabilities = probabilities[members]
                if float(np.ptp(stratum_probabilities)) > 1e-12:
                    diagnostics.append(
                        Diagnostic(
                            code="SAMPLE_STRATUM_INCONSISTENT",
                            severity="ERROR",
                            message=(
                                f"stratum {stratum!r} has non-constant inclusion "
                                "probabilities; a fixed-size design draws a fixed count "
                                "from each stratum, so pi must equal draws/size within it"
                            ),
                            scope=stratum,
                        )
                    )
                    continue
                exact = float(stratum_probabilities[0]) * len(members)
                if abs(exact - round(exact)) > 1e-6 * len(members):
                    diagnostics.append(
                        Diagnostic(
                            code="SAMPLE_DRAWS_NOT_INTEGER",
                            severity="ERROR",
                            message=(
                                f"stratum {stratum!r} implies {exact:.6g} draws from "
                                f"{len(members)} frame items; a fixed-size design needs an "
                                "integer draw count"
                            ),
                            scope=stratum,
                        )
                    )
                    continue
                planned[stratum] = int(round(exact))
            for cohort_id in plan.cohorts:
                for stratum, count in planned.items():
                    realized_stratum = sum(
                        1
                        for index, prompt_id in enumerate(prompt_ids)
                        if strata[index] == stratum
                        and observed_counts.get((cohort_id, prompt_id), 0) > 0
                    )
                    if realized_stratum != count:
                        diagnostics.append(
                            Diagnostic(
                                code="SAMPLE_DRAWS_MISMATCH",
                                severity="ERROR",
                                message=(
                                    f"cohort {cohort_id!r} stratum {stratum!r} realized "
                                    f"{realized_stratum} draws against a planned {count}; a "
                                    "fixed-size design must realize its plan exactly or the "
                                    "declared probabilities misstate the sample"
                                ),
                                scope=f"{cohort_id}:{stratum}",
                                details={
                                    "planned_draws": count,
                                    "realized_draws": realized_stratum,
                                },
                            )
                        )
            return NormalizedData(
                observations=observations,
                manifest=manifest,
                groups=groups,
                diagnostics=diagnostics,
                cohorts=cohort_summaries,
                source_hashes=source_hashes,
                judge_verdicts=judge_verdicts,
            )
        for cohort_id in plan.cohorts:
            realized = sum(
                1
                for prompt_id in prompt_ids
                if observed_counts.get((cohort_id, prompt_id), 0) > 0
            )
            details = {
                "frame_size": len(prompt_ids),
                "realized_sample_size": realized,
                "expected_sample_size": expected_size,
                "standard_deviation": deviation,
            }
            if realized == 0:
                diagnostics.append(
                    Diagnostic(
                        code="SAMPLE_EMPTY",
                        severity="ERROR",
                        message=f"cohort {cohort_id!r} has no sampled frame items",
                        scope=cohort_id,
                        details=details,
                    )
                )
            elif deviation > 0 and abs(realized - expected_size) > 4 * deviation:
                diagnostics.append(
                    Diagnostic(
                        code="SAMPLE_SIZE_IMPLAUSIBLE",
                        severity="ERROR",
                        message=(
                            f"cohort {cohort_id!r} realized {realized} sampled frame items "
                            f"against an expected {expected_size:.1f} (sd {deviation:.1f}); the "
                            "declared inclusion probabilities do not describe this run, "
                            "which would bias the estimate"
                        ),
                        scope=cohort_id,
                        details=details,
                    )
                )

    return NormalizedData(
        observations=observations,
        manifest=manifest,
        groups=groups,
        diagnostics=diagnostics,
        cohorts=cohort_summaries,
        source_hashes=source_hashes,
        judge_verdicts=judge_verdicts,
    )
