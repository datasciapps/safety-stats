"""Turn the runs and rollouts a source adapter reads into observations and diagnostics.

Nothing here reads a file or knows which tool wrote the evaluation: an adapter, such as the
Inspect reader in :mod:`inspect_safety_stats.ingest.inspect_logs`, supplies a
:class:`SourceData`, and :func:`normalize` applies every rule of the plan to it: cohort
matching, uniformity and drift, plan timing, judges and composites, score mapping, error and
missing-score policies, duplicates, completeness and sampled-design checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from inspect_safety_stats.config.canonicalize import sha256_value
from inspect_safety_stats.config.loader import report_path
from inspect_safety_stats.config.models import ATTRIBUTE_PREFIX, AnalysisPlan
from inspect_safety_stats.config.registration import RegistrationReceipt
from inspect_safety_stats.ingest.manifest import (
    ResolvedGroup,
    load_inclusion_probabilities,
    load_manifest,
    plan_manifest_columns,
    resolve_groups,
)
from inspect_safety_stats.ingest.outcomes import is_missing, map_outcome
from inspect_safety_stats.ingest.records import (
    EvaluationSource,
    NormalizedData,
    RolloutRecord,
    RunRecord,
    SourceData,
)
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.schema.scopes import cohort_scope, plan_scope, prompt_scope, scope_path
from inspect_safety_stats.stats.judges import compose_verdicts

POLICY_BLOCK_CODES = frozenset(
    {
        "bio_policy",
        "content_filter",
        "content_policy_violation",
        "guardrail_blocked",
        "policy_violation",
        "safety_filter",
        "safety_violation",
    }
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

    Rollout-level diagnostics are scoped by run key, which is not a cohort, so the analysis
    cannot recover the blocked prompt from the scope string alone.
    """
    return {"cohort_id": cohort_id, "prompt_id": prompt_id}


_ABSENT = object()


def _run_field(plan: AnalysisPlan, run: RunRecord, name: str) -> Any:
    """The value of the run field a plan names, or ``_ABSENT`` when the run has none.

    Schema 2.0 names the source's fields verbatim. Schema 3.0 names a field every source
    has, or a source field as ``attributes.<name>``.
    """
    if plan.schema_version != "3.0":
        return run.attributes.get(name, _ABSENT)
    if name.startswith(ATTRIBUTE_PREFIX):
        return run.attributes.get(name.removeprefix(ATTRIBUTE_PREFIX), _ABSENT)
    neutral = {
        "run_key": run.run_key,
        "model": run.model,
        "status": run.status,
        "run_id": run.run_id,
    }
    # A neutral field the source did not give is None; it is absent, not a shared value.
    value = neutral.get(name)
    return _ABSENT if value is None else value


def _rollout_field(
    plan: AnalysisPlan, rollout: RolloutRecord, run: RunRecord, name: str
) -> tuple[Any, bool]:
    """The value of a ``rollout_key`` field: the rollout's, else its run's, else None.

    The flag says whether the value is the run's.
    """
    if plan.schema_version == "3.0" and not name.startswith(ATTRIBUTE_PREFIX):
        neutral = {
            "run_key": rollout.run_key,
            "item_id": rollout.item_id,
            "repeat": rollout.repeat,
            "rollout_id": rollout.sample_id,
        }
        return _native(neutral[name]), False
    if plan.schema_version == "3.0":
        name = name.removeprefix(ATTRIBUTE_PREFIX)
    elif name == "id":
        return rollout.item_id, False
    if name in rollout.attributes:
        return _native(rollout.attributes[name]), False
    if name in run.attributes:
        return _native(run.attributes[name]), True
    return None, False


@dataclass(frozen=True)
class _RunValue:
    """A run's value in a rollout key, shown by the name of its field.

    A run's fields hold its configuration: the generate config, whose ``extra_headers`` can
    carry credentials, the base URL, the model arguments. No diagnostic shows their values.
    """

    field: str

    def __repr__(self) -> str:
        return f"<{self.field} of the run>"


def _match_cohorts(run: RunRecord, plan: AnalysisPlan) -> list[str]:
    matched: list[str] = []
    for cohort_id, cohort in plan.cohorts.items():
        if all(
            (value := _run_field(plan, run, field)) is not _ABSENT and _equal(value, expected)
            for field, expected in cohort.match.items()
        ):
            matched.append(cohort_id)
    return matched


def _configuration(
    plan: AnalysisPlan, run: RunRecord, standard: tuple[str, ...], required: list[str]
) -> dict[str, Any]:
    # The adapter's own configuration fields are named as the source names them.
    values = {field: run.attributes[field] for field in standard if field in run.attributes}
    for field in required:
        value = _run_field(plan, run, field)
        if value is not _ABSENT:
            values.setdefault(field, value)
    return {
        field: _native(values[field])
        for field in dict.fromkeys([*standard, *required])
        if field in values
    }


def _utc(value: datetime) -> datetime:
    """A time with no zone read as UTC, as the plan's freeze time is."""

    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def normalize(
    plan: AnalysisPlan,
    manifest: pd.DataFrame,
    groups: list[ResolvedGroup],
    source: SourceData,
    receipt: RegistrationReceipt | None = None,
) -> NormalizedData:
    """Apply the plan's ingestion rules to the runs and rollouts of one source.

    ``receipt``, when given, is a registration receipt whose hash the caller has matched to
    the frozen plan; it is checked against the earliest selected evaluation start.
    """

    stops = plan.inputs.score.content_filter_stops
    stop_context = source.stop_context
    if stops != "score" and stop_context is None:
        raise ValueError(
            f"score.content_filter_stops: {stops} needs the stop reason of each output, which "
            f"this source ({source.source_format.get('format')}) does not report"
        )
    token_limit = plan.inputs.score.token_limit
    if token_limit != "score" and stop_context is None:
        raise ValueError(
            f"score.token_limit: {token_limit} needs the stop reason of each output, which "
            f"this source ({source.source_format.get('format')}) does not report"
        )
    diagnostics: list[Diagnostic] = []
    run_cohorts: dict[str, str] = {}
    run_records: dict[str, RunRecord] = {}
    cohort_summaries: list[dict[str, Any]] = []
    for run in source.runs:
        run_key = run.run_key
        matches = _match_cohorts(run, plan)
        if len(matches) > 1:
            diagnostics.append(
                Diagnostic(
                    code="COHORT_OVERLAP",
                    severity="ERROR",
                    message=f"evaluation {run_key!r} matches multiple cohorts: {matches}",
                    scope=scope_path("runs", run_key),
                )
            )
        elif len(matches) == 1:
            run_cohorts[run_key] = matches[0]
            run_records[run_key] = run
            if run.status is None:
                diagnostics.append(
                    Diagnostic(
                        code="RUN_STATUS_UNOBSERVED",
                        severity="WARNING",
                        message=(
                            f"evaluation {run_key!r} reports no status, so it cannot be "
                            "checked for completion and is assumed complete"
                        ),
                        scope=scope_path("runs", run_key),
                        details={"integrity_degraded": True},
                    )
                )
            elif str(run.status) != "success":
                diagnostics.append(
                    Diagnostic(
                        code="LOG_INCOMPLETE",
                        severity="ERROR",
                        message=f"evaluation {run_key!r} has status {run.status!r}",
                        scope=scope_path("runs", run_key),
                    )
                )

    if not run_cohorts:
        diagnostics.append(
            Diagnostic(
                code="COHORT_EMPTY",
                severity="ERROR",
                message="no selected evaluation matched any declared cohort",
            )
        )

    for cohort_id, cohort in plan.cohorts.items():
        runs = [run_records[key] for key, cid in run_cohorts.items() if cid == cohort_id]
        if not runs:
            diagnostics.append(
                Diagnostic(
                    code="COHORT_EMPTY",
                    severity="ERROR",
                    message=f"cohort {cohort_id!r} contains no evaluations",
                    scope=cohort_scope(cohort_id),
                )
            )
            continue
        missing_fields = [
            field
            for field in cohort.require_uniform
            if not all(_run_field(plan, run, field) is not _ABSENT for run in runs)
        ]
        for field in missing_fields:
            diagnostics.append(
                Diagnostic(
                    code="PROVENANCE_UNOBSERVED",
                    severity="WARNING",
                    message=f"cohort {cohort_id!r} cannot verify uniform field {field!r}",
                    scope=cohort_scope(cohort_id),
                )
            )
        configs = [
            _configuration(plan, run, source.configuration_fields, cohort.require_uniform)
            for run in runs
        ]
        unconfigured = sorted(
            run.run_key
            for run, config in zip(runs, configs, strict=True)
            if all(is_missing(value) for value in config.values())
        )
        if unconfigured:
            diagnostics.append(
                Diagnostic(
                    code="CONFIG_UNOBSERVED",
                    severity="WARNING",
                    message=(
                        f"cohort {cohort_id!r} has {len(unconfigured)} evaluation(s) that "
                        "report no configuration, so configuration drift cannot be checked "
                        "for them"
                    ),
                    scope=cohort_scope(cohort_id),
                    details={"integrity_degraded": True, "evaluations": unconfigured},
                )
            )
        fingerprints = {sha256_value(config) for config in configs}
        if len(fingerprints) > 1:
            diagnostics.append(
                Diagnostic(
                    code="CONFIG_DRIFT",
                    severity="ERROR",
                    message=f"cohort {cohort_id!r} contains incompatible evaluation configurations",
                    scope=cohort_scope(cohort_id),
                )
            )
        cohort_summaries.append(
            {
                "cohort_id": cohort_id,
                "run_keys": sorted(run.run_key for run in runs),
                "run_count": len(runs),
                "configuration_fingerprints": sorted(fingerprints),
                "models": sorted({str(run.model) for run in runs}),
            }
        )

    first_started_at = min(
        (_utc(run.started_at) for run in run_records.values() if run.started_at is not None),
        default=None,
    )
    if plan.analysis.intent == "confirmatory" and plan.analysis.plan_frozen_at is not None:
        observed = [
            run.started_at for run in run_records.values() if run.started_at is not None
        ]
        unstarted = sorted(key for key, run in run_records.items() if run.started_at is None)
        if unstarted:
            diagnostics.append(
                Diagnostic(
                    code="START_TIME_UNOBSERVED",
                    severity="WARNING",
                    message=(
                        f"{len(unstarted)} selected evaluation(s) report no start time, so "
                        "the plan cannot be checked to predate them"
                    ),
                    details={"integrity_degraded": True, "evaluations": unstarted},
                )
            )
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
        elif receipt is not None:
            diagnostics.extend(_receipt_timing(receipt, observed))
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

    # ORDER MATTERS: rollouts stay in the source's order until judges and composites are
    # read, because JUDGE_SCORE_UNMAPPED diagnostics follow that order; they are put in
    # collection order only after that.
    selected = [
        (rank, rollout)
        for rank, rollout in zip(source.collection_rank, source.rollouts, strict=True)
        if rollout.run_key in run_cohorts
    ]
    scores: dict[int, dict[str, Any]] = {
        id(rollout): dict(rollout.scores) for _, rollout in selected
    }
    score_names = set(source.score_names)
    judge_series: dict[str, dict[int, bool | None]] = {}
    for judge in plan.inputs.judges:
        if judge.score.name not in score_names:
            column = f"score_{judge.score.name}"
            diagnostics.append(
                Diagnostic(
                    code="JUDGE_SCORE_MISSING",
                    severity="ERROR",
                    message=(
                        f"judge {judge.id!r} score column {column!r} is absent from selected logs"
                    ),
                    scope=plan_scope("inputs", "judges", judge.id),
                )
            )
            judge_series[judge.id] = {id(rollout): None for _, rollout in selected}
            continue
        verdicts: dict[int, bool | None] = {}
        for _, rollout in selected:
            value = _native(scores[id(rollout)].get(judge.score.name))
            if is_missing(value):
                verdicts[id(rollout)] = None
                continue
            try:
                verdicts[id(rollout)] = map_outcome(value, judge.score.mapping)
            except ValueError as exc:
                diagnostics.append(
                    Diagnostic(
                        code="JUDGE_SCORE_UNMAPPED",
                        severity="ERROR",
                        message=f"judge {judge.id!r}: {exc}",
                        scope=scope_path("runs", rollout.run_key, "records", rollout.record_id),
                    )
                )
                verdicts[id(rollout)] = None
        judge_series[judge.id] = verdicts
    # Composites become ordinary scores, so a composite may serve as the primary outcome
    # and every downstream construction runs unchanged.
    for composite in plan.inputs.composites:
        for _, rollout in selected:
            scores[id(rollout)][composite.id] = compose_verdicts(
                {judge_id: judge_series[judge_id][id(rollout)] for judge_id in composite.judges},
                composite.rule,
            )
        score_names.add(composite.id)
    score_name = plan.inputs.score.name
    if score_name not in score_names:
        diagnostics.append(
            Diagnostic(
                code="SCORE_MISSING",
                severity="ERROR",
                message=f"score column {f'score_{score_name}'!r} is absent from selected logs",
            )
        )
    manifest_by_prompt = manifest.set_index("prompt_id", drop=False)
    unexpected = sorted({rollout.item_id for _, rollout in selected} - set(manifest["prompt_id"]))
    for prompt_id in unexpected[:100]:
        diagnostics.append(
            Diagnostic(
                code="PROMPT_UNEXPECTED",
                severity="ERROR",
                message=f"observed prompt {prompt_id!r} is absent from the design manifest",
                scope=scope_path("prompts", prompt_id),
                details={"prompt_id": prompt_id},
            )
        )

    selected.sort(key=lambda ranked: ranked[0])
    rollout_keys: set[tuple[Any, ...]] = set()
    observations: list[Observation] = []
    judge_verdicts: dict[str, dict[str, bool | None]] = {}
    source_hashes: dict[str, str | None] = {}
    # How each source is named in reports. The real path still reads and hashes the file.
    source_names: dict[str, str] = {}
    # Transcripts that could not be read, by report name: the reason and how many errors.
    unread_transcripts: dict[str, tuple[str, int]] = {}
    for collection_order, (_, rollout) in enumerate(selected, start=1):
        run_key = rollout.run_key
        prompt_id = rollout.item_id
        if prompt_id not in manifest_by_prompt.index:
            continue
        if rollout.sample_id is None or rollout.identity is None:
            # The same error the reader raised when it looked the sample id up.
            raise KeyError("sample_id")
        sample_id, identity = rollout.sample_id, rollout.identity
        run = run_records[run_key]
        cohort_id = run_cohorts[run_key]
        key_fields = [
            (field, *_rollout_field(plan, rollout, run, field))
            for field in plan.design.rollout_key
        ]
        rollout_key = tuple(value for _, value, _ in key_fields)
        if rollout_key in rollout_keys:
            shown = tuple(
                _RunValue(field) if of_run else value for field, value, of_run in key_fields
            )
            diagnostics.append(
                Diagnostic(
                    code="ROLLOUT_DUPLICATE",
                    severity="ERROR",
                    message=f"duplicate rollout key {shown!r}",
                    scope=scope_path("runs", run_key, "prompts", prompt_id),
                    details={"prompt_id": prompt_id},
                )
            )
        rollout_keys.add(rollout_key)
        scope = scope_path("runs", run_key, "prompts", prompt_id, "repeats", rollout.repeat)
        raw_error = _native(rollout.error)
        if isinstance(raw_error, str) and not raw_error.strip():
            raw_error = None
        raw_score = _native(scores[id(rollout)].get(score_name))
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
        # A content-filter stop that delivered nothing; one that delivered content before the
        # decline is scored like any other answer.
        stop = None
        # An answer without a score that stopped at the token limit, when the plan looks for
        # one; a truncated answer that was scored keeps its score.
        truncated = False
        if (
            raw_error is None
            and (stops != "score" or token_limit != "score")
            and stop_context is not None
        ):
            found = stop_context(rollout)
            if found is not None and found.stop_reason == "content_filter":
                if stops != "score" and not found.content_delivered:
                    stop = found
            elif found is not None and found.stop_reason == "max_tokens":
                truncated = token_limit != "score" and is_missing(raw_score)
        if raw_error is not None:
            context = source.error_context(rollout)
            if context.unread_reason is not None:
                name = report_path(rollout.source, plan)
                reason, count = unread_transcripts.get(name, (context.unread_reason, 0))
                unread_transcripts[name] = (reason, count + 1)
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
                        scope=scope,
                        details=_blocked_prompt(cohort_id, prompt_id),
                    )
                )
            elif (
                plan.inputs.score.errors == "end_to_end"
                and plan.inputs.estimand == "end_to_end_system"
                and context.stage == "target_generation"
                and context.content_delivered is False
                and context.provider_error_code in POLICY_BLOCK_CODES
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
                        scope=scope,
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
                        scope=scope,
                        details={
                            "integrity_degraded": True,
                            "error_stage": context.stage,
                            "content_delivered": context.content_delivered,
                            "provider_error_code": context.provider_error_code,
                        },
                    )
                )
        elif stop is not None:
            content_delivered = False
            block = {
                "stop_reason": "content_filter",
                "block_type": stop.block_type,
                "block_category": stop.block_category,
                "block_error": stop.block_error,
            }
            if stops == "safe":
                failure = False
                outcome_status = "target_policy_blocked_safe"
                diagnostics.append(
                    Diagnostic(
                        code="TARGET_POLICY_BLOCKED_SAFE",
                        severity="INFO",
                        message=(
                            "a provider's content filter stopped the output before it "
                            "delivered content; the plan counts the block as safe"
                        ),
                        scope=scope,
                        details=block,
                    )
                )
            else:
                failure = True
                outcome_status = "classified_failure"
                diagnostics.append(
                    Diagnostic(
                        code="TARGET_POLICY_BLOCKED_REFUSAL",
                        severity="INFO",
                        message=(
                            "a provider's content filter stopped the output before it "
                            "delivered content; the plan counts the block as a refusal"
                        ),
                        scope=scope,
                        details=block,
                    )
                )
        elif truncated and token_limit in {"safe", "failure"}:
            failure = token_limit == "failure"
            outcome_status = "classified_failure" if failure else "classified_safe"
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
                        scope=scope,
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
                        scope=scope,
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
                        scope=scope,
                        details=_blocked_prompt(cohort_id, prompt_id),
                    )
                )
        if truncated:
            counted_as = {"safe": "safe", "failure": "a failure", "missing": "a missing score"}
            diagnostics.append(
                Diagnostic(
                    code="TOKEN_LIMIT_REACHED",
                    severity="INFO",
                    message=(
                        "the answer stopped at the generation token limit without a score; "
                        f"the plan counts it as {counted_as[token_limit]}"
                    ),
                    scope=scope,
                    details={"stop_reason": "max_tokens", "token_limit": token_limit},
                )
            )
        log_path = rollout.source
        if log_path not in source_hashes:
            source_hashes[log_path] = source.source_sha256(log_path)
            source_names[log_path] = report_path(log_path, plan)
        config = _configuration(
            plan,
            run,
            source.configuration_fields,
            plan.cohorts[cohort_id].require_uniform,
        )
        design_row = manifest_by_prompt.loc[prompt_id]
        observation_id = sha256_value({**identity, "score": score_name})
        if judge_series:
            judge_verdicts[observation_id] = {
                judge_id: series[id(rollout)] for judge_id, series in judge_series.items()
            }
        observations.append(
            Observation(
                observation_id=observation_id,
                source=source_names[log_path],
                source_sha256=source_hashes[log_path],
                run_key=run_key,
                run_id=_native(run.run_id),
                rollout_id=sample_id,
                item_id=rollout.item_id,
                repeat=int(rollout.repeat),
                prompt_id=prompt_id,
                question_id=str(design_row["question_id"]),
                task_id=str(design_row["task_id"]),
                cohort_id=cohort_id,
                score_name=score_name,
                score_raw=raw_score,
                failure=failure,
                outcome_status=outcome_status,
                error=raw_error,
                error_stage=error_stage,
                content_delivered=content_delivered,
                provider_error_code=provider_error_code,
                retries=int(_native(rollout.retries) or 0),
                model=run.model if run.model is not None else "unknown",
                configuration_fingerprint=sha256_value(config),
                collection_order=collection_order,
                token_limit_unscored=None if token_limit == "score" else truncated,
            )
        )

    for name, (reason, count) in sorted(unread_transcripts.items()):
        diagnostics.append(
            Diagnostic(
                code="TRANSCRIPT_UNREAD",
                severity="WARNING",
                message=(
                    f"the transcript of {name} could not be read ({reason}), so the stage of "
                    f"its {count} sample error(s) is unknown and none counts as a policy block"
                ),
                scope=scope_path("logs", name),
                details={"source": name, "reason": reason, "sample_errors": count},
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
                        scope=prompt_scope(cohort_id, prompt_id),
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
                            scope=prompt_scope(cohort_id, prompt_id),
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
                            scope=prompt_scope(cohort_id, prompt_id),
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
                            scope=scope_path("strata", stratum),
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
                            scope=scope_path("strata", stratum),
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
                                scope=cohort_scope(cohort_id, "strata", stratum),
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
                source_format=dict(source.source_format),
                first_started_at=first_started_at,
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
                        scope=cohort_scope(cohort_id),
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
                        scope=cohort_scope(cohort_id),
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
        source_format=dict(source.source_format),
        first_started_at=first_started_at,
    )


def _receipt_timing(receipt: RegistrationReceipt, observed: list[Any]) -> list[Diagnostic]:
    """Whether a receipt that belongs to the plan predates the earliest observed start."""

    registered = receipt.registered_at.isoformat()
    details = {"plan_sha256": receipt.plan_sha256, "registered_at": registered}
    if observed and receipt.registered_at > min(observed):
        return [
            Diagnostic(
                code="PLAN_RECEIPT_POSTHOC",
                severity="ERROR",
                message=(
                    f"the plan was registered at {registered}, later than the earliest "
                    "selected evaluation"
                ),
                details=details,
            )
        ]
    checked = (
        "before the earliest selected evaluation"
        if observed
        else "but no selected evaluation reports a start time to check it against"
    )
    return [
        Diagnostic(
            code="PLAN_REGISTERED",
            severity="INFO",
            message=(
                f"the plan's hash was registered at {registered}, {checked}; the receipt's "
                "time is only as trustworthy as the external timestamp that anchors it"
            ),
            details=details,
        )
    ]


def read_inputs(
    plan: AnalysisPlan, source: EvaluationSource
) -> tuple[pd.DataFrame, list[ResolvedGroup], SourceData]:
    """Load the plan's design manifest and groups, and read the source, without normalizing.

    Split from ``ingest`` so a caller can normalize what was read more than once.
    """

    # ORDER MATTERS: the manifest is loaded before the source is read, so a broken manifest
    # is reported before any log is opened, as it always was.
    manifest = load_manifest(plan.inputs.design_manifest, plan_manifest_columns(plan))
    groups = resolve_groups(plan, manifest)
    return manifest, groups, source.read()


def ingest(plan: AnalysisPlan, source: EvaluationSource) -> NormalizedData:
    """Load the plan's design manifest, read the source and normalize it."""

    return normalize(plan, *read_inputs(plan, source))
