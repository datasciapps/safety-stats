"""The Inspect AI adapter: reads Inspect evaluation logs into runs and rollouts.

Everything that knows Inspect's log format lives here: the columns of ``evals_df`` and
``samples_df``, Inspect's identifiers, its collection order, and how a sample error's stage is
read from the transcript. The rules of the plan are applied by
:func:`inspect_safety_stats.ingest.normalize.normalize`.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

try:
    from inspect_ai.analysis import EvalColumn, EvalColumns, evals_df, samples_df
    from inspect_ai.log import EvalLog, read_eval_log
except ImportError as error:  # pragma: no cover - exercised in a fresh interpreter
    # Only a missing package of the extra (pyproject.toml) means the extra is not installed.
    # Any other import failure, such as a module that an installed Inspect cannot find or a
    # pyarrow built for another NumPy, keeps its own message: "install the extra" would send
    # the user to install what they already have.
    import importlib.util

    if all(importlib.util.find_spec(package) is not None for package in ("inspect_ai", "pyarrow")):
        raise
    raise ImportError(
        "Reading Inspect AI logs needs the inspect extra: "
        "pip install 'inspect-safety-stats[inspect]'. "
        "Rollout files (--runs and --rollouts) need no extra."
    ) from error

from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.files import file_sha256
from inspect_safety_stats.ingest.normalize import POLICY_BLOCK_CODES, ingest
from inspect_safety_stats.ingest.outcomes import is_missing
from inspect_safety_stats.ingest.records import (
    ErrorContext,
    NormalizedData,
    RolloutRecord,
    RunRecord,
    SampleOrder,
    SourceData,
    StopContext,
)

__all__ = [
    "ErrorContext",
    "InspectLogs",
    "InspectRollouts",
    "LogInput",
    "NormalizedData",
    "ingest_inspect_logs",
    "read_inspect_rollouts",
    "read_inspect_runs",
    "read_inspect_sample_orders",
    "read_inspect_source",
]

LogInput = str | Path | EvalLog | Sequence[str | Path] | Sequence[EvalLog] | None


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
    return next((code for code in sorted(POLICY_BLOCK_CODES) if code in text), None)


# What reading a log raises when the log is missing, truncated or not a log. Any other
# exception is a defect, and propagates rather than passing for an unknown error stage.
_UNREADABLE_LOG = (OSError, ValueError, zipfile.BadZipFile)


@dataclass(frozen=True)
class _Transcripts:
    """The error contexts of one log's samples, or why the log could not be read."""

    contexts: dict[str, ErrorContext]
    unread_reason: str | None = None


def _error_context(
    *,
    log_path: str,
    eval_id: str,
    sample_id: str,
    provided: dict[str, EvalLog],
    cache: dict[str, _Transcripts],
) -> ErrorContext:
    cache_key = eval_id if eval_id in provided else log_path
    if cache_key not in cache:
        contexts: dict[str, ErrorContext] = {}
        unread_reason: str | None = None
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
        except _UNREADABLE_LOG as error:
            contexts = {}
            # The exception's type only: its message may name local paths, which reports
            # do not record.
            unread_reason = type(error).__name__
        cache[cache_key] = _Transcripts(contexts, unread_reason)
    transcripts = cache[cache_key]
    return transcripts.contexts.get(
        sample_id,
        ErrorContext(
            stage="unknown",
            content_delivered=None,
            provider_error_code=None,
            unread_reason=transcripts.unread_reason,
        ),
    )


def _carries_content(message: Any) -> bool:
    """Whether a generated message carries any content: text, readable reasoning, a tool call
    or any other part, such as an image or a provider's own tool use. Reasoning that the
    provider returned only encrypted (redacted) or only as a signature is not readable."""

    if getattr(message, "tool_calls", None):
        return True
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return bool(content.strip())
    for part in content or []:
        kind = getattr(part, "type", None)
        if kind == "text":
            texts = [getattr(part, "text", None)]
        elif kind == "reasoning":
            texts = [getattr(part, "summary", None)]
            if not getattr(part, "redacted", False):
                texts.append(getattr(part, "reasoning", None))
        else:
            return True
        if any(isinstance(text, str) and text.strip() for text in texts):
            return True
    return False


def _delivers_content(output: Any) -> bool:
    """Whether a model output delivered content. An output reported as an API error delivered
    none, whatever stand-in text Inspect put in its place."""

    if output is None or getattr(output, "error", None) is not None:
        return False
    choices = getattr(output, "choices", None) or []
    return any(_carries_content(getattr(choice, "message", None)) for choice in choices)


def _message_id(output: Any) -> str | None:
    choices = getattr(output, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    identifier = getattr(message, "id", None)
    return identifier if isinstance(identifier, str) else None


def _content_before_stop(sample: Any) -> Literal["earlier_generation", "unverifiable"] | None:
    """Whether the target delivered content in a sample before the generation that a content
    filter stopped, which is the sample's output.

    The generations before the stopped one are the model events that the transcript records
    before it, such as the earlier turns of a conversation or the steps of an agent; the
    scorers' model calls come after it. Any model's content counts, since the transcript does
    not say which calls the end-to-end system delivered. When the transcript does not record
    the stopped generation, content delivered before it cannot be ruled out.
    """

    stopped = _message_id(getattr(sample, "output", None))
    generations = [
        event for event in (sample.events or []) if getattr(event, "event", None) == "model"
    ]
    recorded = [
        index
        for index, event in enumerate(generations)
        if stopped is not None and _message_id(getattr(event, "output", None)) == stopped
    ]
    if not recorded:
        return "unverifiable"
    earlier = generations[: recorded[-1]]
    if any(
        getattr(event, "error", None) is None
        and _delivers_content(getattr(event, "output", None))
        for event in earlier
    ):
        return "earlier_generation"
    return None


def _stop_of(output: Any) -> StopContext | None:
    """The content-filter stop of a sample's output, or None when no content filter stopped it.

    Claude 5-series models report a safety-classifier decline as a normal answer with stop
    reason `refusal`, which Inspect records as `content_filter`, with the provider's category in
    the choice's `stop_details` (Inspect 0.3.286 and later). A decline that arrives as an API
    error comes back with the output's `error` set and Inspect's stand-in apology as content.
    Only the output itself is read: :func:`_sample_stop` adds the sample's earlier generations.
    """

    if output is None or getattr(output, "stop_reason", None) != "content_filter":
        return None
    choices = getattr(output, "choices", None) or []
    details = getattr(choices[0], "stop_details", None) if choices else None
    delivered = _delivers_content(output)
    return StopContext(
        content_delivered=delivered,
        block_type=getattr(details, "type", None),
        block_category=getattr(details, "category", None),
        block_error=getattr(output, "error", None) is not None,
        delivered_in="stopped_output" if delivered else None,
    )


def _sample_stop(sample: Any) -> StopContext | None:
    """The content-filter stop of a sample, which delivered content when the stopped output or
    any earlier generation of the sample did. A stop after an earlier turn delivered content
    did not keep that content from the user (statistical methods, section 4.6)."""

    stop = _stop_of(getattr(sample, "output", None))
    if stop is None or stop.content_delivered:
        return stop
    before = _content_before_stop(sample)
    if before is None:
        return stop
    return replace(stop, content_delivered=True, delivered_in=before)


def _token_limit_stop_of(output: Any) -> StopContext | None:
    """The stop of a sample's output at the generation token limit, or None for any other end.

    Inspect records an answer that ran out of generation tokens with stop reason `max_tokens`.
    Stop reason `model_length`, a conversation that outgrew the context window, and a sample
    that hit one of Inspect's own sample limits, such as one of type `token`, are not read:
    such a rollout counts by its score, as docs/plan-reference.md says.
    """

    if output is None or getattr(output, "stop_reason", None) != "max_tokens":
        return None
    completion = getattr(output, "completion", "") or ""
    return StopContext(
        content_delivered=bool(completion.strip()),
        block_type=None,
        block_category=None,
        block_error=False,
        stop_reason="max_tokens",
    )


def _stop_context(
    *,
    log_path: str,
    eval_id: str,
    sample_id: str,
    provided: dict[str, EvalLog],
    cache: dict[str, dict[str, StopContext]],
) -> StopContext | None:
    # Read only when the plan asks for content-filter or token-limit stops. Unlike error
    # contexts, a log that cannot be read is an error: the plan's outcome of a stop would
    # otherwise depend on it.
    cache_key = eval_id if eval_id in provided else log_path
    if cache_key not in cache:
        log = provided.get(eval_id) or read_eval_log(log_path)
        stops: dict[str, StopContext] = {}
        for sample in log.samples or []:
            stop = _sample_stop(sample) or _token_limit_stop_of(sample.output)
            if stop is not None:
                stops[str(sample.uuid)] = stop
        cache[cache_key] = stops
    return cache[cache_key].get(sample_id)


# The Inspect versions whose log format this adapter reads; keep in step with pyproject.toml.
SOURCE_FORMAT = {"format": "inspect_logs", "inspect_version_constraint": ">=0.3.259,<0.4"}

_CONFIGURATION_FIELDS = (
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
)


# The run fields read from each log: Inspect's defaults, and the scorers with their options, so
# a report can record the configuration of each declared judge. ``evals_df`` leaves them out.
_RUN_COLUMNS = [*EvalColumns, EvalColumn("scorers", path="eval.scorers")]


def _parse_time(value: Any) -> datetime | None:
    if is_missing(value):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    result = parsed.to_pydatetime()
    return result if isinstance(result, datetime) else None


def _runs(evaluations: pd.DataFrame) -> list[RunRecord]:
    return [
        RunRecord(
            run_key=str(row["eval_id"]),
            attributes={str(key): value for key, value in row.items()},
            status=None if is_missing(row.get("status")) else row["status"],
            started_at=_parse_time(row.get("created")),
            model=str(row["model"]) if "model" in row else None,
            run_id=row.get("run_id"),
            source=str(row.get("log", "")),
        )
        for _, row in evaluations.iterrows()
    ]


@dataclass(frozen=True)
class InspectRollouts:
    """The rollouts of Inspect logs, with what :class:`SourceData` needs to know about them."""

    rollouts: list[RolloutRecord]
    collection_rank: list[int]
    score_names: frozenset[str]


def _rollouts(samples: pd.DataFrame, runs: list[RunRecord]) -> InspectRollouts:
    run_sources = {run.run_key: run.source for run in runs}
    # Rollouts are collected in Inspect's order: by evaluation, sample id and epoch.
    collected = (
        samples.reset_index(drop=True)
        .sort_values(["eval_id", "id", "epoch"], kind="stable")
        .index
    )
    collection_rank = [0] * len(samples)
    for rank, position in enumerate(collected):
        collection_rank[position] = rank
    score_columns = [column for column in samples.columns if column.startswith("score_")]
    rollouts: list[RolloutRecord] = []
    for label, row in samples.iterrows():
        eval_id = str(row["eval_id"])
        sample_id = str(row["sample_id"]) if "sample_id" in row else None
        rollouts.append(
            RolloutRecord(
                run_key=eval_id,
                record_id=label,
                item_id=str(row["id"]),
                repeat=row["epoch"],
                sample_id=sample_id,
                identity=None
                if sample_id is None
                else {
                    "eval_id": eval_id,
                    "sample_id": sample_id,
                    "inspect_id": str(row["id"]),
                    "epoch": int(row["epoch"]),
                },
                attributes={str(key): value for key, value in row.items()},
                scores={column.removeprefix("score_"): row[column] for column in score_columns},
                error=row.get("error"),
                retries=row.get("retries"),
                source=str(row.get("log", run_sources.get(eval_id, ""))),
            )
        )
    return InspectRollouts(
        rollouts=rollouts,
        collection_rank=collection_rank,
        score_names=frozenset(column.removeprefix("score_") for column in score_columns),
    )


def read_inspect_runs(logs: LogInput) -> list[RunRecord]:
    """Read the runs of Inspect logs, one per evaluation."""

    evaluations = evals_df(logs, columns=_RUN_COLUMNS)
    if evaluations.empty:
        raise ValueError("no Inspect evaluation logs were found")
    return _runs(evaluations)


def read_inspect_rollouts(
    logs: LogInput, runs: list[RunRecord], *, parallel: bool | int = False
) -> InspectRollouts:
    """Read the rollouts of Inspect logs, one per sample and epoch."""

    samples = samples_df(logs, parallel=parallel)
    if samples.empty:
        raise ValueError("selected Inspect logs contain no sample records")
    return _rollouts(samples, runs)


def read_inspect_source(logs: LogInput, *, parallel: bool | int = False) -> SourceData:
    """Read Inspect evaluation logs into the runs and rollouts :func:`normalize` takes."""

    # ORDER MATTERS: both frames are read before either is checked, so that a run without
    # samples reports the same error it always did.
    evaluations = evals_df(logs, columns=_RUN_COLUMNS)
    samples = samples_df(logs, parallel=parallel)
    if evaluations.empty:
        raise ValueError("no Inspect evaluation logs were found")
    if samples.empty:
        raise ValueError("selected Inspect logs contain no sample records")
    runs = _runs(evaluations)
    collected = _rollouts(samples, runs)
    provided = _provided_eval_logs(logs)
    cache: dict[str, _Transcripts] = {}
    stop_cache: dict[str, dict[str, StopContext]] = {}

    def error_context(rollout: RolloutRecord) -> ErrorContext:
        return _error_context(
            log_path=rollout.source,
            eval_id=rollout.run_key,
            sample_id=str(rollout.sample_id),
            provided=provided,
            cache=cache,
        )

    def stop_context(rollout: RolloutRecord) -> StopContext | None:
        return _stop_context(
            log_path=rollout.source,
            eval_id=rollout.run_key,
            sample_id=str(rollout.sample_id),
            provided=provided,
            cache=stop_cache,
        )

    return SourceData(
        runs=runs,
        rollouts=collected.rollouts,
        collection_rank=collected.collection_rank,
        score_names=collected.score_names,
        configuration_fields=_CONFIGURATION_FIELDS,
        error_context=error_context,
        source_sha256=file_sha256,
        source_format=SOURCE_FORMAT,
        stop_context=stop_context,
    )


def read_inspect_sample_orders(
    sources: Mapping[str, str], logs: LogInput = None
) -> dict[str, SampleOrder]:
    """The recorded sample order of each run, from its log header.

    ``sources`` maps each run's ``eval_id`` to the log it was read from. A log passed in
    ``logs`` as an ``EvalLog`` is used as given; any other is read from disk, header only.
    Inspect records ``sample_shuffle`` in the eval config and lists the sample ids in the
    shuffled order in ``eval.dataset.sample_ids``, which is the order it schedules them in:
    every sample's first epoch, then every sample's second, and so on.
    """

    provided = _provided_eval_logs(logs)
    orders: dict[str, SampleOrder] = {}
    for run_key, source in sources.items():
        log = provided.get(run_key) or read_eval_log(source, header_only=True)
        ids = log.eval.dataset.sample_ids
        orders[run_key] = SampleOrder(
            run_key=run_key,
            source=source,
            shuffle=log.eval.config.sample_shuffle,
            shuffled=log.eval.dataset.shuffled,
            sample_ids=None if ids is None else tuple(str(item) for item in ids),
        )
    return orders


@dataclass(frozen=True)
class InspectLogs:
    """Inspect AI evaluation logs, as an evaluation source."""

    logs: LogInput
    parallel: bool | int = False

    def read(self) -> SourceData:
        return read_inspect_source(self.logs, parallel=self.parallel)


def ingest_inspect_logs(
    plan: AnalysisPlan,
    logs: LogInput,
    *,
    parallel: bool | int = False,
) -> NormalizedData:
    return ingest(plan, InspectLogs(logs, parallel=parallel))
