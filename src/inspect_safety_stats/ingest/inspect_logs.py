"""The Inspect AI adapter: reads Inspect evaluation logs into runs and rollouts.

Everything that knows Inspect's log format lives here: the columns of ``evals_df`` and
``samples_df``, Inspect's identifiers, its collection order, and how a sample error's stage is
read from the transcript. The rules of the plan are applied by
:func:`inspect_safety_stats.ingest.normalize.normalize`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from inspect_ai.analysis import evals_df, samples_df
from inspect_ai.log import EvalLog, read_eval_log

from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.normalize import POLICY_BLOCK_CODES, ingest
from inspect_safety_stats.ingest.outcomes import is_missing
from inspect_safety_stats.ingest.records import (
    ErrorContext,
    NormalizedData,
    RolloutRecord,
    RunRecord,
    SourceData,
)

__all__ = [
    "ErrorContext",
    "InspectLogs",
    "LogInput",
    "NormalizedData",
    "ingest_inspect_logs",
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


def _hash_file(path: str) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _parse_time(value: Any) -> datetime | None:
    if is_missing(value):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    result = parsed.to_pydatetime()
    return result if isinstance(result, datetime) else None


def read_inspect_source(logs: LogInput, *, parallel: bool | int = False) -> SourceData:
    """Read Inspect evaluation logs into the runs and rollouts :func:`normalize` takes."""

    evaluations = evals_df(logs)
    samples = samples_df(logs, parallel=parallel)
    if evaluations.empty:
        raise ValueError("no Inspect evaluation logs were found")
    if samples.empty:
        raise ValueError("selected Inspect logs contain no sample records")

    runs: list[RunRecord] = []
    eval_logs: dict[str, Any] = {}
    for _, row in evaluations.iterrows():
        eval_id = str(row["eval_id"])
        eval_logs[eval_id] = row.get("log", "")
        runs.append(
            RunRecord(
                run_key=eval_id,
                attributes={str(key): value for key, value in row.items()},
                status=row.get("status", "success"),
                started_at=_parse_time(row.get("created")),
                model=str(row["model"]) if "model" in row else None,
                run_id=row.get("run_id"),
                source=str(row.get("log", "")),
            )
        )

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
        rollouts.append(
            RolloutRecord(
                run_key=eval_id,
                record_id=label,
                item_id=str(row["id"]),
                repeat=row["epoch"],
                sample_id=str(row["sample_id"]),
                identity={
                    "eval_id": eval_id,
                    "sample_id": str(row["sample_id"]),
                    "inspect_id": str(row["id"]),
                    "epoch": int(row["epoch"]),
                },
                attributes={str(key): value for key, value in row.items()},
                scores={column.removeprefix("score_"): row[column] for column in score_columns},
                error=row.get("error"),
                retries=row.get("retries"),
                source=str(row.get("log", eval_logs.get(eval_id, ""))),
            )
        )
    provided = _provided_eval_logs(logs)
    cache: dict[str, dict[str, ErrorContext]] = {}

    def error_context(rollout: RolloutRecord) -> ErrorContext:
        return _error_context(
            log_path=rollout.source,
            eval_id=rollout.run_key,
            sample_id=rollout.sample_id,
            provided=provided,
            cache=cache,
        )

    return SourceData(
        runs=runs,
        rollouts=rollouts,
        collection_rank=collection_rank,
        score_names=frozenset(column.removeprefix("score_") for column in score_columns),
        configuration_fields=_CONFIGURATION_FIELDS,
        error_context=error_context,
        source_sha256=_hash_file,
    )


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
