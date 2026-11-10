"""The neutral adapter: reads runs and rollouts from plain JSONL or CSV files.

Any evaluation system can be analysed by writing two files: one record per run, and one
record per rollout. The format is described in ``docs/rollout-files.md``. A file whose name
ends in ``.csv`` is read as CSV; any other file as JSON Lines, one object per line.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, get_args

from inspect_safety_stats.ingest.files import file_sha256
from inspect_safety_stats.ingest.records import (
    ErrorContext,
    RolloutRecord,
    RunRecord,
    SourceData,
)

__all__ = ["RolloutFiles", "read_rollout_files"]

# Fields with a meaning of their own; every other field is an attribute that cohort
# ``match``, ``require_uniform`` and ``rollout_key`` may name.
RUN_FIELDS = ("run_key", "model", "status", "started_at", "run_id")
ROLLOUT_FIELDS = (
    "run_key",
    "item_id",
    "repeat",
    "rollout_id",
    "scores",
    "error",
    "error_stage",
    "content_delivered",
    "provider_error_code",
    "retries",
)
SCORE_PREFIX = "score_"
ErrorStage = Literal["target_generation", "post_generation", "unknown"]
# The run fields that make up a run's configuration, besides those a cohort requires uniform.
CONFIGURATION_FIELDS = ("model",)


def _csv_value(text: str) -> Any:
    """A CSV cell: empty is missing, a JSON number or boolean is that value, else text."""

    if text == "":
        return None
    if text in {"true", "false"}:
        return text == "true"
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else text


def _records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return [
                {str(key): _csv_value(value or "") for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not a JSON object: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: not a JSON object")
            records.append(record)
    return records


def _required_text(record: Mapping[str, Any], field: str, where: str) -> str:
    value = record.get(field)
    if value is None or isinstance(value, bool) or str(value) == "":
        raise ValueError(f"{where}: field {field!r} is required")
    return str(value)


def _started_at(value: Any, where: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{where}: started_at {value!r} is not an ISO 8601 time") from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _repeat(value: Any, where: str) -> int:
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: repeat {value!r} is not a positive integer")
    return value


def _scores(record: Mapping[str, Any], where: str) -> dict[str, Any]:
    nested = record.get("scores")
    if nested is not None and not isinstance(nested, dict):
        raise ValueError(f"{where}: scores must be an object of score name to value")
    scores = {str(name): value for name, value in (nested or {}).items()}
    for field, value in record.items():
        if field.startswith(SCORE_PREFIX):
            name = field.removeprefix(SCORE_PREFIX)
            if name in scores:
                raise ValueError(f"{where}: score {name!r} is given twice")
            scores[name] = value
    return scores


def _error_context(record: Mapping[str, Any], where: str) -> ErrorContext:
    stage: ErrorStage = "unknown"
    given = record.get("error_stage")
    if given is not None:
        if given not in get_args(ErrorStage):
            raise ValueError(
                f"{where}: error_stage {given!r} is not one of {list(get_args(ErrorStage))}"
            )
        stage = given
    delivered = record.get("content_delivered")
    if delivered is not None and not isinstance(delivered, bool):
        raise ValueError(f"{where}: content_delivered {delivered!r} is not true or false")
    code = record.get("provider_error_code")
    return ErrorContext(
        stage=stage,
        content_delivered=delivered,
        provider_error_code=None if code is None else str(code),
    )


def read_rollout_files(runs: str | Path, rollouts: str | Path) -> SourceData:
    """Read a runs file and a rollouts file into the runs and rollouts ``normalize`` takes."""

    runs_path, rollouts_path = Path(runs), Path(rollouts)
    run_records: list[RunRecord] = []
    seen: set[str] = set()
    for number, record in enumerate(_records(runs_path), start=1):
        where = f"{runs_path}: run {number}"
        run_key = _required_text(record, "run_key", where)
        if run_key in seen:
            raise ValueError(f"{where}: run_key {run_key!r} appears more than once")
        seen.add(run_key)
        model = record.get("model")
        run_records.append(
            RunRecord(
                run_key=run_key,
                attributes=dict(record),
                status=record.get("status"),
                started_at=_started_at(record.get("started_at"), where),
                model=None if model is None else str(model),
                run_id=record.get("run_id"),
                source=str(runs_path),
            )
        )
    if not run_records:
        raise ValueError(f"{runs_path} contains no runs")

    rollout_records: list[RolloutRecord] = []
    contexts: dict[int, ErrorContext] = {}
    score_names: set[str] = set()
    for number, record in enumerate(_records(rollouts_path), start=1):
        where = f"{rollouts_path}: rollout {number}"
        run_key = _required_text(record, "run_key", where)
        if run_key not in seen:
            raise ValueError(f"{where}: run_key {run_key!r} names no run in {runs_path}")
        item_id = _required_text(record, "item_id", where)
        repeat = _repeat(record.get("repeat"), where)
        rollout_id = str(record.get("rollout_id") or f"{run_key}:{item_id}:{repeat}")
        scores = _scores(record, where)
        score_names.update(scores)
        rollout = RolloutRecord(
            run_key=run_key,
            record_id=number,
            item_id=item_id,
            repeat=repeat,
            sample_id=rollout_id,
            identity={
                "run_key": run_key,
                "rollout_id": rollout_id,
                "item_id": item_id,
                "repeat": repeat,
            },
            attributes={**record, "repeat": repeat, "rollout_id": rollout_id},
            scores=scores,
            error=record.get("error"),
            retries=record.get("retries"),
            source=str(rollouts_path),
        )
        contexts[id(rollout)] = _error_context(record, where)
        rollout_records.append(rollout)
    if not rollout_records:
        raise ValueError(f"{rollouts_path} contains no rollouts")

    # Rollouts are collected by run, item and repeat, as the Inspect adapter collects them.
    order = sorted(
        range(len(rollout_records)),
        key=lambda index: (
            rollout_records[index].run_key,
            rollout_records[index].item_id,
            rollout_records[index].repeat,
        ),
    )
    collection_rank = [0] * len(rollout_records)
    for rank, index in enumerate(order):
        collection_rank[index] = rank

    def error_context(rollout: RolloutRecord) -> ErrorContext:
        return contexts[id(rollout)]

    return SourceData(
        runs=run_records,
        rollouts=rollout_records,
        collection_rank=collection_rank,
        score_names=frozenset(score_names),
        configuration_fields=CONFIGURATION_FIELDS,
        error_context=error_context,
        source_sha256=file_sha256,
    )


@dataclass(frozen=True)
class RolloutFiles:
    """A runs file and a rollouts file, as an evaluation source."""

    runs: str | Path
    rollouts: str | Path

    def read(self) -> SourceData:
        return read_rollout_files(self.runs, self.rollouts)
