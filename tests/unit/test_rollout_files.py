from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from inspect_safety_stats.ingest.records import ErrorContext
from inspect_safety_stats.ingest.rollout_files import RolloutFiles, read_rollout_files


def _jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _runs(tmp_path: Path, *records: dict[str, object]) -> Path:
    return _jsonl(tmp_path / "runs.jsonl", list(records) or [{"run_key": "r1", "model": "m"}])


def test_jsonl_records_become_runs_and_rollouts(tmp_path: Path) -> None:
    runs = _runs(
        tmp_path,
        {
            "run_key": "r1",
            "model": "m",
            "status": "success",
            "started_at": "2026-01-01T00:00:00Z",
            "run_id": "batch-1",
            "temperature": 0.7,
        },
    )
    rollouts = _jsonl(
        tmp_path / "rollouts.jsonl",
        [
            {"run_key": "r1", "item_id": "p2", "repeat": 1, "scores": {"harmful": "PASS"}},
            {
                "run_key": "r1",
                "item_id": "p1",
                "repeat": 2,
                "rollout_id": "x-7",
                "score_harmful": 1,
                "error": "refused by provider",
                "error_stage": "target_generation",
                "content_delivered": False,
                "provider_error_code": "content_filter",
                "retries": 2,
                "shard": "a",
            },
            {"run_key": "r1", "item_id": "p1", "scores": {"harmful": "FAIL", "other": 0.5}},
        ],
    )
    data = RolloutFiles(runs=runs, rollouts=rollouts).read()

    (run,) = data.runs
    assert run.run_key == "r1"
    assert run.model == "m"
    assert run.status == "success"
    assert run.started_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert run.run_id == "batch-1"
    assert run.attributes["temperature"] == 0.7
    assert run.source == str(runs)

    first, second, third = data.rollouts
    assert (first.item_id, first.repeat, first.sample_id) == ("p2", 1, "r1:p2:1")
    assert second.sample_id == "x-7"
    assert second.identity == {"run_key": "r1", "rollout_id": "x-7", "item_id": "p1", "repeat": 2}
    assert dict(second.scores) == {"harmful": 1}
    assert second.attributes["shard"] == "a"
    assert (second.error, second.retries) == ("refused by provider", 2)
    assert third.repeat == 1
    assert data.score_names == frozenset({"harmful", "other"})
    # Collected by run, item and repeat.
    assert list(data.collection_rank) == [2, 1, 0]
    assert data.error_context(second) == ErrorContext(
        stage="target_generation", content_delivered=False, provider_error_code="content_filter"
    )
    assert data.error_context(first) == ErrorContext(
        stage="unknown", content_delivered=None, provider_error_code=None
    )
    assert data.source_sha256(str(rollouts)) is not None
    assert data.source_sha256(str(tmp_path / "absent")) is None


def test_csv_cells_are_typed_explicitly(tmp_path: Path) -> None:
    runs = tmp_path / "runs.csv"
    runs.write_text("run_key,model,status\nr1,m,\n", encoding="utf-8")
    rollouts = tmp_path / "rollouts.csv"
    rollouts.write_text(
        "run_key,item_id,repeat,score_harmful,score_label,content_delivered,error\n"
        "r1,p1,1,1,PASS,false,\n"
        "r1,p1,2,0.25,,true,timeout\n",
        encoding="utf-8",
    )
    data = read_rollout_files(runs, rollouts)
    # An empty status cell is missing, and a run without a status counts as complete.
    assert data.runs[0].status == "success"
    first, second = data.rollouts
    assert dict(first.scores) == {"harmful": 1, "label": "PASS"}
    assert dict(second.scores) == {"harmful": 0.25, "label": None}
    assert first.error is None
    assert second.error == "timeout"
    assert data.error_context(second).content_delivered is True


@pytest.mark.parametrize(
    ("runs", "rollouts", "message"),
    [
        ([{"model": "m"}], [], "field 'run_key' is required"),
        ([{"run_key": "r1"}, {"run_key": "r1"}], [], "appears more than once"),
        ([{"run_key": "r1"}], [{"run_key": "r2", "item_id": "p1"}], "names no run"),
        ([{"run_key": "r1"}], [{"run_key": "r1"}], "field 'item_id' is required"),
        ([{"run_key": "r1"}], [{"run_key": "r1", "item_id": "p", "repeat": 0}], "repeat 0"),
        ([{"run_key": "r1"}], [{"run_key": "r1", "item_id": "p", "repeat": True}], "repeat True"),
        (
            [{"run_key": "r1"}],
            [{"run_key": "r1", "item_id": "p", "error_stage": "later"}],
            "error_stage 'later'",
        ),
        (
            [{"run_key": "r1"}],
            [{"run_key": "r1", "item_id": "p", "content_delivered": "yes"}],
            "content_delivered 'yes'",
        ),
        (
            [{"run_key": "r1"}],
            [{"run_key": "r1", "item_id": "p", "scores": {"a": 1}, "score_a": 0}],
            "score 'a' is given twice",
        ),
        ([{"run_key": "r1", "started_at": "soon"}], [], "not an ISO 8601 time"),
        ([], [], "contains no runs"),
        ([{"run_key": "r1"}], [], "contains no rollouts"),
    ],
)
def test_malformed_files_are_rejected_with_the_record_named(
    tmp_path: Path, runs: list[dict[str, object]], rollouts: list[dict[str, object]], message: str
) -> None:
    runs_path = _jsonl(tmp_path / "runs.jsonl", runs)
    rollouts_path = _jsonl(tmp_path / "rollouts.jsonl", rollouts)
    with pytest.raises(ValueError, match=message):
        read_rollout_files(runs_path, rollouts_path)


def test_a_line_that_is_not_a_json_object_is_rejected(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text('{"run_key": "r1", "item_id": "p"}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"rollouts.jsonl:2: not a JSON object"):
        read_rollout_files(runs, rollouts)
