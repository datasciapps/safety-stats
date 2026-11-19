"""The labelling protocol's plan rules and its queue checks (roadmap F41)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from inspect_safety_stats.config.models import JudgeCalibrationSpec
from inspect_safety_stats.ingest.validation import check_labelled_prefix


def _spec(protocol: dict[str, Any] | None, **fields: Any) -> JudgeCalibrationSpec:
    values: dict[str, Any] = {"mode": "validated", "validation_manifest": "labels.csv"}
    values["family"] = "judge_error"
    values.update(fields)
    if protocol is not None:
        values["protocol"] = protocol
    return JudgeCalibrationSpec.model_validate(values)


def test_a_calibration_without_a_protocol_dumps_as_before() -> None:
    assert "protocol" not in _spec(None).model_dump(mode="json")


@pytest.mark.parametrize(
    ("protocol", "fields", "message"),
    [
        ({"queue": "q.csv", "stopping": "fixed_size"}, {}, "fixed_size requires a size"),
        (
            {"queue": "q.csv", "stopping": "label_independent", "size": 5},
            {},
            "size applies only to stopping: fixed_size",
        ),
        ({"queue": "q.csv", "stopping": "fixed_size", "size": True}, {}, "not a boolean"),
        ({"queue": "q.csv", "stopping": "fixed_size", "size": 0}, {}, "greater than 0"),
        ({"queue": "q.csv", "stopping": "whenever"}, {}, "stopping"),
        (
            {"queue": "q.csv", "stopping": "data_dependent"},
            {"sampling": "simple_random"},
            "data_dependent requires sampling: simple_random_within_judge_stratum",
        ),
        (
            {"queue": "q.csv", "stopping": "data_dependent"},
            {
                "sampling": "stratified_within_judge_stratum",
                "validation_strata": "strata.csv",
            },
            "data_dependent requires sampling: simple_random_within_judge_stratum",
        ),
    ],
)
def test_plan_rules(protocol: dict[str, Any], fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _spec(protocol, **fields)


def test_an_assumed_calibration_takes_no_protocol() -> None:
    with pytest.raises(ValidationError, match="labelling protocol requires mode: validated"):
        JudgeCalibrationSpec.model_validate(
            {
                "mode": "assumed",
                "scenarios": [
                    {
                        "id": "s",
                        "rho_1": {"lower": 0.9, "upper": 1.0},
                        "rho_0": {"lower": 0.0, "upper": 0.1},
                    }
                ],
                "primary": "s",
                "protocol": {"queue": "q.csv", "stopping": "label_independent"},
            }
        )


def _queue(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "queue.csv"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_the_whole_queue_labelled_is_a_prefix(tmp_path: Path) -> None:
    path = _queue(tmp_path, "observation_id\na\nb\n")
    prefix = check_labelled_prefix(path, {"a": True, "b": False})
    assert (prefix.queue_rows, prefix.labelled) == (2, 2)


@pytest.mark.parametrize(
    ("text", "tier", "message"),
    [
        ("id\na\n", None, "missing the observation_id column"),
        ("observation_id,tier\na,x\n,x\n", None, "row 2 has a blank observation_id"),
        ("observation_id\na\nb\na\n", None, "lists observation 'a' twice, at rows 1 and 3"),
        ("observation_id\na\n", "calibration", "no tier column"),
        ("observation_id,tier\nb,other\na,calibration\n", "calibration", "not in the"),
    ],
)
def test_queue_checks(tmp_path: Path, text: str, tier: str | None, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_labelled_prefix(_queue(tmp_path, text), {"a": True, "b": False}, tier=tier)
