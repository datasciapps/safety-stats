"""The labelling protocol's plan rules and its queue checks (roadmap F41)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from inspect_safety_stats.analysis.judge_calibration import (
    _fine_queue_shares,
    _partial_queue_diagnostic,
    _queue_shares,
)
from inspect_safety_stats.config.models import JudgeCalibrationSpec
from inspect_safety_stats.ingest.validation import check_labelled_prefix
from inspect_safety_stats.stats.judge_calibration import FineStrata, JudgeStrata
from inspect_safety_stats.stats.misclassification import JudgeStratumCounts


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
        # The queue's tier cells are read stripped, so a padded tier would select nothing.
        (
            {"queue": "q.csv", "tier": " calibration", "stopping": "label_independent"},
            {},
            "tier ' calibration' begins or ends with whitespace",
        ),
        (
            {"queue": "q.csv", "tier": "calibration\t", "stopping": "label_independent"},
            {},
            "begins or ends with whitespace",
        ),
        (
            {"queue": "q.csv", "tier": "  ", "stopping": "label_independent"},
            {},
            "begins or ends with whitespace",
        ),
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
    assert prefix.observation_ids == ("a", "b")


def test_a_tier_keeps_its_own_rows_in_queue_order(tmp_path: Path) -> None:
    text = "observation_id,tier\nc,other\na, calibration\nd,other\nb,calibration\n"
    path = _queue(tmp_path, text)
    prefix = check_labelled_prefix(path, {"a": True}, tier="calibration")
    assert (prefix.observation_ids, prefix.queue_rows, prefix.labelled) == (("a", "b"), 2, 1)


def test_the_queue_hash_is_of_the_bytes_the_check_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The queue is rewritten as soon as it has been read once: a check that hashed one read
    # and parsed a second would record the first version's hash for the second's rows.
    first = b"observation_id\na\nb\n"
    path = tmp_path / "queue.csv"
    path.write_bytes(first)
    read_bytes = Path.read_bytes

    def read_then_rewrite(self: Path) -> bytes:
        content = read_bytes(self)
        if self == path:
            path.write_bytes(first + b"c\n")
        return content

    monkeypatch.setattr(Path, "read_bytes", read_then_rewrite)
    prefix = check_labelled_prefix(path, {"a": True, "b": False})
    assert prefix.queue_sha256 == hashlib.sha256(first).hexdigest()
    assert prefix.queue_rows == 2


@pytest.mark.parametrize(
    "suffix",
    [".gz", ".bz2", ".zip", ".xz", ".zst", ".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".GZ"],
)
def test_a_compressed_queue_is_read_as_pandas_reads_its_path(
    tmp_path: Path, suffix: str
) -> None:
    # The queue is parsed from the bytes it is hashed from, which carry no file name, so the
    # check must take the compression pandas infers from the name, as reading the path did.
    if suffix == ".zst":
        pytest.importorskip("zstandard")
    path = tmp_path / f"queue.csv{suffix}"
    rows = pd.DataFrame({"observation_id": ["a", "b", "c"], "tier": ["x", "x", "y"]})
    # A byte-order mark and CRLF line endings inside the archive read as they do uncompressed.
    rows.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\r\n")
    by_path = pd.read_csv(path, dtype=str, keep_default_na=False)

    prefix = check_labelled_prefix(path, {"a": True}, tier="x")

    assert prefix.observation_ids == tuple(by_path.loc[by_path["tier"] == "x", "observation_id"])
    assert prefix.observation_ids == ("a", "b")
    assert (prefix.queue_rows, prefix.labelled) == (2, 1)
    # The hash is of the file as stored, compressed.
    assert prefix.queue_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


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


def _strata(unsafe: tuple[int, int], safe: tuple[int, int]) -> JudgeStrata:
    """Both judge strata as (counted, size): labels, or queue rows counted as labels."""

    return JudgeStrata(
        JudgeStratumCounts(labelled=unsafe[0], truly_unsafe=0, stratum_size=unsafe[1]),
        JudgeStratumCounts(labelled=safe[0], truly_unsafe=0, stratum_size=safe[1]),
    )


@pytest.mark.parametrize(
    ("labelled", "queued", "flagged"),
    [
        # A queue that lists every rollout of each labelled stratum leaves only its order.
        (_strata((5, 10), (5, 30)), _strata((10, 10), (30, 30)), []),
        # A stratum without labels is completed with certainty: its queue rows do not matter.
        (_strata((0, 10), (5, 30)), _strata((4, 10), (30, 30)), []),
        (_strata((5, 10), (5, 30)), _strata((10, 10), (29, 30)), ["judge_safe"]),
        (_strata((5, 10), (5, 30)), _strata((5, 10), (5, 30)), ["judge_unsafe", "judge_safe"]),
    ],
)
def test_only_labelled_strata_the_queue_lists_in_part_are_flagged(
    labelled: JudgeStrata, queued: JudgeStrata, flagged: list[str]
) -> None:
    found = _partial_queue_diagnostic(
        _queue_shares(labelled, queued), None, ("c", None), "cohorts/c"
    )
    details = {} if found is None else found.details
    names = [item["judge_stratum"] for item in details.get("judge_strata", [])]
    assert names == flagged


def _fine(**strata: tuple[int, int]) -> FineStrata:
    """Fine strata as (counted, size), each named by its judge stratum's initial and a digit."""

    fine: FineStrata = {}
    for name, (counted, size) in strata.items():
        judge = "judge_unsafe" if name.startswith("u") else "judge_safe"
        fine.setdefault(judge, {})[name] = JudgeStratumCounts(
            labelled=counted, truly_unsafe=0, stratum_size=size
        )
    return fine


@pytest.mark.parametrize(
    ("labelled", "queued", "flagged"),
    [
        # The judge-safe stratum is listed in part, but only in its unlabelled fine stratum,
        # which is completed with certainty.
        (
            _fine(u0=(3, 10), s0=(5, 20), s1=(0, 10)),
            _fine(u0=(10, 10), s0=(20, 20), s1=(4, 10)),
            [],
        ),
        (
            _fine(u0=(3, 10), s0=(5, 20), s1=(1, 10)),
            _fine(u0=(10, 10), s0=(20, 20), s1=(4, 10)),
            [("judge_safe", "s1")],
        ),
        (
            _fine(u0=(3, 10), u1=(2, 5), s0=(5, 20)),
            _fine(u0=(9, 10), u1=(5, 5), s0=(19, 20)),
            [("judge_unsafe", "u0"), ("judge_safe", "s0")],
        ),
    ],
)
def test_in_a_stratified_design_labelled_fine_strata_listed_in_part_are_flagged(
    labelled: FineStrata, queued: FineStrata, flagged: list[tuple[str, str]]
) -> None:
    found = _partial_queue_diagnostic(
        _fine_queue_shares(labelled, queued), None, ("c", None), "cohorts/c"
    )
    details = {} if found is None else found.details
    assert "judge_strata" not in details
    names = [(item["judge_stratum"], item["stratum"]) for item in details.get("fine_strata", [])]
    assert names == flagged


def test_the_fine_strata_warning_names_one_stratum_or_counts_several() -> None:
    labelled = _fine(u0=(3, 10), u1=(2, 5), s0=(5, 20))
    one = _partial_queue_diagnostic(
        _fine_queue_shares(labelled, _fine(u0=(10, 10), u1=(5, 5), s0=(19, 20))),
        "calibration",
        ("c", "g"),
        "cohorts/c/groups/g",
    )
    several = _partial_queue_diagnostic(
        _fine_queue_shares(labelled, _fine(u0=(9, 10), u1=(4, 5), s0=(19, 20))),
        None,
        ("c", None),
        "cohorts/c",
    )

    assert one is not None and several is not None
    caveat = (
        "; the labels are a random sample of such a stratum only if the queue's rows of it "
        "were drawn from all of its rollouts at random before labelling, which the analysis "
        "cannot check"
    )
    assert one.message == (
        "the labelling queue tier 'calibration' lists 19 of the 20 classified rollouts of "
        "declared stratum 's0' in the judge-safe stratum of group 'g' in cohort 'c'" + caveat
    )
    assert one.details == {
        "fine_strata": [
            {
                "judge_stratum": "judge_safe",
                "stratum": "s0",
                "queue_rows": 19,
                "labelled": 5,
                "stratum_size": 20,
            }
        ]
    }
    assert several.message == (
        "the labelling queue lists part of 3 labelled declared strata of cohort 'c', such as "
        "9 of the 10 classified rollouts of declared stratum 'u0' in the judge-unsafe stratum"
        + caveat
    )
