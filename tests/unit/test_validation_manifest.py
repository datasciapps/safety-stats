from pathlib import Path

import pytest

from inspect_safety_stats.ingest.validation import (
    judge_stratum_counts,
    load_validation_manifest,
)
from inspect_safety_stats.schema.observations import Observation


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text(body, encoding="utf-8")
    return path


def _observation(observation_id: str, cohort: str, failure: bool | None) -> Observation:
    return Observation(
        observation_id=observation_id,
        source="log.eval",
        run_key="e1",
        rollout_id=observation_id,
        item_id=observation_id,
        repeat=1,
        prompt_id="p1",
        question_id="q1",
        task_id="t1",
        cohort_id=cohort,
        score_name="s",
        failure=failure,
        model="m",
        configuration_fingerprint="f",
        collection_order=1,
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    labels = load_validation_manifest(
        _write(tmp_path, "observation_id,human_label\na,unsafe\nb,SAFE\n")
    )
    assert labels == {"a": True, "b": False}


def test_manifest_rejects_ambiguous_or_malformed_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid human_label"):
        load_validation_manifest(
            _write(tmp_path, "observation_id,human_label\na,unclear\n")
        )
    with pytest.raises(ValueError, match="duplicate observation_id"):
        load_validation_manifest(
            _write(tmp_path, "observation_id,human_label\na,safe\na,unsafe\n")
        )
    with pytest.raises(ValueError, match="missing required columns"):
        load_validation_manifest(_write(tmp_path, "observation_id\na\n"))
    with pytest.raises(ValueError, match="no rows"):
        load_validation_manifest(_write(tmp_path, "observation_id,human_label\n"))


def test_strata_are_built_per_cohort_and_exclude_unclassified() -> None:
    observations = [
        _observation("a", "candidate", True),
        _observation("b", "candidate", True),
        _observation("c", "candidate", False),
        _observation("d", "candidate", None),
        _observation("e", "baseline", False),
    ]
    counts, unmatched = judge_stratum_counts(
        observations, {"a": True, "c": False, "d": True, "zz": False}
    )

    # The unclassified row and the unknown identifier are both reported, not absorbed.
    assert unmatched == ["d", "zz"]

    candidate = counts["candidate"]
    assert candidate["judge_unsafe"].stratum_size == 2
    assert candidate["judge_unsafe"].labelled == 1
    assert candidate["judge_unsafe"].truly_unsafe == 1
    assert candidate["judge_safe"].stratum_size == 1
    assert candidate["judge_safe"].labelled == 1
    assert candidate["judge_safe"].truly_unsafe == 0

    # A cohort with no judge-unsafe observations yields only the safe stratum.
    assert set(counts["baseline"]) == {"judge_safe"}
