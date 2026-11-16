from __future__ import annotations

import pandas as pd
import pytest

from inspect_safety_stats.ingest.pilot import draw_pilot


def _manifest(sizes: dict[str, int]) -> pd.DataFrame:
    rows = [
        {
            "prompt_id": f"{category}-{index:03d}",
            "question_id": f"{category}-q{index:03d}",
            "task_id": "task",
            "category": category,
        }
        for category, size in sizes.items()
        for index in range(size)
    ]
    return pd.DataFrame(rows)


def _drawn(frame: pd.DataFrame) -> set[str]:
    return set(frame.loc[frame["pilot_selected"], "prompt_id"])


def test_draws_the_requested_count_from_every_stratum() -> None:
    manifest = _manifest({"a": 10, "b": 4, "c": 25})
    frame = draw_pilot(manifest, per_stratum=3, strata=["category"], seed=7)
    counts = frame.loc[frame["pilot_selected"]].groupby("category").size().to_dict()
    assert counts == {"a": 3, "b": 3, "c": 3}
    assert frame["prompt_id"].tolist() == manifest["prompt_id"].tolist()


def test_inclusion_probability_is_draws_over_stratum_size() -> None:
    """These are the values a fixed-size sampled design checks: constant within a stratum."""
    frame = draw_pilot(_manifest({"a": 10, "b": 4}), per_stratum=2, strata=["category"], seed=1)
    probabilities = frame.groupby("pilot_stratum")["pilot_inclusion_probability"].unique()
    assert probabilities["a"].tolist() == [pytest.approx(0.2, abs=0)]
    assert probabilities["b"].tolist() == [pytest.approx(0.5, abs=0)]


def test_without_strata_the_manifest_is_one_stratum() -> None:
    frame = draw_pilot(_manifest({"a": 10, "b": 10}), per_stratum=5, strata=[], seed=3)
    assert frame["pilot_selected"].sum() == 5
    assert set(frame["pilot_stratum"]) == {"all"}
    assert set(frame["pilot_inclusion_probability"]) == {0.25}


def test_strata_combine_several_columns() -> None:
    manifest = _manifest({"a": 6, "b": 6})
    manifest["task_id"] = ["t1", "t2"] * 6
    frame = draw_pilot(manifest, per_stratum=2, strata=["task_id", "category"], seed=3)
    assert set(frame["pilot_stratum"]) == {"t1/a", "t2/a", "t1/b", "t2/b"}
    assert frame.groupby("pilot_stratum")["pilot_selected"].sum().tolist() == [2, 2, 2, 2]


def test_the_draw_ignores_row_order() -> None:
    manifest = _manifest({"a": 30, "b": 30})
    shuffled = manifest.sample(frac=1.0, random_state=11).reset_index(drop=True)
    first = draw_pilot(manifest, per_stratum=4, strata=["category"], seed=5)
    second = draw_pilot(shuffled, per_stratum=4, strata=["category"], seed=5)
    assert _drawn(first) == _drawn(second)


def test_a_larger_pilot_extends_a_smaller_one() -> None:
    manifest = _manifest({"a": 30, "b": 30})
    small = draw_pilot(manifest, per_stratum=3, strata=["category"], seed=5)
    large = draw_pilot(manifest, per_stratum=8, strata=["category"], seed=5)
    assert _drawn(small) < _drawn(large)


def test_the_seed_changes_the_draw() -> None:
    manifest = _manifest({"a": 30})
    draws = {
        frozenset(_drawn(draw_pilot(manifest, per_stratum=4, strata=[], seed=seed)))
        for seed in range(5)
    }
    assert len(draws) == 5


def test_the_draw_is_not_the_first_prompts() -> None:
    """The bias this replaces: a pilot of the first prompts by identifier."""
    manifest = _manifest({"a": 100})
    frame = draw_pilot(manifest, per_stratum=8, strata=["category"], seed=20261029)
    first = set(manifest["prompt_id"].sort_values().head(8))
    assert _drawn(frame) != first


def test_every_prompt_is_drawn_about_equally_often() -> None:
    """Over many seeds each prompt's selection rate approaches draws over stratum size.

    With 2,000 seeds and p = 0.2, a count's standard deviation is about 18; the band is
    more than five of them either side of the expectation of 400.
    """
    manifest = _manifest({"a": 10})
    counts = pd.Series(0, index=manifest["prompt_id"])
    for seed in range(2000):
        counts[sorted(_drawn(draw_pilot(manifest, per_stratum=2, strata=[], seed=seed)))] += 1
    assert counts.between(300, 500).all(), counts.to_dict()


def test_a_stratum_smaller_than_the_draw_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"1 strata hold fewer than 5 prompts: b \(4\)"):
        draw_pilot(_manifest({"a": 10, "b": 4}), per_stratum=5, strata=["category"], seed=1)


def test_missing_strata_values_are_rejected() -> None:
    manifest = _manifest({"a": 4, "b": 4})
    manifest.loc[2, "category"] = None
    with pytest.raises(ValueError, match="missing values"):
        draw_pilot(manifest, per_stratum=1, strata=["category"], seed=1)
    manifest.loc[2, "category"] = " "
    with pytest.raises(ValueError, match="missing values"):
        draw_pilot(manifest, per_stratum=1, strata=["category"], seed=1)


def test_strata_labels_that_would_merge_are_rejected() -> None:
    manifest = pd.DataFrame(
        {
            "prompt_id": ["p1", "p2"],
            "first": ["a/b", "a"],
            "second": ["c", "b/c"],
        }
    )
    with pytest.raises(ValueError, match="share a label"):
        draw_pilot(manifest, per_stratum=1, strata=["first", "second"], seed=1)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"per_stratum": 0}, "positive integer"),
        ({"per_stratum": True}, "positive integer"),
        ({"strata": ["absent"]}, "not in the design manifest"),
        ({"strata": ["category", "category"]}, "distinct"),
        ({"seed": True}, "seed must be an integer"),
    ],
)
def test_invalid_arguments_are_rejected(arguments: dict[str, object], message: str) -> None:
    options: dict[str, object] = {"per_stratum": 1, "strata": ["category"], "seed": 1}
    options.update(arguments)
    with pytest.raises(ValueError, match=message):
        draw_pilot(_manifest({"a": 3}), **options)  # type: ignore[arg-type]


def test_a_pilot_frame_cannot_be_drawn_from_again() -> None:
    frame = draw_pilot(_manifest({"a": 3}), per_stratum=1, strata=["category"], seed=1)
    with pytest.raises(ValueError, match="pilot_selected"):
        draw_pilot(frame, per_stratum=1, strata=["category"], seed=1)


def test_duplicate_prompts_are_rejected() -> None:
    manifest = _manifest({"a": 3})
    manifest.loc[1, "prompt_id"] = manifest.loc[0, "prompt_id"]
    with pytest.raises(ValueError, match="duplicate"):
        draw_pilot(manifest, per_stratum=1, strata=["category"], seed=1)
