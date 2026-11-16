"""The judge flip rate under content-preserving wrappers (#122), on a fake judge.

The fake judge is talked out of each unsafe verdict, under each wrapper independently, with a
known probability, so the true flip rates are known in closed form.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from inspect_safety_stats.config.models import JudgeFlipSpec
from inspect_safety_stats.ingest.judge_flip import load_judge_flip_manifest
from inspect_safety_stats.stats.judge_flip import judge_flip_rate

WRAPPERS = ("refusal_prefix", "benign_padding", "educational", "story")


def fake_judge(
    observation_ids: list[str], flip_probability: float, rng: np.random.Generator
) -> dict[str, dict[str, bool]]:
    """Rescored verdicts from a judge that flips each wrapper with ``flip_probability``."""

    return {
        key: {wrapper: bool(rng.random() >= flip_probability) for wrapper in WRAPPERS}
        for key in observation_ids
    }


def test_counts_any_wrapper_and_each_wrapper() -> None:
    verdicts = {
        "a": dict.fromkeys(WRAPPERS, True),
        "b": {**dict.fromkeys(WRAPPERS, True), "story": False},
        "c": {**dict.fromkeys(WRAPPERS, False)},
        "d": dict.fromkeys(WRAPPERS, True),
    }

    result = judge_flip_rate(["a", "b", "c", "d"], verdicts, WRAPPERS, sampling="all", alpha=0.05)

    assert result.judge_positive == 4
    assert result.any_wrapper is not None
    assert (result.any_wrapper.flipped, result.any_wrapper.rescored) == (2, 4)
    assert result.any_wrapper.estimate == 0.5
    assert result.by_wrapper["story"].flipped == 2
    assert result.by_wrapper["refusal_prefix"].flipped == 1
    assert list(result.by_wrapper) == list(WRAPPERS)


def test_the_interval_is_the_two_sided_clopper_pearson_interval() -> None:
    from scipy.stats import beta

    verdicts = {f"o{i}": dict.fromkeys(WRAPPERS, i >= 3) for i in range(20)}
    result = judge_flip_rate(list(verdicts), verdicts, WRAPPERS, sampling="all", alpha=0.10)

    rate = result.any_wrapper
    assert rate is not None
    assert (rate.flipped, rate.rescored) == (3, 20)
    assert rate.lower == pytest.approx(beta.ppf(0.05, 3, 18), rel=1e-12, abs=0)
    assert rate.upper == pytest.approx(beta.ppf(0.95, 4, 17), rel=1e-12, abs=0)


def test_no_flip_has_a_zero_lower_bound_and_all_flips_a_unit_upper_bound() -> None:
    none = {f"o{i}": dict.fromkeys(WRAPPERS, True) for i in range(10)}
    every = {f"o{i}": dict.fromkeys(WRAPPERS, False) for i in range(10)}

    low = judge_flip_rate(list(none), none, WRAPPERS, sampling="all", alpha=0.05).any_wrapper
    high = judge_flip_rate(list(every), every, WRAPPERS, sampling="all", alpha=0.05).any_wrapper

    assert low is not None and high is not None
    assert (low.estimate, low.lower) == (0.0, 0.0)
    assert low.upper == pytest.approx(1 - 0.025 ** (1 / 10), rel=1e-12, abs=0)
    assert (high.estimate, high.upper) == (1.0, 1.0)


@pytest.mark.parametrize("flip_probability", [0.02, 0.2, 0.6])
def test_the_intervals_cover_the_fake_judges_known_rates(flip_probability: float) -> None:
    rng = np.random.default_rng(122)
    alpha = 0.10
    replicates = 2000
    any_true = -math.expm1(len(WRAPPERS) * math.log1p(-flip_probability))
    covered_any = 0
    covered_wrapper = 0
    for _ in range(replicates):
        ids = [f"o{i}" for i in range(30)]
        result = judge_flip_rate(
            ids, fake_judge(ids, flip_probability, rng), WRAPPERS, sampling="all", alpha=alpha
        )
        assert result.any_wrapper is not None
        covered_any += result.any_wrapper.lower <= any_true <= result.any_wrapper.upper
        story = result.by_wrapper["story"]
        covered_wrapper += story.lower <= flip_probability <= story.upper
    # Clopper-Pearson is conservative, so coverage is at least 1 - alpha; the slack is three
    # Monte Carlo standard errors.
    slack = 3 * math.sqrt(alpha * (1 - alpha) / replicates)
    assert covered_any / replicates >= 1 - alpha - slack
    assert covered_wrapper / replicates >= 1 - alpha - slack


def test_a_large_sample_recovers_the_fake_judges_rates() -> None:
    rng = np.random.default_rng(7)
    ids = [f"o{i}" for i in range(20000)]
    result = judge_flip_rate(
        ids, fake_judge(ids, 0.1, rng), WRAPPERS, sampling="all", alpha=0.05
    )

    assert result.any_wrapper is not None
    assert result.any_wrapper.estimate == pytest.approx(1 - 0.9**4, abs=0.01)
    for rate in result.by_wrapper.values():
        assert rate.estimate == pytest.approx(0.1, abs=0.01)


def test_a_simple_random_sample_need_not_cover_every_positive() -> None:
    verdicts = {"a": dict.fromkeys(WRAPPERS, False)}

    result = judge_flip_rate(
        ["a", "b", "c"], verdicts, WRAPPERS, sampling="simple_random", alpha=0.05
    )

    assert result.judge_positive == 3
    assert result.any_wrapper is not None
    assert result.any_wrapper.rescored == 1


def test_a_run_without_judge_positives_has_no_rate() -> None:
    result = judge_flip_rate([], {}, WRAPPERS, sampling="all", alpha=0.05)

    assert result.any_wrapper is None
    assert result.by_wrapper == {}


@pytest.mark.parametrize(
    ("positive", "verdicts", "sampling", "message"),
    [
        (["a"], {"z": dict.fromkeys(WRAPPERS, True)}, "simple_random", "not judge-positive"),
        (["a", "b"], {"a": dict.fromkeys(WRAPPERS, True)}, "all", "were not rescored"),
        (["a"], {}, "simple_random", "rescores none"),
        (["a"], {"a": {"story": True}}, "all", "every rescored completion"),
        (
            ["a"],
            {"a": {**dict.fromkeys(WRAPPERS, True), "other": True}},
            "all",
            "undeclared wrappers",
        ),
    ],
)
def test_a_manifest_that_breaks_the_design_fails_loudly(
    positive: list[str],
    verdicts: dict[str, dict[str, bool]],
    sampling: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        judge_flip_rate(positive, verdicts, WRAPPERS, sampling=sampling, alpha=0.05)  # type: ignore[arg-type]


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "flips.csv"
    path.write_text(body, encoding="utf-8")
    return path


def test_manifest_round_trip(tmp_path: Path) -> None:
    verdicts = load_judge_flip_manifest(
        _write(tmp_path, "observation_id,wrapper,verdict\na,story,SAFE\na,educational,unsafe\n")
    )

    assert verdicts == {"a": {"story": False, "educational": True}}


def test_manifest_with_a_header_only_reads_as_empty(tmp_path: Path) -> None:
    assert load_judge_flip_manifest(_write(tmp_path, "observation_id,wrapper,verdict\n")) == {}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("observation_id,wrapper\na,story\n", "missing required columns"),
        ("observation_id,wrapper,verdict\na,story,unclear\n", "invalid verdict"),
        ("observation_id,wrapper,verdict\na,story,safe\na,story,unsafe\n", "more than one row"),
        ("observation_id,wrapper,verdict\n,story,safe\n", "blank observation_id"),
        ("observation_id,wrapper,verdict\na,,safe\n", "blank wrapper"),
    ],
)
def test_manifest_rejects_malformed_rows(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_judge_flip_manifest(_write(tmp_path, body))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"wrappers": []}, "at least 1"),
        ({"wrappers": ["story", "story"]}, "unique"),
        ({"wrappers": [" "]}, "non-empty"),
        ({"warn_above": 0}, "greater than 0"),
        ({"warn_above": 1}, "less than 1"),
        ({"warn_above": True}, "not a boolean"),
        ({"alpha": 1.5}, "less than 1"),
        ({"sampling": "stratified"}, "sampling"),
    ],
)
def test_the_plan_section_rejects_invalid_values(fields: dict[str, object], message: str) -> None:
    base: dict[str, object] = {
        "manifest": "flips.csv",
        "wrappers": list(WRAPPERS),
        "warn_above": 0.1,
    }
    with pytest.raises(ValidationError, match=message):
        JudgeFlipSpec.model_validate({**base, **fields})
