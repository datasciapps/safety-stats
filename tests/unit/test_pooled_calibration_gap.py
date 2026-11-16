"""Which rows a cohort's pooled judge labels can certify (#95).

The labels weight every classified rollout of the cohort equally, so they certify only a
weighted mean, in a fixed design, that does the same.
"""

from __future__ import annotations

from inspect_safety_stats.analysis.judge_calibration import _pooled_calibration_gap, _PromptRollouts


def _gap(
    population: dict[str, _PromptRollouts],
    *,
    weights: dict[str, float],
    design_type: str = "fixed",
    measure_type: str = "weighted_mean",
) -> str | None:
    return _pooled_calibration_gap(
        population, design_type=design_type, measure_type=measure_type, weights=weights
    )


EVEN = {"p1": _PromptRollouts(200, 200, 200), "p2": _PromptRollouts(200, 200, 200)}
HALVES = {"p1": 0.5, "p2": 0.5}


def test_a_whole_cohort_mean_that_weights_rollouts_equally_is_certified() -> None:
    assert _gap(EVEN, weights=HALVES) is None


def test_unequal_prompt_weights_are_not() -> None:
    assert _gap(EVEN, weights={"p1": 0.9, "p2": 0.1}) == (
        "the row gives some of the cohort's rollouts 9 times the weight of others"
    )


def test_unequal_rollout_counts_are_not() -> None:
    population = {"p1": _PromptRollouts(200, 200, 200), "p2": _PromptRollouts(100, 100, 100)}
    assert _gap(population, weights=HALVES) == (
        "the row gives some of the cohort's rollouts 2 times the weight of others"
    )


def test_a_row_over_part_of_the_cohort_is_not() -> None:
    assert _gap(EVEN, weights={"p1": 1.0}) == (
        "the row covers only part of the cohort's labelled rollouts"
    )


def test_rollouts_a_policy_excludes_still_weigh_in_a_direct_mean() -> None:
    # Prompt bounds drop the ten excluded rollouts of p2, but direct-mean intervals divide by
    # them, so there p2's classified rollouts weigh less than p1's.
    population = {"p1": _PromptRollouts(200, 200, 200), "p2": _PromptRollouts(200, 200, 210)}
    gap = _gap(population, weights=HALVES)
    assert gap is not None and "1.05 times" in gap


def test_unknown_outcomes_kept_in_the_bounds_weigh_alike_in_both() -> None:
    # p2's ten unclassified rollouts stay in both denominators, so every classified rollout
    # still weighs 0.5 / 200.
    population = {"p1": _PromptRollouts(200, 200, 200), "p2": _PromptRollouts(190, 200, 200)}
    assert _gap(population, weights=HALVES) is None


def test_a_prompt_without_classified_rollouts_is_outside_the_labelled_population() -> None:
    population = {**EVEN, "p3": _PromptRollouts(0, 50, 50)}
    assert _gap(population, weights={"p1": 0.4, "p2": 0.4, "p3": 0.2}) is None


def test_tail_measures_and_sampled_designs_are_not_certified() -> None:
    maximum = _gap(EVEN, weights=HALVES, measure_type="maximum")
    assert maximum is not None and maximum.startswith("a maximum measure concentrates")
    sampled = _gap(EVEN, weights=HALVES, design_type="sampled")
    assert sampled is not None and sampled.startswith("a sampled design estimates a frame mean")


def test_a_cohort_without_classified_rollouts_has_nothing_to_certify() -> None:
    assert _gap({"p1": _PromptRollouts(0, 20, 20)}, weights={"p1": 1.0}) == (
        "the cohort has no classified rollouts"
    )
