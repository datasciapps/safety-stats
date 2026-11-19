import itertools
import math
import random
from statistics import NormalDist

import numpy as np
import pytest

from inspect_safety_stats.stats import (
    exact_mcnemar,
    exact_mcnemar_partial,
    newcombe_paired_interval,
    newcombe_paired_interval_partial,
)
from inspect_safety_stats.stats.paired import (
    _newcombe_bounds_from_counts,
    _reachable_cell_counts,
    partial_pairing,
)


def test_exact_mcnemar_counts_and_direction() -> None:
    candidate = [True, True, False, False, True]
    baseline = [False, True, True, False, False]
    result = exact_mcnemar(candidate, baseline)
    assert result.candidate_failure_baseline_safe == 2
    assert result.candidate_safe_baseline_failure == 1
    assert result.discordant == 3
    assert result.estimate == pytest.approx(0.2)
    assert 0 <= result.p_value_two_sided <= 1


def test_exact_mcnemar_requires_pairs() -> None:
    with pytest.raises(ValueError):
        exact_mcnemar([True], [])


def test_newcombe_paired_interval_contains_estimate() -> None:
    candidate = [True, True, False, False, True, False, False, True]
    baseline = [False, True, True, False, False, False, False, True]
    result = newcombe_paired_interval(candidate, baseline)
    assert result.estimate == pytest.approx(0.125)
    assert -1 <= result.lower <= result.estimate <= result.upper <= 1


def test_newcombe_paired_interval_handles_boundary_rates() -> None:
    result = newcombe_paired_interval([False] * 10, [False] * 10)
    assert result.estimate == 0
    assert result.lower <= 0 <= result.upper


def test_newcombe_method_10_matches_published_table() -> None:
    candidate = [True] * 20 + [True] * 12 + [False] * 2 + [False] * 16
    baseline = [True] * 20 + [False] * 12 + [True] * 2 + [False] * 16
    result = newcombe_paired_interval(
        candidate,
        baseline,
        alpha_lower=0.025,
        alpha_upper=0.025,
    )
    assert result.estimate == pytest.approx(0.2)
    assert result.lower == pytest.approx(0.0562, abs=5e-5)
    assert result.upper == pytest.approx(0.3292, abs=5e-5)


def _completions(
    candidate: list[bool | None], baseline: list[bool | None]
) -> list[tuple[list[bool], list[bool]]]:
    unknown = [("c", i) for i, v in enumerate(candidate) if v is None] + [
        ("b", i) for i, v in enumerate(baseline) if v is None
    ]
    completions = []
    for values in itertools.product([False, True], repeat=len(unknown)):
        cand = list(candidate)
        base = list(baseline)
        for (arm, index), value in zip(unknown, values, strict=True):
            (cand if arm == "c" else base)[index] = value
        completions.append(([bool(v) for v in cand], [bool(v) for v in base]))
    return completions


def _random_partial_pairs(seed: int) -> tuple[list[bool | None], list[bool | None]]:
    rng = random.Random(seed)
    size = rng.randint(1, 9)
    choices: list[bool | None] = [True, False, None]
    weights = [0.35, 0.45, 0.2]
    candidate = rng.choices(choices, weights=weights, k=size)
    baseline = rng.choices(choices, weights=weights, k=size)
    return candidate, baseline


def test_partial_pairing_counts_the_unscored_arms() -> None:
    pairing = partial_pairing([True, None, False, None], [None, False, True, None])
    assert pairing.pair_count == 4
    assert pairing.scored_pair_count == 1
    assert pairing.candidate_unscored_pair_count == 1
    assert pairing.baseline_unscored_pair_count == 1
    assert pairing.both_unscored_pair_count == 1


def test_partial_methods_reduce_to_the_complete_ones_without_unscored_pairs() -> None:
    candidate = [True, True, False, False, True, False, False, True]
    baseline = [False, True, True, False, False, False, False, True]
    complete_test = exact_mcnemar(candidate, baseline)
    partial_test = exact_mcnemar_partial(list(candidate), list(baseline))
    assert partial_test.p_value_candidate_worse == complete_test.p_value_candidate_worse
    assert partial_test.p_value_candidate_better == complete_test.p_value_candidate_better
    assert partial_test.p_value_two_sided == pytest.approx(
        complete_test.p_value_two_sided, rel=1e-12, abs=0
    )
    assert partial_test.estimate_lower == partial_test.estimate_upper == complete_test.estimate
    complete_interval = newcombe_paired_interval(candidate, baseline)
    partial_interval = newcombe_paired_interval_partial(list(candidate), list(baseline))
    assert partial_interval.completion_count == 1
    assert partial_interval.lower == pytest.approx(complete_interval.lower, rel=1e-12, abs=1e-15)
    assert partial_interval.upper == pytest.approx(complete_interval.upper, rel=1e-12, abs=1e-15)


def test_vectorised_newcombe_matches_the_scalar_interval() -> None:
    rng = random.Random(3)
    for _ in range(300):
        cells = [rng.randint(0, 12) for _ in range(4)]
        if sum(cells) == 0:
            continue
        n11, n10, n01, n00 = cells
        candidate = [True] * n11 + [True] * n10 + [False] * n01 + [False] * n00
        baseline = [True] * n11 + [False] * n10 + [True] * n01 + [False] * n00
        alpha_lower, alpha_upper = rng.choice([(0.05, 0.05), (0.01, 0.04), (0.025, 0.025)])
        scalar = newcombe_paired_interval(
            candidate, baseline, alpha_lower=alpha_lower, alpha_upper=alpha_upper
        )
        estimate, lower, upper = _newcombe_bounds_from_counts(
            *(np.asarray([value]) for value in cells),
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
        )
        assert float(estimate[0]) == pytest.approx(scalar.estimate, rel=1e-12, abs=1e-15)
        assert float(lower[0]) == pytest.approx(scalar.lower, rel=1e-12, abs=1e-15)
        assert float(upper[0]) == pytest.approx(scalar.upper, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("seed", range(60))
def test_reachable_cell_counts_are_exactly_the_completions(seed: int) -> None:
    candidate, baseline = _random_partial_pairs(seed)
    expected = set()
    for cand, base in _completions(candidate, baseline):
        result = exact_mcnemar(cand, base)
        expected.add(
            (
                result.candidate_failure_baseline_failure,
                result.candidate_failure_baseline_safe,
                result.candidate_safe_baseline_failure,
                result.candidate_safe_baseline_safe,
            )
        )
    n11, n10, n01, n00 = _reachable_cell_counts(candidate, baseline)
    reachable = set(zip(n11.tolist(), n10.tolist(), n01.tolist(), n00.tolist(), strict=True))
    assert reachable == expected


@pytest.mark.parametrize("seed", range(60))
def test_partial_bounds_hold_for_every_completion(seed: int) -> None:
    candidate, baseline = _random_partial_pairs(seed)
    test = exact_mcnemar_partial(candidate, baseline)
    interval = newcombe_paired_interval_partial(candidate, baseline)
    for cand, base in _completions(candidate, baseline):
        completed_test = exact_mcnemar(cand, base)
        assert test.estimate_lower <= completed_test.estimate <= test.estimate_upper
        assert completed_test.p_value_candidate_worse <= test.p_value_candidate_worse
        assert completed_test.p_value_candidate_better <= test.p_value_candidate_better
        assert completed_test.p_value_two_sided <= test.p_value_two_sided + 1e-12
        completed_interval = newcombe_paired_interval(cand, base)
        assert interval.lower <= completed_interval.lower
        assert completed_interval.upper <= interval.upper
        assert interval.estimate_lower <= completed_interval.estimate <= interval.estimate_upper


def test_partial_bounds_are_attained_by_some_completion() -> None:
    candidate: list[bool | None] = [True, None, False, False, True, None, False, True]
    baseline: list[bool | None] = [False, True, None, False, False, None, False, True]
    intervals = [newcombe_paired_interval(c, b) for c, b in _completions(candidate, baseline)]
    interval = newcombe_paired_interval_partial(candidate, baseline)
    assert interval.lower == pytest.approx(min(i.lower for i in intervals), rel=1e-12, abs=1e-15)
    assert interval.upper == pytest.approx(max(i.upper for i in intervals), rel=1e-12, abs=1e-15)


def test_partial_outcomes_must_be_booleans_or_none() -> None:
    with pytest.raises(TypeError):
        exact_mcnemar_partial([1], [None])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        newcombe_paired_interval_partial([], [])


def _reference_newcombe(
    n11: int, n10: int, n01: int, n00: int, *, alpha_lower: float, alpha_upper: float
) -> tuple[float, float]:
    """Newcombe method 10 from the four cells, with the correlation's integers kept exact.

    Independent of the package: Python integers for the products of counts, ``math.sqrt`` for
    their square root and ``statistics.NormalDist`` for the quantiles.
    """

    n = n11 + n10 + n01 + n00
    candidate_rate = (n11 + n10) / n
    baseline_rate = (n11 + n01) / n
    estimate = candidate_rate - baseline_rate
    numerator = n11 * n00 - n10 * n01
    corrected = max(numerator - n / 2, 0.0) if numerator > 0 else float(numerator)
    product = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    correlation = min(1.0, max(-1.0, corrected / math.sqrt(product))) if product else 0.0

    def wilson(rate: float, z: float) -> tuple[float, float]:
        scale = 1 + z * z / n
        centre = (rate + z * z / (2 * n)) / scale
        radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / scale
        return max(0.0, centre - radius), min(1.0, centre + radius)

    def distance(first: float, second: float) -> float:
        return math.sqrt(max(0.0, first**2 + second**2 - 2 * correlation * first * second))

    z = NormalDist().inv_cdf(1 - alpha_lower)
    candidate_low, _ = wilson(candidate_rate, z)
    _, baseline_high = wilson(baseline_rate, z)
    lower = estimate - distance(candidate_rate - candidate_low, baseline_high - baseline_rate)
    z = NormalDist().inv_cdf(1 - alpha_upper)
    _, candidate_high = wilson(candidate_rate, z)
    baseline_low, _ = wilson(baseline_rate, z)
    upper = estimate + distance(candidate_high - candidate_rate, baseline_rate - baseline_low)
    return max(-1.0, lower), min(1.0, upper)


def _paired_lists(n11: int, n10: int, n01: int, n00: int) -> tuple[list[bool], list[bool]]:
    candidate = [True] * n11 + [True] * n10 + [False] * n01 + [False] * n00
    baseline = [True] * n11 + [False] * n10 + [True] * n01 + [False] * n00
    return candidate, baseline


# Cell counts (n11, n10, n01, n00) around the counts at which the product of the four margins,
# the correlation's denominator, outgrows a fixed-width integer. With all four margins m it is
# m ** 4: past 2 ** 63 from m = 55,109 (110,218 pairs) and past 2 ** 64 from m = 65,536.
_MARGIN_OVERFLOW_TABLES = {
    "below_2_63": (13_777, 41_331, 41_331, 13_777),
    "past_2_63": (13_777, 41_332, 41_332, 13_777),
    "below_2_64": (16_384, 49_151, 49_151, 16_384),
    "past_2_64": (16_384, 49_152, 49_152, 16_384),
    "past_5_times_2_64_positive": (60_000, 40_000, 40_000, 60_000),
}


@pytest.mark.parametrize(
    "cells", list(_MARGIN_OVERFLOW_TABLES.values()), ids=list(_MARGIN_OVERFLOW_TABLES)
)
def test_newcombe_keeps_the_correlation_when_the_margin_product_outgrows_integers(
    cells: tuple[int, int, int, int],
) -> None:
    # The int64 product wrapped: past 2 ** 63 its square root was NaN or 0, so the correlation
    # became 0, and past 2 ** 64 + 2 ** 63 it was a wrong positive number. As an integer past
    # 2 ** 64 the scalar interval's product was an object NumPy could not take a root of.
    expected_lower, expected_upper = _reference_newcombe(
        *cells, alpha_lower=0.01, alpha_upper=0.04
    )
    scalar = newcombe_paired_interval(*_paired_lists(*cells), alpha_lower=0.01, alpha_upper=0.04)
    assert scalar.lower == pytest.approx(expected_lower, rel=1e-12, abs=0)
    assert scalar.upper == pytest.approx(expected_upper, rel=1e-12, abs=0)
    _, lower, upper = _newcombe_bounds_from_counts(
        *(np.asarray([value], dtype=np.int64) for value in cells),
        alpha_lower=0.01,
        alpha_upper=0.04,
    )
    assert float(lower[0]) == pytest.approx(expected_lower, rel=1e-12, abs=0)
    assert float(upper[0]) == pytest.approx(expected_upper, rel=1e-12, abs=0)


def test_partial_newcombe_bounds_hold_when_the_margin_product_outgrows_integers() -> None:
    # One candidate outcome unscored in a table whose margins all reach 2 ** 16; the true
    # correlation is -0.5, and the int64 product made it 0, which narrows the interval.
    complete_candidate, complete_baseline = _paired_lists(16_384, 49_152, 49_152, 16_384)
    candidate: list[bool | None] = list(complete_candidate)
    baseline: list[bool | None] = list(complete_baseline)
    candidate[-1] = None  # a pair whose baseline is safe
    expected = [
        _reference_newcombe(*cells, alpha_lower=0.05, alpha_upper=0.05)
        for cells in [(16_384, 49_152, 49_152, 16_384), (16_384, 49_153, 49_152, 16_383)]
    ]
    interval = newcombe_paired_interval_partial(candidate, baseline)
    assert interval.completion_count == 2
    assert interval.lower == pytest.approx(min(low for low, _ in expected), rel=1e-12, abs=0)
    assert interval.upper == pytest.approx(max(high for _, high in expected), rel=1e-12, abs=0)


@pytest.mark.parametrize(
    "cells",
    [(100, 400, 400, 100), (5_000, 20_000, 20_000, 5_000)],
    ids=["margins_500", "margins_25000"],
)
def test_vectorised_newcombe_does_not_depend_on_the_integer_type_of_the_counts(
    cells: tuple[int, int, int, int],
) -> None:
    # The completions enumerate the counts in NumPy's default integer, which is int32 on
    # Windows with NumPy 1. There the product of the four margins overflows once they reach
    # 216 each, and the Wilson term's 4 * trials ** 2 from 23,171 pairs.
    expected_lower, expected_upper = _reference_newcombe(
        *cells, alpha_lower=0.05, alpha_upper=0.05
    )
    for dtype in (np.int32, np.int64):
        _, lower, upper = _newcombe_bounds_from_counts(
            *(np.asarray([value], dtype=dtype) for value in cells),
            alpha_lower=0.05,
            alpha_upper=0.05,
        )
        assert float(lower[0]) == pytest.approx(expected_lower, rel=1e-12, abs=0), dtype
        assert float(upper[0]) == pytest.approx(expected_upper, rel=1e-12, abs=0), dtype
