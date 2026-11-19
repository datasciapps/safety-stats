import itertools
import random

import numpy as np
import pytest

from inspect_safety_stats.stats import (
    exact_mcnemar,
    exact_mcnemar_partial,
    newcombe_paired_interval,
    newcombe_paired_interval_partial,
    paired,
)
from inspect_safety_stats.stats.clusters import ClusterBootstrapResult, paired_cluster_bootstrap
from inspect_safety_stats.stats.paired import (
    _newcombe_bounds_from_counts,
    favourable_and_adverse_completions,
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
    tables = paired._completion_tables(candidate, baseline)
    reachable = [
        table
        for block in paired._table_blocks(tables)
        for table in zip(*(cells.tolist() for cells in block), strict=True)
    ]
    assert len(reachable) == len(set(reachable))
    assert set(reachable) == expected


def test_partial_newcombe_evaluates_its_tables_a_block_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 150 pairs unscored in both arms reach 585,276 tables. Evaluating them at once, after
    # filtering a dense grid of 151**3 candidates, took memory that grew with the cube of the
    # unscored pairs: gigabytes near the limit.
    evaluate = paired._newcombe_bounds_from_counts
    sizes: list[int] = []

    def recording(*cells: np.ndarray, **alphas: float) -> tuple[np.ndarray, ...]:
        sizes.append(len(cells[0]))
        return evaluate(*cells, **alphas)

    monkeypatch.setattr(paired, "_newcombe_bounds_from_counts", recording)
    candidate: list[bool | None] = [True, False] * 20 + [None] * 150
    baseline: list[bool | None] = [False, True] * 20 + [None] * 150
    interval = newcombe_paired_interval_partial(candidate, baseline)
    assert sum(sizes) == interval.completion_count == 585_276
    assert max(sizes) <= 1 << 18


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


@pytest.mark.parametrize("seed", range(20))
def test_extreme_completions_bound_the_cluster_bootstrap_of_every_completion(seed: int) -> None:
    # fixed_cluster_bootstrap takes its lower bound from the favourable completion and its upper
    # bound from the adverse one. Its resampling draws do not depend on the outcomes, and every
    # replicate mean grows with each pair's difference, so no completion goes beyond either.
    rng = random.Random(seed)
    size = rng.randint(2, 8)
    choices: list[bool | None] = [True, False, None]
    candidate = rng.choices(choices, weights=[0.35, 0.4, 0.25], k=size)
    baseline = rng.choices(choices, weights=[0.35, 0.4, 0.25], k=size)
    clusters = [index if index < 2 else rng.randint(0, 2) for index in range(size)]

    def bootstrap(cand: list[bool], base: list[bool]) -> ClusterBootstrapResult:
        differences = [float(c) - float(b) for c, b in zip(cand, base, strict=True)]
        return paired_cluster_bootstrap(differences, clusters, replicates=100, seed=seed)

    favourable, adverse = (
        bootstrap(*completion)
        for completion in favourable_and_adverse_completions(candidate, baseline)
    )
    for cand, base in _completions(candidate, baseline):
        completed = bootstrap(cand, base)
        assert favourable.estimate <= completed.estimate <= adverse.estimate
        # np.quantile interpolates, which can round by an ulp.
        assert favourable.lower <= completed.lower + 1e-12
        assert completed.upper <= adverse.upper + 1e-12


def test_partial_outcomes_must_be_booleans_or_none() -> None:
    with pytest.raises(TypeError):
        exact_mcnemar_partial([1], [None])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        newcombe_paired_interval_partial([], [])
