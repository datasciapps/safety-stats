import pytest

from inspect_safety_stats.stats import exact_mcnemar, newcombe_paired_interval


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
