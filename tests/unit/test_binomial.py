import math

import pytest
from scipy.stats import binom

from inspect_safety_stats.stats import (
    binomial_lower_bound,
    binomial_upper_bound,
    missingness_robust_binomial_bounds,
    zero_failure_sample_size,
)


def test_zero_failure_bound_has_closed_form() -> None:
    upper = binomial_upper_bound(0, 299, alpha=0.05)
    assert upper == pytest.approx(1 - 0.05 ** (1 / 299), abs=1e-14)
    assert upper <= 0.01


def test_zero_failure_planner_is_minimal() -> None:
    n = zero_failure_sample_size(threshold=0.01, alpha=0.05)
    assert n == 299
    assert 1 - 0.05 ** (1 / n) <= 0.01
    assert 1 - 0.05 ** (1 / (n - 1)) > 0.01


def test_boundaries() -> None:
    assert binomial_lower_bound(0, 10, alpha=0.05) == 0
    assert binomial_upper_bound(10, 10, alpha=0.05) == 1


def test_missingness_robust_bounds_cover_every_binary_completion() -> None:
    bounds = missingness_robust_binomial_bounds(
        failures=2,
        classified=7,
        total=10,
        alpha_lower=0.05,
        alpha_upper=0.05,
    )
    assert bounds.lower == binomial_lower_bound(2, 10, alpha=0.05)
    assert bounds.upper == binomial_upper_bound(5, 10, alpha=0.05)
    assert bounds.method == "clopper_pearson_one_sided_missingness_robust"
    for completed_failures in range(2, 6):
        assert bounds.lower <= binomial_lower_bound(completed_failures, 10, alpha=0.05)
        assert bounds.upper >= binomial_upper_bound(completed_failures, 10, alpha=0.05)


@pytest.mark.parametrize("failures,trials", [(-1, 10), (11, 10), (0, 0)])
def test_invalid_counts(failures: int, trials: int) -> None:
    with pytest.raises(ValueError):
        binomial_upper_bound(failures, trials, alpha=0.05)


def test_zero_failure_sample_size_uses_exact_not_rule_of_three() -> None:
    n = zero_failure_sample_size(threshold=0.1, alpha=0.05)
    assert n == math.ceil(math.log(0.05) / math.log(0.9))


@pytest.mark.parametrize("probability", [0.0, 0.01, 0.1, 0.5, 0.9, 1.0])
def test_one_sided_bounds_have_at_least_nominal_coverage(probability: float) -> None:
    alpha = 0.05
    for trials in range(1, 21):
        upper_coverage = sum(
            binom.pmf(failures, trials, probability)
            for failures in range(trials + 1)
            if binomial_upper_bound(failures, trials, alpha=alpha) >= probability
        )
        lower_coverage = sum(
            binom.pmf(failures, trials, probability)
            for failures in range(trials + 1)
            if binomial_lower_bound(failures, trials, alpha=alpha) <= probability
        )
        assert upper_coverage >= 1 - alpha - 1e-12
        assert lower_coverage >= 1 - alpha - 1e-12


def test_simultaneous_bounds_cover_fixed_design_risk() -> None:
    probabilities = (0.1, 0.3)
    trials = 5
    endpoint_alpha = 0.05 / len(probabilities)
    lower_coverage = 0.0
    upper_coverage = 0.0
    true_risk = sum(probabilities) / len(probabilities)
    for first in range(trials + 1):
        for second in range(trials + 1):
            mass = binom.pmf(first, trials, probabilities[0]) * binom.pmf(
                second, trials, probabilities[1]
            )
            lower = (
                sum(
                    binomial_lower_bound(count, trials, alpha=endpoint_alpha)
                    for count in (first, second)
                )
                / 2
            )
            upper = (
                sum(
                    binomial_upper_bound(count, trials, alpha=endpoint_alpha)
                    for count in (first, second)
                )
                / 2
            )
            if lower <= true_risk:
                lower_coverage += mass
            if true_risk <= upper:
                upper_coverage += mass
    assert lower_coverage >= 0.95 - 1e-12
    assert upper_coverage >= 0.95 - 1e-12
