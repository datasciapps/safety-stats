import itertools
import math

import numpy as np
import pytest
from scipy.stats import binom

from inspect_safety_stats.stats import (
    weighted_bounded_kl_chernoff_mean_bounds,
    weighted_empirical_bernstein_mean_bounds,
    weighted_hoeffding_mean_bounds,
    weighted_kl_chernoff_mean_bounds,
    worst_of_k_estimate,
    worst_of_k_kl_chernoff_bounds,
)


def test_weighted_hoeffding_matches_closed_form() -> None:
    result = weighted_hoeffding_mean_bounds(
        failures=[1, 3],
        unclassified=[0, 0],
        trials=[10, 20],
        weights=[0.25, 0.75],
        alpha_lower=0.025,
        alpha_upper=0.025,
    )
    concentration_variance = 0.25**2 / 10 + 0.75**2 / 20
    width = math.sqrt(0.5 * concentration_variance * math.log(1 / 0.025))
    estimate = 0.25 * 0.1 + 0.75 * 0.15
    assert result.completion_estimate_lower == pytest.approx(estimate)
    assert result.completion_estimate_upper == pytest.approx(estimate)
    assert result.lower == pytest.approx(max(0, estimate - width))
    assert result.upper == pytest.approx(min(1, estimate + width))
    assert result.effective_sample_size == pytest.approx(1 / concentration_variance)


def test_unclassified_outcomes_expand_completion_range() -> None:
    result = weighted_hoeffding_mean_bounds(
        failures=[1, 0],
        unclassified=[2, 1],
        trials=[10, 10],
        weights=[0.5, 0.5],
        alpha_lower=0.025,
        alpha_upper=0.025,
    )
    assert result.completion_estimate_lower == pytest.approx(0.05)
    assert result.completion_estimate_upper == pytest.approx(0.20)
    for first_missing_failures in range(3):
        for second_missing_failures in range(2):
            complete_estimate = 0.5 * (1 + first_missing_failures) / 10 + 0.5 * (
                second_missing_failures / 10
            )
            assert result.lower <= max(0, complete_estimate - result.half_width_lower)
            assert result.upper >= min(1, complete_estimate + result.half_width_upper)


def test_empirical_bernstein_is_variance_adaptive_at_zero_failures() -> None:
    empirical_bernstein = weighted_empirical_bernstein_mean_bounds(
        failures=[0, 0],
        unclassified=[0, 0],
        trials=[500, 500],
        weights=[0.5, 0.5],
        alpha_lower=0.025,
        alpha_upper=0.025,
    )
    hoeffding = weighted_hoeffding_mean_bounds(
        failures=[0, 0],
        unclassified=[0, 0],
        trials=[500, 500],
        weights=[0.5, 0.5],
        alpha_lower=0.025,
        alpha_upper=0.025,
    )
    assert empirical_bernstein.method == "weighted_empirical_bernstein"
    assert empirical_bernstein.variance_upper_upper is not None
    assert empirical_bernstein.variance_upper_upper < 0.25 * (
        empirical_bernstein.concentration_variance
    )
    assert empirical_bernstein.upper < hoeffding.upper


def test_cluster_sensitivity_treats_each_prompt_as_one_independent_unit() -> None:
    rollout = weighted_empirical_bernstein_mean_bounds(
        failures=[1, 2, 0],
        unclassified=[0, 0, 0],
        trials=[20, 20, 20],
        weights=[0.2, 0.3, 0.5],
        alpha_lower=0.025,
        alpha_upper=0.025,
    )
    cluster = weighted_empirical_bernstein_mean_bounds(
        failures=[1, 2, 0],
        unclassified=[0, 0, 0],
        trials=[20, 20, 20],
        weights=[0.2, 0.3, 0.5],
        alpha_lower=0.025,
        alpha_upper=0.025,
        independence_unit="prompt_cluster",
    )
    repeated_with_same_prompt_rates = weighted_empirical_bernstein_mean_bounds(
        failures=[10, 20, 0],
        unclassified=[0, 0, 0],
        trials=[200, 200, 200],
        weights=[0.2, 0.3, 0.5],
        alpha_lower=0.025,
        alpha_upper=0.025,
        independence_unit="prompt_cluster",
    )
    assert rollout.effective_sample_size == pytest.approx(20 / (0.2**2 + 0.3**2 + 0.5**2))
    assert cluster.effective_sample_size == pytest.approx(1 / (0.2**2 + 0.3**2 + 0.5**2))
    assert cluster.half_width_upper > rollout.half_width_upper
    assert repeated_with_same_prompt_rates.lower == pytest.approx(cluster.lower)
    assert repeated_with_same_prompt_rates.upper == pytest.approx(cluster.upper)


def test_empirical_bernstein_unclassified_completion_is_monotone() -> None:
    result = weighted_empirical_bernstein_mean_bounds(
        failures=[1, 0],
        unclassified=[2, 1],
        trials=[10, 10],
        weights=[0.5, 0.5],
        alpha_lower=0.05,
        alpha_upper=0.05,
    )
    assert result.completion_estimate_lower == pytest.approx(0.05)
    assert result.completion_estimate_upper == pytest.approx(0.20)
    assert result.lower <= result.completion_estimate_lower
    assert result.upper >= result.completion_estimate_upper


@pytest.mark.parametrize(
    ("failures", "unclassified", "true_labels"),
    [
        # Mostly failing mass; the unknown is truly safe, so all-failure is the adverse case.
        ([0, 550, 0], [1, 0, 0], [0, 550, 0]),
        # The mirror: mostly safe mass; the unknown truly failed, so all-safe is adverse.
        ([0, 0, 16_500], [1, 0, 0], [1, 0, 16_500]),
    ],
)
def test_empirical_bernstein_bounds_contain_the_bounds_of_the_true_labels(
    failures: list[int], unclassified: list[int], true_labels: list[int]
) -> None:
    """One heavy unknown among light rollouts, at a Bonferroni-sized endpoint alpha.

    Completing the unknown adversely also lowered the variance bound, by more than it moved
    the estimate. The robust upper endpoint fell 8.6e-4 below that of the true labels, and in
    the mirror case the robust lower endpoint rose 8.6e-4 above it.
    """
    design = {"trials": [1, 550, 16_500], "weights": [0.01, 0.825, 0.165]}
    alphas = {"alpha_lower": 1e-3, "alpha_upper": 1e-3}
    reported = weighted_empirical_bernstein_mean_bounds(
        failures=failures, unclassified=unclassified, **design, **alphas
    )
    truth = weighted_empirical_bernstein_mean_bounds(
        failures=true_labels, unclassified=[0, 0, 0], **design, **alphas
    )
    assert reported.upper >= truth.upper
    assert reported.lower <= truth.lower


def test_equal_atomic_weight_kl_zero_failure_upper_bound() -> None:
    alpha = 0.0011363636363636363
    result = weighted_kl_chernoff_mean_bounds(
        failures=[0] * 10,
        unclassified=[0] * 10,
        trials=[8] * 10,
        weights=[0.1] * 10,
        alpha_lower=alpha,
        alpha_upper=alpha,
    )
    assert result.method == "weighted_kl_chernoff"
    assert result.chernoff_kernel == "bernoulli_kl_equal_atomic_weights"
    assert result.lower == 0
    assert result.upper == pytest.approx(1 - alpha ** (1 / 80))
    assert result.effective_sample_size == pytest.approx(80)


def test_weighted_kl_uses_rigorous_unequal_weight_fallback() -> None:
    alpha = 0.025
    result = weighted_kl_chernoff_mean_bounds(
        failures=[0, 0],
        unclassified=[0, 0],
        trials=[10, 20],
        weights=[0.25, 0.75],
        alpha_lower=alpha,
        alpha_upper=alpha,
    )
    maximum_increment = 0.75 / 20
    assert result.chernoff_kernel == "weighted_poisson_chernoff_max_increment"
    assert result.maximum_increment == pytest.approx(maximum_increment)
    assert result.upper == pytest.approx(maximum_increment * math.log(1 / alpha))


def test_weighted_kl_unclassified_completion_is_monotone() -> None:
    result = weighted_kl_chernoff_mean_bounds(
        failures=[1, 0],
        unclassified=[2, 1],
        trials=[10, 10],
        weights=[0.5, 0.5],
        alpha_lower=0.05,
        alpha_upper=0.05,
    )
    assert result.completion_estimate_lower == pytest.approx(0.05)
    assert result.completion_estimate_upper == pytest.approx(0.20)
    assert result.lower <= result.completion_estimate_lower
    assert result.upper >= result.completion_estimate_upper


@pytest.mark.parametrize(
    ("trials", "weights"),
    [((10, 10), (0.5, 0.5)), ((10, 20), (0.25, 0.75))],
)
def test_weighted_kl_endpoints_increase_with_observed_failures(
    trials: tuple[int, int], weights: tuple[float, float]
) -> None:
    results = [
        weighted_kl_chernoff_mean_bounds(
            failures=[failures, 0],
            unclassified=[0, 0],
            trials=trials,
            weights=weights,
            alpha_lower=0.025,
            alpha_upper=0.025,
        )
        for failures in range(trials[0] + 1)
    ]
    adjacent = list(zip(results, results[1:], strict=False))
    assert all(left.lower <= right.lower for left, right in adjacent)
    assert all(left.upper <= right.upper for left, right in adjacent)


def test_weighted_kl_cluster_bound_ignores_extra_within_prompt_replication() -> None:
    original = weighted_kl_chernoff_mean_bounds(
        failures=[1, 2, 0],
        unclassified=[0, 0, 0],
        trials=[20, 20, 20],
        weights=[1 / 3] * 3,
        alpha_lower=0.025,
        alpha_upper=0.025,
        independence_unit="prompt_cluster",
    )
    replicated = weighted_kl_chernoff_mean_bounds(
        failures=[10, 20, 0],
        unclassified=[0, 0, 0],
        trials=[200, 200, 200],
        weights=[1 / 3] * 3,
        alpha_lower=0.025,
        alpha_upper=0.025,
        independence_unit="prompt_cluster",
    )
    assert original.chernoff_kernel == "bernoulli_kl_equal_atomic_weights"
    assert replicated.lower == pytest.approx(original.lower)
    assert replicated.upper == pytest.approx(original.upper)


@pytest.mark.parametrize(
    "probabilities",
    [(0.0, 0.0), (0.05, 0.4), (0.2, 0.8), (0.6, 0.6), (1.0, 1.0)],
)
def test_weighted_hoeffding_has_finite_sample_coverage(
    probabilities: tuple[float, float],
) -> None:
    trials = np.array([4, 5])
    weights = np.array([0.3, 0.7])
    true_mean = float(np.dot(weights, probabilities))
    coverage = 0.0
    for first in range(trials[0] + 1):
        for second in range(trials[1] + 1):
            probability_mass = float(
                binom.pmf(first, trials[0], probabilities[0])
                * binom.pmf(second, trials[1], probabilities[1])
            )
            result = weighted_hoeffding_mean_bounds(
                failures=[first, second],
                unclassified=[0, 0],
                trials=trials,
                weights=weights,
                alpha_lower=0.025,
                alpha_upper=0.025,
            )
            if result.lower <= true_mean <= result.upper:
                coverage += probability_mass
    assert coverage >= 0.95 - 1e-12


@pytest.mark.parametrize(
    "probabilities",
    [(0.0, 0.0), (0.05, 0.4), (0.2, 0.8), (0.6, 0.6), (1.0, 1.0)],
)
def test_empirical_bernstein_has_finite_sample_coverage(
    probabilities: tuple[float, float],
) -> None:
    trials = np.array([4, 5])
    weights = np.array([0.3, 0.7])
    true_mean = float(np.dot(weights, probabilities))
    coverage = 0.0
    for first in range(trials[0] + 1):
        for second in range(trials[1] + 1):
            probability_mass = float(
                binom.pmf(first, trials[0], probabilities[0])
                * binom.pmf(second, trials[1], probabilities[1])
            )
            result = weighted_empirical_bernstein_mean_bounds(
                failures=[first, second],
                unclassified=[0, 0],
                trials=trials,
                weights=weights,
                alpha_lower=0.025,
                alpha_upper=0.025,
            )
            if result.lower <= true_mean <= result.upper:
                coverage += probability_mass
    assert coverage >= 0.95 - 1e-12


@pytest.mark.parametrize(
    ("trials", "weights", "probabilities"),
    [
        ((4, 4), (0.5, 0.5), (0.05, 0.4)),
        ((4, 5), (0.3, 0.7), (0.05, 0.4)),
        ((4, 5), (0.3, 0.7), (0.2, 0.8)),
        ((4, 5), (0.3, 0.7), (0.6, 0.6)),
    ],
)
def test_weighted_kl_chernoff_has_finite_sample_coverage(
    trials: tuple[int, int],
    weights: tuple[float, float],
    probabilities: tuple[float, float],
) -> None:
    true_mean = float(np.dot(weights, probabilities))
    coverage = 0.0
    for first in range(trials[0] + 1):
        for second in range(trials[1] + 1):
            probability_mass = float(
                binom.pmf(first, trials[0], probabilities[0])
                * binom.pmf(second, trials[1], probabilities[1])
            )
            result = weighted_kl_chernoff_mean_bounds(
                failures=[first, second],
                unclassified=[0, 0],
                trials=trials,
                weights=weights,
                alpha_lower=0.025,
                alpha_upper=0.025,
            )
            if result.lower <= true_mean <= result.upper:
                coverage += probability_mass
    assert coverage >= 0.95 - 1e-12


def test_direct_mean_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        weighted_hoeffding_mean_bounds(
            failures=[0, 0],
            unclassified=[0, 0],
            trials=[1, 1],
            weights=[0.2, 0.2],
            alpha_lower=0.025,
            alpha_upper=0.025,
        )


@pytest.mark.parametrize(
    ("trials", "weights", "probabilities", "k"),
    [
        ((10,), (1.0,), (0.05,), 2),
        ((10,), (1.0,), (0.1,), 3),
        ((10,), (1.0,), (0.05,), 5),
        ((8, 8), (0.5, 0.5), (0.05, 0.05), 3),
        ((8, 8), (0.5, 0.5), (0.1, 0.1), 5),
        ((8, 8), (0.5, 0.5), (0.02, 0.3), 2),
        ((6, 9), (0.3, 0.7), (0.05, 0.05), 5),
        ((6, 9), (0.3, 0.7), (0.4, 0.1), 3),
        ((60,), (1.0,), (0.3,), 2),
        ((40, 40), (0.5, 0.5), (0.1, 0.2), 2),
    ],
)
def test_worst_of_k_bounds_have_one_sided_finite_sample_coverage(
    trials: tuple[int, ...],
    weights: tuple[float, ...],
    probabilities: tuple[float, ...],
    k: int,
) -> None:
    # Exact enumeration over the binomial outcomes. Entering every rollout as its own block
    # at the prompt's share, instead of floor(n / k) blocks, covers as little as 40% here.
    # The small designs have too few blocks to tell the estimand apart, so the last two have
    # many: bounding the mean failure rate in place of 1 - (1 - p)^k, from the same blocks,
    # covers the risk from above only 44% and 67% of the time there.
    alpha = 0.1
    risk = float(np.dot(weights, [1 - (1 - p) ** k for p in probabilities]))
    lower_coverage = upper_coverage = 0.0
    for counts in itertools.product(*(range(n + 1) for n in trials)):
        mass = math.prod(
            float(binom.pmf(count, n, p))
            for count, n, p in zip(counts, trials, probabilities, strict=True)
        )
        result = worst_of_k_kl_chernoff_bounds(
            list(counts),
            [0] * len(trials),
            list(trials),
            list(weights),
            k=k,
            alpha_lower=alpha,
            alpha_upper=alpha,
        )
        lower_coverage += mass * (result.lower <= risk)
        upper_coverage += mass * (risk <= result.upper)
    assert lower_coverage >= 1 - alpha - 1e-12
    assert upper_coverage >= 1 - alpha - 1e-12


def test_worst_of_k_bounds_at_one_attempt_are_the_rollout_level_bounds() -> None:
    arguments = {
        "failures": [3, 0, 7],
        "unclassified": [1, 2, 0],
        "trials": [20, 15, 20],
        "weights": [0.25, 0.25, 0.5],
        "alpha_lower": 0.01,
        "alpha_upper": 0.02,
    }
    worst = worst_of_k_kl_chernoff_bounds(**arguments, k=1)  # type: ignore[arg-type]
    rollout = weighted_kl_chernoff_mean_bounds(**arguments)  # type: ignore[arg-type]
    assert worst.lower == pytest.approx(rollout.lower, rel=1e-9, abs=0)
    assert worst.upper == pytest.approx(rollout.upper, rel=1e-9, abs=0)
    assert worst.chernoff_kernel == rollout.chernoff_kernel


def test_worst_of_k_bounds_enter_each_prompt_as_floor_n_over_k_blocks() -> None:
    # With 20 rollouts, k = 7 leaves two blocks per prompt and k = 11 one, the prompt level.
    failures, trials, weights = [2, 5], [20, 20], [0.5, 0.5]
    for k, blocks in ((7, 2), (11, 1)):
        shares = [
            worst_of_k_estimate([failures[i]], [trials[i]], [1.0], k=k) for i in range(2)
        ]
        expected = weighted_bounded_kl_chernoff_mean_bounds(
            np.repeat(shares, blocks),
            np.repeat(shares, blocks),
            np.full(2 * blocks, 0.5 / blocks),
            alpha_lower=0.025,
            alpha_upper=0.025,
            independence_profile="rollout",
        )
        result = worst_of_k_kl_chernoff_bounds(
            failures, [0, 0], trials, weights, k=k, alpha_lower=0.025, alpha_upper=0.025
        )
        assert (result.lower, result.upper) == (expected.lower, expected.upper)
        assert result.completion_estimate_lower == pytest.approx(
            worst_of_k_estimate(failures, trials, weights, k=k), rel=1e-12, abs=0
        )


def test_worst_of_k_bounds_complete_unclassified_rollouts_both_ways() -> None:
    def bounds(failures: list[int], unclassified: list[int]) -> tuple[float, float]:
        result = worst_of_k_kl_chernoff_bounds(
            failures,
            unclassified,
            [12, 12],
            [0.5, 0.5],
            k=4,
            alpha_lower=0.05,
            alpha_upper=0.05,
        )
        return result.lower, result.upper

    lower, upper = bounds([1, 2], [2, 1])
    for extra in itertools.product(range(3), range(2)):
        labelled_lower, labelled_upper = bounds([1 + extra[0], 2 + extra[1]], [0, 0])
        assert lower <= labelled_lower
        assert labelled_upper <= upper


def test_worst_of_k_bounds_refuse_what_the_design_cannot_bound() -> None:
    arguments = {"alpha_lower": 0.05, "alpha_upper": 0.05}
    with pytest.raises(ValueError, match="at least 5 attempted rollouts"):
        worst_of_k_kl_chernoff_bounds([0, 0], [0, 0], [10, 4], [0.5, 0.5], k=5, **arguments)
    with pytest.raises(ValueError, match="integer of at least one"):
        worst_of_k_kl_chernoff_bounds([0], [0], [4], [1.0], k=True, **arguments)
    with pytest.raises(ValueError, match="integer of at least one"):
        worst_of_k_kl_chernoff_bounds([0], [0], [4], [1.0], k=0, **arguments)


@pytest.mark.parametrize(
    ("centers", "message"),
    [
        ([0.5], "one center per prompt"),
        ([[0.5, 0.5], [0.5]], "one center per prompt"),
        ([[[0.5, 0.5]]], "one center per prompt"),
        ([0.5, 1.5], r"\[0, 1\]"),
        ([[0.5, 0.5], [0.5, float("nan")]], r"\[0, 1\]"),
    ],
)
def test_empirical_bernstein_rejects_invalid_variance_centers(
    centers: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        weighted_empirical_bernstein_mean_bounds(
            failures=[1, 2],
            unclassified=[0, 0],
            trials=[4, 5],
            weights=[0.3, 0.7],
            alpha_lower=0.025,
            alpha_upper=0.025,
            variance_centers=centers,
        )


@pytest.mark.parametrize("independence_unit", ["rollout", "prompt_cluster"])
@pytest.mark.parametrize("probabilities", [(0.0, 0.0), (0.05, 0.4), (0.6, 0.6), (1.0, 1.0)])
def test_centered_empirical_bernstein_has_finite_sample_coverage(
    independence_unit: str, probabilities: tuple[float, float]
) -> None:
    # Any center fixed in advance keeps the guarantee, including one far from the data.
    trials = np.array([4, 5])
    weights = np.array([0.3, 0.7])
    true_mean = float(np.dot(weights, probabilities))
    for centers in ([0.0, 1.0], [0.5, 0.5], [1.0, 0.0]):
        coverage = 0.0
        for first in range(trials[0] + 1):
            for second in range(trials[1] + 1):
                mass = float(
                    binom.pmf(first, trials[0], probabilities[0])
                    * binom.pmf(second, trials[1], probabilities[1])
                )
                result = weighted_empirical_bernstein_mean_bounds(
                    failures=[first, second],
                    unclassified=[0, 0],
                    trials=trials,
                    weights=weights,
                    alpha_lower=0.025,
                    alpha_upper=0.025,
                    independence_unit=independence_unit,
                    variance_centers=centers,
                )
                if result.lower <= true_mean <= result.upper:
                    coverage += mass
        assert coverage >= 0.95 - 1e-12, centers


def test_centered_empirical_bernstein_contains_every_completion_of_unknowns() -> None:
    common = dict(trials=[6, 8], weights=[0.5, 0.5], alpha_lower=0.05, alpha_upper=0.05)
    unknown = weighted_empirical_bernstein_mean_bounds(
        failures=[1, 0], unclassified=[2, 3], variance_centers=[0.5, 0.0], **common
    )
    for extra_first in range(3):
        for extra_second in range(4):
            known = weighted_empirical_bernstein_mean_bounds(
                failures=[1 + extra_first, extra_second],
                unclassified=[0, 0],
                variance_centers=[0.5, 0.0],
                **common,
            )
            assert unknown.lower <= known.lower + 1e-12
            assert known.upper <= unknown.upper + 1e-12


@pytest.mark.parametrize("independence_unit", ["rollout", "prompt_cluster"])
def test_default_variance_centers_are_the_uncentered_proxies(independence_unit: str) -> None:
    # Without centers the engine keeps its two proxies, the second moments of the outcomes
    # and of their complements, each with a quarter of the endpoint budget.
    failures, unknown = np.array([1, 0, 3]), np.array([1, 2, 0])
    trials, weights = np.array([30, 40, 50]), np.array([0.2, 0.3, 0.5])
    common = dict(
        failures=failures,
        unclassified=unknown,
        trials=trials,
        weights=weights,
        alpha_lower=0.05,
        alpha_upper=0.025,
        independence_unit=independence_unit,
    )
    default = weighted_empirical_bernstein_mean_bounds(**common)
    explicit = weighted_empirical_bernstein_mean_bounds(
        **common, variance_centers=[np.zeros(3), np.ones(3)]
    )
    assert default == explicit
    if independence_unit == "rollout":
        a2, n = weights**2 / trials, trials
        positive = float(np.sum(a2 * (failures + unknown) / n))
        negative = float(np.sum(a2 * (n - failures) / n))
        c2, c4, largest = float(np.sum(a2)), float(np.sum(weights**4 / n**3)), max(weights / n)
    else:
        positive = float(np.sum(weights**2 * ((failures + unknown) / trials) ** 2))
        negative = float(np.sum(weights**2 * (1 - failures / trials) ** 2))
        c2, c4, largest = float(np.sum(weights**2)), float(np.sum(weights**4)), max(weights)
    endpoints = ((0.05, default.variance_upper_lower), (0.025, default.variance_upper_upper))
    for alpha, variance in endpoints:
        padding = math.sqrt(0.5 * c4 * math.log(4 / alpha))
        expected = min(c2 / 4, positive + padding, negative + padding)
        assert variance == pytest.approx(expected, rel=1e-12, abs=0)
    assert default.maximum_increment == pytest.approx(largest, abs=0)


def test_variance_padding_uses_the_range_of_each_centered_term() -> None:
    # About 1/2, a squared deviation of an observation in [0, 1] lies in [0, 1/4].
    weights = np.full(50, 1 / 50)
    result = weighted_empirical_bernstein_mean_bounds(
        failures=np.full(50, 5),
        unclassified=np.zeros(50, dtype=int),
        trials=np.full(50, 10),
        weights=weights,
        alpha_lower=0.05,
        alpha_upper=0.05,
        independence_unit="prompt_cluster",
        variance_centers=np.full(50, 0.5),
    )
    padding = math.sqrt(0.5 * float(np.sum(weights**4)) / 16 * math.log(2 / 0.05))
    assert result.variance_upper_lower == pytest.approx(padding, rel=1e-12, abs=0)


def test_more_variance_centerings_share_the_same_budget() -> None:
    # The centerings split one budget. A second centering that is not the smallest proxy
    # only enlarges the padding of the one that is.
    common = dict(
        failures=[0, 1],
        unclassified=[0, 0],
        trials=[200, 200],
        weights=[0.5, 0.5],
        alpha_lower=0.05,
        alpha_upper=0.05,
    )
    one = weighted_empirical_bernstein_mean_bounds(**common, variance_centers=[0.0, 0.0])
    two = weighted_empirical_bernstein_mean_bounds(
        **common, variance_centers=[[0.0, 0.0], [1.0, 1.0]]
    )
    assert one.variance_upper_lower is not None and two.variance_upper_lower is not None
    assert one.variance_upper_lower < two.variance_upper_lower
