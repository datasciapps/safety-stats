import numpy as np
import pytest

from safety_stats.stats.design_based import (
    FixedSizeDesign,
    InclusionDesign,
    fixed_size_mean_bounds,
    horvitz_thompson_mean_bounds,
)
from safety_stats.stats.direct_mean import weighted_bounded_kl_chernoff_mean_bounds


def _frame(size: int) -> np.ndarray:
    return np.full(size, 1.0 / size)


def test_census_reduces_exactly_to_the_direct_bounded_mean() -> None:
    """With every inclusion probability at one there is no sampling uncertainty to add."""
    rng = np.random.default_rng(3)
    weights = _frame(80)
    rates = rng.beta(0.5, 10, size=80)
    design = InclusionDesign(weights, np.ones(80), np.ones(80, dtype=bool))

    sampled = horvitz_thompson_mean_bounds(
        design, rates, rates, alpha_lower=0.025, alpha_upper=0.025,
        method="weighted_kl_chernoff",
    )
    direct = weighted_bounded_kl_chernoff_mean_bounds(
        rates, rates, weights, alpha_lower=0.025, alpha_upper=0.025,
        independence_profile="rollout",
    )
    assert sampled.inflation == pytest.approx(1.0)
    assert sampled.lower == pytest.approx(direct.lower)
    assert sampled.upper == pytest.approx(direct.upper)


def test_estimator_is_unbiased_for_the_frame_mean() -> None:
    """Horvitz-Thompson targets the whole frame, not the realized sample."""
    rng = np.random.default_rng(17)
    size = 200
    weights = _frame(size)
    probabilities = np.full(size, 0.3)
    risks = rng.beta(0.4, 8.0, size=size)
    target = float(weights @ risks)

    draws = []
    for _ in range(3000):
        included = rng.random(size) < probabilities
        rates = np.where(included, risks, 0.0)
        design = InclusionDesign(weights, probabilities, included)
        draws.append(
            horvitz_thompson_mean_bounds(
                design, rates, rates, alpha_lower=0.025, alpha_upper=0.025
            ).estimate
        )
    assert float(np.mean(draws)) == pytest.approx(target, abs=3 * float(np.std(draws)) / 55)


def test_bounds_cover_the_frame_mean_under_repeated_sampling() -> None:
    rng = np.random.default_rng(5)
    size, rollouts = 150, 20
    weights = _frame(size)
    probabilities = np.full(size, 0.4)

    covered = 0
    for _ in range(200):
        risks = rng.beta(0.5, 12.0, size=size)
        target = float(weights @ risks)
        included = rng.random(size) < probabilities
        rates = np.zeros(size)
        rates[included] = rng.binomial(rollouts, risks[included]) / rollouts
        bounds = horvitz_thompson_mean_bounds(
            InclusionDesign(weights, probabilities, included),
            rates,
            rates,
            alpha_lower=0.025,
            alpha_upper=0.025,
        )
        covered += bounds.lower <= target <= bounds.upper
    # A finite-sample concentration bound is conservative; it must not undercover.
    assert covered >= 195


def test_rarely_sampled_items_concentrate_weight_and_widen_the_bound() -> None:
    size = 100
    weights = _frame(size)
    rates = np.full(size, 0.05)
    included = np.ones(size, dtype=bool)

    even = np.full(size, 0.5)
    skewed = np.full(size, 0.5)
    skewed[0] = 0.02

    balanced = horvitz_thompson_mean_bounds(
        InclusionDesign(weights, even, included), rates, rates,
        alpha_lower=0.025, alpha_upper=0.025,
    )
    concentrated = horvitz_thompson_mean_bounds(
        InclusionDesign(weights, skewed, included), rates, rates,
        alpha_lower=0.025, alpha_upper=0.025,
    )
    assert concentrated.maximum_weight > balanced.maximum_weight
    assert concentrated.effective_sample_size < balanced.effective_sample_size
    assert concentrated.upper - concentrated.lower > balanced.upper - balanced.lower


def test_lower_inclusion_probability_costs_precision() -> None:
    size = 200
    weights = _frame(size)
    rates = np.full(size, 0.05)
    included = np.ones(size, dtype=bool)

    widths = []
    for probability in (1.0, 0.5, 0.2):
        bounds = horvitz_thompson_mean_bounds(
            InclusionDesign(weights, np.full(size, probability), included),
            rates, rates, alpha_lower=0.025, alpha_upper=0.025,
        )
        assert bounds.inflation == pytest.approx(1 / probability)
        widths.append(bounds.upper - bounds.lower)
    assert widths[0] < widths[1] < widths[2]


def test_unclassified_outcomes_widen_through_completion() -> None:
    size = 60
    weights = _frame(size)
    included = np.ones(size, dtype=bool)
    design = InclusionDesign(weights, np.full(size, 0.5), included)
    complete = horvitz_thompson_mean_bounds(
        design, np.full(size, 0.1), np.full(size, 0.1),
        alpha_lower=0.025, alpha_upper=0.025,
    )
    partial = horvitz_thompson_mean_bounds(
        design, np.full(size, 0.1), np.full(size, 0.2),
        alpha_lower=0.025, alpha_upper=0.025,
    )
    assert partial.upper > complete.upper
    assert partial.lower == pytest.approx(complete.lower)


def test_invalid_designs_are_rejected() -> None:
    weights = _frame(10)
    included = np.ones(10, dtype=bool)
    with pytest.raises(ValueError, match="must lie in"):
        InclusionDesign(weights, np.zeros(10), included)
    with pytest.raises(ValueError, match="must lie in"):
        InclusionDesign(weights, np.full(10, 1.5), included)
    with pytest.raises(ValueError, match="at least one frame item"):
        InclusionDesign(weights, np.full(10, 0.5), np.zeros(10, dtype=bool))
    with pytest.raises(ValueError, match="sum to one"):
        InclusionDesign(np.full(10, 0.5), np.full(10, 0.5), included)
    with pytest.raises(ValueError, match="aligned"):
        InclusionDesign(weights, np.full(9, 0.5), included)


def test_unsampled_items_must_carry_no_observation() -> None:
    size = 10
    design = InclusionDesign(
        _frame(size), np.full(size, 0.5), np.array([False] + [True] * (size - 1))
    )
    rates = np.full(size, 0.2)
    with pytest.raises(ValueError, match="unsampled frame items"):
        horvitz_thompson_mean_bounds(
            design, rates, rates, alpha_lower=0.025, alpha_upper=0.025
        )
    with pytest.raises(ValueError, match="unsupported design-based method"):
        horvitz_thompson_mean_bounds(
            design,
            np.where(design.sampled, 0.2, 0.0),
            np.where(design.sampled, 0.2, 0.0),
            alpha_lower=0.025,
            alpha_upper=0.025,
            method="bootstrap",  # type: ignore[arg-type]
        )


def _srswor(size: int, draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(size, dtype=bool)
    mask[rng.choice(size, draws, replace=False)] = True
    return np.zeros(size, dtype=int), mask


def test_fixed_size_equal_weights_carry_no_inflation() -> None:
    """A fixed-size draw with equal weights is a plain sample mean, so K = 1."""
    rng = np.random.default_rng(9)
    size, draws = 120, 40
    strata, mask = _srswor(size, draws, rng)
    rates = np.zeros(size)
    rates[mask] = 0.05
    bounds = fixed_size_mean_bounds(
        FixedSizeDesign(strata, _frame(size), np.array([draws]), mask),
        rates, rates, alpha_lower=0.025, alpha_upper=0.025,
    )
    assert bounds.inflation == pytest.approx(1.0)
    assert bounds.estimator == "stratified_without_replacement"
    assert bounds.sampled_count == draws
    assert bounds.maximum_weight == pytest.approx(1 / draws)


def test_fixed_size_bounds_cover_the_frame_mean() -> None:
    rng = np.random.default_rng(23)
    size, draws, rollouts = 160, 50, 20
    covered = 0
    for _ in range(200):
        risks = rng.beta(0.5, 12.0, size=size)
        weights = _frame(size)
        target = float(weights @ risks)
        strata, mask = _srswor(size, draws, rng)
        rates = np.zeros(size)
        rates[mask] = rng.binomial(rollouts, risks[mask]) / rollouts
        bounds = fixed_size_mean_bounds(
            FixedSizeDesign(strata, weights, np.array([draws]), mask),
            rates, rates, alpha_lower=0.025, alpha_upper=0.025,
        )
        covered += bounds.lower <= target <= bounds.upper
    assert covered >= 195


def test_fixed_size_rejects_a_sample_that_misses_its_plan() -> None:
    rng = np.random.default_rng(1)
    size, draws = 40, 10
    strata, mask = _srswor(size, draws, rng)
    short = mask.copy()
    short[np.flatnonzero(short)[0]] = False
    with pytest.raises(ValueError, match="exactly its planned draws"):
        FixedSizeDesign(strata, _frame(size), np.array([draws]), short)
    with pytest.raises(ValueError, match="cannot draw more items"):
        FixedSizeDesign(strata, _frame(size), np.array([size + 1]), mask)


def test_fixed_size_handles_a_target_confined_to_part_of_the_frame() -> None:
    """A hierarchy node consumes stratum draws it does not use; K reflects that."""
    rng = np.random.default_rng(31)
    size, draws = 100, 50
    strata, mask = _srswor(size, draws, rng)
    weights = np.zeros(size)
    weights[:20] = 1 / 20          # target covers a fifth of the frame
    rates = np.zeros(size)
    rates[mask] = 0.1
    bounds = fixed_size_mean_bounds(
        FixedSizeDesign(strata, weights, np.array([draws]), mask),
        rates, rates, alpha_lower=0.025, alpha_upper=0.025,
    )
    assert bounds.inflation == pytest.approx(size / 20)
    assert bounds.lower <= 0.1 <= bounds.upper
