"""Detectable rates planned under the interval method analyze uses."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.optimize import brentq

from inspect_safety_stats.config.methods import DIRECT_MEAN_METHODS, DirectMeanMethod
from inspect_safety_stats.stats.direct_mean import (
    DirectMeanBounds,
    weighted_bounded_betting_mean_bounds,
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
)
from inspect_safety_stats.stats.planning import (
    INVERTIBLE_METHODS,
    certification_floor,
    detectable_failure_rates,
    fixed_design_detectable_failure_rates,
)

ANALYSIS: dict[DirectMeanMethod, Callable[..., DirectMeanBounds]] = {
    "weighted_hoeffding": weighted_bounded_hoeffding_mean_bounds,
    "weighted_empirical_bernstein": weighted_bounded_empirical_bernstein_mean_bounds,
    "weighted_kl_chernoff": weighted_bounded_kl_chernoff_mean_bounds,
}
METHODS = list(ANALYSIS)
ALPHA = 0.0125
POWER = 0.8
DRAWS = 1000
# A rate planned at power 0.8 must be decided in at least 0.8 of the draws, less three Monte
# Carlo standard errors (sqrt(0.8 * 0.2 / 1000) = 0.0126): a sound plan whose power were
# exactly 0.8 would fail about once in 700 seeds. A rate on the decision boundary itself, with
# no power side, is decided about half the time and fails it.
POWER_FLOOR = POWER - 3 * math.sqrt(POWER * (1 - POWER) / DRAWS)


def _analysis(
    method: DirectMeanMethod, values: np.ndarray, weights: np.ndarray
) -> DirectMeanBounds:
    return ANALYSIS[method](
        values,
        values,
        weights,
        alpha_lower=ALPHA,
        alpha_upper=ALPHA,
        independence_profile="test",
    )


def _weights(kind: str) -> np.ndarray:
    if kind == "equal":
        return np.full(800, 1 / 800)
    weights = np.random.default_rng(7).uniform(0.5, 1.5, size=800)
    return weights / weights.sum()


def _power(
    method: DirectMeanMethod,
    weights: np.ndarray,
    rate: float,
    decided: Callable[[DirectMeanBounds], bool],
    *,
    spread: bool,
    seed: int,
) -> float:
    """The share of DRAWS simulated designs, with weighted failure rate ``rate``, decided.

    With ``spread`` the blocks' rates differ around ``rate``, keeping its weighted mean: the
    guarantee does not assume a common rate. A common rate maximises the variance of the
    estimate, so it is the harder case.
    """
    rng = np.random.default_rng(seed)
    block_rates = np.full(len(weights), rate)
    if spread:
        offsets = rng.uniform(-1.0, 1.0, size=len(weights))
        offsets -= float(np.dot(weights, offsets))
        block_rates += 0.5 * min(rate, 1 - rate) * offsets / float(np.max(np.abs(offsets)))
    assert float(np.dot(weights, block_rates)) == pytest.approx(rate, rel=1e-12, abs=1e-15)
    assert 0 <= block_rates.min() and block_rates.max() <= 1
    hits = sum(
        decided(_analysis(method, (rng.random(len(weights)) < block_rates) * 1.0, weights))
        for _ in range(DRAWS)
    )
    return hits / DRAWS


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["equal", "unequal"])
@pytest.mark.parametrize("spread", [False, True])
def test_planned_rates_reach_the_target_power_under_the_analysis_interval(
    method: DirectMeanMethod, kind: str, spread: bool
) -> None:
    weights = _weights(kind)
    rates = detectable_failure_rates(
        weights,
        method=method,
        alpha=ALPHA,
        power=POWER,
        acceptable_upper=0.08,
        unacceptable_lower=0.08,
    )
    assert rates.maximum_certifiable_failure_rate is not None
    assert rates.minimum_detectable_failure_rate is not None
    certified = _power(
        method,
        weights,
        rates.maximum_certifiable_failure_rate,
        lambda bounds: bounds.upper <= 0.08,
        spread=spread,
        seed=11,
    )
    detected = _power(
        method,
        weights,
        rates.minimum_detectable_failure_rate,
        lambda bounds: bounds.lower >= 0.08,
        spread=spread,
        seed=12,
    )
    assert certified >= POWER_FLOOR
    assert detected >= POWER_FLOOR


def _poisson_root(observed: float, target: float, *, above: bool) -> float:
    """The mean on one side of ``observed`` whose Poisson divergence from it is ``target``.

    Computed independently of the planner, for its exact-value checks; ``1.0`` when no mean up
    to 1 reaches the target from above.
    """

    def excess(mean: float) -> float:
        return observed * math.log(observed / mean) - observed + mean - target

    if above:
        return 1.0 if excess(1.0) <= 0 else float(brentq(excess, observed, 1.0, xtol=1e-15))
    return float(brentq(excess, 1e-300, observed, xtol=1e-15))


@pytest.mark.parametrize("method", METHODS)
def test_the_kernel_on_the_safe_outcomes_plans_high_failure_rates(
    method: DirectMeanMethod,
) -> None:
    """Under unequal weights the planner also applies the Poisson kernel to the safe outcomes.

    The Poisson divergence is not symmetric, so near a high threshold the bound on the safe
    outcomes is the sharpest of the three power bounds, on both sides. (Under equal weights
    the Bernoulli KL kernel is symmetric and it adds nothing.)
    """
    weights = np.random.default_rng(7).uniform(0.9, 1.1, size=400)
    weights /= weights.sum()
    acceptable_upper, unacceptable_lower = 0.95, 0.8
    rates = detectable_failure_rates(
        weights,
        method=method,
        alpha=ALPHA,
        power=POWER,
        acceptable_upper=acceptable_upper,
        unacceptable_lower=unacceptable_lower,
    )
    certifiable = rates.maximum_certifiable_failure_rate
    detectable = rates.minimum_detectable_failure_rate
    assert certifiable is not None and detectable is not None
    if method != "weighted_empirical_bernstein":
        # Both endpoints depend on the data only through the estimate, so the decision
        # boundaries follow from the analysis interval on blocks that all equal it.
        def endpoint(estimate: float, side: int) -> float:
            values = np.full(len(weights), estimate)
            bounds = _analysis(method, values, weights)
            return bounds.upper if side else bounds.lower

        passing = float(brentq(lambda s: endpoint(s, 1) - acceptable_upper, 0, 1, xtol=1e-15))
        failing = float(brentq(lambda s: endpoint(s, 0) - unacceptable_lower, 0, 1, xtol=1e-15))
        target = float(np.max(weights)) * math.log(1 / (1 - POWER))
        radius = math.sqrt(0.5 * float(np.sum(np.square(weights))) * math.log(1 / (1 - POWER)))
        safe_pass = 1 - _poisson_root(1 - passing, target, above=True)
        safe_fail = 1 - _poisson_root(1 - failing, target, above=False)
        assert certifiable == pytest.approx(safe_pass, rel=0, abs=1e-9)
        assert detectable == pytest.approx(safe_fail, rel=0, abs=1e-9)
        # The other two bounds are weaker here.
        assert safe_pass > max(passing - radius, _poisson_root(passing, target, above=False))
        assert safe_fail < min(failing + radius, _poisson_root(failing, target, above=True))
    for spread in (False, True):
        assert (
            _power(
                method,
                weights,
                certifiable,
                lambda bounds: bounds.upper <= acceptable_upper,
                spread=spread,
                seed=21,
            )
            >= POWER_FLOOR
        )
        assert (
            _power(
                method,
                weights,
                detectable,
                lambda bounds: bounds.lower >= unacceptable_lower,
                spread=spread,
                seed=22,
            )
            >= POWER_FLOOR
        )


@pytest.mark.parametrize("method", METHODS)
def test_certification_floor_is_the_analysis_upper_bound_on_all_safe_blocks(
    method: DirectMeanMethod,
) -> None:
    weights = _weights("unequal")
    rates = detectable_failure_rates(weights, method=method, alpha=ALPHA, power=0.9)
    zeros = np.zeros(len(weights))
    ones = np.ones(len(weights))
    assert rates.certification_floor == pytest.approx(
        _analysis(method, zeros, weights).upper, rel=1e-12, abs=0
    )
    assert rates.rejection_ceiling == pytest.approx(
        _analysis(method, ones, weights).lower, rel=1e-12, abs=0
    )
    assert certification_floor(weights, method=method, alpha=ALPHA) == pytest.approx(
        rates.certification_floor, rel=1e-12, abs=0
    )


def test_the_planner_inverts_every_registered_method_but_betting() -> None:
    assert set(DIRECT_MEAN_METHODS) - INVERTIBLE_METHODS == {"weighted_betting"}
    assert INVERTIBLE_METHODS == set(METHODS)


def test_betting_is_refused_for_rates_but_has_an_exact_certification_floor() -> None:
    weights = _weights("unequal")
    with pytest.raises(ValueError, match="cannot invert the weighted_betting interval"):
        detectable_failure_rates(
            weights, method="weighted_betting", alpha=ALPHA, power=0.9, acceptable_upper=0.1
        )
    zeros = np.zeros(len(weights))
    expected = weighted_bounded_betting_mean_bounds(
        zeros,
        zeros,
        weights,
        alpha_lower=ALPHA,
        alpha_upper=ALPHA,
        independence_profile="test",
    ).upper
    floor = certification_floor(weights, method="weighted_betting", alpha=ALPHA)
    assert floor == pytest.approx(expected, rel=1e-12, abs=0)
    assert 0 < floor < 1


@pytest.mark.parametrize("kind", ["equal", "unequal"])
def test_empirical_bernstein_planning_takes_the_widest_interval_of_an_estimate(kind: str) -> None:
    # The planner bounds the EB interval of any data by the one at the largest second moments
    # the estimate allows; observed data never gives a wider one.
    from inspect_safety_stats.stats.planning import _planned_endpoints

    weights = _weights(kind)
    rng = np.random.default_rng(3)
    for _ in range(50):
        values = (rng.random(len(weights)) < rng.uniform(0.0, 0.3)) * rng.uniform(
            0, 1, size=len(weights)
        )
        observed = _analysis("weighted_empirical_bernstein", values, weights)
        lower, upper = _planned_endpoints(
            "weighted_empirical_bernstein", weights, float(np.dot(weights, values)), ALPHA
        )
        assert upper >= observed.upper - 1e-12
        assert lower <= observed.lower + 1e-12


def test_kl_chernoff_planning_is_less_pessimistic_than_the_hoeffding_reference() -> None:
    weights = np.full(2000, 1 / 2000)
    reference = fixed_design_detectable_failure_rates(
        [1] * 2000,
        weights,
        threshold=0.03,
        support_alpha=ALPHA,
        rejection_alpha=ALPHA,
        power=0.9,
        independence_unit="prompt_cluster",
    )
    planned = detectable_failure_rates(
        weights,
        method="weighted_kl_chernoff",
        alpha=ALPHA,
        power=0.9,
        acceptable_upper=0.03,
        unacceptable_lower=0.03,
    )
    # At a rare-event threshold Hoeffding cannot certify any rate, while KL-Chernoff can.
    assert reference.maximum_certifiable_failure_rate is None
    assert planned.maximum_certifiable_failure_rate is not None
    assert 0 < planned.maximum_certifiable_failure_rate < 0.03
    assert reference.minimum_detectable_failure_rate is not None
    assert planned.minimum_detectable_failure_rate is not None
    assert planned.minimum_detectable_failure_rate < reference.minimum_detectable_failure_rate


@pytest.mark.parametrize("method", METHODS)
def test_a_threshold_no_outcome_can_cross_is_unattainable(method: DirectMeanMethod) -> None:
    weights = np.full(10, 0.1)
    rates = detectable_failure_rates(
        weights,
        method=method,
        alpha=ALPHA,
        power=0.9,
        acceptable_upper=0.01,
        unacceptable_lower=0.99,
    )
    assert rates.certification_floor > 0.01
    assert rates.maximum_certifiable_failure_rate is None
    assert rates.rejection_ceiling < 0.99
    assert rates.minimum_detectable_failure_rate is None


def test_planning_rejects_weights_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        detectable_failure_rates([0.5, 0.4], method="weighted_kl_chernoff", alpha=0.05, power=0.9)
