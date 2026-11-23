"""The single-valued bound M = inf_delta [U_delta + delta] (#127): P(M < R) <= R."""

import math

import numpy as np
import pytest
from scipy.stats import binom

from safety_stats.stats import (
    binomial_upper_bound,
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
)
from safety_stats.stats.single_valued import (
    clopper_pearson_single_valued_bound,
    single_valued_bound,
)

RISKS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.9)


def _miss_probability(bound_for_failures, trials: int, risk: float) -> float:
    """Exact P(M < R) for S ~ Binomial(trials, risk), with M non-decreasing in S."""
    total = 0.0
    for failures in range(trials + 1):
        if bound_for_failures(failures) >= risk:
            break
        total += float(binom.pmf(failures, trials, risk))
    return total


@pytest.mark.parametrize("trials", [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000])
def test_clopper_pearson_miss_probability_never_exceeds_the_risk(trials: int) -> None:
    values = [clopper_pearson_single_valued_bound(s, trials).value for s in range(trials + 1)]
    assert all(a <= b for a, b in zip(values, values[1:], strict=False))
    for risk in RISKS:
        miss = _miss_probability(lambda s: values[s], trials, risk)
        assert miss <= risk, (trials, risk, miss)


def test_clopper_pearson_reproduces_the_figures_in_the_issue() -> None:
    def miss(trials: int, risk: float) -> float:
        return _miss_probability(
            lambda s: clopper_pearson_single_valued_bound(s, trials).value, trials, risk
        )

    assert miss(1000, 0.05) == pytest.approx(7.19e-4, rel=2e-3, abs=0)
    assert miss(1000, 0.01) == pytest.approx(4.32e-5, rel=2e-3, abs=0)
    assert miss(20, 0.4) == pytest.approx(1.60e-2, rel=5e-3, abs=0)
    bound = clopper_pearson_single_valued_bound(55, 1000)
    assert bound.value == pytest.approx(0.0808, abs=5e-5)
    assert bound.value > binomial_upper_bound(55, 1000, alpha=0.05)


def test_value_is_attained_at_the_reported_delta_and_is_not_beaten_by_a_fine_scan() -> None:
    bound = clopper_pearson_single_valued_bound(55, 1000)
    assert bound.value == pytest.approx(bound.upper_at_delta + bound.delta, rel=0, abs=1e-15)
    assert bound.upper_at_delta == binomial_upper_bound(55, 1000, alpha=bound.delta)
    deltas = np.linspace(1e-4, 0.05, 20001)
    scan = min(binomial_upper_bound(55, 1000, alpha=float(d)) + float(d) for d in deltas)
    # Never below the attained infimum, and within the scan's resolution of it.
    assert bound.value <= scan + 1e-12
    assert bound.value >= scan - 1e-5


def test_all_failures_give_one() -> None:
    assert clopper_pearson_single_valued_bound(10, 10).value == 1.0


def test_a_family_that_rises_with_delta_is_refused() -> None:
    with pytest.raises(ValueError, match="increases with delta"):
        single_valued_bound(lambda delta: delta / 2)


def test_a_non_finite_family_is_refused() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        single_valued_bound(lambda delta: math.nan)


ENGINES = {
    "weighted_hoeffding": weighted_bounded_hoeffding_mean_bounds,
    "weighted_empirical_bernstein": weighted_bounded_empirical_bernstein_mean_bounds,
    "weighted_kl_chernoff": weighted_bounded_kl_chernoff_mean_bounds,
}


@pytest.mark.parametrize("method", sorted(ENGINES))
@pytest.mark.parametrize("trials", [5, 20, 50])
def test_direct_mean_engines_miss_probability_never_exceeds_the_risk(
    method: str, trials: int
) -> None:
    """Equal-weight binary rollouts: each engine's M depends only on the failure count."""
    engine = ENGINES[method]
    weights = np.full(trials, 1 / trials)

    def bound(failures: int) -> float:
        outcomes = np.zeros(trials)
        outcomes[:failures] = 1.0
        return single_valued_bound(
            lambda delta: engine(
                outcomes,
                outcomes,
                weights,
                alpha_lower=delta,
                alpha_upper=delta,
                independence_profile="rollout",
            ).upper
        ).value

    values = [bound(s) for s in range(trials + 1)]
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:], strict=False))
    for risk in RISKS:
        miss = _miss_probability(lambda s: values[s], trials, risk)
        assert miss <= risk, (method, trials, risk, miss)
