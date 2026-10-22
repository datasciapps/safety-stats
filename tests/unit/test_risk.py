import math

import numpy as np
import pytest
from pydantic import ValidationError

from inspect_safety_stats.config.models import RiskMeasureSpec
from inspect_safety_stats.stats import (
    entropic_risk,
    maximum_risk,
    threshold_exceedance,
    threshold_excess,
    top_k_mean,
    weighted_cvar,
    weighted_mean,
    worst_of_k_estimate,
    worst_of_k_risk,
)


def test_weighted_mean_and_maximum() -> None:
    values = [0.0, 0.1, 0.5]
    weights = [0.5, 0.25, 0.25]
    assert weighted_mean(values, weights) == pytest.approx(0.15)
    assert maximum_risk(values, weights) == 0.5


def test_cvar_uses_fractional_boundary_mass() -> None:
    assert weighted_cvar([0.0, 0.5], [0.99, 0.01], beta=0.9) == pytest.approx(0.05)


def test_equal_weight_cvar_matches_top_k() -> None:
    values = [0.1, 0.2, 0.4, 0.8]
    assert weighted_cvar(values, [0.25] * 4, beta=0.5) == pytest.approx(top_k_mean(values, k=2))


def test_entropic_limits_and_example() -> None:
    values = [0.0, 0.5]
    weights = [0.99, 0.01]
    assert entropic_risk(values, weights, lambda_=1e-9) == pytest.approx(0.005)
    assert entropic_risk(values, weights, lambda_=10) == pytest.approx(0.0905889462)
    assert entropic_risk(values, weights, lambda_=1000) == pytest.approx(0.5, abs=0.005)


def test_threshold_measures() -> None:
    values = [0.0, 0.2, 0.8]
    weights = [0.2, 0.3, 0.5]
    assert threshold_exceedance(values, weights, threshold=0.1) == pytest.approx(0.8)
    assert threshold_excess(values, weights, threshold=0.1) == pytest.approx(0.38)


def test_replication_invariance() -> None:
    original = entropic_risk([0.1, 0.8], [0.6, 0.4], lambda_=5)
    replicated = entropic_risk([0.1, 0.1, 0.8], [0.3, 0.3, 0.4], lambda_=5)
    assert original == pytest.approx(replicated)


def test_monotone_bound_propagation() -> None:
    lower = np.array([0.01, 0.1, 0.2])
    upper = lower + 0.2
    weights = np.array([0.2, 0.3, 0.5])
    assert weighted_cvar(lower, weights, beta=0.5) <= weighted_cvar(upper, weights, beta=0.5)
    assert entropic_risk(lower, weights, lambda_=4) <= entropic_risk(upper, weights, lambda_=4)
    assert math.isfinite(entropic_risk(upper, weights, lambda_=1e5))


def test_worst_of_k_reduces_to_weighted_mean_at_k_one() -> None:
    values = [0.0, 0.1, 0.5]
    weights = [0.5, 0.25, 0.25]
    assert worst_of_k_risk(values, weights, k=1) == pytest.approx(
        weighted_mean(values, weights)
    )


def test_worst_of_k_exact_value() -> None:
    # 1 - (1 - 0.5)^2 = 0.75; weighted: 0.5 * 0 + 0.5 * 0.75 = 0.375.
    assert worst_of_k_risk([0.0, 0.5], [0.5, 0.5], k=2) == pytest.approx(0.375)


def test_worst_of_k_is_monotone_in_k() -> None:
    values = [0.05, 0.2, 0.6]
    weights = [0.5, 0.3, 0.2]
    risks = [worst_of_k_risk(values, weights, k=k) for k in (1, 2, 4, 8)]
    assert risks == sorted(risks)
    assert all(0.0 <= risk <= 1.0 for risk in risks)


def test_worst_of_k_replication_invariance() -> None:
    original = worst_of_k_risk([0.1, 0.8], [0.6, 0.4], k=3)
    replicated = worst_of_k_risk([0.1, 0.1, 0.8], [0.3, 0.3, 0.4], k=3)
    assert original == pytest.approx(replicated)


def test_worst_of_k_certain_failure_and_certain_safety() -> None:
    assert worst_of_k_risk([0.0, 0.0], [0.5, 0.5], k=10) == 0.0
    assert worst_of_k_risk([1.0, 1.0], [0.5, 0.5], k=1) == 1.0


def test_worst_of_k_rejects_values_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        worst_of_k_risk([0.2, 1.2], [0.5, 0.5], k=2)
    with pytest.raises(ValueError, match="probabilities"):
        worst_of_k_risk([-0.1, 0.2], [0.5, 0.5], k=2)


def test_worst_of_k_rejects_invalid_k() -> None:
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="k must be"):
            worst_of_k_risk([0.1, 0.2], [0.5, 0.5], k=bad)  # type: ignore[arg-type]


def test_worst_of_k_spec_requires_k() -> None:
    with pytest.raises(ValueError, match="worst_of_k requires k > 0"):
        RiskMeasureSpec.model_validate({"id": "wk", "type": "worst_of_k"})
    spec = RiskMeasureSpec.model_validate({"id": "wk", "type": "worst_of_k", "k": 4})
    assert spec.k == 4


def test_worst_of_k_stays_accurate_for_tiny_probabilities() -> None:
    """1 - (1 - p)**k loses all precision once 1 - p rounds to one."""
    # 1 - (1 - p)**k = k p - C(k, 2) p**2 + ..., so k p is accurate to about k p**2 here.
    # abs=0: pytest.approx otherwise also accepts anything within 1e-12, which is all of these.
    assert worst_of_k_risk([1e-12], [1.0], k=10) == pytest.approx(1e-11, rel=1e-9, abs=0)
    assert worst_of_k_risk([1e-17], [1.0], k=3) == pytest.approx(3e-17, rel=1e-9, abs=0)
    zero = worst_of_k_risk([0.0, 0.0], [0.5, 0.5], k=7)
    assert zero == 0.0 and math.copysign(1.0, zero) == 1.0
    assert worst_of_k_risk([1.0, 0.25], [0.5, 0.5], k=3) == pytest.approx(
        0.5 * 1.0 + 0.5 * (1 - 0.75**3), rel=1e-15, abs=0
    )


def test_worst_of_k_at_one_attempt_is_exactly_the_weighted_mean() -> None:
    values, weights = [0.1, 0.37, 0.9], [0.2, 0.3, 0.5]
    assert worst_of_k_risk(values, weights, k=1) == weighted_mean(values, weights)


@pytest.mark.parametrize("n", [3, 10, 40])
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("p", [0.01, 0.3, 0.8])
def test_the_worst_of_k_estimate_is_unbiased_where_the_plug_in_understates(
    n: int, k: int, p: float
) -> None:
    truth = 1 - (1 - p) ** k
    pmf = [math.comb(n, f) * p**f * (1 - p) ** (n - f) for f in range(n + 1)]
    unbiased = sum(
        mass * worst_of_k_estimate([f], [n], [1.0], k=k) for f, mass in enumerate(pmf)
    )
    plug_in = sum(mass * worst_of_k_risk([f / n], [1.0], k=k) for f, mass in enumerate(pmf))
    assert unbiased == pytest.approx(truth, rel=1e-12, abs=0)
    # 1 - (1 - p) ** k is concave in p, so by Jensen's inequality the plug-in is too low.
    assert plug_in < truth


def test_the_worst_of_k_estimate_counts_subsets_with_a_failure() -> None:
    # One prompt with 10 failures in 20 rollouts, one with none: C(10, 2) / C(20, 2) of the
    # pairs on the first are failure-free.
    estimate = worst_of_k_estimate([10, 0], [20, 20], [0.5, 0.5], k=2)
    assert estimate == pytest.approx(0.5 * (1 - 45 / 190), rel=1e-15, abs=0)
    assert estimate > worst_of_k_risk([0.5, 0.0], [0.5, 0.5], k=2)
    # Fewer safe rollouts than attempts: every k-subset holds a failure.
    assert worst_of_k_estimate([4], [5], [1.0], k=2) == 1.0
    zero = worst_of_k_estimate([0, 0], [7, 9], [0.5, 0.5], k=3)
    assert zero == 0.0 and math.copysign(1.0, zero) == 1.0
    # At one attempt it is exactly the weighted failure rate.
    assert worst_of_k_estimate([3, 1], [10, 4], [0.5, 0.5], k=1) == 0.5 * 0.3 + 0.5 * 0.25


def test_the_worst_of_k_estimate_stays_accurate_for_tiny_failure_rates() -> None:
    # One failure in a million rollouts, two attempts: exactly 2 / n.
    n = 1_000_000
    assert worst_of_k_estimate([1], [n], [1.0], k=2) == pytest.approx(2 / n, rel=1e-12, abs=0)


def test_the_worst_of_k_estimate_refuses_what_it_cannot_estimate() -> None:
    with pytest.raises(ValueError, match="at least 3 rollouts"):
        worst_of_k_estimate([0], [2], [1.0], k=3)
    with pytest.raises(ValueError, match="between zero and the number of rollouts"):
        worst_of_k_estimate([5], [4], [1.0], k=2)
    with pytest.raises(ValueError, match="integer arrays"):
        worst_of_k_estimate([0.5], [4], [1.0], k=2)
    with pytest.raises(ValueError, match="integer of at least one"):
        worst_of_k_estimate([0], [4], [1.0], k=True)


def test_plans_reject_a_boolean_k() -> None:
    for measure in ("worst_of_k", "top_k_mean"):
        with pytest.raises(ValidationError, match="not a boolean"):
            RiskMeasureSpec.model_validate({"id": "m", "type": measure, "k": True})
        assert RiskMeasureSpec.model_validate({"id": "m", "type": measure, "k": 2}).k == 2
