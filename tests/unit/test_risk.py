import math

import numpy as np
import pytest

from inspect_safety_stats.config.models import RiskMeasureSpec
from inspect_safety_stats.stats import (
    entropic_risk,
    maximum_risk,
    threshold_exceedance,
    threshold_excess,
    top_k_mean,
    weighted_cvar,
    weighted_mean,
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
