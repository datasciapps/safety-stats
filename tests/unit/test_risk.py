import math

import numpy as np
import pytest

from inspect_safety_stats.stats import (
    entropic_risk,
    maximum_risk,
    threshold_exceedance,
    threshold_excess,
    top_k_mean,
    weighted_cvar,
    weighted_mean,
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
