from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class JointMaxTResult:
    lower: np.ndarray
    upper: np.ndarray
    standard_error: np.ndarray
    critical_value_lower: float
    critical_value_upper: float
    target_count: int
    replicates: int
    variance_regularized: np.ndarray


@dataclass(frozen=True)
class JointMaxTUpperResult:
    upper: np.ndarray
    standard_error: np.ndarray
    critical_value: float
    target_count: int
    replicates: int
    variance_regularized: np.ndarray


def jeffreys_binomial_probabilities(failures: ArrayLike, trials: ArrayLike) -> np.ndarray:
    """Boundary-regularized probabilities for parametric Bernoulli resampling."""

    x = np.asarray(failures, dtype=float)
    n = np.asarray(trials, dtype=float)
    if x.ndim != 1 or n.ndim != 1 or len(x) == 0 or len(x) != len(n):
        raise ValueError("failures and trials must be nonempty aligned vectors")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(n)):
        raise ValueError("failures and trials must be finite")
    if np.any(n <= 0) or np.any(x < 0) or np.any(x > n):
        raise ValueError("binomial counts must satisfy 0 <= failures <= trials")
    return (x + 0.5) / (n + 1.0)


def stratified_bernoulli_draws(
    failures: ArrayLike,
    trials: ArrayLike,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Regenerate rollout rates while retaining every fixed prompt stratum."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    n = np.asarray(trials, dtype=np.int64)
    probabilities = jeffreys_binomial_probabilities(failures, n)
    rng = np.random.default_rng(seed)
    return rng.binomial(n=n, p=probabilities, size=(replicates, len(n))) / n


def joint_max_t_intervals(
    estimates: ArrayLike,
    bootstrap_estimates: ArrayLike,
    *,
    alpha: float,
    coverage_mode: Literal["two_sided", "per_direction"] = "two_sided",
) -> JointMaxTResult:
    """Calibrate studentized intervals from a shared tree-wide bootstrap."""

    observed = np.asarray(estimates, dtype=float)
    draws = np.asarray(bootstrap_estimates, dtype=float)
    if observed.ndim != 1 or len(observed) == 0:
        raise ValueError("estimates must be a nonempty vector")
    if draws.ndim != 2 or draws.shape[1] != len(observed) or draws.shape[0] < 2:
        raise ValueError("bootstrap estimates must be replicates by aligned targets")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(draws)):
        raise ValueError("joint bootstrap inputs must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    centers = np.mean(draws, axis=0)
    raw_se = np.std(draws, axis=0, ddof=1)
    floor = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(centers))
    regularized = raw_se <= floor
    standard_error = np.maximum(raw_se, floor)
    standardized = (draws - centers) / standard_error

    if coverage_mode == "two_sided":
        maxima = np.max(np.abs(standardized), axis=1)
        critical = float(np.quantile(maxima, 1 - alpha, method="higher"))
        critical_lower = critical_upper = critical
    else:
        lower_maxima = np.max(-standardized, axis=1)
        upper_maxima = np.max(standardized, axis=1)
        critical_lower = float(np.quantile(lower_maxima, 1 - alpha, method="higher"))
        critical_upper = float(np.quantile(upper_maxima, 1 - alpha, method="higher"))

    return JointMaxTResult(
        lower=observed - critical_lower * standard_error,
        upper=observed + critical_upper * standard_error,
        standard_error=standard_error,
        critical_value_lower=critical_lower,
        critical_value_upper=critical_upper,
        target_count=len(observed),
        replicates=len(draws),
        variance_regularized=regularized,
    )


def joint_max_t_upper_bounds(
    estimates: ArrayLike,
    bootstrap_estimates: ArrayLike,
    *,
    alpha: float,
) -> JointMaxTUpperResult:
    """Calibrate simultaneous one-sided upper bounds with a shared max-T statistic."""

    observed = np.asarray(estimates, dtype=float)
    draws = np.asarray(bootstrap_estimates, dtype=float)
    if observed.ndim != 1 or len(observed) == 0:
        raise ValueError("estimates must be a nonempty vector")
    if draws.ndim != 2 or draws.shape[1] != len(observed) or draws.shape[0] < 2:
        raise ValueError("bootstrap estimates must be replicates by aligned targets")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(draws)):
        raise ValueError("joint bootstrap inputs must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    centers = np.mean(draws, axis=0)
    raw_se = np.std(draws, axis=0, ddof=1)
    floor = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(centers))
    regularized = raw_se <= floor
    standard_error = np.maximum(raw_se, floor)
    standardized = (draws - centers) / standard_error
    maxima = np.max(standardized, axis=1)
    critical = float(np.quantile(maxima, 1 - alpha, method="higher"))

    return JointMaxTUpperResult(
        upper=observed + critical * standard_error,
        standard_error=standard_error,
        critical_value=critical,
        target_count=len(observed),
        replicates=len(draws),
        variance_regularized=regularized,
    )
