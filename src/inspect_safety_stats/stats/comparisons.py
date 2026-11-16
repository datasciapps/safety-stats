from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from inspect_safety_stats.stats.direct_mean import (
    DirectMeanBounds,
    weighted_empirical_bernstein_mean_bounds,
    weighted_hoeffding_mean_bounds,
    weighted_kl_chernoff_mean_bounds,
)

ComparisonMethod = Literal[
    "weighted_hoeffding",
    "weighted_empirical_bernstein",
    "weighted_kl_chernoff",
]
ComparisonIndependenceUnit = Literal["rollout", "prompt_cluster"]


@dataclass(frozen=True)
class DirectComparisonBounds:
    lower: float
    upper: float
    completion_estimate_lower: float
    completion_estimate_upper: float
    half_width_lower: float
    half_width_upper: float
    concentration_variance: float
    effective_sample_size: float
    alpha_lower: float
    alpha_upper: float
    method: str
    independence_unit: str
    variance_upper_lower: float | None = None
    variance_upper_upper: float | None = None
    maximum_increment: float | None = None
    chernoff_kernel: str | None = None
    candidate_lower: float | None = None
    candidate_upper: float | None = None
    baseline_lower: float | None = None
    baseline_upper: float | None = None


@dataclass(frozen=True)
class RatioComparisonBounds:
    candidate_estimate: float
    baseline_estimate: float
    estimate: float | None
    lower: float | None
    upper: float | None
    candidate_bounds: DirectMeanBounds
    baseline_bounds: DirectMeanBounds
    method: str
    independence_unit: str
    estimand: Literal["risk_ratio", "relative_risk_reduction"]
    warning: str | None = None


def _arrays(
    candidate_failures: ArrayLike,
    candidate_unclassified: ArrayLike,
    candidate_trials: ArrayLike,
    baseline_failures: ArrayLike,
    baseline_unclassified: ArrayLike,
    baseline_trials: ArrayLike,
    weights: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = [
        np.asarray(candidate_failures),
        np.asarray(candidate_unclassified),
        np.asarray(candidate_trials),
        np.asarray(baseline_failures),
        np.asarray(baseline_unclassified),
        np.asarray(baseline_trials),
    ]
    w = np.asarray(weights, dtype=float)
    if any(array.ndim != 1 for array in [*arrays, w]):
        raise ValueError("comparison counts and weights must be one-dimensional")
    if len(w) == 0 or any(len(array) != len(w) for array in arrays):
        raise ValueError("comparison counts and weights must be nonempty and aligned")
    if not all(np.issubdtype(array.dtype, np.integer) for array in arrays):
        raise TypeError("comparison counts must be integers")
    cf, cu, cn, bf, bu, bn = arrays
    if np.any(cn <= 0) or np.any(bn <= 0):
        raise ValueError("every comparison arm must have positive attempted rollout counts")
    if (
        np.any(cf < 0)
        or np.any(cu < 0)
        or np.any(cf + cu > cn)
        or np.any(bf < 0)
        or np.any(bu < 0)
        or np.any(bf + bu > bn)
    ):
        raise ValueError("comparison counts must satisfy failure + unclassified <= trials")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("comparison weights must be finite and nonnegative")
    if not math.isclose(float(np.sum(w)), 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("comparison weights must sum to one")
    return cf, cu, cn, bf, bu, bn, w


def _direct_function(method: ComparisonMethod) -> Callable[..., DirectMeanBounds]:
    return {
        "weighted_hoeffding": weighted_hoeffding_mean_bounds,
        "weighted_empirical_bernstein": weighted_empirical_bernstein_mean_bounds,
        "weighted_kl_chernoff": weighted_kl_chernoff_mean_bounds,
    }[method]


def _variance_centers(method: ComparisonMethod, centers: np.ndarray) -> dict[str, np.ndarray]:
    """Variance centers for the engines that estimate a variance proxy."""

    return {"variance_centers": centers} if method == "weighted_empirical_bernstein" else {}


def _transform_rollout_bounds(bounds: DirectMeanBounds) -> DirectComparisonBounds:
    return DirectComparisonBounds(
        lower=max(-1.0, 2 * bounds.lower - 1),
        upper=min(1.0, 2 * bounds.upper - 1),
        completion_estimate_lower=2 * bounds.completion_estimate_lower - 1,
        completion_estimate_upper=2 * bounds.completion_estimate_upper - 1,
        half_width_lower=2 * bounds.half_width_lower,
        half_width_upper=2 * bounds.half_width_upper,
        concentration_variance=4 * bounds.concentration_variance,
        effective_sample_size=bounds.effective_sample_size,
        alpha_lower=bounds.alpha_lower,
        alpha_upper=bounds.alpha_upper,
        method=bounds.method,
        independence_unit=bounds.independence_unit,
        variance_upper_lower=(
            4 * bounds.variance_upper_lower if bounds.variance_upper_lower is not None else None
        ),
        variance_upper_upper=(
            4 * bounds.variance_upper_upper if bounds.variance_upper_upper is not None else None
        ),
        maximum_increment=(
            2 * bounds.maximum_increment if bounds.maximum_increment is not None else None
        ),
        chernoff_kernel=bounds.chernoff_kernel,
    )


def direct_risk_difference_bounds(
    candidate_failures: ArrayLike,
    candidate_unclassified: ArrayLike,
    candidate_trials: ArrayLike,
    baseline_failures: ArrayLike,
    baseline_unclassified: ArrayLike,
    baseline_trials: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    method: ComparisonMethod = "weighted_empirical_bernstein",
    independence_unit: ComparisonIndependenceUnit = "rollout",
) -> DirectComparisonBounds:
    """Bound a fixed-design weighted candidate-minus-baseline failure risk.

    Candidate and baseline rollout counts may differ. Unknown outcomes receive
    adverse arm-specific completions. Rollout inference assumes mutually
    independent attempted generations. Prompt-cluster inference permits arbitrary
    dependence within and across the two arms of each prompt batch.
    """
    if not 0 < alpha_lower < 1 or not 0 < alpha_upper < 1:
        raise ValueError("comparison alpha levels must be in (0, 1)")
    cf, cu, cn, bf, bu, bn, w = _arrays(
        candidate_failures,
        candidate_unclassified,
        candidate_trials,
        baseline_failures,
        baseline_unclassified,
        baseline_trials,
        weights,
    )
    if method == "weighted_kl_chernoff":
        # Sparse safety failures are poorly served by applying Bernoulli-KL to
        # Q=(1+Delta)/2, whose mean is usually near one half. Build one joint
        # two-arm rectangle instead. The two arm endpoints required by each
        # directional contrast endpoint split its local error budget.
        function = _direct_function(method)
        candidate_bounds = function(
            cf,
            cu,
            cn,
            w,
            alpha_lower=alpha_lower / 2,
            alpha_upper=alpha_upper / 2,
            independence_unit=independence_unit,
        )
        baseline_bounds = function(
            bf,
            bu,
            bn,
            w,
            alpha_lower=alpha_upper / 2,
            alpha_upper=alpha_lower / 2,
            independence_unit=independence_unit,
        )
        lower = max(-1.0, candidate_bounds.lower - baseline_bounds.upper)
        upper = min(1.0, candidate_bounds.upper - baseline_bounds.lower)
        completion_lower = (
            candidate_bounds.completion_estimate_lower - baseline_bounds.completion_estimate_upper
        )
        completion_upper = (
            candidate_bounds.completion_estimate_upper - baseline_bounds.completion_estimate_lower
        )
        return DirectComparisonBounds(
            lower=lower,
            upper=upper,
            completion_estimate_lower=completion_lower,
            completion_estimate_upper=completion_upper,
            half_width_lower=max(0.0, completion_lower - lower),
            half_width_upper=max(0.0, upper - completion_upper),
            concentration_variance=(
                candidate_bounds.concentration_variance + baseline_bounds.concentration_variance
            ),
            effective_sample_size=min(
                candidate_bounds.effective_sample_size,
                baseline_bounds.effective_sample_size,
            ),
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            method=method,
            independence_unit=independence_unit,
            maximum_increment=max(
                candidate_bounds.maximum_increment or 0.0,
                baseline_bounds.maximum_increment or 0.0,
            ),
            chernoff_kernel=(
                candidate_bounds.chernoff_kernel
                if candidate_bounds.chernoff_kernel == baseline_bounds.chernoff_kernel
                else "two_arm_mixed"
            ),
            candidate_lower=candidate_bounds.lower,
            candidate_upper=candidate_bounds.upper,
            baseline_lower=baseline_bounds.lower,
            baseline_upper=baseline_bounds.upper,
        )
    if independence_unit == "rollout":
        # q = (1 + candidate risk - baseline risk) / 2. Baseline safe
        # outcomes are q-successes. Leaving both arms' unknowns unclassified
        # produces the adverse endpoint completions after inversion.
        q_failures = np.concatenate([cf, bn - bf - bu])
        q_unclassified = np.concatenate([cu, bu])
        q_trials = np.concatenate([cn, bn])
        q_weights = np.concatenate([w / 2, w / 2])
        # Candidate q-observations are failures and baseline ones are safe outcomes, so a
        # common center sits at the wrong end for one arm. The variance proxy is centered by
        # arm instead, with all four arm-wise choices of 0 and 1: candidate 0 and baseline 1
        # counts both arms' failures and serves two sparse arms, and the others serve arms
        # that fail often. Centers 0/0 and 1/1 are the uncentered proxies.
        zeros, ones = np.zeros_like(w), np.ones_like(w)
        arm_centerings = np.stack(
            [
                np.concatenate([candidate, baseline])
                for candidate in (zeros, ones)
                for baseline in (zeros, ones)
            ]
        )
        bounds = _direct_function(method)(
            q_failures,
            q_unclassified,
            q_trials,
            q_weights,
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            independence_unit="rollout",
            **_variance_centers(method, arm_centerings),
        )
        return _transform_rollout_bounds(bounds)

    # Encode q_j=(1+p_candidate,j-p_baseline,j)/2 exactly as a bounded
    # rational observation. This lets all three direct-mean engines share the
    # same prompt-cluster implementation, including KL/Chernoff.
    q_trials = 2 * cn * bn
    q_failures = cn * bn + cf * bn - (bf + bu) * cn
    q_unclassified = cu * bn + bu * cn
    # Each q_j is 1/2 plus half the arms' rate difference, so the proxy is centered at 1/2.
    # The uncentered proxies 0 and 1 are kept for arms far apart, where q_j nears 0 or 1.
    bounds = _direct_function(method)(
        q_failures,
        q_unclassified,
        q_trials,
        w,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_unit="prompt_cluster",
        **_variance_centers(
            method, np.stack([np.full_like(w, 0.5), np.zeros_like(w), np.ones_like(w)])
        ),
    )
    return _transform_rollout_bounds(bounds)


def direct_ratio_bounds(
    candidate_failures: ArrayLike,
    candidate_unclassified: ArrayLike,
    candidate_trials: ArrayLike,
    baseline_failures: ArrayLike,
    baseline_unclassified: ArrayLike,
    baseline_trials: ArrayLike,
    weights: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    method: ComparisonMethod = "weighted_empirical_bernstein",
    independence_unit: ComparisonIndependenceUnit = "rollout",
    estimand: Literal["risk_ratio", "relative_risk_reduction"] = "risk_ratio",
) -> RatioComparisonBounds:
    """Conservative ratio inference from a simultaneous rectangle for both arms."""
    if estimand == "relative_risk_reduction":
        ratio = direct_ratio_bounds(
            candidate_failures,
            candidate_unclassified,
            candidate_trials,
            baseline_failures,
            baseline_unclassified,
            baseline_trials,
            weights,
            alpha_lower=alpha_upper,
            alpha_upper=alpha_lower,
            method=method,
            independence_unit=independence_unit,
            estimand="risk_ratio",
        )
        return RatioComparisonBounds(
            candidate_estimate=ratio.candidate_estimate,
            baseline_estimate=ratio.baseline_estimate,
            estimate=1 - ratio.estimate if ratio.estimate is not None else None,
            lower=1 - ratio.upper if ratio.upper is not None else None,
            upper=1 - ratio.lower if ratio.lower is not None else None,
            candidate_bounds=ratio.candidate_bounds,
            baseline_bounds=ratio.baseline_bounds,
            method=method,
            independence_unit=independence_unit,
            estimand=estimand,
            warning=ratio.warning,
        )
    cf, cu, cn, bf, bu, bn, w = _arrays(
        candidate_failures,
        candidate_unclassified,
        candidate_trials,
        baseline_failures,
        baseline_unclassified,
        baseline_trials,
        weights,
    )
    function = _direct_function(method)
    # Each directional ratio endpoint uses one endpoint from each arm. Split
    # its local error budget across those endpoints without assuming arm independence.
    candidate_bounds = function(
        cf,
        cu,
        cn,
        w,
        alpha_lower=alpha_lower / 2,
        alpha_upper=alpha_upper / 2,
        independence_unit=independence_unit,
    )
    baseline_bounds = function(
        bf,
        bu,
        bn,
        w,
        alpha_lower=alpha_upper / 2,
        alpha_upper=alpha_lower / 2,
        independence_unit=independence_unit,
    )
    candidate_estimate = float(np.dot(w, cf / cn))
    baseline_estimate = float(np.dot(w, bf / bn))
    estimate = candidate_estimate / baseline_estimate if baseline_estimate > 0 else None
    lower_ratio = (
        candidate_bounds.lower / baseline_bounds.upper if baseline_bounds.upper > 0 else None
    )
    upper_ratio = (
        candidate_bounds.upper / baseline_bounds.lower if baseline_bounds.lower > 0 else None
    )
    warning = None
    if estimate is None:
        warning = "The observed baseline risk is zero, so the ratio estimate is undefined."
    if upper_ratio is None:
        warning = (
            "The baseline lower confidence bound is zero, so no finite upper ratio bound exists."
        )
    return RatioComparisonBounds(
        candidate_estimate=candidate_estimate,
        baseline_estimate=baseline_estimate,
        estimate=estimate,
        lower=lower_ratio,
        upper=upper_ratio,
        candidate_bounds=candidate_bounds,
        baseline_bounds=baseline_bounds,
        method=method,
        independence_unit=independence_unit,
        estimand=estimand,
        warning=warning,
    )
