from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from inspect_safety_stats.stats.direct_mean import (
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
)

DesignBasedMethod = Literal[
    "weighted_hoeffding",
    "weighted_empirical_bernstein",
    "weighted_kl_chernoff",
]

_ENGINES = {
    "weighted_hoeffding": weighted_bounded_hoeffding_mean_bounds,
    "weighted_empirical_bernstein": weighted_bounded_empirical_bernstein_mean_bounds,
    "weighted_kl_chernoff": weighted_bounded_kl_chernoff_mean_bounds,
}


@dataclass(frozen=True)
class InclusionDesign:
    """A declared sampling frame with known independent inclusion probabilities.

    ``design_weights`` are the policy weights over the **whole frame** and must sum to
    one; they are not renormalized to the realized sample. ``inclusion_probabilities``
    must be positive for every frame item, since an item that can never be drawn leaves
    its share of the estimand unidentified.
    """

    design_weights: np.ndarray
    inclusion_probabilities: np.ndarray
    sampled: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.design_weights, dtype=float)
        probabilities = np.asarray(self.inclusion_probabilities, dtype=float)
        sampled = np.asarray(self.sampled, dtype=bool)
        if weights.ndim != 1 or len(weights) == 0:
            raise ValueError("design weights must be a nonempty one-dimensional array")
        if probabilities.shape != weights.shape or sampled.shape != weights.shape:
            raise ValueError("weights, inclusion probabilities, and mask must be aligned")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("design weights must be finite and nonnegative")
        if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("design weights must sum to one over the whole frame")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("inclusion probabilities must be finite")
        if np.any(probabilities <= 0) or np.any(probabilities > 1):
            raise ValueError("every inclusion probability must lie in (0, 1]")
        if not bool(np.any(sampled)):
            raise ValueError("at least one frame item must be sampled")

    @property
    def frame_size(self) -> int:
        return len(self.design_weights)

    @property
    def sampled_count(self) -> int:
        return int(np.count_nonzero(self.sampled))


@dataclass(frozen=True)
class DesignBasedBounds:
    """Finite-sample bounds for a frame mean estimated from a probability sample."""

    estimate: float
    lower: float
    upper: float
    completion_estimate_lower: float
    completion_estimate_upper: float
    inflation: float
    maximum_weight: float
    effective_sample_size: float
    frame_size: int
    sampled_count: int
    alpha_lower: float
    alpha_upper: float
    method: str
    estimator: str = "horvitz_thompson"


def horvitz_thompson_mean_bounds(
    design: InclusionDesign,
    completion_lower: ArrayLike,
    completion_upper: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    method: DesignBasedMethod = "weighted_empirical_bernstein",
) -> DesignBasedBounds:
    """Bound a frame-wide weighted mean from an item sample with known inclusion probabilities.

    For frame item ``j`` with design weight ``w_j``, inclusion probability ``pi_j``, and
    inclusion indicator ``I_j``, the Horvitz-Thompson estimator of ``R = sum_j w_j p_j`` is

        Rhat = sum_j I_j (w_j / pi_j) phat_j = sum_j c_j Y_j,

    with ``c_j = w_j / pi_j`` deterministic and ``Y_j = I_j phat_j`` bounded in ``[0, 1]``.
    Dividing by ``C = sum_j c_j`` makes this an ordinary bounded weighted mean with weights
    summing to one, so the existing concentration engines apply unchanged and the resulting
    bound on ``R / C`` is rescaled by ``C``.

    Two consequences are reported rather than hidden. ``inflation`` is ``C``, the factor by
    which inverse-probability weighting widens the interval; it is at least one and equals
    one only under a census. ``maximum_weight`` is the largest normalized coefficient, which
    governs how much any single item can dominate the bound.

    The construction assumes inclusion is **independent across frame items** -- Poisson
    sampling -- and that the sample was drawn without reference to the outcomes. Fixed-size
    designs induce negatively correlated indicators and are not covered here. Unsampled items
    contribute an exact zero; their share of the estimand is carried by the sampling
    randomness in the bound, not by outcome completion.
    """

    engine = _ENGINES.get(method)
    if engine is None:
        raise ValueError(f"unsupported design-based method {method!r}")
    if not 0 < alpha_lower < 1 or not 0 < alpha_upper < 1:
        raise ValueError("alpha levels must be in (0, 1)")

    lower_rates = np.asarray(completion_lower, dtype=float)
    upper_rates = np.asarray(completion_upper, dtype=float)
    weights = np.asarray(design.design_weights, dtype=float)
    probabilities = np.asarray(design.inclusion_probabilities, dtype=float)
    sampled = np.asarray(design.sampled, dtype=bool)
    if lower_rates.shape != weights.shape or upper_rates.shape != weights.shape:
        raise ValueError("observed rates must align with the sampling frame")
    if not np.all(np.isfinite(lower_rates)) or not np.all(np.isfinite(upper_rates)):
        raise ValueError("observed rates must be finite")
    if np.any(lower_rates < 0) or np.any(upper_rates > 1) or np.any(lower_rates > upper_rates):
        raise ValueError("observed rates must satisfy 0 <= lower <= upper <= 1")
    if np.any(lower_rates[~sampled] != 0) or np.any(upper_rates[~sampled] != 0):
        raise ValueError("unsampled frame items must carry a zero observed rate")

    coefficients = weights / probabilities
    inflation = float(np.sum(coefficients))
    normalized = coefficients / inflation
    bounds = engine(
        np.where(sampled, lower_rates, 0.0),
        np.where(sampled, upper_rates, 0.0),
        normalized,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_profile="frame_item",
    )
    return DesignBasedBounds(
        estimate=float(np.dot(coefficients, np.where(sampled, lower_rates, 0.0))),
        lower=max(0.0, min(1.0, inflation * bounds.lower)),
        upper=min(1.0, inflation * bounds.upper),
        completion_estimate_lower=inflation * bounds.completion_estimate_lower,
        completion_estimate_upper=inflation * bounds.completion_estimate_upper,
        inflation=inflation,
        maximum_weight=float(np.max(normalized)),
        effective_sample_size=bounds.effective_sample_size,
        frame_size=design.frame_size,
        sampled_count=design.sampled_count,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        method=method,
    )
