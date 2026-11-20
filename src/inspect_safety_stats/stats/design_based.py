from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from inspect_safety_stats.config.methods import DEFAULT_STATS_FUNCTION_METHOD, DirectMeanMethod
from inspect_safety_stats.stats.direct_mean import DIRECT_MEAN_ENGINES

DesignBasedMethod = DirectMeanMethod


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
    method: DesignBasedMethod = DEFAULT_STATS_FUNCTION_METHOD,
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

    engines = DIRECT_MEAN_ENGINES.get(method)
    if engines is None:
        raise ValueError(f"unsupported design-based method {method!r}")
    engine = engines.bounded
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


@dataclass(frozen=True)
class FixedSizeDesign:
    """Stratified fixed-size sampling: exactly ``draws[h]`` of stratum ``h``'s frame items.

    A single stratum is simple random sampling without replacement over the whole frame.
    Inclusion indicators are negatively correlated rather than independent, so this design
    is handled by a different reduction from :class:`InclusionDesign`.

    ``frame_weights`` are the target's weights over the whole frame, zero for frame items
    outside the target, and summing to one.
    """

    stratum_index: np.ndarray
    frame_weights: np.ndarray
    draws: np.ndarray
    sampled: np.ndarray

    def __post_init__(self) -> None:
        strata = np.asarray(self.stratum_index, dtype=int)
        weights = np.asarray(self.frame_weights, dtype=float)
        draws = np.asarray(self.draws, dtype=int)
        sampled = np.asarray(self.sampled, dtype=bool)
        if weights.ndim != 1 or len(weights) == 0:
            raise ValueError("frame weights must be a nonempty one-dimensional array")
        if strata.shape != weights.shape or sampled.shape != weights.shape:
            raise ValueError("stratum index, frame weights, and mask must be aligned")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("frame weights must be finite and nonnegative")
        if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("frame weights must sum to one over the target")
        if draws.ndim != 1 or len(draws) == 0:
            raise ValueError("draws must be a nonempty one-dimensional array")
        if np.any(strata < 0) or np.any(strata >= len(draws)):
            raise ValueError("every stratum index must address a declared stratum")
        if np.any(draws < 0):
            raise ValueError("planned draws must be nonnegative")
        sizes = np.bincount(strata, minlength=len(draws))
        if np.any(draws > sizes):
            raise ValueError("a stratum cannot draw more items than it contains")
        realized = np.bincount(strata[sampled], minlength=len(draws))
        if not np.array_equal(realized, draws):
            raise ValueError(
                "a fixed-size design must realize exactly its planned draws per stratum; "
                f"planned {draws.tolist()} but observed {realized.tolist()}"
            )
        if not bool(np.any(sampled)):
            raise ValueError("at least one frame item must be sampled")

    @property
    def frame_size(self) -> int:
        return len(self.frame_weights)

    @property
    def sampled_count(self) -> int:
        return int(np.count_nonzero(self.sampled))


def fixed_size_mean_bounds(
    design: FixedSizeDesign,
    completion_lower: ArrayLike,
    completion_upper: ArrayLike,
    *,
    alpha_lower: float,
    alpha_upper: float,
    method: DesignBasedMethod = DEFAULT_STATS_FUNCTION_METHOD,
) -> DesignBasedBounds:
    """Bound a frame mean estimated by stratified sampling without replacement.

    Within stratum ``h`` of frame size ``N_h``, the target's contribution is estimated by

        (N_h / n_h) * sum over drawn items of a_j * phat_j,

    which is ``N_h A_h`` times a without-replacement sample mean of values
    ``a_j phat_j / A_h`` in ``[0, 1]``, where ``A_h`` is the largest target weight in the
    stratum. Each drawn item therefore carries the deterministic coefficient
    ``N_h A_h / n_h``, and those coefficients sum to ``K = sum_h N_h A_h`` regardless of
    which items were drawn. Normalising by ``K`` gives an ordinary bounded weighted mean.

    Validity rests on Hoeffding's reduction (1963, Theorem 4): a convex function of a
    without-replacement sum is dominated in expectation by the same function of a
    with-replacement sum. The Hoeffding, empirical-Bernstein and KL/Chernoff engines bound
    a moment generating function, which is convex. The experimental betting engine bounds
    none; it bounds the capital of each fixed betting fraction, a product over the drawn
    items of wealth factors raised to a common exponent ``c <= 1`` within a stratum.
    Averaged over the item outcomes, independent across items and of the draw, it is at
    most ``exp(c * sum_j log E f_j)`` by Jensen's inequality, a convex function of a
    without-replacement sum of fixed numbers, so the reduction applies to it and to the
    fixed mixture over fractions. The bound each engine computes under an independence
    assumption therefore remains valid -- and is conservative, since sampling without
    replacement has strictly smaller variance.
    """

    engines = DIRECT_MEAN_ENGINES.get(method)
    if engines is None:
        raise ValueError(f"unsupported design-based method {method!r}")
    engine = engines.bounded
    if not 0 < alpha_lower < 1 or not 0 < alpha_upper < 1:
        raise ValueError("alpha levels must be in (0, 1)")

    lower_rates = np.asarray(completion_lower, dtype=float)
    upper_rates = np.asarray(completion_upper, dtype=float)
    weights = np.asarray(design.frame_weights, dtype=float)
    strata = np.asarray(design.stratum_index, dtype=int)
    draws = np.asarray(design.draws, dtype=int)
    sampled = np.asarray(design.sampled, dtype=bool)
    if lower_rates.shape != weights.shape or upper_rates.shape != weights.shape:
        raise ValueError("observed rates must align with the sampling frame")
    if not np.all(np.isfinite(lower_rates)) or not np.all(np.isfinite(upper_rates)):
        raise ValueError("observed rates must be finite")
    if np.any(lower_rates < 0) or np.any(upper_rates > 1) or np.any(lower_rates > upper_rates):
        raise ValueError("observed rates must satisfy 0 <= lower <= upper <= 1")
    if np.any(lower_rates[~sampled] != 0) or np.any(upper_rates[~sampled] != 0):
        raise ValueError("unsampled frame items must carry a zero observed rate")

    sizes = np.bincount(strata, minlength=len(draws))
    peak = np.zeros(len(draws), dtype=float)
    np.maximum.at(peak, strata, weights)

    coefficients: list[float] = []
    values_lower: list[float] = []
    values_upper: list[float] = []
    for index in np.flatnonzero(sampled):
        stratum = int(strata[index])
        if peak[stratum] <= 0:
            # No target member in this stratum, so it contributes nothing.
            continue
        coefficients.append(float(sizes[stratum]) * peak[stratum] / float(draws[stratum]))
        values_lower.append(float(weights[index] * lower_rates[index] / peak[stratum]))
        values_upper.append(float(weights[index] * upper_rates[index] / peak[stratum]))
    if not coefficients:
        raise ValueError("no sampled frame item carries positive target weight")

    coefficient_array = np.asarray(coefficients, dtype=float)
    inflation = float(np.sum(coefficient_array))
    normalized = coefficient_array / inflation
    bounds = engine(
        np.asarray(values_lower, dtype=float),
        np.asarray(values_upper, dtype=float),
        normalized,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_profile="frame_item",
    )
    return DesignBasedBounds(
        estimate=float(np.dot(coefficient_array, values_lower)),
        lower=max(0.0, min(1.0, inflation * bounds.lower)),
        upper=min(1.0, inflation * bounds.upper),
        completion_estimate_lower=inflation * bounds.completion_estimate_lower,
        completion_estimate_upper=inflation * bounds.completion_estimate_upper,
        inflation=inflation,
        maximum_weight=float(np.max(normalized)),
        effective_sample_size=bounds.effective_sample_size,
        frame_size=design.frame_size,
        sampled_count=len(coefficients),
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        method=method,
        estimator="stratified_without_replacement",
    )
