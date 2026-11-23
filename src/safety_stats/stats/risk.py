from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _arrays(
    values: ArrayLike, weights: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.ndim != 1 or len(x) != len(w) or len(x) == 0:
        raise ValueError(
            "values and weights must be nonempty one-dimensional arrays of equal length"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ValueError("values and weights must be finite")
    if np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be nonnegative with positive total")
    total = float(np.sum(w))
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("weights must sum to one")
    return x, w


def weighted_mean(values: ArrayLike, weights: ArrayLike) -> float:
    """Weighted mean of per-item values, the neutral baseline aggregate.

    ``values`` and ``weights`` are nonempty one-dimensional arrays of equal length. The weights
    must be finite, nonnegative and sum to one. Returns ``sum(w * x)``. Derived in
    docs/statistical-methods.md, section 2.4.1.
    """

    x, w = _arrays(values, weights)
    return float(np.dot(x, w))


def maximum_risk(values: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Worst-item risk: the largest of the per-item values.

    ``weights`` does not change the result. When given, it is only checked like the weights of
    ``weighted_mean`` (same length, nonnegative, summing to one), so that every measure rejects
    the same invalid designs. Derived in docs/statistical-methods.md, section 2.4.2.
    """

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("values must be a nonempty finite one-dimensional array")
    if weights is not None:
        _arrays(x, weights)
    return float(np.max(x))


def weighted_cvar(values: ArrayLike, weights: ArrayLike, *, beta: float) -> float:
    """Finite-design CVaR: the weighted mean of the worst ``1 - beta`` share of the design.

    Items are taken from the largest value down until their weights fill ``1 - beta``, with a
    fractional weight at the boundary. ``beta`` lies in ``[0, 1)``: zero gives the weighted
    mean, and the result approaches the maximum as ``beta`` nears one. Weights follow
    ``weighted_mean``. Derived in docs/statistical-methods.md, section 2.4.3.
    """

    x, w = _arrays(values, weights)
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    remaining = 1.0 - beta
    tail = remaining
    total = 0.0
    for idx in np.argsort(-x, kind="stable"):
        take = min(float(w[idx]), remaining)
        total += take * float(x[idx])
        remaining -= take
        if remaining <= 1e-15:
            break
    return total / tail


def top_k_mean(values: ArrayLike, *, k: int) -> float:
    """Unweighted mean of the ``k`` largest values.

    ``k`` is an integer from one to the number of values. With equal weights and
    ``1 - beta = k / len(values)`` it equals ``weighted_cvar``. Described in
    docs/statistical-methods.md, section 2.4.3.
    """

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("values must be a nonempty finite one-dimensional array")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(x):
        raise ValueError("k must be an integer between one and the number of values")
    partition = np.partition(x, len(x) - k)[-k:]
    return float(np.mean(partition))


def entropic_risk(values: ArrayLike, weights: ArrayLike, *, lambda_: float) -> float:
    """Entropic risk ``log(sum(w * exp(lambda_ * x))) / lambda_`` of the per-item values.

    ``lambda_`` is the risk-aversion parameter, positive and finite. The result tends to the
    weighted mean as ``lambda_`` falls to zero, and is returned as exactly the weighted mean
    below ``1e-7``. It tends to the maximum over positively weighted items as ``lambda_``
    grows. Items with zero weight are ignored. Weights follow ``weighted_mean``. Derived in
    docs/statistical-methods.md, section 2.4.4.
    """

    x, w = _arrays(values, weights)
    if not math.isfinite(lambda_) or lambda_ <= 0:
        raise ValueError("lambda_ must be positive and finite")
    support = w > 0
    xs = x[support]
    ws = w[support]
    if lambda_ < 1e-7:
        return float(np.dot(xs, ws))
    maximum = float(np.max(xs))
    return maximum + math.log(float(np.dot(ws, np.exp(lambda_ * (xs - maximum))))) / lambda_


def threshold_exceedance(
    values: ArrayLike, weights: ArrayLike, *, threshold: float | ArrayLike
) -> float:
    """Total weight of the items whose value exceeds its threshold.

    ``threshold`` is one finite number for every item, or an array of per-item thresholds of
    the same shape as ``values``. An item counts only when strictly above its threshold.
    Weights follow ``weighted_mean``. Derived in docs/statistical-methods.md, section 2.4.6.
    """

    x, w = _arrays(values, weights)
    t = np.asarray(threshold, dtype=float)
    if t.ndim == 0:
        t = np.full_like(x, float(t))
    if t.shape != x.shape or not np.all(np.isfinite(t)):
        raise ValueError("threshold must be finite and scalar or match values")
    return float(np.sum(w[x > t]))


def threshold_excess(
    values: ArrayLike, weights: ArrayLike, *, threshold: float | ArrayLike
) -> float:
    """Weighted mean of how far each value lies above its threshold, ``sum(w * max(x - t, 0))``.

    ``threshold`` is one finite number for every item, or an array of per-item thresholds of
    the same shape as ``values``. Weights follow ``weighted_mean``. Derived in
    docs/statistical-methods.md, section 2.4.6.
    """

    x, w = _arrays(values, weights)
    t = np.asarray(threshold, dtype=float)
    if t.ndim == 0:
        t = np.full_like(x, float(t))
    if t.shape != x.shape or not np.all(np.isfinite(t)):
        raise ValueError("threshold must be finite and scalar or match values")
    return float(np.dot(w, np.maximum(x - t, 0.0)))


def worst_of_k_risk(values: ArrayLike, weights: ArrayLike, *, k: int) -> float:
    """Weighted probability that at least one of ``k`` independent attempts fails.

    Each value is a per-item failure probability, so ``values`` must lie in
    ``[0, 1]``. The per-item transform ``1 - (1 - p) ** k`` is coordinatewise
    monotone, so propagated prompt envelopes remain valid; it reduces to the
    weighted mean at ``k = 1`` and is nondecreasing in ``k``.

    ``1 - (1 - p) ** k`` cancels catastrophically for small ``p``, and returns zero
    once ``1 - p`` rounds to one, so it is evaluated as ``-expm1(k * log1p(-p))``.
    """

    x, w = _arrays(values, weights)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer of at least one")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("values must be probabilities in [0, 1]")
    if k == 1:
        # Exactly the weighted mean, not just up to rounding.
        return float(np.dot(w, x))
    # log1p(-1) is -inf, which expm1 maps to -1, so p = 1 still gives exactly 1. Subtracting
    # from 0.0 rather than negating keeps p = 0 at +0.0 instead of -0.0.
    with np.errstate(divide="ignore"):
        attempts = 0.0 - np.expm1(k * np.log1p(-x))
    return float(np.dot(w, attempts))


def worst_of_k_estimate(
    failures: ArrayLike, rollouts: ArrayLike, weights: ArrayLike, *, k: int
) -> float:
    """Unbiased estimate of the weighted chance that one of ``k`` attempts fails.

    For a prompt with ``f`` failures in ``n >= k`` rollouts, the U-statistic with kernel
    ``max`` is the share of the ``C(n, k)`` subsets of its rollouts that contain a failure,
    ``1 - C(n - f, k) / C(n, k)``. It is unbiased for ``1 - (1 - p) ** k``. The plug-in
    ``1 - (1 - f / n) ** k`` is not: the transform is concave, so by Jensen's inequality the
    plug-in understates the risk. Every prompt needs at least ``k`` rollouts.

    The ratio is the product of ``1 - f / (n - i)`` over ``i < k``, summed in logarithms with
    ``log1p`` and completed with ``expm1``, so tiny failure rates keep their precision.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer of at least one")
    f = np.asarray(failures)
    n = np.asarray(rollouts)
    _, w = _arrays(np.zeros(len(n)), weights)
    if f.shape != n.shape or not all(
        np.issubdtype(array.dtype, np.integer) for array in (f, n)
    ):
        raise ValueError("failures and rollouts must be integer arrays of the same length")
    if np.any(f < 0) or np.any(f > n):
        raise ValueError("failures must lie between zero and the number of rollouts")
    if np.any(n < k):
        raise ValueError(f"every prompt needs at least {k} rollouts to estimate {k} attempts")
    return float(np.dot(w, worst_of_k_shares(f, n, k=k)))


def worst_of_k_shares(failures: ArrayLike, rollouts: ArrayLike, *, k: int) -> NDArray[np.float64]:
    """Per prompt, the share of its ``k``-subsets of rollouts that contain a failure.

    ``1 - C(n - f, k) / C(n, k)`` for ``f`` failures in ``n >= k`` rollouts. The caller checks
    the counts; :func:`worst_of_k_estimate` documents the evaluation.
    """

    f = np.asarray(failures)
    n = np.asarray(rollouts)
    risks = np.empty(len(n))
    for index, (fails, total) in enumerate(zip(f.tolist(), n.tolist(), strict=True)):
        if k == 1 or fails == 0:
            # Exactly f / n, and exactly zero without failures.
            risks[index] = fails / total
        elif total - fails < k:
            # Every k-subset contains a failure.
            risks[index] = 1.0
        else:
            log_ratio = math.fsum(math.log1p(-fails / (total - i)) for i in range(k))
            risks[index] = 0.0 - math.expm1(log_ratio)
    return risks
