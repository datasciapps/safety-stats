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
    x, w = _arrays(values, weights)
    return float(np.dot(x, w))


def maximum_risk(values: ArrayLike, weights: ArrayLike | None = None) -> float:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("values must be a nonempty finite one-dimensional array")
    if weights is not None:
        _arrays(x, weights)
    return float(np.max(x))


def weighted_cvar(values: ArrayLike, weights: ArrayLike, *, beta: float) -> float:
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
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("values must be a nonempty finite one-dimensional array")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(x):
        raise ValueError("k must be an integer between one and the number of values")
    partition = np.partition(x, len(x) - k)[-k:]
    return float(np.mean(partition))


def entropic_risk(values: ArrayLike, weights: ArrayLike, *, lambda_: float) -> float:
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
    x, w = _arrays(values, weights)
    t = np.asarray(threshold, dtype=float)
    if t.ndim == 0:
        t = np.full_like(x, float(t))
    if t.shape != x.shape or not np.all(np.isfinite(t)):
        raise ValueError("threshold must be finite and scalar or match values")
    return float(np.dot(w, np.maximum(x - t, 0.0)))
