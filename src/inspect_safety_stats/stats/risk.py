from __future__ import annotations

import math
from collections.abc import Sequence

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


def power_mean(values: ArrayLike, weights: ArrayLike, *, order: float) -> float:
    x, w = _arrays(values, weights)
    if not math.isfinite(order) or order < 1:
        raise ValueError("order must be finite and at least one")
    if np.any(x < 0):
        raise ValueError("power mean requires nonnegative values")
    if order == 1:
        return float(np.dot(x, w))
    positive = (x > 0) & (w > 0)
    if not np.any(positive):
        return 0.0
    logs = np.log(w[positive]) + order * np.log(x[positive])
    maximum = float(np.max(logs))
    return math.exp((maximum + math.log(float(np.sum(np.exp(logs - maximum))))) / order)


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


def design_quantile(values: ArrayLike, weights: ArrayLike, *, beta: float) -> float:
    x, w = _arrays(values, weights)
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    support = w > 0
    x = x[support]
    w = w[support]
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(w[order])
    idx = int(np.searchsorted(cumulative, beta, side="left"))
    return float(x[order[min(idx, len(x) - 1)]])


def spectral_risk(values: ArrayLike, weights: ArrayLike, *, spectrum: Sequence[float]) -> float:
    x, w = _arrays(values, weights)
    phi = np.asarray(spectrum, dtype=float)
    if phi.ndim != 1 or len(phi) == 0 or not np.all(np.isfinite(phi)) or np.sum(phi) <= 0:
        raise ValueError("spectrum must be a nonempty finite sequence")
    if np.any(phi < 0) or np.any(np.diff(phi) < -1e-12):
        raise ValueError("spectrum must be nonnegative and nondecreasing")
    mass = phi / float(np.sum(phi))
    points = (np.arange(len(phi), dtype=float) + 0.5) / len(phi)
    quantiles = np.array([design_quantile(x, w, beta=float(p)) for p in points])
    return float(np.dot(mass, quantiles))
