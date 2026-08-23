from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _pvalues(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a finite one-dimensional sequence in [0, 1]")
    return p


def bonferroni_allocate(
    alpha: float,
    count: int,
    *,
    weights: Sequence[float] | None = None,
) -> list[float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if weights is None:
        return [alpha / count] * count
    w = np.asarray(weights, dtype=float)
    if w.shape != (count,) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Bonferroni weights must be finite, positive, and match count")
    w = w / np.sum(w)
    return [float(alpha * value) for value in w]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = _pvalues(p_values)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(p[idx]))
        running = max(running, value)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    p = _pvalues(p_values)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.empty(m, dtype=float)
    running = 1.0
    for reverse_rank in range(m - 1, -1, -1):
        idx = order[reverse_rank]
        rank = reverse_rank + 1
        value = min(1.0, float(p[idx]) * m / rank)
        running = min(running, value)
        adjusted_sorted[reverse_rank] = running
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]
