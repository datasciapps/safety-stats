from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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
