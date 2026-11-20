from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np


def bonferroni_allocate(
    alpha: float,
    count: int,
    *,
    weights: Sequence[float] | None = None,
) -> list[float]:
    """Split a family-wise error level ``alpha`` across ``count`` claims, Bonferroni style.

    Without ``weights`` each claim gets ``alpha / count``. With ``weights``, one finite
    positive weight per claim, each claim gets ``alpha`` times its share of their total. The
    shares always sum to ``alpha``, which bounds the chance that any claim errs by ``alpha``
    under any dependence between them. Derived in docs/statistical-methods.md, section 9.2.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError("count must be a positive integer")
    if weights is None:
        return [alpha / count] * count
    w = np.asarray(weights, dtype=float)
    if w.shape != (count,) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Bonferroni weights must be finite, positive, and match count")
    w = w / np.sum(w)
    return [float(alpha * value) for value in w]
