"""Counts held in NumPy integers, as a DataFrame's sums are, give what Python integers give.

``binomial_upper_bound(frame["failed"].sum(), len(frame), alpha=0.05)`` used to raise "counts
must be integers", because ``sum()`` returns ``numpy.int64``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from inspect_safety_stats.stats import (
    binomial_bounds,
    binomial_lower_bound,
    binomial_upper_bound,
    bonferroni_allocate,
    missingness_robust_binomial_bounds,
)
from inspect_safety_stats.stats.adaptive import adaptive_lower_bound
from inspect_safety_stats.stats.planning import (
    certification_sample_size,
    maximum_certifying_failures,
    minimum_rejecting_failures,
    rejection_sample_size,
)

CALLS: dict[str, Callable[[Callable[[int], Any]], Any]] = {
    "binomial_upper_bound": lambda n: binomial_upper_bound(n(3), n(20), alpha=0.05),
    "binomial_lower_bound": lambda n: binomial_lower_bound(n(3), n(20), alpha=0.05),
    "binomial_bounds": lambda n: binomial_bounds(n(3), n(20), alpha_lower=0.025, alpha_upper=0.025),
    "missingness_robust_binomial_bounds": lambda n: missingness_robust_binomial_bounds(
        n(3), n(18), n(20), alpha_lower=0.025, alpha_upper=0.025
    ),
    "bonferroni_allocate": lambda n: bonferroni_allocate(0.05, n(4)),
    "adaptive_lower_bound": lambda n: adaptive_lower_bound(n(18), n(20), alpha=0.05),
    "maximum_certifying_failures": lambda n: maximum_certifying_failures(
        trials=n(200), threshold=0.05, alpha=0.05
    ),
    "minimum_rejecting_failures": lambda n: minimum_rejecting_failures(
        trials=n(200), threshold=0.05, alpha=0.05
    ),
    "certification_sample_size": lambda n: certification_sample_size(
        true_rate=0.01, threshold=0.05, alpha=0.05, power=0.8, max_trials=n(1000)
    ),
    "rejection_sample_size": lambda n: rejection_sample_size(
        true_rate=0.2, threshold=0.05, alpha=0.05, power=0.8, max_trials=n(1000)
    ),
}


@pytest.mark.parametrize("name", list(CALLS))
@pytest.mark.parametrize("integer", [np.int64, np.int32, np.uint16], ids=lambda t: t.__name__)
def test_numpy_integer_counts_give_the_python_result(name: str, integer: type) -> None:
    call = CALLS[name]
    assert call(integer) == call(int)


@pytest.mark.parametrize("value", [np.bool_(True), np.float64(3.0), 3.0])
def test_numpy_booleans_and_floats_are_still_refused(value: object) -> None:
    with pytest.raises(TypeError, match="counts must be integers"):
        binomial_upper_bound(value, 20, alpha=0.05)  # type: ignore[arg-type]
