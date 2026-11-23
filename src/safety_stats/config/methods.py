"""The direct-mean bound methods: their names and the default of each place that takes one.

Plans, the stats modules and the analysis all name a method with ``DirectMeanMethod``, or
with ``ComparisonMethod`` where only the comparison methods are accepted, so a method is
added or removed here and nowhere else. The functions that compute each method are
registered against the same names in ``stats.direct_mean.DIRECT_MEAN_ENGINES``. This module
imports nothing heavy, because plan validation reads it.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

ComparisonMethod = Literal[
    "weighted_hoeffding",
    "weighted_empirical_bernstein",
    "weighted_kl_chernoff",
]
"""The methods a plan's comparisons and comparison sets, and ``stats.comparisons``, accept."""

DirectMeanMethod = Literal[
    ComparisonMethod,
    "weighted_betting",
]
"""Every direct-mean method: direct-mean intervals and ``stats.design_based`` accept them all."""

DIRECT_MEAN_METHODS: Final[tuple[DirectMeanMethod, ...]] = get_args(DirectMeanMethod)
COMPARISON_METHODS: Final[tuple[ComparisonMethod, ...]] = get_args(ComparisonMethod)

# The defaults differ by place. Changing one changes the reports of every plan that leaves
# the method out, so it is a class B change (CONTRIBUTING section 3).
DEFAULT_DIRECT_MEAN_INTERVAL_METHOD: Final[DirectMeanMethod] = "weighted_empirical_bernstein"
"""``outputs.direct_mean_intervals.method`` in a plan."""

DEFAULT_COMPARISON_METHOD: Final[ComparisonMethod] = "weighted_kl_chernoff"
"""``method`` of a plan's comparisons and comparison sets."""

DEFAULT_STATS_FUNCTION_METHOD: Final[ComparisonMethod] = "weighted_empirical_bernstein"
"""``method`` of the functions in ``stats.comparisons`` and ``stats.design_based``."""

__all__ = [
    "COMPARISON_METHODS",
    "DEFAULT_COMPARISON_METHOD",
    "DEFAULT_DIRECT_MEAN_INTERVAL_METHOD",
    "DEFAULT_STATS_FUNCTION_METHOD",
    "DIRECT_MEAN_METHODS",
    "ComparisonMethod",
    "DirectMeanMethod",
]
