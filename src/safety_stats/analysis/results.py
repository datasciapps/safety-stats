"""The risk and comparison rows that the adjusting stages pass on."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from safety_stats.schema.reports import (
    ComparisonResult,
    GroupRiskResult,
    HierarchyRiskResult,
)


@dataclass(frozen=True)
class RiskRows:
    """The group, hierarchy-node and comparison rows of one analysis.

    A stage that adjusts rows (region projection, judge correction, panel correction) takes
    them and returns new ones, and never changes the rows it was given. The order of those
    stages is therefore the order in which each one's result is passed to the next.
    """

    risks: list[GroupRiskResult]
    hierarchy_risks: list[HierarchyRiskResult]
    comparisons: list[ComparisonResult]

    def copy(self) -> RiskRows:
        """A deep copy, for a stage to adjust. Objects the rows share stay shared in it."""
        return copy.deepcopy(self)
