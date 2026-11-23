from __future__ import annotations

import math
from collections.abc import Iterable, Set
from dataclasses import dataclass


@dataclass(frozen=True)
class IndependentFrontier:
    """A collection of targets built from pairwise-disjoint independent atoms."""

    frontier_id: str
    targets: dict[str, frozenset[str]]
    aliases: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class FrontierTargetCalibration:
    """The one-sided error level assigned to one target by ``calibrate_independent_frontiers``.

    ``frontier_alpha`` is the frontier's share of the directional family level, and
    ``endpoint_alpha`` is the Sidak level of each of its ``frontier_size`` targets. Derived in
    docs/statistical-methods.md, section 2.4.10, "Tree-wide calibration by independent
    frontiers".
    """

    target_id: str
    endpoint_alpha: float
    frontier_id: str
    frontier_alpha: float
    frontier_size: int


def sidak_local_alpha(frontier_alpha: float, target_count: int) -> float:
    """Equal local alpha giving exact product coverage for independent targets."""

    if not 0 < frontier_alpha < 1:
        raise ValueError("frontier_alpha must be in (0, 1)")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    return -math.expm1(math.log1p(-frontier_alpha) / target_count)


def any_overlap(sets: Iterable[Set[object]]) -> bool:
    """Whether any two of the sets share an element.

    The same answer as testing every pair, in time linear in the total number of elements.
    """
    seen: set[object] = set()
    for members in sets:
        if not seen.isdisjoint(members):
            return True
        seen.update(members)
    return False


def calibrate_independent_frontiers(
    frontiers: list[IndependentFrontier],
    *,
    directional_family_alpha: float,
    frontier_weights: dict[str, float] | None = None,
) -> dict[str, FrontierTargetCalibration]:
    """Allocate alpha across dependent frontiers and by Sidak within each frontier.

    Frontiers receive equal Bonferroni shares because they may reuse atoms. Within a
    frontier, target atom sets must be pairwise disjoint, so exact product calibration
    applies under mutual independence of the atoms. Aliases identify repeated statistics
    in duplicate hierarchy levels and do not consume additional alpha.
    """

    if not frontiers:
        raise ValueError("at least one frontier is required")
    if not 0 < directional_family_alpha < 1:
        raise ValueError("directional_family_alpha must be in (0, 1)")
    weights = frontier_weights or {frontier.frontier_id: 1.0 for frontier in frontiers}
    if set(weights) != {frontier.frontier_id for frontier in frontiers}:
        raise ValueError("frontier_weights must contain every frontier exactly once")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("frontier weights must be finite and positive")
    total_weight = sum(weights.values())
    result: dict[str, FrontierTargetCalibration] = {}
    for frontier in frontiers:
        frontier_alpha = directional_family_alpha * weights[frontier.frontier_id] / total_weight
        if not frontier.targets:
            raise ValueError(f"frontier {frontier.frontier_id!r} has no targets")
        target_items = list(frontier.targets.items())
        # The pairwise scan only runs to name the offending pair, so that the same error is
        # raised for the same input as before; disjoint frontiers skip it.
        overlapping = any_overlap(atoms for _, atoms in target_items)
        for index, (target_id, atoms) in enumerate(target_items):
            if not atoms:
                raise ValueError(
                    f"target {target_id!r} in frontier {frontier.frontier_id!r} has no atoms"
                )
            for other_id, other_atoms in target_items[index + 1 :] if overlapping else ():
                if atoms & other_atoms:
                    raise ValueError(
                        f"frontier {frontier.frontier_id!r} targets {target_id!r} and "
                        f"{other_id!r} are not disjoint"
                    )
        local_alpha = sidak_local_alpha(frontier_alpha, len(target_items))
        for canonical_id, _ in target_items:
            aliases = frontier.aliases.get(canonical_id, (canonical_id,))
            for target_id in aliases:
                if target_id in result:
                    raise ValueError(f"target {target_id!r} belongs to multiple frontiers")
                result[target_id] = FrontierTargetCalibration(
                    target_id=target_id,
                    endpoint_alpha=local_alpha,
                    frontier_id=frontier.frontier_id,
                    frontier_alpha=frontier_alpha,
                    frontier_size=len(target_items),
                )
    return result
