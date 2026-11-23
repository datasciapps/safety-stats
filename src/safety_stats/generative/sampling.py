"""Counter-based path selection over a compiled generative taxonomy.

Selection randomness is derived from ``(selection_seed, label, index)`` by hashing
rather than from a sequential generator. Draw *t* therefore does not depend on draws
before it, so an interrupted run resumes without duplicating or skipping sampling
units, and a partial run is a prefix of the complete one rather than a different
sample.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from typing import Literal

from safety_stats.generative.ledger import SelectionRecord
from safety_stats.generative.schema import GenerativePlan
from safety_stats.generative.taxonomy import CompiledNode, CompiledTaxonomy

Method = Literal["direct", "proportional_stratified"]


@dataclass(frozen=True)
class StratumAllocation:
    """The draws given to one semantic-seed stratum under proportional stratification.

    ``stratum_weight`` is the stratum node's target path weight; ``draws`` is the declared
    per-stratum minimum plus its largest-remainder share of the rest of the budget.
    """

    stratum_id: str
    stratum_weight: float
    draws: int


@dataclass(frozen=True)
class SamplingDesign:
    """The explicit sampling design that :func:`plan_design` resolves from a plan.

    ``allocations`` is empty under ``direct`` sampling and holds one entry per stratum
    under ``proportional_stratified``. :func:`draw_selections` reads it to select paths.
    """

    method: Method
    total_rewrites: int
    selection_seed: int
    proposal_id: str
    allocations: tuple[StratumAllocation, ...]

    @property
    def draws_by_stratum(self) -> dict[str, int]:
        return {item.stratum_id: item.draws for item in self.allocations}

    @property
    def weight_by_stratum(self) -> dict[str, float]:
        return {item.stratum_id: item.stratum_weight for item in self.allocations}


def _uniform(selection_seed: int, label: str, index: int) -> float:
    """A counter-based pseudo-random uniform, stable across processes and runs."""

    digest = hashlib.sha256(f"{selection_seed}|{label}|{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def _choose(cumulative: list[float], draw: float) -> int:
    index = bisect.bisect_right(cumulative, draw * cumulative[-1])
    return min(index, len(cumulative) - 1)


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Integer allocation whose shares differ from the exact proportions by under one."""

    if total < 0:
        raise ValueError("total must be non-negative")
    exact = {key: weight * total for key, weight in weights.items()}
    allocation = {key: int(math.floor(value)) for key, value in exact.items()}
    shortfall = total - sum(allocation.values())
    order = sorted(exact, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:shortfall]:
        allocation[key] += 1
    return allocation


def stratification_frontier(taxonomy: CompiledTaxonomy) -> tuple[CompiledNode, ...]:
    """The semantic-seed level, which section 3.4 makes a mandatory frontier."""

    return tuple(
        node for node in taxonomy.nodes if node.level == taxonomy.semantic_seed_level
    )


def plan_design(taxonomy: CompiledTaxonomy, plan: GenerativePlan) -> SamplingDesign:
    """Resolve the declared sampling block into an explicit, auditable design."""

    spec = plan.sampling
    total = spec.total_rewrites
    if spec.method == "direct":
        return SamplingDesign(
            method="direct",
            total_rewrites=total,
            selection_seed=plan.evaluation.selection_seed,
            proposal_id="target-direct",
            allocations=(),
        )

    strata = stratification_frontier(taxonomy)
    minimum = spec.minimum_rewrites_per_stratum
    if minimum * len(strata) > total:
        raise ValueError(
            f"sampling.minimum_rewrites_per_stratum ({minimum}) across {len(strata)} strata "
            f"requires at least {minimum * len(strata)} rewrites, but only {total} are budgeted"
        )
    weights = {node.node_id: node.target_path_weight for node in strata}
    remainder = largest_remainder(weights, total - minimum * len(strata))
    allocations = tuple(
        StratumAllocation(
            stratum_id=node.node_id,
            stratum_weight=node.target_path_weight,
            draws=minimum + remainder[node.node_id],
        )
        for node in strata
    )
    empty = [item.stratum_id for item in allocations if item.draws == 0]
    if empty:
        raise ValueError(
            f"{len(empty)} strata receive no draws at a budget of {total}; raise "
            "sampling.total_rewrites or set sampling.minimum_rewrites_per_stratum"
        )
    return SamplingDesign(
        method="proportional_stratified",
        total_rewrites=total,
        selection_seed=plan.evaluation.selection_seed,
        proposal_id=f"proportional-stratified:{taxonomy.fingerprint[:12]}",
        allocations=allocations,
    )


def _terminals_under(taxonomy: CompiledTaxonomy, prefix: str | None) -> list[CompiledNode]:
    terminals = sorted(taxonomy.terminal_paths, key=lambda node: node.node_id)
    if prefix is None:
        return terminals
    return [node for node in terminals if node.node_id.startswith(f"{prefix}|")]


def draw_selections(
    taxonomy: CompiledTaxonomy,
    design: SamplingDesign,
    *,
    stage_id: Literal["pilot", "main"] = "main",
) -> list[SelectionRecord]:
    """Select structural paths, returning ledger rows in stable selection order."""

    records: list[SelectionRecord] = []
    if design.method == "direct":
        terminals = _terminals_under(taxonomy, None)
        cumulative: list[float] = []
        running = 0.0
        for node in terminals:
            running += node.target_path_weight
            cumulative.append(running)
        for index in range(design.total_rewrites):
            node = terminals[_choose(cumulative, _uniform(design.selection_seed, "direct", index))]
            records.append(
                SelectionRecord(
                    sampling_unit_id=f"u-{index:06d}",
                    stage_id=stage_id,
                    selection_order=index,
                    path_hash=node.path_hash,
                    node_id=node.node_id,
                    semantic_seed_id=str(node.semantic_seed_id),
                    stratum=None,
                    target_path_probability=node.target_path_weight,
                    proposal_path_probability=node.target_path_weight,
                    proposal_id=design.proposal_id,
                    selection_seed=design.selection_seed,
                )
            )
        return records

    order = 0
    for ordinal, allocation in enumerate(design.allocations):
        terminals = _terminals_under(taxonomy, allocation.stratum_id)
        if not terminals:
            raise ValueError(f"stratum {allocation.stratum_id} contains no terminal paths")
        cumulative = []
        running = 0.0
        for node in terminals:
            running += node.target_path_weight
            cumulative.append(running)
        for index in range(allocation.draws):
            draw = _uniform(design.selection_seed, allocation.stratum_id, index)
            node = terminals[_choose(cumulative, draw)]
            records.append(
                SelectionRecord(
                    sampling_unit_id=f"u-{ordinal:04d}-{index:06d}",
                    stage_id=stage_id,
                    selection_order=order,
                    path_hash=node.path_hash,
                    node_id=node.node_id,
                    semantic_seed_id=str(node.semantic_seed_id),
                    stratum=allocation.stratum_id,
                    target_path_probability=node.target_path_weight,
                    # Within a stratum the draw is from the conditional target distribution.
                    proposal_path_probability=(
                        node.target_path_weight / allocation.stratum_weight
                    ),
                    proposal_id=design.proposal_id,
                    selection_seed=design.selection_seed,
                )
            )
            order += 1
    return records
