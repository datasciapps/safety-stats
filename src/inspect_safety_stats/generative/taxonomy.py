"""Compile an authored generative plan into one canonical tree.

Every sampler and estimator reads the compiled table produced here, never the
authoring configuration. Authoring shortcuts -- equal weights, values loaded from a
CSV, context-specific overrides -- must not create a second statistical
representation of the same design.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from inspect_safety_stats.generative.schema import ArmSpec, GenerativePlan, reject_reserved

WEIGHT_TOLERANCE = 1e-9
NodeType = Literal["root", "fixed", "arm"]

Assignments = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CompiledNode:
    node_id: str
    parent_id: str | None
    level: str
    value: str
    node_type: NodeType
    conditional_target_weight: float
    target_path_weight: float
    semantic_seed_id: str | None
    rewrite_kernel_id: str | None
    path_hash: str
    depth: int
    is_terminal: bool


@dataclass(frozen=True)
class CompiledTaxonomy:
    root_id: str
    levels: tuple[str, ...]
    semantic_seed_level: str
    nodes: tuple[CompiledNode, ...]
    fingerprint: str
    pruned_zero_weight: tuple[str, ...]
    pruned_by_constraint: tuple[str, ...]

    @property
    def terminal_paths(self) -> tuple[CompiledNode, ...]:
        return tuple(node for node in self.nodes if node.is_terminal)

    def node(self, node_id: str) -> CompiledNode:
        for candidate in self.nodes:
            if candidate.node_id == node_id:
                return candidate
        raise KeyError(node_id)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "level": node.level,
                    "value": node.value,
                    "node_type": node.node_type,
                    "conditional_target_weight": node.conditional_target_weight,
                    "target_path_weight": node.target_path_weight,
                    "semantic_seed_id": node.semantic_seed_id,
                    "rewrite_kernel_id": node.rewrite_kernel_id,
                    "path_hash": node.path_hash,
                }
                for node in self.nodes
            ]
        )


def _node_id(root_id: str, assignments: Assignments) -> str:
    return "|".join([root_id, *(f"{level}={value}" for level, value in assignments)])


def _path_hash(node_id: str) -> str:
    return hashlib.sha256(node_id.encode("utf-8")).hexdigest()


def _read_manifest(path: Path, required: list[str]) -> list[dict[str, str]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"seed manifest {path} is missing columns: {', '.join(missing)}")
    rows: list[dict[str, str]] = []
    for record in frame.to_dict("records"):
        row = {str(key): str(value).strip() for key, value in record.items()}
        for column in required:
            if not row[column]:
                raise ValueError(f"seed manifest {path} has a blank {column}")
            reject_reserved(f"seed manifest {column}", row[column])
        rows.append(row)
    if not rows:
        raise ValueError(f"seed manifest {path} contains no rows")
    return rows


def _fixed_children(
    rows: list[dict[str, str]],
    plan: GenerativePlan,
    fixed_order: list[str],
) -> dict[Assignments, dict[str, float]]:
    """Conditional weights for the manifest-sourced prefix of the hierarchy."""

    children: dict[Assignments, dict[str, float]] = {}
    for depth, level in enumerate(fixed_order):
        spec = plan.taxonomy.fixed_levels[level]
        declared: dict[Assignments, dict[str, float | None]] = {}
        for row in rows:
            prefix: Assignments = tuple((name, row[name]) for name in fixed_order[:depth])
            value = row[level]
            siblings = declared.setdefault(prefix, {})
            if spec.weights == "equal_within_parent":
                siblings[value] = None
                continue
            column = str(spec.weight_column)
            if column not in row:
                raise ValueError(f"seed manifest is missing weight column {column!r}")
            weight = float(row[column])
            if weight < 0.0:
                raise ValueError(f"negative weight {weight} at level {level!r}")
            previous = siblings.get(value)
            if previous is not None and abs(previous - weight) > WEIGHT_TOLERANCE:
                raise ValueError(
                    f"level {level!r} value {value!r} carries inconsistent weights "
                    f"{previous} and {weight}; a level's weight must be constant "
                    "within its parent"
                )
            siblings[value] = weight

        for prefix, siblings in declared.items():
            if spec.weights == "equal_within_parent":
                share = 1.0 / len(siblings)
                children[prefix] = dict.fromkeys(siblings, share)
                continue
            resolved = {value: float(weight or 0.0) for value, weight in siblings.items()}
            total = sum(resolved.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"level {level!r} weights under {prefix or 'the root'} sum to {total}, not 1"
                )
            children[prefix] = resolved
    return children


def _arm_values(arm: ArmSpec, level: str, base_path: Path) -> dict[str, float]:
    if arm.values is not None:
        return {value.id: value.weight for value in arm.values}
    source = base_path / str(arm.values_from)
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    column = str(arm.weight_column)
    for required in (level, column):
        if required not in frame.columns:
            raise ValueError(f"arm source {source} is missing column {required!r}")
    values: dict[str, float] = {}
    for record in frame.to_dict("records"):
        identifier = str(record[level]).strip()
        reject_reserved(f"arm {level!r} value", identifier)
        if identifier in values:
            raise ValueError(f"arm source {source} repeats value {identifier!r}")
        values[identifier] = float(record[column])
    if not values:
        raise ValueError(f"arm source {source} contains no rows")
    return values


def _matches(context: dict[str, str], condition: dict[str, str]) -> bool:
    return all(context.get(level) == value for level, value in condition.items())


def _allowed_values(plan: GenerativePlan, context: dict[str, str], level: str) -> set[str] | None:
    """Intersect every constraint that applies here. None means unconstrained."""

    allowed: set[str] | None = None
    for constraint in plan.taxonomy.constraints:
        if level not in constraint.allow or not _matches(context, constraint.when):
            continue
        permitted = set(constraint.allow[level])
        allowed = permitted if allowed is None else (allowed & permitted)
    return allowed


def _resolve_arm(
    plan: GenerativePlan,
    level: str,
    context: dict[str, str],
    global_values: dict[str, float],
    zero_weight: list[str],
    constrained: list[str],
) -> dict[str, float]:
    arm = plan.taxonomy.arms[level]
    base = dict(global_values)
    contextual = False
    for entry in arm.context_weights:
        if _matches(context, entry.when):
            base = {value.id: value.weight for value in entry.values}
            unknown = set(base) - set(global_values)
            if unknown:
                raise ValueError(
                    f"arm {level!r} context_weights introduce undeclared values: "
                    f"{', '.join(sorted(unknown))}"
                )
            contextual = True
            break

    node = _node_id(plan.taxonomy.root_id, tuple(context.items()))
    for value, weight in sorted(base.items()):
        if weight == 0.0:
            zero_weight.append(f"{node}|{level}={value}")
    declared = {value: weight for value, weight in base.items() if weight > 0.0}
    if not declared:
        raise ValueError(f"arm {level!r} has no positive-weight value under {node}")

    allowed = _allowed_values(plan, context, level)
    if allowed is None:
        survivors = declared
    else:
        survivors = {value: weight for value, weight in declared.items() if value in allowed}
        for value in sorted(set(declared) - set(survivors)):
            constrained.append(f"{node}|{level}={value}")

    if not survivors:
        raise ValueError(
            f"node {node} has positive target weight but every child at level {level!r} "
            "was pruned by a constraint"
        )

    if set(survivors) != set(declared):
        if plan.taxonomy.renormalization == "renormalize_valid_children":
            total = sum(survivors.values())
            return {value: weight / total for value, weight in survivors.items()}
        raise ValueError(
            f"taxonomy.renormalization is 'explicit_context_weights' but level {level!r} "
            f"under {node} is pruned without a context_weights entry covering exactly its "
            f"valid children {sorted(survivors)}"
        )

    total = sum(survivors.values())
    if abs(total - 1.0) > 1e-6:
        source = "context_weights" if contextual else "declared weights"
        raise ValueError(
            f"arm {level!r} {source} under {node} sum to {total}, not 1"
        )
    return survivors


def compile_taxonomy(plan: GenerativePlan, *, base_path: str | Path = ".") -> CompiledTaxonomy:
    """Expand the authored plan into the canonical node table.

    Raises ``ValueError`` for any structure a sampler could not act on: weights that
    do not normalize, a positive-weight node whose children were all pruned, a seed
    appearing under two fixed parents, or a tree larger than the declared guard.
    """

    root_path = Path(base_path)
    taxonomy = plan.taxonomy
    hierarchy = list(taxonomy.hierarchy)
    seed_index = hierarchy.index(taxonomy.semantic_seed_level)
    fixed_order = hierarchy[: seed_index + 1]
    arm_order = hierarchy[seed_index + 1 :]

    rows = _read_manifest(root_path / taxonomy.seed_manifest, fixed_order)
    seed_parents: dict[str, Assignments] = {}
    for row in rows:
        seed = row[taxonomy.semantic_seed_level]
        parent: Assignments = tuple((name, row[name]) for name in fixed_order[:seed_index])
        existing = seed_parents.setdefault(seed, parent)
        if existing != parent:
            raise ValueError(
                f"semantic seed {seed!r} appears under two fixed parent paths; a seed is a "
                "fixed design object with exactly one place in the hierarchy"
            )

    fixed_children = _fixed_children(rows, plan, fixed_order)
    global_arm_values = {
        level: _arm_values(taxonomy.arms[level], level, root_path) for level in arm_order
    }

    zero_weight: list[str] = []
    constrained: list[str] = []
    root_id = taxonomy.root_id
    nodes: list[CompiledNode] = [
        CompiledNode(
            node_id=root_id,
            parent_id=None,
            level="",
            value=root_id,
            node_type="root",
            conditional_target_weight=1.0,
            target_path_weight=1.0,
            semantic_seed_id=None,
            rewrite_kernel_id=None,
            path_hash=_path_hash(root_id),
            depth=0,
            is_terminal=not hierarchy,
        )
    ]

    frontier: list[tuple[Assignments, float, str | None]] = [((), 1.0, None)]
    for depth, level in enumerate(hierarchy, start=1):
        is_fixed = level in taxonomy.fixed_levels
        is_last = depth == len(hierarchy)
        expanded: list[tuple[Assignments, float, str | None]] = []
        for assignments, weight, seed_id in frontier:
            if is_fixed:
                children = fixed_children.get(assignments, {})
                if not children:
                    raise ValueError(
                        f"node {_node_id(root_id, assignments)} has no children at "
                        f"fixed level {level!r}"
                    )
            else:
                children = _resolve_arm(
                    plan,
                    level,
                    dict(assignments),
                    global_arm_values[level],
                    zero_weight,
                    constrained,
                )
            parent_id = _node_id(root_id, assignments)
            for value, conditional in sorted(children.items()):
                if conditional <= 0.0:
                    zero_weight.append(f"{parent_id}|{level}={value}")
                    continue
                child_assignments: Assignments = (*assignments, (level, value))
                child_id = _node_id(root_id, child_assignments)
                child_seed = value if level == taxonomy.semantic_seed_level else seed_id
                path_weight = weight * conditional
                nodes.append(
                    CompiledNode(
                        node_id=child_id,
                        parent_id=parent_id,
                        level=level,
                        value=value,
                        node_type="fixed" if is_fixed else "arm",
                        conditional_target_weight=conditional,
                        target_path_weight=path_weight,
                        semantic_seed_id=child_seed,
                        rewrite_kernel_id=plan.rewrite.kernel_id if is_last else None,
                        path_hash=_path_hash(child_id),
                        depth=depth,
                        is_terminal=is_last,
                    )
                )
                expanded.append((child_assignments, path_weight, child_seed))
            if len(expanded) > taxonomy.maximum_paths:
                raise ValueError(
                    f"compiled tree exceeds taxonomy.maximum_paths "
                    f"({taxonomy.maximum_paths}) while expanding level {level!r}"
                )
        frontier = expanded

    terminal_total = sum(node.target_path_weight for node in nodes if node.is_terminal)
    if abs(terminal_total - 1.0) > 1e-6:
        raise ValueError(f"compiled path weights sum to {terminal_total}, not 1")

    with_children = {node.parent_id for node in nodes}
    for node in nodes:
        if node.is_terminal:
            if node.rewrite_kernel_id is None:
                raise ValueError(f"terminal node {node.node_id} has no rewrite kernel")
        elif node.node_id not in with_children:
            raise ValueError(
                f"node {node.node_id} carries positive weight but reaches no rewrite kernel"
            )

    payload = [
        [
            node.node_id,
            node.parent_id,
            node.level,
            node.value,
            node.node_type,
            round(node.conditional_target_weight, 12),
            round(node.target_path_weight, 12),
            node.semantic_seed_id,
            node.rewrite_kernel_id,
        ]
        for node in sorted(nodes, key=lambda item: item.node_id)
    ]
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return CompiledTaxonomy(
        root_id=root_id,
        levels=tuple(hierarchy),
        semantic_seed_level=taxonomy.semantic_seed_level,
        nodes=tuple(nodes),
        fingerprint=fingerprint,
        pruned_zero_weight=tuple(sorted(set(zero_weight))),
        pruned_by_constraint=tuple(sorted(set(constrained))),
    )
