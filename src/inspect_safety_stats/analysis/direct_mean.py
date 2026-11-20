"""Direct-mean intervals over the prompt population and its hierarchy nodes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from inspect_safety_stats.analysis.estimates import _block_index, prompt_blocked
from inspect_safety_stats.analysis.measures import _coverage_floors, _independence, _parameter
from inspect_safety_stats.config.methods import DirectMeanMethod
from inspect_safety_stats.config.models import (
    AnalysisPlan,
    IndependenceProfileSpec,
    RiskMeasureSpec,
)
from inspect_safety_stats.ingest.manifest import load_inclusion_probabilities
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import Diagnostic, PromptEstimate, RiskIntervalResult
from inspect_safety_stats.schema.scopes import cohort_scope, scope_path
from inspect_safety_stats.stats.design_based import (
    DesignBasedBounds,
    FixedSizeDesign,
    InclusionDesign,
    fixed_size_mean_bounds,
    horvitz_thompson_mean_bounds,
)
from inspect_safety_stats.stats.direct_mean import (
    BETTING_KERNEL,
    DIRECT_MEAN_ENGINES,
    DirectMeanBounds,
    worst_of_k_kl_chernoff_bounds,
)
from inspect_safety_stats.stats.independence import (
    BoundedBlocks,
    direct_bounded_blocks,
    prompt_level_block_ids,
    prompt_outcome_counts,
)
from inspect_safety_stats.stats.single_valued import (
    SINGLE_VALUED_BOUND_GUARANTEE,
    single_valued_bound,
)
from inspect_safety_stats.stats.tree_calibration import (
    IndependentFrontier,
    any_overlap,
    calibrate_independent_frontiers,
)
from inspect_safety_stats.validate.plan import worst_of_k_interval_problems

_BETTING_ASSUMPTION = (
    "Experimental: a common betting fraction, mixed over a prior fixed before any "
    "outcome was seen, is applied to every block with exponent A_g / max A; validity "
    "follows from Jensen's and the weighted AM-GM inequalities and needs no variance "
    "proxy."
)
"""States a betting interval's assumptions and marks it experimental, on fixed and sampled
designs alike (CONTRIBUTING sections 3 and 8.5)."""


@dataclass(frozen=True)
class _HierarchyNode:
    group_id: str
    level: str
    path: dict[str, str]
    prompt_ids: tuple[str, ...]
    weights: tuple[float, ...]
    design_weight: float


@dataclass(frozen=True)
class _EndpointCalibration:
    endpoint_alpha: float
    method: str
    frontier_id: str | None = None
    frontier_alpha: float | None = None
    frontier_size: int | None = None


@dataclass(frozen=True)
class _DirectMeanContext:
    groups: set[str]
    family_id: str | None
    family_alpha: float | None
    one_sided_coverage: float | None
    interval_coverage: float | None
    calibrations: dict[str, _EndpointCalibration]


def _calibration_key(cohort_id: str, target_id: str, profile_id: str) -> str:
    return f"{profile_id}@@{cohort_id}@@{target_id}"


def _path_rows(frame: pd.DataFrame, path_levels: list[str]) -> dict[tuple[str, ...], list[int]]:
    """Row positions of each distinct hierarchy path in ``frame``.

    ORDER MATTERS: paths come in order of first appearance and rows in frame order, as
    drop_duplicates() and one boolean mask per path gave them before. Nodes, target ids,
    frontiers and report rows are all emitted in this order.
    """
    rows: dict[tuple[str, ...], list[int]] = {}
    values = frame[path_levels].astype(str).itertuples(index=False, name=None)
    for position, path_values in enumerate(values):
        rows.setdefault(path_values, []).append(position)
    return rows


def _hierarchy_nodes(plan: AnalysisPlan, data: NormalizedData) -> list[_HierarchyNode]:
    spec = plan.outputs.direct_mean_intervals
    if not (spec.enabled and spec.hierarchy_nodes):
        return []
    selected_groups: set[str] = set(spec.groups or [group.id for group in data.groups])
    manifest = data.manifest.set_index("prompt_id", drop=False)
    levels = [level for level in plan.design.hierarchy if level != "prompt_id"]
    nodes: list[_HierarchyNode] = []
    for group in sorted(data.groups, key=lambda item: item.id):
        if group.id not in selected_groups:
            continue
        group_frame = manifest.loc[list(group.prompt_ids)]
        group_prompts = group_frame["prompt_id"].astype(str).tolist()
        for level_index, level in enumerate(levels):
            path_levels = levels[: level_index + 1]
            for path_values, rows in _path_rows(group_frame, path_levels).items():
                path = dict(zip(path_levels, path_values, strict=True))
                prompt_ids = tuple(group_prompts[row] for row in rows)
                raw_weights = np.asarray(
                    [group.weights[prompt_id] for prompt_id in prompt_ids], dtype=float
                )
                design_weight = float(np.sum(raw_weights))
                if design_weight <= 0:
                    continue
                nodes.append(
                    _HierarchyNode(
                        group_id=group.id,
                        level=level,
                        path=path,
                        prompt_ids=prompt_ids,
                        weights=tuple((raw_weights / design_weight).tolist()),
                        design_weight=design_weight,
                    )
                )
    return nodes


def _frontier_signature(
    entries: list[tuple[str, tuple[str, ...], tuple[float, ...]]],
) -> tuple[tuple[tuple[str, float], ...], ...]:
    return tuple(
        sorted(
            tuple(
                sorted(
                    (prompt_id, round(weight, 14))
                    for prompt_id, weight in zip(prompt_ids, weights, strict=True)
                )
            )
            for _, prompt_ids, weights in entries
        )
    )


def _group_independent_frontiers(
    plan: AnalysisPlan,
    data: NormalizedData,
    nodes: list[_HierarchyNode],
) -> list[IndependentFrontier]:
    selected_groups = set(
        plan.outputs.direct_mean_intervals.groups or [group.id for group in data.groups]
    )
    frontiers: list[IndependentFrontier] = []
    seen: dict[
        tuple[tuple[tuple[str, float], ...], ...],
        tuple[int, dict[tuple[tuple[str, float], ...], str]],
    ] = {}
    for group in sorted(data.groups, key=lambda item: item.id):
        if group.id not in selected_groups:
            continue
        root_id = _hierarchy_region_target_id(group.id, None, {})
        levels: list[tuple[str, list[tuple[str, tuple[str, ...], tuple[float, ...]]]]] = [
            (
                "root",
                [
                    (
                        root_id,
                        tuple(group.prompt_ids),
                        tuple(group.weights[prompt_id] for prompt_id in group.prompt_ids),
                    )
                ],
            )
        ]
        for level in [item for item in plan.design.hierarchy if item != "prompt_id"]:
            level_nodes = [
                node for node in nodes if node.group_id == group.id and node.level == level
            ]
            if not level_nodes:
                continue
            levels.append(
                (
                    level,
                    [
                        (
                            _hierarchy_region_target_id(node.group_id, node.level, node.path),
                            node.prompt_ids,
                            node.weights,
                        )
                        for node in level_nodes
                    ],
                )
            )

        for level, entries in levels:
            atom_sets = [frozenset(prompt_ids) for _, prompt_ids, _ in entries]
            if set().union(*atom_sets) != set(group.prompt_ids):
                raise ValueError(f"hierarchy level {group.id}:{level} does not partition the group")
            if any_overlap(atom_sets):
                raise ValueError(f"hierarchy level {group.id}:{level} contains overlapping nodes")
            signature = _frontier_signature(entries)
            if signature in seen:
                frontier_index, canonical_by_signature = seen[signature]
                frontier = frontiers[frontier_index]
                aliases = dict(frontier.aliases)
                for target_id, prompt_ids, weights in entries:
                    target_signature = tuple(
                        sorted(
                            (prompt_id, round(weight, 14))
                            for prompt_id, weight in zip(prompt_ids, weights, strict=True)
                        )
                    )
                    canonical_id = canonical_by_signature[target_signature]
                    aliases[canonical_id] = (*aliases[canonical_id], target_id)
                frontiers[frontier_index] = IndependentFrontier(
                    frontier_id=f"{frontier.frontier_id}+{level}",
                    targets=frontier.targets,
                    aliases=aliases,
                )
                continue
            targets = {target_id: frozenset(prompt_ids) for target_id, prompt_ids, _ in entries}
            aliases = {target_id: (target_id,) for target_id in targets}
            canonical_by_signature = {
                tuple(
                    sorted(
                        (prompt_id, round(weight, 14))
                        for prompt_id, weight in zip(prompt_ids, weights, strict=True)
                    )
                ): target_id
                for target_id, prompt_ids, weights in entries
            }
            seen[signature] = (len(frontiers), canonical_by_signature)
            frontiers.append(
                IndependentFrontier(
                    frontier_id=f"{group.id}:{level}",
                    targets=targets,
                    aliases=aliases,
                )
            )
    return frontiers


def _has_rows(data: NormalizedData, cohorts: Sequence[str], prompt_ids: Sequence[str]) -> bool:
    """Whether each of the cohorts has a row of each of the prompts."""
    positions = _block_index(data).positions
    return all((cohort, prompt) in positions for cohort in cohorts for prompt in prompt_ids)


def _own_blocks(prompt_ids: Sequence[str], prompt_weights: np.ndarray) -> frozenset[str]:
    """A block that no other target of the frontier holds, standing in for unobserved blocks.

    Its id starts with NUL, which sets it apart from the blocks of observed rollouts, named by
    digests, cohort ids and manifest values. It names the target's prompts of nonzero weight,
    which would form its blocks and which no other target of a frontier shares. Comparisons of
    the same two arms over the same prompts get the same one, as they would get the same
    observed blocks, and so share a region as those would.
    """
    weighted = (prompt for prompt, weight in zip(prompt_ids, prompt_weights, strict=True) if weight)
    return frozenset({"\x00" + "\x1f".join(sorted(weighted))})


def _complete_ledger_blocks(
    data: NormalizedData,
    profile: IndependenceProfileSpec,
    prompt_ids: Sequence[str],
    prompt_weights: np.ndarray,
    comparison_arms: tuple[str, str] | None = None,
) -> frozenset[str] | None:
    """The blocks a target would form if its cohort, or both arms, had a row of every prompt.

    A cohort that matches no evaluation, or misses a prompt, has no such ledger, and the target
    gets no interval: the risk and comparison stages require a complete one. It keeps its place
    and its share of alpha in its frontier all the same, so that every other target is
    calibrated as it would be were the ledger complete. Under the rollout profile no two targets
    share a block, so it gets one of its own; under a profile that blocks by prompt-level
    fields, its prompts' manifest rows name its blocks (``prompt_level_block_ids``). Under any
    other profile its blocks depend on rollouts it does not have, and the result is None: the
    frontier is then taken to hold overlapping targets, each calibrated as a frontier of its
    own, which spends the most alpha.
    """
    if profile.block_by == ["rollout"]:
        return _own_blocks(prompt_ids, prompt_weights)
    return prompt_level_block_ids(
        profile,
        _block_index(data).manifest_values,
        prompt_ids=prompt_ids,
        prompt_weights=prompt_weights,
        comparison_arms=comparison_arms,
    )


def _frontier_overlaps(supports: Iterable[frozenset[str] | None]) -> bool:
    """Whether targets of a frontier may share a block: some do, or some target's are unknown."""
    blocks = list(supports)
    return any(atoms is None for atoms in blocks) or any_overlap(
        atoms for atoms in blocks if atoms is not None
    )


def _direct_target_blocks(
    plan: AnalysisPlan,
    data: NormalizedData,
    profile: IndependenceProfileSpec,
    cohort_id: str,
    prompt_ids: tuple[str, ...],
    prompt_weights: np.ndarray,
) -> frozenset[str] | None:
    """The blocks a direct-mean target forms in a cohort, which its frontier keeps apart.

    Where the cohort has no row of some prompt of a fixed design, the blocks a complete ledger
    would form (``_complete_ledger_blocks``). A sampled design's undrawn items have no rows
    either; plan validation refuses frontier_sidak there (PLAN_SAMPLED_FRONTIER_UNSUPPORTED),
    and ``_direct_mean_context`` forms no frontier for it.
    """
    if plan.design.type == "fixed" and not _has_rows(data, (cohort_id,), prompt_ids):
        return _complete_ledger_blocks(data, profile, prompt_ids, prompt_weights)
    return frozenset(
        direct_bounded_blocks(
            data.observations,
            data.manifest,
            profile,
            cohort_id=cohort_id,
            prompt_ids=prompt_ids,
            prompt_weights=prompt_weights,
            index=_block_index(data),
        ).block_ids
    )


def _direct_mean_context(
    plan: AnalysisPlan,
    data: NormalizedData,
    *,
    nodes: list[_HierarchyNode] | None = None,
    diagnostics: list[Diagnostic] | None = None,
) -> _DirectMeanContext:
    """The calibrated alpha of every direct-mean target.

    A hierarchy that cannot be calibrated is reported in ``diagnostics`` and leaves every target
    uncalibrated; without ``diagnostics`` it raises ValueError. The frontiers are those of the
    plan: a cohort of a fixed design that has no rows of some prompt, such as one that matches
    no evaluation, keeps its targets in them (``_direct_target_blocks``), though they get no
    interval.
    """
    spec = plan.outputs.direct_mean_intervals
    if not spec.enabled:
        return _DirectMeanContext(set(), None, None, None, None, {})
    selected = set(spec.groups or [group.id for group in data.groups])
    family = next(
        (item for item in plan.multiplicity.families if item.id == spec.family),
        None,
    )
    if family is None:
        return _DirectMeanContext(selected, spec.family, None, None, None, {})
    directional_family_alpha, one_sided, joint = _coverage_floors(family.alpha, spec.coverage_mode)
    if directional_family_alpha is None:
        return _DirectMeanContext(selected, family.id, family.alpha, one_sided, joint, {})
    hierarchy_nodes = nodes or []
    independence = _independence(spec)
    profile_ids = [independence.primary, *independence.sensitivities]
    target_ids = [
        *[_hierarchy_region_target_id(group_id, None, {}) for group_id in sorted(selected)],
        *[
            _hierarchy_region_target_id(node.group_id, node.level, node.path)
            for node in hierarchy_nodes
        ],
        # Plan validation keeps these out of frontier_sidak calibration.
        *[
            _worst_of_k_target_id(group_id, measure.id)
            for group_id in sorted(selected)
            for measure in plan.outputs.risk_measures
            if measure.direct_interval
        ],
    ]
    calibrations: dict[str, _EndpointCalibration] = {}
    if not spec.simultaneous:
        for profile_id in profile_ids:
            for cohort_id in sorted(plan.cohorts):
                for target_id in target_ids:
                    calibrations[_calibration_key(cohort_id, target_id, profile_id)] = (
                        _EndpointCalibration(
                            endpoint_alpha=directional_family_alpha,
                            method="pointwise",
                        )
                    )
    elif spec.hierarchy_calibration == "frontier_sidak" and spec.hierarchy_nodes:
        if plan.design.type == "sampled":
            # Plan validation refuses frontier_sidak under a sampled design
            # (PLAN_SAMPLED_FRONTIER_UNSUPPORTED): its undrawn frame items form no blocks, so
            # no frontier is formed and no direct interval is calibrated.
            return _DirectMeanContext(selected, family.id, family.alpha, one_sided, joint, {})
        try:
            base_frontiers = _group_independent_frontiers(plan, data, hierarchy_nodes)
        except ValueError as exc:
            if diagnostics is None:
                raise
            # A level that does not partition its group, such as one whose zero-weight node
            # _hierarchy_nodes left out, has no frontier, so no direct interval is calibrated.
            diagnostics.append(
                Diagnostic(
                    code="DIRECT_MEAN_CALIBRATION_INVALID",
                    severity="ERROR",
                    message=str(exc),
                    scope=scope_path("families", family.id),
                )
            )
            return _DirectMeanContext(selected, family.id, family.alpha, one_sided, joint, {})
        target_details: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for group in data.groups:
            if group.id in selected:
                target_details[_hierarchy_region_target_id(group.id, None, {})] = (
                    group.prompt_ids,
                    np.asarray([group.weights[item] for item in group.prompt_ids]),
                )
        for node in hierarchy_nodes:
            target_details[_hierarchy_region_target_id(node.group_id, node.level, node.path)] = (
                node.prompt_ids,
                np.asarray(node.weights),
            )
        for profile_id in profile_ids:
            profile = plan.design.independence_profiles[profile_id]
            profile_frontiers: list[IndependentFrontier] = []
            for cohort_id in sorted(plan.cohorts):
                for frontier in base_frontiers:
                    known = {
                        target_id: _direct_target_blocks(
                            plan, data, profile, cohort_id, *target_details[target_id]
                        )
                        for target_id in frontier.targets
                    }
                    overlaps = _frontier_overlaps(known.values())
                    supports = {
                        target_id: (
                            _own_blocks(*target_details[target_id]) if atoms is None else atoms
                        )
                        for target_id, atoms in known.items()
                    }
                    if overlaps:
                        for target_id, atoms in supports.items():
                            aliases = frontier.aliases.get(target_id, (target_id,))
                            canonical = _calibration_key(cohort_id, target_id, profile_id)
                            profile_frontiers.append(
                                IndependentFrontier(
                                    frontier_id=(
                                        f"{profile_id}:{cohort_id}:{frontier.frontier_id}:"
                                        f"dependent:{target_id}"
                                    ),
                                    targets={canonical: atoms},
                                    aliases={
                                        canonical: tuple(
                                            _calibration_key(cohort_id, alias, profile_id)
                                            for alias in aliases
                                        )
                                    },
                                )
                            )
                    else:
                        profile_frontiers.append(
                            IndependentFrontier(
                                frontier_id=f"{profile_id}:{cohort_id}:{frontier.frontier_id}",
                                targets={
                                    _calibration_key(cohort_id, target_id, profile_id): atoms
                                    for target_id, atoms in supports.items()
                                },
                                aliases={
                                    _calibration_key(cohort_id, canonical_id, profile_id): tuple(
                                        _calibration_key(cohort_id, alias, profile_id)
                                        for alias in aliases
                                    )
                                    for canonical_id, aliases in frontier.aliases.items()
                                },
                            )
                        )
            frontier_calibrations = calibrate_independent_frontiers(
                profile_frontiers,
                directional_family_alpha=directional_family_alpha,
            )
            calibrations.update(
                {
                    target_id: _EndpointCalibration(
                        endpoint_alpha=item.endpoint_alpha,
                        method="frontier_sidak",
                        frontier_id=item.frontier_id,
                        frontier_alpha=item.frontier_alpha,
                        frontier_size=item.frontier_size,
                    )
                    for target_id, item in frontier_calibrations.items()
                }
            )
    else:
        endpoint_alpha = directional_family_alpha / len(target_ids) / len(plan.cohorts)
        for profile_id in profile_ids:
            for cohort_id in sorted(plan.cohorts):
                for target_id in target_ids:
                    calibrations[_calibration_key(cohort_id, target_id, profile_id)] = (
                        _EndpointCalibration(
                            endpoint_alpha=endpoint_alpha,
                            method="flat_bonferroni",
                        )
                    )
    return _DirectMeanContext(
        selected,
        family.id,
        family.alpha,
        one_sided,
        joint,
        calibrations,
    )


def _direct_bounds(
    method: DirectMeanMethod,
    blocks: BoundedBlocks,
    *,
    endpoint_alpha: float,
    independence_profile: str,
) -> DirectMeanBounds:
    engines = DIRECT_MEAN_ENGINES.get(method)
    if engines is None:
        raise ValueError(f"unsupported direct mean interval method {method!r}")
    function = engines.bounded
    return function(
        blocks.lower,
        blocks.upper,
        blocks.weights,
        alpha_lower=endpoint_alpha,
        alpha_upper=endpoint_alpha,
        independence_profile=independence_profile,
    )


def _direct_interval_result(
    *,
    bounds: DirectMeanBounds,
    role: Literal["primary", "reference", "sensitivity"],
    method: str,
    independence_profile: str,
    independent_block_count: int,
    plan: AnalysisPlan,
    family_id: str | None,
    family_alpha: float | None,
    calibration: _EndpointCalibration,
    one_sided_coverage: float | None,
    interval_coverage: float | None,
) -> RiskIntervalResult:
    spec = plan.outputs.direct_mean_intervals
    primary_profile = _independence(spec).primary
    profile = plan.design.independence_profiles[independence_profile]
    directional_family_alpha = (
        family_alpha / 2
        if family_alpha is not None and spec.coverage_mode == "two_sided"
        else family_alpha
    )
    assumptions = (
        [
            "Attempted rollout outcomes are mutually independent across the fixed node.",
            "Prompt weights and rollout counts are fixed independently of observed outcomes.",
        ]
        if profile.block_by == ["rollout"]
        else [
            f"Blocks defined by {profile.block_by!r} are mutually independent.",
            "Arbitrary dependence is allowed among observations within each declared block.",
            "The block partition and design weights are fixed independently of outcomes.",
        ]
    )
    assumptions.append("Every unclassified attempted outcome has a latent binary safety label.")
    if calibration.method == "frontier_sidak":
        assumptions.extend(
            [
                "Each hierarchy frontier is a partition into nodes with disjoint atomic outcomes.",
                "Independent nodes within a frontier use exact Sidak product calibration.",
                "Potentially dependent frontiers use Bonferroni allocation of "
                "directional family alpha.",
            ]
        )
    if bounds.chernoff_kernel == "bernoulli_kl_equal_atomic_weights":
        assumptions.append(
            "Equal atomic design weights permit inversion of the Bernoulli-KL "
            "Chernoff exponent; identical failure probabilities are not required."
        )
    elif bounds.chernoff_kernel == "weighted_poisson_chernoff_max_increment":
        assumptions.append(
            "Unequal atomic design weights use the weighted Poisson-Chernoff "
            "exponent determined by the largest atomic weight."
        )
    elif bounds.chernoff_kernel == BETTING_KERNEL:
        assumptions.append(_BETTING_ASSUMPTION)
    scope_prefix = "simultaneous" if spec.simultaneous else "pointwise"
    calibration_scope = "_frontier_sidak" if calibration.method == "frontier_sidak" else ""
    return RiskIntervalResult(
        interval_id=(
            f"direct_{method}"
            if independence_profile == primary_profile
            else f"direct_{method}_{independence_profile}"
        ),
        lower=bounds.lower,
        upper=bounds.upper,
        method=method,
        role=role,
        varied_assumption="independence" if role == "sensitivity" else None,
        coverage_scope=(
            f"{scope_prefix}{calibration_scope}_{independence_profile}_fixed_design_mean_family"
        ),
        coverage_mode=spec.coverage_mode,
        family=family_id,
        family_alpha=family_alpha,
        directional_family_alpha=directional_family_alpha,
        alpha_lower=calibration.endpoint_alpha,
        alpha_upper=calibration.endpoint_alpha,
        simultaneous_coverage=interval_coverage,
        lower_one_sided_coverage=one_sided_coverage,
        upper_one_sided_coverage=one_sided_coverage,
        completion_estimate_lower=bounds.completion_estimate_lower,
        completion_estimate_upper=bounds.completion_estimate_upper,
        half_width_lower=bounds.half_width_lower,
        half_width_upper=bounds.half_width_upper,
        effective_sample_size=bounds.effective_sample_size,
        concentration_variance=bounds.concentration_variance,
        variance_upper_lower=bounds.variance_upper_lower,
        variance_upper_upper=bounds.variance_upper_upper,
        maximum_increment=bounds.maximum_increment,
        chernoff_kernel=bounds.chernoff_kernel,
        calibration_method=calibration.method,
        calibration_frontier=calibration.frontier_id,
        calibration_frontier_alpha=calibration.frontier_alpha,
        calibration_frontier_size=calibration.frontier_size,
        independence_profile=independence_profile,
        independent_block_count=independent_block_count,
        missingness_method="all_safe_all_failure_completion",
        assumptions=assumptions,
        eligibility="certifying",
    )


def _worst_of_k_target_id(group_id: str, measure_id: str) -> str:
    """The direct-mean family target of a worst_of_k measure's direct interval on a group."""
    return f"{_hierarchy_region_target_id(group_id, None, {})}::measure={measure_id}"


def _worst_of_k_direct_interval(
    *,
    plan: AnalysisPlan,
    data: NormalizedData,
    measure: RiskMeasureSpec,
    cohort_id: str,
    group_id: str,
    prompt_ids: list[str],
    weights: list[float],
    prompt_results: list[PromptEstimate],
    direct: _DirectMeanContext,
    diagnostics: list[Diagnostic],
) -> tuple[RiskIntervalResult | None, list[str]]:
    """Experimental direct interval of a worst_of_k row, and the row's warnings about it.

    Each prompt's unbiased share enters the KL-Chernoff bound as floor(n / k) blocks, which
    needs a fixed design and independent rollouts. A plan that plan validation refuses, such as
    one whose primary profile does not block by rollout, gets no interval and a warning per
    reason on the row.
    """

    spec = plan.outputs.direct_mean_intervals
    k = int(_parameter(measure, "k", measure.k))
    problems = worst_of_k_interval_problems(plan)
    if problems:
        # validate_plan_references reports each as a PLAN_WORST_OF_K_INTERVAL_UNSUPPORTED error.
        return None, [f"No direct interval of {k} attempts: {problem}." for problem in problems]
    independence = _independence(spec)
    target_id = _worst_of_k_target_id(group_id, measure.id)
    calibration = direct.calibrations.get(
        _calibration_key(cohort_id, target_id, independence.primary)
    )
    if calibration is None:
        return None, ["The direct-mean multiplicity family is unavailable."]
    if any(
        item.n_total <= 0 or prompt_blocked(diagnostics, cohort_id, item.prompt_id)
        for item in prompt_results
    ):
        return None, ["A complete valid attempted-rollout ledger is required."]
    weighted = [
        (prompt_id, weight)
        for prompt_id, weight in zip(prompt_ids, weights, strict=True)
        if weight > 0
    ]
    failures, unclassified, attempted = prompt_outcome_counts(
        data.observations,
        data.manifest,
        cohort_id=cohort_id,
        prompt_ids=[prompt_id for prompt_id, _ in weighted],
        index=_block_index(data),
    )
    warnings: list[str] = []
    if independence.sensitivities:
        warnings.append(
            f"The direct interval of {k} attempts assumes independent rollouts; the sensitivity "
            f"profiles {', '.join(independence.sensitivities)} have no such interval."
        )
    if int(np.min(attempted)) < k:
        warnings.append(
            f"The design cannot bound {k} attempts: a prompt has only "
            f"{int(np.min(attempted))} attempted rollouts, and no estimate is unbiased for "
            "more attempts than rollouts."
        )
        return None, warnings
    try:
        bounds = worst_of_k_kl_chernoff_bounds(
            failures,
            unclassified,
            attempted,
            [weight for _, weight in weighted],
            k=k,
            alpha_lower=calibration.endpoint_alpha,
            alpha_upper=calibration.endpoint_alpha,
        )
    except ValueError as exc:
        diagnostics.append(
            Diagnostic(
                code="DIRECT_MEAN_INTERVAL_INVALID",
                severity="ERROR",
                message=str(exc),
                scope=cohort_scope(cohort_id, "groups", group_id, "measures", measure.id),
            )
        )
        return None, [*warnings, str(exc)]
    directional_family_alpha = (
        direct.family_alpha / 2
        if direct.family_alpha is not None and spec.coverage_mode == "two_sided"
        else direct.family_alpha
    )
    assumptions = [
        "The rollouts of each fixed prompt are independent draws with a common failure "
        f"probability p, so 1 - (1 - p)^{k} is the chance that one of {k} fresh attempts fails.",
        "Prompts are mutually independent.",
        "Prompt weights and rollout counts are fixed independently of observed outcomes.",
        f"Each prompt's share of its {k}-rollout subsets that hold a failure is unbiased for "
        f"1 - (1 - p)^{k}; as a U-statistic it enters the Chernoff bound as floor(n / {k}) "
        "independent [0, 1] blocks (Hoeffding, 1963, section 5).",
        "Every unclassified attempted outcome has a latent binary safety label.",
    ]
    if bounds.chernoff_kernel == "bernoulli_kl_equal_atomic_weights":
        assumptions.append(
            "Equal block weights permit inversion of the Bernoulli-KL Chernoff exponent; "
            "identical failure probabilities across prompts are not required."
        )
    elif bounds.chernoff_kernel == "weighted_poisson_chernoff_max_increment":
        assumptions.append(
            "Unequal block weights use the weighted Poisson-Chernoff exponent determined by "
            "the largest block weight."
        )
    scope_prefix = "simultaneous" if spec.simultaneous else "pointwise"
    return (
        RiskIntervalResult(
            interval_id="direct_worst_of_k_kl_chernoff",
            lower=bounds.lower,
            upper=bounds.upper,
            method="worst_of_k_u_statistic_kl_chernoff",
            role="primary",
            coverage_scope=f"{scope_prefix}_{independence.primary}_fixed_design_mean_family",
            coverage_mode=spec.coverage_mode,
            family=direct.family_id,
            family_alpha=direct.family_alpha,
            directional_family_alpha=directional_family_alpha,
            alpha_lower=calibration.endpoint_alpha,
            alpha_upper=calibration.endpoint_alpha,
            simultaneous_coverage=direct.interval_coverage,
            lower_one_sided_coverage=direct.one_sided_coverage,
            upper_one_sided_coverage=direct.one_sided_coverage,
            completion_estimate_lower=bounds.completion_estimate_lower,
            completion_estimate_upper=bounds.completion_estimate_upper,
            half_width_lower=bounds.half_width_lower,
            half_width_upper=bounds.half_width_upper,
            effective_sample_size=bounds.effective_sample_size,
            concentration_variance=bounds.concentration_variance,
            maximum_increment=bounds.maximum_increment,
            chernoff_kernel=bounds.chernoff_kernel,
            calibration_method=calibration.method,
            independence_profile=independence.primary,
            independent_block_count=int(np.sum(attempted // k)),
            missingness_method="all_safe_all_failure_completion",
            assumptions=assumptions,
            eligibility="diagnostic" if measure.diagnostic_only else "certifying",
        ),
        warnings,
    )


def _sampled_frame_intervals(
    *,
    data: NormalizedData,
    cohort_id: str,
    prompt_ids: tuple[str, ...] | list[str],
    weights: np.ndarray,
    estimates: dict[tuple[str, str], PromptEstimate],
    plan: AnalysisPlan,
    family_id: str | None,
    family_alpha: float | None,
    calibration: _EndpointCalibration,
    one_sided_coverage: float | None,
    interval_coverage: float | None,
) -> list[RiskIntervalResult]:
    """Horvitz-Thompson intervals for a frame mean estimated from an item sample.

    The declared independence profiles do not apply here. Inclusion randomness lives at
    the frame item, so the frame item is the independent block regardless of how rollouts
    within it are grouped.
    """

    spec = plan.outputs.direct_mean_intervals
    column = str(plan.design.inclusion_probability_column)
    stratified = plan.design.sampling == "fixed_size"
    manifest = data.manifest.set_index("prompt_id", drop=False)

    # Poisson inclusion is item-by-item, so only the target's own items matter. A
    # fixed-size draw happens at stratum level, so the whole frame is needed: an item
    # outside the target still consumed one of its stratum's draws.
    scope = list(manifest["prompt_id"].astype(str)) if stratified else list(prompt_ids)
    frame = manifest.loc[scope]
    probabilities = load_inclusion_probabilities(frame, column)
    target_weight = dict(zip(prompt_ids, np.asarray(weights, dtype=float), strict=True))

    sampled = np.zeros(len(scope), dtype=bool)
    lower_rates = np.zeros(len(scope), dtype=float)
    upper_rates = np.zeros(len(scope), dtype=float)
    frame_weights = np.zeros(len(scope), dtype=float)
    for index, prompt_id in enumerate(scope):
        frame_weights[index] = target_weight.get(prompt_id, 0.0)
        item = estimates.get((cohort_id, prompt_id))
        if item is None or item.n_total <= 0:
            continue
        sampled[index] = True
        lower_rates[index] = item.n_failure / item.n_total
        upper_rates[index] = (item.n_failure + item.n_unclassified) / item.n_total

    design: FixedSizeDesign | InclusionDesign
    if stratified:
        labels = (
            frame[str(plan.design.stratum_column)].astype(str).tolist()
            if plan.design.stratum_column
            else ["_all"] * len(scope)
        )
        order = {label: position for position, label in enumerate(sorted(set(labels)))}
        stratum_index = np.asarray([order[label] for label in labels], dtype=int)
        draws = np.bincount(stratum_index[sampled], minlength=len(order))
        design = FixedSizeDesign(stratum_index, frame_weights, draws, sampled)
        estimator = "stratified_without_replacement"
    else:
        design = InclusionDesign(frame_weights, probabilities, sampled)
        estimator = "horvitz_thompson"

    results: list[RiskIntervalResult] = []
    for method in [spec.method, *spec.reference_methods]:
        bounds: DesignBasedBounds
        if isinstance(design, FixedSizeDesign):
            bounds = fixed_size_mean_bounds(
                design,
                lower_rates,
                upper_rates,
                alpha_lower=calibration.endpoint_alpha,
                alpha_upper=calibration.endpoint_alpha,
                method=method,
            )
        else:
            bounds = horvitz_thompson_mean_bounds(
                design,
                lower_rates,
                upper_rates,
                alpha_lower=calibration.endpoint_alpha,
                alpha_upper=calibration.endpoint_alpha,
                method=method,
            )
        # The design-based bounds do not carry their engine's kernel, so the experimental
        # betting rows are recognised by name. They state their own construction and their
        # own reason the without-replacement reduction holds: the other engines' reason is
        # that each bounds a moment generating function, and betting bounds none.
        betting = method == "weighted_betting"
        if not stratified:
            draw = (
                "Frame items were included independently with the declared "
                "probabilities (Poisson sampling)."
            )
        elif betting:
            draw = (
                "Each stratum drew exactly its planned number of frame items without "
                "replacement; the bound uses Hoeffding's reduction on the capital of each "
                "fixed betting fraction, which, averaged over the rollouts, is at most the "
                "exponential of a without-replacement sum of per-item log wealth factors, so "
                "its expectation is dominated by the with-replacement one and the bound is "
                "conservative."
            )
        else:
            draw = (
                "Each stratum drew exactly its planned number of frame items "
                "without replacement; the bound uses Hoeffding's reduction, under "
                "which a convex function of a without-replacement sum is dominated "
                "by the with-replacement one, so it is conservative."
            )
        assumptions = [
            draw,
            "Inclusion was determined before, and independently of, the outcomes.",
            "Design weights are declared over the whole frame and are not "
            "renormalized to the realized sample.",
            f"Inverse-probability weighting widens every endpoint by a factor of "
            f"{bounds.inflation:.6g}.",
            f"The largest single frame item carries {bounds.maximum_weight:.6g} of "
            "the estimator's weight.",
            "Every unclassified attempted outcome has a latent binary safety label.",
        ]
        if betting:
            assumptions.append(_BETTING_ASSUMPTION)
        results.append(
            RiskIntervalResult(
                interval_id=f"{estimator}_{method}",
                lower=bounds.lower,
                upper=bounds.upper,
                method=method,
                role="primary" if method == spec.method else "reference",
                coverage_scope=(
                    f"{'simultaneous' if spec.simultaneous else 'pointwise'}"
                    "_sampled_frame_mean_family"
                ),
                coverage_mode=spec.coverage_mode,
                family=family_id,
                family_alpha=family_alpha,
                alpha_lower=calibration.endpoint_alpha,
                alpha_upper=calibration.endpoint_alpha,
                simultaneous_coverage=interval_coverage,
                lower_one_sided_coverage=one_sided_coverage,
                upper_one_sided_coverage=one_sided_coverage,
                completion_estimate_lower=bounds.completion_estimate_lower,
                completion_estimate_upper=bounds.completion_estimate_upper,
                effective_sample_size=bounds.effective_sample_size,
                maximum_increment=bounds.maximum_weight,
                chernoff_kernel=BETTING_KERNEL if betting else None,
                calibration_method=calibration.method,
                calibration_frontier=calibration.frontier_id,
                calibration_frontier_alpha=calibration.frontier_alpha,
                calibration_frontier_size=calibration.frontier_size,
                independence_profile="frame_item",
                independent_block_count=bounds.sampled_count,
                missingness_method="all_safe_all_failure_completion",
                assumptions=assumptions,
                eligibility="certifying",
            )
        )
    return results


def _direct_intervals(
    *,
    data: NormalizedData,
    cohort_id: str,
    prompt_ids: tuple[str, ...] | list[str],
    weights: np.ndarray,
    plan: AnalysisPlan,
    family_id: str | None,
    family_alpha: float | None,
    target_id: str,
    calibrations: dict[str, _EndpointCalibration],
    one_sided_coverage: float | None,
    interval_coverage: float | None,
    estimates: dict[tuple[str, str], PromptEstimate] | None = None,
) -> list[RiskIntervalResult]:
    spec = plan.outputs.direct_mean_intervals
    independence = _independence(spec)
    if plan.design.type == "sampled":
        if estimates is None:
            raise RuntimeError("a sampled design's direct-mean intervals need the prompt estimates")
        return _sampled_frame_intervals(
            data=data,
            cohort_id=cohort_id,
            prompt_ids=prompt_ids,
            weights=weights,
            estimates=estimates,
            plan=plan,
            family_id=family_id,
            family_alpha=family_alpha,
            calibration=calibrations[
                _calibration_key(cohort_id, target_id, independence.primary)
            ],
            one_sided_coverage=one_sided_coverage,
            interval_coverage=interval_coverage,
        )
    methods = [spec.method, *spec.reference_methods]
    results: list[RiskIntervalResult] = []
    profile_ids = [independence.primary, *independence.sensitivities]
    for profile_id in profile_ids:
        profile = plan.design.independence_profiles[profile_id]
        blocks = direct_bounded_blocks(
            data.observations,
            data.manifest,
            profile,
            cohort_id=cohort_id,
            prompt_ids=prompt_ids,
            prompt_weights=weights,
            index=_block_index(data),
        )
        calibration = calibrations[_calibration_key(cohort_id, target_id, profile_id)]
        for method in methods:
            bounds = _direct_bounds(
                method,
                blocks,
                endpoint_alpha=calibration.endpoint_alpha,
                independence_profile=profile_id,
            )
            results.append(
                _direct_interval_result(
                    bounds=bounds,
                    role=(
                        "sensitivity"
                        if profile_id != independence.primary
                        else "primary"
                        if method == spec.method
                        else "reference"
                    ),
                    method=method,
                    independence_profile=profile_id,
                    independent_block_count=blocks.count,
                    plan=plan,
                    family_id=family_id,
                    family_alpha=family_alpha,
                    calibration=calibration,
                    one_sided_coverage=one_sided_coverage,
                    interval_coverage=interval_coverage,
                )
            )
            if spec.single_valued_bound:
                _add_single_valued_bound(results[-1], method, blocks, profile_id)
    return results


def _add_single_valued_bound(
    interval: RiskIntervalResult,
    method: DirectMeanMethod,
    blocks: BoundedBlocks,
    independence_profile: str,
) -> None:
    """Attach M = inf_delta [U_delta + delta] for this node's own engine (#127).

    U_delta is the engine's upper endpoint at endpoint alpha delta for this node alone, not
    the family-calibrated endpoint, so M carries no multiplicity adjustment and no alpha.
    """
    bound = single_valued_bound(
        lambda delta: _direct_bounds(
            method,
            blocks,
            endpoint_alpha=delta,
            independence_profile=independence_profile,
        ).upper
    )
    interval.single_valued_bound = bound.value
    interval.single_valued_bound_delta = bound.delta
    interval.single_valued_bound_guarantee = SINGLE_VALUED_BOUND_GUARANTEE


def _hierarchy_region_target_id(group_id: str, level: str | None, path: dict[str, str]) -> str:
    if level is None:
        return f"{group_id}::root"
    path_text = "::".join(f"{key}={value}" for key, value in path.items())
    return f"{group_id}::{level}::{path_text}"
