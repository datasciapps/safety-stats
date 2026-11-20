"""Comparison targets, calibration and intervals between cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from inspect_safety_stats.analysis import interval_assumptions
from inspect_safety_stats.analysis.direct_mean import (
    _complete_ledger_blocks,
    _EndpointCalibration,
    _frontier_overlaps,
    _frontier_signature,
    _has_rows,
    _own_blocks,
    _path_rows,
)
from inspect_safety_stats.analysis.estimates import _block_index
from inspect_safety_stats.analysis.measures import _coverage_floors, _independence
from inspect_safety_stats.config.methods import ComparisonMethod
from inspect_safety_stats.config.models import AnalysisPlan, ComparisonSpec
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.observations import Observation, observation_field
from inspect_safety_stats.schema.reports import Diagnostic, Eligibility, RiskIntervalResult
from inspect_safety_stats.schema.scopes import scope_path
from inspect_safety_stats.stats.clusters import paired_cluster_bootstrap
from inspect_safety_stats.stats.comparisons import DirectComparisonBounds, RatioComparisonBounds
from inspect_safety_stats.stats.direct_mean import COMPARISON_ENGINES, DirectMeanBounds
from inspect_safety_stats.stats.independence import BoundedBlocks, comparison_contrast_blocks
from inspect_safety_stats.stats.paired import (
    CompletionLimitError,
    PartialPairing,
    exact_mcnemar,
    exact_mcnemar_partial,
    favourable_and_adverse_completions,
    newcombe_paired_interval,
    newcombe_paired_interval_partial,
    partial_pairing,
)
from inspect_safety_stats.stats.tree_calibration import (
    IndependentFrontier,
    any_overlap,
    calibrate_independent_frontiers,
)


def _observation_attr(observation: Observation, name: str) -> Any:
    if not hasattr(observation, observation_field(name)):
        raise ValueError(f"pair or cluster key {name!r} is unavailable")
    return getattr(observation, observation_field(name))


@dataclass(frozen=True)
class _ComparisonTarget:
    result_id: str
    level: str | None
    path: dict[str, str]
    prompt_ids: tuple[str, ...]
    weights: tuple[float, ...]
    design_weight: float


def _comparison_targets(
    plan: AnalysisPlan,
    data: NormalizedData,
    spec: ComparisonSpec,
) -> list[_ComparisonTarget]:
    group = next(item for item in data.groups if item.id == spec.group)
    prompts = tuple(group.prompt_ids)
    weights = tuple(group.weights[prompt_id] for prompt_id in prompts)
    targets = [
        _ComparisonTarget(
            result_id=spec.id,
            level=None,
            path={},
            prompt_ids=prompts,
            weights=weights,
            design_weight=1.0,
        )
    ]
    if not spec.hierarchy_nodes:
        return targets
    manifest = data.manifest.set_index("prompt_id", drop=False).loc[list(prompts)]
    levels = [level for level in plan.design.hierarchy if level != "prompt_id"]
    manifest_prompts = manifest["prompt_id"].astype(str).tolist()
    for level_index, level in enumerate(levels):
        path_levels = levels[: level_index + 1]
        for values, rows in _path_rows(manifest, path_levels).items():
            path = dict(zip(path_levels, values, strict=True))
            node_prompts = tuple(manifest_prompts[row] for row in rows)
            raw = np.asarray([group.weights[prompt_id] for prompt_id in node_prompts])
            design_weight = float(np.sum(raw))
            if design_weight <= 0:
                continue
            token = "/".join(f"{key}={value}" for key, value in path.items())
            targets.append(
                _ComparisonTarget(
                    result_id=f"{spec.id}::{level}::{token}",
                    level=level,
                    path=path,
                    prompt_ids=node_prompts,
                    weights=tuple((raw / design_weight).tolist()),
                    design_weight=design_weight,
                )
            )
    return targets


def _comparison_target_key(spec_id: str, result_id: str, profile_id: str) -> str:
    return f"{profile_id}@@{spec_id}@@{result_id}"


def _comparison_region_alias_key(
    spec: ComparisonSpec, *, allocation_weight: float
) -> tuple[Any, ...]:
    if spec.method == "weighted_kl_chernoff":
        return (
            "two_arm_kl_region",
            spec.candidate,
            spec.baseline,
            spec.group,
            spec.measure,
            spec.coverage_mode,
            spec.hierarchy_calibration,
            spec.hierarchy_region,
            allocation_weight,
        )
    return ("estimand_specific_region", spec.id)


def _comparison_independent_frontiers(
    plan: AnalysisPlan,
    data: NormalizedData,
    spec: ComparisonSpec,
    targets: list[_ComparisonTarget],
    profile_id: str,
) -> list[IndependentFrontier]:
    if not targets:
        return []
    profile = plan.design.independence_profiles[profile_id]

    def support(target: _ComparisonTarget) -> frozenset[str] | None:
        """The target's contrast blocks, or those both arms would form with a complete ledger.

        A sampled design's undrawn items have no rows either, and comparison_contrast_blocks
        refuses them, as before (``_direct_target_blocks``).
        """
        arms = (spec.candidate, spec.baseline)
        weights = np.asarray(target.weights)
        if plan.design.type == "fixed" and not _has_rows(data, arms, target.prompt_ids):
            return _complete_ledger_blocks(
                data, profile, target.prompt_ids, weights, comparison_arms=arms
            )
        return frozenset(
            comparison_contrast_blocks(
                data.observations,
                data.manifest,
                profile,
                candidate=spec.candidate,
                baseline=spec.baseline,
                prompt_ids=target.prompt_ids,
                prompt_weights=weights,
                index=_block_index(data),
            ).block_ids
        )

    def blocks_or_own(target: _ComparisonTarget, blocks: frozenset[str] | None) -> frozenset[str]:
        if blocks is None:
            return _own_blocks(target.prompt_ids, np.asarray(target.weights))
        return blocks

    if spec.hierarchy_calibration == "flat_bonferroni":
        return [
            IndependentFrontier(
                frontier_id=f"{profile_id}:{spec.id}:{target.result_id}",
                targets={
                    _comparison_target_key(spec.id, target.result_id, profile_id): blocks_or_own(
                        target, support(target)
                    )
                },
                aliases={},
            )
            for target in targets
        ]

    root = next(target for target in targets if target.level is None)
    levels: list[tuple[str, list[_ComparisonTarget]]] = [("root", [root])]
    for level in dict.fromkeys(target.level for target in targets if target.level is not None):
        levels.append((level, [target for target in targets if target.level == level]))

    frontiers: list[IndependentFrontier] = []
    seen: dict[
        tuple[tuple[tuple[str, float], ...], ...],
        tuple[int, dict[tuple[tuple[str, float], ...], str]],
    ] = {}
    for level, level_targets in levels:
        atom_sets = [frozenset(target.prompt_ids) for target in level_targets]
        if set().union(*atom_sets) != set(root.prompt_ids):
            raise ValueError(
                f"comparison hierarchy level {spec.id}:{level} does not partition the group"
            )
        if any_overlap(atom_sets):
            raise ValueError(
                f"comparison hierarchy level {spec.id}:{level} contains overlapping nodes"
            )
        entries = [
            (
                _comparison_target_key(spec.id, target.result_id, profile_id),
                target.prompt_ids,
                target.weights,
            )
            for target in level_targets
        ]
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
        target_by_id = {
            _comparison_target_key(spec.id, target.result_id, profile_id): target
            for target in level_targets
        }
        known = {target_id: support(target_by_id[target_id]) for target_id, *_ in entries}
        overlapping = _frontier_overlaps(known.values())
        targets_by_id = {
            target_id: blocks_or_own(target_by_id[target_id], blocks)
            for target_id, blocks in known.items()
        }
        if overlapping:
            for target_id, atoms in targets_by_id.items():
                frontiers.append(
                    IndependentFrontier(
                        frontier_id=f"{profile_id}:{spec.id}:{level}:dependent:{target_id}",
                        targets={target_id: atoms},
                        aliases={target_id: (target_id,)},
                    )
                )
            continue
        aliases = {target_id: (target_id,) for target_id in targets_by_id}
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
                frontier_id=f"{profile_id}:{spec.id}:{level}",
                targets=targets_by_id,
                aliases=aliases,
            )
        )
    return frontiers


def _comparison_calibrations(
    plan: AnalysisPlan,
    data: NormalizedData,
    targets: dict[str, list[_ComparisonTarget]],
    diagnostics: list[Diagnostic],
) -> dict[str, _EndpointCalibration]:
    families = {family.id: family for family in plan.multiplicity.families}
    result: dict[str, _EndpointCalibration] = {}
    simultaneous_by_family: dict[str, list[ComparisonSpec]] = {}
    for spec in plan.expanded_comparisons():
        if not spec.family:
            continue
        family = families.get(spec.family)
        if family is None:
            continue
        directional_alpha, _, _ = _coverage_floors(family.alpha, spec.coverage_mode)
        independence = _independence(spec)
        if not spec.simultaneous:
            for profile_id in [independence.primary, *independence.sensitivities]:
                for target in targets.get(spec.id, []):
                    result[_comparison_target_key(spec.id, target.result_id, profile_id)] = (
                        _EndpointCalibration(directional_alpha, "pointwise")
                    )
        else:
            simultaneous_by_family.setdefault(family.id, []).append(spec)

    for family_id, specs in simultaneous_by_family.items():
        family = families[family_id]
        directional_alpha = min(
            _coverage_floors(family.alpha, spec.coverage_mode)[0] for spec in specs
        )
        profile_ids = sorted(
            {
                profile_id
                for spec in specs
                for profile_id in [
                    _independence(spec).primary,
                    *_independence(spec).sensitivities,
                ]
            }
        )
        for profile_id in profile_ids:
            frontiers: list[IndependentFrontier] = []
            frontier_weights: dict[str, float] = {}
            frontier_aliases: dict[tuple[tuple[Any, ...], tuple[tuple[str, ...], ...]], int] = {}
            try:
                for spec in specs:
                    independence = _independence(spec)
                    if profile_id not in {independence.primary, *independence.sensitivities}:
                        continue
                    compiled = _comparison_independent_frontiers(
                        plan, data, spec, targets.get(spec.id, []), profile_id
                    )
                    for frontier in compiled:
                        allocation_weight = (
                            spec.allocation_weight
                            if family.method == "weighted_bonferroni"
                            else 1.0
                        )
                        if not (
                            spec.method == "weighted_kl_chernoff"
                            and spec.hierarchy_calibration == "frontier_sidak"
                        ):
                            frontiers.append(frontier)
                            frontier_weights[frontier.frontier_id] = allocation_weight
                            continue
                        frontier_level = frontier.frontier_id.split(":", maxsplit=1)[-1]
                        frontier = IndependentFrontier(
                            frontier_id=(
                                f"{profile_id}:two_arm_kl:{spec.candidate}:{spec.baseline}:"
                                f"{spec.group}:{spec.measure}:{frontier_level}"
                            ),
                            targets=frontier.targets,
                            aliases=frontier.aliases,
                        )
                        atom_signature = tuple(
                            sorted(
                                tuple(sorted(atom_ids)) for atom_ids in frontier.targets.values()
                            )
                        )
                        alias_key = (
                            _comparison_region_alias_key(spec, allocation_weight=allocation_weight),
                            atom_signature,
                        )
                        if alias_key not in frontier_aliases:
                            frontier_aliases[alias_key] = len(frontiers)
                            frontiers.append(frontier)
                            frontier_weights[frontier.frontier_id] = allocation_weight
                            continue
                        existing_index = frontier_aliases[alias_key]
                        existing = frontiers[existing_index]
                        canonical_by_atoms = {
                            atoms: target_id for target_id, atoms in existing.targets.items()
                        }
                        aliases = dict(existing.aliases)
                        for target_id, atoms in frontier.targets.items():
                            canonical_id = canonical_by_atoms[atoms]
                            aliases[canonical_id] = (
                                *aliases.get(canonical_id, (canonical_id,)),
                                *frontier.aliases.get(target_id, (target_id,)),
                            )
                        frontiers[existing_index] = IndependentFrontier(
                            frontier_id=existing.frontier_id,
                            targets=existing.targets,
                            aliases=aliases,
                        )
                calibrated = calibrate_independent_frontiers(
                    frontiers,
                    directional_family_alpha=directional_alpha,
                    frontier_weights=frontier_weights,
                )
            except ValueError as exc:
                diagnostics.append(
                    Diagnostic(
                        code="COMPARISON_CALIBRATION_INVALID",
                        severity="ERROR",
                        message=str(exc),
                        scope=scope_path(
                            "families", family_id, "profiles", profile_id
                        ),
                    )
                )
                continue
            for target_id, item in calibrated.items():
                method = (
                    "frontier_sidak"
                    if any(spec.hierarchy_calibration == "frontier_sidak" for spec in specs)
                    else "flat_bonferroni"
                )
                result[target_id] = _EndpointCalibration(
                    endpoint_alpha=item.endpoint_alpha,
                    method=method,
                    frontier_id=item.frontier_id,
                    frontier_alpha=item.frontier_alpha,
                    frontier_size=item.frontier_size,
                )
    return result


def _bounded_comparison_mean(
    method: ComparisonMethod,
    blocks: BoundedBlocks,
    *,
    alpha_lower: float,
    alpha_upper: float,
    profile_id: str,
) -> DirectMeanBounds:
    function = COMPARISON_ENGINES[method].bounded
    return function(
        blocks.lower,
        blocks.upper,
        blocks.weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_profile=profile_id,
    )


def _difference_from_blocks(
    method: ComparisonMethod,
    blocks: BoundedBlocks,
    *,
    alpha: float,
    profile_id: str,
) -> DirectComparisonBounds:
    bounds = _bounded_comparison_mean(
        method,
        blocks,
        alpha_lower=alpha,
        alpha_upper=alpha,
        profile_id=profile_id,
    )
    return DirectComparisonBounds(
        lower=max(-1.0, 2 * bounds.lower - 1),
        upper=min(1.0, 2 * bounds.upper - 1),
        completion_estimate_lower=2 * bounds.completion_estimate_lower - 1,
        completion_estimate_upper=2 * bounds.completion_estimate_upper - 1,
        half_width_lower=2 * bounds.half_width_lower,
        half_width_upper=2 * bounds.half_width_upper,
        concentration_variance=4 * bounds.concentration_variance,
        effective_sample_size=bounds.effective_sample_size,
        alpha_lower=alpha,
        alpha_upper=alpha,
        method=method,
        independence_unit=profile_id,
        variance_upper_lower=(
            4 * bounds.variance_upper_lower if bounds.variance_upper_lower is not None else None
        ),
        variance_upper_upper=(
            4 * bounds.variance_upper_upper if bounds.variance_upper_upper is not None else None
        ),
        maximum_increment=(
            2 * bounds.maximum_increment if bounds.maximum_increment is not None else None
        ),
        chernoff_kernel=bounds.chernoff_kernel,
    )


def _difference_rectangle_from_blocks(
    method: ComparisonMethod,
    candidate_blocks: BoundedBlocks,
    baseline_blocks: BoundedBlocks,
    *,
    alpha: float,
    profile_id: str,
) -> DirectComparisonBounds:
    candidate = _bounded_comparison_mean(
        method,
        candidate_blocks,
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        profile_id=profile_id,
    )
    baseline = _bounded_comparison_mean(
        method,
        baseline_blocks,
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        profile_id=profile_id,
    )
    completion_lower = candidate.completion_estimate_lower - baseline.completion_estimate_upper
    completion_upper = candidate.completion_estimate_upper - baseline.completion_estimate_lower
    lower = max(-1.0, candidate.lower - baseline.upper)
    upper = min(1.0, candidate.upper - baseline.lower)
    return DirectComparisonBounds(
        lower=lower,
        upper=upper,
        completion_estimate_lower=completion_lower,
        completion_estimate_upper=completion_upper,
        half_width_lower=max(0.0, completion_lower - lower),
        half_width_upper=max(0.0, upper - completion_upper),
        concentration_variance=(candidate.concentration_variance + baseline.concentration_variance),
        effective_sample_size=min(candidate.effective_sample_size, baseline.effective_sample_size),
        alpha_lower=alpha,
        alpha_upper=alpha,
        method=method,
        independence_unit=profile_id,
        maximum_increment=max(
            candidate.maximum_increment or 0.0, baseline.maximum_increment or 0.0
        ),
        chernoff_kernel=(
            candidate.chernoff_kernel
            if candidate.chernoff_kernel == baseline.chernoff_kernel
            else "two_arm_mixed"
        ),
        candidate_lower=candidate.lower,
        candidate_upper=candidate.upper,
        baseline_lower=baseline.lower,
        baseline_upper=baseline.upper,
    )


def _ratio_from_blocks(
    method: ComparisonMethod,
    candidate_blocks: BoundedBlocks,
    baseline_blocks: BoundedBlocks,
    *,
    alpha: float,
    profile_id: str,
    estimand: Literal["risk_ratio", "relative_risk_reduction"],
) -> RatioComparisonBounds:
    candidate = _bounded_comparison_mean(
        method,
        candidate_blocks,
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        profile_id=profile_id,
    )
    baseline = _bounded_comparison_mean(
        method,
        baseline_blocks,
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        profile_id=profile_id,
    )
    candidate_estimate = candidate.completion_estimate_lower
    baseline_estimate = baseline.completion_estimate_lower
    ratio_estimate = candidate_estimate / baseline_estimate if baseline_estimate > 0 else None
    ratio_lower = candidate.lower / baseline.upper if baseline.upper > 0 else None
    ratio_upper = candidate.upper / baseline.lower if baseline.lower > 0 else None
    warning = None
    if ratio_estimate is None:
        warning = "The observed baseline risk is zero, so the ratio estimate is undefined."
    if ratio_upper is None:
        warning = "The baseline lower bound is zero, so no finite upper ratio bound exists."
    if estimand == "relative_risk_reduction":
        estimate = 1 - ratio_estimate if ratio_estimate is not None else None
        lower = 1 - ratio_upper if ratio_upper is not None else None
        upper = 1 - ratio_lower if ratio_lower is not None else None
    else:
        estimate, lower, upper = ratio_estimate, ratio_lower, ratio_upper
    return RatioComparisonBounds(
        candidate_estimate=candidate_estimate,
        baseline_estimate=baseline_estimate,
        estimate=estimate,
        lower=lower,
        upper=upper,
        candidate_bounds=candidate,
        baseline_bounds=baseline,
        method=method,
        independence_unit=profile_id,
        estimand=estimand,
        warning=warning,
    )


def _comparison_interval(
    *,
    bounds: DirectComparisonBounds | RatioComparisonBounds,
    spec: ComparisonSpec,
    method: str,
    role: Literal["primary", "reference", "sensitivity"],
    independence_profile: str,
    independent_block_count: int,
    profile_block_by: list[str],
    comparison_coupling: Literal["independent", "joint"],
    family_alpha: float,
    calibration: _EndpointCalibration,
) -> RiskIntervalResult:
    directional_family_alpha, one_sided, joint = _coverage_floors(family_alpha, spec.coverage_mode)
    lower = bounds.lower
    upper = bounds.upper
    is_ratio = isinstance(bounds, RatioComparisonBounds)
    uses_two_arm_region = is_ratio or (
        isinstance(bounds, DirectComparisonBounds) and bounds.candidate_lower is not None
    )
    candidate_lower: float | None
    candidate_upper: float | None
    baseline_lower: float | None
    baseline_upper: float | None
    if profile_block_by == ["rollout"]:
        assumptions = [
            (
                interval_assumptions.ROLLOUT_INDEPENDENCE_WITHIN_ARMS.text()
                if uses_two_arm_region
                else interval_assumptions.ROLLOUT_INDEPENDENCE_ACROSS_ARMS.text()
            ),
            interval_assumptions.FIXED_ALIGNED_WEIGHTS_AND_COUNTS.text(),
        ]
    else:
        assumptions = [
            interval_assumptions.BLOCK_INDEPENDENCE.text(block_by=profile_block_by),
            interval_assumptions.WITHIN_BLOCK_DEPENDENCE_COUPLED.text(
                coupling=comparison_coupling
            ),
            interval_assumptions.FIXED_ALIGNED_BLOCK_PARTITION.text(),
        ]
    if is_ratio:
        if not isinstance(bounds, RatioComparisonBounds):
            raise TypeError(f"comparison {spec.id!r} has a ratio estimand but no ratio bounds")
        assumptions.append(
            "Ratio bounds use a simultaneous rectangular confidence set for the two arm means."
        )
        interval_warnings = [bounds.warning] if bounds.warning else []
        completion_lower = completion_upper = None
        half_width_lower = half_width_upper = None
        effective_sample_size = min(
            bounds.candidate_bounds.effective_sample_size,
            bounds.baseline_bounds.effective_sample_size,
        )
        concentration_variance = variance_lower = variance_upper = maximum_increment = None
        chernoff_kernel = bounds.candidate_bounds.chernoff_kernel
        candidate_lower = bounds.candidate_bounds.lower
        candidate_upper = bounds.candidate_bounds.upper
        baseline_lower = bounds.baseline_bounds.lower
        baseline_upper = bounds.baseline_bounds.upper
    else:
        if not isinstance(bounds, DirectComparisonBounds):
            raise TypeError(f"comparison {spec.id!r} has a difference estimand but ratio bounds")
        interval_warnings = []
        completion_lower = bounds.completion_estimate_lower
        completion_upper = bounds.completion_estimate_upper
        half_width_lower = bounds.half_width_lower
        half_width_upper = bounds.half_width_upper
        effective_sample_size = bounds.effective_sample_size
        concentration_variance = bounds.concentration_variance
        variance_lower = bounds.variance_upper_lower
        variance_upper = bounds.variance_upper_upper
        maximum_increment = bounds.maximum_increment
        chernoff_kernel = bounds.chernoff_kernel
        candidate_lower = bounds.candidate_lower
        candidate_upper = bounds.candidate_upper
        baseline_lower = bounds.baseline_lower
        baseline_upper = bounds.baseline_upper
        if uses_two_arm_region:
            assumptions.append(
                "The risk-difference bound uses a simultaneous rectangular confidence set "
                "for the two sparse arm means."
            )
    assumptions.append(interval_assumptions.LATENT_BINARY_LABEL.text())
    if calibration.method == "frontier_sidak":
        assumptions.extend(
            [
                "Comparison siblings in a frontier use disjoint prompt blocks from both arms.",
                "Independent sibling blocks use exact Sidak product calibration.",
                "Potentially dependent comparison frontiers use Bonferroni allocation.",
            ]
        )
    eligibility: Eligibility = (
        "certifying" if lower is not None and upper is not None else "invalid"
    )
    scope_prefix = (
        "simultaneous_frontier_sidak"
        if calibration.method == "frontier_sidak"
        else "simultaneous"
        if spec.simultaneous
        else "pointwise"
    )
    return RiskIntervalResult(
        interval_id=(
            f"direct_{method}"
            if spec.independence is not None and independence_profile == spec.independence.primary
            else f"direct_{method}_{independence_profile}"
        ),
        lower=lower,
        upper=upper,
        method=method,
        role=role,
        varied_assumption="independence" if role == "sensitivity" else None,
        coverage_scope=(scope_prefix + f"_{independence_profile}_fixed_design_comparison_family"),
        coverage_mode=spec.coverage_mode,
        family=spec.family,
        family_alpha=family_alpha,
        directional_family_alpha=directional_family_alpha,
        alpha_lower=calibration.endpoint_alpha,
        alpha_upper=calibration.endpoint_alpha,
        simultaneous_coverage=joint if spec.simultaneous else None,
        lower_one_sided_coverage=one_sided,
        upper_one_sided_coverage=one_sided,
        completion_estimate_lower=completion_lower,
        completion_estimate_upper=completion_upper,
        half_width_lower=half_width_lower,
        half_width_upper=half_width_upper,
        effective_sample_size=effective_sample_size,
        concentration_variance=concentration_variance,
        variance_upper_lower=variance_lower,
        variance_upper_upper=variance_upper,
        maximum_increment=maximum_increment,
        chernoff_kernel=chernoff_kernel,
        calibration_method=calibration.method,
        calibration_frontier=calibration.frontier_id,
        calibration_frontier_alpha=calibration.frontier_alpha,
        calibration_frontier_size=calibration.frontier_size,
        candidate_lower=candidate_lower,
        candidate_upper=candidate_upper,
        baseline_lower=baseline_lower,
        baseline_upper=baseline_upper,
        independence_profile=independence_profile,
        independent_block_count=independent_block_count,
        missingness_method="adverse_arm_specific_completion",
        assumptions=assumptions,
        eligibility=eligibility,
        warnings=interval_warnings,
    )


def _paired_comparison_diagnostics(
    spec: ComparisonSpec,
    data: NormalizedData,
    prompt_ids: set[str],
) -> tuple[int | None, dict[str, int], float | None, int | None, dict[str, Any], str | None]:
    """The paired diagnostics of a comparison over ``prompt_ids``.

    Returns the pair count, the discordant counts, the exact McNemar p-value, the cluster
    count, the outputs of each diagnostic method, and the reason newcombe_paired is not
    reported when it was asked for and cannot be computed, or None. Raises ValueError when
    the pairing or the plan's paired fields are broken.
    """
    if not spec.pair_key or not spec.diagnostic_methods:
        return None, {}, None, None, {}, None
    # Unscored rows stay in: a pair with an unscored arm is bounded by the completions of its
    # unknown outcome instead of breaking the pairing (L1). A key absent from one arm still does.
    arm_rows = {
        arm: [
            row
            for row in data.observations
            if row.cohort_id == arm and row.prompt_id in prompt_ids
        ]
        for arm in (spec.candidate, spec.baseline)
    }
    maps: dict[str, dict[tuple[Any, ...], Observation]] = {}
    for arm, rows in arm_rows.items():
        mapped: dict[tuple[Any, ...], Observation] = {}
        for row in rows:
            key = tuple(_observation_attr(row, field) for field in spec.pair_key or [])
            if key in mapped:
                raise ValueError(f"duplicate pair key in {arm}: {key!r}")
            mapped[key] = row
        maps[arm] = mapped
    candidate_keys = set(maps[spec.candidate])
    baseline_keys = set(maps[spec.baseline])
    if candidate_keys != baseline_keys:
        raise ValueError(
            f"broken pairing: {len(candidate_keys - baseline_keys)} candidate-only and "
            f"{len(baseline_keys - candidate_keys)} baseline-only pairs"
        )
    keys = sorted(candidate_keys, key=repr)
    cand_outcomes = [maps[spec.candidate][key].failure for key in keys]
    base_outcomes = [maps[spec.baseline][key].failure for key in keys]
    pairing = partial_pairing(cand_outcomes, base_outcomes)
    if pairing.scored_pair_count == pairing.pair_count:
        cand = [bool(value) for value in cand_outcomes]
        base = [bool(value) for value in base_outcomes]
        return _complete_paired_diagnostics(spec, maps, keys, cand, base)
    return _partial_paired_diagnostics(spec, maps, keys, cand_outcomes, base_outcomes, pairing)


def _complete_paired_diagnostics(
    spec: ComparisonSpec,
    maps: dict[str, dict[tuple[Any, ...], Observation]],
    keys: list[tuple[Any, ...]],
    cand: list[bool],
    base: list[bool],
) -> tuple[int | None, dict[str, int], float | None, int | None, dict[str, Any], str | None]:
    mcnemar = exact_mcnemar(cand, base)
    discordant = {
        "candidate_safe_baseline_failure": mcnemar.candidate_safe_baseline_failure,
        "candidate_failure_baseline_safe": mcnemar.candidate_failure_baseline_safe,
    }
    outputs: dict[str, Any] = {}
    p_value = None
    if "exact_mcnemar" in spec.diagnostic_methods:
        p_value = mcnemar.p_value_two_sided
        outputs["exact_mcnemar"] = {
            "estimate": mcnemar.estimate,
            "p_value_two_sided": mcnemar.p_value_two_sided,
            "p_value_candidate_worse": mcnemar.p_value_candidate_worse,
            "p_value_candidate_better": mcnemar.p_value_candidate_better,
        }
    if "newcombe_paired" in spec.diagnostic_methods:
        paired_interval = newcombe_paired_interval(cand, base)
        outputs["newcombe_paired"] = {
            "estimate": paired_interval.estimate,
            "lower": paired_interval.lower,
            "upper": paired_interval.upper,
            "alpha_lower": paired_interval.alpha_lower,
            "alpha_upper": paired_interval.alpha_upper,
            "method": paired_interval.method,
            "label": "diagnostic_requires_independent_pairs",
        }
    cluster_count = None
    if "fixed_cluster_bootstrap" in spec.diagnostic_methods:
        if not spec.cluster_key:
            raise ValueError("fixed_cluster_bootstrap requires cluster_key")
        differences = [float(c) - float(b) for c, b in zip(cand, base, strict=True)]
        clusters = [_observation_attr(maps[spec.candidate][key], spec.cluster_key) for key in keys]
        boot = paired_cluster_bootstrap(
            differences,
            clusters,
            replicates=spec.bootstrap_replicates,
            seed=spec.bootstrap_seed,
        )
        cluster_count = boot.cluster_count
        outputs["fixed_cluster_bootstrap"] = {
            "estimate": boot.estimate,
            "lower": boot.lower,
            "upper": boot.upper,
            "minimum_cluster_size": boot.minimum_cluster_size,
            "maximum_cluster_size": boot.maximum_cluster_size,
            "cluster_size_ratio": boot.cluster_size_ratio,
            "maximum_leave_one_cluster_out_shift": boot.maximum_leave_one_cluster_out_shift,
            "replicates": boot.replicates,
            "seed": boot.seed,
            "label": "diagnostic_fixed_design_resampling",
        }
    return len(keys), discordant, p_value, cluster_count, outputs, None


def _partial_paired_diagnostics(
    spec: ComparisonSpec,
    maps: dict[str, dict[tuple[Any, ...], Observation]],
    keys: list[tuple[Any, ...]],
    cand: list[bool | None],
    base: list[bool | None],
    pairing: PartialPairing,
) -> tuple[int | None, dict[str, int], float | None, int | None, dict[str, Any], str | None]:
    """Paired diagnostics when some pairs have an unscored arm.

    Every bound and p-value holds for each completion of the unscored outcomes. The point
    estimate is not identified, so each method reports estimate_lower and estimate_upper in its
    place. The discordant counts are those of the scored pairs. When the completions are too
    many for newcombe_paired to evaluate, it alone is left out, and the last value returned
    says why.
    """

    mcnemar = exact_mcnemar_partial(cand, base)
    scored = mcnemar.scored
    discordant = {
        "candidate_safe_baseline_failure": (
            scored.candidate_safe_baseline_failure if scored is not None else 0
        ),
        "candidate_failure_baseline_safe": (
            scored.candidate_failure_baseline_safe if scored is not None else 0
        ),
    }
    outputs: dict[str, Any] = {
        "partial_pairing": {
            "pair_count": pairing.pair_count,
            "scored_pair_count": pairing.scored_pair_count,
            "candidate_unscored_pair_count": pairing.candidate_unscored_pair_count,
            "baseline_unscored_pair_count": pairing.baseline_unscored_pair_count,
            "both_unscored_pair_count": pairing.both_unscored_pair_count,
            "completion": "every_completion_of_unscored_outcomes",
        }
    }
    p_value = None
    if "exact_mcnemar" in spec.diagnostic_methods:
        p_value = mcnemar.p_value_two_sided
        outputs["exact_mcnemar"] = {
            "estimate_lower": mcnemar.estimate_lower,
            "estimate_upper": mcnemar.estimate_upper,
            "p_value_two_sided": mcnemar.p_value_two_sided,
            "p_value_candidate_worse": mcnemar.p_value_candidate_worse,
            "p_value_candidate_better": mcnemar.p_value_candidate_better,
        }
    newcombe_withheld = None
    if "newcombe_paired" in spec.diagnostic_methods:
        try:
            paired_interval = newcombe_paired_interval_partial(cand, base)
        except CompletionLimitError as exc:
            newcombe_withheld = (
                f"{exc}, so newcombe_paired is not reported; the other paired diagnostics are"
            )
        else:
            outputs["newcombe_paired"] = {
                "estimate_lower": paired_interval.estimate_lower,
                "estimate_upper": paired_interval.estimate_upper,
                "lower": paired_interval.lower,
                "upper": paired_interval.upper,
                "alpha_lower": paired_interval.alpha_lower,
                "alpha_upper": paired_interval.alpha_upper,
                "method": paired_interval.method,
                "completion_count": paired_interval.completion_count,
                "label": "diagnostic_requires_independent_pairs",
            }
    cluster_count = None
    if "fixed_cluster_bootstrap" in spec.diagnostic_methods:
        if not spec.cluster_key:
            raise ValueError("fixed_cluster_bootstrap requires cluster_key")
        clusters = [_observation_attr(maps[spec.candidate][key], spec.cluster_key) for key in keys]
        # The resampling draws depend only on the clusters and the seed, so every replicate
        # mean, and with it every quantile, is lowest at the favourable completion and highest
        # at the adverse one.
        (fav_cand, fav_base), (adv_cand, adv_base) = favourable_and_adverse_completions(
            cand, base
        )
        favourable = paired_cluster_bootstrap(
            [float(c) - float(b) for c, b in zip(fav_cand, fav_base, strict=True)],
            clusters,
            replicates=spec.bootstrap_replicates,
            seed=spec.bootstrap_seed,
        )
        adverse = paired_cluster_bootstrap(
            [float(c) - float(b) for c, b in zip(adv_cand, adv_base, strict=True)],
            clusters,
            replicates=spec.bootstrap_replicates,
            seed=spec.bootstrap_seed,
        )
        cluster_count = favourable.cluster_count
        outputs["fixed_cluster_bootstrap"] = {
            "estimate_lower": favourable.estimate,
            "estimate_upper": adverse.estimate,
            "lower": favourable.lower,
            "upper": adverse.upper,
            "minimum_cluster_size": favourable.minimum_cluster_size,
            "maximum_cluster_size": favourable.maximum_cluster_size,
            "cluster_size_ratio": favourable.cluster_size_ratio,
            "maximum_leave_one_cluster_out_shift": max(
                favourable.maximum_leave_one_cluster_out_shift,
                adverse.maximum_leave_one_cluster_out_shift,
            ),
            "replicates": favourable.replicates,
            "seed": favourable.seed,
            "label": "diagnostic_fixed_design_resampling",
        }
    return len(keys), discordant, p_value, cluster_count, outputs, newcombe_withheld

