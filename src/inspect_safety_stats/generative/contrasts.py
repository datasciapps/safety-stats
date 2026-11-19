"""Arm contrasts and a simultaneous family of slices on a generative design.

Experimental. An *arm contrast* compares two values of one transformation level -- wrapper A
against wrapper B -- within one cohort. Each sampling unit carries one value of the level, so
the two arms hold different units and nothing pairs them unit by unit; what they share is the
seeds. The contrast holds the seed mix fixed:

\\[
\\Delta = T_a - T_b,\\qquad T_v = \\sum_{s \\in S} \\pi_s R_s(v),
\\]

with \\(S\\) the seeds whose subtree holds both values, \\(\\pi_s\\) the declared seed weights
renormalized over \\(S\\), and \\(R_s(v)\\) the declared-weight risk of the paths under seed
\\(s\\) that carry value \\(v\\). A context weight or constraint that makes one value likelier
under some seeds than others therefore cannot move the contrast. When every seed holds both
values with the same conditional weight, \\(T_v\\) is the slice risk and \\(\\Delta\\) the
difference of two slices.

A declared slice is the same construction with \\(\\pi_s\\) proportional to the weight of the
slice's paths under each seed, which makes \\(T_v\\) exactly the slice risk.

Each arm is bounded on its own, by the construction its sampling design licenses:

* ``direct`` -- conditional on the number of draws landing in the arm, those draws are
  independent and identically distributed from the declared distribution restricted to it.
  Each draw's outcome is multiplied by the known ratio \\(r = \\pi_s W_v / w(s, v)\\), which
  makes its mean \\(T_v\\); divided by the largest ratio it lies in [0, 1], so the Bernoulli-KL
  bound applies (Hoeffding 1963). When the ratio is constant it equals one, and the arm is
  bounded exactly like a slice.
* ``proportional_stratified`` -- the strata are seeds with fixed draw counts, and
  \\(T_v = \\sum_s \\pi_s w(s) / (w(s, v) n_s) \\sum_{j \\in s} 1\\{L_j = v\\} Y_j\\) is a
  fixed-coefficient weighted sum of independent [0, 1] observations, bounded by the weighted
  KL/Chernoff engine.

The two arms of a contrast are combined by a union bound. Simultaneous coverage over the
declared family comes from the tree's frontier calibration: slices of one level whose draws
are disjoint share a Sidak frontier, every other member is a frontier of its own, and
frontiers share the error budget by Bonferroni.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from inspect_safety_stats.generative.inference import (
    CoverageMode,
    UnitOutcome,
    _bounds,
    _point_estimate,
)
from inspect_safety_stats.generative.ledger import SelectionRecord
from inspect_safety_stats.generative.sampling import SamplingDesign
from inspect_safety_stats.generative.taxonomy import CompiledTaxonomy
from inspect_safety_stats.stats.direct_mean import weighted_bounded_kl_chernoff_mean_bounds
from inspect_safety_stats.stats.tree_calibration import (
    IndependentFrontier,
    any_overlap,
    calibrate_independent_frontiers,
)

RATIO_KL = "importance_ratio_bernoulli_kl"
STRATIFIED_HT = "stratified_horvitz_thompson_kl_chernoff"
RATIO_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SliceMember:
    """A declared slice: every path carrying ``level=value``."""

    level: str
    value: str

    @property
    def member_id(self) -> str:
        return f"{self.level}={self.value}"


@dataclass(frozen=True)
class ContrastMember:
    """A declared contrast between two values of one level."""

    id: str
    level: str
    candidate: str
    baseline: str


@dataclass(frozen=True)
class ArmEstimate:
    """The seed-weighted risk of one value of a level."""

    level: str
    value: str
    seeds: int
    draws: int
    trials: int
    failures: int
    unclassified: int
    estimate: float | None
    lower: float
    upper: float
    method: str
    alpha_lower: float
    alpha_upper: float
    maximum_ratio: float


@dataclass(frozen=True)
class FamilySlice:
    """A declared slice, bounded at the level its frontier in the family gives it."""

    member_id: str
    level: str
    value: str
    weight: float
    arm: ArmEstimate
    frontier_id: str
    frontier_size: int
    endpoint_alpha: float


@dataclass(frozen=True)
class ContrastResult:
    """A seed-matched arm contrast, its interval and the two arms behind it."""

    id: str
    level: str
    candidate_value: str
    baseline_value: str
    estimand: str
    shared_seeds: int
    estimate: float | None
    lower: float
    upper: float
    alpha: float
    coverage_mode: CoverageMode
    frontier_id: str
    frontier_size: int
    endpoint_alpha: float
    candidate: ArmEstimate
    baseline: ArmEstimate


@dataclass(frozen=True)
class SliceFamilyResult:
    """Every declared slice and contrast, covered simultaneously at ``1 - alpha``."""

    alpha: float
    coverage_mode: CoverageMode
    slices: tuple[FamilySlice, ...]
    contrasts: tuple[ContrastResult, ...]


def _value_of(node_id: str, level: str) -> str | None:
    for part in node_id.split("|")[1:]:
        name, _, value = part.partition("=")
        if name == level:
            return value
    return None


@dataclass(frozen=True)
class _Geometry:
    """Declared weights by seed: of the seed, and of each value of one level under it."""

    seed: dict[str, float]
    by_value: dict[str, dict[str, float]]


def _geometry(taxonomy: CompiledTaxonomy, level: str) -> _Geometry:
    seeds_at = taxonomy.levels.index(taxonomy.semantic_seed_level)
    if level not in taxonomy.levels:
        raise ValueError(f"{level!r} is not a level of this taxonomy")
    if taxonomy.levels.index(level) <= seeds_at:
        raise ValueError(
            f"{level!r} is at or above the semantic seed level {taxonomy.semantic_seed_level!r}; "
            "slices and contrasts in the family are over arm levels, which lie below it"
        )
    seed: dict[str, float] = {}
    by_value: dict[str, dict[str, float]] = {}
    for node in taxonomy.terminal_paths:
        seed_id = str(node.semantic_seed_id)
        seed[seed_id] = seed.get(seed_id, 0.0) + node.target_path_weight
        value = _value_of(node.node_id, level)
        if value is None:
            raise ValueError(f"terminal path {node.node_id} carries no value of {level!r}")
        cell = by_value.setdefault(value, {})
        cell[seed_id] = cell.get(seed_id, 0.0) + node.target_path_weight
    return _Geometry(seed=seed, by_value=by_value)


def _cells(geometry: _Geometry, level: str, value: str) -> dict[str, float]:
    cells = geometry.by_value.get(value)
    if not cells:
        raise ValueError(f"no terminal path carries {level}={value}")
    return {seed: weight for seed, weight in cells.items() if weight > 0.0}


def _unevaluated(unit_id: str) -> UnitOutcome:
    return UnitOutcome(unit_id, trials=1, failures=0, unclassified=1)


def _arm(
    design: SamplingDesign,
    selections: Sequence[SelectionRecord],
    by_unit: dict[str, UnitOutcome],
    geometry: _Geometry,
    *,
    level: str,
    value: str,
    seed_weights: dict[str, float],
    alpha_lower: float,
    alpha_upper: float,
    one_sided: bool,
) -> tuple[ArmEstimate, frozenset[str]]:
    """Bound ``sum_s seed_weights[s] * R_s(value)``; also return the units it reads."""

    cells = _cells(geometry, level, value)
    missing = sorted(seed for seed in seed_weights if seed not in cells)
    if missing:
        raise ValueError(f"seed {missing[0]} holds no path with {level}={value}")
    in_arm = [
        record
        for record in selections
        if record.semantic_seed_id in seed_weights and _value_of(record.node_id, level) == value
    ]
    if not in_arm:
        raise ValueError(f"no draw landed on {level}={value}")
    arm_outcomes = [
        by_unit.get(r.sampling_unit_id, _unevaluated(r.sampling_unit_id)) for r in in_arm
    ]
    counts = {
        "draws": len(in_arm),
        "trials": sum(item.trials for item in arm_outcomes),
        "failures": sum(item.failures for item in arm_outcomes),
        "unclassified": sum(item.unclassified for item in arm_outcomes),
    }

    if design.method == "direct":
        mass = sum(cells[seed] for seed in seed_weights)
        ratio_by_seed = {seed: pi * mass / cells[seed] for seed, pi in seed_weights.items()}
        ratios = np.array(list(ratio_by_seed.values()), dtype=float)
        maximum = float(ratios.max())
        if np.allclose(ratios, 1.0, rtol=0.0, atol=RATIO_TOLERANCE):
            # Constant ratio: the arm is a plain slice of the restricted distribution.
            equal = np.full(len(arm_outcomes), 1.0 / len(arm_outcomes))
            lower, upper, method, _ = _bounds(
                arm_outcomes, equal, alpha_lower=alpha_lower, alpha_upper=alpha_upper,
                one_sided=one_sided,
            )
            estimate = _point_estimate(arm_outcomes, equal)
            maximum = 1.0
        else:
            scaled = np.array(
                [ratio_by_seed[record.semantic_seed_id] / maximum for record in in_arm]
            )
            low = np.array([o.failures / o.trials for o in arm_outcomes]) * scaled
            high = scaled * np.array(
                [(o.failures + o.unclassified) / o.trials for o in arm_outcomes]
            )
            bounds = weighted_bounded_kl_chernoff_mean_bounds(
                low, high, np.full(len(in_arm), 1.0 / len(in_arm)),
                alpha_lower=alpha_lower, alpha_upper=alpha_upper,
                independence_profile="sampling_unit",
            )
            lower = 0.0 if one_sided else max(0.0, bounds.lower * maximum)
            upper = min(1.0, bounds.upper * maximum)
            method = RATIO_KL
            classified = [
                (ratio_by_seed[r.semantic_seed_id], o)
                for r, o in zip(in_arm, arm_outcomes, strict=True)
                if o.trials > o.unclassified
            ]
            estimate = (
                sum(ratio * o.failures / (o.trials - o.unclassified) for ratio, o in classified)
                / len(classified)
                if classified
                else None
            )
        reads = frozenset(record.sampling_unit_id for record in in_arm)
    else:
        per_seed: dict[str, list[SelectionRecord]] = {}
        for record in selections:
            if record.semantic_seed_id in seed_weights:
                per_seed.setdefault(record.semantic_seed_id, []).append(record)
        unsampled = sorted(set(seed_weights) - set(per_seed))
        if unsampled:
            raise ValueError(f"stratum for seed {unsampled[0]} received no draws")
        multiplier = {
            seed: seed_weights[seed] * geometry.seed[seed] / cells[seed] for seed in seed_weights
        }
        records = [record for seed in sorted(per_seed) for record in per_seed[seed]]
        coefficients = np.array(
            [multiplier[r.semantic_seed_id] / len(per_seed[r.semantic_seed_id]) for r in records]
        )
        total = float(coefficients.sum())
        low_values, high_values = [], []
        for record in records:
            outcome = by_unit.get(record.sampling_unit_id, _unevaluated(record.sampling_unit_id))
            hit = _value_of(record.node_id, level) == value
            low_values.append(outcome.failures / outcome.trials if hit else 0.0)
            high_values.append(
                (outcome.failures + outcome.unclassified) / outcome.trials if hit else 0.0
            )
        bounds = weighted_bounded_kl_chernoff_mean_bounds(
            np.array(low_values), np.array(high_values), coefficients / total,
            alpha_lower=alpha_lower, alpha_upper=alpha_upper,
            independence_profile="sampling_unit",
        )
        lower = 0.0 if one_sided else max(0.0, bounds.lower * total)
        upper = min(1.0, bounds.upper * total)
        method = STRATIFIED_HT
        maximum = float(max(multiplier.values()))
        estimate = 0.0
        for seed, members in sorted(per_seed.items()):
            seen = [
                by_unit.get(r.sampling_unit_id, _unevaluated(r.sampling_unit_id)) for r in members
            ]
            hits = [
                (o, _value_of(r.node_id, level) == value)
                for r, o in zip(members, seen, strict=True)
                if o.trials > o.unclassified
            ]
            if not hits:
                estimate = None
                break
            share = sum(
                o.failures / (o.trials - o.unclassified) for o, hit in hits if hit
            ) / len(hits)
            estimate += multiplier[seed] * share
        reads = frozenset(record.sampling_unit_id for record in records)

    return (
        ArmEstimate(
            level=level,
            value=value,
            seeds=len(seed_weights),
            estimate=estimate,
            lower=lower,
            upper=upper,
            method=method,
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            maximum_ratio=maximum,
            **counts,
        ),
        reads,
    )


def _slice_seed_weights(geometry: _Geometry, member: SliceMember) -> tuple[dict[str, float], float]:
    cells = _cells(geometry, member.level, member.value)
    mass = sum(cells.values())
    return {seed: weight / mass for seed, weight in sorted(cells.items())}, mass


def _shared_seed_weights(geometry: _Geometry, member: ContrastMember) -> dict[str, float]:
    if member.candidate == member.baseline:
        raise ValueError(f"contrast {member.id!r} compares {member.candidate!r} with itself")
    candidate = _cells(geometry, member.level, member.candidate)
    baseline = _cells(geometry, member.level, member.baseline)
    shared = sorted(set(candidate) & set(baseline))
    if not shared:
        raise ValueError(
            f"contrast {member.id!r}: no semantic seed holds both {member.candidate!r} and "
            f"{member.baseline!r}"
        )
    mass = sum(geometry.seed[seed] for seed in shared)
    return {seed: geometry.seed[seed] / mass for seed in shared}


def check_slice_family(
    taxonomy: CompiledTaxonomy,
    design: SamplingDesign,
    selections: Sequence[SelectionRecord],
    *,
    slices: Sequence[SliceMember] = (),
    contrasts: Sequence[ContrastMember] = (),
) -> None:
    """Raise what :func:`estimate_slice_family` would raise on these selections.

    Whether a member can be estimated depends on the compiled taxonomy and the selected
    paths alone: an unknown value, an arm no draw lands on, two values that share no seed.
    Outcomes never decide it, because a unit without one is read as unevaluated. Running
    the family over the selections with every unit unevaluated therefore refuses exactly
    what the analysis of the evaluated run would, and can run before anything is generated.
    """

    estimate_slice_family(taxonomy, design, selections, (), slices=slices, contrasts=contrasts)


def estimate_slice_family(
    taxonomy: CompiledTaxonomy,
    design: SamplingDesign,
    selections: Sequence[SelectionRecord],
    outcomes: Sequence[UnitOutcome],
    *,
    slices: Sequence[SliceMember] = (),
    contrasts: Sequence[ContrastMember] = (),
    alpha: float = 0.05,
    coverage_mode: CoverageMode = "one_sided_upper",
) -> SliceFamilyResult:
    """Bound every declared slice and arm contrast simultaneously at ``1 - alpha``.

    Coverage is per direction under ``two_sided``, as for the simultaneous tree. A member
    that cannot be estimated -- an unknown value, no draw in an arm, no shared seed -- is an
    error, never a silent omission, because dropping it would change the family the error
    budget was spent on. :func:`check_slice_family` raises the same error before generation.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not slices and not contrasts:
        raise ValueError("a slice family needs at least one slice or contrast")
    if len(set(slices)) != len(slices):
        raise ValueError("a slice family declares the same slice twice")
    if len({item.id for item in contrasts}) != len(contrasts):
        raise ValueError("a slice family declares the same contrast id twice")

    by_unit = {outcome.sampling_unit_id: outcome for outcome in outcomes}
    one_sided = coverage_mode == "one_sided_upper"
    geometries: dict[str, _Geometry] = {}

    def geometry(level: str) -> _Geometry:
        if level not in geometries:
            geometries[level] = _geometry(taxonomy, level)
        return geometries[level]

    # The units each member reads, needed before any bound to lay out the frontiers. The
    # endpoint alpha does not change which units are read, so a provisional pass at alpha
    # gives them.
    def reads(level: str, value: str, weights: dict[str, float]) -> frozenset[str]:
        _, units = _arm(
            design, selections, by_unit, geometry(level), level=level, value=value,
            seed_weights=weights, alpha_lower=alpha, alpha_upper=alpha, one_sided=one_sided,
        )
        return units

    slice_weights = {item: _slice_seed_weights(geometry(item.level), item) for item in slices}
    contrast_weights = {
        item.id: _shared_seed_weights(geometry(item.level), item) for item in contrasts
    }

    frontiers: list[IndependentFrontier] = []
    by_level: dict[str, dict[str, frozenset[str]]] = {}
    for item in slices:
        by_level.setdefault(item.level, {})[f"slice:{item.member_id}"] = reads(
            item.level, item.value, slice_weights[item][0]
        )
    for level, targets in by_level.items():
        if any_overlap(targets.values()):
            # Draws read by more than one slice: their coverage events are not independent.
            for target_id, units in targets.items():
                frontiers.append(IndependentFrontier(target_id, {target_id: units}, {}))
        else:
            frontiers.append(IndependentFrontier(f"slices:{level}", targets, {}))
    for contrast in contrasts:
        weights = contrast_weights[contrast.id]
        units = reads(contrast.level, contrast.candidate, weights) | reads(
            contrast.level, contrast.baseline, weights
        )
        target_id = f"contrast:{contrast.id}"
        frontiers.append(IndependentFrontier(target_id, {target_id: units}, {}))
    calibration = calibrate_independent_frontiers(frontiers, directional_family_alpha=alpha)

    def endpoint(target_id: str) -> float:
        local = calibration[target_id].endpoint_alpha
        return local if one_sided else local / 2.0

    slice_results = []
    for item in slices:
        target_id = f"slice:{item.member_id}"
        weights, mass = slice_weights[item]
        side = endpoint(target_id)
        arm, _ = _arm(
            design, selections, by_unit, geometry(item.level), level=item.level,
            value=item.value, seed_weights=weights, alpha_lower=side, alpha_upper=side,
            one_sided=one_sided,
        )
        slice_results.append(
            FamilySlice(
                member_id=item.member_id,
                level=item.level,
                value=item.value,
                weight=mass,
                arm=arm,
                frontier_id=calibration[target_id].frontier_id,
                frontier_size=calibration[target_id].frontier_size,
                endpoint_alpha=side,
            )
        )

    contrast_results = []
    for contrast in contrasts:
        target_id = f"contrast:{contrast.id}"
        weights = contrast_weights[contrast.id]
        side = endpoint(target_id)
        # A difference has two arms, so each side of the interval is a union over both.
        arms = [
            _arm(
                design, selections, by_unit, geometry(contrast.level), level=contrast.level,
                value=value, seed_weights=weights, alpha_lower=side / 2.0,
                alpha_upper=side / 2.0, one_sided=False,
            )[0]
            for value in (contrast.candidate, contrast.baseline)
        ]
        candidate, baseline = arms
        contrast_results.append(
            ContrastResult(
                id=contrast.id,
                level=contrast.level,
                candidate_value=contrast.candidate,
                baseline_value=contrast.baseline,
                estimand="seed_matched_risk_difference",
                shared_seeds=len(weights),
                estimate=(
                    None
                    if candidate.estimate is None or baseline.estimate is None
                    else candidate.estimate - baseline.estimate
                ),
                lower=-1.0 if one_sided else max(-1.0, candidate.lower - baseline.upper),
                upper=min(1.0, candidate.upper - baseline.lower),
                alpha=alpha,
                coverage_mode=coverage_mode,
                frontier_id=calibration[target_id].frontier_id,
                frontier_size=calibration[target_id].frontier_size,
                endpoint_alpha=side,
                candidate=candidate,
                baseline=baseline,
            )
        )

    return SliceFamilyResult(
        alpha=alpha,
        coverage_mode=coverage_mode,
        slices=tuple(slice_results),
        contrasts=tuple(contrast_results),
    )
