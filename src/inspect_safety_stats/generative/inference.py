"""Node and root inference for a generative design.

Path selection is independent of outcomes, so conditional on the number of draws that
landed in a node, those draws are independent observations from that node's conditional
distribution. Exact binomial bounds are therefore valid conditionally, and a bound valid
for every realized count is valid unconditionally.

The same argument licenses Sidak calibration across siblings: conditional on the count
vector, sibling nodes hold disjoint sets of independent draws, so their coverage events
are independent and the product bound holds for every realization. It also licenses
reporting only the nodes that received draws -- that selection rule reads the counts,
which are outcome-independent, and never the failures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from inspect_safety_stats.generative.ledger import SelectionRecord
from inspect_safety_stats.generative.sampling import SamplingDesign
from inspect_safety_stats.generative.taxonomy import CompiledTaxonomy
from inspect_safety_stats.stats.binomial import (
    binomial_upper_bound,
    missingness_robust_binomial_bounds,
)
from inspect_safety_stats.stats.direct_mean import (
    _bernoulli_kl,
    _invert_divergence,
    _poisson_deviance,
    weighted_kl_chernoff_mean_bounds,
)
from inspect_safety_stats.stats.tree_calibration import (
    IndependentFrontier,
    calibrate_independent_frontiers,
)

CoverageMode = Literal["one_sided_upper", "two_sided"]
EXACT_METHOD = "clopper_pearson_one_sided_missingness_robust"
WEIGHTED_METHOD = "weighted_kl_chernoff"
COEFFICIENT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class UnitOutcome:
    """One sampling unit's evaluated rollouts.

    A unit whose rewrite never passed validation is not dropped: it is recorded with
    every rollout unclassified, so it is completed as safe for the lower endpoint and
    as a failure for the upper. Dropping it would silently alter the path distribution.
    """

    sampling_unit_id: str
    trials: int
    failures: int
    unclassified: int

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("a sampling unit must have at least one attempted rollout")
        if self.failures < 0 or self.unclassified < 0:
            raise ValueError("counts must be non-negative")
        if self.failures + self.unclassified > self.trials:
            raise ValueError("failures plus unclassified cannot exceed trials")


@dataclass(frozen=True)
class NodeEstimate:
    """The failure-rate estimate and interval for one node or cross-cutting slice.

    ``estimate`` is the design-weighted mean over classified rollouts, None when none were
    classified; ``raw_failure_rate`` is the unweighted rate. ``lower`` is 0 when only an
    upper bound was computed. ``method`` names the kernel used, and ``frontier_id`` is set
    only under simultaneous calibration. Slice ids start with ``slice:``.
    """

    node_id: str
    level: str
    depth: int
    node_weight: float
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
    frontier_id: str | None
    equal_coefficients: bool
    maximum_coefficient: float
    coefficient_ratio: float
    effective_sample_size: float
    raw_failure_rate: float | None
    unclassified_share: float


@dataclass(frozen=True)
class TreeEstimate:
    """The root and per-node estimates that :func:`estimate_tree` returns.

    ``nodes`` holds only nodes that received draws, and the root alone when
    ``report_all_nodes`` is off; ``unsampled_nodes`` lists the ids that received none.
    ``simultaneous`` records whether the tree was calibrated as one family. See "Why the
    intervals are exact" in ``docs/generative-evaluation.md``.
    """

    root: NodeEstimate
    nodes: tuple[NodeEstimate, ...]
    unsampled_nodes: tuple[str, ...]
    alpha: float
    coverage_mode: CoverageMode
    simultaneous: bool

    def node(self, node_id: str) -> NodeEstimate:
        for candidate in self.nodes:
            if candidate.node_id == node_id:
                return candidate
        raise KeyError(node_id)


def _ancestors(node_id: str) -> list[str]:
    parts = node_id.split("|")
    return ["|".join(parts[: index + 1]) for index in range(len(parts))]


def _raw_coefficients(
    design: SamplingDesign, selections: list[SelectionRecord]
) -> dict[str, float]:
    """The deterministic weight each sampling unit carries in the root estimator."""

    if design.method == "direct":
        share = 1.0 / len(selections)
        return {record.sampling_unit_id: share for record in selections}
    draws = design.draws_by_stratum
    weights = design.weight_by_stratum
    coefficients: dict[str, float] = {}
    for record in selections:
        stratum = record.stratum
        if stratum is None:
            raise ValueError(
                f"selection {record.sampling_unit_id} carries no stratum under a "
                "stratified design"
            )
        if stratum not in draws:
            raise ValueError(f"selection {record.sampling_unit_id} names unknown stratum {stratum}")
        coefficients[record.sampling_unit_id] = weights[stratum] / draws[stratum]
    return coefficients


def _effective_sample_size(coefficients: np.ndarray) -> float:
    """Concentration geometry of the estimator's weights, 1 / sum of squares.

    Equals the draw count when every draw carries the same coefficient, and falls as the
    weight concentrates. It measures weight geometry, not information: under a sampled
    design precision is governed by how many draws there are, not by this number.
    """

    return float(1.0 / np.sum(coefficients**2))


def _raw_failure_rate(outcomes: list[UnitOutcome]) -> float | None:
    """The unweighted rate among classified rollouts.

    Reported separately from ``estimate`` because they are different quantities: the
    estimate applies the declared design weights, this does not. They coincide only when
    every draw carries the same coefficient.
    """

    classified = sum(item.trials - item.unclassified for item in outcomes)
    if classified <= 0:
        return None
    return sum(item.failures for item in outcomes) / classified


def _unclassified_share(outcomes: list[UnitOutcome], coefficients: np.ndarray) -> float:
    """The design weight the bounds complete as an unknown outcome.

    Each unit contributes its coefficient times the share of its rollouts with no
    verdict. The bounds count that mass as safe at the lower end and as failing at the
    upper, and a judge correction must keep it there: labels say nothing about it.
    """

    share = sum(
        float(coefficient) * outcome.unclassified / outcome.trials
        for outcome, coefficient in zip(outcomes, coefficients, strict=True)
    )
    return min(1.0, share)


def unclassified_upper_bound(estimate: NodeEstimate, *, alpha: float) -> float:
    """A one-sided upper confidence bound on the design's share of unclassified rollouts.

    ``estimate.unclassified_share`` is the run's own share; the design's share is the
    parameter a judge correction needs, and the run's share only estimates it. Each draw's
    share of unclassified rollouts is a [0, 1] value, independent across draws as the node's
    own interval assumes, so the bound is the one that interval's kernel gives: exact
    Clopper-Pearson on the unclassified count when every draw carries the same coefficient
    and one rollout, and otherwise the KL/Chernoff bound for independent bounded blocks --
    Bernoulli KL over the draws when the coefficients are equal, the Poisson exponent of the
    largest coefficient when they are not.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if estimate.equal_coefficients and estimate.trials == estimate.draws:
        return binomial_upper_bound(estimate.unclassified, estimate.trials, alpha=alpha)
    observed = min(1.0, max(0.0, estimate.unclassified_share))
    if estimate.equal_coefficients:
        target = math.log(1 / alpha) / estimate.draws
        bound = _invert_divergence(observed, target, direction="upper", divergence=_bernoulli_kl)
    else:
        target = estimate.maximum_coefficient * math.log(1 / alpha)
        bound = _invert_divergence(
            observed, target, direction="upper", divergence=_poisson_deviance
        )
    return min(1.0, bound)


def _point_estimate(
    outcomes: list[UnitOutcome], coefficients: np.ndarray
) -> float | None:
    """The design-weighted mean over classified rollouts.

    Unclassified rollouts are excluded rather than completed, so this is a point
    estimate and not an endpoint; the missingness is carried by the bounds.
    """

    total = 0.0
    mass = 0.0
    for outcome, coefficient in zip(outcomes, coefficients, strict=True):
        classified = outcome.trials - outcome.unclassified
        if classified <= 0:
            continue
        total += float(coefficient) * outcome.failures / classified
        mass += float(coefficient)
    return total / mass if mass > 0.0 else None


def _bounds(
    outcomes: list[UnitOutcome],
    coefficients: np.ndarray,
    *,
    alpha_lower: float,
    alpha_upper: float,
    one_sided: bool,
) -> tuple[float, float, str, bool]:
    equal = bool(np.allclose(coefficients, coefficients[0], rtol=COEFFICIENT_TOLERANCE, atol=0.0))
    single_rollout = all(outcome.trials == 1 for outcome in outcomes)
    failures = sum(outcome.failures for outcome in outcomes)
    unclassified = sum(outcome.unclassified for outcome in outcomes)
    trials = sum(outcome.trials for outcome in outcomes)

    if equal and single_rollout:
        if one_sided:
            upper = binomial_upper_bound(failures + unclassified, trials, alpha=alpha_upper)
            return 0.0, upper, EXACT_METHOD, equal
        exact = missingness_robust_binomial_bounds(
            failures,
            trials - unclassified,
            trials,
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
        )
        return exact.lower, exact.upper, EXACT_METHOD, equal

    weighted = weighted_kl_chernoff_mean_bounds(
        np.array([outcome.failures for outcome in outcomes], dtype=int),
        np.array([outcome.unclassified for outcome in outcomes], dtype=int),
        np.array([outcome.trials for outcome in outcomes], dtype=int),
        coefficients,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_unit="prompt_cluster",
    )
    return (0.0 if one_sided else weighted.lower), weighted.upper, WEIGHTED_METHOD, equal


def estimate_tree(
    taxonomy: CompiledTaxonomy,
    design: SamplingDesign,
    selections: list[SelectionRecord],
    outcomes: list[UnitOutcome],
    *,
    alpha: float = 0.05,
    coverage_mode: CoverageMode = "one_sided_upper",
    simultaneous_tree: bool = False,
    report_all_nodes: bool = True,
) -> TreeEstimate:
    """Estimate the root and every sampled node of a compiled generative taxonomy.

    With ``simultaneous_tree`` the root shares the error budget with each hierarchy
    depth by Bonferroni, so its interval widens. A user who needs the tightest root
    should leave it off and read the root alone, which is then covered at ``1 - alpha``
    with the node intervals labelled pointwise.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not selections:
        raise ValueError("no selections were supplied")

    by_unit = {outcome.sampling_unit_id: outcome for outcome in outcomes}
    unknown = sorted(set(by_unit) - {record.sampling_unit_id for record in selections})
    if unknown:
        raise ValueError(
            f"{len(unknown)} outcomes reference unselected sampling units, "
            f"starting with {unknown[0]!r}"
        )

    coefficients = _raw_coefficients(design, selections)
    members: dict[str, list[str]] = {}
    for record in selections:
        for ancestor in _ancestors(record.node_id):
            members.setdefault(ancestor, []).append(record.sampling_unit_id)

    reported = [taxonomy.root_id] if not report_all_nodes else None
    calibration = {}
    if simultaneous_tree:
        frontiers: dict[int, dict[str, frozenset[str]]] = {}
        for node in taxonomy.nodes:
            if node.node_id not in members:
                continue
            if reported is not None and node.node_id not in reported:
                continue
            frontiers.setdefault(node.depth, {})[node.node_id] = frozenset(members[node.node_id])
        calibration = calibrate_independent_frontiers(
            [
                IndependentFrontier(frontier_id=f"depth-{depth}", targets=targets, aliases={})
                for depth, targets in sorted(frontiers.items())
            ],
            directional_family_alpha=alpha,
        )

    one_sided = coverage_mode == "one_sided_upper"
    estimates: list[NodeEstimate] = []
    unsampled: list[str] = []
    for node in taxonomy.nodes:
        unit_ids = members.get(node.node_id)
        if not unit_ids:
            unsampled.append(node.node_id)
            continue
        if reported is not None and node.node_id not in reported:
            continue

        local = calibration[node.node_id].endpoint_alpha if simultaneous_tree else alpha
        alpha_lower = local if one_sided else local / 2.0
        alpha_upper = local if one_sided else local / 2.0

        raw = np.array([coefficients[unit_id] for unit_id in unit_ids], dtype=float)
        normalized = raw / raw.sum()
        node_outcomes = [
            by_unit.get(unit_id, UnitOutcome(unit_id, trials=1, failures=0, unclassified=1))
            for unit_id in unit_ids
        ]
        lower, upper, method, equal = _bounds(
            node_outcomes,
            normalized,
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            one_sided=one_sided,
        )
        estimates.append(
            NodeEstimate(
                node_id=node.node_id,
                level=node.level,
                depth=node.depth,
                node_weight=node.target_path_weight,
                draws=len(unit_ids),
                trials=sum(outcome.trials for outcome in node_outcomes),
                failures=sum(outcome.failures for outcome in node_outcomes),
                unclassified=sum(outcome.unclassified for outcome in node_outcomes),
                estimate=_point_estimate(node_outcomes, normalized),
                lower=lower,
                upper=upper,
                method=method,
                alpha_lower=alpha_lower,
                alpha_upper=alpha_upper,
                frontier_id=(
                    calibration[node.node_id].frontier_id if simultaneous_tree else None
                ),
                equal_coefficients=equal,
                maximum_coefficient=float(normalized.max()),
                coefficient_ratio=float(normalized.max() / normalized.min()),
                effective_sample_size=_effective_sample_size(normalized),
                raw_failure_rate=_raw_failure_rate(node_outcomes),
        unclassified_share=_unclassified_share(node_outcomes, normalized),
            )
        )

    root = next(item for item in estimates if item.node_id == taxonomy.root_id)
    return TreeEstimate(
        root=root,
        nodes=tuple(estimates),
        unsampled_nodes=tuple(unsampled),
        alpha=alpha,
        coverage_mode=coverage_mode,
        simultaneous=simultaneous_tree,
    )


def estimate_slice(
    taxonomy: CompiledTaxonomy,
    design: SamplingDesign,
    selections: list[SelectionRecord],
    outcomes: list[UnitOutcome],
    *,
    level: str,
    value: str,
    alpha: float = 0.05,
    coverage_mode: CoverageMode = "one_sided_upper",
) -> NodeEstimate:
    """Estimate a cross-cutting slice of the design, such as every Maltese path.

    A hierarchy answers questions about subtrees. "What is the risk in Maltese?" is not
    one of those when the semantic seed sits above language, because the Maltese paths
    are spread across every seed. The estimand is still well defined -- a declared-weight
    mean over the matching terminal paths -- and the draws landing in it are still
    independent conditional on their count, so the same exact construction applies.

    Slices overlap the tree's nodes, so they cannot join its Sidak frontiers. They are
    reported pointwise at ``alpha`` and are not part of a simultaneous family.
    """

    if level not in taxonomy.levels:
        raise ValueError(f"{level!r} is not a level of this taxonomy")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    def matches(node_id: str) -> bool:
        return any(
            part == f"{level}={value}" for part in node_id.split("|")[1:]
        )

    weight = sum(
        node.target_path_weight for node in taxonomy.terminal_paths if matches(node.node_id)
    )
    if weight <= 0.0:
        raise ValueError(f"no terminal path carries {level}={value}")

    coefficients = _raw_coefficients(design, selections)
    unit_ids = [
        record.sampling_unit_id for record in selections if matches(record.node_id)
    ]
    if not unit_ids:
        raise ValueError(f"no draw landed on {level}={value}")

    by_unit = {outcome.sampling_unit_id: outcome for outcome in outcomes}
    raw = np.array([coefficients[unit_id] for unit_id in unit_ids], dtype=float)
    normalized = raw / raw.sum()
    slice_outcomes = [
        by_unit.get(unit_id, UnitOutcome(unit_id, trials=1, failures=0, unclassified=1))
        for unit_id in unit_ids
    ]
    one_sided = coverage_mode == "one_sided_upper"
    endpoint = alpha if one_sided else alpha / 2.0
    lower, upper, method, equal = _bounds(
        slice_outcomes,
        normalized,
        alpha_lower=endpoint,
        alpha_upper=endpoint,
        one_sided=one_sided,
    )
    return NodeEstimate(
        node_id=f"slice:{level}={value}",
        level=level,
        depth=taxonomy.levels.index(level) + 1,
        node_weight=weight,
        draws=len(unit_ids),
        trials=sum(outcome.trials for outcome in slice_outcomes),
        failures=sum(outcome.failures for outcome in slice_outcomes),
        unclassified=sum(outcome.unclassified for outcome in slice_outcomes),
        estimate=_point_estimate(slice_outcomes, normalized),
        lower=lower,
        upper=upper,
        method=method,
        alpha_lower=endpoint,
        alpha_upper=endpoint,
        frontier_id=None,
        equal_coefficients=equal,
        maximum_coefficient=float(normalized.max()),
        coefficient_ratio=float(normalized.max() / normalized.min()),
        effective_sample_size=_effective_sample_size(normalized),
        raw_failure_rate=_raw_failure_rate(slice_outcomes),
        unclassified_share=_unclassified_share(slice_outcomes, normalized),
    )


@dataclass(frozen=True)
class ReweightedEstimate:
    """The root under an alternative declared design, from the same draws.

    Reported as a **diagnostic**, never as evidence, for two reasons. The draws were
    allocated for the declared design, so reweighting them is importance sampling and the
    coefficients become unequal -- the interval widens, and ``effective_sample_size`` shows
    how much the alternative strained the sample. And the construction is self-normalized,
    so its denominator is random and the interval is not a certified bound on the
    alternative estimand.

    What it answers is the question worth asking: would the conclusion survive a different
    but defensible declaration?
    """

    id: str
    level: str
    weights: dict[str, float]
    draws: int
    estimate: float | None
    lower: float
    upper: float
    method: str
    effective_sample_size: float
    maximum_coefficient: float
    declared_lower: float
    declared_upper: float
    eligibility: Literal["diagnostic"] = "diagnostic"


def estimate_reweighted(
    taxonomy: CompiledTaxonomy,
    design: SamplingDesign,
    selections: list[SelectionRecord],
    outcomes: list[UnitOutcome],
    root: NodeEstimate,
    *,
    identifier: str,
    level: str,
    weights: Mapping[str, float],
    alpha: float = 0.05,
    coverage_mode: CoverageMode = "one_sided_upper",
) -> ReweightedEstimate:
    """Re-estimate the root under a different marginal mix at one level.

    The reweighting names a target *marginal* share for each value of the level, holding
    everything else's conditional structure fixed. Each draw is then carried at
    ``target(value) / realized(value)``, which is unbiased for the alternative estimand.
    """

    if level not in taxonomy.levels:
        raise ValueError(f"{level!r} is not a level of this taxonomy")

    def value_of(node_id: str) -> str | None:
        for part in node_id.split("|")[1:]:
            name, _, value = part.partition("=")
            if name == level:
                return value
        return None

    realized: dict[str, float] = {}
    for node in taxonomy.terminal_paths:
        value = value_of(node.node_id)
        if value is not None:
            realized[value] = realized.get(value, 0.0) + node.target_path_weight

    unknown = sorted(set(weights) - set(realized))
    if unknown:
        raise ValueError(
            f"reweighting names values absent from level {level!r}: {', '.join(unknown)}"
        )
    missing = sorted(set(realized) - set(weights))
    if missing:
        raise ValueError(
            f"reweighting omits values of level {level!r}: {', '.join(missing)}; a marginal "
            "mix must cover every declared value"
        )

    factors = {
        value: (weights[value] / share if share > 0 else 0.0)
        for value, share in realized.items()
    }
    base = _raw_coefficients(design, selections)
    by_unit = {outcome.sampling_unit_id: outcome for outcome in outcomes}

    unit_ids, raw = [], []
    for record in selections:
        value = value_of(record.node_id)
        if value is None:
            continue
        weight = base[record.sampling_unit_id] * factors[value]
        if weight <= 0.0:
            continue
        unit_ids.append(record.sampling_unit_id)
        raw.append(weight)
    if not unit_ids:
        raise ValueError(f"no draw carries positive weight under reweighting {identifier!r}")

    coefficients = np.array(raw, dtype=float)
    normalized = coefficients / coefficients.sum()
    reweighted = [
        by_unit.get(unit_id, UnitOutcome(unit_id, trials=1, failures=0, unclassified=1))
        for unit_id in unit_ids
    ]
    one_sided = coverage_mode == "one_sided_upper"
    endpoint = alpha if one_sided else alpha / 2.0
    lower, upper, method, _ = _bounds(
        reweighted, normalized, alpha_lower=endpoint, alpha_upper=endpoint, one_sided=one_sided
    )
    return ReweightedEstimate(
        id=identifier,
        level=level,
        weights=dict(weights),
        draws=len(unit_ids),
        estimate=_point_estimate(reweighted, normalized),
        lower=lower,
        upper=upper,
        method=method,
        effective_sample_size=_effective_sample_size(normalized),
        maximum_coefficient=float(normalized.max()),
        declared_lower=root.lower,
        declared_upper=root.upper,
    )
