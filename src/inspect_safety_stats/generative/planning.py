"""What a budget will buy, before any of it is spent.

A generative plan is written *before* the evaluation exists, and the spend follows it:
rewrite calls, then rollouts, then judging. `sampling.total_rewrites` is therefore the
decision that costs money, and it is made with no data.

Everything here is computed from the compiled taxonomy plus one assumed failure rate.
Nothing is evaluated and nothing is generated. The forecast assumes risk is **uniform
across the design** at the declared rate; real heterogeneity moves individual nodes either
way, so treat node-level figures as what the geometry alone permits, not a prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from inspect_safety_stats.generative.sampling import plan_design
from inspect_safety_stats.generative.schema import GenerativePlan
from inspect_safety_stats.generative.taxonomy import CompiledTaxonomy
from inspect_safety_stats.stats.binomial import binomial_upper_bound
from inspect_safety_stats.stats.planning import certification_power


@dataclass(frozen=True)
class LevelForecast:
    """What one hierarchy level can support at a given budget."""

    level: str
    nodes: int
    median_draws: float
    smallest_draws: float
    median_upper: float
    worst_upper: float
    expected_unsampled: float


@dataclass(frozen=True)
class BudgetForecast:
    """What one budget is projected to support, as computed by :func:`forecast`.

    Every figure assumes risk uniform at ``assumed_rate``. ``coefficient_ratio`` above one
    means the weighted fallback replaces the exact kernel, and it is infinite when the
    stratified design cannot be resolved at this budget. ``certification_power`` is None
    unless a threshold was given.
    """

    total_rewrites: int
    assumed_rate: float
    alpha: float
    root_upper: float
    root_draws: int
    paths_with_a_draw: float
    expected_unsampled_nodes: float
    levels: tuple[LevelForecast, ...]
    rewrite_calls: float
    exhausted_units: float
    target_rollouts: int
    coefficient_ratio: float
    starved_strata: int
    certification_power: float | None


def _expected_upper(draws: float, assumed_rate: float, alpha: float) -> float:
    """The bound a node of this size would report if the design were homogeneous."""

    trials = max(1, int(round(draws)))
    return binomial_upper_bound(int(round(assumed_rate * trials)), trials, alpha=alpha)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        return 0.0
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _design_geometry(
    taxonomy: CompiledTaxonomy, plan: GenerativePlan, total_rewrites: int
) -> tuple[float, int]:
    """The coefficient ratio the declared design would realize, and starved strata.

    A ratio above one means the sharp exact kernel is unavailable and the weighted
    fallback applies. Under `proportional_stratified` this is normally the
    `minimum_rewrites_per_stratum` floor pushing allocation off proportional.
    """

    if plan.sampling.method == "direct":
        return 1.0, 0
    original = plan.sampling.total_rewrites
    try:
        plan.sampling.total_rewrites = total_rewrites
        try:
            design = plan_design(taxonomy, plan)
        except ValueError:
            # Too small a budget for the declared floor, or strata with no draws.
            return math.inf, len(
                [n for n in taxonomy.nodes if n.level == taxonomy.semantic_seed_level]
            )
    finally:
        plan.sampling.total_rewrites = original
    coefficients = [item.stratum_weight / item.draws for item in design.allocations]
    starved = sum(1 for item in design.allocations if item.draws == 0)
    return max(coefficients) / min(coefficients), starved


def forecast(
    taxonomy: CompiledTaxonomy,
    plan: GenerativePlan,
    *,
    total_rewrites: int,
    assumed_rate: float,
    acceptance_rate: float = 1.0,
    threshold: float | None = None,
) -> BudgetForecast:
    """Project what one budget supports, from the taxonomy geometry alone."""

    if total_rewrites <= 0:
        raise ValueError("total_rewrites must be positive")
    if not 0.0 <= assumed_rate < 1.0:
        raise ValueError("assumed_rate must be in [0, 1)")
    if not 0.0 < acceptance_rate <= 1.0:
        raise ValueError("acceptance_rate must be in (0, 1]")

    alpha = plan.inference.alpha
    root_upper = _expected_upper(total_rewrites, assumed_rate, alpha)

    by_level: dict[str, list[float]] = {}
    unsampled = 0.0
    for node in taxonomy.nodes:
        if not node.level:
            continue
        by_level.setdefault(node.level, []).append(node.target_path_weight)
        unsampled += (1.0 - node.target_path_weight) ** total_rewrites

    levels = []
    for level in taxonomy.levels:
        weights = by_level.get(level, [])
        if not weights:
            continue
        draws = [weight * total_rewrites for weight in weights]
        levels.append(
            LevelForecast(
                level=level,
                nodes=len(weights),
                median_draws=_median(draws),
                smallest_draws=min(draws),
                median_upper=_expected_upper(_median(draws), assumed_rate, alpha),
                worst_upper=_expected_upper(min(draws), assumed_rate, alpha),
                expected_unsampled=sum((1.0 - w) ** total_rewrites for w in weights),
            )
        )

    touched = sum(
        1.0 - (1.0 - node.target_path_weight) ** total_rewrites
        for node in taxonomy.terminal_paths
    ) / len(taxonomy.terminal_paths)

    # A capped geometric retry: attempts until the validator accepts, or the cap.
    attempts = plan.rewrite.validation.maximum_attempts
    survives = (1.0 - acceptance_rate) ** attempts
    calls_per_unit = (1.0 - survives) / acceptance_rate
    ratio, starved = _design_geometry(taxonomy, plan, total_rewrites)

    power = None
    if threshold is not None:
        if not 0.0 <= assumed_rate < threshold < 1.0:
            raise ValueError("threshold must exceed the assumed rate and be below one")
        power = certification_power(
            trials=total_rewrites, true_rate=assumed_rate, threshold=threshold, alpha=alpha
        )

    accepted = total_rewrites * (1.0 - survives)
    return BudgetForecast(
        total_rewrites=total_rewrites,
        assumed_rate=assumed_rate,
        alpha=alpha,
        root_upper=root_upper,
        root_draws=total_rewrites,
        paths_with_a_draw=touched,
        expected_unsampled_nodes=unsampled,
        levels=tuple(levels),
        rewrite_calls=total_rewrites * calls_per_unit,
        exhausted_units=total_rewrites * survives,
        target_rollouts=int(round(accepted * plan.target.rollouts_per_rewrite)),
        coefficient_ratio=ratio,
        starved_strata=starved,
        certification_power=power,
    )


def rewrites_for_bound(
    *, assumed_rate: float, target_upper: float, alpha: float, maximum: int = 5_000_000
) -> int | None:
    """The smallest budget whose expected root bound reaches a target."""

    if not 0.0 <= assumed_rate < target_upper < 1.0:
        raise ValueError("target_upper must exceed the assumed rate and be below one")
    low, high = 1, 1
    while _expected_upper(high, assumed_rate, alpha) > target_upper:
        low, high = high, high * 2
        if high > maximum:
            return None
    while low < high:
        middle = (low + high) // 2
        if _expected_upper(middle, assumed_rate, alpha) <= target_upper:
            high = middle
        else:
            low = middle + 1
    return low
