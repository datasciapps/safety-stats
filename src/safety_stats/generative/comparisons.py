"""Cohort comparisons on a generative design.

With ``target.shared_rewrites`` both arms evaluate the *same* accepted rewrite, so each
sampling unit yields a paired difference \\(D_j = X^C_j - X^B_j \\in \\{-1, 0, 1\\}\\).
The dependence between the arms lives entirely inside a unit, where it is unrestricted;
independence is only claimed across units, which is what path selection and rewrite
generation actually deliver. Concordant units contribute exactly zero, which is where the
pairing pays.

Two constructions, chosen the same way node inference chooses its kernel:

* equal coefficients and one rollout per rewrite -- the discordant counts are two cells of
  a trinomial, each marginally binomial, so exact Clopper--Pearson applies to each and a
  union bound combines them. In the sparse-discordance regime this package targets, that
  is markedly tighter than a bounded-mean bound.
* otherwise -- \\(Q_j = (1 + D_j)/2 \\in [0, 1]\\) is an ordinary bounded observation, and
  the weighted engine bounds its mean before rescaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from safety_stats.generative.inference import (
    CoverageMode,
    NodeEstimate,
    UnitOutcome,
    _bounds,
    _effective_sample_size,
    _point_estimate,
    _raw_coefficients,
    _raw_failure_rate,
    _unclassified_share,
)
from safety_stats.generative.ledger import SelectionRecord
from safety_stats.generative.sampling import SamplingDesign
from safety_stats.generative.taxonomy import CompiledTaxonomy
from safety_stats.stats.binomial import binomial_lower_bound, binomial_upper_bound

PAIRED_EXACT = "paired_binomial_cells"
PAIRED_WEIGHTED = "paired_weighted_kl_chernoff"
INDEPENDENT_ARMS = "independent_arm_rectangle"


@dataclass(frozen=True)
class ComparisonResult:
    """A bound on the candidate-minus-baseline risk difference at one node.

    Built by :func:`compare_cohorts`. Under ``one_sided_upper`` ``lower`` is -1. The
    discordant counts are set only for the exact paired method and are None otherwise.
    ``candidate`` and ``baseline`` are the per-arm estimates over the same units.
    """

    node_id: str
    estimand: Literal["risk_difference"]
    candidate_cohort: str
    baseline_cohort: str
    draws: int
    estimate: float | None
    lower: float
    upper: float
    method: str
    alpha: float
    coverage_mode: CoverageMode
    paired: bool
    discordant_candidate_only: int | None
    discordant_baseline_only: int | None
    candidate: NodeEstimate
    baseline: NodeEstimate


def units_in_node(selections: list[SelectionRecord], node_id: str) -> list[str]:
    """Sampling units whose selected path lies at or below a node."""

    prefix = f"{node_id}|"
    return [
        record.sampling_unit_id
        for record in selections
        if record.node_id == node_id or record.node_id.startswith(prefix)
    ]


def _arm_estimate(
    taxonomy: CompiledTaxonomy,
    design: SamplingDesign,
    selections: list[SelectionRecord],
    outcomes: list[UnitOutcome],
    unit_ids: list[str],
    node_id: str,
    *,
    alpha_lower: float,
    alpha_upper: float,
    one_sided: bool,
) -> NodeEstimate:
    node = taxonomy.node(node_id)
    coefficients = _raw_coefficients(design, selections)
    by_unit = {outcome.sampling_unit_id: outcome for outcome in outcomes}
    raw = np.array([coefficients[unit_id] for unit_id in unit_ids], dtype=float)
    normalized = raw / raw.sum()
    arm = [
        by_unit.get(unit_id, UnitOutcome(unit_id, trials=1, failures=0, unclassified=1))
        for unit_id in unit_ids
    ]
    lower, upper, method, equal = _bounds(
        arm, normalized, alpha_lower=alpha_lower, alpha_upper=alpha_upper, one_sided=one_sided
    )
    return NodeEstimate(
        node_id=node_id,
        level=node.level,
        depth=node.depth,
        node_weight=node.target_path_weight,
        draws=len(unit_ids),
        trials=sum(item.trials for item in arm),
        failures=sum(item.failures for item in arm),
        unclassified=sum(item.unclassified for item in arm),
        estimate=_point_estimate(arm, normalized),
        lower=lower,
        upper=upper,
        method=method,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        frontier_id=None,
        equal_coefficients=equal,
        maximum_coefficient=float(normalized.max()),
        coefficient_ratio=float(normalized.max() / normalized.min()),
        effective_sample_size=_effective_sample_size(normalized),
        raw_failure_rate=_raw_failure_rate(arm),
        unclassified_share=_unclassified_share(arm, normalized),
    )


def _completions(outcome: UnitOutcome) -> tuple[int, int]:
    """The smallest and largest failure count consistent with an outcome's labels."""

    return outcome.failures, outcome.failures + outcome.unclassified


def compare_cohorts(
    taxonomy: CompiledTaxonomy,
    design: SamplingDesign,
    selections: list[SelectionRecord],
    candidate_outcomes: list[UnitOutcome],
    baseline_outcomes: list[UnitOutcome],
    *,
    candidate_cohort: str = "candidate",
    baseline_cohort: str = "baseline",
    node_id: str | None = None,
    alpha: float = 0.05,
    coverage_mode: CoverageMode = "one_sided_upper",
    paired: bool = True,
) -> ComparisonResult:
    """Bound the risk difference between two cohorts at a node.

    ``paired`` requires both arms to have evaluated the same sampling units, which is what
    ``target.shared_rewrites`` arranges. Set it to ``False`` when the arms were generated
    with distinct rewrite salts; the arms are then bounded separately and combined by a
    union bound, which is valid but forfeits the concordance that makes pairing worthwhile.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    target = node_id or taxonomy.root_id
    unit_ids = units_in_node(selections, target)
    if not unit_ids:
        raise ValueError(f"no draw landed on {target}")

    one_sided = coverage_mode == "one_sided_upper"
    # A difference has two arms, so each side of the interval is a union over both.
    side_alpha = alpha if one_sided else alpha / 2.0

    candidate_arm = _arm_estimate(
        taxonomy, design, selections, candidate_outcomes, unit_ids, target,
        alpha_lower=side_alpha / 2.0, alpha_upper=side_alpha / 2.0, one_sided=False,
    )
    baseline_arm = _arm_estimate(
        taxonomy, design, selections, baseline_outcomes, unit_ids, target,
        alpha_lower=side_alpha / 2.0, alpha_upper=side_alpha / 2.0, one_sided=False,
    )

    if not paired:
        lower = -1.0 if one_sided else candidate_arm.lower - baseline_arm.upper
        estimate = (
            None
            if candidate_arm.estimate is None or baseline_arm.estimate is None
            else candidate_arm.estimate - baseline_arm.estimate
        )
        return ComparisonResult(
            node_id=target,
            estimand="risk_difference",
            candidate_cohort=candidate_cohort,
            baseline_cohort=baseline_cohort,
            draws=len(unit_ids),
            estimate=estimate,
            lower=lower,
            upper=candidate_arm.upper - baseline_arm.lower,
            method=INDEPENDENT_ARMS,
            alpha=alpha,
            coverage_mode=coverage_mode,
            paired=False,
            discordant_candidate_only=None,
            discordant_baseline_only=None,
            candidate=candidate_arm,
            baseline=baseline_arm,
        )

    candidate_by_unit = {item.sampling_unit_id: item for item in candidate_outcomes}
    baseline_by_unit = {item.sampling_unit_id: item for item in baseline_outcomes}
    missing = [
        unit_id
        for unit_id in unit_ids
        if unit_id not in candidate_by_unit or unit_id not in baseline_by_unit
    ]
    if missing and len(missing) == len(unit_ids):
        raise ValueError(
            "a paired comparison needs both arms to have evaluated the same sampling "
            "units; no unit appears in both"
        )

    differences_low: list[int] = []
    differences_high: list[int] = []
    unit_trials: list[int] = []
    for unit_id in unit_ids:
        unevaluated = UnitOutcome(unit_id, trials=1, failures=0, unclassified=1)
        candidate = candidate_by_unit.get(unit_id, unevaluated)
        base = baseline_by_unit.get(unit_id, unevaluated)
        if candidate.trials != base.trials:
            raise ValueError(
                f"sampling unit {unit_id} has {candidate.trials} candidate rollouts and "
                f"{base.trials} baseline rollouts; a paired difference needs the same count"
            )
        candidate_low, candidate_high = _completions(candidate)
        base_low, base_high = _completions(base)
        differences_low.append(candidate_low - base_high)
        differences_high.append(candidate_high - base_low)
        unit_trials.append(candidate.trials)

    trials = set(unit_trials)
    coefficients = _raw_coefficients(design, selections)
    raw = np.array([coefficients[unit_id] for unit_id in unit_ids], dtype=float)
    normalized = raw / raw.sum()
    equal = bool(np.allclose(normalized, normalized[0], rtol=1e-12, atol=0.0))

    total = len(unit_ids)
    estimate = float(
        np.dot(normalized, (np.array(differences_low) + np.array(differences_high)) / 2.0)
    )
    if trials == {1}:
        estimate = float(np.dot(normalized, np.array(differences_low)))

    if equal and trials == {1}:
        # Discordant units under the completion that maximises, then minimises, the
        # difference. Each cell count is marginally binomial; a union bound combines them.
        high = np.array(differences_high)
        low = np.array(differences_low)
        candidate_only_high = int(np.sum(high > 0))
        baseline_only_high = int(np.sum(high < 0))
        candidate_only_low = int(np.sum(low > 0))
        baseline_only_low = int(np.sum(low < 0))
        cell_alpha = side_alpha / 2.0
        upper = binomial_upper_bound(
            candidate_only_high, total, alpha=cell_alpha
        ) - binomial_lower_bound(baseline_only_high, total, alpha=cell_alpha)
        lower = (
            -1.0
            if one_sided
            else binomial_lower_bound(candidate_only_low, total, alpha=cell_alpha)
            - binomial_upper_bound(baseline_only_low, total, alpha=cell_alpha)
        )
        return ComparisonResult(
            node_id=target,
            estimand="risk_difference",
            candidate_cohort=candidate_cohort,
            baseline_cohort=baseline_cohort,
            draws=total,
            estimate=estimate,
            lower=max(lower, -1.0),
            upper=min(upper, 1.0),
            method=PAIRED_EXACT,
            alpha=alpha,
            coverage_mode=coverage_mode,
            paired=True,
            discordant_candidate_only=candidate_only_low,
            discordant_baseline_only=baseline_only_low,
            candidate=candidate_arm,
            baseline=baseline_arm,
        )

    # Q = (1 + D) / 2 on a scale of 2 rollouts encodes {-1, 0, 1} exactly, and the engine's
    # own missingness completion reproduces the conservative pairing completion.
    scale = 2 * max(trials)
    encoded = [
        UnitOutcome(
            unit_ids[index],
            trials=scale,
            failures=low + max(trials),
            unclassified=high - low,
        )
        for index, (low, high) in enumerate(zip(differences_low, differences_high, strict=True))
    ]
    q_lower, q_upper, _, _ = _bounds(
        encoded,
        normalized,
        alpha_lower=side_alpha,
        alpha_upper=side_alpha,
        one_sided=False,
    )
    return ComparisonResult(
        node_id=target,
        estimand="risk_difference",
        candidate_cohort=candidate_cohort,
        baseline_cohort=baseline_cohort,
        draws=total,
        estimate=estimate,
        lower=-1.0 if one_sided else max(2.0 * q_lower - 1.0, -1.0),
        upper=min(2.0 * q_upper - 1.0, 1.0),
        method=PAIRED_WEIGHTED,
        alpha=alpha,
        coverage_mode=coverage_mode,
        paired=True,
        discordant_candidate_only=None,
        discordant_baseline_only=None,
        candidate=candidate_arm,
        baseline=baseline_arm,
    )
