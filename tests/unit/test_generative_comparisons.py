"""Paired and unpaired cohort comparisons on a generative design."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_safety_stats.generative.comparisons import (
    INDEPENDENT_ARMS,
    PAIRED_EXACT,
    PAIRED_WEIGHTED,
    compare_cohorts,
    units_in_node,
)
from inspect_safety_stats.generative.inference import UnitOutcome
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy
from tests.generative_fixtures import SEEDS, base_plan, materialize


def prepare(tmp_path: Path, method: str = "direct", total: int = 600, minimum: int = 0):
    plan = base_plan()
    plan["sampling"] = {"method": method, "total_rewrites": total}
    if minimum:
        plan["sampling"]["minimum_rewrites_per_stratum"] = minimum
    parsed = load_generative_plan(materialize(tmp_path, plan, SEEDS))
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    design = plan_design(taxonomy, parsed)
    return taxonomy, design, draw_selections(taxonomy, design)


def risk_of(node_id: str) -> float:
    return 0.12 if "language=maltese" in node_id else 0.03


def arms(selections: list[Any], rng: np.random.Generator, ratio: float, coupling: float = 1.0):
    """Both arms on the same rewrites; ``coupling`` shares the latent difficulty draw."""

    shared = rng.random(len(selections))
    independent = rng.random(len(selections))
    couple = rng.random(len(selections)) < coupling
    candidate_draw = np.where(couple, shared, independent)
    candidate, baseline = [], []
    for index, record in enumerate(selections):
        risk = risk_of(record.node_id)
        candidate.append(
            UnitOutcome(record.sampling_unit_id, 1, int(candidate_draw[index] < risk * ratio), 0)
        )
        baseline.append(UnitOutcome(record.sampling_unit_id, 1, int(shared[index] < risk), 0))
    return candidate, baseline


def constant(selections: list[Any], failures: int) -> list[UnitOutcome]:
    return [
        UnitOutcome(record.sampling_unit_id, 1, 1 if index < failures else 0, 0)
        for index, record in enumerate(selections)
    ]


def test_units_in_node_selects_the_subtree(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path)
    everything = units_in_node(selections, taxonomy.root_id)
    assert len(everything) == len(selections)
    branch = units_in_node(selections, "all|problem_class=chem")
    assert 0 < len(branch) < len(selections)
    assert set(branch) < set(everything)


def test_paired_comparison_uses_the_exact_cell_construction(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path)
    candidate, baseline = arms(selections, np.random.default_rng(0), ratio=1.3)
    result = compare_cohorts(taxonomy, design, selections, candidate, baseline)
    assert result.method == PAIRED_EXACT
    assert result.paired
    assert result.draws == len(selections)
    assert result.discordant_candidate_only is not None
    assert result.estimand == "risk_difference"


def test_identical_arms_give_a_difference_of_exactly_zero(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path)
    same = constant(selections, 40)
    result = compare_cohorts(taxonomy, design, selections, same, list(same))
    assert result.discordant_candidate_only == 0
    assert result.discordant_baseline_only == 0
    assert result.estimate == pytest.approx(0.0)
    # No discordant unit at all, so the bound collapses towards zero.
    assert 0.0 < result.upper < 0.01


def test_pairing_is_never_worse_than_treating_the_arms_separately(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, total=1200)
    candidate, baseline = arms(selections, np.random.default_rng(3), ratio=1.3, coupling=1.0)
    together = compare_cohorts(taxonomy, design, selections, candidate, baseline)
    apart = compare_cohorts(
        taxonomy, design, selections, candidate, baseline, paired=False
    )
    assert apart.method == INDEPENDENT_ARMS
    assert not apart.paired
    assert together.upper < apart.upper


def test_the_paired_upper_bound_covers_the_true_difference(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, total=600)
    ratio = 1.3
    weights = {}
    for record in selections:
        weights[record.node_id] = weights.get(record.node_id, 0) + 1
    truth = sum(
        count * risk_of(node_id) * (ratio - 1.0) for node_id, count in weights.items()
    ) / len(selections)

    rng = np.random.default_rng(101)
    covered = 0
    replicates = 200
    for _ in range(replicates):
        candidate, baseline = arms(selections, rng, ratio=ratio, coupling=1.0)
        result = compare_cohorts(taxonomy, design, selections, candidate, baseline, alpha=0.05)
        covered += result.upper >= truth
    assert covered / replicates >= 0.95


def test_two_sided_mode_returns_a_finite_lower_bound(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path)
    candidate, baseline = arms(selections, np.random.default_rng(5), ratio=1.3)
    one_sided = compare_cohorts(taxonomy, design, selections, candidate, baseline)
    two_sided = compare_cohorts(
        taxonomy, design, selections, candidate, baseline, coverage_mode="two_sided"
    )
    assert one_sided.lower == -1.0
    assert -1.0 < two_sided.lower < two_sided.upper
    # Spending alpha on both sides costs width on the upper one.
    assert two_sided.upper > one_sided.upper


def test_a_worse_candidate_moves_the_estimate_up(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, total=1200)
    rng = np.random.default_rng(9)
    worse = compare_cohorts(
        taxonomy, design, selections, *arms(selections, rng, ratio=2.0)
    )
    better = compare_cohorts(
        taxonomy, design, selections, *arms(selections, rng, ratio=0.5)
    )
    assert worse.estimate is not None and better.estimate is not None
    assert worse.estimate > 0 > better.estimate


def test_unclassified_outcomes_widen_the_difference_conservatively(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path)
    candidate, baseline = arms(selections, np.random.default_rng(7), ratio=1.2)
    murky = [
        UnitOutcome(item.sampling_unit_id, 1, 0, 1) if index < 30 else item
        for index, item in enumerate(candidate)
    ]
    clean = compare_cohorts(taxonomy, design, selections, candidate, baseline)
    widened = compare_cohorts(taxonomy, design, selections, murky, baseline)
    assert widened.upper > clean.upper


def test_a_missing_arm_outcome_is_completed_not_dropped(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path)
    candidate, baseline = arms(selections, np.random.default_rng(2), ratio=1.2)
    result = compare_cohorts(taxonomy, design, selections, candidate[:400], baseline)
    assert result.draws == len(selections)
    full = compare_cohorts(taxonomy, design, selections, candidate, baseline)
    assert result.upper > full.upper


def test_unequal_coefficients_fall_back_to_the_weighted_construction(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(
        tmp_path, method="proportional_stratified", total=600, minimum=150
    )
    candidate, baseline = arms(selections, np.random.default_rng(4), ratio=1.3)
    result = compare_cohorts(taxonomy, design, selections, candidate, baseline)
    assert result.method == PAIRED_WEIGHTED
    assert result.paired
    assert result.discordant_candidate_only is None
    assert -1.0 <= result.upper <= 1.0


def test_comparison_can_target_a_node(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, total=1200)
    candidate, baseline = arms(selections, np.random.default_rng(6), ratio=1.3)
    node = "all|problem_class=chem|semantic_seed_id=s001"
    result = compare_cohorts(
        taxonomy, design, selections, candidate, baseline, node_id=node
    )
    assert result.node_id == node
    assert result.draws == len(units_in_node(selections, node))
    assert result.draws < len(selections)
    assert result.candidate.node_id == node


def test_mismatched_rollout_counts_are_rejected(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, total=100)
    candidate = [UnitOutcome(r.sampling_unit_id, 2, 0, 0) for r in selections]
    baseline = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    with pytest.raises(ValueError, match="paired difference needs the same count"):
        compare_cohorts(taxonomy, design, selections, candidate, baseline)


def test_arms_sharing_no_units_cannot_be_paired(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, total=100)
    candidate = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    baseline = [UnitOutcome(f"other-{r.sampling_unit_id}", 1, 0, 0) for r in selections]
    with pytest.raises(ValueError, match="no unit appears in both"):
        compare_cohorts(taxonomy, design, selections, candidate, baseline)


def test_an_empty_node_is_rejected(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, total=20)
    candidate = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    with pytest.raises(ValueError, match="no draw landed on"):
        compare_cohorts(
            taxonomy,
            design,
            selections,
            candidate,
            list(candidate),
            node_id="all|problem_class=ghost",
        )
