"""Counter-based sampler: reproducibility, allocation, and convergence to the design."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from inspect_safety_stats.generative.sampling import (
    draw_selections,
    largest_remainder,
    plan_design,
    stratification_frontier,
)
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy
from tests.unit.test_generative_taxonomy import SEEDS, base_plan, materialize


def build(tmp_path: Path, method: str, total: int, minimum: int = 0, seeds: str = SEEDS,
          selection_seed: int | None = None,
          explicit_seed_weights: bool = False) -> tuple[Any, Any, Any]:
    plan = base_plan()
    plan["sampling"] = {"method": method, "total_rewrites": total}
    if explicit_seed_weights:
        plan["taxonomy"]["fixed_levels"]["semantic_seed_id"] = {
            "weights": "explicit",
            "weight_column": "seed_weight",
        }
    if minimum:
        plan["sampling"]["minimum_rewrites_per_stratum"] = minimum
    if selection_seed is not None:
        plan["evaluation"]["selection_seed"] = selection_seed
    parsed = load_generative_plan(materialize(tmp_path, plan, seeds))
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    design = plan_design(taxonomy, parsed)
    return taxonomy, design, draw_selections(taxonomy, design)


def test_direct_frequencies_converge_to_declared_weights(tmp_path: Path) -> None:
    taxonomy, _, selections = build(tmp_path, "direct", 40000)
    counts = Counter(record.node_id for record in selections)
    for node in taxonomy.terminal_paths:
        realized = counts[node.node_id] / len(selections)
        assert realized == pytest.approx(node.target_path_weight, abs=0.01)


def test_selection_is_reproducible_and_seed_dependent(tmp_path: Path) -> None:
    _, _, first = build(tmp_path, "direct", 500)
    _, _, again = build(tmp_path, "direct", 500)
    assert first == again
    _, _, other = build(tmp_path, "direct", 500, selection_seed=99)
    assert [r.node_id for r in other] != [r.node_id for r in first]


def test_a_shorter_run_is_a_prefix_of_a_longer_one(tmp_path: Path) -> None:
    """Counter-based selection means an interrupted run resumes without duplication."""

    _, _, short = build(tmp_path, "direct", 200)
    _, _, long = build(tmp_path, "direct", 500)
    assert long[:200] == short


def test_sampling_unit_ids_are_unique(tmp_path: Path) -> None:
    for method in ("direct", "proportional_stratified"):
        _, _, selections = build(tmp_path, method, 600)
        identifiers = [record.sampling_unit_id for record in selections]
        assert len(set(identifiers)) == len(identifiers)
        assert [record.selection_order for record in selections] == list(range(len(selections)))


def test_stratified_allocation_is_proportional_within_one_draw(tmp_path: Path) -> None:
    taxonomy, design, selections = build(tmp_path, "proportional_stratified", 1000)
    assert len(selections) == 1000
    for allocation in design.allocations:
        assert abs(allocation.draws - allocation.stratum_weight * 1000) < 1.0


def test_stratified_draws_stay_inside_their_stratum(tmp_path: Path) -> None:
    _, design, selections = build(tmp_path, "proportional_stratified", 600)
    counts = Counter(record.stratum for record in selections)
    for allocation in design.allocations:
        assert counts[allocation.stratum_id] == allocation.draws
    for record in selections:
        assert record.node_id.startswith(f"{record.stratum}|")
        # Within a stratum the draw comes from the conditional target distribution.
        assert record.proposal_path_probability == pytest.approx(
            record.target_path_probability / design.weight_by_stratum[str(record.stratum)]
        )


def test_minimum_floor_is_respected(tmp_path: Path) -> None:
    _, design, selections = build(tmp_path, "proportional_stratified", 300, minimum=80)
    assert len(selections) == 300
    assert all(allocation.draws >= 80 for allocation in design.allocations)


def test_infeasible_minimum_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires at least"):
        build(tmp_path, "proportional_stratified", 100, minimum=40)


def test_a_stratum_receiving_no_draws_is_rejected(tmp_path: Path) -> None:
    seeds = "problem_class,semantic_seed_id,problem_class_weight,seed_weight\n"
    seeds += "chem,s001,1.0,0.999\n"
    seeds += "chem,s002,1.0,0.001\n"
    with pytest.raises(ValueError, match="receive no draws"):
        build(
            tmp_path,
            "proportional_stratified",
            10,
            seeds=seeds,
            explicit_seed_weights=True,
        )


def test_stratification_frontier_is_the_semantic_seed_level(tmp_path: Path) -> None:
    taxonomy, _, _ = build(tmp_path, "direct", 10)
    frontier = stratification_frontier(taxonomy)
    assert {node.value for node in frontier} == {"s001", "s002", "s003"}
    assert sum(node.target_path_weight for node in frontier) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("weights", "total", "expected_sum"),
    [
        ({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 100, 100),
        ({"a": 0.5, "b": 0.5}, 7, 7),
        ({"a": 1.0}, 13, 13),
    ],
)
def test_largest_remainder_conserves_the_total(weights, total, expected_sum) -> None:
    allocation = largest_remainder(weights, total)
    assert sum(allocation.values()) == expected_sum
    for key, weight in weights.items():
        assert abs(allocation[key] - weight * total) < 1.0
