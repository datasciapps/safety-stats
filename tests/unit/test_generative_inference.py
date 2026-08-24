"""Root and node inference: exactness, coverage, and conservative missingness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_safety_stats.generative.inference import (
    EXACT_METHOD,
    WEIGHTED_METHOD,
    UnitOutcome,
    estimate_slice,
    estimate_tree,
)
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy
from tests.generative_fixtures import SEEDS, base_plan, materialize

# Seven equally weighted seeds: 1/7 divides no round budget, so integer allocation
# cannot be exactly proportional and the coefficient wrinkle is exposed.
SEVEN_SEEDS = "problem_class,semantic_seed_id,problem_class_weight,seed_weight\n" + "".join(
    f"chem,s{index:03d},1.0,0.142857142857\n" for index in range(1, 8)
)


def prepare(
    tmp_path: Path, method: str, total: int, seeds: str = SEEDS, minimum: int = 0
) -> tuple[Any, Any, Any]:
    plan = base_plan()
    plan["sampling"] = {"method": method, "total_rewrites": total}
    if minimum:
        plan["sampling"]["minimum_rewrites_per_stratum"] = minimum
    parsed = load_generative_plan(materialize(tmp_path, plan, seeds))
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    design = plan_design(taxonomy, parsed)
    return taxonomy, design, draw_selections(taxonomy, design)


def risk_of(node_id: str) -> float:
    """A deliberately heterogeneous truth: maltese and roleplay paths fail more."""

    risk = 0.02
    if "language=maltese" in node_id:
        risk += 0.10
    if "wrapper=roleplay" in node_id:
        risk += 0.05
    return risk


def true_root_risk(taxonomy: Any) -> float:
    return sum(node.target_path_weight * risk_of(node.node_id) for node in taxonomy.terminal_paths)


def simulate(selections: list[Any], rng: np.random.Generator) -> list[UnitOutcome]:
    return [
        UnitOutcome(
            record.sampling_unit_id,
            trials=1,
            failures=int(rng.random() < risk_of(record.node_id)),
            unclassified=0,
        )
        for record in selections
    ]


def test_direct_sampling_gives_the_exact_kernel(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 500)
    outcomes = simulate(selections, np.random.default_rng(0))
    tree = estimate_tree(taxonomy, design, selections, outcomes, alpha=0.05)
    assert tree.root.method == EXACT_METHOD
    assert tree.root.equal_coefficients
    assert tree.root.coefficient_ratio == pytest.approx(1.0)
    assert tree.root.draws == 500


def test_the_estimator_is_unbiased(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 400)
    truth = true_root_risk(taxonomy)
    rng = np.random.default_rng(11)
    estimates = []
    for _ in range(400):
        tree = estimate_tree(
            taxonomy,
            design,
            selections,
            simulate(selections, rng),
            alpha=0.05,
            report_all_nodes=False,
        )
        assert tree.root.estimate is not None
        estimates.append(tree.root.estimate)
    # Selections are fixed here, so this checks the outcome average, not path selection.
    assert float(np.mean(estimates)) == pytest.approx(truth, abs=0.01)


def test_root_upper_bound_covers_the_truth(tmp_path: Path) -> None:
    """One-sided coverage must reach its nominal level over repeated evaluations."""

    taxonomy, design, selections = prepare(tmp_path, "direct", 300)
    truth = true_root_risk(taxonomy)
    rng = np.random.default_rng(7)
    covered = 0
    replicates = 400
    for _ in range(replicates):
        tree = estimate_tree(
            taxonomy,
            design,
            selections,
            simulate(selections, rng),
            alpha=0.05,
            report_all_nodes=False,
        )
        covered += tree.root.upper >= truth
    assert covered / replicates >= 0.95


def test_node_intervals_cover_their_conditional_risk(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 600)
    target = "all|problem_class=chem|semantic_seed_id=s001|language=maltese"
    conditional = sum(
        node.target_path_weight * risk_of(node.node_id)
        for node in taxonomy.terminal_paths
        if node.node_id.startswith(f"{target}|")
    ) / taxonomy.node(target).target_path_weight

    rng = np.random.default_rng(3)
    covered = 0
    replicates = 200
    for _ in range(replicates):
        tree = estimate_tree(
            taxonomy, design, selections, simulate(selections, rng), alpha=0.05
        )
        covered += tree.node(target).upper >= conditional
    assert covered / replicates >= 0.95


def test_integer_allocation_penalty_is_small(tmp_path: Path) -> None:
    """Proportional stratification stays near-free once integer allocation is applied.

    Seven strata at 1/7 force unequal coefficients at most budgets, which drops the
    sharp Bernoulli-KL kernel for the weighted fallback. The cost is bounded and
    shrinks as the budget grows.
    """

    def outcomes_for(sel: list[Any]) -> list[UnitOutcome]:
        return [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in sel]
    penalties = {}
    for total in (300, 1000, 3000):
        taxonomy, design, selections = prepare(tmp_path, "direct", total, SEVEN_SEEDS)
        direct = estimate_tree(
            taxonomy, design, selections, outcomes_for(selections), report_all_nodes=False
        )
        taxonomy, design, selections = prepare(
            tmp_path, "proportional_stratified", total, SEVEN_SEEDS
        )
        stratified = estimate_tree(
            taxonomy, design, selections, outcomes_for(selections), report_all_nodes=False
        )
        assert stratified.root.method == WEIGHTED_METHOD
        assert not stratified.root.equal_coefficients
        penalties[total] = stratified.root.upper / direct.root.upper

    assert penalties[300] < 1.05
    assert penalties[3000] < 1.01
    assert penalties[3000] < penalties[300]


def test_a_budget_that_divides_the_weights_keeps_the_exact_kernel(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(
        tmp_path, "proportional_stratified", 700, SEVEN_SEEDS
    )
    outcomes = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    tree = estimate_tree(taxonomy, design, selections, outcomes, report_all_nodes=False)
    assert tree.root.method == EXACT_METHOD
    assert tree.root.coefficient_ratio == pytest.approx(1.0)


def test_unsampled_nodes_are_reported_not_estimated(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 12)
    outcomes = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    tree = estimate_tree(taxonomy, design, selections, outcomes)
    assert tree.unsampled_nodes
    reported = {node.node_id for node in tree.nodes}
    assert reported.isdisjoint(tree.unsampled_nodes)
    assert len(reported) + len(tree.unsampled_nodes) == len(taxonomy.nodes)


def test_a_unit_without_an_outcome_is_completed_conservatively(tmp_path: Path) -> None:
    """A rewrite that never passed validation must not silently leave the design."""

    taxonomy, design, selections = prepare(tmp_path, "direct", 200)
    clean = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    complete = estimate_tree(taxonomy, design, selections, clean, report_all_nodes=False)

    partial = clean[:150]
    incomplete = estimate_tree(taxonomy, design, selections, partial, report_all_nodes=False)
    assert incomplete.root.draws == 200
    assert incomplete.root.unclassified == 50
    assert incomplete.root.lower <= complete.root.lower
    assert incomplete.root.upper > complete.root.upper


def test_explicitly_unclassified_rollouts_widen_the_interval(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 200)
    clean = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    murky = [UnitOutcome(r.sampling_unit_id, 1, 0, 1) for r in selections[:20]] + clean[20:]
    widened = estimate_tree(taxonomy, design, selections, murky, report_all_nodes=False)
    tight = estimate_tree(taxonomy, design, selections, clean, report_all_nodes=False)
    assert widened.root.upper > tight.root.upper


def test_simultaneous_calibration_widens_and_names_its_frontier(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 600)
    outcomes = simulate(selections, np.random.default_rng(5))
    pointwise = estimate_tree(taxonomy, design, selections, outcomes, alpha=0.05)
    together = estimate_tree(
        taxonomy, design, selections, outcomes, alpha=0.05, simultaneous_tree=True
    )
    assert together.root.upper > pointwise.root.upper
    assert together.root.frontier_id == "depth-0"
    assert pointwise.root.frontier_id is None
    for node in together.nodes:
        assert node.frontier_id == f"depth-{node.depth}"
        assert node.alpha_upper < 0.05


def test_outcomes_for_unselected_units_are_rejected(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 20)
    outcomes = [UnitOutcome("not-a-unit", 1, 0, 0)]
    with pytest.raises(ValueError, match="unselected sampling units"):
        estimate_tree(taxonomy, design, selections, outcomes)


def test_unit_outcome_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="cannot exceed trials"):
        UnitOutcome("u", trials=1, failures=1, unclassified=1)
    with pytest.raises(ValueError, match="at least one attempted rollout"):
        UnitOutcome("u", trials=0, failures=0, unclassified=0)


def test_slice_weight_and_draws_partition_the_design(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 600)
    outcomes = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    cuts = [
        estimate_slice(
            taxonomy, design, selections, outcomes, level="language", value=value
        )
        for value in ("english", "maltese")
    ]
    assert sum(cut.node_weight for cut in cuts) == pytest.approx(1.0)
    assert sum(cut.draws for cut in cuts) == len(selections)
    assert all(cut.node_id.startswith("slice:language=") for cut in cuts)


def test_slice_uses_the_exact_kernel_under_direct_sampling(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 600)
    outcomes = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    cut = estimate_slice(
        taxonomy, design, selections, outcomes, level="wrapper", value="roleplay"
    )
    assert cut.method == EXACT_METHOD
    assert cut.equal_coefficients
    assert cut.frontier_id is None


def test_slice_covers_its_declared_weight_mean(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 600)
    matching = [
        node for node in taxonomy.terminal_paths if "language=maltese" in node.node_id
    ]
    mass = sum(node.target_path_weight for node in matching)
    truth = sum(
        node.target_path_weight * risk_of(node.node_id) for node in matching
    ) / mass

    rng = np.random.default_rng(19)
    covered = 0
    replicates = 200
    for _ in range(replicates):
        cut = estimate_slice(
            taxonomy,
            design,
            selections,
            simulate(selections, rng),
            level="language",
            value="maltese",
            alpha=0.05,
        )
        covered += cut.upper >= truth
    assert covered / replicates >= 0.95


def test_slice_rejects_an_unknown_level_or_value(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 100)
    outcomes = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    with pytest.raises(ValueError, match="not a level of this taxonomy"):
        estimate_slice(taxonomy, design, selections, outcomes, level="ghost", value="x")
    with pytest.raises(ValueError, match="no terminal path carries"):
        estimate_slice(taxonomy, design, selections, outcomes, level="language", value="klingon")


def test_effective_sample_size_equals_the_draw_count_under_equal_coefficients(
    tmp_path: Path,
) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 600)
    outcomes = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    tree = estimate_tree(taxonomy, design, selections, outcomes, report_all_nodes=False)
    assert tree.root.effective_sample_size == pytest.approx(600.0)


def test_effective_sample_size_falls_when_the_weight_concentrates(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(
        tmp_path, "proportional_stratified", 600, SEVEN_SEEDS, minimum=80
    )
    outcomes = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    tree = estimate_tree(taxonomy, design, selections, outcomes, report_all_nodes=False)
    assert not tree.root.equal_coefficients
    assert tree.root.effective_sample_size < tree.root.draws


def test_the_raw_rate_is_reported_apart_from_the_weighted_estimate(tmp_path: Path) -> None:
    """They coincide only when every draw carries the same coefficient."""

    taxonomy, design, selections = prepare(tmp_path, "direct", 600)
    outcomes = [
        UnitOutcome(record.sampling_unit_id, 1, int(index < 30), 0)
        for index, record in enumerate(selections)
    ]
    tree = estimate_tree(taxonomy, design, selections, outcomes, report_all_nodes=False)
    assert tree.root.raw_failure_rate == pytest.approx(30 / 600)
    assert tree.root.estimate == pytest.approx(tree.root.raw_failure_rate)

    taxonomy, design, selections = prepare(
        tmp_path, "proportional_stratified", 600, SEVEN_SEEDS, minimum=80
    )
    outcomes = [
        UnitOutcome(record.sampling_unit_id, 1, int(index < 30), 0)
        for index, record in enumerate(selections)
    ]
    tree = estimate_tree(taxonomy, design, selections, outcomes, report_all_nodes=False)
    assert tree.root.raw_failure_rate != pytest.approx(tree.root.estimate)


def test_unclassified_rollouts_leave_the_raw_rate_undefined(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, "direct", 100)
    outcomes = [UnitOutcome(r.sampling_unit_id, 1, 0, 1) for r in selections]
    tree = estimate_tree(taxonomy, design, selections, outcomes, report_all_nodes=False)
    assert tree.root.raw_failure_rate is None
    assert tree.root.estimate is None
