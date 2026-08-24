"""The shipped generative example must keep working, and keep covering its known answer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from inspect_safety_stats.generative.execution import (
    execute_rewrites,
    load_seed_texts,
    write_prompt_store,
)
from inspect_safety_stats.generative.inference import estimate_slice, estimate_tree
from inspect_safety_stats.generative.inspect_bridge import join_outcomes, materialize_for_cohorts
from inspect_safety_stats.generative.kernels import (
    DeterministicFakeKernel,
    resolve_kernel_fingerprint,
)
from inspect_safety_stats.generative.ledger import write_rewrites
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "generative_multilingual"
sys.path.insert(0, str(EXAMPLE))

import simulated_target  # noqa: E402


def pipeline(plan_name: str, budget: int, seed: int = 0):
    plan = load_generative_plan(EXAMPLE / plan_name)
    plan.sampling.total_rewrites = budget
    taxonomy = compile_taxonomy(plan, base_path=EXAMPLE)
    fingerprint = resolve_kernel_fingerprint(plan, base_path=EXAMPLE)
    design = plan_design(taxonomy, plan)
    selections = draw_selections(taxonomy, design)
    result = execute_rewrites(
        plan,
        selections,
        DeterministicFakeKernel(),
        fingerprint,
        load_seed_texts(plan, base_path=EXAMPLE),
    )
    samples = simulated_target.evaluate(
        selections, result.accepted, np.random.default_rng(seed)
    )
    joined = join_outcomes(plan, selections, result, samples, cohort_id="baseline")
    return plan, taxonomy, design, selections, result, joined


def test_both_shipped_plans_compile_to_the_same_taxonomy() -> None:
    direct = compile_taxonomy(
        load_generative_plan(EXAMPLE / "plan-direct.yaml"), base_path=EXAMPLE
    )
    stratified = compile_taxonomy(
        load_generative_plan(EXAMPLE / "plan-stratified.yaml"), base_path=EXAMPLE
    )
    assert direct.fingerprint == stratified.fingerprint
    assert len(direct.terminal_paths) == 264
    assert len(direct.pruned_by_constraint) == 24


def test_the_maltese_constraint_and_context_weights_take_effect() -> None:
    taxonomy = compile_taxonomy(
        load_generative_plan(EXAMPLE / "plan-direct.yaml"), base_path=EXAMPLE
    )
    assert not any(
        "language=maltese" in node.node_id and "wrapper=nested_task" in node.node_id
        for node in taxonomy.nodes
    )
    maltese = [
        node
        for node in taxonomy.nodes
        if node.level == "wrapper" and "language=maltese" in node.node_id
    ]
    assert {round(node.conditional_target_weight, 3) for node in maltese} == {0.625, 0.375}

    english_style = {
        node.value: round(node.conditional_target_weight, 3)
        for node in taxonomy.nodes
        if node.level == "style" and "language=english" in node.node_id
    }
    assert english_style == {"plain": 0.4, "conversational": 0.3, "formal": 0.3}


@pytest.mark.parametrize("plan_name", ["plan-direct.yaml", "plan-stratified.yaml"])
def test_the_example_runs_end_to_end_and_covers_its_known_answer(plan_name: str) -> None:
    plan, taxonomy, design, selections, result, joined = pipeline(plan_name, 800)
    assert not joined.has_errors
    assert len(result.accepted) == 800

    tree = estimate_tree(
        taxonomy, design, selections, list(joined.outcomes), alpha=0.05, report_all_nodes=False
    )
    truth = simulated_target.true_node_risk(taxonomy, taxonomy.root_id)
    assert tree.root.draws == 800
    assert tree.root.upper >= truth
    assert tree.root.estimate == pytest.approx(truth, abs=0.02)


def test_the_maltese_slice_finds_the_concentrated_risk() -> None:
    plan, taxonomy, design, selections, result, joined = pipeline("plan-direct.yaml", 1200)
    outcomes = list(joined.outcomes)
    maltese = estimate_slice(
        taxonomy, design, selections, outcomes, level="language", value="maltese"
    )
    english = estimate_slice(
        taxonomy, design, selections, outcomes, level="language", value="english"
    )
    assert maltese.node_weight == pytest.approx(0.2)
    assert english.node_weight == pytest.approx(0.4)
    # The declared truth puts Maltese well above English; the estimates should see it.
    assert maltese.estimate is not None and english.estimate is not None
    assert maltese.estimate > english.estimate


def test_the_example_writes_no_prompt_text_into_its_ledger(tmp_path: Path) -> None:
    plan, taxonomy, design, selections, result, joined = pipeline("plan-direct.yaml", 200)
    write_rewrites(list(result.rewrites), tmp_path / "rewrites.csv")
    ledger = (tmp_path / "rewrites.csv").read_text(encoding="utf-8")
    assert "synthesis instructions" not in ledger
    assert "select-agent" not in ledger

    write_prompt_store(result.accepted, tmp_path / "prompts.csv")
    assert "select-agent" in (tmp_path / "prompts.csv").read_text(encoding="utf-8")


def test_the_materialized_dataset_is_valid_and_paired(tmp_path: Path) -> None:
    plan, taxonomy, design, selections, result, joined = pipeline("plan-direct.yaml", 200)
    plan.target.cohorts["candidate"] = plan.target.cohorts["baseline"].model_copy()
    datasets = materialize_for_cohorts(plan, selections, result, tmp_path / "datasets")
    assert datasets["baseline"] == datasets["candidate"]

    rows = [
        json.loads(line)
        for line in datasets["baseline"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 200
    assert len({row["id"] for row in rows}) == 200
    assert all(row["input"] and row["metadata"]["path_hash"] for row in rows)
