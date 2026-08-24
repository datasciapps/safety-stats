"""Synthetic-tree fixtures and compilation invariants for the generative taxonomy."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy

SEEDS = """problem_class,semantic_seed_id,problem_class_weight,seed_weight
chem,s001,0.6,0.5
chem,s002,0.6,0.5
bio,s003,0.4,1.0
"""


def base_plan() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "evaluation": {
            "id": "test-generative",
            "title": "Synthetic generative tree",
            "intent": "exploratory",
            "selection_seed": 271828,
        },
        "taxonomy": {
            "seed_manifest": "seeds.csv",
            "hierarchy": ["problem_class", "semantic_seed_id", "language", "wrapper"],
            "semantic_seed_level": "semantic_seed_id",
            "fixed_levels": {
                "problem_class": {"weights": "explicit", "weight_column": "problem_class_weight"},
                "semantic_seed_id": {"weights": "equal_within_parent"},
            },
            "arms": {
                "language": {
                    "values": [
                        {"id": "english", "weight": 0.5},
                        {"id": "maltese", "weight": 0.5},
                    ]
                },
                "wrapper": {
                    "values": [
                        {"id": "plain", "weight": 0.5},
                        {"id": "roleplay", "weight": 0.25},
                        {"id": "nested", "weight": 0.25},
                    ]
                },
            },
        },
        "rewrite": {
            "kernel_id": "test-kernel",
            "provider": "fake",
            "model": "fake-model",
            "prompt_template": "template.txt",
            "temperature": 1.0,
            "max_tokens": 256,
            "validation": {"validator_id": "semantic-equivalence-v1"},
        },
        "target": {
            "cohorts": {"baseline": {"model": "model-a"}},
            "score": {
                "name": "harmful",
                "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
            },
        },
        "sampling": {"method": "direct", "total_rewrites": 400},
    }


def materialize(tmp_path: Path, plan: dict[str, Any], seeds: str = SEEDS) -> Path:
    (tmp_path / "seeds.csv").write_text(seeds, encoding="utf-8")
    (tmp_path / "template.txt").write_text("Rewrite: {seed}\n", encoding="utf-8")
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


def compile_from(tmp_path: Path, plan: dict[str, Any], seeds: str = SEEDS):
    path = materialize(tmp_path, plan, seeds)
    return compile_taxonomy(load_generative_plan(path), base_path=tmp_path)


def test_compiles_expected_tree_and_weights(tmp_path: Path) -> None:
    tree = compile_from(tmp_path, base_plan())

    # 1 root + 2 problem classes + 3 seeds + 6 languages + 18 wrappers.
    assert len(tree.nodes) == 30
    assert len(tree.terminal_paths) == 18
    assert sum(node.target_path_weight for node in tree.terminal_paths) == pytest.approx(1.0)

    chem_plain = tree.node(
        "all|problem_class=chem|semantic_seed_id=s001|language=english|wrapper=plain"
    )
    # 0.6 problem class x 0.5 seed x 0.5 language x 0.5 wrapper.
    assert chem_plain.target_path_weight == pytest.approx(0.075)
    assert chem_plain.semantic_seed_id == "s001"
    assert chem_plain.rewrite_kernel_id == "test-kernel"
    assert chem_plain.is_terminal


def test_ancestor_weight_equals_declared_descendant_sum(tmp_path: Path) -> None:
    tree = compile_from(tmp_path, base_plan())
    for node in tree.nodes:
        children = [child for child in tree.nodes if child.parent_id == node.node_id]
        if not children:
            continue
        total = sum(child.target_path_weight for child in children)
        assert total == pytest.approx(node.target_path_weight)
        conditional = sum(child.conditional_target_weight for child in children)
        assert conditional == pytest.approx(1.0)


def test_path_hashes_are_unique_and_stable(tmp_path: Path) -> None:
    tree = compile_from(tmp_path, base_plan())
    hashes = {node.path_hash for node in tree.nodes}
    assert len(hashes) == len(tree.nodes)
    again = compile_from(tmp_path, base_plan())
    assert [node.path_hash for node in again.nodes] == [node.path_hash for node in tree.nodes]


def test_arm_declaration_order_does_not_change_the_fingerprint(tmp_path: Path) -> None:
    plan = base_plan()
    reordered = copy.deepcopy(plan)
    reordered["taxonomy"]["arms"]["wrapper"]["values"] = list(
        reversed(reordered["taxonomy"]["arms"]["wrapper"]["values"])
    )
    assert (
        compile_from(tmp_path, plan).fingerprint
        == compile_from(tmp_path, reordered).fingerprint
    )


def test_equal_weights_and_explicit_equal_weights_agree(tmp_path: Path) -> None:
    plan = base_plan()
    explicit = copy.deepcopy(plan)
    explicit["taxonomy"]["fixed_levels"]["semantic_seed_id"] = {
        "weights": "explicit",
        "weight_column": "seed_weight",
    }
    assert (
        compile_from(tmp_path, plan).fingerprint
        == compile_from(tmp_path, explicit).fingerprint
    )


def test_constraint_prunes_and_renormalizes(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["renormalization"] = "renormalize_valid_children"
    plan["taxonomy"]["constraints"] = [
        {
            "id": "maltese-wrapper-compatibility",
            "when": {"language": "maltese"},
            "allow": {"wrapper": ["plain", "roleplay"]},
        }
    ]
    tree = compile_from(tmp_path, plan)

    assert len(tree.terminal_paths) == 3 * (3 + 2)
    maltese = [
        node
        for node in tree.terminal_paths
        if "language=maltese" in node.node_id
        and node.node_id.startswith("all|problem_class=chem|semantic_seed_id=s001|")
    ]
    # 0.5 and 0.25 renormalized over their surviving total of 0.75.
    assert sorted(node.conditional_target_weight for node in maltese) == pytest.approx(
        [1 / 3, 2 / 3]
    )
    assert sum(node.target_path_weight for node in tree.terminal_paths) == pytest.approx(1.0)
    assert any("wrapper=nested" in entry for entry in tree.pruned_by_constraint)


def test_constraints_without_a_renormalization_rule_are_rejected(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["constraints"] = [
        {"id": "c1", "when": {"language": "maltese"}, "allow": {"wrapper": ["plain"]}}
    ]
    with pytest.raises(ValidationError, match="renormalization"):
        load_generative_plan(materialize(tmp_path, plan))


def test_explicit_context_weights_required_when_declared(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["renormalization"] = "explicit_context_weights"
    plan["taxonomy"]["constraints"] = [
        {"id": "c1", "when": {"language": "maltese"}, "allow": {"wrapper": ["plain", "roleplay"]}}
    ]
    with pytest.raises(ValueError, match="context_weights entry covering"):
        compile_from(tmp_path, plan)

    plan["taxonomy"]["arms"]["wrapper"]["context_weights"] = [
        {
            "when": {"language": "maltese"},
            "values": [{"id": "plain", "weight": 0.7}, {"id": "roleplay", "weight": 0.3}],
        }
    ]
    tree = compile_from(tmp_path, plan)
    assert sum(node.target_path_weight for node in tree.terminal_paths) == pytest.approx(1.0)
    weights = {
        node.value: node.conditional_target_weight
        for node in tree.terminal_paths
        if node.node_id.startswith("all|problem_class=bio|semantic_seed_id=s003|language=maltese|")
    }
    assert weights == pytest.approx({"plain": 0.7, "roleplay": 0.3})


def test_context_weights_cannot_introduce_undeclared_values(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["arms"]["wrapper"]["context_weights"] = [
        {
            "when": {"language": "maltese"},
            "values": [{"id": "plain", "weight": 0.5}, {"id": "invented", "weight": 0.5}],
        }
    ]
    with pytest.raises(ValueError, match="undeclared values"):
        compile_from(tmp_path, plan)


def test_constraint_that_empties_a_level_is_rejected(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["renormalization"] = "renormalize_valid_children"
    plan["taxonomy"]["constraints"] = [
        {"id": "c1", "when": {"language": "maltese"}, "allow": {"wrapper": ["absent"]}}
    ]
    with pytest.raises(ValueError, match="pruned by a constraint"):
        compile_from(tmp_path, plan)


def test_seed_under_two_fixed_parents_is_rejected(tmp_path: Path) -> None:
    seeds = """problem_class,semantic_seed_id,problem_class_weight,seed_weight
chem,s001,0.5,1.0
bio,s001,0.5,1.0
"""
    with pytest.raises(ValueError, match="two fixed parent paths"):
        compile_from(tmp_path, base_plan(), seeds)


def test_arm_weights_that_do_not_normalize_are_rejected(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["arms"]["language"]["values"] = [
        {"id": "english", "weight": 0.5},
        {"id": "maltese", "weight": 0.4},
    ]
    with pytest.raises(ValueError, match="sum to 0.9, not 1"):
        compile_from(tmp_path, plan)


def test_fixed_level_weights_that_do_not_normalize_are_rejected(tmp_path: Path) -> None:
    seeds = """problem_class,semantic_seed_id,problem_class_weight,seed_weight
chem,s001,0.6,1.0
bio,s002,0.6,1.0
"""
    with pytest.raises(ValueError, match="not 1"):
        compile_from(tmp_path, base_plan(), seeds)


def test_inconsistent_fixed_weight_within_a_parent_is_rejected(tmp_path: Path) -> None:
    seeds = """problem_class,semantic_seed_id,problem_class_weight,seed_weight
chem,s001,0.6,1.0
chem,s002,0.3,1.0
bio,s003,0.4,1.0
"""
    with pytest.raises(ValueError, match="inconsistent weights"):
        compile_from(tmp_path, base_plan(), seeds)


def test_zero_weight_branches_leave_the_estimand(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["arms"]["wrapper"]["values"] = [
        {"id": "plain", "weight": 0.5},
        {"id": "roleplay", "weight": 0.5},
        {"id": "nested", "weight": 0.0},
    ]
    tree = compile_from(tmp_path, plan)
    assert len(tree.terminal_paths) == 3 * 2 * 2
    assert all("wrapper=nested" not in node.node_id for node in tree.nodes)
    assert any("wrapper=nested" in entry for entry in tree.pruned_zero_weight)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["taxonomy"].update(hierarchy=["a", "a"]), "repeats a level"),
        (
            lambda p: p["taxonomy"]["arms"].update(orphan={"values": [{"id": "x", "weight": 1.0}]}),
            "absent from the hierarchy",
        ),
        (
            lambda p: p["taxonomy"].update(semantic_seed_level="language"),
            "must be exactly the hierarchy prefix",
        ),
        (
            lambda p: p["taxonomy"]["fixed_levels"].update(
                language={"weights": "equal_within_parent"}
            ),
            "declared as both fixed and arm",
        ),
    ],
)
def test_invalid_nesting_is_rejected(tmp_path: Path, mutate, message: str) -> None:
    plan = base_plan()
    mutate(plan)
    with pytest.raises(ValidationError, match=message):
        load_generative_plan(materialize(tmp_path, plan))


def test_constraint_must_condition_strictly_above_what_it_prunes(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["renormalization"] = "renormalize_valid_children"
    plan["taxonomy"]["constraints"] = [
        {"id": "c1", "when": {"wrapper": "plain"}, "allow": {"language": ["english"]}}
    ]
    with pytest.raises(ValidationError, match="at or above the level it conditions on"):
        load_generative_plan(materialize(tmp_path, plan))


def test_constraints_cannot_prune_a_fixed_level(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["renormalization"] = "renormalize_valid_children"
    plan["taxonomy"]["constraints"] = [
        {"id": "c1", "when": {"problem_class": "chem"}, "allow": {"semantic_seed_id": ["s001"]}}
    ]
    with pytest.raises(ValidationError, match="not an arm"):
        load_generative_plan(materialize(tmp_path, plan))


def test_maximum_paths_guards_the_expansion(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["maximum_paths"] = 4
    with pytest.raises(ValueError, match="maximum_paths"):
        compile_from(tmp_path, plan)


def test_reserved_separator_in_an_identifier_is_rejected(tmp_path: Path) -> None:
    seeds = """problem_class,semantic_seed_id,problem_class_weight,seed_weight
chem,s|001,1.0,1.0
"""
    with pytest.raises(ValueError, match="reserved character"):
        compile_from(tmp_path, base_plan(), seeds)


def test_unknown_plan_field_is_rejected(tmp_path: Path) -> None:
    plan = base_plan()
    plan["sampling"]["adaptive"] = True
    with pytest.raises(ValidationError, match="adaptive"):
        load_generative_plan(materialize(tmp_path, plan))


def test_confirmatory_plan_requires_a_freeze_time(tmp_path: Path) -> None:
    plan = base_plan()
    plan["evaluation"]["intent"] = "confirmatory"
    with pytest.raises(ValidationError, match="plan_frozen_at"):
        load_generative_plan(materialize(tmp_path, plan))


def constrained_plan() -> dict[str, Any]:
    plan = base_plan()
    plan["taxonomy"]["renormalization"] = "renormalize_valid_children"
    plan["taxonomy"]["constraints"] = [
        {
            "id": "maltese-wrapper-compatibility",
            "when": {"language": "maltese"},
            "allow": {"wrapper": ["plain", "roleplay"]},
        },
        {
            "id": "bio-english-only",
            "when": {"problem_class": "bio"},
            "allow": {"wrapper": ["plain"]},
        },
    ]
    return plan


def test_every_positive_weight_node_routes_to_a_rewrite_kernel(tmp_path: Path) -> None:
    tree = compile_from(tmp_path, constrained_plan())
    children: dict[str, list[Any]] = {}
    for node in tree.nodes:
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node)

    for node in tree.nodes:
        assert node.target_path_weight > 0.0
        if node.is_terminal:
            assert node.rewrite_kernel_id == "test-kernel"
        else:
            assert children.get(node.node_id), f"{node.node_id} reaches no kernel"


def test_sibling_weights_normalize_after_the_constraint_rule(tmp_path: Path) -> None:
    tree = compile_from(tmp_path, constrained_plan())
    children: dict[str, list[Any]] = {}
    for node in tree.nodes:
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node)

    assert tree.pruned_by_constraint  # the rule actually fired
    for parent_id, group in children.items():
        parent = tree.node(parent_id)
        assert sum(child.conditional_target_weight for child in group) == pytest.approx(1.0)
        assert sum(child.target_path_weight for child in group) == pytest.approx(
            parent.target_path_weight
        )
    assert sum(node.target_path_weight for node in tree.terminal_paths) == pytest.approx(1.0)


def test_the_compiled_tree_is_acyclic_by_construction(tmp_path: Path) -> None:
    """A node id contains its parent's id as a prefix, so a cycle cannot be expressed."""

    tree = compile_from(tmp_path, constrained_plan())
    for node in tree.nodes:
        if node.parent_id is None:
            continue
        assert node.node_id.startswith(f"{node.parent_id}|")
        assert node.depth == tree.node(node.parent_id).depth + 1
