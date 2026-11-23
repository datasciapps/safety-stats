"""Compile the authored plan and show what the canonical taxonomy actually contains.

Run:
    python examples/generative_multilingual/compile_tree.py
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from safety_stats.generative.schema import load_generative_plan
from safety_stats.generative.taxonomy import compile_taxonomy

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(HERE / "plan-direct.yaml"))
    arguments = parser.parse_args()

    plan = load_generative_plan(arguments.plan)
    taxonomy = compile_taxonomy(plan, base_path=HERE)

    print(f"taxonomy fingerprint : {taxonomy.fingerprint[:16]}")
    print(f"nodes                : {len(taxonomy.nodes)}")
    print(f"terminal paths       : {len(taxonomy.terminal_paths)}")

    print("\nnodes per level")
    counts = Counter(node.level or "(root)" for node in taxonomy.nodes)
    for level in ("(root)", *taxonomy.levels):
        print(f"  {level:<18} {counts[level]:>5}")

    print("\nweight invariants")
    total = sum(node.target_path_weight for node in taxonomy.terminal_paths)
    children = defaultdict(list)
    for node in taxonomy.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)
    worst = max(
        abs(sum(child.conditional_target_weight for child in group) - 1.0)
        for group in children.values()
    )
    print(f"  terminal path weights sum to      {total:.12f}")
    print(f"  worst sibling normalization error {worst:.2e}")

    print("\nwhat the constraint removed")
    print(f"  paths pruned by constraint : {len(taxonomy.pruned_by_constraint)}")
    for entry in taxonomy.pruned_by_constraint[:3]:
        print(f"    {entry.split('|', 3)[-1]}")

    print("\ncontext weights in effect (style, by language)")
    for language in ("english", "french"):
        weights = {
            node.value: node.conditional_target_weight
            for node in taxonomy.nodes
            if node.level == "style"
            and node.parent_id is not None
            and node.parent_id.endswith(f"language={language}")
            and "semantic_seed_id=cs-01" in node.node_id
        }
        rendered = ", ".join(f"{key}={value:.2f}" for key, value in sorted(weights.items()))
        print(f"  {language:<8} {rendered}")

    print("\nrenormalization after pruning (wrapper, by language)")
    for language in ("english", "maltese"):
        weights = {
            node.value: node.conditional_target_weight
            for node in taxonomy.nodes
            if node.level == "wrapper"
            and node.parent_id is not None
            and f"language={language}|" in node.node_id
            and node.parent_id.endswith("style=plain")
            and "semantic_seed_id=cs-01" in node.node_id
        }
        rendered = ", ".join(f"{key}={value:.3f}" for key, value in sorted(weights.items()))
        print(f"  {language:<8} {rendered}")

    print("\nheaviest terminal paths")
    for node in sorted(
        taxonomy.terminal_paths, key=lambda item: -item.target_path_weight
    )[:5]:
        print(f"  {node.target_path_weight:.5f}  {node.node_id.split('|', 2)[-1]}")


if __name__ == "__main__":
    main()
