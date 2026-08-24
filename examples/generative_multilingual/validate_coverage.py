"""Measure whether the reported intervals actually cover, over many evaluations.

A single run shows one interval. The guarantee is about the long run: at alpha = 0.05 a
one-sided upper bound may sit below the truth in about one evaluation in twenty, and a
run where that happens is the guarantee working as specified, not failing.

Each replicate redraws the sample as well as the outcomes, so what is measured is the
whole procedure -- path selection, rewrite generation and rollouts -- not just the noise
in one fixed sample.

Run:
    python examples/generative_multilingual/validate_coverage.py --replicates 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import simulated_target

from inspect_safety_stats.generative.execution import execute_rewrites, load_seed_texts
from inspect_safety_stats.generative.inference import estimate_slice, estimate_tree
from inspect_safety_stats.generative.inspect_bridge import join_outcomes
from inspect_safety_stats.generative.kernels import (
    DeterministicFakeKernel,
    resolve_kernel_fingerprint,
)
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy

HERE = Path(__file__).parent
LANGUAGES = ("english", "french", "german", "maltese")


def one_replicate(plan, taxonomy, fingerprint, texts, seed: int):
    """Redraw the whole evaluation under a fresh selection seed."""

    plan.evaluation.selection_seed = seed
    design = plan_design(taxonomy, plan)
    selections = draw_selections(taxonomy, design)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    samples = simulated_target.evaluate(
        selections, result.accepted, np.random.default_rng(seed)
    )
    joined = join_outcomes(plan, selections, result, samples, cohort_id="baseline")
    return design, selections, list(joined.outcomes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="plan-direct.yaml")
    parser.add_argument("--replicates", type=int, default=400)
    parser.add_argument("--budget", type=int, default=0, help="override sampling.total_rewrites")
    arguments = parser.parse_args()

    plan = load_generative_plan(HERE / arguments.plan)
    if arguments.budget:
        plan.sampling.total_rewrites = arguments.budget
    taxonomy = compile_taxonomy(plan, base_path=HERE)
    fingerprint = resolve_kernel_fingerprint(plan, base_path=HERE)
    texts = load_seed_texts(plan, base_path=HERE)
    alpha = plan.inference.alpha

    root_truth = simulated_target.true_node_risk(taxonomy, taxonomy.root_id)
    slice_truth = {}
    for language in LANGUAGES:
        matching = [
            node for node in taxonomy.terminal_paths if f"language={language}" in node.node_id
        ]
        mass = sum(node.target_path_weight for node in matching)
        slice_truth[language] = sum(
            node.target_path_weight * simulated_target.path_risk(node.node_id)
            for node in matching
        ) / mass

    print(f"plan       : {arguments.plan}")
    print(f"budget     : {plan.sampling.total_rewrites} rewrites")
    print(f"replicates : {arguments.replicates}")
    print(f"alpha      : {alpha} (one-sided upper)\n")
    print(f"true root risk: {root_truth:.5f}\n")

    root_covered = 0
    root_uppers: list[float] = []
    root_estimates: list[float] = []
    slice_covered = dict.fromkeys(LANGUAGES, 0)
    for replicate in range(arguments.replicates):
        design, selections, outcomes = one_replicate(
            plan, taxonomy, fingerprint, texts, seed=1_000_000 + replicate
        )
        tree = estimate_tree(
            taxonomy, design, selections, outcomes, alpha=alpha, report_all_nodes=False
        )
        root_covered += tree.root.upper >= root_truth
        root_uppers.append(tree.root.upper)
        if tree.root.estimate is not None:
            root_estimates.append(tree.root.estimate)
        for language in LANGUAGES:
            cut = estimate_slice(
                taxonomy, design, selections, outcomes, level="language", value=language,
                alpha=alpha,
            )
            slice_covered[language] += cut.upper >= slice_truth[language]

    replicates = arguments.replicates
    print("root")
    print(f"  empirical coverage   {root_covered / replicates:.3f}   (nominal >= {1 - alpha})")
    print(f"  mean point estimate  {np.mean(root_estimates):.5f}   (truth {root_truth:.5f})")
    print(f"  mean upper bound     {np.mean(root_uppers):.5f}")
    print(f"  bound slack          {np.mean(root_uppers) - root_truth:.5f}\n")

    print("cross-cutting language slices")
    print(f"  {'language':<10} {'weight':>7} {'true':>8} {'coverage':>9}")
    for language in LANGUAGES:
        weight = sum(
            node.target_path_weight
            for node in taxonomy.terminal_paths
            if f"language={language}" in node.node_id
        )
        print(
            f"  {language:<10} {weight:>7.3f} {slice_truth[language]:>8.4f} "
            f"{slice_covered[language] / replicates:>9.3f}"
        )

    failures = [
        name
        for name, covered in slice_covered.items()
        if covered / replicates < 1 - alpha - 3 * (alpha * (1 - alpha) / replicates) ** 0.5
    ]
    print()
    if root_covered / replicates < 1 - alpha:
        print("  ROOT COVERAGE BELOW NOMINAL -- this would be a real defect.")
    elif failures:
        print(f"  slice coverage below nominal for: {', '.join(failures)}")
    else:
        print("  every interval met its nominal coverage.")


if __name__ == "__main__":
    main()
