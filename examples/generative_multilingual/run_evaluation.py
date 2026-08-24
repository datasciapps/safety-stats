"""Run one complete generative evaluation offline, from plan to interval.

The target model is simulated (see ``simulated_target.py``) so the answer is known and
the reported interval can be checked against it. Every other step -- selection, rewrite
execution, dataset materialization, the outcome join, inference -- is the real code path.

Run:
    python examples/generative_multilingual/run_evaluation.py
    python examples/generative_multilingual/run_evaluation.py --plan plan-stratified.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import simulated_target

from inspect_safety_stats.generative.comparisons import compare_cohorts
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
from inspect_safety_stats.generative.ledger import write_rewrites, write_selections
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="plan-direct.yaml")
    parser.add_argument("--output", default="run/generative", help="where artifacts are written")
    parser.add_argument("--seed", type=int, default=0, help="seed for the simulated target only")
    arguments = parser.parse_args()

    plan = load_generative_plan(HERE / arguments.plan)
    taxonomy = compile_taxonomy(plan, base_path=HERE)
    fingerprint = resolve_kernel_fingerprint(plan, base_path=HERE)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    design = plan_design(taxonomy, plan)
    selections = draw_selections(taxonomy, design)
    print(f"plan            : {arguments.plan}")
    print(f"taxonomy        : {len(taxonomy.terminal_paths)} terminal paths")
    print(f"design          : {design.method}, {design.total_rewrites} rewrites")
    if design.allocations:
        draws = sorted(item.draws for item in design.allocations)
        print(f"                  {len(design.allocations)} strata, draws {draws}")

    result = execute_rewrites(
        plan,
        selections,
        DeterministicFakeKernel(),
        fingerprint,
        load_seed_texts(plan, base_path=HERE),
    )
    print(f"kernel          : {fingerprint.kernel_id} @ {fingerprint.sha256[:12]}")
    print(
        f"rewrites        : {len(result.accepted)} accepted, "
        f"{result.duplicates_rejected} duplicates rejected, "
        f"{len(result.exhausted)} exhausted"
    )

    write_selections(selections, output / "selections.csv")
    write_rewrites(list(result.rewrites), output / "rewrites.csv")
    write_prompt_store(result.accepted, output / "prompts.restricted.csv")
    datasets = materialize_for_cohorts(plan, selections, result, output / "datasets")
    print(f"materialized    : {datasets['baseline']}")

    samples = simulated_target.evaluate_pair(
        selections, result.accepted, np.random.default_rng(arguments.seed)
    )
    joined = join_outcomes(plan, selections, list(result.rewrites), samples, cohort_id="baseline")
    candidate_joined = join_outcomes(
        plan, selections, list(result.rewrites), samples, cohort_id="candidate"
    )
    for diagnostic in joined.diagnostics:
        print(f"  [{diagnostic.severity}] {diagnostic.code}: {diagnostic.message}")

    tree = estimate_tree(
        taxonomy,
        design,
        selections,
        list(joined.outcomes),
        alpha=plan.inference.alpha,
        coverage_mode=plan.inference.coverage_mode,
        simultaneous_tree=plan.inference.simultaneous_tree,
        report_all_nodes=plan.inference.report_all_nodes,
    )

    truth = simulated_target.true_node_risk(taxonomy, taxonomy.root_id)
    root = tree.root
    print("\nroot")
    print(f"  true risk           {truth:.5f}")
    print(f"  point estimate      {root.estimate:.5f}")
    print(f"  {int((1 - plan.inference.alpha) * 100)}% upper bound     {root.upper:.5f}")
    print(f"  covers the truth    {root.upper >= truth}")
    print(f"  method              {root.method}")
    print(f"  failures / draws    {root.failures} / {root.draws}")
    if not root.equal_coefficients:
        print(f"  coefficient ratio   {root.coefficient_ratio:.5f}")

    print(f"\nnodes reported {len(tree.nodes)}, unsampled {len(tree.unsampled_nodes)}")
    print("\nrisk by language -- a cross-cutting slice, not a subtree")
    print("  language sits below semantic_seed_id, so 'English' is 32 separate nodes.")
    print("  The slice estimator bounds the declared-weight mean over all of them.\n")
    header = (
        f"  {'language':<10} {'weight':>7} {'draws':>6} "
        f"{'true':>8} {'est':>8} {'upper':>8}  covers"
    )
    print(header)
    for value in ("english", "french", "german", "maltese"):
        cut = estimate_slice(
            taxonomy,
            design,
            selections,
            list(joined.outcomes),
            level="language",
            value=value,
            alpha=plan.inference.alpha,
            coverage_mode=plan.inference.coverage_mode,
        )
        slice_truth = sum(
            node.target_path_weight * simulated_target.path_risk(node.node_id)
            for node in taxonomy.terminal_paths
            if f"language={value}" in node.node_id
        ) / cut.node_weight
        print(
            f"  {value:<10} {cut.node_weight:>7.3f} {cut.draws:>6} {slice_truth:>8.4f} "
            f"{cut.estimate:>8.4f} {cut.upper:>8.4f}  {cut.upper >= slice_truth}"
        )

    print(
        "\n  A 'False' above is not a defect. Each slice is pointwise at alpha=0.05, so each\n"
        "  misses about one run in twenty, and four of them miss at least once about one run\n"
        "  in five. validate_coverage.py measures the long-run rate, which is what the\n"
        "  guarantee is about."
    )

    print("\ncandidate versus baseline on the same rewrites")
    truth = simulated_target.true_difference(taxonomy)
    together = compare_cohorts(
        taxonomy,
        design,
        selections,
        list(candidate_joined.outcomes),
        list(joined.outcomes),
        alpha=plan.inference.alpha,
        coverage_mode=plan.inference.coverage_mode,
    )
    apart = compare_cohorts(
        taxonomy,
        design,
        selections,
        list(candidate_joined.outcomes),
        list(joined.outcomes),
        alpha=plan.inference.alpha,
        coverage_mode=plan.inference.coverage_mode,
        paired=False,
    )
    print(f"  true risk difference   {truth:+.5f}")
    print(f"  point estimate         {together.estimate:+.5f}")
    print(f"  95% upper bound        {together.upper:+.5f}   covers: {together.upper >= truth}")
    print(f"  method                 {together.method}")
    if together.discordant_candidate_only is not None:
        concordant = together.draws - (
            together.discordant_candidate_only + together.discordant_baseline_only
        )
        print(
            f"  discordant units       {together.discordant_candidate_only} candidate-only, "
            f"{together.discordant_baseline_only} baseline-only, {concordant} concordant"
        )
    widening = apart.upper / together.upper
    print(f"  ignoring the pairing   {apart.upper:+.5f}   ({widening:.2f}x wider)")
    print(
        "  Concordant units contribute exactly zero to the difference, so the bound reads\n"
        "  only the units where the two models disagreed."
    )

    print("\ndeepest reportable nodes are thin by design")
    thin = [node for node in tree.nodes if node.level == "wrapper"]
    print(f"  wrapper-level nodes reported : {len(thin)}")
    print(f"  median draws each            : {sorted(node.draws for node in thin)[len(thin) // 2]}")
    print(f"  nodes with no draw at all    : {len(tree.unsampled_nodes)}")


if __name__ == "__main__":
    main()
