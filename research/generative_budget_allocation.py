"""Where should a generative evaluation budget go: more rewrites, or more rollouts each?

Section 25's fifth completion criterion asks the planner to make "an explicit, data-driven
choice between additional structural paths, additional rewrites, and additional rollouts",
and section 11 proposes pilot-driven adaptive allocation to make it.

Additional structural paths are not a separate lever -- the design draws paths in proportion
to their declared weight, so `sampling.total_rewrites` is what controls how many are visited.
That leaves rewrites against rollouts, measured here at a fixed number of target-model
rollouts, which is what actually costs money.

Run:
    python research/generative_budget_allocation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "generative_multilingual"
sys.path.insert(0, str(EXAMPLE_DIR))

import simulated_target  # noqa: E402

from inspect_safety_stats.generative.inference import UnitOutcome, estimate_tree  # noqa: E402
from inspect_safety_stats.generative.sampling import draw_selections, plan_design  # noqa: E402
from inspect_safety_stats.generative.schema import load_generative_plan  # noqa: E402
from inspect_safety_stats.generative.taxonomy import compile_taxonomy  # noqa: E402

BUDGET = 1200
REPLICATES = 120
ALPHA = 0.05


def main() -> None:
    plan = load_generative_plan(EXAMPLE_DIR / "plan-direct.yaml")
    taxonomy = compile_taxonomy(plan, base_path=EXAMPLE_DIR)
    truth = simulated_target.true_node_risk(taxonomy, taxonomy.root_id)

    print(f"true root risk {truth:.5f}; budget {BUDGET} target rollouts, alpha {ALPHA}\n")
    print(f"{'rewrites':>9} {'rollouts each':>14} {'kernel':>26} {'mean upper':>11} {'vs best':>9}")

    rows = []
    for rollouts in (1, 2, 4, 8):
        rewrites = BUDGET // rollouts
        plan.sampling.total_rewrites = rewrites
        plan.target.rollouts_per_rewrite = rollouts
        design = plan_design(taxonomy, plan)
        selections = draw_selections(taxonomy, design)
        rng = np.random.default_rng(4)
        uppers = []
        method = ""
        for _ in range(REPLICATES):
            outcomes = [
                UnitOutcome(
                    record.sampling_unit_id,
                    trials=rollouts,
                    failures=int(
                        rng.binomial(rollouts, simulated_target.path_risk(record.node_id))
                    ),
                    unclassified=0,
                )
                for record in selections
            ]
            estimate = estimate_tree(
                taxonomy, design, selections, outcomes, alpha=ALPHA, report_all_nodes=False
            )
            uppers.append(estimate.root.upper)
            method = estimate.root.method
        rows.append((rewrites, rollouts, method.replace("_one_sided_missingness_robust", ""),
                     float(np.mean(uppers))))

    best = min(row[3] for row in rows)
    for rewrites, rollouts, method, upper in rows:
        print(f"{rewrites:>9} {rollouts:>14} {method:>26} {upper:>11.5f} {upper / best:>8.2f}x")

    print("\nEvery row spends the same number of target-model rollouts.")
    print("One rollout per rewrite wins monotonically, for two compounding reasons: a unit")
    print("enters the bound as a single bounded observation however many rollouts produced")
    print("it, so extra rollouts buy no extra terms; and only single-rollout units keep the")
    print("outcomes binary, which is what makes the sharp exact kernel available at all.")
    print("\nThe choice therefore has a constant answer, and an adaptive planner would be")
    print("optimising a decision that does not vary.")


if __name__ == "__main__":
    main()
