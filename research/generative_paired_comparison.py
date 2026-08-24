"""Does pairing on shared rewrites buy anything under a finite-sample bound?

Slice 0 established that variance reduction does not automatically improve the bound this
package reports -- inverse-probability weighting reduced RMSE while widening the interval.
Pairing is also a variance-reduction device, so the same question has to be asked before
it is offered: measured on the reported bound, not on the estimator.

Run:
    python research/generative_paired_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "generative_multilingual"
sys.path.insert(0, str(EXAMPLE_DIR))

import simulated_target  # noqa: E402

from inspect_safety_stats.generative.comparisons import compare_cohorts  # noqa: E402
from inspect_safety_stats.generative.inference import UnitOutcome  # noqa: E402
from inspect_safety_stats.generative.sampling import draw_selections, plan_design  # noqa: E402
from inspect_safety_stats.generative.schema import load_generative_plan  # noqa: E402
from inspect_safety_stats.generative.taxonomy import compile_taxonomy  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "generative_multilingual"


def outcomes(risks: np.ndarray, latent: np.ndarray, unit_ids: list[str]) -> list[UnitOutcome]:
    return [
        UnitOutcome(unit_id, trials=1, failures=int(draw < risk), unclassified=0)
        for unit_id, risk, draw in zip(unit_ids, risks, latent, strict=True)
    ]


def replicate(taxonomy, design, selections, ratio, coupling, rng):
    """One evaluation of both arms on the same rewrites.

    ``coupling`` is the probability that a unit's two arms share their latent draw. At 1.0
    the arms respond identically to prompt difficulty and only the risk gap creates
    discordance; at 0.0 the arms are independent given the prompt.
    """

    unit_ids = [record.sampling_unit_id for record in selections]
    base = np.array([simulated_target.path_risk(record.node_id) for record in selections])
    shared = rng.random(len(unit_ids))
    independent = rng.random(len(unit_ids))
    couple = rng.random(len(unit_ids)) < coupling
    candidate_latent = np.where(couple, shared, independent)
    return (
        outcomes(base * ratio, candidate_latent, unit_ids),
        outcomes(base, shared, unit_ids),
    )


def main() -> None:
    plan = load_generative_plan(EXAMPLE / "plan-direct.yaml")
    taxonomy = compile_taxonomy(plan, base_path=EXAMPLE)
    truth_base = simulated_target.true_node_risk(taxonomy, taxonomy.root_id)

    print("Root risk difference, one-sided upper bound at alpha=0.05.")
    print(f"Baseline true root risk {truth_base:.5f}.\n")

    for coupling in (1.0, 0.5, 0.0):
        print(f"coupling = {coupling:.1f}")
        print(f"  {'budget':>7} {'ratio':>6} {'true':>9} {'paired':>9} {'unpaired':>9} {'gain':>7}")
        for budget in (600, 2400):
            plan.sampling.total_rewrites = budget
            design = plan_design(taxonomy, plan)
            selections = draw_selections(taxonomy, design)
            for ratio in (1.0, 1.2):
                truth = truth_base * (ratio - 1.0)
                rng = np.random.default_rng(20)
                paired_upper, unpaired_upper = [], []
                covered = 0
                replicates = 60
                for _ in range(replicates):
                    candidate, baseline = replicate(
                        taxonomy, design, selections, ratio, coupling, rng
                    )
                    together = compare_cohorts(
                        taxonomy, design, selections, candidate, baseline, alpha=0.05
                    )
                    apart = compare_cohorts(
                        taxonomy, design, selections, candidate, baseline,
                        alpha=0.05, paired=False,
                    )
                    paired_upper.append(together.upper)
                    unpaired_upper.append(apart.upper)
                    covered += together.upper >= truth
                mean_paired = float(np.mean(paired_upper))
                mean_unpaired = float(np.mean(unpaired_upper))
                flag = "" if covered / replicates >= 0.95 else "  COVERAGE LOW"
                print(
                    f"  {budget:>7} {ratio:>6.1f} {truth:>9.5f} {mean_paired:>9.5f} "
                    f"{mean_unpaired:>9.5f} {mean_unpaired / mean_paired:>6.2f}x{flag}"
                )
        print()


if __name__ == "__main__":
    main()
