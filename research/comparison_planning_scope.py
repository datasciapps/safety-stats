"""Which comparison intervals the resolvable differences of ``plan comparison`` hold for.

``stats.planning.comparison_resolvable_differences`` plans the ``weighted_hoeffding``
difference of a ``comparisons`` entry, which bounds Q = (1 + difference) / 2 directly. A
``comparison_sets`` entry, whatever its method, and ``weighted_kl_chernoff`` everywhere bound
each arm at half the alpha and subtract the endpoints (a two-arm rectangle). This script
simulates the example of docs/model-comparisons.md, "Planning a comparison": 100 equally
weighted prompts, 10 unpaired rollouts per arm, margin 0.05, alpha 0.05 split over both
bounds, power 0.9. At the planned largest supportable difference it counts the runs whose
upper bound falls at or below the margin, for each method as a ``comparisons`` entry
("entry") and as a two-arm rectangle ("rectangle"). The planner guarantees at least 0.9
only for weighted_hoeffding as an entry. It also prints each construction's upper
half-width when both arms observe the same failure rate, spread as evenly over the prompts
as the counts allow.

Section 7.7 of docs/statistical-methods.md quotes its output.

Run:
    python research/comparison_planning_scope.py
"""

from __future__ import annotations

import numpy as np

from inspect_safety_stats.stats.comparisons import (
    COMPARISON_ENGINES,
    direct_risk_difference_bounds,
)
from inspect_safety_stats.stats.planning import comparison_resolvable_differences

PROMPTS = 100
ROLLOUTS = 10
MARGIN = 0.05
ALPHA = 0.025  # one-sided: 0.05 split over both bounds
POWER = 0.9
BASELINE_RATES = (0.12, 0.3, 0.5)
OBSERVED_RATES = (0.02, 0.05, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5)
REPLICATES = 10_000
SEED = 20261119
METHODS = ("weighted_hoeffding", "weighted_empirical_bernstein", "weighted_kl_chernoff")

WEIGHTS = np.full(PROMPTS, 1 / PROMPTS)
TRIALS = np.full(PROMPTS, ROLLOUTS)
NONE = np.zeros(PROMPTS, dtype=int)


def upper_bounds(method: str, candidate: np.ndarray, baseline: np.ndarray) -> tuple[float, float]:
    """The upper bound of the difference as a comparisons entry and as a two-arm rectangle."""

    entry = direct_risk_difference_bounds(
        candidate, NONE, TRIALS, baseline, NONE, TRIALS, WEIGHTS,
        alpha_lower=ALPHA, alpha_upper=ALPHA, method=method,  # type: ignore[arg-type]
    )
    arm = COMPARISON_ENGINES[method].counts  # type: ignore[index]
    candidate_arm = arm(
        candidate, NONE, TRIALS, WEIGHTS,
        alpha_lower=ALPHA / 2, alpha_upper=ALPHA / 2, independence_unit="rollout",
    )
    baseline_arm = arm(
        baseline, NONE, TRIALS, WEIGHTS,
        alpha_lower=ALPHA / 2, alpha_upper=ALPHA / 2, independence_unit="rollout",
    )
    return entry.upper, min(1.0, candidate_arm.upper - baseline_arm.lower)


def main() -> None:
    plan = comparison_resolvable_differences(
        TRIALS, TRIALS, WEIGHTS, margin=MARGIN, alpha_lower=ALPHA, alpha_upper=ALPHA,
        power=POWER,
    )
    difference = plan.maximum_supportable_difference
    if difference is None:
        raise SystemExit("the example has no supportable difference")
    print(f"planned largest supportable difference {difference:.4f} at power {POWER}; "
          f"{REPLICATES} runs per baseline rate")
    rng = np.random.default_rng(SEED)
    for baseline_rate in BASELINE_RATES:
        hits = {(method, kind): 0 for method in METHODS for kind in ("entry", "rectangle")}
        for _ in range(REPLICATES):
            candidate = rng.binomial(TRIALS, baseline_rate + difference)
            baseline = rng.binomial(TRIALS, baseline_rate)
            for method in METHODS:
                entry, rectangle = upper_bounds(method, candidate, baseline)
                hits[method, "entry"] += entry <= MARGIN
                hits[method, "rectangle"] += rectangle <= MARGIN
        print(f"baseline rate {baseline_rate}: share of runs with upper bound <= margin")
        for (method, kind), count in hits.items():
            share = count / REPLICATES
            error = (share * (1 - share) / REPLICATES) ** 0.5
            print(f"  {method:30s} {kind:9s} {share:.4f} (standard error {error:.4f})")
    print("upper half-width (upper bound minus the estimate 0) with the same observed failure "
          "rate in both arms")
    print("  rate  " + "  ".join(f"{method.removeprefix('weighted_'):>24s}" for method in METHODS))
    for rate in OBSERVED_RATES:
        total = round(rate * PROMPTS * ROLLOUTS)
        failures = np.full(PROMPTS, total // PROMPTS)
        failures[: total % PROMPTS] += 1
        widths = [upper_bounds(method, failures, failures) for method in METHODS]
        print(f"  {rate:<4}  " + "  ".join(
            f"entry {entry:.4f} rect {rectangle:.4f}" for entry, rectangle in widths
        ))


if __name__ == "__main__":
    main()
