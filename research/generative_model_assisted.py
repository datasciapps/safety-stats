"""Does model-assisted estimation improve the bound this package reports?

Model-assisted estimation -- GREG in survey sampling, two-level MLMC in Giles's
formulation -- splits the estimand with a telescoping identity:

    R = E[f(X)]  +  E[Y - f(X)]
        level 0      level 1
        cheap        expensive, but concentrated when f couples well to Y

Slice 0 established that variance reduction does not automatically improve a
finite-sample bound, so this is measured before being built, the same way.

Two variants are possible here and they fail differently.

* A **prompt-level** predictor can in principle track individual outcomes, but it needs
  the generated prompt, so it pays the rewrite cost that level 1 also pays. The ratio
  between level costs is floored by that shared step.
* A **path-level** predictor is free -- E[f] is computable exactly as a weighted sum over
  the compiled taxonomy, with no level-0 sampling and no level-0 alpha -- but it can only
  capture between-path structure.

Run:
    python research/generative_model_assisted.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "generative_multilingual"
sys.path.insert(0, str(EXAMPLE_DIR))

import simulated_target  # noqa: E402

from inspect_safety_stats.generative.schema import load_generative_plan  # noqa: E402
from inspect_safety_stats.generative.taxonomy import compile_taxonomy  # noqa: E402
from inspect_safety_stats.stats.binomial import (  # noqa: E402
    binomial_lower_bound,
    binomial_upper_bound,
)

ALPHA = 0.05
BUDGET = 1200.0
COST_RATIOS = (1, 3, 10, 100)
PREDICTORS = (
    ("perfect", 1.00, 1.0000),
    ("excellent", 0.95, 0.9990),
    ("good", 0.80, 0.9900),
    ("fair", 0.60, 0.9700),
)


def _taxonomy() -> tuple[np.ndarray, np.ndarray, float]:
    plan = load_generative_plan(EXAMPLE_DIR / "plan-direct.yaml")
    taxonomy = compile_taxonomy(plan, base_path=EXAMPLE_DIR)
    weights = np.array([node.target_path_weight for node in taxonomy.terminal_paths])
    risks = np.array([simulated_target.path_risk(node.node_id) for node in taxonomy.terminal_paths])
    return weights, risks, float(weights @ risks)


def direct_bound(risk: float, rng: np.random.Generator, replicates: int) -> float:
    """One level: every evaluation spent on the target, bounded exactly."""

    total = int(BUDGET)
    return float(
        np.mean([
            binomial_upper_bound(int(rng.binomial(total, risk)), total, alpha=ALPHA)
            for _ in range(replicates)
        ])
    )


def two_level_bound(
    risk: float,
    sensitivity: float,
    specificity: float,
    cost_ratio: float,
    expensive: int,
    rng: np.random.Generator,
    replicates: int,
) -> float:
    """Bound R with `expensive` level-1 draws and whatever cheap draws the budget leaves.

    A level-1 sample costs one target evaluation plus one predictor call; a level-0 sample
    costs only the predictor call. Because the predictor is binary, the residual is nonzero
    only where f and Y disagree, so the level-1 term uses the same exact two-cell
    construction as a paired comparison.
    """

    level_one_cost = 1.0 + 1.0 / cost_ratio
    cheap = int((BUDGET - expensive * level_one_cost) * cost_ratio)
    if cheap <= 0 or expensive < 2:
        return float("inf")

    false_positive = (1 - specificity) * (1 - risk)
    false_negative = (1 - sensitivity) * risk
    predictor_rate = risk + false_positive - false_negative
    if not 0.0 < predictor_rate < 1.0:
        return float("inf")

    bounds = []
    for _ in range(replicates):
        level_zero = binomial_upper_bound(
            int(rng.binomial(cheap, predictor_rate)), cheap, alpha=ALPHA / 2
        )
        missed = int(rng.binomial(expensive, false_negative))
        spurious = int(rng.binomial(expensive, false_positive))
        level_one = binomial_upper_bound(
            missed, expensive, alpha=ALPHA / 4
        ) - binomial_lower_bound(spurious, expensive, alpha=ALPHA / 4)
        bounds.append(level_zero + level_one)
    return float(np.mean(bounds))


def best_two_level(
    risk: float,
    sensitivity: float,
    specificity: float,
    cost_ratio: float,
    rng: np.random.Generator,
    replicates: int = 40,
) -> tuple[float, int]:
    """The allocation is chosen to minimise the reported bound, not the variance.

    That distinction is the point: MLMC's own allocation rule optimises mean squared
    error, and slice 0 showed that optimising the estimator can widen the interval.
    """

    ceiling = BUDGET / (1 + 1 / cost_ratio) - 1
    grid = np.unique(np.linspace(20, ceiling, 40).astype(int))
    scored = [
        (two_level_bound(risk, sensitivity, specificity, cost_ratio, int(n), rng, replicates),
         int(n))
        for n in grid
    ]
    return min(scored)


def empirical_bernstein_upper(values: np.ndarray, alpha: float) -> float:
    """Maurer-Pontil one-sided upper bound for a mean of iid values in [0, 1]."""

    count = len(values)
    log = float(np.log(1.0 / alpha))
    return (
        float(np.mean(values))
        + float(np.sqrt(2.0 * np.var(values, ddof=1) * log / count))
        + 7.0 * log / (3.0 * (count - 1))
    )


def prompt_level(risk: float) -> None:
    rng = np.random.default_rng(0)
    baseline = direct_bound(risk, rng, 200)
    print("Variant 1: a prompt-level predictor, the true two-level construction.\n")
    print(f"  direct sampling upper bound: {baseline:.5f}\n")
    header = f"  {'predictor':>10} {'sens':>5} {'spec':>7} "
    header += "".join(f"{'k=' + str(k):>13}" for k in COST_RATIOS)
    print(header)
    for name, sensitivity, specificity in PREDICTORS:
        row = f"  {name:>10} {sensitivity:>5.2f} {specificity:>7.4f} "
        for cost_ratio in COST_RATIOS:
            bound, expensive = best_two_level(risk, sensitivity, specificity, cost_ratio, rng)
            row += f"{baseline / bound:>7.2f}x n={expensive:<4}"
        print(row)
    print("\n  Values are direct/two-level: above 1.00 means the two-level bound is tighter.")
    print("  n is the level-1 sample count the optimiser chose.")


def path_level(weights: np.ndarray, risks: np.ndarray, risk: float) -> None:
    rng = np.random.default_rng(0)
    total, replicates = int(BUDGET), 400
    direct, assisted, residual_variance, raw_variance = [], [], [], []
    for _ in range(replicates):
        drawn = rng.choice(len(risks), size=total, p=weights)
        outcomes = (rng.random(total) < risks[drawn]).astype(float)
        direct.append(binomial_upper_bound(int(outcomes.sum()), total, alpha=ALPHA))
        residual = outcomes - risks[drawn]
        rescaled = (residual + 1.0) / 2.0
        assisted.append(risk + (2.0 * empirical_bernstein_upper(rescaled, ALPHA) - 1.0))
        residual_variance.append(float(np.var(residual, ddof=1)))
        raw_variance.append(float(np.var(outcomes, ddof=1)))

    exact, model = float(np.mean(direct)), float(np.mean(assisted))
    removed = 100 * (1 - float(np.mean(residual_variance)) / float(np.mean(raw_variance)))
    print("\n\nVariant 2: a path-level predictor, where E[f] is exact and costs nothing.\n")
    print(f"  direct, exact Clopper-Pearson  {exact:.5f}")
    print(f"  model-assisted, oracle path f  {model:.5f}")
    print(f"                                 {exact / model:.2f}x")
    print(f"  variance removed by the oracle {removed:.1f}%")
    print("\n  Level 0 is free and exact, and the predictor is the best one that could")
    print("  exist at this level. The residual still needs a bounded-mean inequality")
    print("  rather than an exact binomial one, and that costs more than it saves.")


def main() -> None:
    weights, risks, risk = _taxonomy()
    print(f"Worked-example design: root risk {risk:.5f}, "
          f"budget {int(BUDGET)} full evaluations, alpha {ALPHA}.\n")
    prompt_level(risk)
    path_level(weights, risks, risk)


if __name__ == "__main__":
    main()
