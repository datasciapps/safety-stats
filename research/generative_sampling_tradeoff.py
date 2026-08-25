"""Does variance reduction help a package that reports finite-sample bounds?

Slice 0 for the generative extension. The original proposal, now archived at
docs/archive/generative-hierarchy-evaluation-plan.md, proposes in order
hierarchical stratification (slice 3), cost-aware Neyman allocation (slice 4), and
cross-entropy adaptive importance sampling (slice 7). All three are variance-reduction
techniques. This asks whether they improve the number this package actually reports.

Run:  python research/generative_sampling_tradeoff.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from inspect_safety_stats.stats.direct_mean import (
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
)

ENGINES = {
    "kl": weighted_bounded_kl_chernoff_mean_bounds,
    "eb": weighted_bounded_empirical_bernstein_mean_bounds,
}


@dataclass(frozen=True)
class Tree:
    """A finite structural hierarchy: each path has a target weight and a true risk."""

    weights: np.ndarray
    risks: np.ndarray
    arms: np.ndarray

    @property
    def root(self) -> float:
        return float(self.weights @ self.risks)


def build_tree(rng: np.random.Generator, *, seeds: int, arms: int, regime: str) -> Tree:
    count = seeds * arms
    seed_of = np.repeat(np.arange(seeds), arms)
    arm_of = np.tile(np.arange(arms), seeds)
    base = rng.gamma(0.6, 1.0, size=seeds)
    base /= base.mean()
    if regime == "concentrated":
        effect = np.full(arms, 0.05)
        effect[0] = 8.0
    elif regime == "graded":
        effect = np.geomspace(0.15, 4.0, arms)
    else:
        effect = np.linspace(0.7, 1.4, arms)
    risks = np.clip(0.02 * base[seed_of] * effect[arm_of], 1e-6, 0.95)
    return Tree(np.full(count, 1.0 / count), risks, arm_of)


def bound_width(
    values: np.ndarray, coefficients: np.ndarray, scale: float, *, engine: str, alpha: float
) -> float:
    """Width of the finite-sample bound for ``scale * sum(coefficients * values)``."""
    total = float(coefficients.sum())
    bounds = ENGINES[engine](
        values,
        values,
        coefficients / total,
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        independence_profile="sampling_unit",
    )
    upper = min(1.0, scale * total * bounds.upper)
    lower = max(0.0, scale * total * bounds.lower)
    return float(upper - lower)


def _evaluate(tree: Tree, rng: np.random.Generator, path: int) -> float:
    return float(rng.random() < tree.risks[path])


def direct(tree: Tree, rng: np.random.Generator, budget: int, **kw: object) -> tuple[float, float]:
    """Draw paths from the declared target distribution. All coefficients equal 1/T."""
    paths = rng.choice(len(tree.weights), budget, p=tree.weights)
    y = (rng.random(budget) < tree.risks[paths]).astype(float)
    return float(y.mean()), bound_width(
        y, np.full(budget, 1.0 / budget), 1.0, **kw  # type: ignore[arg-type]
    )


def stratified(
    tree: Tree, rng: np.random.Generator, budget: int, *, neyman: bool, **kw: object
) -> tuple[float, float]:
    strata = np.unique(tree.arms)
    mass = np.array([tree.weights[tree.arms == h].sum() for h in strata])
    if neyman:
        # A pilot is needed to estimate the within-stratum spread Neyman allocates on.
        per = max(2, int(budget * 0.25) // len(strata))
        sigma = np.empty(len(strata))
        for i, h in enumerate(strata):
            idx = np.flatnonzero(tree.arms == h)
            y = np.array([_evaluate(tree, rng, rng.choice(idx)) for _ in range(per)])
            sigma[i] = max(float(np.sqrt(y.mean() * (1 - y.mean()))), 0.05)
        share = mass * sigma
        remaining = budget - per * len(strata)
        alloc = np.maximum(1, np.floor(remaining * share / share.sum()).astype(int))
    else:
        alloc = np.maximum(1, np.floor(budget * mass / mass.sum()).astype(int))

    values, coefficients = [], []
    for i, h in enumerate(strata):
        idx = np.flatnonzero(tree.arms == h)
        y = np.array([_evaluate(tree, rng, rng.choice(idx)) for _ in range(alloc[i])])
        values.append(y)
        coefficients.append(np.full(alloc[i], mass[i] / alloc[i]))
    values = np.concatenate(values)
    coefficients = np.concatenate(coefficients)
    return float(coefficients @ values), bound_width(
        values, coefficients, 1.0, **kw  # type: ignore[arg-type]
    )


def oracle_importance(
    tree: Tree, rng: np.random.Generator, budget: int, *, epsilon: float, **kw: object
) -> tuple[float, float]:
    """Importance sampling handed the second-moment-optimal proposal for free.

    q* proportional to w(h) * sqrt(p_h) is optimal for a Bernoulli outcome -- not
    w(h) * p_h. Giving it away removes any question about learner quality, so this
    upper-bounds what a cross-entropy learner could ever reach.
    """
    strata = np.unique(tree.arms)
    mass = np.array([tree.weights[tree.arms == h].sum() for h in strata])
    risk = np.array([tree.risks[tree.arms == h].mean() for h in strata])
    optimal = mass * np.sqrt(risk)
    optimal /= optimal.sum()
    proposal = epsilon * mass + (1 - epsilon) * optimal

    picks = rng.choice(len(strata), budget, p=proposal)
    ratio = mass[picks] / proposal[picks]
    y = np.array(
        [_evaluate(tree, rng, rng.choice(np.flatnonzero(tree.arms == strata[h]))) for h in picks]
    )
    contribution = ratio * y
    # The defensive mixture bounds the likelihood ratio, which is what makes a
    # finite-sample bound available at all -- and what it charges for.
    cap = float(mass.max() / proposal.min())
    return float(contribution.mean()), bound_width(
        contribution / cap, np.full(budget, 1.0 / budget), cap, **kw  # type: ignore[arg-type]
    )


def compare(
    regime: str, *, seeds: int, arms: int, budget: int, reps: int, engine: str, alpha: float
) -> None:
    rng = np.random.default_rng(1)
    tree = build_tree(rng, seeds=seeds, arms=arms, regime=regime)
    truth = tree.root
    designs = {
        "direct target sampling": lambda: direct(tree, rng, budget, engine=engine, alpha=alpha),
        "stratified (proportional)": lambda: stratified(
            tree, rng, budget, neyman=False, engine=engine, alpha=alpha
        ),
        "stratified (Neyman)": lambda: stratified(
            tree, rng, budget, neyman=True, engine=engine, alpha=alpha
        ),
        "oracle importance sampling": lambda: oracle_importance(
            tree, rng, budget, epsilon=0.2, engine=engine, alpha=alpha
        ),
    }
    print(f"--- {regime}: {len(tree.weights)} paths, root risk {truth:.5f}")
    print(f"{'design':<30} {'RMSE':>9} {'vs direct':>10} {'bound width':>12} {'vs direct':>10}")
    results = {}
    for name, run in designs.items():
        estimates, widths = zip(*[run() for _ in range(reps)], strict=True)
        results[name] = (
            float(np.sqrt(np.mean((np.array(estimates) - truth) ** 2))),
            float(np.mean(widths)),
        )
    base_rmse, base_width = results["direct target sampling"]
    for name, (rmse, width) in results.items():
        print(
            f"{name:<30} {rmse:>9.5f} {rmse / base_rmse:>9.2f}x "
            f"{width:>12.5f} {width / base_width:>9.2f}x"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--reps", type=int, default=250)
    parser.add_argument("--engine", choices=sorted(ENGINES), default="kl")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    print(
        f"Equal budget of {args.budget} target-model evaluations, {args.reps} replications, "
        f"{args.engine} bound at alpha={args.alpha}.\n"
        "RMSE is the classical score. Bound width is what this package would report.\n"
    )
    for regime in ("concentrated", "graded", "diffuse"):
        compare(
            regime,
            seeds=25,
            arms=8,
            budget=args.budget,
            reps=args.reps,
            engine=args.engine,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
