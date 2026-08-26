"""Can sequential Monte Carlo focus compute on the risky branches of the tree?

SMC splits into two halves and they fail differently.

**Without resampling** it is sequential importance sampling with a factorized proposal --
exactly the cross-entropy slice, which `generative_sampling_tradeoff.py` already measured
and closed: the likelihood ratios become unequal coefficients and the bound charges for
that even as the estimator improves.

**With resampling** -- the part that makes it SMC -- particles interact. Whether particle
*i* survives depends on particle *j*'s weight, so the estimator stops being a sum of
independent bounded terms, which is the only object the engines here can bound. The
estimator is not worse; it is outside the class.

There is one honest way to bound it anyway: run K *independent* replicates and treat their
K estimates as K independent bounded observations. That is valid, and it is what this
script measures. The cost is the point -- the bound then rests on K observations rather
than on the budget, so it collapses as replicates get larger and fewer.

What this does and does not establish. The resampling step here acts on a single-level
draw, so it tests the *coupling* faithfully but barely tests sequential adaptation -- a real
SMC would traverse the hierarchy level by level with a proposal that adapts as it descends.
That limitation does not rescue the conclusion, because the penalty measured below is
independent of how good the estimator gets: the bound rests on the replicate count, so an
SMC with half the RMSE would still be bounded by K observations rather than by the budget.
Better focusing does not buy back independence.

Run:
    python research/generative_smc.py
"""

from __future__ import annotations

import numpy as np

from inspect_safety_stats.stats.binomial import binomial_upper_bound
from inspect_safety_stats.stats.direct_mean import (
    weighted_bounded_empirical_bernstein_mean_bounds,
)

ALPHA = 0.05
BUDGET = 1200
REPLICATES = 200


def build_tree(rng: np.random.Generator, *, seeds: int, arms: int) -> tuple[np.ndarray, np.ndarray]:
    """One risky arm among many, which is where SMC is supposed to earn its keep."""

    count = seeds * arms
    seed_of = np.repeat(np.arange(seeds), arms)
    arm_of = np.tile(np.arange(arms), seeds)
    base = rng.gamma(0.6, 1.0, size=seeds)
    base /= base.mean()
    effect = np.full(arms, 0.05)
    effect[0] = 8.0
    risks = np.clip(0.02 * base[seed_of] * effect[arm_of], 1e-6, 0.95)
    return np.full(count, 1.0 / count), risks


def direct(weights, risks, rng, budget):
    """Every draw carries 1/T, so the sharp exact kernel applies."""

    paths = rng.choice(len(weights), budget, p=weights)
    y = (rng.random(budget) < risks[paths]).astype(int)
    estimate = float(y.mean())
    upper = binomial_upper_bound(int(y.sum()), budget, alpha=ALPHA)
    return estimate, upper


def smc_replicate(weights, risks, rng, particles, *, proposal):
    """One SMC pass with resampling, returning a single estimate in [0, 1].

    Resampling is what concentrates effort on the risky arm, and is also what couples the
    particles to each other.
    """

    paths = rng.choice(len(weights), particles, p=proposal)
    ratios = weights[paths] / proposal[paths]
    ratios /= ratios.sum()
    # Systematic resampling on the importance weights: the interaction step.
    positions = (rng.random() + np.arange(particles)) / particles
    kept = paths[np.searchsorted(np.cumsum(ratios), positions)]
    y = (rng.random(particles) < risks[kept]).astype(float)
    return float(y.mean())


def smc_bounded(weights, risks, rng, budget, *, replicates, proposal):
    """K independent replicates, bounded as K independent observations in [0, 1]."""

    per = max(2, budget // replicates)
    estimates = np.array(
        [smc_replicate(weights, risks, rng, per, proposal=proposal) for _ in range(replicates)]
    )
    # Each replicate is one independent observation in [0, 1]; nothing is unclassified,
    # so the lower and upper completions coincide.
    bounds = weighted_bounded_empirical_bernstein_mean_bounds(
        estimates,
        estimates,
        np.full(replicates, 1.0 / replicates),
        alpha_lower=ALPHA,
        alpha_upper=ALPHA,
        independence_profile="sampling_unit",
    )
    return float(estimates.mean()), float(min(1.0, bounds.upper))


def main() -> None:
    rng = np.random.default_rng(0)
    weights, risks = build_tree(rng, seeds=25, arms=8)
    truth = float(weights @ risks)
    # The proposal SMC would learn: proportional to weight times root-risk.
    proposal = weights * np.sqrt(risks)
    proposal /= proposal.sum()

    print(f"tree: {len(weights)} paths, root risk {truth:.5f}, "
          f"risky arm {risks.max() / risks.min():.0f}x the quietest")
    print(f"budget {BUDGET} evaluations, alpha {ALPHA}, {REPLICATES} trials\n")

    direct_est, direct_up = [], []
    for _ in range(REPLICATES):
        e, u = direct(weights, risks, rng, BUDGET)
        direct_est.append(e)
        direct_up.append(u)
    d_rmse = float(np.sqrt(np.mean((np.array(direct_est) - truth) ** 2)))
    d_upper = float(np.mean(direct_up))
    print(f"{'design':<34}{'RMSE':>10}{'vs direct':>11}{'bound':>10}{'vs direct':>11}")
    print(f"  {'direct sampling':<32}{d_rmse:>10.5f}{'1.00x':>11}{d_upper:>10.5f}{'1.00x':>11}")

    for k in (400, 100, 25, 8):
        est, up = [], []
        for _ in range(REPLICATES):
            e, u = smc_bounded(weights, risks, rng, BUDGET, replicates=k, proposal=proposal)
            est.append(e)
            up.append(u)
        rmse = float(np.sqrt(np.mean((np.array(est) - truth) ** 2)))
        upper = float(np.mean(up))
        label = f"SMC, {k} replicates x {BUDGET // k}"
        print(f"  {label:<32}{rmse:>10.5f}{d_rmse / rmse:>10.2f}x{upper:>10.5f}"
              f"{upper / d_upper:>10.2f}x")

    print("\nThe two columns move in opposite directions, and neither favours SMC here.")
    print("RMSE only reaches parity with direct sampling at the fewest, largest replicates,")
    print("where the bound is already an order of magnitude wider. The bound rests on the")
    print("replicate count rather than the budget, because independence is what the engines")
    print("need and resampling is what spends it.")


if __name__ == "__main__":
    main()
