"""Coverage of one-sided upper bounds where optstop's width rule stops a run (#126).

A fixed design of 200 items with 10 rollouts each, the size of optstop's illustrative
evaluation, runs in a random order fixed before the run. Item failure probabilities are fixed
and heterogeneous: drawn once per scenario from a Beta distribution with mean ``theta`` and
concentration 4, then held fixed while outcomes and orders are resampled. After every item
(10 rollouts, optstop's reanalysis interval) the run stops once the 97% equal-tailed
Beta(1, 1) posterior interval of the pooled failure rate is at most 0.05 wide, after at least
5 items: optstop's validation settings, on the pooled rate rather than its hierarchical
model, which narrows no slower.

At the stopping item tau the script reports how often each one-sided upper bound at
alpha = 0.05 falls below theta:

- the anytime-valid bound of ``stats/sequential.py`` (rollouts independent given the item);
- the fixed-horizon Clopper-Pearson bound on the pooled rollouts of the first tau items, the
  bound a run that is analysed as if tau had been fixed in advance gets;
- the fixed-horizon KL/Chernoff bound of the first tau items with items as the unit
  (``weighted_bounded_kl_chernoff_mean_bounds``), which does not take the rollouts of one item
  as independent draws of theta, and so shows how much of the anytime bound's width is item
  heterogeneity rather than the price of anytime validity;
- the Clopper-Pearson bound at the full horizon, without stopping, for reference.

Run:
    python research/optstop_coverage.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta

from safety_stats.stats.direct_mean import weighted_bounded_kl_chernoff_mean_bounds
from safety_stats.stats.sequential import anytime_upper_bound

ALPHA = 0.05
ITEMS = 200
EPOCHS = 10
WIDTH = 0.05
CREDIBILITY = 0.97
MINIMUM_ITEMS = 5
CONCENTRATION = 4.0
THETAS = (0.01, 0.04, 0.3, 0.5)
RUNS = 2000
SEED = 20261116


def clopper_pearson_upper(failures: np.ndarray | int, trials: np.ndarray | int) -> np.ndarray:
    failures = np.asarray(failures)
    trials = np.asarray(trials)
    upper = beta.ppf(1 - ALPHA, failures + 1, np.maximum(trials - failures, 1))
    return np.where(failures >= trials, 1.0, upper)


def stopping_item(failures: np.ndarray) -> int:
    """The number of items after which the width rule stops, or all of them."""

    cumulative = np.cumsum(failures)
    trials = EPOCHS * np.arange(1, ITEMS + 1)
    tail = (1 - CREDIBILITY) / 2
    width = beta.ppf(1 - tail, cumulative + 1, trials - cumulative + 1) - beta.ppf(
        tail, cumulative + 1, trials - cumulative + 1
    )
    eligible = np.flatnonzero((width <= WIDTH) & (np.arange(1, ITEMS + 1) >= MINIMUM_ITEMS))
    return int(eligible[0]) + 1 if len(eligible) else ITEMS


def scenario(theta: float, rng: np.random.Generator) -> dict[str, float]:
    rates = rng.beta(theta * CONCENTRATION, (1 - theta) * CONCENTRATION, ITEMS)
    target = float(rates.mean())
    weights = np.full(ITEMS, 1 / ITEMS)
    stops, anytime, fixed, items, full = [], [], [], [], []
    for _ in range(RUNS):
        failures = rng.binomial(EPOCHS, rates)
        order = rng.permutation(ITEMS)
        drawn = failures[order]
        tau = stopping_item(drawn)
        bound = anytime_upper_bound(
            drawn[:tau] / EPOCHS, order[:tau], weights, alpha=ALPHA, independent_rollouts=EPOCHS
        )
        stops.append(tau)
        anytime.append(bound.upper)
        fixed.append(float(clopper_pearson_upper(int(drawn[:tau].sum()), tau * EPOCHS)))
        shares = drawn[:tau] / EPOCHS
        items.append(
            weighted_bounded_kl_chernoff_mean_bounds(
                shares,
                shares,
                np.full(tau, 1 / tau),
                alpha_lower=ALPHA,
                alpha_upper=ALPHA,
                independence_profile="prompt_cluster",
            ).upper
        )
        full.append(float(clopper_pearson_upper(int(failures.sum()), ITEMS * EPOCHS)))
    return {
        "theta": target,
        "median_items": float(np.median(stops)),
        "anytime_miss": float(np.mean(np.asarray(anytime) < target)),
        "fixed_at_stop_miss": float(np.mean(np.asarray(fixed) < target)),
        "items_at_stop_miss": float(np.mean(np.asarray(items) < target)),
        "fixed_full_miss": float(np.mean(np.asarray(full) < target)),
        "anytime_median_upper": float(np.median(anytime)),
        "fixed_at_stop_median_upper": float(np.median(fixed)),
        "items_at_stop_median_upper": float(np.median(items)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(
        f"{ITEMS} items x {EPOCHS} rollouts, stop at {CREDIBILITY:.0%} width <= {WIDTH}, "
        f"one-sided alpha {ALPHA}, {RUNS} runs per row"
    )
    header = (
        "theta   items at stop | miss: anytime  CP at stop  KL items at stop  CP full run | "
        "median upper: anytime  CP at stop  KL items at stop"
    )
    print(header)
    for theta in THETAS:
        row = scenario(theta, rng)
        print(
            f"{row['theta']:.4f}  {row['median_items']:>13.0f} | "
            f"{row['anytime_miss']:>13.3f}  {row['fixed_at_stop_miss']:>10.3f}  "
            f"{row['items_at_stop_miss']:>16.3f}  {row['fixed_full_miss']:>11.3f} | "
            f"{row['anytime_median_upper']:>21.4f}  {row['fixed_at_stop_median_upper']:>10.4f}  "
            f"{row['items_at_stop_median_upper']:>16.4f}"
        )


if __name__ == "__main__":
    main()
