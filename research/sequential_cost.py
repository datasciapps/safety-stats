"""What looking after every draw costs, for a named sequence against fixed designs.

The design for anytime-valid confidence sequences (docs/sequential-design.md, section "Cost")
compares a confidence sequence with the fixed-horizon analysis. This script measures that
comparison for two sequences, on draws with replacement of a Bernoulli failure outcome, each
a one-sided upper bound at alpha = 0.05 against a pass threshold of 0.05, with at most 1,200
draws:

- **the betting mixture**: against each candidate mean ``m``, the capital
  ``prod_s (1 - f + f (1 - X_s) / (1 - m))``, mixed over the fixed bet fractions ``f`` and
  weights of the betting bounds (``_betting_prior``, #3), with the running intersection over
  draws. It is the with-replacement form of the sequence of step 1 of #126;
- **the Hoeffding-spending reference**: the Hoeffding bound at ``alpha / (t (t + 1))`` after
  draw ``t``, with the running intersection.

A run passes at the first draw whose upper bound is at or below the threshold. The fixed
designs are one-sided Clopper-Pearson at a number of draws fixed in advance.

Run:
    python research/sequential_cost.py
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta, binom

from inspect_safety_stats.stats.direct_mean import _betting_prior

ALPHA = 0.05
THRESHOLD = 0.05
HORIZON = 1200
PASS_RATES = (0.005, 0.03, 0.05)
PASS_RUNS = 20_000
WIDTH_SIZES = (100, 300, 1200)
WIDTH_RATES = (0.005, 0.03, 0.05)
WIDTH_RUNS = 1000
CHUNK = 500
BISECTION_STEPS = 50
SEED = 20261119

FRACTIONS, LOG_PRIOR = _betting_prior()
LOG_LEVEL = math.log(1 / ALPHA)


def betting_log_capital(failures: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Log mixture capital after each draw, for failure counts ``failures`` (runs x draws).

    ``mean`` holds one candidate mean per run. A safe draw multiplies the capital at fraction
    ``f`` by ``1 + f m / (1 - m)`` and a failure by ``1 - f``.
    """

    draws = np.arange(1, failures.shape[1] + 1)
    safe = draws[None, :] - failures
    up = np.log1p(FRACTIONS[None, :] * (mean / (1 - mean))[:, None])  # runs x fractions
    down = np.log1p(-FRACTIONS)
    terms = (
        LOG_PRIOR[None, None, :]
        + safe[:, :, None] * up[:, None, :]
        + failures[:, :, None] * down[None, None, :]
    )
    peak = terms.max(axis=2)
    return peak + np.log(np.exp(terms - peak[:, :, None]).sum(axis=2))


def betting_pass_draw(failures: np.ndarray) -> np.ndarray:
    """The first draw at which the betting mixture passes, or 0 if it never does."""

    capital = betting_log_capital(failures, np.full(len(failures), THRESHOLD))
    passed = capital >= LOG_LEVEL
    return np.where(passed.any(axis=1), passed.argmax(axis=1) + 1, 0)


def hoeffding_upper(failures: np.ndarray) -> np.ndarray:
    """The Hoeffding-spending upper bound after each draw, as a running minimum."""

    draws = np.arange(1, failures.shape[1] + 1)
    bound = failures / draws + np.sqrt(np.log(draws * (draws + 1) / ALPHA) / (2 * draws))
    return np.minimum.accumulate(np.minimum(bound, 1.0), axis=1)


def betting_upper(failures: np.ndarray) -> np.ndarray:
    """The betting mixture's upper bound after the last draw: the running intersection."""

    inside = np.zeros(len(failures))
    outside = np.ones(len(failures))
    for _ in range(BISECTION_STEPS):
        middle = (inside + outside) / 2
        rejected = betting_log_capital(failures, middle).max(axis=1) >= LOG_LEVEL
        outside = np.where(rejected, middle, outside)
        inside = np.where(rejected, inside, middle)
    return outside


def clopper_pearson_upper(failures: np.ndarray, draws: int) -> np.ndarray:
    upper = beta.ppf(1 - ALPHA, failures + 1, np.maximum(draws - failures, 1))
    return np.where(failures >= draws, 1.0, upper)


def fixed_pass_probability(draws: int, rate: float) -> float:
    """How often a fixed design of ``draws`` passes at ``rate``: its bound at the threshold."""

    counts = np.arange(draws + 1)
    passing = counts[clopper_pearson_upper(counts, draws) <= THRESHOLD]
    return float(binom.cdf(passing.max(), draws, rate)) if len(passing) else 0.0


def smallest_fixed_design(rate: float, probability: float) -> int:
    """The fewest draws fixed in advance that pass at ``rate`` at least ``probability``."""

    for draws in range(1, 20 * HORIZON):
        if fixed_pass_probability(draws, rate) >= probability:
            return draws
    raise ValueError("no fixed design reaches that pass probability")


def failure_paths(rng: np.random.Generator, runs: int, draws: int, rate: float) -> np.ndarray:
    return np.cumsum(rng.random((runs, draws)) < rate, axis=1)


def passing(rng: np.random.Generator) -> None:
    print(f"\nPassing, {PASS_RUNS} runs per rate, at most {HORIZON} draws")
    zero_failure = smallest_fixed_design(0.0, 1.0)
    for rate in PASS_RATES:
        stops, hoeffding = [], 0
        for _ in range(PASS_RUNS // CHUNK):
            failures = failure_paths(rng, CHUNK, HORIZON, rate)
            stops.append(betting_pass_draw(failures))
            hoeffding += int((hoeffding_upper(failures)[:, -1] <= THRESHOLD).sum())
        stop = np.concatenate(stops)
        passed = stop > 0
        share = float(passed.mean())
        mean_draws = float(np.where(passed, stop, HORIZON).mean())
        median = float(np.median(stop[passed])) if passed.any() else math.nan
        print(f"true rate {rate}:")
        print(
            f"  betting mixture: passes {share:.3f} (s.e. "
            f"{math.sqrt(share * (1 - share) / PASS_RUNS):.4f}); median draws among passes "
            f"{median:.0f}; mean draws to pass or horizon {mean_draws:.0f}"
        )
        print(f"  Hoeffding-spending reference: passes {hoeffding / PASS_RUNS:.3f}")
        print(
            f"  fixed design of {HORIZON}: passes {fixed_pass_probability(HORIZON, rate):.3f}; "
            f"zero-failure design of {zero_failure}: passes "
            f"{fixed_pass_probability(zero_failure, rate):.3f}"
        )
        if rate < THRESHOLD:
            targets = sorted({0.5, 0.9, 0.99, *([round(share, 3)] if share < 0.99 else [])})
            fewest = ", ".join(f"{t}: {smallest_fixed_design(rate, t)}" for t in targets)
            print(f"  fewest fixed draws that pass with probability at least {fewest}")


def widths(rng: np.random.Generator) -> None:
    print(
        f"\nUpper bound minus estimate at a fixed number of draws, median ratio to "
        f"Clopper-Pearson's, {WIDTH_RUNS} runs per cell"
    )
    print("draws  rate    betting mixture  Hoeffding-spending")
    for draws in WIDTH_SIZES:
        for rate in WIDTH_RATES:
            betting_ratio, hoeffding_ratio = [], []
            for _ in range(WIDTH_RUNS // CHUNK):
                failures = failure_paths(rng, CHUNK, draws, rate)
                estimate = failures[:, -1] / draws
                exact = clopper_pearson_upper(failures[:, -1], draws) - estimate
                betting_ratio.append((betting_upper(failures) - estimate) / exact)
                hoeffding_ratio.append((hoeffding_upper(failures)[:, -1] - estimate) / exact)
            print(
                f"{draws:>5}  {rate:<6} {np.median(np.concatenate(betting_ratio)):>15.2f}  "
                f"{np.median(np.concatenate(hoeffding_ratio)):>18.2f}"
            )


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"one-sided alpha {ALPHA}, pass threshold {THRESHOLD}, draws with replacement")
    passing(rng)
    widths(rng)


if __name__ == "__main__":
    main()
