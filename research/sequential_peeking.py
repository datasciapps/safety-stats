"""How often re-checking a fixed-sample bound after every draw passes an unsafe system.

The design for anytime-valid confidence sequences (docs/sequential-design.md) asks for a
peeking demonstration: an ordinary one-sided Clopper-Pearson upper bound, recomputed after
every rollout, with the run stopped as soon as the bound falls to the pass threshold. Each
look is valid on its own; taken together they are not.

The true failure rate here equals the threshold, so every pass is a false pass, and a single
look at the horizon would pass at most alpha of the time.

Run:
    python research/sequential_peeking.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta

ALPHA = 0.05
THRESHOLD = 0.05
TRUE_RATE = 0.05
HORIZON = 1200
RUNS = 200_000
CHUNK = 10_000
SEED = 20261116


def passing_counts(horizon: int) -> np.ndarray:
    """The largest failure count at each n whose upper bound passes, or -1 if none does."""

    limits = np.full(horizon, -1)
    for index in range(horizon):
        n = index + 1
        failures = 0
        # The upper bound rises with the failure count, so the passing counts are a prefix.
        while failures < n and beta.ppf(1 - ALPHA, failures + 1, n - failures) <= THRESHOLD:
            failures += 1
        limits[index] = failures - 1
    return limits


def main() -> None:
    limits = passing_counts(HORIZON)
    rng = np.random.default_rng(SEED)
    passed_anytime = 0
    passed_at_horizon = 0
    for _ in range(RUNS // CHUNK):
        failures = np.cumsum(rng.random((CHUNK, HORIZON)) < TRUE_RATE, axis=1)
        passed_anytime += int(np.any(failures <= limits, axis=1).sum())
        passed_at_horizon += int((failures[:, -1] <= limits[-1]).sum())
    print(f"true rate {TRUE_RATE}, threshold {THRESHOLD}, one-sided alpha {ALPHA}, {RUNS} runs")
    print(f"single look at n = {HORIZON}: passes {passed_at_horizon / RUNS:.3f}")
    print(f"look after every draw up to n = {HORIZON}: passes {passed_anytime / RUNS:.3f}")


if __name__ == "__main__":
    main()
