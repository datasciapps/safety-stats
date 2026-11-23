"""Where does a sampled design's uncertainty come from, and what would reduce it?

Decomposes the estimator's variance into the part caused by which items were drawn and
the part caused by rollout noise inside them, then shows what interval width a given
number of drawn items actually buys.

The decomposition needs per-item risks, so run it against a completed census or a pilot.
Used prospectively it answers the only question worth asking before spending budget:
more items, more rollouts, or better strata?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from safety_stats.stats.design_based import FixedSizeDesign, fixed_size_mean_bounds


def _load(report_dir: Path, group: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    design = [
        row
        for row in csv.DictReader((report_dir / "design.resolved.csv").open())
        if row["group_id"] == group
    ]
    if not design:
        raise SystemExit(f"group {group!r} is absent from {report_dir}")
    cohort = report["cohorts"][0]["cohort_id"]
    estimates = {
        item["prompt_id"]: item
        for item in report["prompt_estimates"]
        if item["cohort_id"] == cohort
    }
    prompt_ids = [row["prompt_id"] for row in design]
    weights = np.array([float(row["weight"]) for row in design])
    risks = np.array(
        [estimates[p]["n_failure"] / estimates[p]["n_total"] for p in prompt_ids]
    )
    rollouts = np.array([estimates[p]["n_total"] for p in prompt_ids], dtype=float)
    return weights, risks, rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Completed census or pilot report directory")
    parser.add_argument("--group", default="all-behaviors")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--replications", type=int, default=200)
    args = parser.parse_args()

    weights, risks, rollouts = _load(args.report, args.group)
    size = len(weights)
    census = float(weights @ risks)
    census_sd = float(np.sqrt(np.sum(weights**2 * risks * (1 - risks) / rollouts)))
    print(f"Frame: {size} items, weighted mean {census:.5f}")
    print(
        f"Risk spread: {int((risks == 0).sum())} zero-risk items, "
        f"max {risks.max():.3f}, census standard error {census_sd:.5f}\n"
    )

    print("Variance decomposition at a range of designs")
    print(f"{'pi':>6} {'rollouts':>9} {'sd':>9} {'item sampling':>14} {'rollout noise':>14}")
    for probability in (0.10, 0.25, 0.50):
        for per_item in (1, int(np.median(rollouts))):
            pi = np.full(size, probability)
            coefficient = weights / pi
            sampling = float(np.sum(coefficient**2 * pi * (1 - pi) * risks**2))
            rollout = float(np.sum(coefficient**2 * pi * risks * (1 - risks) / per_item))
            total = sampling + rollout
            print(
                f"{probability:>6.2f} {per_item:>9} {np.sqrt(total):>9.5f} "
                f"{sampling / total:>13.1%} {rollout / total:>13.1%}"
            )
    print(
        "\nRollout noise is the only component a two-stage decomposition could remove, and\n"
        "width scales as the square root of variance: removing a 2% share buys 1%.\n"
    )

    print("What drawing n items actually buys (stratum-free draw, finite-sample width)")
    print(f"{'n drawn':>8} {'budget':>8} {'mean width':>11} {'covers census':>14}")
    rng = np.random.default_rng(20260824)
    endpoint = args.alpha / 2
    for fraction in (1.0, 0.5, 0.25, 0.125):
        drawn = max(2, int(round(size * fraction)))
        widths, covered = [], 0
        for _ in range(args.replications):
            mask = np.zeros(size, dtype=bool)
            mask[rng.choice(size, drawn, replace=False)] = True
            observed = np.where(mask, risks, 0.0)
            bounds = fixed_size_mean_bounds(
                FixedSizeDesign(np.zeros(size, dtype=int), weights, np.array([drawn]), mask),
                observed,
                observed,
                alpha_lower=endpoint,
                alpha_upper=endpoint,
                method="weighted_kl_chernoff",
            )
            widths.append(bounds.upper - bounds.lower)
            covered += bounds.lower <= census <= bounds.upper
        print(
            f"{drawn:>8} {fraction:>8.0%} {np.mean(widths):>11.4f} "
            f"{covered / args.replications:>14.3f}"
        )

    print(
        "\nSpend guidance. Item-sampling variance dominates unless the frame is nearly a\n"
        "census, so precision comes from drawing more items -- never from more rollouts per\n"
        "item, which this estimator does not read. The other lever is stratifying on a\n"
        "variable that predicts risk, which reduces the between-item spread the sampling\n"
        "term is built from."
    )


if __name__ == "__main__":
    main()
