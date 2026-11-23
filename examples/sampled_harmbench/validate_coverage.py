"""Check that a sampled design recovers the answer the census already established.

Repeatedly redraws from the completed census and asks whether the design-based interval
covers the census weighted mean. This is the one check a simulation cannot supply: the
prompt risks are the real ones this model and judge produced, not draws from an assumed
distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from safety_stats.stats.design_based import (
    FixedSizeDesign,
    InclusionDesign,
    fixed_size_mean_bounds,
    horvitz_thompson_mean_bounds,
)

FRAME = Path(__file__).with_name("frame.csv")
GROUP = "all-behaviors"


def _census(report_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    design = [
        row
        for row in csv.DictReader((report_dir / "design.resolved.csv").open())
        if row["group_id"] == GROUP
    ]
    cohort = report["cohorts"][0]["cohort_id"]
    estimates = {
        item["prompt_id"]: item
        for item in report["prompt_estimates"]
        if item["cohort_id"] == cohort
    }
    prompt_ids = [row["prompt_id"] for row in design]
    weights = np.array([float(row["weight"]) for row in design])
    rates = np.array(
        [
            estimates[prompt_id]["n_failure"] / estimates[prompt_id]["n_total"]
            for prompt_id in prompt_ids
        ]
    )
    return prompt_ids, weights, rates, float(weights @ rates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census_report", type=Path, help="Completed census report directory")
    parser.add_argument("--replications", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--method", default="weighted_kl_chernoff")
    args = parser.parse_args()

    prompt_ids, weights, rates, census = _census(args.census_report)
    frame = {row["prompt_id"]: row for row in csv.DictReader(FRAME.open())}
    size = len(prompt_ids)
    rng = np.random.default_rng(20260824)
    endpoint = args.alpha / 2

    print(f"Frame: {size} items. Census domain-balanced mean = {census:.6f}\n")

    probabilities = np.array(
        [float(frame[p]["inclusion_probability"]) for p in prompt_ids]
    )
    covered, widths = 0, []
    for _ in range(args.replications):
        mask = rng.random(size) < probabilities
        if not mask.any():
            continue
        observed = np.where(mask, rates, 0.0)
        bounds = horvitz_thompson_mean_bounds(
            InclusionDesign(weights, probabilities, mask),
            observed,
            observed,
            alpha_lower=endpoint,
            alpha_upper=endpoint,
            method=args.method,
        )
        covered += bounds.lower <= census <= bounds.upper
        widths.append(bounds.upper - bounds.lower)
    print(
        f"{'Poisson':<26} coverage {covered / len(widths):.3f}   "
        f"mean width {np.mean(widths):.4f}   inflation {bounds.inflation:.2f}"
    )

    groups: dict[str, list[int]] = defaultdict(list)
    for index, prompt_id in enumerate(prompt_ids):
        groups[frame[prompt_id]["risk_domain"]].append(index)
    labels = sorted(groups)
    strata = np.zeros(size, dtype=int)
    draws = np.zeros(len(labels), dtype=int)
    for position, label in enumerate(labels):
        members = groups[label]
        strata[members] = position
        draws[position] = round(
            float(frame[prompt_ids[members[0]]]["stratified_inclusion_probability"])
            * len(members)
        )
    covered, widths = 0, []
    for _ in range(args.replications):
        mask = np.zeros(size, dtype=bool)
        for position, label in enumerate(labels):
            mask[rng.choice(groups[label], draws[position], replace=False)] = True
        observed = np.where(mask, rates, 0.0)
        bounds = fixed_size_mean_bounds(
            FixedSizeDesign(strata, weights, draws, mask),
            observed,
            observed,
            alpha_lower=endpoint,
            alpha_upper=endpoint,
            method=args.method,
        )
        covered += bounds.lower <= census <= bounds.upper
        widths.append(bounds.upper - bounds.lower)
    print(
        f"{'Stratified fixed-size':<26} coverage {covered / args.replications:.3f}   "
        f"mean width {np.mean(widths):.4f}   inflation {bounds.inflation:.2f}"
    )
    print(f"{'':<26} draws per domain {dict(zip(labels, draws.tolist(), strict=True))}")

    print("\nWhat a smaller budget buys (stratified, scaled proportionally):")
    print(f"{'budget':>8} {'drawn':>7} {'coverage':>9} {'mean width':>11}")
    for fraction in (1.0, 0.5, 0.33, 0.25, 0.17):
        scaled = np.maximum(2, np.round(np.bincount(strata) * fraction).astype(int))
        covered, widths = 0, []
        for _ in range(args.replications):
            mask = np.zeros(size, dtype=bool)
            for position, label in enumerate(labels):
                mask[rng.choice(groups[label], scaled[position], replace=False)] = True
            observed = np.where(mask, rates, 0.0)
            bounds = fixed_size_mean_bounds(
                FixedSizeDesign(strata, weights, scaled, mask),
                observed,
                observed,
                alpha_lower=endpoint,
                alpha_upper=endpoint,
                method=args.method,
            )
            covered += bounds.lower <= census <= bounds.upper
            widths.append(bounds.upper - bounds.lower)
        print(
            f"{fraction:>8.0%} {int(scaled.sum()):>7} {covered / args.replications:>9.3f} "
            f"{np.mean(widths):>11.4f}"
        )


if __name__ == "__main__":
    main()
