"""Freeze a probability sample of the HarmBench frame and subset a completed log to it.

The draw uses only the frame and a declared seed -- never an outcome. That is what makes
the retrospective check meaningful: the sample is chosen exactly as it would have been
before the evaluation ran, then the already-recorded outcomes for those items are kept.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

FRAME = Path(__file__).with_name("frame.csv")


def draw(seed: int, *, stratified: bool) -> set[str]:
    """Return the drawn prompt identifiers for one frozen sample."""
    rows = list(csv.DictReader(FRAME.open()))
    rng = random.Random(seed)
    if not stratified:
        column = "inclusion_probability"
        return {
            row["prompt_id"] for row in rows if rng.random() < float(row[column])
        }
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[row["risk_domain"]].append(row)
    drawn: set[str] = set()
    for stratum in sorted(strata):
        members = strata[stratum]
        probability = float(members[0]["stratified_inclusion_probability"])
        count = round(probability * len(members))
        drawn.update(row["prompt_id"] for row in rng.sample(members, count))
    return drawn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census_log", type=Path, help="Completed 400-behaviour .eval log")
    parser.add_argument("output_log", type=Path, help="Destination subset .eval log")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Draw a fixed count per risk domain instead of independent inclusion",
    )
    args = parser.parse_args()

    selected = draw(args.seed, stratified=args.stratified)
    source = read_eval_log(args.census_log)
    samples = [
        sample for sample in source.samples or [] if str(sample.id) in selected
    ]
    observed = {str(sample.id) for sample in samples}
    missing = sorted(selected - observed)
    if missing:
        raise ValueError(
            f"census log is missing {len(missing)} drawn frame items, so the realized "
            f"sample would not match the declared design: {missing[:5]}"
        )

    subset = source.model_copy(
        update={
            "samples": samples,
            "results": None,
            "reductions": None,
            "location": str(args.output_log),
        }
    )
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(subset, args.output_log, format="eval")
    design = "stratified fixed-size" if args.stratified else "Poisson"
    print(f"{design} draw, seed {args.seed}: {len(selected)} of 300 frame items")
    print(f"Wrote {len(samples)} sample records to {args.output_log}")


if __name__ == "__main__":
    main()
