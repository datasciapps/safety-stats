"""Create the eight-sample worked log from a completed SimpleQA Inspect log."""

from __future__ import annotations

import argparse
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

PROMPT_IDS = {"9", "13", "17", "7", "5", "14", "11", "18"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Completed full SimpleQA .eval log")
    parser.add_argument("output", type=Path, help="Destination .eval log")
    args = parser.parse_args()

    source = read_eval_log(args.source)
    selected = [sample for sample in source.samples or [] if str(sample.id) in PROMPT_IDS]
    observed = {str(sample.id) for sample in selected}
    if observed != PROMPT_IDS:
        raise ValueError(f"source log is missing IDs: {sorted(PROMPT_IDS - observed)}")

    subset = source.model_copy(
        update={
            "samples": selected,
            "results": None,
            "reductions": None,
            "location": str(args.output),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(subset, args.output, format="eval")
    print(f"Wrote {len(selected)} real samples to {args.output}")


if __name__ == "__main__":
    main()
