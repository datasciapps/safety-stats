"""Read the generated prompts, grouped by seed, and flag what a glance would miss.

Unless the plan sets rewrite.validation.invoke, nothing in the package validates that a
rewrite honoured its declared arms -- the kernel's own verdict is trusted. Reading the
prompts is then the only check, and this makes that quick.

Two things it flags that are easy to miss by eye. A rewrite that *names* its framing rather
than enacting it ("direct_request:", "Rollenspiel:") means the arm was echoed, not applied,
and the hierarchy is then claiming structure the prompts do not carry. And near-duplicates
within a seed slip past deduplication, which by default compares exact hashes
(rewrite.deduplication.near_duplicate: normalized_text catches only differences of case,
punctuation, symbols, spacing and invisible characters).

Run:
    python examples/generative_live/review_prompts.py run/live
    python examples/generative_live/review_prompts.py run/live --seed ba-02
"""

from __future__ import annotations

import argparse
import csv
import difflib
from pathlib import Path

LABELS = (
    "direct_request",
    "roleplay",
    "rollenspiel",
    "nested_task",
    "wrapper:",
    "style:",
    "task:",
)
NEAR_DUPLICATE = 0.85


def path_values(node_id: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in node_id.split("|")[1:] if "=" in part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", help="a materialization output directory")
    parser.add_argument("--seed", default=None, help="show only this semantic seed")
    arguments = parser.parse_args()

    root = Path(arguments.artifacts)
    selections = {
        row["sampling_unit_id"]: row
        for row in csv.DictReader((root / "selections.csv").open(encoding="utf-8"))
    }
    prompts = {
        row["sampling_unit_id"]: row["prompt"]
        for row in csv.DictReader((root / "prompts.restricted.csv").open(encoding="utf-8"))
    }
    attempts = {
        row["sampling_unit_id"]: row["rewrite_attempts"]
        for row in csv.DictReader((root / "rewrites.csv").open(encoding="utf-8"))
    }

    by_seed: dict[str, list[str]] = {}
    for unit_id in prompts:
        by_seed.setdefault(selections[unit_id]["semantic_seed_id"], []).append(unit_id)

    flagged_labels: list[str] = []
    flagged_duplicates: list[str] = []

    for seed_id in sorted(by_seed):
        if arguments.seed and seed_id != arguments.seed:
            continue
        units = sorted(
            by_seed[seed_id],
            key=lambda u: tuple(path_values(selections[u]["node_id"]).get(k, "") for k in
                                ("language", "style", "wrapper")),
        )
        print(f"\n=== {seed_id}  ({len(units)} rewrites) ===")
        for unit_id in units:
            values = path_values(selections[unit_id]["node_id"])
            arms = " ".join(
                f"{values.get(k, '?'):<14}" for k in ("language", "style", "wrapper")
            )
            text = prompts[unit_id]
            named = [label for label in LABELS if label in text.lower()]
            marker = "  <- names its framing" if named else ""
            print(f"  [{arms}] attempts={attempts[unit_id]}{marker}")
            print(f"    {text}")
            if named:
                flagged_labels.append(unit_id)

        for i, left in enumerate(units):
            for right in units[i + 1 :]:
                ratio = difflib.SequenceMatcher(
                    None, prompts[left], prompts[right]
                ).ratio()
                if ratio >= NEAR_DUPLICATE:
                    flagged_duplicates.append(f"{seed_id}: {left} ~ {right} ({ratio:.2f})")

    print("\n--- review ---")
    print(f"  rewrites read            {sum(len(v) for v in by_seed.values())}")
    print(f"  naming their framing     {len(flagged_labels)}")
    for entry in flagged_duplicates:
        print(f"  near-duplicate           {entry}")
    if not flagged_labels and not flagged_duplicates:
        print("  nothing flagged; read them anyway -- only you can judge equivalence")


if __name__ == "__main__":
    main()
