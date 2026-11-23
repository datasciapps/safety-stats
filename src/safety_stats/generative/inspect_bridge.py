"""Materialize an Inspect dataset from accepted rewrites.

The dataset written here is the restricted materialization artifact: it necessarily
carries prompt text, because that is what Inspect evaluates. Nothing downstream of the
join does -- outcomes are keyed by identifier, and the statistical ledgers, reports and
CSV outputs carry hashes only.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_safety_stats.generative.execution import AcceptedRewrite, ExecutionResult
from inspect_safety_stats.generative.join import (
    EvaluatedSample,
    JoinResult,
    join_outcomes,
    samples_from_observations,
)
from inspect_safety_stats.generative.ledger import SelectionRecord
from inspect_safety_stats.generative.schema import GenerativePlan

__all__ = [
    "EvaluatedSample",
    "JoinResult",
    "join_outcomes",
    "materialize_dataset",
    "materialize_for_cohorts",
    "samples_from_observations",
]


def materialize_dataset(
    selections: list[SelectionRecord],
    result: ExecutionResult,
    path: str | Path,
) -> int:
    """Write accepted rewrites as an Inspect JSONL dataset. Returns the sample count.

    A single-turn rewrite becomes a string ``input``; a conversation becomes a list of
    ``{"role", "content"}`` messages, which Inspect reads as chat messages, so the target
    answers the conversation's final user turn.
    """

    by_unit = {record.sampling_unit_id: record for record in selections}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in result.accepted:
            selection = by_unit.get(entry.sampling_unit_id)
            if selection is None:
                raise ValueError(
                    f"accepted rewrite {entry.sampling_unit_id} has no matching selection"
                )
            handle.write(
                json.dumps(
                    {
                        "id": entry.prompt_id,
                        "input": _dataset_input(entry),
                        "metadata": {
                            "sampling_unit_id": entry.sampling_unit_id,
                            "path_hash": selection.path_hash,
                            "node_id": selection.node_id,
                            "semantic_seed_id": selection.semantic_seed_id,
                            "stratum": selection.stratum,
                            "target_path_probability": selection.target_path_probability,
                            "prompt_sha256": entry.prompt_sha256,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            written += 1
    return written


def _dataset_input(entry: AcceptedRewrite) -> str | list[dict[str, str]]:
    if entry.messages is None:
        return entry.prompt
    return [{"role": turn.role, "content": turn.content} for turn in entry.messages]


def materialize_for_cohorts(
    plan: GenerativePlan,
    selections: list[SelectionRecord],
    result: ExecutionResult,
    directory: str | Path,
) -> dict[str, Path]:
    """One dataset per cohort.

    Under ``target.shared_rewrites`` every cohort receives the identical file, which is
    what makes the arms paired: the same accepted rewrite is evaluated by each model, so
    a difference between them cannot be an artefact of different prompts.
    """

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if not plan.target.shared_rewrites:
        raise ValueError(
            "independent rewrites per cohort require executing the kernel once per "
            "cohort with a distinct rewrite_salt; pass each result to "
            "materialize_dataset directly"
        )
    shared = root / "dataset.jsonl"
    materialize_dataset(selections, result, shared)
    return dict.fromkeys(plan.target.cohorts, shared)
