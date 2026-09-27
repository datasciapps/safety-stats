"""Freeze a published attack artifact into a fixed-design manifest.

Adversarial prompts are produced by searching against a target model's failures, so they are
not draws from a declared distribution and cannot serve as a rewrite kernel. On the fixed path
that does not matter: the declared set *is* the estimand, and the claim is about this frozen
set rather than about every prompt near an anchor.

The input is a JailbreakBench attack artifact (MIT, github.com/JailbreakBench/artifacts):

    {"parameters": {"method": "PAIR", "model": "vicuna-13b-v1.5",
                    "attack_type": "black_box", "defense": null, ...},
     "jailbreaks": [{"index": 0, "goal": "...", "behavior": "...", "category": "...",
                     "prompt": "...", "response": "...", "jailbroken": true}, ...]}

Two fields are deliberately dropped. ``response`` and ``jailbroken`` are *their* evaluation,
of *their* target, under *their* judge. The point of freezing the prompts is to run them
against a target we choose and bound the failure rate with our own grader.

Every row records the model the attack was optimised against. A set generated against model A
cannot fairly compare A with B, so that column is what keeps a later comparison honest.

Usage:
    python build_design.py \
        --behaviors ../real_jailbreakbench/jbb_behaviors_harmful.csv \
        --artifact PAIR/black_box/vicuna-13b-v1.5.json \
        --artifact GCG/white_box/vicuna-13b-v1.5.json \
        --output-dir run/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DESIGN_COLUMNS = (
    "prompt_id",
    "question_id",
    "task_id",
    "anchor_id",
    "behavior",
    "category_id",
    "perturbation",
    "attack_type",
    "generator_model",
    "artifact",
    "artifact_sha256",
    "goal_drift",
    "expected_rollouts",
    "zero_tolerance",
    "mandatory",
)


@dataclass(frozen=True)
class Artifact:
    """One published attack run: a method, the target it was optimised against, its prompts."""

    path: Path
    method: str
    model: str
    attack_type: str
    defense: str | None
    sha256: str
    items: list[dict[str, Any]]

    @property
    def perturbation(self) -> str:
        return slug(self.method)


def slug(value: str) -> str:
    """Lower-case identifier safe for a manifest column and a node id."""

    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def load_behaviors(path: str | Path) -> dict[int, dict[str, str]]:
    """Read the JBB-Behaviors harmful split, keyed by its published index."""

    behaviors: dict[int, dict[str, str]] = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            missing = [c for c in ("Index", "Goal", "Behavior", "Category") if c not in row]
            if missing:
                raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
            behaviors[int(row["Index"])] = row
    if not behaviors:
        raise ValueError(f"{path} contains no behaviours")
    return behaviors


def load_artifact(path: str | Path) -> Artifact:
    """Read one artifact file and hash it, so a run records which bytes it came from."""

    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    for key in ("parameters", "jailbreaks"):
        if key not in payload:
            raise ValueError(f"{path.name} is not an attack artifact: no {key!r} key")
    parameters = payload["parameters"]
    for key in ("method", "model"):
        if not parameters.get(key):
            raise ValueError(f"{path.name} does not declare parameters.{key}")
    return Artifact(
        path=path,
        method=str(parameters["method"]),
        model=str(parameters["model"]),
        attack_type=str(parameters.get("attack_type") or "unspecified"),
        defense=parameters.get("defense"),
        sha256=hashlib.sha256(raw).hexdigest(),
        items=list(payload["jailbreaks"]),
    )


@dataclass(frozen=True)
class Alignment:
    """Which anchor text to use, and which anchors cannot be trusted.

    Published artifacts are built against a particular revision of the behaviours file. On the
    JailbreakBench artifacts, five of the hundred goals differ from the revision this repository
    pins: the artifact wording is longer and more specific. Attaching those attacked prompts to
    our shorter anchor would mean the unmodified arm asks a different question from the one the
    attack was built from, which is exactly the comparison the design is for.
    """

    goals: dict[int, str]
    drifted: set[int]
    conflicting: set[int]


def align(artifacts: list[Artifact], behaviors: dict[int, dict[str, str]]) -> Alignment:
    """Compare every artifact's goal against the behaviours file, and against each other."""

    seen: dict[int, dict[str, str]] = {}
    for artifact in artifacts:
        for item in artifact.items:
            index = int(item["index"])
            if index not in behaviors:
                raise ValueError(
                    f"{artifact.path.name} item {index} has no matching behaviour; the artifact "
                    "was built against a different version of JBB-Behaviors"
                )
            seen.setdefault(index, {})[artifact.path.name] = str(item.get("goal", "")).strip()

    goals = {index: behaviors[index]["Goal"].strip() for index in behaviors}
    drifted: set[int] = set()
    conflicting: set[int] = set()
    for index, per_artifact in seen.items():
        texts = set(per_artifact.values())
        if len(texts) > 1:
            conflicting.add(index)
            continue
        text = texts.pop()
        if text != goals[index]:
            drifted.add(index)
            goals[index] = text
    return Alignment(goals=goals, drifted=drifted, conflicting=conflicting)


def anchor_rows(
    behaviors: dict[int, dict[str, str]],
    goals: dict[int, str],
    *,
    task_id: str,
    rollouts: int,
    drifted: set[int],
    excluded: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """The unmodified behaviours, so the attacked arm has something to be compared against."""

    rows, prompts = [], []
    for index in sorted(behaviors):
        if index in excluded:
            continue
        behavior = behaviors[index]
        prompt_id = f"anchor-{index:03d}"
        rows.append({
            "prompt_id": prompt_id,
            "question_id": f"q-{index:03d}",
            "task_id": task_id,
            "anchor_id": f"q-{index:03d}",
            "behavior": slug(behavior["Behavior"]),
            "category_id": slug(behavior["Category"]),
            "perturbation": "none",
            "attack_type": "none",
            "generator_model": "none",
            "artifact": "jbb-behaviors",
            "artifact_sha256": "",
            "goal_drift": "true" if index in drifted else "false",
            "expected_rollouts": rollouts,
            "zero_tolerance": "false",
            "mandatory": "true",
        })
        prompts.append({"id": prompt_id, "input": goals[index], "question_id": f"q-{index:03d}",
                        "perturbation": "none"})
    return rows, prompts


def attack_rows(
    artifact: Artifact,
    behaviors: dict[int, dict[str, str]],
    *,
    task_id: str,
    rollouts: int,
    drifted: set[int],
    excluded: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    """Rows for one artifact, skipping anchors whose text could not be aligned."""

    rows, prompts, skipped = [], [], []
    for item in artifact.items:
        index = int(item["index"])
        if index in excluded:
            continue
        text = item.get("prompt")
        if not text or not str(text).strip():
            skipped.append(f"{index:03d} (no prompt recorded)")
            continue
        prompt_id = f"{artifact.perturbation}-{slug(artifact.model)}-{index:03d}"
        rows.append({
            "prompt_id": prompt_id,
            "question_id": f"q-{index:03d}",
            "task_id": task_id,
            "anchor_id": f"q-{index:03d}",
            "behavior": slug(behaviors[index]["Behavior"]),
            "category_id": slug(behaviors[index]["Category"]),
            "perturbation": artifact.perturbation,
            "attack_type": slug(artifact.attack_type),
            "generator_model": slug(artifact.model),
            "artifact": artifact.path.name,
            "artifact_sha256": artifact.sha256,
            "goal_drift": "true" if index in drifted else "false",
            "expected_rollouts": rollouts,
            "zero_tolerance": "false",
            "mandatory": "true",
        })
        prompts.append({"id": prompt_id, "input": str(text), "question_id": f"q-{index:03d}",
                        "perturbation": artifact.perturbation})
    return rows, prompts, skipped


def build(
    artifacts: list[Artifact],
    behaviors: dict[int, dict[str, str]],
    *,
    task_id: str = "jbb-harmful-compliance",
    rollouts: int = 3,
    include_anchors: bool = True,
    allow_goal_drift: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Build the manifest rows and their prompt text, with a report of what was left out.

    An anchor whose wording drifted between revisions is dropped by default: keeping it would
    compare an attacked prompt against a differently worded original. ``allow_goal_drift`` keeps
    those anchors and uses the artifact's wording for the unmodified arm, so both arms ask the
    same question; the rows are then marked ``goal_drift``.
    """

    alignment = align(artifacts, behaviors)
    excluded = set(alignment.conflicting)
    if not allow_goal_drift:
        excluded |= alignment.drifted

    rows: list[dict[str, Any]] = []
    prompts: list[dict[str, str]] = []
    skipped: dict[str, list[str]] = {}
    if include_anchors:
        rows, prompts = anchor_rows(
            behaviors, alignment.goals, task_id=task_id, rollouts=rollouts,
            drifted=alignment.drifted, excluded=excluded,
        )
    for artifact in artifacts:
        extra_rows, extra_prompts, missing = attack_rows(
            artifact, behaviors, task_id=task_id, rollouts=rollouts,
            drifted=alignment.drifted, excluded=excluded,
        )
        rows.extend(extra_rows)
        prompts.extend(extra_prompts)
        skipped[artifact.path.name] = missing
    seen = {row["prompt_id"] for row in rows}
    if len(seen) != len(rows):
        raise ValueError("two artifacts produced the same prompt_id; pass each artifact once")
    report = {
        "skipped_without_prompt": skipped,
        "drifted": sorted(alignment.drifted),
        "conflicting": sorted(alignment.conflicting),
        "excluded": sorted(excluded),
    }
    return rows, prompts, report


def write_design(rows: list[dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(DESIGN_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def write_prompts(prompts: list[dict[str, str]], path: str | Path) -> None:
    """Prompt text lives apart from the manifest, so reports can stay free of it."""

    with Path(path).open("w", encoding="utf-8") as stream:
        for prompt in prompts:
            stream.write(json.dumps(prompt, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--behaviors", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path, action="append", dest="artifacts")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-id", default="jbb-harmful-compliance")
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--no-anchors", action="store_true",
                        help="omit the unmodified behaviours, leaving only attacked prompts")
    parser.add_argument("--allow-goal-drift", action="store_true",
                        help="keep anchors whose wording changed between revisions, using the "
                             "artifact's wording for the unmodified arm")
    args = parser.parse_args()

    behaviors = load_behaviors(args.behaviors)
    artifacts = [load_artifact(p) for p in args.artifacts]
    rows, prompts, report = build(
        artifacts, behaviors, task_id=args.task_id, rollouts=args.rollouts,
        include_anchors=not args.no_anchors, allow_goal_drift=args.allow_goal_drift,
    )
    skipped = report["skipped_without_prompt"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_design(rows, args.output_dir / "design.csv")
    write_prompts(prompts, args.output_dir / "prompts.jsonl")

    print(f"behaviours: {len(behaviors)}")
    for artifact in artifacts:
        missing = skipped.get(artifact.path.name, [])
        kept = sum(1 for row in rows if row["artifact"] == artifact.path.name)
        print(f"  {artifact.method:>6} vs {artifact.model:<22} {kept:3} prompts, "
              f"{len(missing):3} skipped, sha256 {artifact.sha256[:12]}")
    if report["drifted"]:
        kept = "kept, using the artifact wording" if args.allow_goal_drift else "dropped"
        print(f"  goal drift vs the behaviours file at {report['drifted']}: {kept}")
    if report["conflicting"]:
        print(f"  artifacts disagree with each other at {report['conflicting']}: dropped")
    print(f"rows: {len(rows)} over {len({row['question_id'] for row in rows})} anchors")
    print(f"wrote {args.output_dir / 'design.csv'} and {args.output_dir / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
