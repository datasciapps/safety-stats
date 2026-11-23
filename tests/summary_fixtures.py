"""Small analyses, written as rollout files, whose report.md is checked against report.json.

Each variant is a fixed design of prompts in two categories, analysed with a schema 3.0 plan
from the neutral rollout files, so no Inspect log and no monkeypatching is needed. The
variants cover what the summary's claims block depends on: a confirmatory and an exploratory
analysis, invalid integrity, a zero-tolerance failure, an adaptive-attacker claim and an
assumed judge calibration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from safety_stats import analyze
from safety_stats.ingest.rollout_files import RolloutFiles
from safety_stats.schema.reports import SafetyReport

COHORTS = {"baseline": "model/baseline", "candidate": "model/candidate"}


@dataclass(frozen=True)
class Variant:
    """One analysis: the plan's changes from ``_plan`` and the outcomes of its rollouts."""

    name: str
    prompts: int = 20
    rollouts: int = 30
    # Failing rollouts per prompt, by cohort: a prompt's index modulo the list's length picks
    # the count, so a short list repeats over the prompts.
    failures: dict[str, list[int]] = field(
        default_factory=lambda: {"baseline": [12, 3, 0, 1], "candidate": [2, 0, 0, 0]}
    )
    intent: str = "confirmatory"
    # Plan entries replaced or added at the top level, and under inputs.
    plan: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    # Risk measures declared after the weighted mean.
    measures: tuple[dict[str, Any], ...] = ()
    # Bounds on each category and on its difference, which spend alpha of their own.
    hierarchy_nodes: bool = True
    # The independence of the direct means. The category profile comes before the prompt
    # profile, so that the summary must pick the prompt-level one by its blocks and not by
    # its place.
    independence: dict[str, Any] = field(
        default_factory=lambda: {
            "primary": "rollout",
            "sensitivities": ["by_category", "by_prompt"],
        }
    )
    # Comparisons declared after the risk difference and ratio of candidate and baseline. A
    # family one names that the plan does not declare is added, with Bonferroni at 0.05.
    comparisons: tuple[dict[str, Any], ...] = ()
    # Extra manifest columns, by name, as a function of the prompt index.
    columns: dict[str, list[Any]] = field(default_factory=dict)
    # Rollouts left out of the files, as (cohort, prompt index, repeat).
    missing: tuple[tuple[str, int, int], ...] = ()


def claim_spec(
    cohort: str,
    *,
    acceptable_upper: float,
    unacceptable_lower: float,
    mandatory: bool = True,
) -> dict[str, Any]:
    """A claim on ``cohort``'s weighted mean over all prompts, in the gate family."""

    return {
        "id": f"{cohort}_acceptable",
        "title": f"{cohort.capitalize()} risk is acceptable",
        "cohort": cohort,
        "group": "all",
        "measure": "mean",
        "family": "gate",
        "acceptable_upper": acceptable_upper,
        "unacceptable_lower": unacceptable_lower,
        "mandatory": mandatory,
    }


def _nodes(variant: Variant) -> dict[str, Any]:
    if not variant.hierarchy_nodes:
        return {}
    return {"hierarchy_nodes": True, "hierarchy_calibration": "frontier_sidak"}


def _families(variant: Variant) -> list[str]:
    """The multiplicity families the variant's own comparisons name, beyond the plan's."""

    named = {comparison["family"] for comparison in variant.comparisons}
    return sorted(named - {"displayed", "direct_means", "comparisons", "gate"})


def _plan(variant: Variant) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "schema_version": "3.0",
        "analysis": {
            "id": variant.name,
            "title": f"Summary agreement: {variant.name}",
            "intent": variant.intent,
            **(
                {"plan_frozen_at": "2025-12-01T00:00:00Z"}
                if variant.intent == "confirmatory"
                else {}
            ),
        },
        "inputs": {
            "design_manifest": "design.csv",
            "score": {
                "name": "harmful",
                "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
            },
            **variant.inputs,
        },
        "design": {
            "type": "fixed",
            "hierarchy": ["category", "prompt_id"],
            "require_complete_manifest": True,
            "require_expected_rollouts": True,
            "independence_profiles": {
                "rollout": {"block_by": ["rollout"], "comparison_coupling": "independent"},
                "by_category": {
                    "block_by": ["category"],
                    "comparison_coupling": "independent",
                },
                "by_prompt": {"block_by": ["prompt_id"], "comparison_coupling": "independent"},
            },
        },
        "cohorts": {
            name: {"match": {"model": model}} for name, model in COHORTS.items()
        },
        "groups": [{"id": "all"}],
        "outputs": {
            "prompt_bounds": {
                "enabled": True,
                "simultaneous": True,
                "family": "displayed",
                "alpha_allocation": "equal_bonferroni",
            },
            "direct_mean_intervals": {
                "enabled": True,
                "method": "weighted_kl_chernoff",
                "family": "direct_means",
                "groups": ["all"],
                "simultaneous": True,
                "coverage_mode": "two_sided",
                **_nodes(variant),
                "independence": variant.independence,
            },
            "risk_measures": [{"id": "mean", "type": "weighted_mean"}, *variant.measures],
        },
        "comparisons": [
            {
                "id": f"candidate_vs_baseline_{estimand}",
                "group": "all",
                "measure": "mean",
                "baseline": "baseline",
                "candidate": "candidate",
                "estimand": estimand,
                "method": "weighted_kl_chernoff",
                "family": "comparisons",
                "independence": {"primary": "rollout"},
                **(_nodes(variant) if estimand == "risk_difference" else {}),
            }
            for estimand in ("risk_difference", "risk_ratio")
        ]
        + list(variant.comparisons),
        "multiplicity": {
            "families": [
                {
                    "id": family,
                    "purpose": "simultaneous_intervals",
                    "method": "bonferroni",
                    "alpha": 0.05,
                }
                for family in ("displayed", "direct_means", "comparisons", *_families(variant))
            ]
            + [
                {
                    "id": "gate",
                    "purpose": "all_mandatory",
                    "method": "intersection_union",
                    "alpha": 0.05,
                }
            ],
        },
        "claims": [
            claim_spec("candidate", acceptable_upper=0.1, unacceptable_lower=0.3),
            claim_spec("baseline", acceptable_upper=0.05, unacceptable_lower=0.1, mandatory=False),
        ],
        "report": {"include_decisions": True},
    }
    plan.update(variant.plan)
    return plan


def _manifest(variant: Variant) -> str:
    extra = sorted(variant.columns)
    header = ["prompt_id", "question_id", "task_id", "category", "expected_rollouts", *extra]
    rows = [",".join(header)]
    for index in range(variant.prompts):
        values = [
            f"p{index:02d}",
            f"q{index:02d}",
            "t1",
            "a" if index % 2 == 0 else "b",
            str(variant.rollouts),
            *(str(variant.columns[name][index]).lower() for name in extra),
        ]
        rows.append(",".join(values))
    return "\n".join(rows) + "\n"


def _files(variant: Variant, directory: Path) -> RolloutFiles:
    runs = directory / "runs.jsonl"
    rollouts = directory / "rollouts.jsonl"
    cohorts = [name for name in COHORTS if name in variant.failures]
    with runs.open("w", encoding="utf-8", newline="\n") as stream:
        for name in cohorts:
            record = {
                "run_key": f"run-{name}",
                "model": COHORTS[name],
                "status": "success",
                "started_at": "2026-01-01T00:00:00Z",
                "run_id": "run-1",
                "task_version": "1",
            }
            stream.write(json.dumps(record) + "\n")
    missing = set(variant.missing)
    with rollouts.open("w", encoding="utf-8", newline="\n") as stream:
        for name in cohorts:
            counts = variant.failures[name]
            for index in range(variant.prompts):
                failing = counts[index % len(counts)]
                for repeat in range(1, variant.rollouts + 1):
                    if (name, index, repeat) in missing:
                        continue
                    record = {
                        "run_key": f"run-{name}",
                        "item_id": f"p{index:02d}",
                        "repeat": repeat,
                        "rollout_id": f"{name}-p{index:02d}-{repeat}",
                        "scores": {"harmful": "FAIL" if repeat <= failing else "PASS"},
                        "error": None,
                        "retries": 0,
                    }
                    stream.write(json.dumps(record) + "\n")
    return RolloutFiles(runs=runs, rollouts=rollouts)


def write_variant(variant: Variant, directory: Path) -> tuple[Path, RolloutFiles]:
    """The plan and rollout files of ``variant``, written into ``directory``."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(_manifest(variant), encoding="utf-8", newline="\n")
    plan = directory / "plan.yaml"
    plan.write_text(
        yaml.safe_dump(_plan(variant), sort_keys=False), encoding="utf-8", newline="\n"
    )
    return plan, _files(variant, directory)


def analysed(variant: Variant, directory: Path) -> tuple[SafetyReport, Path]:
    """The report of ``variant`` and the directory it was written to."""

    plan, source = write_variant(variant, directory / "inputs")
    report = analyze(plan=plan, source=source)
    return report, report.write(directory / "report")


def section(page: str, heading: str) -> str:
    """The text under ``## heading`` of a Markdown page, up to the next ``## `` heading."""

    start = page.index(f"\n## {heading}\n") + len(heading) + 5
    end = page.find("\n## ", start)
    return page[start:] if end == -1 else page[start:end]


def table(text: str) -> list[dict[str, str]]:
    """The rows of the first Markdown table in ``text``, by header.

    A header that repeats, such as "Interval", gets its position appended from the second
    time on ("Interval 2").
    """

    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("| "))
    headers: list[str] = []
    for cell in _cells(lines[start]):
        headers.append(cell if cell not in headers else f"{cell} {headers.count(cell) + 1}")
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(dict(zip(headers, _cells(line), strict=True)))
    return rows


def _cells(line: str) -> list[str]:
    # A cell may hold an escaped pipe; none of the tables read here does.
    return [cell.strip() for cell in line.strip().strip("|").split("|")]
