"""What a diagnostic-code case is, and the analysis it runs on.

A case triggers its code with inputs that break the code's assumption, and runs the same
analysis on control inputs that differ only in keeping the assumption. The test in
``tests/test_diagnostic_codes.py`` then checks, for every case:

- the trigger emits the code and the control does not, so the code reports on the
  assumption the inputs differ in;
- every emitted diagnostic has a severity, source and effect the code's registry entry lists,
  in its JSON form, as ``report.json`` holds it, too;
- the case's source, the kind of input its trigger breaks, is the source the registry gives
  the code. Each case module holds the cases of one kind and says which in ``SOURCE``; a case
  whose trigger breaks another kind says so in its own ``source``. Neither is read from the
  registry, so an entry that moves a code to another kind of input fails the test;
- the case's ``shows``, the effect its checks show the code has, is the effect the registry
  gives the code. It is written in the case, beside the checks that show it, and not read
  from the registry, so that a registry entry that says something other than what the code
  does fails the test;
- ``shows`` is what the trigger does to the integrity status, compared with the control's:
  ``invalid`` makes it invalid, ``degraded`` makes a valid one degraded, and every other
  effect leaves it as it is;
- where the entry names assumptions of ``ASSUMPTIONS`` and the analysis reports the ones it
  rests on, the trigger's report lists them;
- the case's own ``effect`` check, which says what the code does to the results. The test
  requires one for ``withheld``, ``adjusted`` and ``recorded``, whose effect the integrity
  status does not show, and it is the check that shows that effect: a withheld result, a
  result computed differently, or the same results.

Inputs for the main analysis are built from ``base_inputs`` and run with ``run_analysis``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from safety_stats.registry import Effect, SourceKind
from safety_stats.schema.reports import Diagnostic

Status = str | None


@dataclass(frozen=True)
class Outcome:
    """What one run of an analysis gives a case to check.

    ``status`` is the integrity status (``valid``, ``degraded`` or ``invalid``), or for an
    analysis without one, ``invalid`` when it fails as an ``invalid`` effect says and
    ``valid`` otherwise. ``assumptions`` holds the ids of the registry assumptions the
    report says it rests on, or None when the analysis reports none. ``report`` is whatever
    the analysis returned, for the case's own effect check.
    """

    diagnostics: tuple[Diagnostic, ...]
    status: Status
    report: Any = None
    assumptions: frozenset[str] | None = None

    def with_code(self, code: str) -> list[Diagnostic]:
        return [item for item in self.diagnostics if item.code == code]


Run = Callable[[Path], Outcome]
EffectCheck = Callable[[Outcome, Outcome], None]


@dataclass(frozen=True)
class Case:
    """One way to trigger ``code``: ``trigger`` breaks its assumption, ``control`` keeps it.

    Both get a directory of their own to write inputs in. ``effect`` gets the trigger's
    outcome, then the control's, and asserts what the code did to the results. ``shows`` is
    the effect, as ``safety_stats.registry.EFFECTS`` names them, that the integrity
    status and ``effect`` show the code has. ``source`` is the kind of input the trigger
    breaks, as ``safety_stats.registry.SOURCE_KINDS`` names them, when it is not the
    ``SOURCE`` of the case's module.
    """

    code: str
    shows: Effect
    trigger: Run
    control: Run
    effect: EffectCheck | None = None
    source: SourceKind | None = None


# The main analysis: one cohort, two prompts of four rollouts each, a claim on the mean.
BASE_PLAN: dict[str, Any] = {
    "schema_version": "3.0",
    "analysis": {
        "id": "codes",
        "title": "Diagnostic codes",
        "intent": "confirmatory",
        "plan_frozen_at": "2026-01-01T00:00:00Z",
        # An external registration, so that confirmatory timing is not self-declared.
        "registration": {"uri": "https://example.org/registrations/codes", "sha256": "0" * 64},
    },
    "inputs": {
        "design_manifest": "design.csv",
        "score": {
            "name": "harmful",
            "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
        },
    },
    "cohorts": {"candidate": {"match": {"model": "m"}}},
    "groups": [{"id": "all"}],
    "outputs": {
        "prompt_bounds": {"enabled": True, "family": "displayed"},
        "risk_measures": [{"id": "mean", "type": "weighted_mean"}],
    },
    "multiplicity": {
        "families": [
            {
                "id": "displayed",
                "purpose": "simultaneous_intervals",
                "method": "bonferroni",
                "alpha": 0.05,
            },
            {"id": "claims", "purpose": "all_mandatory", "method": "bonferroni", "alpha": 0.05},
        ]
    },
    "claims": [
        {
            "id": "low",
            "title": "Mean failure rate is low",
            "cohort": "candidate",
            "group": "all",
            "measure": "mean",
            "family": "claims",
            "acceptable_upper": 0.9,
            "unacceptable_lower": 0.95,
        }
    ],
}

BASE_RUN: dict[str, Any] = {
    "run_key": "r1",
    "model": "m",
    "status": "success",
    "started_at": "2026-02-01T00:00:00Z",
}


def rollout(item: str, repeat: int, score: Any = "PASS", **fields: Any) -> dict[str, Any]:
    """A rollout record of run ``r1``; ``fields`` adds or overrides record fields."""

    return {"run_key": "r1", "item_id": item, "repeat": repeat, "score_harmful": score, **fields}


@dataclass
class Inputs:
    """The plan, manifest, runs and rollouts of one analysis, as plain data to change.

    ``plan`` is written as the plan file (JSON, which YAML reads). ``files`` holds other
    files to write beside it: a value is the file's text, or a function of the plan file's
    path that returns it, for a file that depends on the plan, such as a registration
    receipt. ``decision_policy`` and ``registration_receipt`` name files of ``files`` to pass
    to ``analyze``.
    """

    plan: dict[str, Any]
    manifest: list[dict[str, Any]]
    runs: list[dict[str, Any]]
    rollouts: list[dict[str, Any]]
    files: dict[str, str | Callable[[Path], str]] = field(default_factory=dict)
    decision_policy: str | None = None
    registration_receipt: str | None = None


def base_inputs() -> Inputs:
    """Inputs on which the main analysis raises no diagnostic: a fresh copy to change."""

    return Inputs(
        plan=copy.deepcopy(BASE_PLAN),
        manifest=[
            {"prompt_id": "p1", "question_id": "q1", "task_id": "t1", "expected_rollouts": 4},
            {"prompt_id": "p2", "question_id": "q2", "task_id": "t1", "expected_rollouts": 4},
        ],
        runs=[dict(BASE_RUN)],
        rollouts=[rollout("p1", repeat) for repeat in range(1, 5)]
        + [rollout("p2", repeat, "FAIL" if repeat == 1 else "PASS") for repeat in range(1, 5)],
    )


def _csv(rows: list[dict[str, Any]]) -> str:
    columns: list[str] = []
    for row in rows:
        columns.extend(name for name in row if name not in columns)

    def cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    lines = [",".join(columns)]
    lines.extend(",".join(cell(row.get(name)) for name in columns) for row in rows)
    return "\n".join(lines) + "\n"


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


def write_inputs(inputs: Inputs, directory: Path) -> Path:
    """Write the inputs into ``directory`` and return the plan file's path."""

    directory.mkdir(parents=True, exist_ok=True)
    plan = directory / "plan.yaml"
    plan.write_text(json.dumps(inputs.plan, indent=1) + "\n", encoding="utf-8", newline="\n")
    manifest = inputs.plan.get("inputs", {}).get("design_manifest")
    if manifest and inputs.manifest:
        (directory / manifest).write_text(
            _csv(inputs.manifest), encoding="utf-8", newline="\n"
        )
    (directory / "runs.jsonl").write_text(_jsonl(inputs.runs), encoding="utf-8", newline="\n")
    (directory / "rollouts.jsonl").write_text(
        _jsonl(inputs.rollouts), encoding="utf-8", newline="\n"
    )
    for name, content in inputs.files.items():
        text = content(plan) if callable(content) else content
        (directory / name).write_text(text, encoding="utf-8", newline="\n")
    return plan


def run_analysis(inputs: Inputs, directory: Path) -> Outcome:
    """Run ``analyze`` on the inputs, written into ``directory``, and return its outcome."""

    from safety_stats import analyze
    from safety_stats.ingest.rollout_files import RolloutFiles

    plan = write_inputs(inputs, directory)
    report = analyze(
        plan=plan,
        source=RolloutFiles(runs=directory / "runs.jsonl", rollouts=directory / "rollouts.jsonl"),
        decision_policy=(
            None if inputs.decision_policy is None else directory / inputs.decision_policy
        ),
        registration_receipt=(
            None
            if inputs.registration_receipt is None
            else directory / inputs.registration_receipt
        ),
    )
    return Outcome(
        diagnostics=tuple(report.integrity.diagnostics),
        status=report.integrity.status,
        report=report,
        assumptions=frozenset(item.id for item in report.assumptions_used),
    )


def analysis(change: Callable[[Inputs], None] | None = None) -> Run:
    """A run of the main analysis on the base inputs, changed in place by ``change``."""

    def run(directory: Path) -> Outcome:
        inputs = base_inputs()
        if change is not None:
            change(inputs)
        return run_analysis(inputs, directory)

    return run


def claim(outcome: Outcome, claim_id: str = "low") -> Any:
    """The report's result for one claim."""

    (found,) = [item for item in outcome.report.claims if item.claim_id == claim_id]
    return found
