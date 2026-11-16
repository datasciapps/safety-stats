"""Anytime-valid bounds for runs stopped early (#126), end to end through real Inspect logs.

A fixed design of 40 prompts runs on Inspect's mock model in a shuffled order with a recorded
seed, and an early-stopping manager, standing in for optstop, halts it after 25 samples. Four
prompts always fail, so theta is 0.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState, generate
from inspect_ai.util._early_stopping import EarlyStop

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.schema.reports import Diagnostic, SafetyReport
from inspect_safety_stats.stats.sequential import anytime_upper_bound

PROMPTS = 40
STOP_AFTER = 25
SEED = 11

PLAN = """
schema_version: "2.0"
analysis: {id: stopped, title: Stopped early, intent: confirmatory,
           plan_frozen_at: 2025-01-01T00:00:00Z}
inputs:
  design_manifest: design.csv
  score:
    name: verdict
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
    missing: conservative_bounds
    errors: block
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  rollout_key: [eval_id, id, epoch]
  require_complete_manifest: false
  require_expected_rollouts: false
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
    prompt: {block_by: [prompt_id], comparison_coupling: independent}
    question: {block_by: [question_id], comparison_coupling: independent}
cohorts:
  mock: {match: {model: mockllm/model}}
groups:
  - id: all
    filter: {suite: s}
outputs:
  prompt_bounds: {enabled: false}
  risk_measures:
    - {id: mean, type: weighted_mean}
@ANYTIME@
multiplicity:
  families:
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - {id: loose, title: Loose, cohort: mock, group: all, measure: mean, family: gate,
     acceptable_upper: 0.8, unacceptable_lower: 0.9}
  - {id: tight, title: Tight, cohort: mock, group: all, measure: mean, family: gate,
     acceptable_upper: 0.05, unacceptable_lower: 0.5}
report: {include_observations: true, decimal_places: 6}
"""

SECTION = (
    "  anytime_bounds:\n"
    "    independence_profile: {profile}\n"
    "    family: {family}\n"
)


def _failing(prompt: str) -> bool:
    return int(prompt[1:]) % 10 == 0


@scorer(metrics=[])
def verdict():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value="FAIL" if _failing(str(state.sample_id)) else "PASS")

    return score


class StopAfter:
    """An early-stopping manager that halts the run after a fixed number of samples."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.completed = 0

    async def start_task(self, task: Any, samples: list[Sample], epochs: int) -> str:
        return "stop_after"

    async def schedule_sample(self, id: str | int, epoch: int) -> EarlyStop | None:
        if self.completed >= self.count:
            return EarlyStop(id=id, epoch=epoch, reason="stop_after")
        return None

    async def complete_sample(self, id: str | int, epoch: int, scores: Any) -> None:
        self.completed += 1

    async def complete_task(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def _run_from_the_drive_that_holds_the_logs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work from the drive of pytest's temporary root, where these tests write real logs (#35)."""

    monkeypatch.chdir(tmp_path_factory.getbasetemp())


def _run(log_dir: Path, shuffle: bool | int | None) -> str:
    task = Task(
        dataset=MemoryDataset([Sample(id=f"p{i}", input=f"q{i}") for i in range(PROMPTS)]),
        solver=generate(),
        scorer=verdict(),
        early_stopping=StopAfter(STOP_AFTER),
    )
    (log,) = inspect_eval(
        task,
        model="mockllm/model",
        log_dir=str(log_dir),
        sample_shuffle=shuffle,
        max_samples=1,
        display="none",
    )
    # Inspect gives the location relative to the working directory of the run.
    return str(Path(log.location).resolve())


@pytest.fixture(scope="module")
def logs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("anytime-logs")
    return {
        "seeded": _run(root / "seeded", SEED),
        "unseeded": _run(root / "unseeded", True),
        "unshuffled": _run(root / "unshuffled", None),
    }


def _analyze(directory: Path, log: str, section: str = "") -> SafetyReport:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite\n"
        + "".join(f"p{i},q{i},t1,s\n" for i in range(PROMPTS)),
        encoding="utf-8",
    )
    plan = directory / "plan.yaml"
    plan.write_text(PLAN.replace("@ANYTIME@\n", section).strip() + "\n", encoding="utf-8")
    return analyze(plan=load_plan(plan), logs=log)


def _found(report: SafetyReport, code: str) -> list[Diagnostic]:
    return [item for item in report.integrity.diagnostics if item.code == code]


def test_a_stopped_shuffled_run_gets_the_bound_of_its_recorded_prefix(
    logs: dict[str, str], tmp_path: Path
) -> None:
    header = read_eval_log(logs["seeded"], header_only=True)
    order = [str(item) for item in header.eval.dataset.sample_ids or []]
    prefix = order[:STOP_AFTER]

    report = _analyze(
        tmp_path, logs["seeded"], SECTION.format(profile="prompt", family="gate")
    )

    (diagnostic,) = _found(report, "ANYTIME_UPPER_BOUND")
    details = diagnostic.details
    assert diagnostic.severity == "INFO"
    assert details["sample_shuffle"] == SEED
    assert details["population"] == PROMPTS
    assert details["draws"] == details["complete_draws"] == STOP_AFTER
    assert details["rollouts_counted_as_failures"] == 0
    assert details["independent_rollouts"] == 1
    expected = anytime_upper_bound(
        [1.0 if _failing(prompt) else 0.0 for prompt in prefix],
        np.array([int(prompt[1:]) for prompt in prefix]),
        np.full(PROMPTS, 1 / PROMPTS),
        alpha=0.05,
    )
    assert details["upper"] == expected.upper
    assert details["alpha"] == 0.05
    assert {item["claim_id"]: item["certified"] for item in details["claims"]} == {
        "loose": expected.upper <= 0.8,
        "tight": False,
    }
    assert expected.upper <= 0.8


def test_the_bound_changes_no_claim(logs: dict[str, str], tmp_path: Path) -> None:
    plain = _analyze(tmp_path / "plain", logs["seeded"])
    with_bound = _analyze(
        tmp_path / "bound", logs["seeded"], SECTION.format(profile="prompt", family="gate")
    )
    assert [claim.model_dump() for claim in with_bound.claims] == [
        claim.model_dump() for claim in plain.claims
    ]
    assert with_bound.decision == plain.decision


@pytest.mark.parametrize(
    ("run", "why"),
    [
        ("unshuffled", "Inspect does not shuffle by default"),
        ("unseeded", "without a recorded seed"),
    ],
)
def test_a_run_without_a_recorded_order_is_refused(
    logs: dict[str, str], tmp_path: Path, run: str, why: str
) -> None:
    report = _analyze(tmp_path, logs[run], SECTION.format(profile="prompt", family="gate"))

    (diagnostic,) = _found(report, "ANYTIME_ORDER_UNRECORDED")
    assert diagnostic.severity == "ERROR"
    assert why in diagnostic.message
    assert not _found(report, "ANYTIME_UPPER_BOUND")
    assert report.integrity.status == "invalid"


@pytest.mark.parametrize(
    ("profile", "family", "problem"),
    [
        ("question", "gate", "must block by rollout or by prompt_id"),
        ("absent", "gate", "is not declared in design.independence_profiles"),
        ("prompt", "absent", "is not declared in multiplicity.families"),
    ],
)
def test_a_plan_the_bound_cannot_serve_is_refused(
    logs: dict[str, str], tmp_path: Path, profile: str, family: str, problem: str
) -> None:
    report = _analyze(
        tmp_path, logs["seeded"], SECTION.format(profile=profile, family=family)
    )

    (diagnostic,) = _found(report, "PLAN_ANYTIME_BOUNDS_INVALID")
    assert problem in diagnostic.message
    assert not _found(report, "ANYTIME_UPPER_BOUND")


def test_plans_without_the_section_keep_their_dump(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text(PLAN.replace("@ANYTIME@\n", ""), encoding="utf-8")
    assert "anytime_bounds" not in load_plan(plan).model_dump(mode="json")["outputs"]
