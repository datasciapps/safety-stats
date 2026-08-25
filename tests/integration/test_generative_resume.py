"""Resuming an interrupted materialization.

A provider-backed rewrite run is long, costly and interruptible. Counter-based selection
makes resumption safe -- draw t depends only on the seed and t, so the units a resumed run
re-derives are exactly the ones the interrupted run would have gone on to draw -- but the
ledgers have to survive the interruption for that to be usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from inspect_safety_stats.generative.execution import read_prompt_store
from inspect_safety_stats.generative.kernels import RewriteOutcome, RewriteRequest
from inspect_safety_stats.generative.ledger import read_rewrites
from inspect_safety_stats.generative.runner import materialize_run
from tests.integration.test_generative_cli import write_plan

runner = CliRunner()


class Interrupted(RuntimeError):
    """Stands in for a rate limit, a dropped connection or a spend cap."""


@dataclass
class ScriptedKernel:
    """A kernel whose behaviour is fixed, so runs can be compared like for like.

    ``survive`` bounds how many units it handles before dying the way a provider run dies.
    ``repeat`` makes it return one prompt per semantic seed, which forces deduplication.
    ``reject`` makes every attempt fail validation. ``error_every`` reports an operational
    error rather than raising.
    """

    survive: int = 10**6
    repeat: bool = False
    reject: bool = False
    error_every: int = 0
    kernel_id: str = "fake-rewriter"
    seen: int = 0

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        if self.seen >= self.survive:
            raise Interrupted("provider stopped responding")
        self.seen += 1
        if self.reject:
            return RewriteOutcome(status="invalid", detail="never valid")
        if self.error_every and self.seen % self.error_every == 0:
            return RewriteOutcome(status="error", detail="transient provider error")
        if self.repeat:
            return RewriteOutcome(
                status="valid", prompt=f"one per seed {request.semantic_seed_id}"
            )
        return RewriteOutcome(status="valid", prompt=f"rewrite of {request.sampling_unit_id}")


def run_with(kernel: ScriptedKernel, plan: Path, artifacts: Path, *, resume: bool = False):
    """Materialize with a scripted kernel, returning None if it was interrupted."""

    import inspect_safety_stats.generative.runner as runner_module

    original = runner_module.resolve_kernel
    runner_module.resolve_kernel = lambda spec: kernel  # type: ignore[assignment]
    try:
        if kernel.survive < 10**6:
            with pytest.raises(Interrupted):
                materialize_run(plan, artifacts, resume=resume)
            return None
        return materialize_run(plan, artifacts, resume=resume)
    finally:
        runner_module.resolve_kernel = original  # type: ignore[assignment]


def test_an_interrupted_run_leaves_a_usable_ledger(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, total=200)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(survive=120), plan, artifacts)

    assert (artifacts / "selections.csv").is_file(), "selections must be written before execution"
    records = read_rewrites(artifacts / "rewrites.csv")
    assert len(records) == 120
    prompts = read_prompt_store(artifacts / "prompts.restricted.csv")
    assert len(prompts) == 120


def test_resume_executes_only_what_is_outstanding(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, total=200)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(survive=120), plan, artifacts)

    run = run_with(ScriptedKernel(), plan, artifacts, resume=True)
    assert run is not None
    assert run.resumed == 120
    assert run.selections == 200
    assert run.accepted == 200
    assert len(read_rewrites(artifacts / "rewrites.csv")) == 200


def test_resume_reaches_the_same_state_as_an_uninterrupted_run(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, total=200)
    straight = run_with(ScriptedKernel(), plan, tmp_path / "straight")
    broken = tmp_path / "broken"
    run_with(ScriptedKernel(survive=120), plan, broken)
    resumed = run_with(ScriptedKernel(), plan, broken, resume=True)

    assert straight is not None and resumed is not None
    assert resumed.accepted == straight.accepted
    by_unit = {
        record.sampling_unit_id: record.prompt_sha256
        for record in read_rewrites(broken / "rewrites.csv")
    }
    for record in read_rewrites(tmp_path / "straight" / "rewrites.csv"):
        assert by_unit[record.sampling_unit_id] == record.prompt_sha256

    straight_dataset = (tmp_path / "straight" / "datasets" / "dataset.jsonl").read_text()
    resumed_dataset = (broken / "datasets" / "dataset.jsonl").read_text()
    assert sorted(resumed_dataset.splitlines()) == sorted(straight_dataset.splitlines())


def test_resume_is_not_destructive_if_it_is_itself_interrupted(tmp_path: Path) -> None:
    """A second crash must not lose the first run's work."""

    plan = write_plan(tmp_path, total=200)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(survive=100), plan, artifacts)
    assert len(read_rewrites(artifacts / "rewrites.csv")) == 100

    run_with(ScriptedKernel(survive=50), plan, artifacts, resume=True)
    assert len(read_rewrites(artifacts / "rewrites.csv")) == 150

    run = run_with(ScriptedKernel(), plan, artifacts, resume=True)
    assert run is not None
    assert run.resumed == 150
    assert run.accepted == 200


def test_exhausted_units_are_final_and_are_not_retried(tmp_path: Path) -> None:
    """Retrying them would grant more than maximum_attempts and shift the estimand."""

    plan = write_plan(tmp_path, total=50)
    artifacts = tmp_path / "run"
    first = run_with(ScriptedKernel(reject=True), plan, artifacts)
    assert first is not None and first.accepted == 0 and first.exhausted == 50

    # A working kernel on resume must not rescue them.
    second = run_with(ScriptedKernel(), plan, artifacts, resume=True)
    assert second is not None
    assert second.resumed == 50
    assert second.accepted == 0
    assert second.exhausted == 50
    records = read_rewrites(artifacts / "rewrites.csv")
    assert all(record.validation_status == "invalid" for record in records)
    assert all(record.rewrite_attempts == 5 for record in records)


def test_errored_units_are_retried_because_they_say_nothing_about_the_kernel(
    tmp_path: Path,
) -> None:
    plan = write_plan(tmp_path, total=60)
    artifacts = tmp_path / "run"
    first = run_with(ScriptedKernel(error_every=5), plan, artifacts)
    assert first is not None
    assert first.errored == 12
    assert first.accepted == 48

    second = run_with(ScriptedKernel(), plan, artifacts, resume=True)
    assert second is not None
    assert second.resumed == 48        # only the successes carried over
    assert second.accepted == 60       # the errored twelve ran again and succeeded
    assert second.errored == 0


def test_resume_rebuilds_deduplication_state(tmp_path: Path) -> None:
    """A resumed run must reject repeats a single run would have rejected."""

    plan = write_plan(tmp_path, total=40)
    straight = run_with(ScriptedKernel(repeat=True), plan, tmp_path / "straight")
    assert straight is not None

    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(repeat=True, survive=15), plan, artifacts)
    resumed = run_with(ScriptedKernel(repeat=True), plan, artifacts, resume=True)
    assert resumed is not None

    # One accepted rewrite per semantic seed. Without rebuilt deduplication state the
    # resumed run would accept repeats that the uninterrupted one rejected.
    assert straight.accepted == 3
    assert resumed.accepted == straight.accepted


def test_resume_refuses_a_directory_from_a_different_design(tmp_path: Path) -> None:
    plan_path = write_plan(tmp_path, total=100)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(survive=40), plan_path, artifacts)

    edited = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    edited["evaluation"]["selection_seed"] = 999
    plan_path.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="belongs to a different design"):
        materialize_run(plan_path, artifacts, resume=True)


def test_overwrite_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, total=20)
    with pytest.raises(ValueError, match="mutually exclusive"):
        materialize_run(plan, tmp_path / "run", overwrite=True, resume=True)


def test_a_non_empty_directory_names_both_ways_out(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, total=20)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    with pytest.raises(FileExistsError, match="--overwrite to replace it or --resume"):
        materialize_run(plan, artifacts)


def test_resume_through_the_cli(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, total=100)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(survive=60), plan, artifacts)

    result = runner.invoke(
        app,
        ["generative", "materialize", "--plan", str(plan),
         "--output", str(artifacts), "--resume"],
    )
    assert result.exit_code == 0, result.stdout
    assert "Resumed: 60 units carried over, 40 executed" in result.stdout
