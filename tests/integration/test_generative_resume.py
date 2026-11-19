"""Resuming an interrupted materialization.

A provider-backed rewrite run is long, costly and interruptible. Counter-based selection
makes resumption safe -- draw t depends only on the seed and t, so the units a resumed run
re-derives are exactly the ones the interrupted run would have gone on to draw -- but the
ledgers have to survive the interruption for that to be usable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from inspect_safety_stats.generative.execution import check_attempts, read_prompt_store
from inspect_safety_stats.generative.kernels import RewriteOutcome, RewriteRequest
from inspect_safety_stats.generative.ledger import read_attempts, read_rewrites
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


def test_a_rerun_errored_unit_is_recorded_once(tmp_path: Path) -> None:
    """The error row a unit leaves must not stay beside the row of its rerun."""

    plan = write_plan(tmp_path, total=60)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(error_every=5), plan, artifacts)
    before = (artifacts / "rewrites.csv").read_bytes().splitlines(keepends=True)
    finished = [line for line in before if b",error," not in line]
    assert len(before) - len(finished) == 12

    resumed = run_with(ScriptedKernel(), plan, artifacts, resume=True)
    assert resumed is not None and resumed.accepted == 60
    records = read_rewrites(artifacts / "rewrites.csv")
    assert len(records) == 60
    assert {record.validation_status for record in records} == {"valid"}
    after = (artifacts / "rewrites.csv").read_bytes().splitlines(keepends=True)
    assert after[: len(finished)] == finished, "the finished rows are kept byte for byte"


def test_an_interrupted_rerun_of_errored_units_leaves_a_readable_ledger(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, total=60)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(error_every=5), plan, artifacts)
    run_with(ScriptedKernel(survive=5), plan, artifacts, resume=True)
    records = read_rewrites(artifacts / "rewrites.csv")
    assert len(records) == 48 + 5
    assert not (artifacts / "rewrites.csv.partial").exists()

    run = run_with(ScriptedKernel(), plan, artifacts, resume=True)
    assert run is not None and run.resumed == 53 and run.accepted == 60
    assert len(read_rewrites(artifacts / "rewrites.csv")) == 60


def test_resume_repairs_a_ledger_that_kept_error_rows_beside_their_reruns(
    tmp_path: Path,
) -> None:
    """Earlier versions appended a rerun's row after the unit's error row."""

    plan = write_plan(tmp_path, total=60)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(error_every=5), plan, artifacts)
    lines = (artifacts / "rewrites.csv").read_bytes().splitlines(keepends=True)
    errors = [line for line in lines if b",error," in line]
    run_with(ScriptedKernel(), plan, artifacts, resume=True)
    lines = (artifacts / "rewrites.csv").read_bytes().splitlines(keepends=True)
    (artifacts / "rewrites.csv").write_bytes(b"".join([lines[0], *errors, *lines[1:]]))
    with pytest.raises(ValueError, match="repeats a sampling_unit_id"):
        read_rewrites(artifacts / "rewrites.csv")

    run = run_with(ScriptedKernel(), plan, artifacts, resume=True)
    assert run is not None and run.resumed == 60 and run.accepted == 60
    assert len(read_rewrites(artifacts / "rewrites.csv")) == 60


def test_resume_continues_ledgers_a_development_version_wrote_with_crlf(tmp_path: Path) -> None:
    """Before resume replaced them, the ledgers were appended to with CRLF line endings.

    No released version wrote them that way, but a development directory may hold such
    files. A resume writes them afresh, so their rows end like the ones it appends, even
    when it drops no row: the interrupted run here recorded no error.
    """

    plan = write_plan(tmp_path, total=60)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(survive=30), plan, artifacts)
    names = ("rewrites.csv", "prompts.restricted.csv")
    for name in names:
        path = artifacts / name
        assert b"\r" not in path.read_bytes(), name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    prompts = {
        key: entry.prompt
        for key, entry in read_prompt_store(artifacts / "prompts.restricted.csv").items()
    }
    assert len(prompts) == 30

    run = run_with(ScriptedKernel(), plan, artifacts, resume=True)
    assert run is not None and run.resumed == 30 and run.accepted == 60
    for name in names:
        assert b"\r" not in (artifacts / name).read_bytes(), name
    assert len(read_rewrites(artifacts / "rewrites.csv")) == 60
    stored = read_prompt_store(artifacts / "prompts.restricted.csv")
    assert len(stored) == 60
    assert {key: stored[key].prompt for key in prompts} == prompts


def near_duplicate_plan(tmp_path: Path, total: int) -> Path:
    """A plan that also records attempts.csv, so a resume replaces all three ledgers."""

    path = write_plan(tmp_path, total=total)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["rewrite"].setdefault("deduplication", {})["near_duplicate"] = "normalized_text"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "where",
    [
        "rewrites.csv header",
        "rewrites.csv first row",
        "rewrites.csv last row",
        "prompts.restricted.csv row",
        "attempts.csv row",
    ],
)
def test_a_resume_killed_while_it_replaces_the_ledgers_loses_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """Each ledger is either as it was or fully replaced, never cut short.

    The resume replaces rewrites.csv, prompts.restricted.csv and attempts.csv in that
    order, each through a sibling file. Killed while it writes one of them, it must leave
    that one byte for byte as it was, and the ones before it complete.
    """

    plan = near_duplicate_plan(tmp_path, total=60)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(error_every=5), plan, artifacts)
    names = ("rewrites.csv", "prompts.restricted.csv", "attempts.csv")
    before = {name: (artifacts / name).read_bytes() for name in names}
    lines = {name: before[name].splitlines(keepends=True) for name in names}
    errored = {line.split(b",")[0] for line in lines["rewrites.csv"] if b",error," in line}
    assert len(errored) == 12
    kept = {
        name: b"".join(
            line for index, line in enumerate(lines[name])
            if index == 0 or line.split(b",")[0] not in errored
        )
        for name in names
    }
    assert kept["prompts.restricted.csv"] == before["prompts.restricted.csv"]
    assert kept["rewrites.csv"] != before["rewrites.csv"] != kept["attempts.csv"]

    # One writerow call per header and per row, ledger after ledger.
    rows = {name: len(kept[name].splitlines()) for name in names}
    first = {"rewrites.csv": 1}
    first["prompts.restricted.csv"] = first["rewrites.csv"] + rows["rewrites.csv"]
    first["attempts.csv"] = first["prompts.restricted.csv"] + rows["prompts.restricted.csv"]
    failing = {
        "rewrites.csv header": first["rewrites.csv"],
        "rewrites.csv first row": first["rewrites.csv"] + 1,
        "rewrites.csv last row": first["prompts.restricted.csv"] - 1,
        "prompts.restricted.csv row": first["prompts.restricted.csv"] + 10,
        "attempts.csv row": first["attempts.csv"] + 10,
    }[where]
    failed = where.split(" ")[0]

    writerow = csv.DictWriter.writerow
    calls = 0

    def dying(self: csv.DictWriter, row: dict) -> object:
        nonlocal calls
        calls += 1
        if calls == failing:
            raise Interrupted("killed while the ledgers were being replaced")
        return writerow(self, row)

    monkeypatch.setattr(csv.DictWriter, "writerow", dying)
    with pytest.raises(Interrupted):
        run_with(ScriptedKernel(), plan, artifacts, resume=True)
    monkeypatch.setattr(csv.DictWriter, "writerow", writerow)

    order = list(names)
    for name in names:
        now = (artifacts / name).read_bytes()
        if order.index(name) < order.index(failed):
            assert now == kept[name], f"{name} was replaced before the failure"
        else:
            assert now == before[name], f"{name} is unchanged"
    assert sorted(item.name for item in artifacts.glob("*.partial")) == []

    run = run_with(ScriptedKernel(), plan, artifacts, resume=True)
    assert run is not None and run.resumed == 48 and run.accepted == 60
    records = read_rewrites(artifacts / "rewrites.csv")
    assert len(records) == 60
    check_attempts(records, read_attempts(artifacts / "attempts.csv"))
    assert len(read_prompt_store(artifacts / "prompts.restricted.csv")) == 60
    assert sorted(item.name for item in artifacts.glob("*.partial")) == []


def test_overwrite_removes_a_partial_ledger_a_killed_resume_left(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, total=20)
    artifacts = tmp_path / "run"
    run_with(ScriptedKernel(), plan, artifacts)
    for name in ("rewrites.csv", "prompts.restricted.csv"):
        (artifacts / f"{name}.partial").write_bytes(b"half a ledger\n")
    materialize_run(plan, artifacts, overwrite=True)
    assert sorted(item.name for item in artifacts.glob("*.partial")) == []


def test_a_run_killed_between_a_units_two_rows_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted unit is two writes, its prompt row and its rewrite row; a kill can land
    between them. Before, the rewrite row came first, and the unit it left recorded as
    accepted without its text made every resume refuse the directory."""

    from inspect_safety_stats.generative.ledger import IncrementalCsvWriter

    plan = write_plan(tmp_path, total=20)
    artifacts = tmp_path / "run"
    original = IncrementalCsvWriter.append
    rows = 0

    def killed_after_eleven_rows(self: IncrementalCsvWriter, row: dict[str, object]) -> None:
        nonlocal rows
        if self._path.name in {"rewrites.csv", "prompts.restricted.csv"}:
            if rows == 11:
                raise Interrupted("killed between a unit's two rows")
            rows += 1
        original(self, row)

    monkeypatch.setattr(IncrementalCsvWriter, "append", killed_after_eleven_rows)
    with pytest.raises(Interrupted):
        run_with(ScriptedKernel(), plan, artifacts)
    monkeypatch.setattr(IncrementalCsvWriter, "append", original)

    run = run_with(ScriptedKernel(), plan, artifacts, resume=True)
    assert run is not None and run.resumed == 5 and run.accepted == 20
    assert len(read_rewrites(artifacts / "rewrites.csv")) == 20
    with (artifacts / "prompts.restricted.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 20
