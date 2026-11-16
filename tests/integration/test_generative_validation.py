"""The package invoking the validator a plan declares (experimental).

``rewrite.validation.validator_id`` used to be declared and fingerprinted only, with the
kernel's own verdict trusted. With ``rewrite.validation.invoke`` the package calls the
validator on every rewrite and records each attempt, so fidelity can be reported by arm.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from inspect_safety_stats.generative.execution import execute_rewrites, load_seed_texts
from inspect_safety_stats.generative.kernels import (
    DeterministicFakeKernel,
    DeterministicFakeValidator,
    RewriteOutcome,
    RewriteRequest,
    RewriteValidator,
    ValidationVerdict,
    resolve_kernel_fingerprint,
)
from inspect_safety_stats.generative.ledger import read_attempts, read_rewrites
from inspect_safety_stats.generative.runner import materialize_run, resolve_validator
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy
from tests.cli_output import plain
from tests.generative_fixtures import SEEDS_WITH_TEXT, base_plan, materialize

runner = CliRunner()


def plan_document(
    *, invoke: bool = True, validator_id: str = "fake-validator", **rewrite: Any
) -> dict[str, Any]:
    plan = base_plan()
    plan["sampling"] = {"method": "direct", "total_rewrites": 60}
    plan["rewrite"]["validation"] = {"validator_id": validator_id, "invoke": invoke}
    plan["rewrite"].update(rewrite)
    return plan


def prepare(tmp_path: Path, document: dict[str, Any]) -> tuple[Any, ...]:
    parsed = load_generative_plan(materialize(tmp_path, document, SEEDS_WITH_TEXT))
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    selections = draw_selections(taxonomy, plan_design(taxonomy, parsed))
    fingerprint = resolve_kernel_fingerprint(parsed, base_path=tmp_path)
    texts = load_seed_texts(parsed, base_path=tmp_path)
    return parsed, selections, fingerprint, texts


@dataclass
class RecordingValidator:
    """Accepts everything, remembering what it was shown."""

    validator_id: str = "fake-validator"
    seen: list[tuple[RewriteRequest, str]] = field(default_factory=list)

    def validate(self, request: RewriteRequest, prompt: str) -> ValidationVerdict:
        self.seen.append((request, prompt))
        return ValidationVerdict(status="valid")


@dataclass
class FixedValidator:
    status: str
    validator_id: str = "fake-validator"

    def validate(self, request: RewriteRequest, prompt: str) -> ValidationVerdict:
        return ValidationVerdict(status=self.status, detail="fixed")  # type: ignore[arg-type]


def test_the_fake_validator_satisfies_the_protocol() -> None:
    assert isinstance(DeterministicFakeValidator(), RewriteValidator)


def test_a_plan_that_does_not_invoke_keeps_its_fingerprint(tmp_path: Path) -> None:
    """Existing runs must not change kernel hash because a setting was added."""

    implicit = plan_document()
    del implicit["rewrite"]["validation"]["invoke"]
    explicit = plan_document(invoke=False)
    invoked = plan_document(invoke=True)
    hashes = []
    for name, document in (("a", implicit), ("b", explicit), ("c", invoked)):
        directory = tmp_path / name
        directory.mkdir()
        parsed, _, fingerprint, _ = prepare(directory, document)
        hashes.append(fingerprint.sha256)
    assert hashes[0] == hashes[1]
    assert hashes[2] != hashes[0], "a validated run must not pool with an unvalidated one"


def test_invoking_without_a_validator_fails_loudly(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path, plan_document())
    with pytest.raises(ValueError, match="must be supplied"):
        execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)


def test_a_validator_the_plan_does_not_invoke_is_refused(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path, plan_document(invoke=False))
    with pytest.raises(ValueError, match="does not set rewrite.validation.invoke"):
        execute_rewrites(
            plan,
            selections,
            DeterministicFakeKernel(),
            fingerprint,
            texts,
            validator=DeterministicFakeValidator(),
        )


def test_the_validator_sees_every_rewrite_the_kernel_returns(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path, plan_document())
    validator = RecordingValidator()
    result = execute_rewrites(
        plan, selections, DeterministicFakeKernel(), fingerprint, texts, validator=validator
    )
    assert len(validator.seen) == len(selections)
    shown = {request.sampling_unit_id: prompt for request, prompt in validator.seen}
    assert {item.sampling_unit_id: item.prompt for item in result.accepted} == shown
    for request, _ in validator.seen:
        assert request.seed_text == texts[request.semantic_seed_id]


def test_a_rejecting_validator_exhausts_every_unit(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path, plan_document())
    result = execute_rewrites(
        plan,
        selections,
        DeterministicFakeKernel(),
        fingerprint,
        texts,
        validator=FixedValidator("invalid"),
    )
    assert not result.accepted
    assert len(result.exhausted) == len(selections)
    assert len(result.attempts) == 5 * len(selections)
    assert {item.validator_status for item in result.attempts} == {"invalid"}
    assert {item.kernel_status for item in result.attempts} == {"valid"}
    assert not any(item.accepted for item in result.attempts)


def test_a_validator_error_is_operational(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path, plan_document())
    result = execute_rewrites(
        plan,
        selections,
        DeterministicFakeKernel(),
        fingerprint,
        texts,
        validator=FixedValidator("error"),
    )
    assert len(result.errored) == len(selections)
    assert result.first_error == "validator: fixed"
    assert all(record.rewrite_attempts == 1 for record in result.rewrites)


def test_attempts_agree_with_the_rewrite_ledger(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path, plan_document())
    result = execute_rewrites(
        plan,
        selections,
        DeterministicFakeKernel(invalid_every=4),
        fingerprint,
        texts,
        validator=DeterministicFakeValidator(reject_every=3),
    )
    statuses = {item.validator_status for item in result.attempts}
    assert {"valid", "invalid", "not_run"} <= statuses
    by_unit: dict[str, list[Any]] = {}
    for item in result.attempts:
        by_unit.setdefault(item.sampling_unit_id, []).append(item)
    for record in result.rewrites:
        rows = by_unit[record.sampling_unit_id]
        assert [row.attempt for row in rows] == list(range(1, record.rewrite_attempts + 1))
        kept = [row for row in rows if row.accepted]
        if record.is_evaluable():
            assert len(kept) == 1 and kept[0] is rows[-1]
            assert kept[0].prompt_sha256 == record.prompt_sha256
            assert kept[0].validator_status == "valid"
        else:
            assert kept == []
    for item in result.attempts:
        if item.kernel_status != "valid":
            assert item.validator_status == "not_run" and item.prompt_sha256 is None


def test_a_duplicate_is_rejected_before_the_validator_is_asked(tmp_path: Path) -> None:
    @dataclass
    class Repeating:
        kernel_id: str = "repeating"

        def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
            return RewriteOutcome(status="valid", prompt=f"same {request.semantic_seed_id}")

    plan, selections, fingerprint, texts = prepare(tmp_path, plan_document())
    validator = RecordingValidator()
    result = execute_rewrites(
        plan, selections, Repeating(), fingerprint, texts, validator=validator
    )
    assert len(validator.seen) == 3, "one per semantic seed; repeats never reach it"
    duplicates = [item for item in result.attempts if item.duplicate == "exact"]
    assert len(duplicates) == result.duplicates_rejected > 0
    assert {item.validator_status for item in duplicates} == {"not_run"}


def test_without_invoke_nothing_is_recorded(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path, plan_document(invoke=False))
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    assert result.attempts == ()


def write_run_plan(tmp_path: Path, **kwargs: Any) -> Path:
    return materialize(tmp_path, plan_document(**kwargs), SEEDS_WITH_TEXT)


def test_materialize_writes_the_attempt_ledger(tmp_path: Path) -> None:
    plan = write_run_plan(tmp_path)
    run = materialize_run(plan, tmp_path / "run", validator="fake:reject_every=3")
    attempts = read_attempts(tmp_path / "run" / "attempts.csv")
    rewrites = read_rewrites(tmp_path / "run" / "rewrites.csv")
    assert len(attempts) == sum(record.rewrite_attempts for record in rewrites)
    assert sum(item.accepted for item in attempts) == run.accepted
    assert run.validator_id == "fake-validator"
    assert run.validator_mismatch is None
    text = (tmp_path / "run" / "attempts.csv").read_text(encoding="utf-8")
    assert "How do I" not in text, "the attempt ledger carries hashes, never prompt text"


def test_materialize_without_invoke_writes_no_attempt_ledger(tmp_path: Path) -> None:
    plan = write_run_plan(tmp_path, invoke=False)
    materialize_run(plan, tmp_path / "run")
    assert not (tmp_path / "run" / "attempts.csv").exists()


def test_materialize_reports_a_validator_the_plan_does_not_name(tmp_path: Path) -> None:
    plan = write_run_plan(tmp_path, validator_id="semantic-equivalence-v1")
    run = materialize_run(plan, tmp_path / "run", validator="fake")
    assert run.validator_mismatch is not None
    assert "semantic-equivalence-v1" in run.validator_mismatch


def test_the_validator_option_on_the_command_line(tmp_path: Path) -> None:
    plan = write_run_plan(tmp_path)
    output = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generative", "materialize", "--plan", str(plan), "--output", str(output),
         "--validator", "fake:reject_every=4"],
    )
    assert result.exit_code == 0, result.output
    assert "Validator: fake-validator" in plain(result.output)
    assert (output / "attempts.csv").is_file()

    missing = runner.invoke(
        app,
        ["generative", "materialize", "--plan", str(plan), "--output", str(tmp_path / "x")],
    )
    assert missing.exit_code == 3
    assert "pass --validator" in plain(missing.output)


def test_a_validator_from_a_file(tmp_path: Path) -> None:
    source = tmp_path / "my_validator.py"
    source.write_text(
        "from inspect_safety_stats.generative import ValidationVerdict\n"
        "class Strict:\n"
        "    validator_id = 'strict-v1'\n"
        "    def validate(self, request, prompt):\n"
        "        return ValidationVerdict(status='valid')\n"
        "def build():\n"
        "    return Strict()\n",
        encoding="utf-8",
    )
    resolved = resolve_validator(f"{source}:build")
    assert resolved.validator_id == "strict-v1"
    with pytest.raises(ValueError, match="reject_every=N"):
        resolve_validator("fake:sometimes")
    with pytest.raises(ValueError, match="validator"):
        resolve_validator("nonsense")


def test_resume_drops_the_attempts_of_unfinished_units(tmp_path: Path) -> None:
    """An interrupted unit's attempts are on disk before its rewrite row; it runs again."""

    plan = write_run_plan(tmp_path)
    artifacts = tmp_path / "run"

    class Interrupted(RuntimeError):
        pass

    @dataclass
    class Dying:
        survive: int
        validator_id: str = "fake-validator"
        calls: int = 0

        def validate(self, request: RewriteRequest, prompt: str) -> ValidationVerdict:
            self.calls += 1
            if self.calls > self.survive:
                raise Interrupted
            # Odd calls reject, so every unit leaves an attempt row before it finishes.
            return ValidationVerdict(status="invalid" if self.calls % 2 else "valid")

    import inspect_safety_stats.generative.runner as runner_module

    original = runner_module.resolve_validator
    try:
        runner_module.resolve_validator = lambda spec: Dying(survive=41)  # type: ignore[assignment]
        with pytest.raises(Interrupted):
            materialize_run(plan, artifacts, validator="dying")
        partial = read_attempts(artifacts / "attempts.csv")
        finished = {record.sampling_unit_id for record in read_rewrites(artifacts / "rewrites.csv")}
        assert {item.sampling_unit_id for item in partial} - finished, "a unit was cut short"

        runner_module.resolve_validator = (  # type: ignore[assignment]
            lambda spec: Dying(survive=10**6)
        )
        run = materialize_run(plan, artifacts, validator="dying", resume=True)
    finally:
        runner_module.resolve_validator = original  # type: ignore[assignment]

    attempts = read_attempts(artifacts / "attempts.csv")
    rewrites = read_rewrites(artifacts / "rewrites.csv")
    assert run.accepted == len(rewrites) == 60
    assert len(attempts) == sum(record.rewrite_attempts for record in rewrites)


def test_resume_refuses_a_partial_run_without_an_attempt_ledger(tmp_path: Path) -> None:
    unvalidated = write_run_plan(tmp_path, invoke=False)
    artifacts = tmp_path / "run"
    materialize_run(unvalidated, artifacts)
    document = copy.deepcopy(plan_document())
    validated = materialize(tmp_path, document, SEEDS_WITH_TEXT)
    with pytest.raises(ValueError, match="no attempts.csv"):
        materialize_run(validated, artifacts, validator="fake", resume=True)
