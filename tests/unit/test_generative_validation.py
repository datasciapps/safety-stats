"""Invoking the rewrite validator, and recording which check rejected a rewrite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_safety_stats.generative.execution import (
    acceptance_by_level,
    execute_rewrites,
    load_seed_texts,
)
from inspect_safety_stats.generative.kernels import (
    DeterministicFakeKernel,
    DeterministicFakeValidator,
    RewriteOutcome,
    RewriteRequest,
    ValidationVerdict,
    resolve_kernel_fingerprint,
)
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy
from tests.generative_fixtures import SEEDS_WITH_TEXT, base_plan, materialize

# The fixture plan's default rewrite.validation.maximum_attempts.
ATTEMPTS = 5


@dataclass
class RepeatingKernel:
    """Returns one prompt per seed, so every later unit of that seed draws a duplicate."""

    kernel_id: str = "repeating"

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        return RewriteOutcome(status="valid", prompt=f"fixed rewrite of {request.semantic_seed_id}")


@dataclass
class CountingValidator:
    """Accepts everything, and keeps what it was shown."""

    validator_id: str = "counting"
    shown: list[str] = field(default_factory=list)

    def validate(self, request: RewriteRequest, candidate: str) -> ValidationVerdict:
        self.shown.append(candidate)
        return ValidationVerdict(status="equivalent")


@dataclass
class UnreachableValidator:
    """Stands in for a validator whose provider is down."""

    validator_id: str = "unreachable"

    def validate(self, request: RewriteRequest, candidate: str) -> ValidationVerdict:
        return ValidationVerdict(status="error", detail="provider unavailable")


def prepare(tmp_path: Path, **validation: Any) -> tuple[Any, ...]:
    plan = base_plan()
    plan["sampling"] = {"method": "direct", "total_rewrites": 60}
    plan["rewrite"]["validation"].update(validation)
    parsed = load_generative_plan(materialize(tmp_path, plan, SEEDS_WITH_TEXT))
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    selections = draw_selections(taxonomy, plan_design(taxonomy, parsed))
    fingerprint = resolve_kernel_fingerprint(parsed, base_path=tmp_path)
    return parsed, selections, fingerprint, load_seed_texts(parsed, base_path=tmp_path)


def counts(record: Any) -> tuple[int, int, int]:
    return (record.kernel_rejections, record.validator_rejections, record.duplicate_rejections)


def test_a_validator_rejection_is_recorded_as_one(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(
        plan, selections, DeterministicFakeKernel(), fingerprint, texts,
        validator=DeterministicFakeValidator(reject_every=1),
    )
    assert not result.accepted
    assert len(result.exhausted) == len(selections)
    assert result.validator_rejected == ATTEMPTS * len(selections)
    for record in result.rewrites:
        assert record.validation_status == "invalid"
        assert record.validator_id == "fake-validator"
        assert counts(record) == (0, ATTEMPTS, 0)


def test_duplicate_exhaustion_is_told_apart_from_validator_rejection(tmp_path: Path) -> None:
    """Both used to end as a bare "invalid"; only the counts can tell them apart."""

    plan, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(
        plan, selections, RepeatingKernel(), fingerprint, texts, validator=CountingValidator()
    )
    accepted = {item.sampling_unit_id for item in result.accepted}
    exhausted = [record for record in result.rewrites if record.sampling_unit_id not in accepted]
    assert exhausted
    assert all(record.validation_status == "invalid" for record in exhausted)
    assert all(counts(record) == (0, 0, ATTEMPTS) for record in exhausted)


def test_duplicates_never_reach_the_validator(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path)
    validator = CountingValidator()
    result = execute_rewrites(
        plan, selections, RepeatingKernel(), fingerprint, texts, validator=validator
    )
    # One accepted rewrite per seed, and those are the only candidates it was shown.
    assert len(result.accepted) == 3
    assert len(validator.shown) == 3


def test_the_kernels_own_rejection_is_counted_separately(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path)
    validator = CountingValidator()
    result = execute_rewrites(
        plan, selections, DeterministicFakeKernel(invalid_every=1), fingerprint, texts,
        validator=validator,
    )
    assert validator.shown == []
    assert result.kernel_rejected == ATTEMPTS * len(selections)
    assert all(counts(record) == (ATTEMPTS, 0, 0) for record in result.rewrites)


def test_a_validator_error_is_operational_not_a_rejection(tmp_path: Path) -> None:
    """An unreachable validator says nothing about the rewrite, so the unit runs again."""

    plan, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(
        plan, selections, DeterministicFakeKernel(), fingerprint, texts,
        validator=UnreachableValidator(),
    )
    assert len(result.errored) == len(selections)
    assert not result.exhausted
    assert all(record.validation_status == "error" for record in result.rewrites)
    assert result.first_error == "validator: provider unavailable"


def test_on_invalid_fail_stops_at_the_first_validator_rejection(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path, on_invalid="fail")
    result = execute_rewrites(
        plan, selections, DeterministicFakeKernel(), fingerprint, texts,
        validator=DeterministicFakeValidator(reject_every=1),
    )
    assert all(record.rewrite_attempts == 1 for record in result.rewrites)
    assert all(counts(record) == (0, 1, 0) for record in result.rewrites)


def test_without_a_validator_the_kernel_is_trusted_and_the_ledger_says_so(
    tmp_path: Path,
) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    assert len(result.accepted) == len(selections)
    assert all(record.validator_id is None for record in result.rewrites)
    assert all(counts(record) == (0, 0, 0) for record in result.rewrites)


def test_acceptance_splits_rejections_by_check(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(
        plan, selections, DeterministicFakeKernel(), fingerprint, texts,
        validator=DeterministicFakeValidator(reject_every=1),
    )
    rows = acceptance_by_level(selections, list(result.rewrites))
    assert rows
    for row in rows:
        assert row.validator_rejections == ATTEMPTS * row.units
        assert row.kernel_rejections == 0
        assert row.duplicate_rejections == 0


def test_acceptance_does_not_invent_counts_for_an_older_ledger(tmp_path: Path) -> None:
    """A run from before rejections were recorded reports them as unknown, not as zero."""

    plan, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    older = [
        record.model_copy(
            update={
                "kernel_rejections": None,
                "validator_rejections": None,
                "duplicate_rejections": None,
            }
        )
        for record in result.rewrites
    ]
    rows = acceptance_by_level(selections, older)
    assert rows
    assert all(row.validator_rejections is None for row in rows)
    assert all(row.accepted == row.units for row in rows)
