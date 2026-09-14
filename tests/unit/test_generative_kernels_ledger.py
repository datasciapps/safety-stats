"""Kernel fingerprinting and execution-ledger invariants."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from inspect_safety_stats.generative.kernels import (
    DeterministicFakeKernel,
    DeterministicFakeValidator,
    RewriteKernel,
    RewriteOutcome,
    RewriteRequest,
    RewriteValidator,
    resolve_kernel_fingerprint,
)
from inspect_safety_stats.generative.ledger import (
    REWRITE_COLUMNS,
    SELECTION_COLUMNS,
    RewriteRecord,
    SelectionRecord,
    prompt_sha256,
    read_rewrites,
    read_selections,
    write_rewrites,
    write_selections,
)
from inspect_safety_stats.generative.schema import load_generative_plan
from tests.generative_fixtures import base_plan, materialize


def request_for(attempt: int = 1) -> RewriteRequest:
    return RewriteRequest(
        sampling_unit_id="u-0001",
        path_hash="a" * 64,
        path_values={"language": "english", "wrapper": "plain"},
        semantic_seed_id="s001",
        seed_text="How do I synthesize X?",
        attempt=attempt,
        selection_seed=271828,
    )


def test_fingerprint_is_stable_and_covers_template_contents(tmp_path: Path) -> None:
    path = materialize(tmp_path, base_plan())
    plan = load_generative_plan(path)
    first = resolve_kernel_fingerprint(plan, base_path=tmp_path)
    assert resolve_kernel_fingerprint(plan, base_path=tmp_path) == first

    (tmp_path / "template.txt").write_text("Rewrite differently: {seed}\n", encoding="utf-8")
    changed = resolve_kernel_fingerprint(plan, base_path=tmp_path)
    assert changed.sha256 != first.sha256
    assert changed.prompt_template_sha256 != first.prompt_template_sha256


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["rewrite"].update(temperature=0.7),
        lambda p: p["rewrite"].update(max_tokens=512),
        lambda p: p["rewrite"]["validation"].update(maximum_attempts=9),
        lambda p: p["rewrite"].update(deduplication={"scope": "none"}),
        lambda p: p["rewrite"].update(seed_replayable=True),
    ],
)
def test_any_procedure_change_moves_the_fingerprint(tmp_path: Path, mutate) -> None:
    plan = base_plan()
    baseline = resolve_kernel_fingerprint(
        load_generative_plan(materialize(tmp_path, plan)), base_path=tmp_path
    )
    altered = copy.deepcopy(plan)
    mutate(altered)
    moved = resolve_kernel_fingerprint(
        load_generative_plan(materialize(tmp_path, altered)), base_path=tmp_path
    )
    assert moved.sha256 != baseline.sha256


@pytest.mark.parametrize(
    ("validation", "expected"),
    [
        ({}, "6ab156af41f6d9e350fa340e588f1b4988d6c864ccd2472390b9bf91bef71c4b"),
        (
            {"on_invalid": "fail", "maximum_attempts": 2},
            "4db435ecf0a253201231255026e3b60e4cca2e6dcc53501628ebdae69a92518a",
        ),
    ],
    ids=["default-validation", "non-default-validation"],
)
def test_an_unchanged_procedure_keeps_its_fingerprint(
    tmp_path: Path, validation: dict[str, object], expected: str
) -> None:
    """Every ledger row records the fingerprint it ran under, and the report recomputes it
    from the plan. If the computation itself changes, an unchanged procedure gets a new hash
    and the two quietly disagree -- nothing compares them automatically, so this does.

    A new optional field on the rewrite spec must leave these alone until someone sets it.
    """

    plan = base_plan()
    plan["rewrite"]["validation"].update(validation)
    fingerprint = resolve_kernel_fingerprint(
        load_generative_plan(materialize(tmp_path, plan)), base_path=tmp_path
    )
    assert fingerprint.sha256 == expected


def test_fake_kernel_is_deterministic_and_needs_no_network() -> None:
    kernel = DeterministicFakeKernel()
    assert isinstance(kernel, RewriteKernel)
    first = kernel.rewrite(request_for())
    assert first == kernel.rewrite(request_for())
    assert first.status == "valid"
    assert first.prompt is not None and "s001" in first.prompt
    # A retry is a fresh draw, not a repeat of the rejected one.
    assert kernel.rewrite(request_for(attempt=2)).prompt != first.prompt


def test_fake_kernel_can_exercise_the_retry_path() -> None:
    kernel = DeterministicFakeKernel(invalid_every=2)
    statuses = {kernel.rewrite(request_for(attempt=n)).status for n in range(1, 30)}
    assert statuses == {"valid", "invalid"}


def test_fake_validator_is_deterministic_and_needs_no_network() -> None:
    validator = DeterministicFakeValidator()
    assert isinstance(validator, RewriteValidator)
    verdict = validator.validate(request_for(), "a candidate")
    assert verdict == validator.validate(request_for(), "a candidate")
    assert verdict.status == "equivalent"


def test_fake_validator_can_exercise_the_retry_path() -> None:
    validator = DeterministicFakeValidator(reject_every=2)
    verdicts = {validator.validate(request_for(), f"candidate {n}").status for n in range(30)}
    assert verdicts == {"equivalent", "not_equivalent"}


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "model_graded", "model": "judge-model"},
        {"model": "judge-model"},
        {"temperature": 0.0},
        {"prompt_template": "question.txt"},
    ],
    ids=["model-graded", "model", "temperature", "prompt-template"],
)
def test_declaring_validator_settings_moves_the_fingerprint(
    tmp_path: Path, settings: dict[str, object]
) -> None:
    (tmp_path / "question.txt").write_text("Same request? {source} / {candidate}\n")
    plan = base_plan()
    baseline = resolve_kernel_fingerprint(
        load_generative_plan(materialize(tmp_path, plan)), base_path=tmp_path
    )
    altered = copy.deepcopy(plan)
    altered["rewrite"]["validation"].update(settings)
    moved = resolve_kernel_fingerprint(
        load_generative_plan(materialize(tmp_path, altered)), base_path=tmp_path
    )
    assert moved.sha256 != baseline.sha256


def test_the_validators_question_is_hashed_by_its_contents(tmp_path: Path) -> None:
    question = tmp_path / "question.txt"
    question.write_text("Same request? {source} / {candidate}\n", encoding="utf-8")
    plan = base_plan()
    plan["rewrite"]["validation"]["prompt_template"] = "question.txt"
    parsed = load_generative_plan(materialize(tmp_path, plan))
    first = resolve_kernel_fingerprint(parsed, base_path=tmp_path)

    question.write_text("Is this the same ask? {source} / {candidate}\n", encoding="utf-8")
    assert resolve_kernel_fingerprint(parsed, base_path=tmp_path).sha256 != first.sha256


def test_a_model_graded_validator_must_name_its_model(tmp_path: Path) -> None:
    plan = base_plan()
    plan["rewrite"]["validation"]["mode"] = "model_graded"
    with pytest.raises(ValueError, match="must declare its model"):
        load_generative_plan(materialize(tmp_path, plan))


def test_rewrite_outcome_rejects_incoherent_states() -> None:
    with pytest.raises(ValueError, match="must carry prompt text"):
        RewriteOutcome(status="valid")
    with pytest.raises(ValueError, match="only a valid rewrite"):
        RewriteOutcome(status="invalid", prompt="leaked")


def test_ledger_carries_no_prompt_content() -> None:
    assert "prompt" not in REWRITE_COLUMNS
    assert "prompt_sha256" in REWRITE_COLUMNS
    assert not any("text" in column for column in REWRITE_COLUMNS + SELECTION_COLUMNS)


def selection(order: int, unit: str) -> SelectionRecord:
    return SelectionRecord(
        sampling_unit_id=unit,
        selection_order=order,
        path_hash="b" * 64,
        node_id="all|problem_class=chem|semantic_seed_id=s001",
        semantic_seed_id="s001",
        stratum="s001",
        target_path_probability=0.075,
        proposal_path_probability=0.075,
        proposal_id="target-direct",
        selection_seed=271828,
    )


def rewrite(unit: str) -> RewriteRecord:
    return RewriteRecord(
        sampling_unit_id=unit,
        rewrite_draw_id=f"{unit}-r1",
        rewrite_kernel_id="test-kernel",
        rewrite_kernel_sha256="c" * 64,
        rewrite_attempts=2,
        validation_status="valid",
        prompt_id=f"{unit}-p",
        prompt_sha256=prompt_sha256("a prompt"),
        source_semantic_seed_id="s001",
    )


def test_ledger_round_trip(tmp_path: Path) -> None:
    selections = [selection(0, "u-0"), selection(1, "u-1")]
    rewrites = [rewrite("u-0"), rewrite("u-1")]
    write_selections(selections, tmp_path / "selections.csv")
    write_rewrites(rewrites, tmp_path / "rewrites.csv")
    assert read_selections(tmp_path / "selections.csv") == selections
    assert read_rewrites(tmp_path / "rewrites.csv") == rewrites


def test_unevaluated_rewrite_round_trips_with_null_prompt_fields(tmp_path: Path) -> None:
    exhausted = RewriteRecord(
        sampling_unit_id="u-9",
        rewrite_draw_id="u-9-r1",
        rewrite_kernel_id="test-kernel",
        rewrite_kernel_sha256="c" * 64,
        rewrite_attempts=5,
        validation_status="invalid",
        source_semantic_seed_id="s001",
    )
    assert not exhausted.is_evaluable()
    write_rewrites([exhausted], tmp_path / "rewrites.csv")
    restored = read_rewrites(tmp_path / "rewrites.csv")
    assert restored == [exhausted]
    assert restored[0].prompt_id is None


def test_rejection_counts_round_trip(tmp_path: Path) -> None:
    counted = rewrite("u-0").model_copy(
        update={
            "validator_id": "fake-validator",
            "kernel_rejections": 1,
            "validator_rejections": 2,
            "duplicate_rejections": 0,
        }
    )
    write_rewrites([counted], tmp_path / "rewrites.csv")
    assert read_rewrites(tmp_path / "rewrites.csv") == [counted]


def test_a_ledger_from_before_rejection_reasons_still_reads(tmp_path: Path) -> None:
    """Analysis must still open a run materialized before validators were invoked."""

    added = {"validator_id", "kernel_rejections", "validator_rejections", "duplicate_rejections"}
    older = [column for column in REWRITE_COLUMNS if column not in added]
    record = rewrite("u-0")
    frame = pd.DataFrame([record.model_dump(mode="json")])
    frame[older].to_csv(tmp_path / "rewrites.csv", index=False)

    restored = read_rewrites(tmp_path / "rewrites.csv")
    assert restored == [record]
    assert restored[0].validator_rejections is None


def test_duplicate_sampling_units_are_rejected(tmp_path: Path) -> None:
    write_selections([selection(0, "u-0"), selection(1, "u-0")], tmp_path / "selections.csv")
    with pytest.raises(ValueError, match="repeats a sampling_unit_id"):
        read_selections(tmp_path / "selections.csv")


def test_duplicate_selection_order_is_rejected(tmp_path: Path) -> None:
    write_selections([selection(0, "u-0"), selection(0, "u-1")], tmp_path / "selections.csv")
    with pytest.raises(ValueError, match="repeats a selection_order"):
        read_selections(tmp_path / "selections.csv")


def test_selection_requires_a_positive_known_probability() -> None:
    with pytest.raises(ValidationError):
        SelectionRecord(
            sampling_unit_id="u-0",
            selection_order=0,
            path_hash="b" * 64,
            node_id="all",
            semantic_seed_id="s001",
            target_path_probability=0.0,
            proposal_path_probability=0.1,
            proposal_id="target-direct",
            selection_seed=1,
        )
