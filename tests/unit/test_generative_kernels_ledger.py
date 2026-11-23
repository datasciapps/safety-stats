"""Kernel fingerprinting and execution-ledger invariants."""

from __future__ import annotations

import copy
import errno
import os
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from safety_stats.generative.execution import prompt_store_writer
from safety_stats.generative.kernels import (
    DeterministicFakeKernel,
    RewriteKernel,
    RewriteOutcome,
    RewriteRequest,
    resolve_kernel_fingerprint,
)
from safety_stats.generative.ledger import (
    REWRITE_COLUMNS,
    SELECTION_COLUMNS,
    AttemptRecord,
    RewriteRecord,
    SelectionRecord,
    attempt_ledger_writer,
    prompt_sha256,
    read_rewrites,
    read_selections,
    rewrite_ledger_writer,
    write_rewrites,
    write_selections,
)
from safety_stats.generative.schema import load_generative_plan
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


@pytest.mark.parametrize("label", ["007", "1.0", "true"])
def test_ledgers_round_trip_labels_that_read_as_numbers(tmp_path: Path, label: str) -> None:
    """Read with pandas' inference, `007` became 7 and `true` a boolean: the run's own ledger
    could not be read back."""

    chosen = SelectionRecord(
        **{
            **selection(0, label).model_dump(),
            "node_id": label,
            "semantic_seed_id": label,
            "stratum": label,
            "proposal_id": label,
        }
    )
    rewritten = RewriteRecord(
        **{
            **rewrite(label).model_dump(),
            "rewrite_draw_id": label,
            "rewrite_kernel_id": label,
            "prompt_id": label,
            "source_semantic_seed_id": label,
        }
    )
    write_selections([chosen], tmp_path / "selections.csv")
    write_rewrites([rewritten], tmp_path / "rewrites.csv")
    assert read_selections(tmp_path / "selections.csv") == [chosen]
    assert read_rewrites(tmp_path / "rewrites.csv") == [rewritten]


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


@pytest.mark.parametrize(
    "writer", [rewrite_ledger_writer, attempt_ledger_writer, prompt_store_writer]
)
def test_replacing_a_ledger_writes_what_appending_would(tmp_path: Path, writer) -> None:
    """A resume rewrites a ledger it then appends to, so both must write the same bytes."""

    if writer is rewrite_ledger_writer:
        rows = [rewrite(f"u-{index}").model_dump(mode="json") for index in range(3)]
    elif writer is prompt_store_writer:
        rows = [
            {"sampling_unit_id": f"u-{index}", "prompt_id": f"p-{index}", "prompt": text}
            for index, text in enumerate(["one\rline", "two\r\nlines", 'a "quoted", text'])
        ]
    else:
        rows = [
            AttemptRecord(sampling_unit_id=f"u-{index}", attempt=1, kernel_status="valid")
            .model_dump(mode="json")
            for index in range(3)
        ]
    appended, replaced = tmp_path / "appended.csv", tmp_path / "replaced.csv"
    with writer(appended) as ledger:
        for row in rows:
            ledger.append(row)
    replaced.write_bytes(b"stale rows\n" * 5)
    writer(replaced).replace(iter(rows))
    assert replaced.read_bytes() == appended.read_bytes()
    assert appended.read_bytes().count(b"\n") == 4 + (writer is prompt_store_writer)
    assert b"\r\n" not in appended.read_bytes().replace(b"two\r\nlines", b"")
    assert sorted(item.name for item in tmp_path.iterdir()) == ["appended.csv", "replaced.csv"]


def test_a_replaced_ledger_reaches_the_disk_before_its_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Renamed before its rows were on disk, a ledger could be left empty by a power loss."""

    import safety_stats.generative.ledger as ledger_module

    rows = [rewrite(f"u-{index}").model_dump(mode="json") for index in range(3)]
    expected = tmp_path / "expected.csv"
    with rewrite_ledger_writer(expected) as ledger:
        for row in rows:
            ledger.append(row)
    path, partial = tmp_path / "rewrites.csv", tmp_path / "rewrites.csv.partial"
    path.write_bytes(b"stale rows\n")
    events: list[tuple[str, bool, bool]] = []
    fsync, replace = os.fsync, os.replace

    def syncing(descriptor: int) -> None:
        complete = partial.exists() and partial.read_bytes() == expected.read_bytes()
        events.append(("fsync", complete, stat.S_ISDIR(os.fstat(descriptor).st_mode)))
        fsync(descriptor)

    def replacing(source: Path, target: Path) -> None:
        events.append(("replace", Path(source) == partial, Path(target) == path))
        replace(source, target)

    monkeypatch.setattr(ledger_module.os, "fsync", syncing)
    monkeypatch.setattr(ledger_module.os, "replace", replacing)
    rewrite_ledger_writer(path).replace(rows)
    assert events[:2] == [("fsync", True, False), ("replace", True, True)]
    assert events[2:] == ([] if os.name == "nt" else [("fsync", False, True)])
    assert path.read_bytes() == expected.read_bytes()


@pytest.mark.skipif(os.name == "nt", reason="Windows syncs no directory")
@pytest.mark.parametrize(("code", "raises"), [(errno.EINVAL, False), (errno.EIO, True)])
def test_a_directory_that_cannot_be_synced_fails_only_for_a_real_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, raises: bool
) -> None:
    import safety_stats.generative.ledger as ledger_module

    fsync = os.fsync

    def syncing(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(code, os.strerror(code))
        fsync(descriptor)

    monkeypatch.setattr(ledger_module.os, "fsync", syncing)
    path = tmp_path / "rewrites.csv"
    if raises:
        with pytest.raises(OSError):
            rewrite_ledger_writer(path).replace([])
    else:
        rewrite_ledger_writer(path).replace([])
    assert path.read_bytes().count(b"\n") == 1, "the rename happened either way"
