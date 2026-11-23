"""Fidelity by arm, reported from the attempt ledger (experimental).

Acceptance by level says how many units ran out of attempts; fidelity says why attempts
failed, so an arm whose rewrites the validator rejects more often is visible even when
retries keep its acceptance level with the others.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from safety_stats.generative.execution import FidelityRow
from safety_stats.generative.kernels import (
    RewriteOutcome,
    RewriteRequest,
    ValidationVerdict,
)
from safety_stats.generative.ledger import read_attempts, read_rewrites, write_attempts
from safety_stats.generative.runner import analyze_run, materialize_run
from tests.generative_fixtures import SEEDS_WITH_TEXT, base_plan, materialize


def plan_path(
    tmp_path: Path,
    *,
    invoke: bool = True,
    validator_id: str = "fake-validator",
    deduplication: dict[str, Any] | None = None,
) -> Path:
    plan = base_plan()
    plan["sampling"] = {"method": "direct", "total_rewrites": 120}
    plan["rewrite"]["validation"] = {"validator_id": validator_id, "invoke": invoke}
    if deduplication is not None:
        plan["rewrite"]["deduplication"] = deduplication
    return materialize(tmp_path, plan, SEEDS_WITH_TEXT)


def stub_logs(monkeypatch: pytest.MonkeyPatch, artifacts: Path) -> None:
    """Stand in for an Inspect run in which every accepted prompt passed."""

    prompt_ids = [
        str(record.prompt_id)
        for record in read_rewrites(artifacts / "rewrites.csv")
        if record.is_evaluable()
    ]
    evals = pd.DataFrame([{"eval_id": "e", "model": "model-a", "status": "success"}])
    samples = pd.DataFrame(
        [
            {"eval_id": "e", "id": prompt_id, "epoch": 1, "score_harmful": "PASS"}
            for prompt_id in prompt_ids
        ]
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.samples_df", lambda logs, **_: samples
    )


@dataclass
class DislikesMaltese:
    """Rejects the first attempt of every Maltese rewrite and nothing else."""

    validator_id: str = "fake-validator"

    def validate(self, request: RewriteRequest, prompt: str) -> ValidationVerdict:
        if request.path_values.get("language") == "maltese" and request.attempt == 1:
            return ValidationVerdict(status="invalid", detail="not equivalent")
        return ValidationVerdict(status="valid")


def run(
    monkeypatch: pytest.MonkeyPatch,
    plan: Path,
    artifacts: Path,
    *,
    validator: Any = None,
    kernel: Any = None,
) -> Any:
    import safety_stats.generative.runner as runner_module

    if validator is not None:
        monkeypatch.setattr(runner_module, "resolve_validator", lambda spec: validator)
    if kernel is not None:
        monkeypatch.setattr(runner_module, "resolve_kernel", lambda spec: kernel)
    materialize_run(plan, artifacts, validator="stub" if validator is not None else None)
    stub_logs(monkeypatch, artifacts)
    return analyze_run(plan, artifacts, "logs")


def codes(report: Any) -> list[str]:
    return [item.code for item in report.diagnostics]


def test_fidelity_is_reported_by_arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = plan_path(tmp_path)
    report = run(monkeypatch, plan, tmp_path / "run", validator=DislikesMaltese())
    rows = {(row.level, row.value): row for row in report.fidelity}
    maltese, english = rows[("language", "maltese")], rows[("language", "english")]
    assert maltese.validator_rejection_rate == pytest.approx(0.5, abs=0)
    assert english.validator_rejection_rate == 0.0
    assert maltese.attempts == 2 * maltese.units
    assert english.attempts == english.units
    assert "GENERATIVE_FIDELITY_UNEVEN" in codes(report)
    # Retries hid it from acceptance: every unit was accepted.
    assert "GENERATIVE_ACCEPTANCE_UNEVEN" not in codes(report)

    output = report.write(tmp_path / "report")
    frame = pd.read_csv(output / "fidelity.csv")
    assert list(frame.columns) == [item.name for item in fields(FidelityRow)]
    document = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(document["fidelity"]) == len(report.fidelity)
    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    assert "fidelity.csv" in checksums


def test_a_plan_without_attempts_has_no_fidelity_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = plan_path(tmp_path, invoke=False)
    report = run(monkeypatch, plan, tmp_path / "run")
    assert report.fidelity is None
    output = report.write(tmp_path / "report")
    assert not (output / "fidelity.csv").exists()
    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    assert "fidelity.csv" not in checksums
    assert "fidelity" not in json.loads((output / "report.json").read_text(encoding="utf-8"))


def test_a_validator_the_plan_does_not_name_is_diagnosed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = plan_path(tmp_path, validator_id="semantic-equivalence-v1")
    report = run(monkeypatch, plan, tmp_path / "run", validator=DislikesMaltese())
    assert "GENERATIVE_VALIDATOR_MISMATCH" in codes(report)


def test_a_matching_validator_is_not_diagnosed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = plan_path(tmp_path)
    report = run(monkeypatch, plan, tmp_path / "run", validator=DislikesMaltese())
    assert "GENERATIVE_VALIDATOR_MISMATCH" not in codes(report)
    assert "GENERATIVE_RETRY_DEPENDENCE" not in codes(report)


@dataclass
class Restyling:
    """A first attempt that restyles one wording per seed; retries reword per unit."""

    kernel_id: str = "test-kernel"

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        if request.attempt == 1:
            text = f"Explain: {request.semantic_seed_id}" + "!" * (
                int(request.path_hash[:2], 16) % 5
            )
            return RewriteOutcome(status="valid", prompt=text.upper())
        reworded = f"Explain {request.semantic_seed_id} for {request.sampling_unit_id}"
        return RewriteOutcome(status="valid", prompt=reworded)


def test_duplicate_retries_are_diagnosed_as_dependence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = plan_path(
        tmp_path, invoke=False, deduplication={"near_duplicate": "normalized_text"}
    )
    report = run(monkeypatch, plan, tmp_path / "run", kernel=Restyling())
    found = [item for item in report.diagnostics if item.code == "GENERATIVE_RETRY_DEPENDENCE"]
    assert len(found) == 1 and found[0].severity == "INFO"
    by_language = [row for row in report.fidelity if row.level == "language"]
    assert found[0].details["units"] == sum(row.retried_after_duplicate for row in by_language)
    # One unit per semantic seed kept its first attempt; every later one was retried.
    assert found[0].details["units"] == found[0].details["evaluable_units"] - 3
    assert sum(row.near_duplicates for row in report.fidelity) > 0


def test_a_missing_attempt_ledger_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = plan_path(tmp_path)
    artifacts = tmp_path / "run"
    import safety_stats.generative.runner as runner_module

    monkeypatch.setattr(runner_module, "resolve_validator", lambda spec: DislikesMaltese())
    materialize_run(plan, artifacts, validator="stub")
    stub_logs(monkeypatch, artifacts)
    (artifacts / "attempts.csv").unlink()
    with pytest.raises(ValueError, match="attempts.csv is missing"):
        analyze_run(plan, artifacts, "logs")


def test_an_attempt_ledger_that_disagrees_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = plan_path(tmp_path)
    artifacts = tmp_path / "run"
    import safety_stats.generative.runner as runner_module

    monkeypatch.setattr(runner_module, "resolve_validator", lambda spec: DislikesMaltese())
    materialize_run(plan, artifacts, validator="stub")
    stub_logs(monkeypatch, artifacts)
    attempts = read_attempts(artifacts / "attempts.csv")
    write_attempts(attempts[1:], artifacts / "attempts.csv")
    with pytest.raises(ValueError, match="disagree"):
        analyze_run(plan, artifacts, "logs")


@dataclass
class FailsEveryFifth:
    """Reports an operational error on every fifth call, and accepts every other rewrite."""

    validator_id: str = "fake-validator"
    calls: int = 0

    def validate(self, request: RewriteRequest, prompt: str) -> ValidationVerdict:
        self.calls += 1
        if self.calls % 5 == 0:
            return ValidationVerdict(status="error", detail="validator unavailable")
        return ValidationVerdict(status="valid")


def test_a_run_resumed_after_validator_errors_can_be_analyzed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resume reruns the errored units; their old rows must not block the analysis."""

    import safety_stats.generative.runner as runner_module

    plan = plan_path(tmp_path)
    artifacts = tmp_path / "run"
    monkeypatch.setattr(runner_module, "resolve_validator", lambda spec: FailsEveryFifth())
    first = materialize_run(plan, artifacts, validator="stub")
    assert first.errored == 24

    monkeypatch.setattr(runner_module, "resolve_validator", lambda spec: DislikesMaltese())
    resumed = materialize_run(plan, artifacts, validator="stub", resume=True)
    assert resumed.errored == 0 and resumed.accepted == 120
    stub_logs(monkeypatch, artifacts)
    report = analyze_run(plan, artifacts, "logs")
    assert report.tree.root.draws == 120
    attempts = read_attempts(artifacts / "attempts.csv")
    assert len(attempts) == sum(
        record.rewrite_attempts for record in read_rewrites(artifacts / "rewrites.csv")
    )
    assert not any(item.validator_status == "error" for item in attempts)


def test_uneven_fidelity_says_how_few_judged_rewrites_its_rates_rest_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No sample-size floor applies, so the warning must not call a spread material."""

    plan = plan_path(tmp_path)
    report = run(monkeypatch, plan, tmp_path / "run", validator=DislikesMaltese())
    found = [
        item
        for item in report.diagnostics
        if item.code == "GENERATIVE_FIDELITY_UNEVEN" and "'language'" in item.message
    ]
    assert len(found) == 1
    fewest = min(row.validator_judged for row in report.fidelity if row.level == "language")
    assert fewest > 0
    message = found[0].message
    assert "materially" not in message
    assert "more than five percentage points apart" in message
    assert f"the fewest rewrites it judged for one value is {fewest}," in message
    assert message.endswith("can arise by chance")


@dataclass
class RejectsEverything:
    """A kernel that never returns a usable rewrite, so no attempt reaches the validator."""

    kernel_id: str = "test-kernel"

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        return RewriteOutcome(status="invalid", detail="never usable")


def test_no_validator_mismatch_when_no_rewrite_reached_the_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing was judged and nothing kept, so there is no validator to compare."""

    import safety_stats.generative.runner as runner_module

    plan = plan_path(tmp_path)
    artifacts = tmp_path / "run"
    monkeypatch.setattr(runner_module, "resolve_kernel", lambda spec: RejectsEverything())
    monkeypatch.setattr(runner_module, "resolve_validator", lambda spec: DislikesMaltese())
    run_summary = materialize_run(plan, artifacts, validator="stub")
    assert run_summary.accepted == 0 and run_summary.exhausted == 120
    # No rewrite was accepted, so the log can only hold a sample that matches none.
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.evals_df",
        lambda logs, **_: pd.DataFrame(
            [{"eval_id": "e", "model": "model-a", "status": "success"}]
        ),
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, **_: pd.DataFrame(
            [{"eval_id": "e", "id": "elsewhere", "epoch": 1, "score_harmful": "PASS"}]
        ),
    )
    report = analyze_run(plan, artifacts, "logs")
    assert report.fidelity is not None
    assert all(row.validator_judged == 0 for row in report.fidelity)
    assert "GENERATIVE_VALIDATOR_MISMATCH" not in codes(report)


class KernelStopped(RuntimeError):
    """Stands in for a kernel whose provider stopped responding."""


@dataclass
class StopsAfter:
    """Returns one rewrite per unit, until it stops after ``survive`` calls."""

    survive: int = 10**6
    kernel_id: str = "test-kernel"
    calls: int = 0

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        self.calls += 1
        if self.calls > self.survive:
            raise KernelStopped
        return RewriteOutcome(status="valid", prompt=f"rewrite of {request.sampling_unit_id}")


def test_kept_rewrites_no_validator_judged_are_a_validator_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plan that set validation.invoke only for the resume validated part of the run."""

    import safety_stats.generative.runner as runner_module

    near = {"near_duplicate": "normalized_text"}
    artifacts = tmp_path / "run"
    unvalidated = plan_path(tmp_path, invoke=False, deduplication=near)
    monkeypatch.setattr(runner_module, "resolve_kernel", lambda spec: StopsAfter(40))
    with pytest.raises(KernelStopped):
        materialize_run(unvalidated, artifacts)

    validated = plan_path(tmp_path, invoke=True, deduplication=near)
    monkeypatch.setattr(runner_module, "resolve_kernel", lambda spec: StopsAfter())
    monkeypatch.setattr(runner_module, "resolve_validator", lambda spec: DislikesMaltese())
    materialize_run(validated, artifacts, validator="stub", resume=True)
    stub_logs(monkeypatch, artifacts)
    report = analyze_run(validated, artifacts, "logs")
    found = [item for item in report.diagnostics if item.code == "GENERATIVE_VALIDATOR_MISMATCH"]
    assert len(found) == 1
    assert found[0].message == (
        "the plan declares validator 'fake-validator', but the attempt ledger records 40 "
        "kept rewrite(s) that no validator judged"
    )


def test_no_validator_mismatch_when_the_plan_only_detects_near_duplicates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every kept rewrite is unjudged here, as the plan says, so there is nothing to warn of."""

    plan = plan_path(tmp_path, invoke=False, deduplication={"near_duplicate": "normalized_text"})
    report = run(monkeypatch, plan, tmp_path / "run")
    assert report.fidelity is not None and report.tree.root.draws == 120
    assert all(row.validator_judged == 0 for row in report.fidelity)
    assert "GENERATIVE_VALIDATOR_MISMATCH" not in codes(report)
