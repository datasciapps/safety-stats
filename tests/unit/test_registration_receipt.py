"""Plan registration receipts: hash-plan --register writes one, analyze checks it."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inspect_safety_stats import validate
from inspect_safety_stats.cli import app
from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
from inspect_safety_stats.config.registration import load_receipt, write_receipt
from inspect_safety_stats.ingest.rollout_files import RolloutFiles
from tests.cli_output import plain

runner = CliRunner()

SIMPLEQA = Path(__file__).resolve().parents[2] / "examples" / "real_simpleqa"


def _plan(tmp_path: Path) -> Path:
    shutil.copy(SIMPLEQA / "design.csv", tmp_path / "design.csv")
    shutil.copy(SIMPLEQA / "safety-plan.yaml", tmp_path / "plan.yaml")
    return tmp_path / "plan.yaml"


def test_hash_plan_register_writes_a_receipt_after_the_unchanged_hash_line(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    receipt_path = tmp_path / "plan.receipt.json"
    before = datetime.now(timezone.utc)

    result = runner.invoke(app, ["hash-plan", "--plan", str(plan), "--register", str(receipt_path)])

    after = datetime.now(timezone.utc)
    assert result.exit_code == 0, result.output
    digest = canonical_plan_hash(load_plan(plan))
    assert result.output.splitlines()[0] == digest
    receipt = load_receipt(receipt_path)
    assert receipt.plan_sha256 == digest
    assert receipt.registered_at.utcoffset() is not None
    # Rounded up to the second, so never before the moment the command ran.
    assert receipt.registered_at.microsecond == 0
    assert before <= receipt.registered_at < after + timedelta(seconds=1)
    assert "Anchor the receipt externally" in plain(result.output)
    assert b"\r\n" not in receipt_path.read_bytes()


def test_hash_plan_never_overwrites_a_receipt(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipt_path = tmp_path / "plan.receipt.json"
    receipt_path.write_text("earlier", encoding="utf-8")

    result = runner.invoke(app, ["hash-plan", "--plan", str(plan), "--register", str(receipt_path)])

    assert result.exit_code == 3
    assert "never overwritten" in plain(result.output)
    assert receipt_path.read_text(encoding="utf-8") == "earlier"


def test_hash_plan_without_register_prints_only_the_hash(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    result = runner.invoke(app, ["hash-plan", "--plan", str(plan)])
    assert result.exit_code == 0
    assert result.output == canonical_plan_hash(load_plan(plan)) + "\n"


def test_hash_plan_refuses_to_register_the_inference_hash(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipt_path = tmp_path / "plan.receipt.json"

    result = runner.invoke(
        app,
        ["hash-plan", "--plan", str(plan), "--inference", "--register", str(receipt_path)],
    )

    assert result.exit_code == 2
    assert "omit --inference" in plain(result.output)
    assert not receipt_path.exists()


@pytest.mark.parametrize(
    ("old", "new", "named"),
    [
        ("decimal_places: 6", "decimal_places: true", "report.decimal_places: "),
        ('schema_version: "2.0"', 'schema_version: "2.0"\nunknown_field: 1', "unknown_field: "),
        ('schema_version: "2.0"', 'schema_version: "9.9"', "'9.9'"),
        ('schema_version: "2.0"', 'schema_version: ["unclosed', "Plan invalid: "),
    ],
    ids=["boolean-field", "unknown-field", "schema-version", "yaml-syntax"],
)
def test_hash_plan_reports_an_invalid_plan_on_one_line_and_exits_3(
    tmp_path: Path, old: str, new: str, named: str
) -> None:
    plan = _plan(tmp_path)
    text = plan.read_text(encoding="utf-8")
    assert old in text
    plan.write_text(text.replace(old, new), encoding="utf-8")

    result = runner.invoke(app, ["hash-plan", "--plan", str(plan)])

    assert result.exit_code == 3
    output = plain(result.output)
    assert output.startswith("Plan invalid: ")
    assert named in output
    assert "\n" not in output.strip()
    assert "Traceback" not in output


@pytest.mark.parametrize(
    ("now", "recorded"),
    [
        (datetime(2026, 3, 1, 14, 30, 5, tzinfo=timezone.utc), "2026-03-01T14:30:05Z"),
        (datetime(2026, 3, 1, 14, 30, 5, 1, tzinfo=timezone.utc), "2026-03-01T14:30:06Z"),
        (datetime(2026, 3, 1, 14, 30, 5, 999999, tzinfo=timezone.utc), "2026-03-01T14:30:06Z"),
        (datetime(2026, 3, 1, 23, 59, 59, 500000, tzinfo=timezone.utc), "2026-03-02T00:00:00Z"),
    ],
    ids=["whole-second", "one-microsecond-past", "just-before-the-next", "next-day"],
)
def test_a_receipt_records_utc_rounded_up_to_the_second(
    tmp_path: Path, now: datetime, recorded: str
) -> None:
    # Truncating would record a time before the registration, in the author's favour.
    path = tmp_path / "r.json"
    write_receipt(path, "a" * 64, package_version="0", now=now.astimezone())
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "package_version": "0",
        "plan_sha256": "a" * 64,
        "receipt_version": "1",
        "registered_at": recorded,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"registered_at": "2026-03-01T14:30:05"}, "time zone"),
        ({"plan_sha256": "A" * 64}, "plan_sha256"),
        ({"receipt_version": "2"}, "receipt_version"),
        ({"extra": 1}, "extra"),
    ],
)
def test_a_malformed_receipt_is_rejected(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    receipt = {
        "package_version": "0",
        "plan_sha256": "a" * 64,
        "receipt_version": "1",
        "registered_at": "2026-03-01T14:30:05Z",
        **change,
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_receipt(path)


PLAN = """
schema_version: "2.0"
analysis:
  id: receipt
  title: Registration receipt
  intent: INTENT
  plan_frozen_at: 2026-01-01T00:00:00Z
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
design:
  rollout_key: [run_key, item_id, repeat]
cohorts:
  candidate:
    match: {team: red}
groups:
  - id: all
outputs:
  prompt_bounds:
    enabled: true
    family: displayed
  risk_measures:
    - {id: mean, type: weighted_mean}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
claims: []
"""

STARTED = "2026-02-01T00:00:00Z"


def _source(tmp_path: Path, started_at: str | None = STARTED) -> RolloutFiles:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,2\n", encoding="utf-8"
    )
    run = {"run_key": "r1", "team": "red", "status": "success", "model": "m"}
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps({**run, "started_at": started_at}) + "\n", encoding="utf-8")
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        "".join(
            json.dumps({"run_key": "r1", "item_id": "p1", "repeat": r, "score_harmful": "PASS"})
            + "\n"
            for r in (1, 2)
        ),
        encoding="utf-8",
    )
    return RolloutFiles(runs=runs, rollouts=rollouts)


def _codes(
    tmp_path: Path,
    *,
    registered_at: str | None = "2026-01-15T00:00:00+00:00",
    sha256: str | None = None,
    intent: str = "confirmatory",
    started_at: str | None = STARTED,
) -> dict[str, dict[str, object]]:
    plan = tmp_path / "plan.yaml"
    plan.write_text(PLAN.replace("INTENT", intent), encoding="utf-8")
    source = _source(tmp_path, started_at)
    receipt = None
    if registered_at is not None:
        receipt = tmp_path / "receipt.json"
        receipt.unlink(missing_ok=True)
        write_receipt(
            receipt,
            sha256 or canonical_plan_hash(load_plan(plan)),
            package_version="0",
            now=datetime.fromisoformat(registered_at),
        )
    summary = validate(plan=plan, source=source, registration_receipt=receipt)
    return {
        item.code: {"severity": item.severity, "message": item.message, **item.details}
        for item in summary.diagnostics
    }


def test_without_a_receipt_timing_stays_self_declared(tmp_path: Path) -> None:
    codes = _codes(tmp_path, registered_at=None)
    assert codes["PLAN_SELF_DECLARED"]["severity"] == "WARNING"
    assert not {"PLAN_REGISTERED", "PLAN_RECEIPT_POSTHOC", "PLAN_RECEIPT_MISMATCH"} & set(codes)


def test_a_receipt_that_predates_generation_replaces_the_self_declared_warning(
    tmp_path: Path,
) -> None:
    codes = _codes(tmp_path)
    assert "PLAN_SELF_DECLARED" not in codes
    registered = codes["PLAN_REGISTERED"]
    assert registered["severity"] == "INFO"
    assert registered["registered_at"] == "2026-01-15T00:00:00+00:00"
    assert "before the earliest selected evaluation" in str(registered["message"])


def test_a_receipt_later_than_the_first_generation_is_an_error(tmp_path: Path) -> None:
    codes = _codes(tmp_path, registered_at="2026-02-01T00:00:01+00:00")
    assert codes["PLAN_RECEIPT_POSTHOC"]["severity"] == "ERROR"
    assert not {"PLAN_REGISTERED", "PLAN_SELF_DECLARED"} & set(codes)


def test_a_receipt_at_the_first_generation_time_predates_it(tmp_path: Path) -> None:
    # Registration at the same second as the start is not later than it, as for
    # plan_frozen_at.
    codes = _codes(tmp_path, registered_at="2026-02-01T00:00:00+00:00")
    assert "PLAN_REGISTERED" in codes


@pytest.mark.parametrize(
    "started_at",
    ["2026-02-01T00:00:00.300Z", "2026-02-01T00:00:00Z"],
    ids=["start-to-the-millisecond", "start-truncated-as-inspect-records-it"],
)
def test_a_receipt_written_within_the_second_after_the_first_generation_is_an_error(
    tmp_path: Path, started_at: str
) -> None:
    # Registered 0.6 s after generation started: a receipt truncated to 00:00:00 would read
    # as no later than either start, and so pass.
    codes = _codes(
        tmp_path, registered_at="2026-02-01T00:00:00.900+00:00", started_at=started_at
    )
    posthoc = codes["PLAN_RECEIPT_POSTHOC"]
    assert posthoc["severity"] == "ERROR"
    assert posthoc["registered_at"] == "2026-02-01T00:00:01+00:00"
    assert not {"PLAN_REGISTERED", "PLAN_SELF_DECLARED"} & set(codes)


def test_a_receipt_for_another_plan_is_an_error_and_proves_no_timing(tmp_path: Path) -> None:
    codes = _codes(tmp_path, sha256="0" * 64)
    assert codes["PLAN_RECEIPT_MISMATCH"]["severity"] == "ERROR"
    assert codes["PLAN_SELF_DECLARED"]["severity"] == "WARNING"
    assert "PLAN_REGISTERED" not in codes


def test_a_receipt_for_another_plan_is_an_error_for_an_exploratory_plan_too(
    tmp_path: Path,
) -> None:
    codes = _codes(tmp_path, sha256="0" * 64, intent="exploratory")
    assert codes["PLAN_RECEIPT_MISMATCH"]["severity"] == "ERROR"


def test_a_receipt_says_when_there_is_no_start_time_to_check(tmp_path: Path) -> None:
    codes = _codes(tmp_path, started_at=None)
    assert "START_TIME_UNOBSERVED" in codes
    assert "no selected evaluation reports a start time" in str(
        codes["PLAN_REGISTERED"]["message"]
    )


def test_validate_reports_a_malformed_receipt_instead_of_crashing(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text(PLAN.replace("INTENT", "confirmatory"), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    summary = validate(plan=plan, source=_source(tmp_path), registration_receipt=receipt)
    assert summary.status == "invalid"
    assert any(
        item.code == "INGEST_FAILED" and "registration receipt" in item.message
        for item in summary.diagnostics
    )


@pytest.mark.parametrize("unreadable", ["missing", "directory"])
def test_validate_reports_an_unreadable_receipt_instead_of_crashing(
    tmp_path: Path, unreadable: str
) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text(PLAN.replace("INTENT", "confirmatory"), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    if unreadable == "directory":
        receipt.mkdir()
    summary = validate(plan=plan, source=_source(tmp_path), registration_receipt=receipt)
    assert summary.status == "invalid"
    assert any(
        item.code == "INGEST_FAILED" and "receipt.json" in item.message
        for item in summary.diagnostics
    )


def test_analyze_cli_checks_the_receipt_end_to_end(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipt = tmp_path / "plan.receipt.json"
    write_receipt(
        receipt,
        canonical_plan_hash(load_plan(plan)),
        package_version="0",
        now=datetime(2000, 1, 1, tzinfo=timezone.utc),
    )
    result = runner.invoke(
        app,
        [
            "analyze",
            "--plan",
            str(plan),
            "--logs",
            str(SIMPLEQA / "simpleqa-8-real.eval"),
            "--output",
            str(tmp_path / "report"),
            "--registration-receipt",
            str(receipt),
        ],
    )
    assert result.exit_code == 0, plain(result.output)
    report = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    codes = {item["code"] for item in report["integrity"]["diagnostics"]}
    assert "PLAN_REGISTERED" in codes
    assert "PLAN_SELF_DECLARED" not in codes
