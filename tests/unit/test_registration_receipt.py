"""Plan registration receipts: hash-plan --register writes one, analyze checks it."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
from inspect_safety_stats.config.registration import load_receipt, write_receipt
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
    before = datetime.now(timezone.utc).replace(microsecond=0)

    result = runner.invoke(app, ["hash-plan", "--plan", str(plan), "--register", str(receipt_path)])

    assert result.exit_code == 0, result.output
    digest = canonical_plan_hash(load_plan(plan))
    assert result.output.splitlines()[0] == digest
    receipt = load_receipt(receipt_path)
    assert receipt.plan_sha256 == digest
    assert receipt.registered_at.utcoffset() is not None
    assert before <= receipt.registered_at <= datetime.now(timezone.utc)
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


def test_a_receipt_records_utc_to_the_second(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    local = datetime(2026, 3, 1, 14, 30, 5, 999, tzinfo=timezone.utc).astimezone()
    write_receipt(path, "a" * 64, package_version="0", now=local)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "package_version": "0",
        "plan_sha256": "a" * 64,
        "receipt_version": "1",
        "registered_at": "2026-03-01T14:30:05Z",
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
