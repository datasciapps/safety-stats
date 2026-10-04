from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "graded_attempts" / "run_example.py"


def _module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("graded_attempts_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_runs_offline_and_reports_harm_and_attempts(tmp_path: Path) -> None:
    destination = _module().run(tmp_path, epochs=8)
    summary = (destination / "report.md").read_text(encoding="utf-8")
    assert "## Mean harm" in summary
    assert "## Risk over repeated attempts" in summary
    assert "| model-a | all | 4 |" in summary
    assert "model-b vs model-a" in summary
    assert "## By category" in summary
    assert (destination / "report.html").is_file()


@pytest.mark.parametrize("epochs", [3])
def test_example_refuses_fewer_rollouts_than_attempts(tmp_path: Path, epochs: int) -> None:
    destination = _module().run(tmp_path, epochs=epochs)
    summary = (destination / "report.md").read_text(encoding="utf-8")
    assert "ATTEMPT_RISK_INTERVAL_INVALID" in summary
