from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inspect_safety_stats import analyze
from inspect_safety_stats.scaffold import scaffold_plan
from tests.mock_inspect_logs import write_mock_logs


@pytest.fixture(scope="module")
def mock_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_mock_logs(tmp_path_factory.mktemp("summary") / "logs")


def _plan(
    mock_logs: Path, directory: Path, *, group_by: str | None = "category", **report: object
) -> Path:
    result = scaffold_plan(
        mock_logs, directory, failure_values=["FAIL"], group_by=group_by,
        baseline="mockllm/alpha",
    )
    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    if report:
        plan["report"] = report
    result.plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return result.plan_path


def _summary(plan: Path, logs: Path, output: Path) -> str:
    written = analyze(plan=plan, logs=str(logs)).write(output)
    assert (written / "report-full.md").is_file()
    return (written / "report.md").read_text(encoding="utf-8")


def test_summary_covers_risk_comparison_and_breakdown(mock_logs: Path, tmp_path: Path) -> None:
    summary = _summary(_plan(mock_logs, tmp_path / "plan"), mock_logs, tmp_path / "report")
    assert "## Failure risk" in summary
    assert "| alpha | all | 10.0% |" in summary
    assert "## Comparisons" in summary
    assert "beta vs alpha" in summary
    assert "no difference resolved" in summary
    assert "## By category" in summary
    assert "beta − alpha" in summary
    assert "## Tail and threshold measures" not in summary
    assert "## Claims" not in summary


def test_imprecise_results_are_flagged_with_a_budget(mock_logs: Path, tmp_path: Path) -> None:
    summary = _summary(_plan(mock_logs, tmp_path / "plan"), mock_logs, tmp_path / "report")
    assert "## Precision" in summary
    assert "†" in summary
    assert "the current rollouts" in summary


def test_a_loose_precision_target_suppresses_flags(mock_logs: Path, tmp_path: Path) -> None:
    plan = _plan(mock_logs, tmp_path / "plan", group_by=None, precision_target=0.49)
    summary = _summary(plan, mock_logs, tmp_path / "report")
    assert "## Precision" not in summary
    assert "†" not in summary


def test_tail_measures_get_their_own_section(mock_logs: Path, tmp_path: Path) -> None:
    plan_path = _plan(mock_logs, tmp_path / "plan")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan["outputs"]["risk_measures"].append({"id": "worst", "type": "maximum"})
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    summary = _summary(plan_path, mock_logs, tmp_path / "report")
    assert "## Tail and threshold measures" in summary
    assert "per-prompt bounds that tight need about" in summary


def test_tail_measures_warn_when_rollouts_cannot_support_them(
    mock_logs: Path, tmp_path: Path
) -> None:
    plan_path = _plan(mock_logs, tmp_path / "plan")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan["outputs"]["risk_measures"].append({"id": "cvar_90", "type": "cvar", "beta": 0.9})
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    report = analyze(plan=plan_path, logs=str(mock_logs))
    warning = next(
        item for item in report.integrity.diagnostics if item.code == "TAIL_MEASURES_UNINFORMATIVE"
    )
    assert warning.severity == "WARNING"
    assert warning.details["median_rollouts_per_prompt"] == 4
    assert warning.details["rollouts_per_prompt_needed"] > 4
    assert report.integrity.status == "valid"


def test_mean_only_plans_do_not_warn_about_tail_measures(
    mock_logs: Path, tmp_path: Path
) -> None:
    report = analyze(plan=_plan(mock_logs, tmp_path / "plan"), logs=str(mock_logs))
    assert all(item.code != "TAIL_MEASURES_UNINFORMATIVE" for item in report.integrity.diagnostics)
