"""The one-page report.md, and report-full.md beside it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from inspect_safety_stats import analyze
from inspect_safety_stats.registry import CODES
from inspect_safety_stats.scaffold import scaffold_plan
from tests.mock_inspect_logs import write_mock_logs
from tests.report_rendering import rendering_differences


@pytest.fixture(scope="module")
def mock_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_mock_logs(tmp_path_factory.mktemp("summary") / "logs")


def _plan(
    mock_logs: Path,
    directory: Path,
    *,
    group_by: str | None = "category",
    measures: list[dict[str, object]] | None = None,
    **report: object,
) -> Path:
    result = scaffold_plan(
        mock_logs,
        directory,
        failure_values=["FAIL"],
        group_by=group_by,
        baseline="mockllm/alpha",
    )
    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    if report:
        plan.setdefault("report", {}).update(report)
    if measures:
        plan["outputs"]["risk_measures"].extend(measures)
    result.plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return result.plan_path


def _written(plan: Path, logs: Path, output: Path) -> Path:
    written = analyze(plan=plan, logs=str(logs)).write(output)
    assert (written / "report-full.md").is_file()
    return written


def _summary(plan: Path, logs: Path, output: Path) -> str:
    return (_written(plan, logs, output) / "report.md").read_text(encoding="utf-8")


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


def test_the_full_report_keeps_every_construction(mock_logs: Path, tmp_path: Path) -> None:
    written = _written(_plan(mock_logs, tmp_path / "plan"), mock_logs, tmp_path / "report")
    full = (written / "report-full.md").read_text(encoding="utf-8")
    assert "## Fixed-design risk measures" in full
    assert "## Data integrity" in full
    assert "## Limitations" in full
    checksums = json.loads((written / "checksums.json").read_text(encoding="utf-8"))
    assert "report-full.md" in json.dumps(checksums)


def test_imprecise_results_are_flagged_with_a_budget(mock_logs: Path, tmp_path: Path) -> None:
    summary = _summary(_plan(mock_logs, tmp_path / "plan"), mock_logs, tmp_path / "report")
    assert "## Precision" in summary
    assert "†" in summary
    assert "the current rollouts" in summary


def test_a_loose_precision_target_suppresses_flags(mock_logs: Path, tmp_path: Path) -> None:
    plan = _plan(mock_logs, tmp_path / "plan", group_by=None, precision_target=0.49)
    written = _written(plan, mock_logs, tmp_path / "report")
    summary = (written / "report.md").read_text(encoding="utf-8")
    assert "## Precision" not in summary
    assert "†" not in summary
    # The target the plan set is in report.json, so the page renders from it alone (L38).
    stored = json.loads((written / "report.json").read_text(encoding="utf-8"))
    assert stored["presentation"]["precision_target"] == 0.49
    assert rendering_differences(written) == []


def test_the_default_precision_target_is_not_recorded(mock_logs: Path, tmp_path: Path) -> None:
    written = _written(_plan(mock_logs, tmp_path / "plan"), mock_logs, tmp_path / "report")
    stored = json.loads((written / "report.json").read_text(encoding="utf-8"))
    assert "precision_target" not in stored["presentation"]
    assert rendering_differences(written) == []


def test_tail_measures_get_their_own_section(mock_logs: Path, tmp_path: Path) -> None:
    plan = _plan(mock_logs, tmp_path / "plan", measures=[{"id": "worst", "type": "maximum"}])
    summary = _summary(plan, mock_logs, tmp_path / "report")
    assert "## Tail and threshold measures" in summary
    assert "per-prompt bounds that tight need about" in summary


def test_warnings_name_their_code_assumption_and_effect(mock_logs: Path, tmp_path: Path) -> None:
    # A tail measure at the mock logs' few rollouts per prompt warns TAIL_MEASURES_UNINFORMATIVE.
    plan = _plan(mock_logs, tmp_path / "plan", measures=[{"id": "worst", "type": "maximum"}])
    summary = _summary(plan, mock_logs, tmp_path / "report")
    entry = CODES["TAIL_MEASURES_UNINFORMATIVE"]
    warnings = summary.split("## Warnings", 1)[1].split("\n## ", 1)[0]
    row = next(line for line in warnings.splitlines() if "TAIL_MEASURES_UNINFORMATIVE" in line)
    assert entry.assumption in row
    assert f"**{entry.effect}**" in row
    assert entry.consequence in row
    assert "`plan/outputs/risk_measures`" in row
    # The warnings come before the results they qualify.
    assert summary.index("## Warnings") < summary.index("## Failure risk")


def test_a_report_without_warnings_has_no_warnings_block(
    mock_logs: Path, tmp_path: Path
) -> None:
    written = _written(_plan(mock_logs, tmp_path / "plan"), mock_logs, tmp_path / "report")
    stored = json.loads((written / "report.json").read_text(encoding="utf-8"))
    summary = (written / "report.md").read_text(encoding="utf-8")
    shown = [
        item
        for item in stored["integrity"]["diagnostics"]
        if item["severity"] in {"ERROR", "WARNING"}
    ]
    assert ("## Warnings" in summary) == bool(shown)


def test_the_assumptions_used_are_listed(mock_logs: Path, tmp_path: Path) -> None:
    written = _written(_plan(mock_logs, tmp_path / "plan"), mock_logs, tmp_path / "report")
    stored = json.loads((written / "report.json").read_text(encoding="utf-8"))
    summary = (written / "report.md").read_text(encoding="utf-8")
    section = summary.split("## Assumptions used", 1)[1].split("\n## ", 1)[0]
    assert stored["assumptions_used"]
    for assumption in stored["assumptions_used"]:
        assert f"**{assumption['id']}.** {assumption['statement']}" in section
        for setting in assumption["settings"]:
            if setting["default"]:
                assert f"`{setting['path']}`" in section
