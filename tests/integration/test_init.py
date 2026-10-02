from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from inspect_safety_stats import analyze
from inspect_safety_stats.cli import app
from inspect_safety_stats.scaffold import scaffold_plan
from tests.mock_inspect_logs import write_mock_logs


@pytest.fixture(scope="module")
def mock_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_mock_logs(tmp_path_factory.mktemp("mock") / "logs")


def test_init_refuses_to_guess_which_values_are_failures(mock_logs: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--failure-values"):
        scaffold_plan(mock_logs, tmp_path)
    assert not (tmp_path / "safety-plan.yaml").exists()


def test_init_plan_analyzes_end_to_end(mock_logs: Path, tmp_path: Path) -> None:
    result = scaffold_plan(
        mock_logs,
        tmp_path,
        failure_values=["FAIL"],
        group_by="category",
        baseline="mockllm/alpha",
    )
    assert result.cohorts == {"alpha": "mockllm/alpha", "beta": "mockllm/beta"}
    assert result.reference == "alpha"
    assert result.safe_values == ["PASS"]
    assert (result.prompts, result.rollouts_per_prompt) == (5, 4)

    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    assert plan["analysis"]["intent"] == "exploratory"
    assert plan["design"]["hierarchy"] == ["category", "prompt_id"]
    assert plan["comparison_sets"][0]["contrasts"] == {
        "type": "versus_reference",
        "reference": "alpha",
    }
    assert "&id" not in result.plan_path.read_text(encoding="utf-8")

    report = analyze(plan=result.plan_path, logs=str(mock_logs))
    assert report.integrity.status == "valid"
    root = {
        risk.cohort_id: risk
        for risk in report.group_risks
        if risk.group_id == "all" and risk.measure_id == "mean"
    }
    assert set(root) == {"alpha", "beta"}
    for risk in root.values():
        assert risk.estimate == pytest.approx(0.1)
        assert risk.lower is not None and risk.upper is not None
        assert risk.lower <= 0.1 <= risk.upper
    levels = {item.level for item in report.hierarchy_risks}
    assert levels == {"category"}
    overall = [item for item in report.comparisons if item.level is None]
    assert {item.estimand for item in overall} == {"risk_difference", "risk_ratio"}


def test_init_without_baseline_picks_a_reference_and_says_so(
    mock_logs: Path, tmp_path: Path
) -> None:
    result = scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"])
    assert result.reference == "alpha"
    assert any("reference" in note for note in result.notes)
    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    assert plan["design"]["hierarchy"] == ["prompt_id"]
    assert plan["outputs"]["direct_mean_intervals"]["hierarchy_nodes"] is False


def test_init_rejects_unknown_inputs(mock_logs: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="never occurs"):
        scaffold_plan(mock_logs, tmp_path, failure_values=["UNSAFE"])
    with pytest.raises(ValueError, match="no field 'colour'"):
        scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"], group_by="colour")
    with pytest.raises(ValueError, match="baseline 'gamma'"):
        scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"], baseline="gamma")
    with pytest.raises(ValueError, match="no Inspect evaluation logs"):
        scaffold_plan(tmp_path, tmp_path / "out", failure_values=["FAIL"])


def test_init_refuses_two_runs_of_one_model(mock_logs: Path, tmp_path: Path) -> None:
    second = write_mock_logs(tmp_path / "again", models=("mockllm/alpha",), epochs=2)
    with pytest.raises(ValueError, match="more than one successful evaluation per model"):
        scaffold_plan([mock_logs, second], tmp_path / "out", failure_values=["FAIL"])


def test_init_command_writes_files_and_refuses_to_overwrite(
    mock_logs: Path, tmp_path: Path
) -> None:
    arguments = ["init", "--logs", str(mock_logs), "--failure-values", "FAIL",
                 "--output", str(tmp_path)]
    first = CliRunner().invoke(app, arguments)
    assert first.exit_code == 0, first.output
    assert "Cohort alpha: mockllm/alpha (reference)" in first.output
    assert (tmp_path / "design.csv").is_file()
    second = CliRunner().invoke(app, arguments)
    assert second.exit_code == 3
    assert "--overwrite" in second.output
