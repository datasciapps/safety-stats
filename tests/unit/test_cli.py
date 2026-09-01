from pathlib import Path

from typer.testing import CliRunner

from inspect_safety_stats.cli import app

runner = CliRunner()


def test_zero_failure_cli() -> None:
    result = runner.invoke(
        app,
        ["plan", "zero-failure", "--threshold", "0.01", "--alpha", "0.05"],
    )
    assert result.exit_code == 0
    assert "Required rollouts: 299" in result.stdout


def test_schema_cli_emits_json() -> None:
    result = runner.invoke(app, ["schema", "report"])
    assert result.exit_code == 0
    assert '"SafetyReport"' in result.stdout


def test_certification_planning_cli() -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "certification",
            "--threshold",
            "0.1",
            "--anticipated-rate",
            "0.02",
            "--power",
            "0.8",
        ],
    )
    assert result.exit_code == 0
    assert "Maximum certifying failures" in result.stdout
    assert "Maximum certifiable failure rate at target power" in result.stdout
    assert "Minimum detectable failure rate at target power" in result.stdout


def test_detectable_rate_cli() -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "detectable-rate",
            "--trials",
            "200",
            "--threshold",
            "0.1",
            "--alpha",
            "0.05",
            "--power",
            "0.8",
        ],
    )
    assert result.exit_code == 0
    assert "Maximum certifiable failure rate" in result.stdout
    assert "Minimum detectable failure rate" in result.stdout
    assert "Support-side detectable gap" in result.stdout
    assert "Rejection-side detectable gap" in result.stdout


def test_cli_help_lists_visualization_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "visualize" in result.stdout
    assert "analyze" in result.stdout
    assert "plan" in result.stdout


def test_zero_failure_family_size_cli() -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "zero-failure",
            "--threshold",
            "0.1",
            "--alpha",
            "0.05",
            "--family-size",
            "8",
        ],
    )
    assert result.exit_code == 0
    assert "Required rollouts: 49" in result.stdout
    assert "Local alpha: 0.00625" in result.stdout


def test_fixed_design_planning_cli(tmp_path: Path) -> None:
    manifest = tmp_path / "design.csv"
    manifest.write_text(
        "prompt_id,question_id,task_id,expected_rollouts\n"
        "p1,q1,t1,20\n"
        "p2,q1,t1,20\n"
        "p3,q2,t1,20\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "plan",
            "fixed-design",
            "--manifest",
            str(manifest),
            "--threshold",
            "0.5",
            "--weights",
            "equal_prompt",
        ],
    )
    assert result.exit_code == 0
    assert "Prompts: 3" in result.stdout
    assert "Total rollouts: 60" in result.stdout
    assert "Effective sample size: 60" in result.stdout
    assert "Maximum certifiable failure rate at target power:" in result.stdout


def test_fixed_design_planning_requires_rollout_source(tmp_path: Path) -> None:
    manifest = tmp_path / "design.csv"
    manifest.write_text(
        "prompt_id,question_id,task_id\np1,q1,t1\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["plan", "fixed-design", "--manifest", str(manifest), "--threshold", "0.5"],
    )
    assert result.exit_code != 0
    assert "expected_rollouts" in result.output
