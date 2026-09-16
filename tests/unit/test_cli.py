import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from inspect_safety_stats import cli as cli_module
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


def test_direct_script_invocation_registers_all_commands() -> None:
    """Running cli.py as a script exposes the same commands as the entry point.

    The ``__main__`` guard used to sit above the generative command
    definitions, so ``python cli.py generative --help`` exited successfully
    with an empty command group instead of listing the subcommands.
    """
    assert cli_module.__file__ is not None
    cli_path = Path(cli_module.__file__)
    result = subprocess.run(
        [sys.executable, str(cli_path), "generative", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for command in ("compile", "materialize", "analyze", "budget"):
        assert command in result.stdout
