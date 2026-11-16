import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from inspect_safety_stats import cli as cli_module
from inspect_safety_stats.cli import app
from tests.cli_output import plain

runner = CliRunner()

SIMPLEQA = Path(__file__).resolve().parents[2] / "examples" / "real_simpleqa"


def _simpleqa_plan(tmp_path: Path, *, score_name: str = "model_graded_qa") -> Path:
    """Copy the bundled offline SimpleQA plan, optionally pointing it at another scorer."""
    shutil.copy(SIMPLEQA / "design.csv", tmp_path / "design.csv")
    text = (SIMPLEQA / "safety-plan.yaml").read_text(encoding="utf-8")
    assert "name: model_graded_qa" in text
    plan = tmp_path / "plan.yaml"
    plan.write_text(text.replace("name: model_graded_qa", f"name: {score_name}"), encoding="utf-8")
    return plan


def _analyze(plan: Path, output: Path, *extra: str) -> tuple[int, str]:
    result = runner.invoke(
        app,
        [
            "analyze",
            "--plan",
            str(plan),
            "--logs",
            str(SIMPLEQA / "simpleqa-8-real.eval"),
            "--output",
            str(output),
            *extra,
        ],
    )
    return result.exit_code, result.stdout


def test_analyze_exits_zero_for_a_valid_analysis(tmp_path: Path) -> None:
    code, stdout = _analyze(_simpleqa_plan(tmp_path), tmp_path / "report")
    assert code == 0
    assert "Integrity: valid" in stdout


def test_analyze_exits_three_but_writes_the_report_when_integrity_is_invalid(
    tmp_path: Path,
) -> None:
    plan = _simpleqa_plan(tmp_path, score_name="nonexistent_scorer")
    code, stdout = _analyze(plan, tmp_path / "report")
    assert code == 3
    assert "Integrity: invalid" in stdout
    assert "SCORE_MISSING" in (tmp_path / "report" / "report.md").read_text(encoding="utf-8")


def test_invalid_integrity_wins_over_decision_exit_codes(tmp_path: Path) -> None:
    plan = _simpleqa_plan(tmp_path, score_name="nonexistent_scorer")
    code, _ = _analyze(plan, tmp_path / "report", "--decision-exit-codes")
    assert code == 3


@pytest.mark.parametrize(("extra", "expected"), [((), 0), (("--decision-exit-codes",), 2)])
def test_allow_invalid_restores_the_completed_analysis_exit_code(
    tmp_path: Path, extra: tuple[str, ...], expected: int
) -> None:
    plan = _simpleqa_plan(tmp_path, score_name="nonexistent_scorer")
    code, _ = _analyze(plan, tmp_path / "report", "--allow-invalid", *extra)
    # An invalid analysis can never PASS, so decision exit codes report INCONCLUSIVE.
    assert code == expected


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
    assert "expected_rollouts" in plain(result.output)


def test_fixed_design_planning_rejects_an_unused_weight_column(tmp_path: Path) -> None:
    """--weight-column only applies to explicit weights; ignoring it silently would mislead."""
    manifest = tmp_path / "design.csv"
    manifest.write_text(
        "prompt_id,question_id,task_id,expected_rollouts,weight\np1,q1,t1,20,1\n",
        encoding="utf-8",
    )
    arguments = ["plan", "fixed-design", "--manifest", str(manifest), "--threshold", "0.5"]
    rejected = runner.invoke(app, [*arguments, "--weight-column", "weight"])
    assert rejected.exit_code != 0
    assert "--weights explicit" in plain(rejected.output)
    accepted = runner.invoke(
        app, [*arguments, "--weights", "explicit", "--weight-column", "weight"]
    )
    assert accepted.exit_code == 0, accepted.output


def test_cli_errors_render_in_tests_as_they_do_on_ci() -> None:
    """tests/conftest.py forces CI's rendering, so raw-text assertions fail locally as well.

    If this fails, typer no longer honours the variable the fixture sets, or something
    rendered CLI output before the fixture applied. Local runs would then pass assertions
    that fail on CI.
    """
    result = runner.invoke(app, ["plan", "zero-failure", "--alpha", "0.05"])
    assert result.exit_code != 0
    assert "\x1b[" in result.output, "typer rendered without colour; see tests/conftest.py"
    assert "Missing option '--threshold'" in plain(result.output)


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
    # A failing script (an import or path problem) is only debuggable from its output.
    output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, output
    for command in ("compile", "materialize", "analyze", "budget"):
        assert command in result.stdout, output


def test_every_command_explains_itself_in_help() -> None:
    """A command without a docstring shows an empty line in --help."""
    import typer.main

    def leaves(command: Any, path: tuple[str, ...]) -> list[tuple[str, str]]:
        # Typer's groups need not subclass the click package's Group, so look for children.
        children = getattr(command, "commands", None)
        if isinstance(children, dict):
            return [
                leaf
                for name, child in sorted(children.items())
                for leaf in leaves(child, (*path, name))
            ]
        return [(" ".join(path), (command.help or "").strip())]

    commands = leaves(typer.main.get_command(app), ())
    # Guards the walk itself: one that stopped at the root would pass without looking.
    assert len(commands) > 10
    assert [name for name, text in commands if not text] == []
