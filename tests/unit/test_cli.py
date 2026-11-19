import json
import math
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


def test_analyze_prints_the_headline_rates_and_the_summary_path(tmp_path: Path) -> None:
    code, stdout = _analyze(_simpleqa_plan(tmp_path), tmp_path / "report")
    assert code == 0
    # Plain typer.echo lines, uncoloured; plain() would join them into one.
    lines = stdout.splitlines()
    start = lines.index("Failure rate (weighted mean):")
    rates = [line for line in lines[start + 1 :] if line.startswith("  ")]
    assert rates
    assert all(" / " in line and "%" in line for line in rates)
    assert f"Summary: {tmp_path / 'report' / 'report.md'}" in lines


def _invoke_analyze(plan: Path, output: Path) -> Any:
    return runner.invoke(
        app,
        [
            "analyze",
            "--plan",
            str(plan),
            "--logs",
            str(SIMPLEQA / "simpleqa-8-real.eval"),
            "--output",
            str(output),
        ],
    )


def test_analyze_prints_each_warning_code_on_stderr_after_the_headline(tmp_path: Path) -> None:
    from inspect_safety_stats.registry import CODES

    output = tmp_path / "report"
    result = _invoke_analyze(_simpleqa_plan(tmp_path), output)
    assert result.exit_code == 0
    index_md = output / "index.md"
    header = (
        f"Warnings by code, each explained in {index_md} under "
        '"Errors and warnings by code":'
    )
    # Plain typer.echo lines, uncoloured; plain() would join them into one.
    stderr = result.stderr.splitlines()
    assert stderr[0] == header
    codes = ("PLAN_SELF_DECLARED", "TAIL_MEASURES_UNINFORMATIVE")
    assert stderr[1:] == [
        f"  WARNING {code} (1 occurrence): {CODES[code].assumption}" for code in codes
    ]
    assert header not in result.stdout
    # After the headline, before the paths; index.md has the section the header names.
    lines = result.output.splitlines()
    assert (
        lines.index("Failure rate (weighted mean):")
        < lines.index(header)
        < lines.index(f"Summary: {output / 'report.md'}")
    )
    index_text = index_md.read_text(encoding="utf-8")
    assert "\n## Errors and warnings by code\n" in index_text
    assert all(f"| `{code}` | warning | 1 |" in index_text for code in codes)


def test_analyze_prints_errors_by_code_when_integrity_is_invalid(tmp_path: Path) -> None:
    output = tmp_path / "report"
    result = _invoke_analyze(_simpleqa_plan(tmp_path, score_name="nonexistent_scorer"), output)
    assert result.exit_code == 3
    stderr = result.stderr.splitlines()
    assert stderr[0] == (
        f"Errors and warnings by code, each explained in {output / 'index.md'} under "
        '"Errors and warnings by code":'
    )
    assert any(line.startswith("  ERROR SCORE_MISSING (") for line in stderr)


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


def test_fixed_design_planning_needs_only_the_levels_its_weights_name(tmp_path: Path) -> None:
    manifest = tmp_path / "design.csv"
    manifest.write_text("prompt_id,expected_rollouts\np1,20\np2,20\n", encoding="utf-8")
    arguments = ["plan", "fixed-design", "--manifest", str(manifest), "--threshold", "0.5"]
    accepted = runner.invoke(app, [*arguments, "--weights", "equal_prompt"])
    assert accepted.exit_code == 0, accepted.output
    assert "Prompts: 2" in accepted.stdout
    # The default nested hierarchy names task_id and question_id, so they stay required.
    rejected = runner.invoke(app, arguments)
    assert rejected.exit_code != 0
    assert "missing required columns: question_id, task_id" in plain(rejected.output)


def test_comparison_planning_needs_only_the_levels_its_weights_name(tmp_path: Path) -> None:
    manifest = tmp_path / "design.csv"
    manifest.write_text("prompt_id,expected_rollouts\np1,20\np2,20\n", encoding="utf-8")
    arguments = ["plan", "comparison", "--manifest", str(manifest), "--margin", "0.1"]
    accepted = runner.invoke(app, [*arguments, "--weights", "equal_prompt"])
    assert accepted.exit_code == 0, accepted.output
    assert "Prompts: 2" in accepted.stdout
    rejected = runner.invoke(app, arguments)
    assert rejected.exit_code != 0
    assert "missing required columns: question_id, task_id" in plain(rejected.output)


def _comparison_manifest(tmp_path: Path, *, counts: bool = True) -> Path:
    manifest = tmp_path / "design.csv"
    rows = [f"p{index},q{index},t1" + (",100" if counts else "") for index in range(1, 5)]
    header = "prompt_id,question_id,task_id" + (",expected_rollouts" if counts else "")
    manifest.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return manifest


def test_comparison_planning_cli_splits_alpha_and_reports_both_directions(
    tmp_path: Path,
) -> None:
    arguments = [
        "plan",
        "comparison",
        "--manifest",
        str(_comparison_manifest(tmp_path)),
        "--margin",
        "0.05",
        "--power",
        "0.8",
        "--weights",
        "equal_prompt",
    ]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    output = plain(result.output)
    # 800 independent rollouts with range 0.25 / 100 each on the difference; two_sided spends
    # 0.025 on each bound.
    ranges = 800 * (0.25 / 100) ** 2
    gap = math.sqrt(0.5 * ranges * math.log(40)) + math.sqrt(0.5 * ranges * math.log(5))
    assert "Candidate rollouts: 400" in output
    assert "Baseline rollouts: 400" in output
    assert "Independent blocks: 800" in output
    assert f"at or below the margin: {0.05 - gap:.8g}" in output
    assert f"at or above the margin: {0.05 + gap:.8g}" in output
    assert "One-sided alpha per direction: 0.025" in output

    per_direction = runner.invoke(app, [*arguments, "--coverage-mode", "per_direction"])
    assert per_direction.exit_code == 0, per_direction.output
    assert "One-sided alpha per direction: 0.05" in plain(per_direction.output)


def test_comparison_planning_cli_pairs_and_overrides_counts(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "comparison",
            "--manifest",
            str(_comparison_manifest(tmp_path, counts=False)),
            "--margin",
            "-0.02",
            "--candidate-rollouts",
            "30",
            "--baseline-rollouts",
            "20",
            "--independence-unit",
            "pair",
            "--family-size",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    output = plain(result.output)
    assert "Candidate rollouts: 120" in output
    assert "Baseline rollouts: 80" in output
    assert "Independent blocks: 120" in output
    assert "Margin: -0.02" in output
    assert "Local alpha: 0.025" in output
    assert "One-sided alpha per direction: 0.0125" in output


def test_comparison_planning_cli_needs_counts_for_both_arms(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "comparison",
            "--manifest",
            str(_comparison_manifest(tmp_path, counts=False)),
            "--margin",
            "0.05",
            "--candidate-rollouts",
            "30",
        ],
    )
    assert result.exit_code != 0
    assert "--baseline-rollouts" in plain(result.output)


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


def _pilot_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "design.csv"
    rows = [
        f"{category}-{index:02d},q{index},t1,{category},{'' if index == 0 else index}"
        for category in ("a", "b")
        for index in range(6)
    ]
    manifest.write_text(
        "prompt_id,question_id,task_id,category,length\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_pilot_cli_writes_the_frame_with_the_draw(tmp_path: Path) -> None:
    manifest = _pilot_manifest(tmp_path)
    output = tmp_path / "pilot" / "frame.csv"
    result = runner.invoke(
        app,
        [
            "plan",
            "pilot",
            "--manifest",
            str(manifest),
            "--strata",
            "task_id, category",
            "--per-stratum",
            "2",
            "--seed",
            "13",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Strata: 2" in result.stdout
    assert "Pilot prompts: 4 of 12" in result.stdout
    assert "Seed: 13" in result.stdout
    written = output.read_bytes()
    assert b"\r\n" not in written
    lines = written.decode("utf-8").splitlines()
    original = manifest.read_text(encoding="utf-8").splitlines()
    assert lines[0] == original[0] + ",pilot_stratum,pilot_inclusion_probability,pilot_selected"
    # The manifest's own text comes through, including the gap in an integer column.
    assert [line.rsplit(",", 3)[0] for line in lines[1:]] == original[1:]
    assert sum(line.endswith(",True") for line in lines[1:]) == 4
    assert {line.split(",")[-2] for line in lines[1:]} == {"0.3333333333333333"}


def test_pilot_cli_and_python_draw_the_same_prompts(tmp_path: Path) -> None:
    """draw_pilot on the manifest read as its docs say draws what the command draws."""
    import pandas as pd

    from inspect_safety_stats.ingest.pilot import draw_pilot

    manifest = tmp_path / "design.csv"
    manifest.write_text(
        "prompt_id,question_id,task_id\n"
        + "".join(f"{index:04d},q{index},t{index % 2}\n" for index in range(1, 21)),
        encoding="utf-8",
    )
    output = tmp_path / "frame.csv"
    result = runner.invoke(
        app,
        [
            "plan",
            "pilot",
            "--manifest",
            str(manifest),
            "--per-stratum",
            "3",
            "--seed",
            "5",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    written = pd.read_csv(output, dtype=str, keep_default_na=False)
    by_command = set(written.loc[written["pilot_selected"] == "True", "prompt_id"])
    as_written = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    frame = draw_pilot(as_written, per_stratum=3, strata=["task_id"], seed=5)
    assert set(frame.loc[frame["pilot_selected"], "prompt_id"]) == by_command
    assert len(by_command) == 6


def test_pilot_cli_rejects_a_stratum_smaller_than_the_draw(tmp_path: Path) -> None:
    manifest = _pilot_manifest(tmp_path)
    output = tmp_path / "frame.csv"
    result = runner.invoke(
        app,
        [
            "plan",
            "pilot",
            "--manifest",
            str(manifest),
            "--strata",
            "category",
            "--per-stratum",
            "7",
            "--seed",
            "1",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "2 strata hold fewer than 7 prompts" in plain(result.output)
    assert not output.exists()


def test_pilot_cli_will_not_overwrite_the_manifest(tmp_path: Path) -> None:
    manifest = _pilot_manifest(tmp_path)
    before = manifest.read_bytes()
    result = runner.invoke(
        app,
        [
            "plan",
            "pilot",
            "--manifest",
            str(manifest),
            "--per-stratum",
            "1",
            "--seed",
            "1",
            "--output",
            str(manifest),
        ],
    )
    assert result.exit_code != 0
    assert "must not overwrite" in plain(result.output)
    assert manifest.read_bytes() == before


def test_pilot_cli_requires_a_seed(tmp_path: Path) -> None:
    manifest = _pilot_manifest(tmp_path)
    result = runner.invoke(
        app,
        [
            "plan",
            "pilot",
            "--manifest",
            str(manifest),
            "--per-stratum",
            "1",
            "--output",
            str(tmp_path / "frame.csv"),
        ],
    )
    assert result.exit_code != 0
    assert "--seed" in plain(result.output)


@pytest.mark.parametrize(
    ("acceptable_upper", "unacceptable_lower", "decision", "expected"),
    [("0.99", "0.995", "PASS", 0), ("0.01", "0.05", "FAIL", 1)],
)
def test_decision_exit_codes_map_pass_to_zero_and_fail_to_one(
    tmp_path: Path, acceptable_upper: str, unacceptable_lower: str, decision: str, expected: int
) -> None:
    plan = _simpleqa_plan(tmp_path)
    text = plan.read_text(encoding="utf-8")
    for old, new in (("0.25", acceptable_upper), ("0.60", unacceptable_lower)):
        assert f"{old}\n" in text
        text = text.replace(f"{old}\n", f"{new}\n")
    plan.write_text(text, encoding="utf-8")
    code, _ = _analyze(plan, tmp_path / "report", "--decision-exit-codes")
    # Decisions are hidden from the terminal by default, so read the report's.
    report = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    assert report["decision"] == decision
    assert code == expected
