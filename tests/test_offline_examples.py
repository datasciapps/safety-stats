"""Every example that runs without a provider runs end to end here, as its README tells a reader to.

Three examples need no API key: the simulated demo (``examples/demo``), the eight-sample SimpleQA
log (``examples/real_simpleqa``) and the simulated multilingual generative example
(``examples/generative_multilingual``). The scripts run as a reader runs them, in a fresh
interpreter from a scratch directory, so a broken import, a path that only resolves from the
repository root or a script that no longer matches the package fails here and not on a reader's
machine. The CLI commands run in process, as the other CLI tests do.

Each test pins the exit code, the integrity status of the report and the numbers the README
quotes. When a change moves one of those numbers on purpose, update the README and the
expectation here in the same commit, and say why. ``tests/test_example_reports.py`` pins the
committed SimpleQA report byte for byte; this module adds the commands around it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from tests.cli_output import plain

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo"
SIMPLEQA = ROOT / "examples" / "real_simpleqa"
MULTILINGUAL = ROOT / "examples" / "generative_multilingual"

# The demo runs two Inspect evaluations; the multilingual scripts take about a second each.
_TIMEOUT_SECONDS = 600


def _run_script(script: Path, *arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run an example script in a fresh interpreter, from ``cwd``, as a reader would."""
    environment = dict(os.environ)
    # Windows consoles default to a legacy code page; the examples print plain ASCII, and the
    # Inspect progress display around them must not fail to encode either.
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=_TIMEOUT_SECONDS,
        check=False,
    )


def _completed(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, f"exit {result.returncode}\n{result.stdout}\n{result.stderr}"
    return result.stdout


def _report(directory: Path) -> dict[str, object]:
    payload = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _integrity_status(report: dict[str, object]) -> object:
    integrity = report["integrity"]
    assert isinstance(integrity, dict)
    return integrity["status"]


# The demo -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """Run ``python examples/demo/run_demo.py`` once; it writes logs/demo and reports/demo."""
    workspace = tmp_path_factory.mktemp("demo")
    stdout = _completed(_run_script(DEMO / "run_demo.py", cwd=workspace))
    return workspace, stdout


def test_the_demo_script_writes_a_valid_report(demo_run: tuple[Path, str]) -> None:
    workspace, _ = demo_run
    assert list((workspace / "logs" / "demo").glob("*.eval")), "the demo wrote no Inspect log"
    report = _report(workspace / "reports" / "demo")
    assert _integrity_status(report) == "valid"
    # The demo plan declares no claims, so there is nothing to decide.
    assert report["decision"] == "INCONCLUSIVE"


def test_the_demo_prints_the_numbers_its_readme_quotes(demo_run: tuple[Path, str]) -> None:
    _, stdout = demo_run
    # examples/demo/README.md, "What to look at".
    assert "  bio      0.050 [0.000, 0.312]" in stdout
    assert "  cyber    0.200 [0.000, 0.462]    0.000 [0.000, 0.256]" in stdout
    assert "Candidate minus baseline, all prompts: -0.196 [-0.372, -0.019]" in stdout
    assert "The whole interval is below zero: the candidate fails less often." in stdout


def test_the_demo_steps_one_at_a_time_agree_with_the_script(
    demo_run: tuple[Path, str], tmp_path: Path
) -> None:
    """The README's step-by-step commands, on the logs the script wrote, give the same report."""
    workspace, stdout = demo_run
    logs = workspace / "logs" / "demo"
    plan = DEMO / "plan.yaml"
    runner = CliRunner()

    validated = runner.invoke(app, ["validate", "--plan", str(plan), "--logs", str(logs)])
    assert validated.exit_code == 0, plain(validated.output)
    assert "Integrity: valid" in plain(validated.output)

    output = tmp_path / "demo"
    analyzed = runner.invoke(
        app, ["analyze", "--plan", str(plan), "--logs", str(logs), "--output", str(output)]
    )
    assert analyzed.exit_code == 0, plain(analyzed.output)
    assert _report(output) == _report(workspace / "reports" / "demo")

    shown = _completed(_run_script(DEMO / "show_results.py", str(output), cwd=workspace))
    # The script ends with show(), so its last lines are what show_results.py prints, apart
    # from the report directory it names.
    assert shown.replace(output.as_posix(), "reports/demo") in stdout


# The SimpleQA log ---------------------------------------------------------------------------


def test_the_simpleqa_log_validates() -> None:
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--plan",
            str(SIMPLEQA / "safety-plan.yaml"),
            "--logs",
            str(SIMPLEQA / "simpleqa-8-real.eval"),
        ],
    )
    assert result.exit_code == 0, plain(result.output)
    assert "Integrity: valid" in plain(result.output)


def test_the_simpleqa_decision_is_inconclusive_and_says_so_in_its_exit_code(
    tmp_path: Path,
) -> None:
    """examples/real_simpleqa/README.md: the mean-risk claim is neither PASS nor FAIL."""
    output = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--plan",
            str(SIMPLEQA / "safety-plan.yaml"),
            "--logs",
            str(SIMPLEQA / "simpleqa-8-real.eval"),
            "--output",
            str(output),
            "--decision-exit-codes",
        ],
    )
    # 2 is ExitCode.INCONCLUSIVE in inspect_safety_stats.cli.
    assert result.exit_code == 2, plain(result.output)
    report = _report(output)
    assert _integrity_status(report) == "valid"
    assert report["decision"] == "INCONCLUSIVE"


# The multilingual generative example --------------------------------------------------------


def test_the_multilingual_taxonomy_compiles(tmp_path: Path) -> None:
    stdout = _completed(_run_script(MULTILINGUAL / "compile_tree.py", cwd=tmp_path))
    assert "terminal paths       : 264" in stdout


# examples/generative_multilingual/README.md, "Running it" and "Pairing is where the comparison
# gets its power". Both quote the direct plan with the default seed.
_MULTILINGUAL_README_LINES = (
    "design          : direct, 1200 rewrites",
    "  point estimate      0.03250",
    "  95% upper bound     0.04224",
    "  failures / draws    39 / 1200",
    "nodes reported 390, unsampled 13",
    "  true risk difference   +0.01226",
    "  95% upper bound        +0.02826   covers: True",
    "  discordant units       24 candidate-only, 5 baseline-only, 1171 concordant",
    "  ignoring the pairing   +0.03883   (1.37x wider)",
)


@pytest.mark.parametrize("plan", ["plan-direct.yaml", "plan-stratified.yaml"])
def test_the_multilingual_evaluation_runs_and_covers_its_known_answer(
    plan: str, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    stdout = _completed(
        _run_script(
            MULTILINGUAL / "run_evaluation.py", "--plan", plan, "--output", str(output),
            cwd=tmp_path,
        )
    )
    assert "rewrites        : 1200 accepted, 0 duplicates rejected, 0 exhausted" in stdout
    assert "  true risk           0.03504" in stdout
    assert "  covers the truth    True" in stdout
    for artifact in ("selections.csv", "rewrites.csv", "prompts.restricted.csv"):
        assert (output / artifact).is_file(), artifact
    assert (output / "datasets" / "dataset.jsonl").is_file()
    if plan == "plan-direct.yaml":
        for line in _MULTILINGUAL_README_LINES:
            assert line in stdout, line


def test_the_multilingual_coverage_check_runs(tmp_path: Path) -> None:
    """A few replicates only: this checks that the script runs, not the coverage it measures."""
    stdout = _completed(
        _run_script(MULTILINGUAL / "validate_coverage.py", "--replicates", "3", cwd=tmp_path)
    )
    assert "empirical coverage" in stdout
