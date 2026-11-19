"""Every example that runs without a provider runs end to end here, as its README tells a reader to.

Four examples need no API key: the simulated demo (``examples/demo``), the simulated harness
comparison (``examples/harness_versions``), the eight-sample SimpleQA log
(``examples/real_simpleqa``) and the simulated multilingual generative example
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
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from tests.cli_output import plain

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo"
HARNESS = ROOT / "examples" / "harness_versions"
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
    # The demo plan, written by init, declares no claims, so there is nothing to decide.
    assert report["decision"] == "INCONCLUSIVE"


def _prose(path: Path) -> str:
    """The page's text with every run of whitespace, line breaks included, as one space."""
    return " ".join(path.read_text(encoding="utf-8").split())


def _interval_row(stdout: str, group: str) -> list[str]:
    """The baseline and candidate estimate, lower and upper bound of one group, as printed."""
    match = re.search(
        rf"^  {group} +(\S+) \[(\S+), (\S+)\] +(\S+) \[(\S+), (\S+)\] *$", stdout, re.M
    )
    assert match, f"no {group} row in the demo output:\n{stdout}"
    return list(match.groups())


def test_the_demo_prints_the_numbers_its_readme_quotes(demo_run: tuple[Path, str]) -> None:
    """The prose of the demo README and of Getting started quotes what the demo prints.

    The numbers are read from the output, not written here, so a change on either side fails.
    """
    _, stdout = demo_run
    readme = _prose(DEMO / "README.md")
    guide = _prose(ROOT / "docs" / "getting-started.md")
    bio = _interval_row(stdout, "bio")
    cyber = _interval_row(stdout, "cyber")
    difference = re.search(
        r"^Candidate minus baseline, all prompts: (\S+) \[(\S+), (\S+)\]$", stdout, re.M
    )
    words = " ".join(stdout.split())
    tail = re.search(r"bounded only at (\d\.\d+),", words)
    needed = re.search(r"About (\d+) rollouts per prompt would be needed\.", words)
    assert difference and tail and needed, stdout
    assert "The whole interval is below zero: the candidate fails less often." in stdout
    # One warning for each model; the script wraps each message over several lines.
    assert stdout.count("WARNING TAIL_MEASURES_UNINFORMATIVE:") == 2
    for cohort in ("baseline", "candidate"):
        assert f"In cohort '{cohort}', group 'all', cvar_90 and worst are built" in words
    assert words.count(f"still bounded only at {tail[1]}") == 2
    assert words.count(f"About {needed[1]} rollouts per prompt would be needed.") == 2

    # examples/demo/README.md, "What to look at".
    assert f"The baseline fails on {float(bio[0]):.0%} of its bio rollouts" in readme
    assert f"the interval runs from {bio[1]} to {bio[2]}" in readme
    assert cyber[3:5] == ["0.000", "0.000"]
    assert "The candidate never fails on a cyber prompt" in readme
    assert f"its upper bound there is still {cyber[5]}" in readme
    estimate, lower, upper = difference.groups()
    assert f"the candidate-minus-baseline difference is {estimate}" in readme
    assert f"with an interval from {lower} to {upper}" in readme
    assert f"bounded only at {tail[1]}" in readme
    assert f"About {needed[1]} rollouts per prompt would be needed" in readme
    # docs/getting-started.md, "Reading the numbers".
    assert f"the baseline fails on {float(bio[0]):.0%} of its 80 bio rollouts" in guide
    assert f"its interval still reaches {float(bio[2]):.2f}" in guide


def test_the_demo_report_keeps_the_comparison_and_warns_about_an_assumption(
    demo_run: tuple[Path, str],
) -> None:
    """The release example shows both configurations compared and at least one warning."""
    workspace, _ = demo_run
    report = _report(workspace / "reports" / "demo")
    comparisons = report["comparisons"]
    assert isinstance(comparisons, list)
    assert {
        (item["comparison_set_id"], item["estimand"]) for item in comparisons if not item["path"]
    } == {("models", "risk_difference"), ("models", "risk_ratio")}
    integrity = report["integrity"]
    assert isinstance(integrity, dict)
    warnings = [item for item in integrity["diagnostics"] if item["severity"] == "WARNING"]
    assert [item["code"] for item in warnings] == ["TAIL_MEASURES_UNINFORMATIVE"] * 2


def test_the_demo_plan_is_the_one_init_writes_plus_two_risk_measures(
    demo_run: tuple[Path, str], tmp_path: Path
) -> None:
    """examples/demo/plan.yaml and design.csv come from init; only the tail measures are added."""
    workspace, _ = demo_run
    result = CliRunner().invoke(
        app,
        [
            "init",
            "--logs",
            str(workspace / "logs" / "demo"),
            "--failure-values",
            "FAIL",
            "--group-by",
            "risk_domain",
            "--baseline",
            "mockllm/baseline",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, plain(result.output)
    assert (tmp_path / "design.csv").read_bytes() == (DEMO / "design.csv").read_bytes()
    written = yaml.safe_load((tmp_path / "safety-plan.yaml").read_text(encoding="utf-8"))
    committed = yaml.safe_load((DEMO / "plan.yaml").read_text(encoding="utf-8"))
    # The analysis id carries the time init ran.
    for plan in (written, committed):
        plan["analysis"].pop("id")
    added = committed["outputs"]["risk_measures"][1:]
    assert added == [
        {"id": "cvar_90", "type": "cvar", "beta": 0.9},
        {"id": "worst", "type": "maximum"},
    ]
    del committed["outputs"]["risk_measures"][1:]
    assert committed == written


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


# The harness-versions example ----------------------------------------------------------------


@pytest.fixture(scope="module")
def harness_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """Run ``python examples/harness_versions/run_example.py`` once, from a scratch directory."""
    workspace = tmp_path_factory.mktemp("harness-versions")
    stdout = _completed(_run_script(HARNESS / "run_example.py", cwd=workspace))
    return workspace, stdout


def test_the_harness_example_prints_the_numbers_its_readme_quotes(
    harness_run: tuple[Path, str],
) -> None:
    workspace, stdout = harness_run
    report = _report(workspace / "reports" / "harness-versions")
    assert _integrity_status(report) == "valid"
    # examples/harness_versions/README.md, "Run it": the block it shows is printed verbatim.
    readme = (HARNESS / "README.md").read_text(encoding="utf-8")
    quoted = readme.split("```text\n", 1)[1].split("```", 1)[0]
    assert quoted in stdout
    assert "v2 minus v1, all prompts: +0.067 [-0.046, +0.176]" in quoted


def test_the_harness_example_steps_one_at_a_time_agree_with_the_script(
    harness_run: tuple[Path, str], tmp_path: Path
) -> None:
    """The README's analyze command, on the logs the script wrote, gives the same report."""
    workspace, _ = harness_run
    output = tmp_path / "harness-versions"
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--plan",
            str(HARNESS / "plan.yaml"),
            "--logs",
            str(workspace / "logs" / "harness-versions"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, plain(result.output)
    assert _report(output) == _report(workspace / "reports" / "harness-versions")


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


# README.md, "What it is for" -----------------------------------------------------------------


def _what_it_is_for() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split("## What it is for\n", 1)[1].split("\n## ", 1)[0]


def _risk(report: dict[str, object], cohort: str) -> str:
    risks = report["group_risks"]
    assert isinstance(risks, list)
    (risk,) = [r for r in risks if r["cohort_id"] == cohort and r["measure_id"] == "mean"]
    return f"{risk['estimate']:.3f} [{risk['lower']:.3f}, {risk['upper']:.3f}]"


def _difference(report: dict[str, object]) -> str:
    comparisons = report["comparisons"]
    assert isinstance(comparisons, list)
    (item,) = [c for c in comparisons if not c["path"] and c["estimand"] == "risk_difference"]
    return f"{item['estimate']:+.3f} [{item['lower']:+.3f}, {item['upper']:+.3f}]"


def test_what_it_is_for_quotes_the_harness_and_demo_numbers(
    harness_run: tuple[Path, str], demo_run: tuple[Path, str]
) -> None:
    section = " ".join(_what_it_is_for().split())
    harness = _report(harness_run[0] / "reports" / "harness-versions")
    assert f"{_risk(harness, 'v1')} and {_risk(harness, 'v2')}" in section
    assert f"v2 minus v1 is {_difference(harness)}" in section
    demo = _report(demo_run[0] / "reports" / "demo")
    assert f"baseline fails at {_risk(demo, 'baseline')}" in section
    assert f"candidate at {_risk(demo, 'candidate')}" in section
    assert f"Candidate minus baseline is {_difference(demo).replace('+', '')}" in section


def test_what_it_is_for_runs_the_threshold_command_as_written(tmp_path: Path) -> None:
    """The SimpleQA command of the section, from a directory standing in for the root."""
    section = _what_it_is_for()
    command = section.split("### Does one system meet a threshold?", 1)[1]
    command = command.split("```bash\n", 1)[1].split("```", 1)[0]
    arguments = command.replace("\\\n", " ").split()
    assert arguments[:2] == ["inspect-safety-stats", "analyze"]
    arguments = [
        str(ROOT / argument) if argument.startswith("examples/") else argument
        for argument in arguments[1:]
    ]
    output = tmp_path / "reports" / "simpleqa"
    arguments[arguments.index("reports/simpleqa")] = str(output)
    result = CliRunner().invoke(app, arguments)
    # 2 is ExitCode.INCONCLUSIVE in inspect_safety_stats.cli, and also Click's code for a usage
    # error, so the report's headline is required as well.
    assert result.exit_code == 2, plain(result.output)
    assert "Integrity: valid" in plain(result.output)
    report = _report(output)
    assert report["decision"] == "INCONCLUSIVE"
    assert f"the mean is {_risk(report, 'candidate')}" in " ".join(section.split())
