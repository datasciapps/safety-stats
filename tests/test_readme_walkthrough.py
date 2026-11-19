"""The README's walkthrough of the offline demo runs as written and shows what the run shows.

The section "Walkthrough: the offline demo" of README.md quotes commands, a Python snippet,
their output and numbers read off the report. This module runs every command and the snippet
of that section in order, on fresh demo logs, and compares what they print and write with what
the section says:

- the ``inspect eval``, ``init``, ``validate`` and ``analyze`` commands, as written, from a
  scratch directory standing in for the repository root (only ``examples/...`` paths are made
  absolute, and ``inspect eval`` runs from the repository root with its ``--log-dir`` moved);
- the output block after ``analyze`` and after the Python snippet, line for line, apart from the
  absolute report paths ``analyze`` prints, which the README shows as relative ones;
- the summary's first line, the table of output files, the index's headline table, the warning
  codes and the numbers the prose quotes.

When a change moves one of these on purpose, update the README and, if the change is to what
is pinned, this module, in the same commit, and say why. ``tests/test_offline_examples.py``
pins the demo script and the demo's own README.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from tests.cli_output import plain

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DEMO = ROOT / "examples" / "demo"
HEADING = "## Walkthrough: the offline demo"

# Two simulated Inspect evaluations of 240 rollouts each.
_TIMEOUT_SECONDS = 600

_FENCE = re.compile(r"^( *)```(\w*)\n(.*?)^\1```$", re.S | re.M)


@dataclass(frozen=True)
class Fence:
    language: str
    body: str


def _section() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(HEADING)
    end = text.index("\n## ", start + len(HEADING))
    return text[start:end]


def _subsection(title: str) -> str:
    section = _section()
    start = section.index(f"### {title}\n")
    end = section.find("\n### ", start + 1)
    return section[start:] if end == -1 else section[start:end]


def _fences(text: str) -> list[Fence]:
    fences = []
    for match in _FENCE.finditer(text):
        indent = len(match[1])
        body = "".join(line[indent:] for line in match[3].splitlines(keepends=True))
        fences.append(Fence(match[2], body))
    return fences


def _commands(body: str) -> list[list[str]]:
    """The commands of a bash block, one per line once continuations are joined."""
    joined = body.replace("\\\n", " ")
    return [shlex.split(line) for line in joined.splitlines() if line.strip()]


def _lines(output: str) -> list[str]:
    """Printed lines without blank lines at either end.

    Inspect prints empty lines to stdout while it reads logs, before anything of this package's
    output; the README leaves them out.
    """
    return output.strip("\n").splitlines()


def _prose(text: str) -> str:
    """Prose with its line breaks folded, so a quoted phrase may span two README lines."""
    return " ".join(text.split())


def _table(text: str, first_header: str) -> list[dict[str, str]]:
    """The rows of the Markdown table whose first column is ``first_header``."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {first_header} |"))
    headers = [cell.strip() for cell in lines[start].strip("|").split("|")]
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def _readme_fences() -> dict[str, list[Fence]]:
    return {
        title: _fences(_subsection(title))
        for title in ("Inputs", "Command line", "Python", "The summary")
    }


# The run ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Walkthrough:
    workspace: Path
    init_output: str
    validate_output: str
    analyze_output: str
    python_output: str

    @property
    def report_dir(self) -> Path:
        return self.workspace / "reports" / "demo"

    def report(self) -> dict[str, object]:
        payload = json.loads((self.report_dir / "report.json").read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        return payload

    def file(self, name: str) -> str:
        return (self.report_dir / name).read_text(encoding="utf-8")


def _absolute_examples(argument: str) -> str:
    return (ROOT / argument).as_posix() if argument.startswith("examples/") else argument


def _invoke(argv: list[str]) -> str:
    assert argv[0] == "inspect-safety-stats", f"README walkthrough runs {argv[0]!r}"
    result = CliRunner().invoke(app, [_absolute_examples(argument) for argument in argv[1:]])
    text = result.output
    assert result.exit_code == 0, f"{shlex.join(argv)} exited {result.exit_code}: {plain(text)}"
    return text


def _inspect_eval(argv: list[str], workspace: Path) -> None:
    """Run the README's ``inspect eval`` from the repository root, logging into ``workspace``."""
    assert argv[:2] == ["inspect", "eval"], argv
    arguments = list(argv[2:])
    log_dir = arguments.index("--log-dir") + 1
    arguments[log_dir] = str(workspace / arguments[log_dir])
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "eval", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=_TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, f"exit {result.returncode}\n{result.stdout}\n{result.stderr}"


@pytest.fixture(scope="module")
def walkthrough(tmp_path_factory: pytest.TempPathFactory) -> Walkthrough:
    """Run the walkthrough's commands and snippet in order, from a scratch directory."""
    fences = _readme_fences()
    workspace = tmp_path_factory.mktemp("walkthrough").resolve()

    (evaluate,), (init,) = (_commands(fence.body) for fence in fences["Inputs"])
    _inspect_eval(evaluate, workspace)

    previous = Path.cwd()
    os.chdir(workspace)
    try:
        init_output = _invoke(init)
        validate, analyze = _commands(fences["Command line"][0].body)
        validate_output = _invoke(validate)
        analyze_output = _invoke(analyze)
        python_output = io.StringIO()
        snippet = fences["Python"][0].body.replace(
            '"examples/demo/plan.yaml"', repr((DEMO / "plan.yaml").as_posix())
        )
        with contextlib.redirect_stdout(python_output):
            exec(compile(snippet, "README.md (walkthrough)", "exec"), {})  # noqa: S102
    finally:
        os.chdir(previous)
    return Walkthrough(
        workspace, init_output, validate_output, analyze_output, python_output.getvalue()
    )


# What the README holds ----------------------------------------------------------------------


def test_the_walkthrough_has_the_blocks_this_module_runs() -> None:
    """Guards the tests below against passing because the README's blocks moved or changed."""
    fences = _readme_fences()
    assert [fence.language for fence in fences["Inputs"]] == ["bash", "bash"]
    assert [fence.language for fence in fences["Command line"]] == ["bash", "text"]
    assert [fence.language for fence in fences["Python"]] == ["python", "text"]
    assert [fence.language for fence in fences["The summary"]] == ["text"]
    assert [command[:2] for command in _commands(fences["Command line"][0].body)] == [
        ["inspect-safety-stats", "validate"],
        ["inspect-safety-stats", "analyze"],
    ]
    assert '"examples/demo/plan.yaml"' in fences["Python"][0].body


# The commands -------------------------------------------------------------------------------


def test_the_logs_are_the_demo_logs_and_init_writes_the_committed_design(
    walkthrough: Walkthrough,
) -> None:
    assert len(list((walkthrough.workspace / "logs" / "demo").glob("*.eval"))) == 2
    written = (walkthrough.workspace / "design.csv").read_bytes()
    assert written == (DEMO / "design.csv").read_bytes()
    assert (walkthrough.workspace / "safety-plan.yaml").is_file()


def test_validate_prints_what_the_walkthrough_says(walkthrough: Walkthrough) -> None:
    assert "`validate` prints `Integrity: valid`." in _prose(_subsection("Command line"))
    assert "Integrity: valid" in plain(walkthrough.validate_output)


def test_analyze_prints_the_block_the_walkthrough_shows(walkthrough: Walkthrough) -> None:
    shown = _readme_fences()["Command line"][1].body
    # analyze prints absolute paths; the README shows them relative to where it ran.
    printed = walkthrough.analyze_output.replace(str(walkthrough.workspace) + os.sep, "")
    printed = printed.replace(os.sep, "/")
    assert _lines(printed) == shown.splitlines()


def test_the_python_snippet_prints_the_block_the_walkthrough_shows(
    walkthrough: Walkthrough,
) -> None:
    shown = _readme_fences()["Python"][1].body
    assert _lines(walkthrough.python_output) == shown.splitlines()
    python_report = walkthrough.workspace / "reports" / "demo-python" / "report.json"
    assert json.loads(python_report.read_text(encoding="utf-8")) == walkthrough.report()


# The report it writes -----------------------------------------------------------------------


def test_the_outputs_table_lists_every_file_analyze_writes(walkthrough: Walkthrough) -> None:
    listed = {
        name
        for row in _table(_subsection("Outputs"), "File")
        for name in re.findall(r"`([^`]+)`", row["File"])
    }
    written = {path.name for path in walkthrough.report_dir.iterdir()}
    assert listed == written


def test_the_summary_opens_with_the_line_the_walkthrough_shows(walkthrough: Walkthrough) -> None:
    (shown,) = _readme_fences()["The summary"]
    first_line = next(
        line for line in walkthrough.file("report.md").splitlines()[1:] if line.strip()
    )
    assert [first_line] == shown.body.splitlines()


def test_the_index_table_is_the_headline_of_index_md(walkthrough: Walkthrough) -> None:
    shown = _table(_subsection("The index"), "Kind")
    written = _table(walkthrough.file("index.md"), "Kind")
    assert len(shown) == 4
    assert [{column: row[column] for column in shown[0]} for row in written] == shown


def test_the_warning_codes_are_the_ones_the_walkthrough_reads(walkthrough: Walkthrough) -> None:
    report = walkthrough.report()
    integrity = report["integrity"]
    assert isinstance(integrity, dict)
    assert integrity["status"] == "valid"
    diagnostics = integrity["diagnostics"]
    assert [(item["code"], item["severity"], item["effect"]) for item in diagnostics] == [
        ("TAIL_MEASURES_UNINFORMATIVE", "WARNING", "recorded")
    ]
    (diagnostic,) = diagnostics
    assert diagnostic["item"] is None
    assert diagnostic["details"]["median_rollouts_per_prompt"] == 20
    assert diagnostic["details"]["rollouts_per_prompt_needed"] == 134
    assert diagnostic["details"]["measures"] == ["cvar_90", "worst"]
    assert "bounded only at 0.291, above the precision target 0.05" in diagnostic["message"]

    reading = _subsection("Reading the warnings")
    assert "**`TAIL_MEASURES_UNINFORMATIVE` (warning, effect recorded).**" in reading
    columns = {row["Column"]: row["In the demo"] for row in _table(reading, "Column")}
    assert columns["Where"] == f"`{diagnostic['scope']}`"
    assert columns["Effect"] == diagnostic["effect"]
    prose = _prose(reading)
    assert "`report.json` at `/integrity/diagnostics/0`" in prose
    assert "a prompt that never fails is still bounded only at 0.291" in prose
    assert "about 134 rollouts per prompt would bring it below" in prose

    warnings = _table(walkthrough.file("report.md"), "Code")
    assert [(row["Code"], row["Where"]) for row in warnings] == [
        ("`TAIL_MEASURES_UNINFORMATIVE` (warning)", "`plan/outputs/risk_measures`")
    ]


# Each entry: a phrase of the walkthrough, a mark of the report.md line it reads off, and what
# that line says.
_QUOTED = [
    (
        "sensitivity beside them (1.1% to 65.3% for the baseline)",
        "| baseline | all | 20.4% |",
        "[1.1%, 65.3%]",
    ),
    (
        "Candidate minus baseline is −19.6 percentage points, from −29.4 to −8.8.",
        "| candidate vs baseline | all |",
        "| -19.6 pts | [-29.4 pts, -8.8 pts] |",
    ),
    (
        "The ratio of the rates is 0.04, from 0.00 to 0.33.",
        "| candidate vs baseline | all |",
        "| 0.04 | [0.00, 0.33] |",
    ),
    (
        "The comparison carries † too.",
        "| candidate vs baseline | all |",
        "candidate fails less †",
    ),
    (
        "the baseline's bio rate is 5.0%, from 0.4% to 18.2%",
        "| bio | 4 |",
        "| bio | 4 | 5.0% [0.4%, 18.2%] †",
    ),
    ("its upper bound there is still 7.4%", "| bio | 4 |", "| 0.0% [0.0%, 7.4%] †"),
    ("its upper bound there is still 7.4%", "| cyber | 4 |", "| 0.0% [0.0%, 7.4%] †"),
    (
        "In bio the difference runs from −18.2 to +7.0 points",
        "| bio | 4 |",
        "[-18.2 pts, +7.0 pts]",
    ),
    (
        "the candidate's `worst` is 0.100 with an interval from 0.002 to 0.446",
        "| candidate | all | worst |",
        "| candidate | all | worst | 0.100 | [0.002, 0.446] †",
    ),
    (
        "would need about 61 rollouts per prompt instead of 20",
        "- baseline · all:",
        "about 61 per prompt",
    ),
]


@pytest.mark.parametrize(
    ("phrase", "mark", "reported"),
    _QUOTED,
    ids=[f"{mark} {phrase[:30]}" for phrase, mark, _ in _QUOTED],
)
def test_each_number_the_walkthrough_quotes_is_in_the_summary(
    walkthrough: Walkthrough, phrase: str, mark: str, reported: str
) -> None:
    assert phrase in _prose(_section()), f"the README no longer says {phrase!r}"
    (line,) = [line for line in walkthrough.file("report.md").splitlines() if mark in line]
    assert reported in line, f"report.md says {line!r}, the README says {phrase!r}"
