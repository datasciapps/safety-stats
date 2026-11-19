"""The committed example reports are exactly what the shipped plan and log produce today.

The example READMEs walk a reader through running ``analyze`` on a bundled plan and log and
then reading the committed report. Those reports once stayed at the package version that
first wrote them while the analysis moved on, so the numbers a reader saw were not the numbers
the command they had just run produced (#27). Reports name their inputs relative to the plan's
directory, so a fresh run writes the same bytes in any checkout and output directory, and
every artifact is compared byte for byte.

When the package version or any output changes, regenerate each report with the command in
its ``analyze`` argument list below, from the repository root and with ``--overwrite``, in a
commit of its own that explains every difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from inspect_safety_stats import __version__
from inspect_safety_stats.cli import app
from tests.cli_output import plain

ROOT = Path(__file__).resolve().parents[1]

# (committed report directory, plan, logs), all relative to the repository root.
EXAMPLE_REPORTS = {
    "simpleqa": (
        "examples/real_simpleqa/report",
        "examples/real_simpleqa/safety-plan.yaml",
        "examples/real_simpleqa/simpleqa-8-real.eval",
    ),
    "xstest": (
        "examples/real_xstest/report-current",
        "examples/real_xstest/safety-plan.yaml",
        "examples/real_xstest/current-log",
    ),
}


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_every_committed_example_report_is_pinned() -> None:
    committed = {
        path.parent.relative_to(ROOT).as_posix()
        for path in (ROOT / "examples").rglob("report.json")
    }
    assert committed == {report for report, _, _ in EXAMPLE_REPORTS.values()}


@pytest.mark.parametrize("name", sorted(EXAMPLE_REPORTS))
def test_committed_example_report_was_written_by_this_version(name: str) -> None:
    report, _, _ = EXAMPLE_REPORTS[name]
    payload = json.loads((ROOT / report / "report.json").read_text(encoding="utf-8"))
    assert payload["package_version"] == __version__, (
        f"{report} was written by {payload['package_version']}; regenerate it with {__version__}"
    )


@pytest.mark.parametrize("name", sorted(EXAMPLE_REPORTS))
def test_committed_example_report_follows_the_report_schema(name: str) -> None:
    """`inspect-safety-stats schema report` is the contract for report.json (CONTRIBUTING 4.3).

    The schema left out `summary`, which every report.json holds, and allows no field it does
    not list, so no report the package wrote validated against it.
    """

    result = CliRunner().invoke(app, ["schema", "report"])
    assert result.exit_code == 0, result.output
    schema = json.loads(result.stdout)
    Draft202012Validator.check_schema(schema)
    report, _, _ = EXAMPLE_REPORTS[name]
    payload = json.loads((ROOT / report / "report.json").read_text(encoding="utf-8"))

    assert "summary" in payload
    errors = [
        f"{error.json_path}: {error.message}"
        for error in Draft202012Validator(schema).iter_errors(payload)
    ]
    assert errors == []


@pytest.mark.parametrize("name", sorted(EXAMPLE_REPORTS))
def test_committed_example_report_matches_a_fresh_run(name: str, tmp_path: Path) -> None:
    report, plan, logs = EXAMPLE_REPORTS[name]
    output = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--plan",
            str(ROOT / plan),
            "--logs",
            str(ROOT / logs),
            "--output",
            str(output),
            "--no-decision-exit-codes",
        ],
    )
    assert result.exit_code == 0, plain(result.output)

    fresh = _files(output)
    committed = _files(ROOT / report)
    assert sorted(committed) == sorted(fresh), f"{report} does not hold the files a run writes"
    differing = [path for path in fresh if fresh[path] != committed[path]]
    assert not differing, f"{report}: {differing} differ from a fresh run; regenerate the report"
