"""The committed example reports are exactly what the shipped plan and log produce today.

The example READMEs walk a reader through running ``analyze`` on a bundled plan and log and
then reading the committed report. Those reports once stayed at the package version that
first wrote them while the analysis moved on, so the numbers a reader saw were not the numbers
the command they had just run produced (#27). Reports name their inputs relative to the plan's
directory, so a fresh run writes the same text in any checkout and output directory.

Every artifact is compared as text, exactly, except its floating-point numbers, which agree to
a relative 1e-9 (1e-12 absolute near zero). The last digits of a computed bound depend on the
platform's maths library and solver build, so the reports generated on Linux differ from a
Windows run in those digits only, while a stale report differs by far more. ``checksums.json``
holds hashes of the other files' bytes, so it is checked against the files it lists instead.

When the package version or any output changes, regenerate each report with the command in
its ``analyze`` argument list below, from the repository root and with ``--overwrite``, in a
commit of its own that explains every difference.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest
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


# A floating-point literal standing on its own: not digits inside a hash such as "ae41e4649b",
# an identifier or a version string, which stay text and are compared exactly.
_FLOAT = re.compile(rb"(?<![\w.])(-?\d+(?:\.\d+(?:[eE][+-]?\d+)?|[eE][+-]?\d+))(?![\w.])")
_REL_TOL = 1e-9
_ABS_TOL = 1e-12


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _mismatch(fresh: bytes, committed: bytes) -> str | None:
    """Where two artifacts first differ beyond floating-point noise, or None if they agree."""
    if fresh == committed:
        return None
    fresh_parts = _FLOAT.split(fresh)
    committed_parts = _FLOAT.split(committed)
    offset = 0
    for index, (new, old) in enumerate(zip(fresh_parts, committed_parts, strict=False)):
        if index % 2 == 0 and new != old:
            common = min(len(new), len(old))
            for position, (a, b) in enumerate(zip(new, old, strict=False)):
                if a != b:
                    common = position
                    break
            at = offset + common
            return f"text at byte {at}: {new[common:][:60]!r} != {old[common:][:60]!r}"
        if index % 2 == 1:
            new_value, old_value = float(new), float(old)
            if not math.isclose(new_value, old_value, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
                return f"number at byte {offset}: {new.decode()} != {old.decode()}"
        offset += len(old)
    if len(fresh_parts) == len(committed_parts):
        return None
    fresh_count, committed_count = len(fresh_parts) // 2, len(committed_parts) // 2
    return f"{fresh_count} numbers in the fresh run, {committed_count} committed"


def _assert_checksums_hold(files: dict[str, bytes], where: str) -> None:
    listed = json.loads(files["checksums.json"])
    assert sorted(listed) == sorted(set(files) - {"checksums.json"}), where
    for name, digest in listed.items():
        assert hashlib.sha256(files[name]).hexdigest() == digest, f"{where}: {name}"


def test_report_comparison_forgives_only_the_last_digits_of_numbers() -> None:
    row = b'0.0920692927984616,1e-05,"q-11",3,ae41e4649b,0.0\n'
    assert _mismatch(row, row) is None
    # The last digits, as a Windows maths library may compute them.
    assert _mismatch(row.replace(b"0.0920692927984616", b"0.09206929279846162"), row) is None
    assert _mismatch(row.replace(b",0.0\n", b",-1e-17\n"), row) is None
    # A changed result, count, identifier or hash is a stale report.
    assert _mismatch(row.replace(b"0.0920692927984616", b"0.0920693"), row) is not None
    assert _mismatch(row.replace(b"1e-05", b"1.00001e-05"), row) is not None
    assert _mismatch(row.replace(b",0.0\n", b",1e-09\n"), row) is not None
    assert _mismatch(row.replace(b",3,", b",4,"), row) is not None
    assert _mismatch(row.replace(b"q-11", b"q-12"), row) is not None
    assert _mismatch(row.replace(b"ae41e4649b", b"ae41e4650b"), row) is not None
    assert _mismatch(row + b"0.5\n", row) is not None


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
    _assert_checksums_hold(fresh, "the fresh run's checksums.json")
    _assert_checksums_hold(committed, f"{report}/checksums.json")
    differing = {
        path: reason
        for path in sorted(fresh)
        if path != "checksums.json"
        and (reason := _mismatch(fresh[path], committed[path])) is not None
    }
    assert not differing, f"{report} differs from a fresh run; regenerate the report: {differing}"
