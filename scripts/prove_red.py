"""Show that tests fail without a source change and pass with it.

A test written for a fix proves nothing unless it fails without the fix, and that is easy to
get wrong: a test once passed against the very precision bug it was written to catch, because
pytest.approx's default absolute tolerance accepted the wrong answer.

This runs the given tests twice: first against the package source as committed at --base
(default HEAD, so the change under test is the uncommitted one), then against the working
tree. It reports PROVEN only if the first run fails and the second passes. The working tree
is never modified: the --base source is exported to a temporary directory and imported from
there.

    python scripts/prove_red.py tests/unit/test_risk.py::test_worst_of_k_exact_value
    python scripts/prove_red.py --base main tests/unit/test_cli.py -q

Arguments other than --base and --src go to pytest unchanged. Exit status: 0 proven, 1 not
proven, 2 could not run. Tests that read source files by path, rather than importing the
package, see the working tree in both runs.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

# The exit status _CHILD uses when a package imports from outside the tree under test.
_WRONG_SOURCE = 97

# Runs in the child interpreter. It hides the working tree's source directory from sys.path
# (an editable install puts it there), checks that every package imports from the tree under
# test, and then runs pytest in the same interpreter, so the check and the tests see the same
# sys.path.
_CHILD = """
import importlib, os, sys

source, hidden, packages, arguments = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]


def key(path):
    return os.path.normcase(os.path.realpath(path or os.curdir))


if hidden:
    sys.path[:] = [entry for entry in sys.path if key(entry) != key(hidden)]
for name in filter(None, packages.split(",")):
    try:
        origin = importlib.import_module(name).__file__
    except Exception as error:  # the tests report it; it does not invalidate the comparison
        print(f"prove_red: importing {name} failed: {error!r}", file=sys.stderr)
        continue
    if not key(origin).startswith(key(source) + os.sep):
        print(f"prove_red: {name} imports from {origin}, not from {source}", file=sys.stderr)
        sys.exit(97)

import pytest

sys.exit(int(pytest.main(arguments)))
"""

# Python 3.12, and the 3.10 and 3.11 security releases, can refuse unsafe archive members.
_EXTRACT: dict[str, Any] = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *arguments], cwd=repository, capture_output=True, check=False)


def _fail(message: str) -> int:
    print(f"prove_red: {message}", file=sys.stderr)
    return 2


def _run(source: Path, hidden: Path | None, pytest_arguments: list[str]) -> int:
    """Run pytest in a child interpreter that imports the packages under ``source``."""
    packages = sorted(path.name for path in source.iterdir() if (path / "__init__.py").is_file())
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (str(source), environment.get("PYTHONPATH")) if entry
    )
    command = [
        sys.executable, "-c", _CHILD, str(source), str(hidden or ""), ",".join(packages),
        "-p", "no:cacheprovider", *pytest_arguments,
    ]
    sys.stdout.flush()
    return subprocess.run(command, env=environment, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show that tests fail without the source change and pass with it.",
        epilog="Other arguments go to pytest. Exit status: 0 proven, 1 not proven, "
        "2 could not run.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--base", default="HEAD", help="commit whose source the tests must fail against "
        "(default: HEAD)"
    )
    parser.add_argument(
        "--src", default="src", help="source directory, relative to the repository root "
        "(default: src)"
    )
    options, pytest_arguments = parser.parse_known_args(argv)
    if not pytest_arguments:
        parser.error("name the tests to run, for example tests/unit/test_risk.py::test_name")

    toplevel = _git(Path.cwd(), "rev-parse", "--show-toplevel")
    if toplevel.returncode:
        return _fail("run this inside a git checkout")
    repository = Path(toplevel.stdout.decode().strip())
    commit = _git(repository, "rev-parse", "--verify", "--quiet", f"{options.base}^{{commit}}")
    if commit.returncode:
        return _fail(f"{options.base!r} is not a commit")
    base = commit.stdout.decode().strip()
    label = f"{options.base} ({base[:10]})"
    source = repository / options.src
    if not source.is_dir():
        return _fail(f"there is no {options.src}/ directory in {repository}")

    changed = _git(repository, "diff", "--quiet", base, "--", options.src).returncode == 1
    untracked = _git(repository, "ls-files", "--others", "--exclude-standard", "--", options.src)
    if not changed and not untracked.stdout.strip():
        return _fail(
            f"{options.src}/ is the same at {label} and in the working tree, so there is no "
            "change to prove. If the change is committed, pass --base with the commit before it."
        )

    with tempfile.TemporaryDirectory(prefix="prove_red-", ignore_cleanup_errors=True) as scratch:
        archive = _git(repository, "archive", "--format=tar", base, "--", options.src)
        if archive.returncode:
            return _fail(f"{options.src}/ does not exist at {label}")
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(scratch, **_EXTRACT)
        print(f"prove_red: testing {options.src}/ at {label}; the tests should fail", flush=True)
        red = _run(Path(scratch) / options.src, source, pytest_arguments)
    if red == _WRONG_SOURCE:
        return _fail("could not import the old source in place of the installed package")
    if red == 0:
        print(
            f"NOT PROVEN: the tests pass against {options.src}/ at {label}, so they do not "
            "detect the change."
        )
        return 1
    if red not in (1, 2):
        return _fail(f"pytest could not run the tests (exit status {red}); check the selection")

    print(f"prove_red: testing {options.src}/ in the working tree; the tests should pass")
    green = _run(source, None, pytest_arguments)
    if green == _WRONG_SOURCE:
        return _fail("the package does not import from the working tree")
    if green != 0:
        print("NOT PROVEN: the tests fail with the change as well.")
        return 1
    how = "cannot even be collected" if red == 2 else "fail"
    print(f"PROVEN: the tests {how} against {options.src}/ at {label} and pass with the change.")
    if red == 2:
        print("A collection error proves less than a failing assertion: check what it was.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
