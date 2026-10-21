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
import sys
import tempfile
from pathlib import Path

from _source_trees import (
    WRONG_SOURCE,
    export_source,
    repository_root,
    resolve_commit,
    run_module,
    source_changed,
)


def _fail(message: str) -> int:
    print(f"prove_red: {message}", file=sys.stderr)
    return 2


def _run(source: Path, hidden: Path | None, pytest_arguments: list[str]) -> int:
    """Run pytest in a child interpreter that imports the packages under ``source``."""
    arguments = ["-p", "no:cacheprovider", *pytest_arguments]
    return run_module(source, "pytest", arguments, hidden=hidden).returncode


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

    repository = repository_root()
    if repository is None:
        return _fail("run this inside a git checkout")
    base = resolve_commit(repository, options.base)
    if base is None:
        return _fail(f"{options.base!r} is not a commit")
    label = f"{options.base} ({base[:10]})"
    source = repository / options.src
    if not source.is_dir():
        return _fail(f"there is no {options.src}/ directory in {repository}")
    if not source_changed(repository, base, options.src):
        return _fail(
            f"{options.src}/ is the same at {label} and in the working tree, so there is no "
            "change to prove. If the change is committed, pass --base with the commit before it."
        )

    with tempfile.TemporaryDirectory(prefix="prove_red-", ignore_cleanup_errors=True) as scratch:
        old_source = export_source(repository, base, options.src, Path(scratch))
        if old_source is None:
            return _fail(f"{options.src}/ does not exist at {label}")
        print(f"prove_red: testing {options.src}/ at {label}; the tests should fail", flush=True)
        red = _run(old_source, source, pytest_arguments)
    if red == WRONG_SOURCE:
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
    if green == WRONG_SOURCE:
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
