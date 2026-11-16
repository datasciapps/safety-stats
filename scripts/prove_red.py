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
    python scripts/prove_red.py --tree tests/integration/test_live_rewrite_kernel.py

By default only src/ is exported, so tests that read other files by path, such as the
examples a test puts on sys.path, see the working tree in both runs, and a fix there cannot
be proven. With --tree the first run happens in a full export of --base instead, made a git
repository, in which only the tests/ directory comes from the working tree: a fix anywhere
outside tests/ can be proven, in examples/ or scripts/ as well as in src/. Test paths are
then read relative to the same directory inside the export as the one prove_red runs from.

Arguments other than --base, --src, --tree and --tests go to pytest unchanged. Exit status:
0 proven, 1 not proven, 2 could not run.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from _source_trees import (
    WRONG_SOURCE,
    export_source,
    export_tree,
    repository_root,
    resolve_commit,
    run_module,
    source_changed,
    tree_changed,
)


def _fail(message: str) -> int:
    print(f"prove_red: {message}", file=sys.stderr)
    return 2


def _run(
    source: Path, hidden: Path | None, pytest_arguments: list[str], cwd: Path | None = None
) -> int:
    """Run pytest in a child interpreter that imports the packages under ``source``."""
    arguments = ["-p", "no:cacheprovider", *pytest_arguments]
    return run_module(source, "pytest", arguments, hidden=hidden, cwd=cwd).returncode


def _red_in_tree(
    repository: Path, base: str, options: argparse.Namespace, pytest_arguments: list[str]
) -> int | str:
    """The status of the first run, in a full export of ``base``; a string says why it failed."""
    with tempfile.TemporaryDirectory(prefix="prove_red-", ignore_cleanup_errors=True) as scratch:
        tree = Path(scratch) / "tree"
        tree.mkdir()
        failure = export_tree(repository, base, options.tests, tree)
        if failure is not None:
            return failure
        old_source = tree / options.src
        if not old_source.is_dir():
            return f"{options.src}/ does not exist at {options.base} ({base[:10]})"
        # The same place in the export as the caller's in the checkout, so that relative test
        # paths name the same tests.
        here = Path.cwd().resolve()
        try:
            cwd = tree / here.relative_to(repository.resolve())
        except ValueError:
            return "run this from inside the checkout"
        print(
            f"prove_red: testing the tree at {options.base} ({base[:10]}) with the working "
            f"tree's {options.tests}/; the tests should fail",
            flush=True,
        )
        return _run(old_source, repository / options.src, pytest_arguments, cwd)


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
    parser.add_argument(
        "--tree", action="store_true", help="run the tests against all of --base, not only its "
        "source directory, keeping the working tree's tests directory; for a fix outside src/"
    )
    parser.add_argument(
        "--tests", default="tests", help="with --tree, the directory taken from the working "
        "tree, relative to the repository root (default: tests)"
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
    what = f"the tree outside {options.tests}/" if options.tree else f"{options.src}/"
    changed = (
        tree_changed(repository, base, options.tests)
        if options.tree
        else source_changed(repository, base, options.src)
    )
    if not changed:
        hint = ""
        if not options.tree and tree_changed(repository, base, options.tests):
            hint = (
                f" Files outside {options.src}/ differ: pass --tree to prove a fix in "
                "examples/, scripts/ or elsewhere."
            )
        return _fail(
            f"{what} is the same at {label} and in the working tree, so there is no change to "
            "prove. If the change is committed, pass --base with the commit before it." + hint
        )

    if options.tree:
        outcome = _red_in_tree(repository, base, options, pytest_arguments)
        if isinstance(outcome, str):
            return _fail(outcome)
        red = outcome
        against = f"the tree at {label}"
    else:
        with tempfile.TemporaryDirectory(
            prefix="prove_red-", ignore_cleanup_errors=True
        ) as scratch:
            old_source = export_source(repository, base, options.src, Path(scratch))
            if old_source is None:
                return _fail(f"{options.src}/ does not exist at {label}")
            print(
                f"prove_red: testing {options.src}/ at {label}; the tests should fail", flush=True
            )
            red = _run(old_source, source, pytest_arguments)
        against = f"{options.src}/ at {label}"
    if red == WRONG_SOURCE:
        return _fail("could not import the old source in place of the installed package")
    if red == 0:
        print(f"NOT PROVEN: the tests pass against {against}, so they do not detect the change.")
        return 1
    if red not in (1, 2):
        return _fail(f"pytest could not run the tests (exit status {red}); check the selection")

    print("prove_red: testing the working tree; the tests should pass")
    green = _run(source, None, pytest_arguments)
    if green == WRONG_SOURCE:
        return _fail("the package does not import from the working tree")
    if green != 0:
        print("NOT PROVEN: the tests fail with the change as well.")
        return 1
    how = "cannot even be collected" if red == 2 else "fail"
    print(f"PROVEN: the tests {how} against {against} and pass with the change.")
    if red == 2:
        print("A collection error proves less than a failing assertion: check what it was.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
