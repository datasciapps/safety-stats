"""Run a module against the package source of a commit, or of the working tree.

Shared by the scripts in this directory. A commit's source is exported to a temporary
directory with git archive and imported from there, so the working tree is never modified.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

# The exit status of a child whose packages import from outside the tree under test.
WRONG_SOURCE = 97

# Runs in the child interpreter and behaves like ``python -m MODULE ARGS``. Before running the
# module it hides one directory from sys.path (an editable install puts the working tree's
# source there) and checks that every package imports from the tree under test. The check and
# the module run in the same interpreter, so they see the same sys.path.
_CHILD = """
import importlib, os, runpy, sys

source, hidden, packages, module = sys.argv[1:5]
del sys.argv[1:5]


def key(path):
    return os.path.normcase(os.path.realpath(path or os.curdir))


if hidden:
    sys.path[:] = [entry for entry in sys.path if key(entry) != key(hidden)]
for name in filter(None, packages.split(",")):
    try:
        origin = importlib.import_module(name).__file__
    except Exception as error:  # the module reports it; it does not invalidate the comparison
        print(f"{name} failed to import: {error!r}", file=sys.stderr)
        continue
    if not key(origin).startswith(key(source) + os.sep):
        print(f"{name} imports from {origin}, not from {source}", file=sys.stderr)
        sys.exit(97)
runpy.run_module(module, run_name="__main__", alter_sys=True)
"""

# Python 3.12, and the 3.10 and 3.11 security releases, can refuse unsafe archive members.
_EXTRACT: dict[str, Any] = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}


def git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *arguments], cwd=repository, capture_output=True, check=False)


def repository_root() -> Path | None:
    """The top of the git checkout that holds the current directory."""
    toplevel = git(Path.cwd(), "rev-parse", "--show-toplevel")
    return None if toplevel.returncode else Path(toplevel.stdout.decode().strip())


def resolve_commit(repository: Path, ref: str) -> str | None:
    commit = git(repository, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    return None if commit.returncode else commit.stdout.decode().strip()


def source_changed(repository: Path, commit: str, src: str) -> bool:
    """Whether ``src`` in the working tree, untracked files included, differs from ``commit``."""
    if git(repository, "diff", "--quiet", commit, "--", src).returncode == 1:
        return True
    untracked = git(repository, "ls-files", "--others", "--exclude-standard", "--", src)
    return bool(untracked.stdout.strip())


def export_source(repository: Path, commit: str, src: str, destination: Path) -> Path | None:
    """Write ``src`` as committed at ``commit`` under ``destination``; None if it is absent."""
    archive = git(repository, "archive", "--format=tar", commit, "--", src)
    if archive.returncode:
        return None
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(destination, **_EXTRACT)
    return destination / src


def run_module(
    source: Path, module: str, arguments: list[str], *, hidden: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``python -m module arguments`` with the packages under ``source`` importable first.

    ``hidden`` is removed from sys.path, so packages that exist only there cannot leak in.
    """
    packages = sorted(path.name for path in source.iterdir() if (path / "__init__.py").is_file())
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (str(source), environment.get("PYTHONPATH")) if entry
    )
    command = [
        sys.executable, "-c", _CHILD, str(source), str(hidden or ""), ",".join(packages), module,
        *arguments,
    ]
    sys.stdout.flush()
    return subprocess.run(command, env=environment, text=True, check=False)
