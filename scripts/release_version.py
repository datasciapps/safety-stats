"""Check a release tag against the package version, and say where the release goes.

``.github/workflows/release.yml`` runs this on every pushed ``v*`` tag, before it builds.

- ``v<version>``, such as ``v0.16.0``, releases the package version to PyPI. It must equal the
  version in ``pyproject.toml`` and ``src/inspect_safety_stats/version.py``.
- ``v<version>rc<N>``, such as ``v0.16.0rc1``, is a release candidate of that version. It is
  built as ``<version>rc<N>`` and goes to TestPyPI only, so a candidate can be installed and
  checked from an index before the release, on the same commit and without a commit that
  changes the version.

Usage::

    python scripts/release_version.py <tag>          # prints version=... and index=...
    python scripts/release_version.py <tag> --apply  # also writes a candidate's version

The printed lines are in the form GitHub Actions reads from ``$GITHUB_OUTPUT``. Any other tag,
or a tag that does not match the package version, exits with status 1 and says why.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = Path("pyproject.toml")
VERSION_MODULE = Path("src") / "inspect_safety_stats" / "version.py"

_TAG = re.compile(r"v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)(?P<candidate>rc[0-9]+)?")
_FINAL = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_PYPROJECT_VERSION = re.compile(r'^version = "(?P<version>[^"]*)"$', re.MULTILINE)
_MODULE_VERSION = re.compile(r'^__version__ = "(?P<version>[^"]*)"$', re.MULTILINE)


class ReleaseTagError(ValueError):
    """A tag this repository does not release from."""


@dataclass(frozen=True)
class Release:
    """What a tag releases: the version the distributions carry, and the index they go to."""

    version: str
    index: str  # "pypi" or "testpypi"


def release_for_tag(tag: str, package_version: str) -> Release:
    """The release a tag stands for, given the package version on the tagged commit."""

    if not _FINAL.fullmatch(package_version):
        raise ReleaseTagError(
            f"the package version {package_version!r} is not a final X.Y.Z version; tag "
            "candidates as v<version>rc<N> and keep the package at the final version"
        )
    match = _TAG.fullmatch(tag)
    if match is None:
        raise ReleaseTagError(
            f"tag {tag!r} is neither v<version> nor v<version>rc<N>, such as "
            f"v{package_version} or v{package_version}rc1"
        )
    if match["version"] != package_version:
        raise ReleaseTagError(
            f"tag {tag!r} is for version {match['version']}, but the package version is "
            f"{package_version}"
        )
    if match["candidate"] is None:
        return Release(version=package_version, index="pypi")
    return Release(version=package_version + match["candidate"], index="testpypi")


def _one_version(pattern: re.Pattern[str], text: str, path: Path) -> re.Match[str]:
    found = list(pattern.finditer(text))
    if len(found) != 1:
        raise ReleaseTagError(f"{path.as_posix()} must state the version exactly once")
    return found[0]


def package_version(root: Path) -> str:
    """The package version, which ``pyproject.toml`` and ``version.py`` must agree on."""

    declared = _one_version(
        _PYPROJECT_VERSION, (root / PYPROJECT).read_text(encoding="utf-8"), PYPROJECT
    )["version"]
    module = _one_version(
        _MODULE_VERSION, (root / VERSION_MODULE).read_text(encoding="utf-8"), VERSION_MODULE
    )["version"]
    if declared != module:
        raise ReleaseTagError(
            f"{PYPROJECT.as_posix()} says {declared} but {VERSION_MODULE.as_posix()} says {module}"
        )
    return declared


def apply_version(root: Path, version: str) -> None:
    """Write ``version`` into ``pyproject.toml`` and ``version.py``, for a candidate's build."""

    for relative, pattern, line in (
        (PYPROJECT, _PYPROJECT_VERSION, f'version = "{version}"'),
        (VERSION_MODULE, _MODULE_VERSION, f'__version__ = "{version}"'),
    ):
        path = root / relative
        text = path.read_text(encoding="utf-8")
        found = _one_version(pattern, text, relative)
        updated = text[: found.start()] + line + text[found.end() :]
        path.write_text(updated, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tag", help="the pushed tag, such as v0.16.0 or v0.16.0rc1")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write a release candidate's version into pyproject.toml and version.py",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    try:
        release = release_for_tag(arguments.tag, package_version(arguments.root))
    except ReleaseTagError as error:
        print(f"release_version: {error}", file=sys.stderr)
        return 1
    if arguments.apply:
        apply_version(arguments.root, release.version)
    print(f"version={release.version}")
    print(f"index={release.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
