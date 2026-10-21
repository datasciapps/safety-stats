"""Repository-level invariants that are easy to break and tedious to check by hand.

Every one of these has been broken at least once during development: an example plan
left declaring a removed field, a documentation link pointing at a deleted file, and a
version bumped in two of the three places that carry it.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from inspect_safety_stats import __version__
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.generative.schema import GenerativePlan, load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy

ROOT = Path(__file__).resolve().parents[1]


def _tracked(pattern: str) -> list[Path]:
    try:
        listing = subprocess.run(
            ["git", "ls-files", pattern],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git checkout
        pytest.skip("repository consistency checks require a git checkout")
    return [ROOT / name for name in listing]


# Everything else belongs in a subdirectory. Scratch scripts once sat at the root unnoticed.
_ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    # Community files a public project carries at its root.
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
}


def test_only_expected_files_are_tracked_at_the_repository_root() -> None:
    tracked = {path.name for path in _tracked("*") if path.parent == ROOT}
    unexpected = sorted(tracked - _ROOT_FILES)
    assert not unexpected, "unexpected files at the repository root: " + ", ".join(unexpected)


def _literal_magnitude(node: ast.expr) -> float | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if not isinstance(node.value, bool):
            return abs(float(node.value))
    return None


def test_approx_tolerances_cannot_make_small_comparisons_vacuous() -> None:
    """pytest.approx always adds an absolute tolerance of 1e-12 unless abs= is given.

    So ``approx(3e-17, rel=1e-9)`` accepts 0.0, and a test written to catch a precision bug
    passes against the very bug. That happened while adding the worst_of_k accuracy test.
    Passing ``rel=`` without ``abs=``, or comparing with a tiny literal without ``abs=``, is
    therefore flagged: state the absolute tolerance explicitly (``abs=0`` for a purely
    relative check).
    """
    flagged: list[str] = []
    # git's tests/*.py pathspec already matches files in subdirectories.
    for path in sorted(set(_tracked("tests/*.py"))):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            if name != "approx":
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            if "abs" in keywords:
                continue
            magnitude = _literal_magnitude(node.args[0]) if node.args else None
            if "rel" in keywords or (magnitude is not None and 0 < magnitude < 1e-6):
                flagged.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert not flagged, (
        "pytest.approx without an explicit abs= tolerance, where the default abs=1e-12 can "
        "make the comparison vacuous: " + ", ".join(sorted(set(flagged)))
    )


def test_every_tracked_example_plan_validates() -> None:
    """A plan shipped as an example must still load under the current schema."""
    checked = 0
    for path in _tracked("examples/**/*.yaml"):
        if "report" in path.parts or path.name.endswith(".resolved.yaml"):
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "schema_version" not in raw:
            continue
        if "analysis" in raw:
            AnalysisPlan.model_validate(raw)
        elif "taxonomy" in raw:
            GenerativePlan.model_validate(raw)
        else:
            continue
        checked += 1
    assert checked >= 8, f"expected the shipped example plans to be checked, saw {checked}"


def test_every_tracked_generative_plan_compiles() -> None:
    """A generative example must compile to a tree, not merely parse.

    Validation checks the document; compilation checks that the seed manifest, the arm
    sources and the constraints it names actually resolve together.
    """

    checked = 0
    for path in _tracked("examples/**/*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "taxonomy" not in raw:
            continue
        plan = load_generative_plan(path)
        taxonomy = compile_taxonomy(plan, base_path=path.parent)
        assert taxonomy.terminal_paths
        total = sum(node.target_path_weight for node in taxonomy.terminal_paths)
        assert abs(total - 1.0) < 1e-9, f"{path.name} path weights sum to {total}"
        checked += 1
    assert checked >= 2, f"expected the generative example plans to be checked, saw {checked}"


def test_no_broken_internal_documentation_links() -> None:
    link = re.compile(r"\]\((?!https?:|#)([^)]+?\.md)(?:#[^)]*)?\)")
    broken: list[str] = []
    for path in _tracked("*.md") + _tracked("**/*.md"):
        for target in link.findall(path.read_text(encoding="utf-8")):
            if not (path.parent / target).is_file():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not broken, "broken internal documentation links: " + ", ".join(sorted(set(broken)))


def _pyproject() -> str:
    return (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_version_is_consistent_everywhere_it_is_written() -> None:
    """The version lives in three files and has to be bumped in all of them."""
    declared = re.search(r'^version = "([^"]+)"', _pyproject(), flags=re.MULTILINE)
    assert declared is not None, "pyproject.toml no longer declares a version"
    assert declared.group(1) == __version__

    profiles = (ROOT / "docs" / "independence-profiles.md").read_text(encoding="utf-8")
    current = re.search(r"remains current in `([0-9]+\.[0-9]+\.[0-9]+)`", profiles)
    assert current is not None, "independence-profiles.md no longer states a current version"
    assert current.group(1) == __version__


def test_declared_python_versions_match_the_requirement() -> None:
    classifiers = set(
        re.findall(r'"Programming Language :: Python :: (3\.[0-9]+)"', _pyproject())
    )
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    if not workflow.is_file():  # pragma: no cover - workflow removed
        pytest.skip("no CI workflow to compare against")
    tested = set(
        yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["test"]["strategy"][
            "matrix"
        ]["python-version"]
    )
    assert classifiers == tested, (
        "CI must test exactly the Python versions the package advertises: "
        f"declared {sorted(classifiers)}, tested {sorted(tested)}"
    )
