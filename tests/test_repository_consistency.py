"""Repository-level invariants that are easy to break and tedious to check by hand.

Every one of these has been broken at least once during development: an example plan
left declaring a removed field, a documentation link pointing at a deleted file, and a
version bumped in two of the three places that carry it.
"""

from __future__ import annotations

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
