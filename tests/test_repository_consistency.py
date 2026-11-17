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
from typing import get_args

import pytest
import yaml

import inspect_safety_stats
from inspect_safety_stats import __version__
from inspect_safety_stats.config.loader import SCHEMA_VERSIONS
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.generative.schema import GenerativePlan, load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy
from inspect_safety_stats.schema.reports import SafetyReport

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
    # Notes for coding agents. CLAUDE.md only imports AGENTS.md.
    "AGENTS.md",
    "CLAUDE.md",
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


def _undefined_tests(text: str) -> list[str]:
    """The test functions named in backticks in ``text`` that no tracked test module defines."""
    definitions = "\n".join(path.read_text(encoding="utf-8") for path in _tracked("tests/*.py"))
    return [
        name
        for name in sorted(set(re.findall(r"`(test_\w+)`", text)))
        if f"def {name}(" not in definitions
    ]


def test_agent_notes_cite_tests_and_files_that_exist() -> None:
    """AGENTS.md ties each past mistake to the test that now catches it.

    Renaming such a test, or moving a file the notes point at, would otherwise leave the notes
    pointing at nothing, and nobody would notice.
    """
    notes = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    missing = _undefined_tests(notes)
    paths = set(re.findall(r"\b((?:scripts|src|tests)/[\w./-]+\.(?:py|md|yaml))", notes))
    missing += sorted(path for path in paths if not (ROOT / path).is_file())
    assert not missing, "AGENTS.md cites what no longer exists: " + ", ".join(missing)


def test_formal_proofs_cite_tests_that_exist() -> None:
    """Each theorem in formal/ names the property test that checks it on the implementation.

    The proof covers the mathematics and the test ties the Python code to it. Renaming the
    test would cut that link silently.
    """
    proofs = _tracked("formal/*.lean")
    if not proofs:  # pragma: no cover - formal proofs removed
        pytest.skip("no formal proofs are tracked")
    missing = _undefined_tests("\n".join(path.read_text(encoding="utf-8") for path in proofs))
    assert not missing, "formal/ cites tests that no longer exist: " + ", ".join(missing)


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


def test_documentation_has_no_url_placeholders() -> None:
    """Install instructions once read ``git clone <repository-url>``, which nobody can run."""
    placeholder = re.compile(r"<[a-z-]+-url>")
    found = [
        f"{path.relative_to(ROOT).as_posix()}: {match}"
        for path in sorted(set(_tracked("*.md") + _tracked("**/*.md")))
        for match in placeholder.findall(path.read_text(encoding="utf-8"))
    ]
    assert not found, "placeholders in tracked documentation: " + ", ".join(found)


def test_documented_clones_enter_the_directory_they_create() -> None:
    """``git clone`` names the checkout after the URL; the next ``cd`` must use that name.

    The installation page once cloned the repository and then entered ``inspect-safety-stats``,
    the package name, which is not the directory git creates.
    """
    clone = re.compile(r"^git clone (\S+)\n(?:cd (\S+))?", flags=re.MULTILINE)
    checked = 0
    wrong: list[str] = []
    for path in sorted(set(_tracked("*.md") + _tracked("**/*.md"))):
        for url, directory in clone.findall(path.read_text(encoding="utf-8")):
            checked += 1
            expected = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
            if directory != expected:
                wrong.append(f"{path.relative_to(ROOT).as_posix()}: cd {directory!r} after {url}")
    assert checked >= 2, f"expected the documented clone commands to be checked, saw {checked}"
    assert not wrong, "clone commands not followed by cd into the checkout: " + ", ".join(wrong)


def _pyproject() -> str:
    return (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_version_is_consistent_everywhere_it_is_written() -> None:
    """The version lives in four files and has to be bumped in all of them."""
    declared = re.search(r'^version = "([^"]+)"', _pyproject(), flags=re.MULTILINE)
    assert declared is not None, "pyproject.toml no longer declares a version"
    assert declared.group(1) == __version__

    profiles = (ROOT / "docs" / "independence-profiles.md").read_text(encoding="utf-8")
    current = re.search(r"remains current in `([0-9]+\.[0-9]+\.[0-9]+)`", profiles)
    assert current is not None, "independence-profiles.md no longer states a current version"
    assert current.group(1) == __version__

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert str(citation["version"]) == __version__, "CITATION.cff states another version"


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


# Split so that this file does not match its own pattern. Reports write Windows paths with
# forward slashes, so this also matches a Windows home directory written that way. Backslash
# forms are not matched: the HarmBench behaviours dataset quotes Windows shell sessions run in
# a guest account's home directory.
_HOME_DIRECTORY = re.compile(rb"/" rb"Users/[A-Za-z]")


def test_no_tracked_file_names_a_local_home_directory() -> None:
    """Example READMEs and committed reports once carried a maintainer's checkout path (#27)."""
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in _tracked("*")
        if path.is_file() and _HOME_DIRECTORY.search(path.read_bytes())
    ]
    assert offenders == []


def test_the_package_has_no_runtime_asserts() -> None:
    """Python strips asserts under -O, so a runtime check must raise (CONTRIBUTING 5.2)."""
    # The imported package, not the checkout, so that scripts/prove_red.py can test it.
    package = Path(inspect_safety_stats.__file__).parent
    offenders = [
        f"{path.relative_to(package.parent).as_posix()}:{node.lineno}"
        for path in sorted(package.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Assert)
    ]
    assert offenders == []


def _versions(text: str) -> set[str]:
    return set(re.findall(r'[`"]([0-9]+\.[0-9]+)[`"]', text))


def test_documented_schema_versions_match_the_code() -> None:
    """The docs name the plan and report schema versions this release reads and writes.

    The API reference once said "plan and report schema version 1.0" two releases after 1.0
    plans stopped loading (#17).
    """
    plan_versions = set(SCHEMA_VERSIONS)
    assert set(get_args(AnalysisPlan.model_fields["schema_version"].annotation)) == plan_versions
    report_versions = {SafetyReport.model_fields["schema_version"].default}

    api = (ROOT / "docs" / "api-reference.md").read_text(encoding="utf-8")
    compatibility = api.split("## Compatibility and stability", 1)[1].split("\n## ", 1)[0]
    reads = re.search(r"reads analysis plans of\s+schema version (.+?)\(", compatibility, re.S)
    writes = re.search(r"writes reports of schema version (`[^`]+`)", compatibility)
    plan_reference = (ROOT / "docs" / "plan-reference.md").read_text(encoding="utf-8")
    declared = re.search(r"`schema_version` is (.+?), and an unquoted", plan_reference, re.S)
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    heading = re.search(r"^### 4\.2 Analysis plan .*$", contributing, re.M)

    found = {
        "docs/api-reference.md plan versions": (reads, plan_versions),
        "docs/api-reference.md report version": (writes, report_versions),
        "docs/plan-reference.md plan versions": (declared, plan_versions),
        "CONTRIBUTING.md section 4.2 heading": (heading, plan_versions),
    }
    wrong = [
        f"{where}: says {sorted(_versions(match.group(0))) if match else 'nothing'}, "
        f"the code has {sorted(expected)}"
        for where, (match, expected) in found.items()
        if match is None or _versions(match.group(0)) != expected
    ]
    assert not wrong, "documented schema versions differ from the code: " + "; ".join(wrong)


# CONTRIBUTING section 4.1 declares the generative surface experimental. Every user-facing
# entry point to it carries the same caveat, so none of them presents it as stable (#24).
_GENERATIVE_CAVEAT = (
    "The generative pipeline is experimental: its public surface, "
    "`inspect_safety_stats.generative.__all__`, may change in any minor release."
)


@pytest.mark.parametrize(
    "relative",
    [
        "README.md",
        "docs/generative-evaluation.md",
        "docs/api-reference.md",
        "src/inspect_safety_stats/generative/__init__.py",
    ],
)
def test_generative_entry_points_carry_the_experimental_caveat(relative: str) -> None:
    text = (ROOT / relative).read_text(encoding="utf-8").replace("``", "`")
    assert _GENERATIVE_CAVEAT in " ".join(text.split())
