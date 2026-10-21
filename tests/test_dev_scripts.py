"""The developer scripts in scripts/ do what they claim, on every supported platform."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="the scripts need git")

_BUGGY = "def double(value):\n    return value + value + 1\n"
_FIXED = "def double(value):\n    return 2 * value\n"


def _toy_test(assertion: str) -> str:
    return f"from toy_prove_red import double\n\n\ndef test_double():\n    assert {assertion}\n"


_CATCHES_THE_BUG = _toy_test("double(2) == 4")
_MISSES_THE_BUG = _toy_test("double(2) > 0")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false", *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def toy(tmp_path: Path) -> Path:
    """A repository whose committed package is buggy and whose working tree fixes it."""
    package = tmp_path / "src" / "toy_prove_red"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_BUGGY, encoding="utf-8")
    # Anchors pytest's rootdir here rather than in whatever directory holds tmp_path.
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "Add a buggy package")
    (package / "__init__.py").write_text(_FIXED, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    return tmp_path


def _prove_red(toy: Path, test: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    (toy / "tests" / "test_toy.py").write_text(test, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "prove_red.py"), *arguments, "tests/test_toy.py", "-q"],
        cwd=toy,
        capture_output=True,
        text=True,
        check=False,
    )


def test_prove_red_accepts_a_test_that_fails_without_the_change(toy: Path) -> None:
    (toy / "tests" / "test_toy.py").write_text(_CATCHES_THE_BUG, encoding="utf-8")
    status = _git(toy, "status", "--porcelain", "--untracked-files=all")
    result = _prove_red(toy, _CATCHES_THE_BUG)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PROVEN: the tests fail against src/ at HEAD" in result.stdout
    # The old source is imported from a temporary export; the working tree is never touched.
    assert (toy / "src" / "toy_prove_red" / "__init__.py").read_text(encoding="utf-8") == _FIXED
    assert _git(toy, "status", "--porcelain", "--untracked-files=all") == status


def test_prove_red_rejects_a_test_that_passes_without_the_change(toy: Path) -> None:
    result = _prove_red(toy, _MISSES_THE_BUG)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "NOT PROVEN: the tests pass against src/ at HEAD" in result.stdout


def test_prove_red_compares_with_an_earlier_commit_once_the_change_is_committed(
    toy: Path,
) -> None:
    _git(toy, "commit", "-q", "-am", "Fix the package")
    nothing_to_prove = _prove_red(toy, _CATCHES_THE_BUG)
    assert nothing_to_prove.returncode == 2, nothing_to_prove.stdout + nothing_to_prove.stderr
    assert "pass --base" in nothing_to_prove.stderr
    proven = _prove_red(toy, _CATCHES_THE_BUG, "--base", "HEAD~1")
    assert proven.returncode == 0, proven.stdout + proven.stderr
    assert "PROVEN: the tests fail against src/ at HEAD~1" in proven.stdout


@pytest.fixture
def compare_artifacts(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("compare_artifacts")


def test_compare_artifacts_finds_one_commit_identical_to_itself() -> None:
    """End to end: export, synthetic inputs, both analyses and the byte comparison.

    On Windows CI the checkout and the temporary directory sit on different drives, which is
    where Inspect loses the drive letter of the synthetic logs unless each run starts on theirs.
    """
    result = subprocess.run(
        [
            sys.executable, str(SCRIPTS / "compare_artifacts.py"), "--base", "HEAD",
            "--new", "HEAD", "--case", "synthetic", "--prompts", "4",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "synthetic: identical" in result.stdout
    assert "exit status 0" in result.stdout


def test_compare_artifacts_names_files_that_differ_or_exist_on_one_side(
    compare_artifacts: ModuleType, tmp_path: Path
) -> None:
    base, new = tmp_path / "base", tmp_path / "new"
    for side in (base, new):
        (side / "nested").mkdir(parents=True)
        (side / "same.json").write_bytes(b"{}\n")
        (side / "nested" / "table.csv").write_bytes(b"a,b\n1,2\n")
    assert compare_artifacts.differing_files(base, new) == []
    (new / "nested" / "table.csv").write_bytes(b"a,b\r\n1,2\r\n")
    (base / "only-base.md").write_bytes(b"x")
    assert compare_artifacts.differing_files(base, new) == ["nested/table.csv", "only-base.md"]


def test_compare_artifacts_bundled_cases_exist(compare_artifacts: ModuleType) -> None:
    for case in compare_artifacts.bundled_cases(ROOT).values():
        assert case.plan.is_file(), case.plan
        assert case.logs.exists(), case.logs


def test_compare_artifacts_shortens_long_lines_in_its_excerpt(
    compare_artifacts: ModuleType, tmp_path: Path
) -> None:
    """CSV artifacts hold whole JSON documents per row; the excerpt must stay readable."""
    base, new = tmp_path / "base.csv", tmp_path / "new.csv"
    base.write_text("node," + "x" * 5000 + "\n", encoding="utf-8")
    new.write_text("node," + "y" * 5000 + "\n", encoding="utf-8")
    excerpt = compare_artifacts._excerpt(base, new)
    assert excerpt.splitlines()
    assert all(len(line) <= 220 for line in excerpt.splitlines())
    assert "more characters]" in excerpt
