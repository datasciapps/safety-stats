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


def _report(directory: Path) -> dict[str, object]:
    import json

    loaded: dict[str, object] = json.loads((directory / "report.json").read_text("utf-8"))
    return loaded


def _rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _interval_ids(report: dict[str, object]) -> set[str]:
    groups = report["group_risks"]
    assert isinstance(groups, list)
    return {interval["interval_id"] for group in groups for interval in group["intervals"]}


def test_compare_artifacts_cases_reach_the_paths_they_exist_for(
    compare_artifacts: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A case that stops reaching its path still compares as identical and protects nothing.

    Builds every offline case with the working tree as the base source and analyses it once.
    """
    from _artifact_cases import BUILDERS, Builder

    from inspect_safety_stats import analyze

    # Inspect drops the drive letter of logs on another Windows drive than the working
    # directory (#35); the cases sit under tmp_path.
    monkeypatch.chdir(tmp_path)
    builder = Builder(ROOT, tmp_path / "inputs", ROOT / "src", None)
    assert set(BUILDERS) <= set(compare_artifacts.CASES)
    out: dict[str, Path] = {}
    for name, build in BUILDERS.items():
        inputs = build(builder)
        out[name] = tmp_path / "out" / name
        if inputs.command == ("analyze",):
            analyze(plan=inputs.plan, logs=inputs.logs).write(out[name])
            continue
        result = subprocess.run(
            [sys.executable, "-m", "inspect_safety_stats.cli", *inputs.command,
             "--plan", str(inputs.plan), "--logs", str(inputs.logs), *inputs.extra,
             "--output", str(out[name])],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    demo = _report(out["demo"])
    assert any(row["diagnostic_outputs"] not in ("", "[]", "{}")
               for row in _rows(out["demo"] / "comparisons.csv"))
    assert demo["comparisons"]

    errors = _report(out["errors"])
    stages = {row["error_stage"] for row in _rows(out["errors"] / "observations.csv")}
    assert {"target_generation", "post_generation", "unknown"} <= stages
    assert errors["policy_events"] and errors["comparison_sets"]
    assert "judge_corrected_declared" in _interval_ids(errors)

    assert "judge_corrected_panel_derived" in _interval_ids(_report(out["judge-panel"]))
    assert "judge_corrected_validated" in _interval_ids(_report(out["judge-validated"]))

    incomplete = _report(out["incomplete"])["integrity"]
    assert isinstance(incomplete, dict)
    assert "LOG_INCOMPLETE" in {item["code"] for item in incomplete["diagnostics"]}

    assert any(name.startswith("horvitz_thompson_")
               for name in _interval_ids(_report(out["sampled-poisson"])))
    assert any(name.startswith("stratified_without_replacement_")
               for name in _interval_ids(_report(out["sampled-stratified"])))

    for name, method in (("generative", "paired_binomial_cells"),
                         ("generative-weighted", "paired_weighted_kl_chernoff")):
        assert {row["method"] for row in _rows(out[name] / "comparisons.csv")} == {method}


@pytest.fixture
def artifact_cases(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("_artifact_cases")


def _fake_log(model: str, status: str = "success") -> object:
    from types import SimpleNamespace

    error = None if status == "success" else SimpleNamespace(message="the provider failed")
    return SimpleNamespace(status=status, error=error, eval=SimpleNamespace(model=model))


def test_compare_artifacts_cases_reject_a_demo_run_that_did_not_succeed(
    artifact_cases: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed build gives both sides the same empty inputs, which compare as identical."""
    import inspect_ai

    calls: list[dict[str, object]] = []

    def failing_eval(*args: object, **kwargs: object) -> list[object]:
        calls.append(kwargs)
        return [_fake_log(model, "error") for model in artifact_cases.DEMO_MODELS]

    monkeypatch.setattr(inspect_ai, "eval", failing_eval)
    builder = artifact_cases.Builder(ROOT, tmp_path, ROOT / "src", None)
    with pytest.raises(RuntimeError, match="did not succeed"):
        artifact_cases.demo_logs(builder)
    # The case reads .eval files, whatever format Inspect would choose by default.
    assert calls and calls[0]["log_format"] == "eval"


def test_compare_artifacts_cases_require_one_log_per_demo_model(
    artifact_cases: ModuleType, tmp_path: Path
) -> None:
    """Editing logs that are not there writes none, and every case reading them is empty."""
    source = tmp_path / "logs"
    source.mkdir()
    with pytest.raises(RuntimeError, match="one log per configuration"):
        artifact_cases._edit_logs(source, tmp_path / "edited", lambda log, index: None)
