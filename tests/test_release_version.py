"""The release workflow publishes only what a tag and the package version agree on."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_version(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("release_version")


def _tree(root: Path, declared: str, module: str) -> Path:
    (root / "src" / "safety_stats").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "safety-stats"\nversion = "{declared}"\n'
        '\n[tool.other]\ntarget-version = "py311"\n',
        encoding="utf-8",
    )
    (root / "src" / "safety_stats" / "version.py").write_text(
        f'__version__ = "{module}"\n', encoding="utf-8"
    )
    return root


def test_a_version_tag_releases_the_package_version_to_pypi(
    release_version: ModuleType,
) -> None:
    release = release_version.release_for_tag("v0.16.0", "0.16.0")
    assert (release.version, release.index) == ("0.16.0", "pypi")


def test_a_candidate_tag_builds_a_candidate_version_for_testpypi(
    release_version: ModuleType,
) -> None:
    release = release_version.release_for_tag("v0.16.0rc2", "0.16.0")
    assert (release.version, release.index) == ("0.16.0rc2", "testpypi")


@pytest.mark.parametrize(
    "tag",
    [
        "v0.15.0",
        "v0.16.1",
        "v0.16.0rc",
        "v0.16.0rc0",
        "v0.16.0rc01",
        "v0.16.0-rc1",
        "0.16.0",
        "v0.16",
        "v0.16.0.post1",
    ],
)
def test_other_tags_release_nothing(release_version: ModuleType, tag: str) -> None:
    with pytest.raises(release_version.ReleaseTagError):
        release_version.release_for_tag(tag, "0.16.0")


def test_a_package_at_a_candidate_version_releases_nothing(
    release_version: ModuleType,
) -> None:
    with pytest.raises(release_version.ReleaseTagError, match="final"):
        release_version.release_for_tag("v0.16.0rc1", "0.16.0rc1")


def test_the_two_version_files_must_agree(release_version: ModuleType, tmp_path: Path) -> None:
    assert release_version.package_version(_tree(tmp_path / "same", "0.16.0", "0.16.0")) == (
        "0.16.0"
    )
    with pytest.raises(release_version.ReleaseTagError, match="version.py says 0.15.0"):
        release_version.package_version(_tree(tmp_path / "differ", "0.16.0", "0.15.0"))


def test_apply_writes_a_candidate_version_into_both_files_only(
    release_version: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _tree(tmp_path, "0.16.0", "0.16.0")
    assert release_version.main(["v0.16.0rc1", "--apply", "--root", str(root)]) == 0
    assert capsys.readouterr().out == "version=0.16.0rc1\nindex=testpypi\n"
    pyproject = (root / "pyproject.toml").read_bytes()
    assert b'\nversion = "0.16.0rc1"\n' in pyproject
    assert b'target-version = "py311"' in pyproject
    assert b"\r" not in pyproject
    assert release_version.package_version(root) == "0.16.0rc1"


def test_without_apply_nothing_is_written(
    release_version: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _tree(tmp_path, "0.16.0", "0.16.0")
    files = [root / "pyproject.toml", root / "src" / "safety_stats" / "version.py"]
    before = [path.read_bytes() for path in files]
    assert release_version.main(["v0.16.0rc1", "--root", str(root)]) == 0
    assert capsys.readouterr().out == "version=0.16.0rc1\nindex=testpypi\n"
    after = [path.read_bytes() for path in files]
    assert after == before


def test_a_mismatched_tag_exits_1_and_changes_nothing(
    release_version: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _tree(tmp_path, "0.16.0", "0.16.0")
    before = (root / "pyproject.toml").read_bytes()
    assert release_version.main(["v0.17.0rc1", "--apply", "--root", str(root)]) == 1
    assert "package version is 0.16.0" in capsys.readouterr().err
    assert (root / "pyproject.toml").read_bytes() == before


def test_this_checkout_states_one_final_version(release_version: ModuleType) -> None:
    version = release_version.package_version(ROOT)
    assert release_version.release_for_tag(f"v{version}", version).index == "pypi"


def test_the_release_workflow_publishes_each_index_from_its_own_environment() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    published = {
        job["environment"]["name"]: job["if"]
        for job in jobs.values()
        if "environment" in job
    }
    assert published == {
        "pypi": "needs.build.outputs.index == 'pypi'",
        "testpypi": "needs.build.outputs.index == 'testpypi'",
    }
    # Only the publishing jobs may mint an OpenID Connect token; nothing stores a PyPI token.
    assert workflow["permissions"] == {"contents": "read"}
    for name, job in jobs.items():
        expected = {"id-token": "write"} if "environment" in job else None
        assert job.get("permissions") == expected, name
    assert "secrets." not in (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )


def test_the_release_workflow_checks_before_it_publishes() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    # YAML 1.1 reads the key `on` as the boolean True.
    trigger = workflow.get("on", workflow.get(True))
    assert trigger == {"push": {"tags": ["v*"]}}
    jobs = workflow["jobs"]
    build = " ".join(step.get("run", "") for step in jobs["build"]["steps"])
    assert "merge-base --is-ancestor" in build
    assert "scripts/release_version.py" in build and "--apply" in build
    assert "twine check --strict" in build
    assert "scripts/check_wheel.sh" in build
    uploads = {}
    for name in ("pypi", "testpypi"):
        assert jobs[name]["needs"] == "build"
        (step,) = [
            step
            for step in jobs[name]["steps"]
            if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@")
        ]
        uploads[name] = step.get("with", {}).get("repository-url")
    assert uploads == {"pypi": None, "testpypi": "https://test.pypi.org/legacy/"}
