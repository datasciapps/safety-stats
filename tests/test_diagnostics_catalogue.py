"""docs/diagnostics.md is the registry as scripts/render_catalogue.py renders it, byte for byte."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from inspect_safety_stats.registry import ASSUMPTIONS, CODES

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


@pytest.fixture
def render_catalogue(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("render_catalogue")


def test_the_committed_catalogue_is_the_rendered_registry(render_catalogue: ModuleType) -> None:
    committed = (ROOT / "docs" / "diagnostics.md").read_bytes().decode("utf-8")
    assert committed == render_catalogue.render(), (
        "docs/diagnostics.md is stale: run python scripts/render_catalogue.py"
    )


def test_the_catalogue_has_a_section_for_every_code_and_assumption(
    render_catalogue: ModuleType,
) -> None:
    headings = set(render_catalogue.render().splitlines())
    assert {f"#### `{code}`" for code in CODES} <= headings
    assert {f"### Assumption `{name}`" for name in ASSUMPTIONS} <= headings


def test_every_link_within_the_catalogue_has_its_heading(render_catalogue: ModuleType) -> None:
    page = render_catalogue.render()
    anchors = {
        render_catalogue._anchor(line.lstrip("#").strip())
        for line in page.splitlines()
        if line.startswith("#")
    }
    targets = set(render_catalogue.re.findall(r"\]\(#([^)]+)\)", page))
    assert targets and targets <= anchors


def test_the_check_reports_a_stale_page(
    render_catalogue: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = tmp_path / "diagnostics.md"
    monkeypatch.setattr(render_catalogue, "ROOT", tmp_path)
    monkeypatch.setattr(render_catalogue, "PAGE", page)
    page.write_text("# An old catalogue\n", encoding="utf-8", newline="\n")
    assert render_catalogue.main(["--check"]) == 1
    assert render_catalogue.main([]) == 0
    assert render_catalogue.main(["--check"]) == 0
