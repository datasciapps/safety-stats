"""Executable checks for consumer-facing agent context files (llms.txt, llms-full.txt, SKILL.md)."""

from __future__ import annotations

import builtins
import importlib
import re
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from inspect_safety_stats import analyze, validate
from inspect_safety_stats.cli import app
from inspect_safety_stats.config import AnalysisPlan
from tests.cli_output import plain

ROOT = Path(__file__).resolve().parents[1]
SIMPLEQA = ROOT / "examples" / "real_simpleqa"
SKILL_DIR = ROOT / "skills" / "inspect-safety-stats"
SKILL_FILE = SKILL_DIR / "SKILL.md"
AGENT_CONTEXT_FILES = [
    ROOT / "llms.txt",
    ROOT / "llms-full.txt",
    SKILL_FILE,
]

_MARKDOWN_LINK = re.compile(r"\]\((?!https?:|mailto:|#)([^)#\s]+)(?:#[^)]*)?\)")
_FENCED_BLOCK = re.compile(r"^```([a-zA-Z0-9_-]+)\n(.*?)^```", re.MULTILINE | re.DOTALL)
_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

runner = CliRunner()


def _fenced_blocks(path: Path, language: str) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    blocks: list[tuple[int, str]] = []
    for match in _FENCED_BLOCK.finditer(text):
        if match.group(1) == language:
            line = text.count("\n", 0, match.start()) + 1
            blocks.append((line, match.group(2)))
    return blocks


def test_llms_txt_follows_index_structure() -> None:
    lines = (ROOT / "llms.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# inspect-safety-stats"
    blockquote_lines = [line for line in lines[1:8] if line.startswith("> ")]
    assert blockquote_lines, "llms.txt must start with an H1 title and summary blockquote"
    headings = [line for line in lines if line.startswith("## ")]
    assert "## Optional" in headings


@pytest.mark.parametrize(
    "doc_path", AGENT_CONTEXT_FILES, ids=lambda p: p.relative_to(ROOT).as_posix()
)
def test_agent_context_links_resolve_to_existing_files(doc_path: Path) -> None:
    text = doc_path.read_text(encoding="utf-8")
    targets = _MARKDOWN_LINK.findall(text)
    assert targets, f"expected relative links in {doc_path.relative_to(ROOT)}"
    missing = [
        f"{doc_path.relative_to(ROOT)} -> {target}"
        for target in targets
        if not (doc_path.parent / target).exists()
    ]
    assert not missing, "broken relative links in agent context:\n" + "\n".join(missing)


def test_skill_frontmatter_follows_agent_skills_spec() -> None:
    raw = SKILL_FILE.read_text(encoding="utf-8")
    assert raw.startswith("---\n"), "SKILL.md must begin with YAML frontmatter"
    _, frontmatter_text, body = raw.split("---\n", maxsplit=2)
    meta = yaml.safe_load(frontmatter_text)
    assert isinstance(meta, dict)

    name = meta.get("name")
    assert isinstance(name, str) and 1 <= len(name) <= 64
    assert _SKILL_NAME.match(name), f"invalid skill name: {name!r}"
    assert name == SKILL_DIR.name, "skill name must match parent directory name"

    description = meta.get("description")
    assert isinstance(description, str) and 1 <= len(description.strip()) <= 1024
    assert len(raw.splitlines()) <= 250, "SKILL.md should stay concise for single-turn loading"
    assert body.strip().startswith("# ")


def test_every_python_snippet_in_agent_context_executes() -> None:
    executed = 0
    for doc_path in AGENT_CONTEXT_FILES:
        for line, code in _fenced_blocks(doc_path, "python"):
            label = f"{doc_path.relative_to(ROOT)}:{line}"
            compiled = compile(code, label, "exec")
            exec(compiled, {})
            executed += 1
    assert executed >= 3


def test_every_yaml_plan_in_agent_context_validates_and_analyzes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)  # keep working directory on SIMPLEQA's drive on Windows (#35)
    checked = 0
    for doc_path in AGENT_CONTEXT_FILES:
        for index, (line, raw_yaml) in enumerate(_fenced_blocks(doc_path, "yaml")):
            loaded = yaml.safe_load(raw_yaml)
            if not isinstance(loaded, dict) or loaded.get("schema_version") != "2.0":
                continue
            plan = AnalysisPlan.model_validate(loaded)
            assert plan.schema_version == "2.0"

            run_dir = tmp_path / f"{doc_path.stem}-{index}-{line}"
            run_dir.mkdir()
            shutil.copy(SIMPLEQA / "design.csv", run_dir / "design.csv")
            plan_path = run_dir / "plan.yaml"
            plan_path.write_text(raw_yaml, encoding="utf-8", newline="\n")

            integrity = validate(plan=plan_path, logs=[SIMPLEQA / "simpleqa-8-real.eval"])
            assert integrity.status == "valid", (
                f"{doc_path.relative_to(ROOT)}:{line} failed validation: {integrity.diagnostics}"
            )
            report = analyze(plan=plan_path, logs=[SIMPLEQA / "simpleqa-8-real.eval"])
            assert report.integrity.status == "valid"
            assert report.group_risks
            checked += 1
    assert checked >= 1


def test_documented_stats_symbols_exist_in_stated_modules() -> None:
    stats_root = importlib.import_module("inspect_safety_stats.stats")
    planning = importlib.import_module("inspect_safety_stats.stats.planning")
    misclassification = importlib.import_module("inspect_safety_stats.stats.misclassification")
    planning_only = {
        "certification_sample_size",
        "maximum_certifying_failures",
        "maximum_certifiable_failure_rate",
        "minimum_detectable_failure_rate",
        "rejection_sample_size",
        "fixed_design_detectable_failure_rates",
    }
    misclassification_only = {"JudgeStratumCounts", "estimate_judge_calibration"}

    text = (ROOT / "llms-full.txt").read_text(encoding="utf-8")
    after_heading = text.split("## Problem-to-function map", maxsplit=1)[1]
    map_section = after_heading.split("\n---", maxsplit=1)[0]

    checked: set[str] = set()
    missing: list[str] = []
    for line in map_section.splitlines():
        if not line.startswith("- "):
            continue
        for symbol in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)\(", line):
            if hasattr(builtins, symbol):
                continue
            checked.add(symbol)
            if symbol in planning_only:
                if not hasattr(planning, symbol):
                    missing.append(f"stats.planning.{symbol}")
            elif symbol in misclassification_only:
                if not hasattr(misclassification, symbol):
                    missing.append(f"stats.misclassification.{symbol}")
            elif not hasattr(stats_root, symbol):
                missing.append(f"stats.{symbol}")
    assert len(checked) >= 25, f"expected >=25 documented callables, checked only {len(checked)}"
    assert not missing, f"documented stats symbols missing from stated modules: {missing}"


def test_documented_cli_commands_succeed_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)  # keep working directory on SIMPLEQA's drive on Windows (#35)
    schema_res = runner.invoke(app, ["schema", "plan"])
    assert schema_res.exit_code == 0
    assert '"schema_version"' in plain(schema_res.output)

    validate_res = runner.invoke(
        app,
        [
            "validate",
            "--plan",
            str(SIMPLEQA / "safety-plan.yaml"),
            "--logs",
            str(SIMPLEQA / "simpleqa-8-real.eval"),
        ],
    )
    assert validate_res.exit_code == 0
    assert "Integrity: valid" in plain(validate_res.output)

    out_dir = tmp_path / "simpleqa-report"
    analyze_res = runner.invoke(
        app,
        [
            "analyze",
            "--plan",
            str(SIMPLEQA / "safety-plan.yaml"),
            "--logs",
            str(SIMPLEQA / "simpleqa-8-real.eval"),
            "--output",
            str(out_dir),
        ],
    )
    assert analyze_res.exit_code == 0
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.html").exists()
