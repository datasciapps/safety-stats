"""Executable checks for consumer-facing agent context files (llms.txt, llms-full.txt, SKILL.md)."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import re
import shlex
import shutil
import textwrap
from pathlib import Path
from typing import Any

import pytest
import typer
import yaml
from typer.core import TyperGroup, TyperOption
from typer.testing import CliRunner

from safety_stats import analyze, validate
from safety_stats.cli import app
from safety_stats.config import AnalysisPlan
from tests.cli_output import plain

ROOT = Path(__file__).resolve().parents[1]
SIMPLEQA = ROOT / "examples" / "real_simpleqa"
SKILL_DIR = ROOT / "skills" / "safety-stats"
SKILL_FILE = SKILL_DIR / "SKILL.md"
AGENT_CONTEXT_FILES = [
    ROOT / "llms.txt",
    ROOT / "llms-full.txt",
    SKILL_FILE,
]

_MARKDOWN_LINK = re.compile(r"\]\((?!https?:|mailto:|#)([^)#\s]+)(?:#[^)]*)?\)")
# A fence may be indented, as under a list item in SKILL.md; its closing fence has the same
# indent, and the block is read dedented.
_FENCED_BLOCK = re.compile(
    r"^([ \t]*)```([a-zA-Z0-9_-]+)\n(.*?)^\1```", re.MULTILINE | re.DOTALL
)
_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

runner = CliRunner()


def _fenced_blocks(path: Path, language: str) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    blocks: list[tuple[int, str]] = []
    for match in _FENCED_BLOCK.finditer(text):
        if match.group(2) == language:
            line = text.count("\n", 0, match.start()) + 1
            blocks.append((line, textwrap.dedent(match.group(3))))
    return blocks


def test_llms_txt_follows_index_structure() -> None:
    lines = (ROOT / "llms.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# safety-stats"
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


# --- Python API: names, signatures and import locations -------------------------------------

_MODULE = r"safety_stats(?:\.[a-z_]+)*"
# One documented call: `name(parameters)` or `name(parameters) -> Return`, a whole code span.
_DOCUMENTED_CALL = re.compile(r"`([A-Za-z_]\w*)\(([^`]*)\)(?: -> ([^`]+))?`")
_SECTION_IMPORT = re.compile(rf"Import\b(.*?)\bfrom `({_MODULE})`")
_LINE_MODULE = re.compile(rf"\((?:in|import from) `({_MODULE})`\)")
_IDENTIFIER_SPAN = re.compile(r"`([A-Za-z_]\w*)`")
# SKILL.md states locations as "`a`, `b` and `c` are in `module`" or "(`a` in `module`)".
_STATED_LOCATION = re.compile(rf"([^;(*.\n]*?)\b(?:is |are )?in `({_MODULE})`")


def _section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    assert heading in text, f"{path.relative_to(ROOT)} has no {heading!r}"
    after = text.split(heading, maxsplit=1)[1]
    # A "##" section runs to the next rule; a "###" section also ends at the next heading.
    end = r"\n---\n|\n## |\n### " if heading.startswith("### ") else r"\n---\n"
    return re.split(end, after, maxsplit=1)[0]


def _documented_calls() -> list[tuple[str, str, str | None, str]]:
    """Each call in llms-full.txt's map as (name, parameters, return, stated module)."""
    calls: list[tuple[str, str, str | None, str]] = []
    unplaced: list[str] = []
    map_text = _section(ROOT / "llms-full.txt", "## Problem-to-function map")
    for block in re.split(r"\n### ", map_text):
        default_module: str | None = None
        named: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith("- "):
                continue
            for names, module in _SECTION_IMPORT.findall(line):
                listed = _IDENTIFIER_SPAN.findall(names)
                if listed:
                    named.update(dict.fromkeys(listed, module))
                else:
                    default_module = module
        for line in block.splitlines():
            if not line.startswith("- "):
                continue
            line_module = _LINE_MODULE.search(line)
            for name, params, returns in _DOCUMENTED_CALL.findall(line):
                if hasattr(builtins, name):
                    continue
                module = (
                    named.get(name)
                    or (line_module.group(1) if line_module else None)
                    or default_module
                )
                if module is None:
                    unplaced.append(name)
                    continue
                calls.append((name, params, returns.strip() or None, module))
    assert not unplaced, f"llms-full.txt states no module for: {unplaced}"
    return calls


def _split_parameters(params: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current = ""
    for char in params:
        if quote:
            quote = None if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def _signature_mismatches(name: str, params: str, returns: str | None, obj: object) -> list[str]:
    signature = inspect.signature(obj)  # type: ignore[arg-type]
    actual = list(signature.parameters.values())
    documented = _split_parameters(params)
    problems: list[str] = []

    actual_names = [parameter.name for parameter in actual]
    documented_names = [part.split("=", 1)[0].strip() for part in documented if part != "*"]
    if documented_names != actual_names:
        problems.append(f"{name}: documents {documented_names}, main has {actual_names}")
        return problems

    star = documented.index("*") if "*" in documented else None
    first_keyword_only = next(
        (i for i, p in enumerate(actual) if p.kind is inspect.Parameter.KEYWORD_ONLY), None
    )
    if star != first_keyword_only:
        problems.append(
            f"{name}: `*` documented before parameter {star}, main {first_keyword_only}"
        )

    documented_defaults = [part for part in documented if part != "*"]
    for part, parameter in zip(documented_defaults, actual, strict=True):
        has_default = parameter.default is not inspect.Parameter.empty
        if "=" not in part:
            if has_default:
                problems.append(f"{name}: {parameter.name} has default {parameter.default!r}")
            continue
        documented_default = ast.literal_eval(part.split("=", 1)[1].strip())
        if not has_default or documented_default != parameter.default:
            problems.append(
                f"{name}: {parameter.name} documented default {documented_default!r},"
                f" main {parameter.default if has_default else 'none'!r}"
            )

    if returns is not None and str(signature.return_annotation) != returns:
        problems.append(f"{name}: returns {returns!r}, main {signature.return_annotation!r}")
    return problems


def test_documented_stats_signatures_match_main() -> None:
    calls = _documented_calls()
    problems: list[str] = []
    for name, params, returns, module_name in calls:
        module = importlib.import_module(module_name)
        if not hasattr(module, name):
            problems.append(f"{module_name}.{name} does not exist")
            continue
        problems.extend(_signature_mismatches(name, params, returns, getattr(module, name)))
    assert len({call[0] for call in calls}) >= 30, f"only {len(calls)} documented calls found"
    assert not problems, "llms-full.txt's map differs from main:\n" + "\n".join(problems)


def test_skill_import_locations_hold() -> None:
    text = SKILL_FILE.read_text(encoding="utf-8")
    stated: list[tuple[str, str]] = []
    for clause, module in _STATED_LOCATION.findall(text):
        stated.extend((name, module) for name in _IDENTIFIER_SPAN.findall(clause))
    missing = [
        f"{module}.{name}"
        for name, module in stated
        if not hasattr(importlib.import_module(module), name)
    ]
    assert len(stated) >= 10, f"expected the stated import locations, found {stated}"
    assert not missing, f"SKILL.md states imports that do not exist: {missing}"


# --- CLI: documented commands and options ---------------------------------------------------

_INLINE_COMMAND = re.compile(r"`(safety-stats [^`]+)`")
_OPTION = re.compile(r"--[a-z][a-z0-9-]*")


def _documented_command_lines(path: Path) -> list[str]:
    """Every `safety-stats ...` invocation in a file, prose and bash blocks."""
    text = path.read_text(encoding="utf-8")
    lines = _INLINE_COMMAND.findall(text)
    for _, block in _fenced_blocks(path, "bash"):
        for line in block.replace("\\\n", " ").splitlines():
            if line.strip().startswith("safety-stats "):
                lines.append(line.strip())
    return lines


def _cli_root() -> TyperGroup:
    # Typer 0.29 vendors click, so its commands are not `click` classes: use typer's own.
    root = typer.main.get_command(app)
    assert isinstance(root, TyperGroup)
    return root


def _resolve(line: str) -> tuple[list[tuple[str, Any]], str]:
    """The commands a documented line names (alternatives split on `|`) and its option text."""
    words = line.split()[1:]
    frontier: list[tuple[str, Any]] = [("", _cli_root())]
    index = 0
    while any(isinstance(command, TyperGroup) for _, command in frontier):
        assert index < len(words), f"{line!r} names a command group, not a command"
        next_frontier: list[tuple[str, Any]] = []
        for path, group in frontier:
            assert isinstance(group, TyperGroup), f"{line!r} mixes groups and commands"
            for name in words[index].split("|"):
                assert name in group.commands, (
                    f"{line!r}: no command {(path + ' ' + name).strip()!r}"
                )
                next_frontier.append(((path + " " + name).strip(), group.commands[name]))
        frontier = next_frontier
        index += 1
    return frontier, " ".join(words[index:])


def _option_names(command: Any) -> set[str]:
    names: set[str] = set()
    for parameter in command.params:
        assert isinstance(parameter, TyperOption), f"{command.name}: argument {parameter.name}"
        names.update(parameter.opts, parameter.secondary_opts)
    return names


def test_documented_cli_commands_and_options_exist() -> None:
    problems: list[str] = []
    checked = 0
    for doc_path in AGENT_CONTEXT_FILES:
        for line in _documented_command_lines(doc_path):
            commands, rest = _resolve(line)
            for path, command in commands:
                for option in _OPTION.findall(rest):
                    checked += 1
                    if option not in _option_names(command) | {"--help"}:
                        problems.append(f"{doc_path.relative_to(ROOT)}: `{path}` has no {option}")
    assert checked >= 50, f"only {checked} documented options found"
    assert not problems, "documented CLI options missing from main:\n" + "\n".join(problems)


def test_the_skills_command_blocks_are_checked() -> None:
    """Guards the check above: SKILL.md's bash blocks sit indented under list items."""
    lines = [
        line
        for _, block in _fenced_blocks(SKILL_FILE, "bash")
        for line in block.splitlines()
        if line.startswith("safety-stats ")
    ]
    assert len(lines) >= 2, lines


# SKILL.md gives the options of validate and analyze in code spans of their own, as in
# "with `--runs runs.jsonl --rollouts rollouts.jsonl` in place of `--logs`".
_OPTION_SPAN = re.compile(r"`(--[a-z][^`]*)`")


def test_the_skills_option_spans_name_options_of_validate_or_analyze() -> None:
    root = _cli_root()
    known = _option_names(root.commands["validate"]) | _option_names(root.commands["analyze"])
    spans = _OPTION_SPAN.findall(SKILL_FILE.read_text(encoding="utf-8"))
    options = [option for span in spans for option in _OPTION.findall(span)]
    assert len(options) >= 4, options
    unknown = sorted(set(options) - known)
    assert not unknown, f"SKILL.md names options validate and analyze do not have: {unknown}"


def test_command_list_covers_every_command_and_option() -> None:
    command_list = _section(ROOT / "llms-full.txt", "### Commands")
    documented: dict[str, set[str]] = {}
    elided: set[str] = set()
    for line in _INLINE_COMMAND.findall(command_list):
        commands, rest = _resolve(line)
        for path, _ in commands:
            documented.setdefault(path, set()).update(_OPTION.findall(rest))
            if rest.strip().endswith("..."):
                elided.add(path)

    leaves: dict[str, Any] = {}

    def collect(command: Any, path: str) -> None:
        if isinstance(command, TyperGroup):
            for name, sub in command.commands.items():
                collect(sub, (path + " " + name).strip())
        else:
            leaves[path] = command

    collect(_cli_root(), "")
    undocumented_commands = sorted(set(leaves) - set(documented))
    undocumented_options = [
        f"{path} {'/'.join(parameter.opts + parameter.secondary_opts)}"
        for path, command in leaves.items()
        if path in documented and path not in elided
        for parameter in command.params
        if isinstance(parameter, TyperOption)
        and parameter.name != "help"
        and not set(parameter.opts + parameter.secondary_opts) & documented[path]
    ]
    assert not undocumented_commands, f"commands missing from the list: {undocumented_commands}"
    assert not undocumented_options, f"options missing from the list: {undocumented_options}"


# --- Documented commands run, and write the documented files ---------------------------------


def _documented_report_files() -> tuple[set[str], set[str]]:
    """The files the report-directory list names, and those it says are written on request."""
    contents = _section(ROOT / "llms-full.txt", "### Report directory contents")
    listed: set[str] = set()
    on_request: set[str] = set()
    for line in contents.splitlines():
        if not line.startswith("- `"):
            continue
        head = line.split(":", 1)[0]
        names = set(re.findall(r"`([\w.-]+\.(?:json|md|html|csv|yaml))`", head))
        listed |= names
        if "Written only when" in line:
            on_request |= names
    return listed, on_request


def test_documented_commands_run_offline_and_write_the_documented_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)  # keep working directory on SIMPLEQA's drive on Windows (#35)
    schema_res = runner.invoke(app, ["schema", "plan"])
    assert schema_res.exit_code == 0
    assert '"schema_version"' in plain(schema_res.output)

    output_dirs: list[Path] = []
    commands = [
        line
        for _, block in _fenced_blocks(ROOT / "llms-full.txt", "bash")
        for line in block.replace("\\\n", " ").splitlines()
        if line.strip()
    ]
    assert len(commands) >= 2
    for index, line in enumerate(commands):
        words = shlex.split(line)
        assert words[0] == "safety-stats", f"not a CLI command: {line!r}"
        args = words[1:]
        if "--output" in args:
            position = args.index("--output") + 1
            args[position] = str(tmp_path / f"output-{index}")
            output_dirs.append(Path(args[position]))
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{line!r} exited {result.exit_code}:\n{plain(result.output)}"
        if args[0] == "validate":
            assert "Integrity: valid" in plain(result.output)

    assert output_dirs, "llms-full.txt runs no command that writes a report directory"
    listed, on_request = _documented_report_files()
    for output_dir in output_dirs:
        written = {path.name for path in output_dir.iterdir()}
        assert written == listed - on_request, (
            f"written but not listed: {sorted(written - listed)};"
            f" listed but not written: {sorted(listed - on_request - written)}"
        )
