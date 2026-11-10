"""Which parts of the package may depend on an evaluation tool, checked on the source.

The core reads runs and rollouts through records, so it works with any source. Only an
adapter may know a tool: it may import that tool's package and the core, never another
tool's adapter. A facade chooses an adapter for its caller, so it may import any adapter,
but only when called: inside a function, or for type checking alone. Importing a facade
therefore loads no tool. Every other import counts wherever it is made.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import inspect_safety_stats

# The package as imported, so that the check reads the source the tests run against.
PACKAGE = Path(inspect_safety_stats.__file__).resolve().parent

# Each adapter, by the source it reads, with the modules that make it up.
ADAPTERS = {
    "inspect": {
        "inspect_safety_stats.ingest.inspect_logs",
        "inspect_safety_stats.generative.logs",
        "inspect_safety_stats.generative.inspect_bridge",
    },
}
# Third-party packages that belong to one tool, and may be imported only by its adapter.
TOOL_PACKAGES = {"inspect": {"inspect_ai"}}
# Modules that pick an adapter for the caller, or re-export one, when called.
FACADES = {
    "inspect_safety_stats",
    "inspect_safety_stats.api",
    "inspect_safety_stats.cli",
    "inspect_safety_stats.generative",
    "inspect_safety_stats.ingest",
}
# Orchestration that drives one tool end to end, and so may load its adapter on import.
TOOL_PIPELINES = {"inspect_safety_stats.generative.runner": "inspect"}


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _imports(path: Path, name: str | None = None) -> dict[str, bool]:
    """Every module the file imports, resolved to absolute names, and whether on import.

    An import runs on import of the file unless it sits in a function body or under
    ``if TYPE_CHECKING:``.
    """
    name = name or _module_name(path)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found: dict[str, bool] = {}

    def add(target: str, eager: bool) -> None:
        found[target] = found.get(target, False) or eager

    def visit(node: ast.AST, eager: bool) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name, eager)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - node.level + 1]
                base = ".".join([*anchor, base] if base else anchor)
            add(base, eager)
            for alias in node.names:
                add(f"{base}.{alias.name}", eager)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for child in ast.iter_child_nodes(node):
                visit(child, False)
            return
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            for child in node.body:
                visit(child, False)
            for child in node.orelse:
                visit(child, eager)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, eager)

    visit(ast.parse(path.read_text(encoding="utf-8")), True)
    return found


MODULES = {_module_name(path): path for path in sorted(PACKAGE.rglob("*.py"))}


def _adapter_of(module: str) -> str | None:
    return next((tool for tool, members in ADAPTERS.items() if module in members), None)


def _tool_of(target: str) -> str | None:
    root = target.split(".")[0]
    return next((tool for tool, packages in TOOL_PACKAGES.items() if root in packages), None)


def _departures(module: str, imports: dict[str, bool]) -> list[str]:
    own = _adapter_of(module) or TOOL_PIPELINES.get(module)
    departures = []
    for target, eager in sorted(imports.items()):
        for tool, what in ((_tool_of(target), "a package of"), (_adapter_of(target), "")):
            if tool is None or tool == own:
                continue
            kind = f"{what} the {tool} tool" if what else f"the {tool} adapter"
            if module not in FACADES:
                departures.append(f"{module} imports {target}, {kind}")
            elif eager:
                departures.append(f"{module} imports {target}, {kind}, on import")
    return departures


def test_the_declared_modules_exist() -> None:
    declared = set().union(*ADAPTERS.values()) | FACADES | set(TOOL_PIPELINES)
    assert sorted(declared - set(MODULES)) == []


@pytest.mark.parametrize("module", sorted(MODULES))
def test_only_adapters_depend_on_an_evaluation_tool(module: str) -> None:
    assert _departures(module, _imports(MODULES[module])) == []


def test_the_check_reports_a_core_module_that_imports_a_tool_even_lazily() -> None:
    planted = {"inspect_ai.log": False, "inspect_safety_stats.ingest.inspect_logs": False}
    assert len(_departures("inspect_safety_stats.ingest.normalize", planted)) == 2


def test_the_check_reports_a_facade_that_loads_an_adapter_on_import() -> None:
    lazy = {"inspect_safety_stats.ingest.inspect_logs": False}
    eager = {"inspect_safety_stats.ingest.inspect_logs": True}
    assert _departures("inspect_safety_stats.api", lazy) == []
    assert len(_departures("inspect_safety_stats.api", eager)) == 1


def test_the_check_reports_an_adapter_that_imports_another_adapter() -> None:
    ADAPTERS["other"] = {"inspect_safety_stats.ingest.other_logs"}
    try:
        found = _departures(
            "inspect_safety_stats.ingest.other_logs",
            {"inspect_safety_stats.ingest.inspect_logs": True},
        )
    finally:
        del ADAPTERS["other"]
    assert len(found) == 1


def test_imports_in_functions_and_for_type_checking_do_not_run_on_import(
    tmp_path: Path,
) -> None:
    source = tmp_path / "probe.py"
    source.write_text(
        "from typing import TYPE_CHECKING\n"
        "import inspect_ai\n"
        "if TYPE_CHECKING:\n"
        "    from inspect_ai.log import EvalLog\n"
        "else:\n"
        "    import inspect_ai.analysis\n"
        "def read():\n"
        "    from inspect_ai.model import get_model\n",
        encoding="utf-8",
    )
    found = _imports(source, "inspect_safety_stats.probe")
    assert found["inspect_ai"] is True
    assert found["inspect_ai.analysis"] is True
    assert found["inspect_ai.log.EvalLog"] is False
    assert found["inspect_ai.model.get_model"] is False
