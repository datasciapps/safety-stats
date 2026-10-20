"""Import behaviour of the package root and the CLI, each checked in a fresh interpreter."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(code: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


@pytest.mark.parametrize("module", ["inspect_safety_stats", "inspect_safety_stats.cli"])
def test_importing_the_package_or_cli_loads_no_heavy_dependency(module: str) -> None:
    loaded = _run(
        f"import sys, {module}\n"
        "print(sorted(name for name in ('inspect_ai', 'pandas', 'scipy') if name in sys.modules))"
    )
    assert loaded == "[]"


@pytest.mark.parametrize(
    "imports",
    [
        "from inspect_safety_stats import validate",
        "import inspect_safety_stats.validate.plan\nfrom inspect_safety_stats import validate",
        "import inspect_safety_stats.api\nfrom inspect_safety_stats import validate",
        "from inspect_safety_stats import analyze\nfrom inspect_safety_stats import validate",
    ],
)
def test_validate_is_the_function_whatever_was_imported_first(imports: str) -> None:
    """The validate subpackage shares the exported function's name."""
    result = _run(
        f"{imports}\n"
        "import inspect_safety_stats as package\n"
        "from inspect_safety_stats.api import validate as source\n"
        "print(validate is source and package.validate is source)"
    )
    assert result == "True"


def test_lazy_exports_are_the_objects_of_their_source_modules() -> None:
    result = _run(
        "import inspect_safety_stats as package\n"
        "from inspect_safety_stats import api, config\n"
        "from inspect_safety_stats.report import visualization\n"
        "from inspect_safety_stats.schema import reports\n"
        "expected = {\n"
        "    'AnalysisPlan': config.AnalysisPlan,\n"
        "    'SafetyReport': reports.SafetyReport,\n"
        "    'analyze': api.analyze,\n"
        "    'load_plan': config.load_plan,\n"
        "    'render_visualization': visualization.render_visualization,\n"
        "    'validate': api.validate,\n"
        "    'write_visualization': visualization.write_visualization,\n"
        "}\n"
        "print(all(getattr(package, name) is value for name, value in expected.items()),\n"
        "      sorted(set(package.__all__) - set(expected)))"
    )
    assert result == "True ['__version__']"
