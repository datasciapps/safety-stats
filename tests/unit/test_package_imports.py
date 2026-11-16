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


@pytest.mark.parametrize(
    "module",
    [
        "inspect_safety_stats",
        "inspect_safety_stats.cli",
        "inspect_safety_stats.ingest",
        "inspect_safety_stats.generative",
        "inspect_safety_stats.stats",
    ],
)
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


@pytest.mark.parametrize(
    "package",
    [
        "inspect_safety_stats.ingest",
        "inspect_safety_stats.generative",
        "inspect_safety_stats.stats",
    ],
)
def test_subpackage_lazy_exports_cover_all_and_keep_submodules_reachable(package: str) -> None:
    """Each public name has a source module, and submodules stay attributes of the package."""
    result = _run(
        f"import importlib, {package} as package\n"
        "exports = package._EXPORTS\n"
        "same = all(getattr(package, name) is getattr(importlib.import_module(module), name)\n"
        "           for name, module in exports.items())\n"
        "submodules = all(getattr(package, name).__name__ == f'{package.__name__}.{name}'\n"
        "                 for name in package._SUBMODULES)\n"
        "print(sorted(exports) == sorted(package.__all__), same, submodules)"
    )
    assert result == "True True True"


@pytest.mark.parametrize(
    "module",
    [
        "inspect_safety_stats.api",
        "inspect_safety_stats.ingest.normalize",
        "inspect_safety_stats.ingest.records",
        "inspect_safety_stats.ingest.rollout_files",
    ],
)
def test_the_analysis_and_the_ingestion_core_load_inspect_only_when_reading_its_logs(
    module: str,
) -> None:
    assert _run(f"import sys, {module}\nprint('inspect_ai' in sys.modules)") == "False"
