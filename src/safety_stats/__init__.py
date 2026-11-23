"""Fixed-design statistical risk analysis for Inspect AI evaluations.

The public names are imported on first use, so that importing the package, or running a
command that reads no logs, does not load Inspect AI, pandas and SciPy.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# ORDER MATTERS: the subpackage import must come before the globals().pop() below. The
# plan-validation subpackage shares its name with the exported validate() function, and
# importing a subpackage for the first time binds it as an attribute of this package. Loading
# it here and then dropping that binding leaves "validate" to __getattr__, which serves the
# function. If the subpackage were first imported later, for example by api, the binding would
# silently replace the function and `from safety_stats import validate` would return
# the subpackage.
import safety_stats.validate as _plan_validation  # noqa: F401
from safety_stats.version import __version__

globals().pop("validate", None)

if TYPE_CHECKING:
    from safety_stats.api import analyze, validate
    from safety_stats.config import AnalysisPlan, load_plan
    from safety_stats.report.visualization import (
        render_visualization,
        write_visualization,
    )
    from safety_stats.schema.reports import SafetyReport

__all__ = [
    "AnalysisPlan",
    "SafetyReport",
    "__version__",
    "analyze",
    "load_plan",
    "render_visualization",
    "validate",
    "write_visualization",
]

_EXPORTS = {
    "AnalysisPlan": "safety_stats.config",
    "SafetyReport": "safety_stats.schema.reports",
    "analyze": "safety_stats.api",
    "load_plan": "safety_stats.config",
    "render_visualization": "safety_stats.report.visualization",
    "validate": "safety_stats.api",
    "write_visualization": "safety_stats.report.visualization",
}

# Subpackages stay reachable as attributes, as they were when this module imported them.
_SUBMODULES = {
    "api",
    "cli",
    "config",
    "decision",
    "generative",
    "ingest",
    "report",
    "schema",
    "stats",
    "version",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
