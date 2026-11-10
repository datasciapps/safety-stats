from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# The public names are imported on first use, so that importing this package, or one of its
# Inspect-free modules, does not load the Inspect reader and with it Inspect AI.
if TYPE_CHECKING:
    from .inspect_logs import ingest_inspect_logs
    from .manifest import ResolvedGroup, load_manifest, resolve_groups
    from .records import EvaluationSource, NormalizedData
    from .rollout_files import RolloutFiles

__all__ = [
    "EvaluationSource",
    "NormalizedData",
    "ResolvedGroup",
    "ingest_inspect_logs",
    "load_manifest",
    "resolve_groups",
    "RolloutFiles",
]

_EXPORTS = {
    "EvaluationSource": "inspect_safety_stats.ingest.records",
    "NormalizedData": "inspect_safety_stats.ingest.records",
    "ResolvedGroup": "inspect_safety_stats.ingest.manifest",
    "ingest_inspect_logs": "inspect_safety_stats.ingest.inspect_logs",
    "load_manifest": "inspect_safety_stats.ingest.manifest",
    "resolve_groups": "inspect_safety_stats.ingest.manifest",
    "RolloutFiles": "inspect_safety_stats.ingest.rollout_files",
}

# Submodules stay reachable as attributes, as they were when this module imported them.
_SUBMODULES = {
    "files",
    "inspect_logs",
    "manifest",
    "normalize",
    "outcomes",
    "records",
    "rollout_files",
    "validation",
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
