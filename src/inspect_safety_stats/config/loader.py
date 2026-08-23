from __future__ import annotations

import json
from pathlib import Path

import yaml

from .canonicalize import sha256_value
from .models import AnalysisPlan


def load_plan(path: str | Path) -> AnalysisPlan:
    source = Path(path).resolve()
    if source.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("analysis plan must be YAML or JSON")
    if not isinstance(raw, dict):
        raise ValueError("analysis plan root must be an object")
    plan = AnalysisPlan.model_validate(raw)
    plan._source_path = source
    manifest = plan.inputs.design_manifest
    if not manifest.is_absolute():
        plan.inputs.design_manifest = (source.parent / manifest).resolve()
    return plan


def canonical_plan_hash(plan: AnalysisPlan) -> str:
    value = plan.model_dump(mode="json", by_alias=True)
    source = plan.source_path()
    manifest = plan.inputs.design_manifest
    if source is not None and manifest.is_absolute():
        try:
            value["inputs"]["design_manifest"] = manifest.relative_to(source.parent).as_posix()
        except ValueError:
            pass
    return sha256_value(value)
