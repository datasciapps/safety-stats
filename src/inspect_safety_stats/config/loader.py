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
    calibration = plan.inputs.judge_calibration
    if calibration is not None and calibration.validation_manifest is not None:
        validation = calibration.validation_manifest
        if not validation.is_absolute():
            calibration.validation_manifest = (source.parent / validation).resolve()
    return plan


def _plan_relative(path: Path, source: Path) -> str | None:
    """The plan-relative POSIX form of a path that load_plan made absolute, if it has one."""
    if not path.is_absolute():
        return None
    try:
        return path.relative_to(source.parent).as_posix()
    except ValueError:
        return None


def canonical_plan_hash(plan: AnalysisPlan) -> str:
    # load_plan resolves the plan's input paths against its own directory. Hash them in the
    # plan-relative form the author wrote, so the same plan hashes the same wherever it lives.
    value = plan.model_dump(mode="json", by_alias=True)
    source = plan.source_path()
    if source is not None:
        manifest = _plan_relative(plan.inputs.design_manifest, source)
        if manifest is not None:
            value["inputs"]["design_manifest"] = manifest
        calibration = plan.inputs.judge_calibration
        if calibration is not None and calibration.validation_manifest is not None:
            validation = _plan_relative(calibration.validation_manifest, source)
            if validation is not None:
                value["inputs"]["judge_calibration"]["validation_manifest"] = validation
    return sha256_value(value)
