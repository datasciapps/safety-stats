from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .canonicalize import sha256_value
from .models import AnalysisPlan

SCHEMA_VERSION = "2.0"


def _check_schema_version(raw: dict[str, Any]) -> None:
    """Accept an unquoted 2.0, and refuse any other version with an explanation.

    YAML and JSON read an unquoted 2.0 as a number, and the schema field is the string "2.0",
    so validation used to reject the current version. An older version failed with a list of
    unknown fields that did not say what had changed.
    """
    if "schema_version" not in raw:
        return
    declared = raw["schema_version"]
    if isinstance(declared, float):
        declared = raw["schema_version"] = str(declared)
    if declared == SCHEMA_VERSION:
        return
    if declared == "1.0":
        raise ValueError(
            f"this plan declares schema_version '1.0', and this release reads "
            f"{SCHEMA_VERSION!r}. Schema 2.0 replaced independence_unit and "
            "cluster_sensitivity with named design.independence_profiles, and does not "
            "translate them, so a plan's independence assumptions cannot change silently "
            "(docs/independence-profiles.md). Write a 2.0 plan, freeze it again, and rerun "
            "analyze: the Inspect logs can be reused"
        )
    raise ValueError(
        f"unsupported plan schema_version {declared!r}; this release reads "
        f'{SCHEMA_VERSION!r} plans, declared as schema_version: "{SCHEMA_VERSION}"'
    )


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
    _check_schema_version(raw)
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
