from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from .canonicalize import sha256_value
from .models import AnalysisPlan

SCHEMA_VERSIONS = ("2.0", "3.0")


def _check_schema_version(raw: dict[str, Any]) -> None:
    """Accept an unquoted 2.0 or 3.0, and refuse any other version with an explanation.

    YAML and JSON read an unquoted 2.0 as a number, and the schema field is a string, so
    validation used to reject a supported version. An older version failed with a list of
    unknown fields that did not say what had changed.
    """
    if "schema_version" not in raw:
        return
    declared = raw["schema_version"]
    if isinstance(declared, float):
        declared = raw["schema_version"] = str(declared)
    if declared in SCHEMA_VERSIONS:
        return
    supported = " and ".join(repr(version) for version in SCHEMA_VERSIONS)
    if declared == "1.0":
        raise ValueError(
            f"this plan declares schema_version '1.0', and this release reads {supported}. "
            "Schema 2.0 replaced independence_unit and cluster_sensitivity with named "
            "design.independence_profiles, and does not translate them, so a plan's "
            "independence assumptions cannot change silently (docs/independence-profiles.md). "
            "Write a 2.0 or 3.0 plan (docs/plan-reference.md#schema-versions), freeze it "
            "again, and rerun analyze: the Inspect logs can be reused"
        )
    raise ValueError(
        f"unsupported plan schema_version {declared!r}; this release reads {supported} "
        f'plans, declared as, for example, schema_version: "{SCHEMA_VERSIONS[-1]}"'
    )


def load_plan(path: str | Path) -> AnalysisPlan:
    """Read and validate an analysis plan from a YAML (``.yaml``, ``.yml``) or JSON file.

    The plan must declare a schema version this release reads (2.0 or 3.0), and an unquoted
    number is accepted. Relative paths to the design manifest and the judge-calibration
    validation manifest are resolved against the plan's own directory, so a plan works from any
    working directory. Raises ``ValueError`` for any other suffix, a root that is not an
    object, or an unsupported schema version, and pydantic's ``ValidationError`` for an
    invalid plan. The fields are documented in docs/plan-reference.md.
    """

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
    if calibration is not None and calibration.validation_strata is not None:
        strata = calibration.validation_strata
        if not strata.is_absolute():
            calibration.validation_strata = (source.parent / strata).resolve()
    if calibration is not None and calibration.protocol is not None:
        queue = calibration.protocol.queue
        if not queue.is_absolute():
            calibration.protocol.queue = (source.parent / queue).resolve()
    return plan


def _plan_relative(path: Path, source: Path) -> str | None:
    """The plan-relative POSIX form of a path that load_plan made absolute, if it has one."""
    if not path.is_absolute():
        return None
    try:
        return path.relative_to(source.parent).as_posix()
    except ValueError:
        return None


def report_path(path: str | Path, plan: AnalysisPlan) -> str:
    """A path as a report records it: relative to the plan's directory, in POSIX form.

    The same analysis then writes the same bytes in any checkout, and a report shared with
    another institution does not reveal the local directory layout. A URI keeps its form, and
    so does a path with no relative form, such as one on another Windows drive, apart from
    using forward slashes.
    """
    text = str(path)
    if "://" in text:
        return text
    source = plan.source_path()
    if source is None:
        return Path(text).as_posix()
    absolute = Path(text).resolve()
    try:
        return Path(os.path.relpath(absolute, source.parent)).as_posix()
    except ValueError:
        return absolute.as_posix()


def report_plan(plan: AnalysisPlan) -> dict[str, Any]:
    """The resolved plan as a report records it, with input paths in their report form."""
    value = plan.model_dump(mode="json", by_alias=True)
    value["inputs"]["design_manifest"] = report_path(plan.inputs.design_manifest, plan)
    calibration = plan.inputs.judge_calibration
    if calibration is not None and calibration.validation_manifest is not None:
        value["inputs"]["judge_calibration"]["validation_manifest"] = report_path(
            calibration.validation_manifest, plan
        )
    if calibration is not None and calibration.validation_strata is not None:
        value["inputs"]["judge_calibration"]["validation_strata"] = report_path(
            calibration.validation_strata, plan
        )
    if calibration is not None and calibration.protocol is not None:
        value["inputs"]["judge_calibration"]["protocol"]["queue"] = report_path(
            calibration.protocol.queue, plan
        )
    return value


def canonical_plan_hash(plan: AnalysisPlan) -> str:
    return sha256_value(canonical_plan_value(plan))


def canonical_plan_value(plan: AnalysisPlan) -> dict[str, Any]:
    """The plan as its hash covers it."""

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
        if calibration is not None and calibration.validation_strata is not None:
            strata = _plan_relative(calibration.validation_strata, source)
            if strata is not None:
                value["inputs"]["judge_calibration"]["validation_strata"] = strata
        if calibration is not None and calibration.protocol is not None:
            queue = _plan_relative(calibration.protocol.queue, source)
            if queue is not None:
                value["inputs"]["judge_calibration"]["protocol"]["queue"] = queue
    return value
