from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

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
    number is accepted. Relative paths to the design manifest, the judge-calibration
    validation manifest and the judge flip manifest are resolved against the plan's own
    directory, so a plan works from any working directory. Raises ``ValueError`` for any
    other suffix, a root that is not an object, or an unsupported schema version, and
    pydantic's ``ValidationError`` for an invalid plan. The fields are documented in
    docs/plan-reference.md.
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
    flip = plan.inputs.judge_flip
    if flip is not None and not flip.manifest.is_absolute():
        flip.manifest = (source.parent / flip.manifest).resolve()
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



def report_message(
    message: str, plan: AnalysisPlan, paths: Iterable[str | os.PathLike[str] | None]
) -> str:
    """An input error's message with each of ``paths`` in it as a report records it.

    An exception names the file it failed on as it opened it, usually by the absolute path the
    plan loader resolved it to. A report names its inputs with ``report_path``, so the message
    follows suit: the same analysis then writes the same bytes in any checkout, and a shared
    report does not reveal the local directory layout. A URI, and a path given relative to the
    working directory, are left as they are.
    """

    forms: dict[str, str] = {}
    for path in paths:
        if path is None:
            continue
        text = os.fspath(path)
        if "://" in text:
            continue
        form = report_path(text, plan)
        for absolute in (text, str(Path(text).resolve())):
            if Path(absolute).is_absolute():
                forms[absolute] = form
                # OSError messages show the path's repr, which doubles Windows separators.
                forms[absolute.replace("\\", "\\\\")] = form
    if not forms:
        return message
    pattern = "|".join(re.escape(text) for text in sorted(forms, key=len, reverse=True))
    return re.sub(pattern, lambda match: forms[match.group(0)], message)

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
    if plan.inputs.judge_flip is not None:
        value["inputs"]["judge_flip"]["manifest"] = report_path(
            plan.inputs.judge_flip.manifest, plan
        )
    if calibration is not None and calibration.protocol is not None:
        value["inputs"]["judge_calibration"]["protocol"]["queue"] = report_path(
            calibration.protocol.queue, plan
        )
    return value


def plan_defaults(plan: AnalysisPlan) -> list[dict[str, Any]]:
    """Every plan field the author left out, with the value its default gave it.

    The complement of ``canonical_plan_value``: together they are the plan an analysis ran.
    Each entry is ``{"path": ..., "value": ...}``. A path joins field names with ``.``, writes
    a list item's position as ``[i]`` and a mapping entry by its key, as in
    ``outputs.risk_measures[0].beta`` or ``cohorts.candidate.require_uniform``. A section left
    out whole has an entry for each of its fields. A field that does not exist in this plan,
    such as ``inputs.sample_join`` under schema 3.0, has no entry. Entries come in the plan's
    field order.
    """

    entries: list[dict[str, Any]] = []
    _collect_defaults(plan, "", entries)
    return entries


def _collect_defaults(model: BaseModel, prefix: str, entries: list[dict[str, Any]]) -> None:
    for name, info in type(model).model_fields.items():
        path = f"{prefix}.{info.alias or name}" if prefix else info.alias or name
        value = getattr(model, name)
        if name not in model.model_fields_set and isinstance(value, BaseModel):
            # A section left out whole: each of its fields is a default. Its dump would leave
            # out experimental fields, so it is not one entry.
            _collect_defaults(value, path, entries)
        elif name not in model.model_fields_set:
            # A None field named in absent_when_none does not exist in this plan, as
            # sample_join does not under schema 3.0. Every other field left out is a default
            # the analysis applied, experimental ones included, even where a model's dump
            # leaves it out to keep older outputs' bytes.
            if value is None and name in getattr(model, "absent_when_none", ()):
                continue
            entries.append({"path": path, "value": to_jsonable_python(value, by_alias=True)})
        elif isinstance(value, BaseModel):
            _collect_defaults(value, path, entries)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, BaseModel):
                    _collect_defaults(item, f"{path}[{index}]", entries)
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, BaseModel):
                    _collect_defaults(item, f"{path}.{key}", entries)


def canonical_plan_hash(plan: AnalysisPlan) -> str:
    """The SHA-256 of what the plan's author wrote, as ``canonical_plan_value`` gives it."""

    return sha256_value(canonical_plan_value(plan))


def canonical_plan_value(plan: AnalysisPlan) -> dict[str, Any]:
    """The plan as its hash covers it: the fields its author wrote, validated and normalized.

    A field the plan leaves out is left out here too, whatever its default, so a release that
    adds a defaulted field or changes a default leaves the hash of every unchanged plan file
    as it was (#56). A field written at its default value is kept: the plan says it, and its
    meaning does not follow a later default. Values are in their validated form, so that, for
    example, an unquoted ``2.0`` and ``"2.0"`` hash the same.
    """

    # load_plan resolves the plan's input paths against its own directory. Hash them in the
    # plan-relative form the author wrote, so the same plan hashes the same wherever it lives.
    value = plan.model_dump(mode="json", by_alias=True, exclude_unset=True)
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
        flip = plan.inputs.judge_flip
        if flip is not None:
            flip_manifest = _plan_relative(flip.manifest, source)
            if flip_manifest is not None:
                value["inputs"]["judge_flip"]["manifest"] = flip_manifest
        if calibration is not None and calibration.protocol is not None:
            queue = _plan_relative(calibration.protocol.queue, source)
            if queue is not None:
                value["inputs"]["judge_calibration"]["protocol"]["queue"] = queue
    return value
