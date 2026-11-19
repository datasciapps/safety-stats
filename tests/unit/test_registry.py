"""The registry of diagnostic codes and assumptions, checked against the package source.

A diagnostic code is any string literal in the package that is written like one, in upper
case words joined by underscores, outside the registry itself and the ``__all__`` lists.
The check reads the source rather than running the analysis, so that a code reached only on a
rare path is held to the registry as surely as a common one.
"""

from __future__ import annotations

import ast
import re
import types
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

import inspect_safety_stats
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.registry import (
    ASSUMPTIONS,
    CODES,
    EFFECTS,
    SOURCE_KINDS,
    checked_by,
)

PACKAGE = Path(inspect_safety_stats.__file__).resolve().parent
REGISTRY = PACKAGE / "registry.py"
CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
SEVERITIES = ("ERROR", "WARNING", "INFO")


def _all_lists(tree: ast.Module) -> set[int]:
    """The ids of the string nodes that name exports in ``__all__``."""

    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            ids.update(id(item) for item in ast.walk(node.value))
    return ids


def _source_codes() -> dict[str, list[str]]:
    """Every code-like string literal in the package, with where it is written."""

    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if path == REGISTRY:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        exports = _all_lists(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and CODE_PATTERN.fullmatch(node.value)
                and node.value not in SEVERITIES
                and id(node) not in exports
            ):
                where = f"{path.relative_to(PACKAGE)}:{node.lineno}"
                found.setdefault(node.value, []).append(where)
    return found


def _emitted_severities() -> dict[str, set[str]]:
    """The severity of every ``Diagnostic(code=..., severity=...)`` written with literals."""

    found: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "Diagnostic"):
                continue
            keywords = {item.arg: item.value for item in node.keywords}
            code, severity = keywords.get("code"), keywords.get("severity")
            if isinstance(code, ast.Constant) and isinstance(severity, ast.Constant):
                found.setdefault(code.value, set()).add(severity.value)
    return found


DEGRADED_DETAIL = "integrity_degraded"


def _degraded_details() -> tuple[dict[str, list[bool]], list[str]]:
    """Which ``Diagnostic(code=...)`` sites write ``integrity_degraded: True`` in ``details``.

    The first value maps each code written as a literal to one flag per site: whether the
    site's ``details`` is a dict literal with that key set to True. The second lists every
    other place in the package that writes the key, which the flags would miss.
    """

    marks: dict[str, list[bool]] = {}
    stray: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        counted: set[int] = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "Diagnostic"):
                continue
            keywords = {item.arg: item.value for item in node.keywords}
            code, details = keywords.get("code"), keywords.get("details")
            marked = False
            if isinstance(details, ast.Dict):
                for key, value in zip(details.keys, details.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == DEGRADED_DETAIL:
                        counted.add(id(key))
                        marked = isinstance(value, ast.Constant) and value.value is True
            if isinstance(code, ast.Constant):
                marks.setdefault(code.value, []).append(marked)
            elif marked:
                stray.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
        stray.extend(
            f"{path.relative_to(PACKAGE)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and node.value == DEGRADED_DETAIL
            and id(node) not in counted
        )
    return marks, stray


def test_the_codes_whose_sites_mark_integrity_degraded_are_the_degraded_ones() -> None:
    # Integrity reads the registry's effect (analysis.inputs._integrity_status); the detail is
    # what the emitting site says of its own finding. An entry whose effect disagrees with its
    # sites, or a site that marks a code the registry does not degrade, fails here.
    marks, stray = _degraded_details()
    degraded = sorted(code for code, entry in CODES.items() if entry.effect == "degraded")
    assert stray == [], "integrity_degraded written outside a Diagnostic(code=...) literal"
    assert sorted(code for code, sites in marks.items() if any(sites)) == degraded
    assert sorted(code for code in degraded if not all(marks.get(code, [False]))) == []


def test_every_code_in_the_source_has_a_registry_entry() -> None:
    missing = {code: sites for code, sites in _source_codes().items() if code not in CODES}
    assert missing == {}


def test_every_registry_entry_names_a_code_the_source_writes() -> None:
    assert sorted(set(CODES) - set(_source_codes())) == []


def test_entries_list_every_severity_their_code_is_emitted_with() -> None:
    wrong = {
        code: sorted(severities)
        for code, severities in _emitted_severities().items()
        if code in CODES and not severities <= set(CODES[code].severities)
    }
    assert wrong == {}


@pytest.mark.parametrize("code", sorted(CODES))
def test_code_entries_are_complete_and_consistent(code: str) -> None:
    entry = CODES[code]
    assert entry.code == code
    assert entry.severities, "an entry lists at least one severity"
    assert set(entry.severities) <= set(SEVERITIES)
    assert list(entry.severities) == sorted(set(entry.severities), key=SEVERITIES.index)
    assert entry.source in SOURCE_KINDS
    assert entry.effect in EFFECTS
    # Any ERROR makes the integrity status invalid (analysis.inputs._integrity_status).
    assert (entry.effect == "invalid") == ("ERROR" in entry.severities)
    for text in (entry.assumption, entry.consequence, entry.remedy):
        assert text.strip() == text and text.endswith("."), text
    assert set(entry.assumptions) <= set(ASSUMPTIONS)


def test_the_registry_lists_its_codes_in_order_once() -> None:
    from inspect_safety_stats import registry

    codes = [entry.code for entry in registry._CODES]
    assert codes == sorted(set(codes))


def _field(model: type[BaseModel], path: str) -> typing.Any:
    """The pydantic field a dotted plan path names; ``[]`` and ``<id>`` step into items."""

    field = None
    current: typing.Any = model
    for part in path.split("."):
        if part == "<id>":
            current = _item_type(current, dict)
            continue
        name = part.removesuffix("[]")
        assert isinstance(current, type) and issubclass(current, BaseModel), path
        field = current.model_fields[name]
        current = field.annotation
        if part.endswith("[]"):
            current = _item_type(current, list)
        else:
            current = _model_in(current)
    return field


def _model_in(annotation: typing.Any) -> typing.Any:
    """The model of an optional or plain annotation, or the annotation itself."""

    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        models = [
            item
            for item in typing.get_args(annotation)
            if isinstance(item, type) and issubclass(item, BaseModel)
        ]
        return models[0] if len(models) == 1 else annotation
    return annotation


def _item_type(annotation: typing.Any, container: type) -> typing.Any:
    assert typing.get_origin(annotation) is container, annotation
    return _model_in(typing.get_args(annotation)[-1])


def _as_plan_text(value: typing.Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


@pytest.mark.parametrize("assumption", sorted(ASSUMPTIONS))
def test_assumption_entries_name_real_plan_fields_and_their_defaults(assumption: str) -> None:
    entry = ASSUMPTIONS[assumption]
    assert entry.id == assumption
    assert entry.source in SOURCE_KINDS
    for text in (entry.statement, entry.if_wrong, entry.remedy):
        assert text.strip() == text and text.endswith("."), text
    if entry.plan_field is None:
        assert entry.default is None
        return
    field = _field(AnalysisPlan, entry.plan_field)
    if field.is_required():
        assert entry.default is None
    else:
        assert entry.default == _as_plan_text(field.get_default(call_default_factory=True))


def test_checked_by_reads_the_code_entries() -> None:
    for assumption in ASSUMPTIONS:
        codes = checked_by(assumption)
        assert list(codes) == sorted(codes)
        assert all(assumption in CODES[code].assumptions for code in codes)
    with pytest.raises(KeyError):
        checked_by("no_such_assumption")
