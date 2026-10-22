"""Every integer field in a plan rejects booleans.

Pydantic's lax mode reads `true` as 1 in an integer field, so `decimal_places: true` silently
printed one decimal place. A plan is a pre-registered document: a typo has to fail. The fields
are found by walking both plan schemas from their roots, so a new integer field, or a new
nested model, cannot opt out by accident.
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Iterator

import pytest
from pydantic import BaseModel, ValidationError

from inspect_safety_stats.config.models import AnalysisPlan, ReportSpec
from inspect_safety_stats.generative.schema import GenerativePlan


def _models_in(annotation: object) -> Iterator[type[BaseModel]]:
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        yield annotation
    for argument in typing.get_args(annotation):
        yield from _models_in(argument)


def _plan_fields() -> dict[str, tuple[type[BaseModel], str]]:
    """Every field of every model reachable from the two plan roots, keyed `module.Model.field`."""
    fields: dict[str, tuple[type[BaseModel], str]] = {}
    seen: set[type[BaseModel]] = set()
    pending: list[type[BaseModel]] = [AnalysisPlan, GenerativePlan]
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        module = model.__module__.removeprefix("inspect_safety_stats.")
        for name, info in model.model_fields.items():
            fields[f"{module}.{model.__qualname__}.{name}"] = (model, name)
            pending.extend(_models_in(info.rebuild_annotation()))
    return dict(sorted(fields.items()))


def _annotation(model: type[BaseModel], field: str) -> object:
    return model.model_fields[field].rebuild_annotation()


def _is_integer(annotation: object) -> bool:
    """An integer, or an optional one: the shapes the boolean probe below can reach."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _is_integer(typing.get_args(annotation)[0])
    if origin in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        return bool(members) and all(_is_integer(member) for member in members)
    return annotation is int


def _contains_integer(annotation: object) -> bool:
    if annotation is int:
        return True
    return any(_contains_integer(argument) for argument in typing.get_args(annotation))


PLAN_FIELDS = _plan_fields()
INTEGER_FIELDS = {
    name: (model, field)
    for name, (model, field) in PLAN_FIELDS.items()
    if _is_integer(_annotation(model, field))
}


def test_the_walk_reaches_every_integer_field_named_in_issue_85() -> None:
    # Guards the discovery itself: a walk that found nothing would make the probe vacuous.
    short_names = {".".join(name.split(".")[-2:]) for name in INTEGER_FIELDS}
    assert short_names >= {
        "ComparisonSpec.bootstrap_replicates",
        "ComparisonSpec.bootstrap_seed",
        "GenerativeMetadata.selection_seed",
        "ReportSpec.decimal_places",
        "RewriteSpec.max_tokens",
        "RewriteValidationSpec.maximum_attempts",
        "RiskMeasureSpec.k",
        "SamplingSpec.minimum_rewrites_per_stratum",
        "SamplingSpec.total_rewrites",
        "TargetSpec.rollouts_per_rewrite",
        "TaxonomySpec.maximum_paths",
    }


def test_no_plan_field_hides_an_integer_from_the_probe() -> None:
    # An integer inside a list, a dict or a wider union would escape the probe below. Give it
    # the PlanInt type from inspect_safety_stats.config.models, and a probe of its own here.
    hidden = [
        name
        for name, (model, field) in PLAN_FIELDS.items()
        if _contains_integer(_annotation(model, field))
        and not _is_integer(_annotation(model, field))
    ]
    assert hidden == []


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize(
    ("model", "field"), [pytest.param(*entry, id=name) for name, entry in INTEGER_FIELDS.items()]
)
def test_every_integer_plan_field_rejects_a_boolean(
    model: type[BaseModel], field: str, value: bool
) -> None:
    key = model.model_fields[field].alias or field
    with pytest.raises(ValidationError) as caught:
        model.model_validate({key: value})
    # Other required fields of the model are reported missing; only this field's error counts.
    messages = [error["msg"] for error in caught.value.errors() if error["loc"][:1] == (key,)]
    assert any("not a boolean" in message for message in messages), messages


def test_only_booleans_are_refused() -> None:
    assert ReportSpec.model_validate({"decimal_places": 4}).decimal_places == 4
    # Lax mode still reads an integral float, as before.
    assert ReportSpec.model_validate({"decimal_places": 4.0}).decimal_places == 4
