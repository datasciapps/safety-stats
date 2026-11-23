"""Every field of both plan schemas, found by walking them from their roots.

The tests that a plan's numeric fields reject booleans find their fields this way, so a new
field, or a new nested model, cannot opt out by accident.
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Iterator

from pydantic import BaseModel

from safety_stats.config.models import AnalysisPlan
from safety_stats.generative.schema import GenerativePlan


def _models_in(annotation: object) -> Iterator[type[BaseModel]]:
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        yield annotation
    for argument in typing.get_args(annotation):
        yield from _models_in(argument)


def plan_fields() -> dict[str, tuple[type[BaseModel], str]]:
    """Every field of every model reachable from the two plan roots, keyed `module.Model.field`."""
    fields: dict[str, tuple[type[BaseModel], str]] = {}
    seen: set[type[BaseModel]] = set()
    pending: list[type[BaseModel]] = [AnalysisPlan, GenerativePlan]
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        module = model.__module__.removeprefix("safety_stats.")
        for name, info in model.model_fields.items():
            fields[f"{module}.{model.__qualname__}.{name}"] = (model, name)
            pending.extend(_models_in(info.rebuild_annotation()))
    return dict(sorted(fields.items()))


def annotation(model: type[BaseModel], field: str) -> object:
    return model.model_fields[field].rebuild_annotation()


def is_scalar(annotation: object, scalar: type) -> bool:
    """`scalar`, or an optional one: the shapes a boolean probe can reach."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return is_scalar(typing.get_args(annotation)[0], scalar)
    if origin in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        return bool(members) and all(is_scalar(member, scalar) for member in members)
    return annotation is scalar


def contains(annotation: object, scalar: type) -> bool:
    if annotation is scalar:
        return True
    return any(contains(argument, scalar) for argument in typing.get_args(annotation))
