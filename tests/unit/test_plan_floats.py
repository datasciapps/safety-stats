"""Every float field in a plan rejects booleans (#107).

Pydantic's lax mode reads `true` as 1.0 and `false` as 0.0 in a float field, and in the values
of a dict of floats. So `acceptable_upper: false` became a claim limit of 0, and reweighting
weights `{a: true, b: false}` read as the distribution (1, 0), which passes the sums-to-one
check. The fields are found by walking both plan schemas from their roots.
"""

from __future__ import annotations

import types
import typing

import pytest
from pydantic import BaseModel, ValidationError

from inspect_safety_stats.config.models import JudgeRateRangeSpec
from tests.plan_fields import annotation, contains, is_scalar, plan_fields


def _is_float_mapping(annotation: object) -> bool:
    """A dict of floats, or an optional one: the shapes the dict probe below can reach."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _is_float_mapping(typing.get_args(annotation)[0])
    if origin in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        return bool(members) and all(_is_float_mapping(member) for member in members)
    if origin is dict:
        return is_scalar(typing.get_args(annotation)[1], float)
    return False


PLAN_FIELDS = plan_fields()
FLOAT_FIELDS = {
    name: (model, field)
    for name, (model, field) in PLAN_FIELDS.items()
    if is_scalar(annotation(model, field), float)
}
FLOAT_MAPPINGS = {
    name: (model, field)
    for name, (model, field) in PLAN_FIELDS.items()
    if _is_float_mapping(annotation(model, field))
}


def _short(names: dict[str, tuple[type[BaseModel], str]]) -> set[str]:
    return {".".join(name.split(".")[-2:]) for name in names}


def test_the_walk_reaches_every_float_field_named_in_issue_107() -> None:
    # Guards the discovery itself: a walk that found nothing would make the probes vacuous.
    assert _short(FLOAT_FIELDS) >= {
        "ArmValueSpec.weight",
        "ClaimSpec.acceptable_upper",
        "ClaimSpec.unacceptable_lower",
        "ComparisonSetSpec.allocation_weight",
        "ComparisonSpec.allocation_weight",
        "FailureIfSpec.value",
        "InferenceSpec.alpha",
        "JudgeCalibrationSpec.alpha",
        "JudgeRateRangeSpec.lower",
        "JudgeRateRangeSpec.upper",
        "MultiplicityFamilySpec.alpha",
        "RewriteSpec.temperature",
        "RiskMeasureSpec.beta",
        "RiskMeasureSpec.lambda_",
        "RiskMeasureSpec.threshold",
    }
    assert _short(FLOAT_MAPPINGS) >= {"PromptBoundsSpec.explicit_alpha", "ReweightingSpec.weights"}


def test_no_plan_field_hides_a_float_from_the_probes() -> None:
    # A float inside a list, a tuple or a wider union would escape both probes. Give it the
    # PlanFloat type from inspect_safety_stats.config.models, and a probe of its own here.
    hidden = [
        name
        for name, (model, field) in PLAN_FIELDS.items()
        if contains(annotation(model, field), float)
        and not is_scalar(annotation(model, field), float)
        and not _is_float_mapping(annotation(model, field))
    ]
    assert hidden == []


def _messages(error: ValidationError, location: tuple[str, ...]) -> list[str]:
    # Other required fields of the model are reported missing; only this location counts.
    return [item["msg"] for item in error.errors() if item["loc"][: len(location)] == location]


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize(
    ("model", "field"), [pytest.param(*entry, id=name) for name, entry in FLOAT_FIELDS.items()]
)
def test_every_float_plan_field_rejects_a_boolean(
    model: type[BaseModel], field: str, value: bool
) -> None:
    key = model.model_fields[field].alias or field
    with pytest.raises(ValidationError) as caught:
        model.model_validate({key: value})
    messages = _messages(caught.value, (key,))
    assert any("not a boolean" in message for message in messages), messages


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize(
    ("model", "field"), [pytest.param(*entry, id=name) for name, entry in FLOAT_MAPPINGS.items()]
)
def test_every_float_in_a_plan_mapping_rejects_a_boolean(
    model: type[BaseModel], field: str, value: bool
) -> None:
    key = model.model_fields[field].alias or field
    with pytest.raises(ValidationError) as caught:
        model.model_validate({key: {"probe": value}})
    messages = _messages(caught.value, (key, "probe"))
    assert any("not a boolean" in message for message in messages), messages


def test_integers_still_validate_as_floats() -> None:
    accuracy = JudgeRateRangeSpec.model_validate({"lower": 0, "upper": 1})
    assert (accuracy.lower, accuracy.upper) == (0.0, 1.0)
    assert isinstance(accuracy.lower, float)
