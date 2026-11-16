"""Every integer field in a plan rejects booleans.

Pydantic's lax mode reads `true` as 1 in an integer field, so `decimal_places: true` silently
printed one decimal place. A plan is a pre-registered document: a typo has to fail. The fields
are found by walking both plan schemas from their roots, so a new integer field, or a new
nested model, cannot opt out by accident.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from inspect_safety_stats.config.models import ReportSpec
from tests.plan_fields import annotation, contains, is_scalar, plan_fields

PLAN_FIELDS = plan_fields()
INTEGER_FIELDS = {
    name: (model, field)
    for name, (model, field) in PLAN_FIELDS.items()
    if is_scalar(annotation(model, field), int)
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
        if contains(annotation(model, field), int)
        and not is_scalar(annotation(model, field), int)
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
