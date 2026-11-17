"""A diagnostic carries the source and effect its code has in the registry."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from inspect_safety_stats.registry import CODES
from inspect_safety_stats.schema.reports import Diagnostic, IntegritySummary


@pytest.mark.parametrize("code", sorted(CODES))
def test_a_diagnostic_takes_source_and_effect_from_its_registry_entry(code: str) -> None:
    entry = CODES[code]
    diagnostic = Diagnostic(code=code, severity=entry.severities[0], message="found")
    assert (diagnostic.source, diagnostic.effect) == (entry.source, entry.effect)


def test_report_json_carries_source_and_effect_after_severity() -> None:
    summary = IntegritySummary(
        status="invalid",
        diagnostics=[Diagnostic(code="COHORT_EMPTY", severity="ERROR", message="empty")],
    )
    written = json.loads(summary.model_dump_json())["diagnostics"][0]
    assert list(written)[:4] == ["code", "severity", "source", "effect"]
    assert (written["source"], written["effect"]) == ("logs", "invalid")


def test_a_report_written_before_the_fields_existed_reads_them_from_the_registry() -> None:
    old = {"code": "COHORT_EMPTY", "severity": "ERROR", "message": "empty", "scope": None}
    diagnostic = Diagnostic.model_validate(old)
    assert (diagnostic.source, diagnostic.effect) == ("logs", "invalid")


def test_a_code_the_registry_does_not_list_is_rejected() -> None:
    with pytest.raises(ValidationError, match="NOT_A_CODE"):
        Diagnostic(code="NOT_A_CODE", severity="ERROR", message="found")


@pytest.mark.parametrize(("field", "value"), [("source", "plan"), ("effect", "recorded")])
def test_a_source_or_effect_that_differs_from_the_registry_is_rejected(
    field: str, value: str
) -> None:
    with pytest.raises(ValidationError, match=field):
        Diagnostic.model_validate(
            {"code": "COHORT_EMPTY", "severity": "ERROR", "message": "empty", field: value}
        )
