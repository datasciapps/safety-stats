"""A diagnostic carries the source and effect its code has in the registry."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from inspect_safety_stats.registry import CODES
from inspect_safety_stats.schema.reports import STORED_REPORT, Diagnostic, IntegritySummary


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


def _stored(diagnostic: dict[str, object]) -> Diagnostic:
    return Diagnostic.model_validate(diagnostic, context={STORED_REPORT: True})


def test_a_stored_report_keeps_a_source_and_effect_the_registry_now_gives_otherwise() -> None:
    # Written when the registry said otherwise: reading it back must neither reject it nor
    # rewrite what it says.
    entry = CODES["CONFIG_UNOBSERVED"]
    assert (entry.source, entry.effect) == ("logs", "degraded")
    diagnostic = _stored(
        {
            "code": "CONFIG_UNOBSERVED",
            "severity": "WARNING",
            "source": "manifest",
            "effect": "recorded",
            "message": "no configuration",
        }
    )
    assert (diagnostic.source, diagnostic.effect) == ("manifest", "recorded")


def test_a_stored_report_takes_only_the_fields_it_lacks_from_the_registry() -> None:
    diagnostic = _stored(
        {"code": "COHORT_EMPTY", "severity": "ERROR", "effect": "recorded", "message": "empty"}
    )
    assert (diagnostic.source, diagnostic.effect) == ("logs", "recorded")


def test_a_stored_report_keeps_a_code_the_registry_no_longer_lists() -> None:
    written = {
        "code": "RETIRED_CODE",
        "severity": "WARNING",
        "source": "logs",
        "effect": "degraded",
        "message": "retired",
    }
    assert (_stored(written).source, _stored(written).effect) == ("logs", "degraded")
    bare = _stored({"code": "RETIRED_CODE", "severity": "INFO", "message": "retired"})
    assert (bare.code, bare.source, bare.effect) == ("RETIRED_CODE", None, None)
    with pytest.raises(ValidationError, match="RETIRED_CODE"):
        Diagnostic.model_validate(written)
