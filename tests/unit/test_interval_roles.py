from __future__ import annotations

import pytest
from pydantic import ValidationError

from inspect_safety_stats.schema.reports import RiskIntervalResult


def _interval(**fields: object) -> RiskIntervalResult:
    return RiskIntervalResult.model_validate(
        {"interval_id": "x", "method": "m", "eligibility": "diagnostic", **fields}
    )


@pytest.mark.parametrize("role", ["primary", "reference"])
def test_only_a_sensitivity_interval_names_a_varied_assumption(role: str) -> None:
    with pytest.raises(ValidationError, match="only a sensitivity interval"):
        _interval(role=role, varied_assumption="independence")


def test_reports_written_before_the_varied_assumption_still_load() -> None:
    # A report.json from before the field has sensitivity intervals without it.
    assert _interval(role="sensitivity").varied_assumption is None
