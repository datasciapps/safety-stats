"""``SafetyReport.presentation`` became a model; reading it as a dictionary still works, warned.

Before the report model recorded its presentation inputs, ``presentation`` was a dictionary,
read as ``report.presentation.get("include_decisions", False)``. Such code keeps working for a
deprecation period, with a DeprecationWarning that names the attribute to read instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inspect_safety_stats.schema.reports import ReportPresentation


def test_dictionary_reads_of_the_presentation_still_work_with_a_deprecation_warning() -> None:
    presentation = ReportPresentation(include_decisions=True, decimal_places=4)

    with pytest.warns(
        DeprecationWarning, match=r"report\.presentation\.include_decisions"
    ) as warned:
        assert presentation["include_decisions"] is True
    # The warning names the line that reads the presentation, not the model.
    assert [Path(item.filename).name for item in warned] == [Path(__file__).name]
    with pytest.warns(DeprecationWarning, match=r"report\.presentation\.decimal_places"):
        assert presentation.get("decimal_places") == 4
    with pytest.warns(DeprecationWarning, match="not a dictionary"):
        assert "include_decisions" in presentation


def test_a_field_the_report_does_not_record_reads_as_a_missing_key() -> None:
    # report.json leaves out a field that is None, so the dictionary had no such key.
    presentation = ReportPresentation()

    with pytest.warns(DeprecationWarning):
        assert presentation.get("primary_independence_profile", "rollout") == "rollout"
    with pytest.warns(DeprecationWarning):
        assert presentation.get("no_such_field") is None
    with pytest.warns(DeprecationWarning):
        assert "judge_calibration" not in presentation
    with pytest.warns(DeprecationWarning), pytest.raises(KeyError, match="decimal_places"):
        presentation["decimal_places"]
