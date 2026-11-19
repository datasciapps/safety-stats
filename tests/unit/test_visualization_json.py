"""The report data embedded in report.html cannot end its script or break a line of it."""

from __future__ import annotations

import json

from inspect_safety_stats.report.visualization import _safe_json


def test_embedded_report_data_escapes_what_could_end_or_break_its_script() -> None:
    # "</script>" would end the data script; U+2028 and U+2029 end a line in older
    # JavaScript string literals. Each is written as a JSON escape instead.
    value = {"text": "a b c</script><!-- & -->"}

    embedded = _safe_json(value)

    for character in (" ", " ", "<", ">", "&"):
        assert character not in embedded, repr(character)
    assert "\\u2028" in embedded
    assert "\\u2029" in embedded
    assert json.loads(embedded) == value
