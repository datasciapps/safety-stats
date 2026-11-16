"""Whether a written report renders again from its report.json alone (L38)."""

from __future__ import annotations

from pathlib import Path

from inspect_safety_stats.report.writer import _rendered
from inspect_safety_stats.schema.reports import SafetyReport


def rendering_differences(directory: Path) -> list[str]:
    """The files of a report directory that its report.json, read back, renders differently.

    Nothing beside report.json is read: not the plan, not the design CSV.
    """

    saved = SafetyReport.model_validate_json(
        (directory / "report.json").read_text(encoding="utf-8")
    )
    return [
        name
        for name, text in _rendered(saved).items()
        if (directory / name).read_bytes() != text.encode("utf-8")
    ]
