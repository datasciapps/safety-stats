"""Render a generative report as a self-contained HTML page.

The fixed-design report has its own template, and this is deliberately a second one
rather than an adapter onto it. The two reports answer different questions -- one about a
materialized prompt set, one about a declared generative process -- and section 2 of the
plan requires that difference to stay visible rather than being flattened into a shared
presentation.

The page embeds its data and needs no network access, so it can be opened from a file or
attached to a review without a server.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from inspect_safety_stats.report.visualization import _safe_json

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from inspect_safety_stats.generative.report import GenerativeReport

_TEMPLATE = Path(__file__).with_name("visualization_template.html")
_DATA_MARKER = "__INSPECT_SAFETY_STATS_GENERATIVE_DATA__"


def render_generative_report(report: GenerativeReport) -> str:
    """Return the complete HTML document for a generative report."""

    template = _TEMPLATE.read_text(encoding="utf-8")
    if template.count(_DATA_MARKER) != 1:
        raise RuntimeError("generative template must contain exactly one data marker")
    return template.replace(_DATA_MARKER, _safe_json(report.to_dict()))


def write_generative_report(report: GenerativeReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_generative_report(report), encoding="utf-8", newline="\n"
    )
    return destination
