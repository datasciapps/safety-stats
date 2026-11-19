from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from inspect_safety_stats.config.canonicalize import canonical_value
from inspect_safety_stats.schema.reports import SafetyReport

_TEMPLATE = Path(__file__).with_name("visualization_template.html")
_DATA_MARKER = "__INSPECT_SAFETY_STATS_DATA__"


def _safe_json(value: Any) -> str:
    """Serialize JSON for an inline script without permitting a closing script tag."""

    return (
        json.dumps(
            canonical_value(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def _load_design(directory: Path) -> list[dict[str, Any]]:
    path = directory / "design.resolved.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row.get("weight") not in {None, ""}:
            row["weight"] = float(row["weight"])
    return rows


def _load_plan(directory: Path) -> dict[str, Any]:
    path = directory / "plan.resolved.yaml"
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _fill_from_directory(report: SafetyReport, directory: Path) -> None:
    """Fill what a report written before the visualization's inputs were in report.json lacks,
    from the files beside it. A report that records them is left as it is."""

    if "prompts" not in report.design:
        report.design["prompts"] = _load_design(directory)
    if report.presentation.primary_independence_profile is None:
        independence = (
            ((_load_plan(directory).get("outputs") or {}).get("direct_mean_intervals") or {})
            .get("independence")
            or {}
        )
        primary = independence.get("primary")
        if primary is not None:
            report.presentation.primary_independence_profile = str(primary)


def _payload(report: SafetyReport) -> dict[str, Any]:
    # The page reads the report alone, so that a saved report.json renders it again.
    return {"report": report.model_dump(mode="json")}


def _report_from_path(source: Path) -> tuple[SafetyReport, Path]:
    report_path = source / "report.json" if source.is_dir() else source
    if not report_path.exists():
        raise FileNotFoundError(f"report JSON does not exist: {report_path}")
    report = SafetyReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    directory = report_path.parent
    _fill_from_directory(report, directory)
    return report, directory


def render_visualization(report: SafetyReport) -> str:
    """Render a self-contained interactive HTML visualization for a report."""

    template = _TEMPLATE.read_text(encoding="utf-8")
    if template.count(_DATA_MARKER) != 1:
        raise RuntimeError("visualization template must contain exactly one data marker")
    return template.replace(_DATA_MARKER, _safe_json(_payload(report)))


def write_visualization(
    source: SafetyReport | str | Path,
    output: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a self-contained HTML visualization from a report object or report directory."""

    if isinstance(source, SafetyReport):
        if output is None:
            raise ValueError("output is required when visualizing a SafetyReport object")
        report = source
        destination = Path(output).resolve()
    else:
        report, directory = _report_from_path(Path(source).resolve())
        destination = Path(output).resolve() if output is not None else directory / "report.html"

    if destination.exists() and not overwrite:
        raise FileExistsError(f"visualization output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_visualization(report), encoding="utf-8", newline="\n")
    return destination
