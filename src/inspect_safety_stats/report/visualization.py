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
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
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


def _payload_from_report(report: SafetyReport) -> dict[str, Any]:
    return {
        "report": report.model_dump(mode="json"),
        "resolved_design": report._resolved_design,
        "resolved_plan": report._resolved_plan,
    }


def _payload_from_path(source: Path) -> tuple[dict[str, Any], Path]:
    report_path = source / "report.json" if source.is_dir() else source
    if not report_path.exists():
        raise FileNotFoundError(f"report JSON does not exist: {report_path}")
    report = SafetyReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    directory = report_path.parent
    return (
        {
            "report": report.model_dump(mode="json"),
            "resolved_design": _load_design(directory),
            "resolved_plan": _load_plan(directory),
        },
        directory,
    )


def render_visualization(report: SafetyReport) -> str:
    """Render a self-contained interactive HTML visualization for a report."""

    template = _TEMPLATE.read_text(encoding="utf-8")
    if template.count(_DATA_MARKER) != 1:
        raise RuntimeError("visualization template must contain exactly one data marker")
    return template.replace(_DATA_MARKER, _safe_json(_payload_from_report(report)))


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
        payload = _payload_from_report(source)
        destination = Path(output).resolve()
    else:
        payload, directory = _payload_from_path(Path(source).resolve())
        destination = Path(output).resolve() if output is not None else directory / "report.html"

    if destination.exists() and not overwrite:
        raise FileExistsError(f"visualization output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    template = _TEMPLATE.read_text(encoding="utf-8")
    if template.count(_DATA_MARKER) != 1:
        raise RuntimeError("visualization template must contain exactly one data marker")
    destination.write_text(
        template.replace(_DATA_MARKER, _safe_json(payload)),
        encoding="utf-8",
        newline="\n",
    )
    return destination
