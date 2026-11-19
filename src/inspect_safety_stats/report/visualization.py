from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import yaml

from inspect_safety_stats.config.canonicalize import canonical_value
from inspect_safety_stats.report.artifacts import write_text
from inspect_safety_stats.schema.reports import STORED_REPORT, SafetyReport

_TEMPLATE = Path(__file__).with_name("visualization_template.html")
_DATA_MARKER = "__INSPECT_SAFETY_STATS_DATA__"
# The decision policy's section goes before this one. It is written only for a report with a
# policy, so that a report without one keeps the template's bytes.
_AUDIT_SECTION = '    <section aria-labelledby="audit-heading">'


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
    data = report.model_dump(mode="json")
    # The page makes scope.what_results_establish a link. `visualize` renders any report.json
    # it is given, and a javascript: or data: address there would run in the page when the
    # link is clicked, so only an https:// address reaches the page. The package writes an
    # https:// address, so its own reports keep their bytes.
    address = data["scope"].get("what_results_establish")
    if address is not None and not (isinstance(address, str) and address.startswith("https://")):
        del data["scope"]["what_results_establish"]
    return {"report": data}


def _report_from_path(source: Path) -> tuple[SafetyReport, Path]:
    report_path = source / "report.json" if source.is_dir() else source
    if not report_path.exists():
        raise FileNotFoundError(f"report JSON does not exist: {report_path}")
    report = SafetyReport.model_validate_json(
        report_path.read_text(encoding="utf-8"), context={STORED_REPORT: True}
    )
    directory = report_path.parent
    _fill_from_directory(report, directory)
    return report, directory


def _decision_policy_section(report: SafetyReport) -> str:
    """A static section on the decision policy (experimental), or "" without one."""

    from inspect_safety_stats.report.summary import decision_policy_note

    record = report.decision_policy
    if record is None:
        return ""

    def cell(text: str, numeric: bool = False) -> str:
        return f'<td class="num">{text}</td>' if numeric else f"<td>{html.escape(text)}</td>"

    def table(head: list[str], rows: list[list[str]]) -> str:
        header = "".join(f"<th>{html.escape(item)}</th>" for item in head)
        body = "".join(f"<tr>{''.join(row)}</tr>" for row in rows)
        return (
            '      <div class="table-wrap"><table class="data-table">'
            f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>\n"
        )

    after = {True: "yes", False: "no", None: "unknown"}
    note = decision_policy_note(report, code=lambda text: text, pointer=False)
    versions = [
        [
            cell(item.id),
            cell(item.set_at),
            cell(after[item.set_after_first_evaluation]),
            cell(", ".join(item.thresholds)),
            cell(item.reason or "—"),
        ]
        for item in record.versions
    ]
    claims = [
        [
            cell(item.claim_id),
            cell(f"{item.acceptable_upper:.4f}", numeric=True),
            cell(f"{item.unacceptable_lower:.4f}", numeric=True),
            cell(item.set_by),
            cell(f"{item.plan_acceptable_upper:.4f}", numeric=True),
            cell(f"{item.plan_unacceptable_lower:.4f}", numeric=True),
        ]
        for item in record.claims
    ]
    return (
        '    <section aria-labelledby="decision-policy-heading">\n'
        '      <div class="section-head"><h2 id="decision-policy-heading">Decision policy</h2>'
        '<span class="section-note">Experimental</span></div>\n'
        f'      <p class="coverage-note">{html.escape(note)} Each version\'s time is as the '
        "policy states it; nothing in the analysis checks it.</p>\n"
        + table(["Version", "Set at", "After first evaluation", "Claims", "Reason"], versions)
        + table(
            [
                "Claim",
                "Acceptable",
                "Unacceptable",
                "Set by",
                "Plan's acceptable",
                "Plan's unacceptable",
            ],
            claims,
        )
        + "    </section>\n\n"
    )


def render_visualization(report: SafetyReport) -> str:
    """Render a self-contained interactive HTML visualization for a report."""

    template = _TEMPLATE.read_text(encoding="utf-8")
    if template.count(_DATA_MARKER) != 1:
        raise RuntimeError("visualization template must contain exactly one data marker")
    page = template.replace(_DATA_MARKER, _safe_json(_payload(report)))
    policy = _decision_policy_section(report)
    if not policy:
        return page
    # After the data: the inline JSON escapes "<", so the audit section is found only in the
    # template, and the policy's own text cannot be taken for the data marker.
    if template.count(_AUDIT_SECTION) != 1:
        raise RuntimeError("visualization template must contain exactly one audit section")
    return page.replace(_AUDIT_SECTION, policy + _AUDIT_SECTION)


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
    write_text(destination, render_visualization(report))
    return destination
