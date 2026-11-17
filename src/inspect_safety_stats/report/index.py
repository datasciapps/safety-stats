"""index.json and index.md: a map of a report directory, for agents and other tools.

The index names every file the report wrote, what each holds and how its rows line up with
``report.json``, then copies the summary level of ``report.json`` (``summary``): the headline
estimates, the decisions when they are shown, and the errors and warnings by code, each with a
JSON pointer into ``report.json``. A headline that a CSV holds a row of also gives that row.
``index.md`` holds the same content as Markdown. ``docs/agent-guide.md`` says how to read both.

Like the other renderings, the index is a function of the report model, plus the names of the
files written beside it, which depend on the plan (``observations.csv``) and so are not in
``report.json``; ``checksums.json`` lists them.
"""

from __future__ import annotations

from typing import Any

from inspect_safety_stats.config.canonicalize import canonical_value
from inspect_safety_stats.report.artifacts import CHECKSUMS, json_text
from inspect_safety_stats.schema.reports import SafetyReport

INDEX_JSON = "index.json"
INDEX_MD = "index.md"
INDEX_VERSION = "1"
# Absolute, as the report is read outside the repository; source docs/agent-guide.md.
AGENT_GUIDE_URL = "https://github.com/aduncan001/safety_stats/blob/main/docs/agent-guide.md"
SUMMARY_PAGE = "report.md"

# What each file holds, and the report.json list whose items its data rows are, in order.
_ARTIFACTS: dict[str, tuple[str, str, str | None]] = {
    "report.json": ("json", "Every result, diagnostic and setting; the source of all others.", None),  # noqa: E501
    "report.md": ("markdown", "One-page summary for people: read first.", None),
    "report-full.md": ("markdown", "The full report: every interval, sensitivity and message.", None),  # noqa: E501
    "report.html": ("html", "Interactive report; embeds report.json.", None),
    "plan.resolved.yaml": ("yaml", "The plan as analysed, defaults filled in.", None),
    "design.resolved.csv": ("csv", "One row per design prompt.", "/design/prompts"),
    "hierarchy-risks.csv": ("csv", "One row per hierarchy node risk.", "/hierarchy_risks"),
    "comparisons.csv": ("csv", "One row per comparison result.", "/comparisons"),
    "comparison-sets.csv": ("csv", "One row per comparison set.", "/comparison_sets"),
    "observations.csv": ("csv", "One row per normalised observation (analysis input).", None),
    "outcome-mapping-sensitivity.csv": (
        "csv",
        "Each displayed result under each alternative outcome mapping.",
        None,
    ),
    "claims.json": ("json", "One record per claim (experimental).", None),
    INDEX_JSON: ("json", "This index.", None),
    INDEX_MD: ("markdown", "This index as Markdown.", None),
    CHECKSUMS: ("json", "SHA-256 of every other file.", None),
}


def _artifacts(names: list[str]) -> list[dict[str, Any]]:
    unknown = sorted(set(names) - set(_ARTIFACTS))
    if unknown:
        raise ValueError(f"the index does not describe {', '.join(unknown)}")
    entries = []
    for name in sorted({*names, INDEX_JSON, INDEX_MD, CHECKSUMS}):
        file_format, holds, rows_of = _ARTIFACTS[name]
        entry: dict[str, Any] = {"name": name, "format": file_format, "holds": holds}
        if rows_of is not None:
            # Data row n (1-based, after the header) is the item at rows_of/<n - 1>.
            entry["rows_of"] = rows_of
        entries.append(entry)
    return entries


def _csv_row(pointer: str, artifacts: list[dict[str, Any]]) -> dict[str, Any] | None:
    for artifact in artifacts:
        rows_of = artifact.get("rows_of")
        if rows_of and pointer.startswith(rows_of + "/"):
            index = pointer[len(rows_of) + 1 :]
            if index.isdigit():
                return {"file": artifact["name"], "row": int(index) + 1}
    return None


def index_document(report: SafetyReport, names: list[str]) -> dict[str, Any]:
    """The index of a report directory holding ``names`` beside index.json and index.md."""

    artifacts = _artifacts(names)
    summary = report.summary.model_dump(mode="json")
    for estimate in summary["headline"]:
        estimate["csv"] = _csv_row(estimate["item"], artifacts)
    return {
        "index_version": INDEX_VERSION,
        "guide": AGENT_GUIDE_URL,
        "summary_page": SUMMARY_PAGE,
        "pointers_into": "report.json",
        "analysis": {
            key: report.analysis.get(key) for key in ("id", "title", "intent", "plan_sha256")
        },
        "package_version": report.package_version,
        "report_schema_version": report.schema_version,
        "artifacts": artifacts,
        "summary": summary,
    }


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def index_markdown(document: dict[str, Any]) -> str:
    """index.md: the content of ``document`` (``index_document``) as Markdown."""

    analysis = document["analysis"]
    summary = document["summary"]
    lines = [
        f"# Index: {_cell(analysis.get('title') or analysis.get('id'))}",
        "",
        f"`{document['summary_page']}` is the one-page summary for people. Pointers such as "
        f"`/group_risks/0` are JSON pointers into `{document['pointers_into']}`. How to read "
        "this index: "
        f"<{document['guide']}>.",
        "",
        f"Analysis `{_cell(analysis.get('id'))}` ({_cell(analysis.get('intent'))}), plan "
        f"SHA-256 `{_cell(analysis.get('plan_sha256'))}`, inspect-safety-stats "
        f"{_cell(document['package_version'])}, report schema "
        f"{_cell(document['report_schema_version'])}, index version "
        f"{document['index_version']}.",
        "",
        f"Data integrity: **{summary['integrity']}** ({summary['error_count']} errors, "
        f"{summary['warning_count']} warnings, {summary['info_count']} informational).",
        "",
    ]
    if summary["decisions_shown"]:
        lines.extend(
            [
                "## Decisions",
                "",
                f"Overall decision: **{summary['decision']}**",
                "",
                "| Claim | Status | Mandatory | Pointer |",
                "|---|---|---|---|",
                *[
                    f"| {_cell(claim['title'])} | {claim['status']} | "
                    f"{'yes' if claim['mandatory'] else 'no'} | `{claim['item']}` |"
                    for claim in summary["claims"]
                ],
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Decisions are hidden (`report.include_decisions` is false); `report.json` "
                "still holds them at `/decision` and `/claims`.",
                "",
            ]
        )
    lines.extend(["## Headline estimates", ""])
    if summary["headline"]:
        lines.extend(
            [
                "| Kind | Model | Baseline | Group | Measure | Estimand | Estimate | Lower "
                "| Upper | Eligibility | Pointer | CSV row |",
                "|---|---|---|---|---|---|---:|---:|---:|---|---|---|",
            ]
        )
        for item in summary["headline"]:
            csv = item["csv"]
            cells = [
                item["kind"],
                _cell(item["cohort"]),
                _cell(item["baseline"]),
                _cell(item["group"]),
                _cell(item["measure"]),
                _cell(item["estimand"]),
                _cell(item["estimate"]),
                _cell(item["lower"]),
                _cell(item["upper"]),
                item["eligibility"],
                f"`{item['item']}`",
                f"`{csv['file']}` row {csv['row']}" if csv else "—",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("No headline estimates.")
    lines.extend(["", "## Errors and warnings by code", ""])
    if summary["warnings"]:
        lines.extend(
            [
                "| Code | Severity | Count | Effect | Diagnostics | Items |",
                "|---|---|---:|---|---|---|",
            ]
        )
        for warning in summary["warnings"]:
            occurrences = warning["occurrences"]
            items = list(dict.fromkeys(item["item"] for item in occurrences if item["item"]))
            cells = [
                f"`{warning['code']}`",
                warning["severity"].lower(),
                str(warning["count"]),
                _cell(warning["effect"]),
                ", ".join(f"`{item['diagnostic']}`" for item in occurrences),
                ", ".join(f"`{item}`" for item in items) or "—",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "What each code assumes and what follows is in `index.json` and in "
                "`report.md`; `docs/diagnostics.md` gives the remedy.",
            ]
        )
    else:
        lines.append("None.")
    lines.extend(["", "## Assumptions used", ""])
    lines.append(
        ", ".join(f"`{item}`" for item in summary["assumptions"])
        + ", in order at `/assumptions_used/<i>`."
        if summary["assumptions"]
        else "None recorded."
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "| File | Format | Holds | Data row n is |",
            "|---|---|---|---|",
            *[
                f"| `{item['name']}` | {item['format']} | {_cell(item['holds'])} | "
                + (f"`{item['rows_of']}/<n-1>`" if "rows_of" in item else "—")
                + " |"
                for item in document["artifacts"]
            ],
        ]
    )
    return "\n".join(lines) + "\n"


def index_files(report: SafetyReport, names: list[str]) -> dict[str, str]:
    """index.json and index.md for a report directory holding ``names``, by file name."""

    document = index_document(report, names)
    return {
        INDEX_JSON: json_text(canonical_value(document)),
        INDEX_MD: index_markdown(document),
    }
