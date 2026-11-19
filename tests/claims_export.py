"""Check claims.json against the report.json beside it: every exported value is a copy.

The check finds in report.json, by each claim's id, estimand and evidence refs, where each of
its values must come from, and does not follow the pointers claims.json gives. So an exporter
that copies faithfully from the wrong result or interval fails it too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Each record field, as (part of the record, field there) -> (pointer in sources, field there).
# A field claims.json exports that this table does not name fails the check, so a new field
# needs its report.json counterpart here.
_RECORD_FIELDS: dict[tuple[str, str], tuple[str, str]] = {
    ("", "claim_id"): ("claim", "claim_id"),
    ("", "title"): ("claim", "title"),
    ("", "status"): ("claim", "status"),
    ("", "descriptive_status"): ("claim", "descriptive_status"),
    ("", "mandatory"): ("claim", "mandatory"),
    ("", "reasons"): ("claim", "reasons"),
    ("estimand", "cohort"): ("estimand", "cohort"),
    ("estimand", "group"): ("estimand", "group"),
    ("estimand", "measure"): ("estimand", "measure"),
    ("estimand", "measure_type"): ("estimand", "measure_type"),
    ("estimand", "comparison"): ("estimand", "comparison"),
    ("estimand", "acceptable_upper"): ("claim", "acceptable_upper"),
    ("estimand", "unacceptable_lower"): ("claim", "unacceptable_lower"),
    ("bound", "estimate"): ("claim", "estimate"),
    ("bound", "lower"): ("claim", "lower"),
    ("bound", "upper"): ("claim", "upper"),
    ("bound", "alpha"): ("claim", "alpha"),
    ("bound", "method"): ("claim", "method"),
    ("bound", "multiplicity_family"): ("claim", "multiplicity_family"),
    ("evidence", "refs"): ("claim", "evidence_refs"),
    ("evidence", "eligibility"): ("evidence", "eligibility"),
    ("evidence", "interval_id"): ("evidence_interval", "interval_id"),
    ("evidence", "interval_method"): ("evidence_interval", "method"),
    ("evidence", "coverage_mode"): ("evidence_interval", "coverage_mode"),
    ("evidence", "coverage_scope"): ("evidence_interval", "coverage_scope"),
    ("evidence", "family_alpha"): ("evidence_interval", "family_alpha"),
    ("evidence", "independence_profile"): ("evidence_interval", "independence_profile"),
    ("evidence", "assumptions"): ("evidence_interval", "assumptions"),
}
# Checked on their own (judge_corrected, judge_correction_unavailable, evidence.test) or
# containers of the fields above.
_UNMAPPED = {
    *(
        ("", name)
        for name in (
            "judge_corrected",
            "judge_correction_unavailable",
            "estimand",
            "bound",
            "evidence",
            "sources",
        )
    ),
    ("evidence", "test"),
}


def _resolve(document: Any, pointer: str) -> Any:
    value = document
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _flat(record: dict[str, Any]) -> dict[tuple[str, str], Any]:
    flat: dict[tuple[str, str], Any] = {}
    for key, value in record.items():
        flat[("", key)] = value
        if isinstance(value, dict) and key != "sources":
            for inner, item in value.items():
                flat[(key, inner)] = item
    return flat


def _interval_id(result: dict[str, Any], refs: list[str]) -> str | None:
    """The interval a claim's bound comes from, as the claims.json docs say.

    A claim decided by a test has the test's bound, computed from the result's primary
    interval. Otherwise it is the interval an ``interval:`` ref names, else the primary one.
    """

    if any(ref.startswith("test:") for ref in refs):
        return result["primary_interval_id"]
    named = [ref.removeprefix("interval:") for ref in refs if ref.startswith("interval:")]
    return named[0] if named else result["primary_interval_id"]


def _expected_sources(report: dict[str, Any], claim_id: str) -> dict[str, str | None]:
    """Where in report.json each value of a claim's record comes from, found by its id."""

    (claim,) = [
        index for index, item in enumerate(report["claims"]) if item["claim_id"] == claim_id
    ]
    (estimand,) = [
        index
        for index, item in enumerate(report["presentation"]["claim_evidence"])
        if item["claim_id"] == claim_id
    ]
    about = report["presentation"]["claim_evidence"][estimand]
    if about["comparison"] is not None:
        # A comparison's hierarchy nodes share its id; the claim is about the overall row.
        found = [
            f"/comparisons/{index}"
            for index, item in enumerate(report["comparisons"])
            if item["comparison_id"] == about["comparison"] and item["level"] is None
        ]
    else:
        found = [
            f"/group_risks/{index}"
            for index, item in enumerate(report["group_risks"])
            if (item["cohort_id"], item["group_id"], item["measure_id"])
            == (about["cohort"], about["group"], about["measure"])
        ]
    if len(found) > 1:
        raise ValueError(f"claim {claim_id!r} is about {len(found)} results: {found}")
    evidence = found[0] if found else None
    interval = None
    if evidence is not None:
        result = _resolve(report, evidence)
        wanted = _interval_id(result, report["claims"][claim]["evidence_refs"])
        positions = [
            index
            for index, item in enumerate(result["intervals"])
            if wanted is not None and item["interval_id"] == wanted
        ]
        interval = f"{evidence}/intervals/{positions[0]}" if positions else None
    return {
        "claim": f"/claims/{claim}",
        "estimand": f"/presentation/claim_evidence/{estimand}",
        "evidence": evidence,
        "evidence_interval": interval,
    }


def round_trip_differences(directory: Path) -> list[str]:
    """Where claims.json differs from the report.json beside it; empty when every value is
    the report's own."""

    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    claims = json.loads((directory / "claims.json").read_text(encoding="utf-8"))
    differences: list[str] = []

    def expect(name: str, exported: Any, recorded: Any) -> None:
        if exported != recorded:
            differences.append(f"{name}: {exported!r} != {recorded!r}")

    expect("report_schema_version", claims["report_schema_version"], report["schema_version"])
    expect("package_version", claims["package_version"], report["package_version"])
    for key, value in claims["analysis"].items():
        expect(f"analysis.{key}", value, report["analysis"].get(key))
    expect("provenance.sources", claims["provenance"]["sources"], report["provenance"]["sources"])
    expect(
        "provenance.manifest_sha256",
        claims["provenance"]["manifest_sha256"],
        report["design"]["manifest_sha256"],
    )
    expect("integrity_status", claims["integrity_status"], report["integrity"]["status"])
    expect("decision", claims["decision"], report["decision"])
    expect(
        "claim ids",
        [item["claim_id"] for item in claims["claims"]],
        [item["claim_id"] for item in report["claims"]],
    )

    def named(code: str) -> set[str]:
        return {
            claim_id
            for diagnostic in report["integrity"]["diagnostics"]
            if diagnostic["code"] == code
            for claim_id in diagnostic["details"]["claim_ids"]
        }

    corrected = named("JUDGE_CORRECTED_CLAIMS")
    uncorrected = named("JUDGE_CORRECTION_UNAVAILABLE")
    claims_by_id = {item["claim_id"]: item for item in report["claims"]}
    for record in claims["claims"]:
        name = record["claim_id"]
        flat = _flat(record)
        unmapped = set(flat) - set(_RECORD_FIELDS) - _UNMAPPED
        if unmapped:
            differences.append(f"{name}: fields without a report.json counterpart: {unmapped}")
        expect(f"{name}.judge_corrected", record["judge_corrected"], name in corrected)
        expect(
            f"{name}.judge_correction_unavailable",
            record["judge_correction_unavailable"],
            name in uncorrected,
        )
        tests = [
            ref.removeprefix("test:")
            for ref in claims_by_id[name]["evidence_refs"]
            if ref.startswith("test:")
        ]
        expect(f"{name}.evidence.test", record["evidence"]["test"], tests[0] if tests else None)
        sources = _expected_sources(report, name)
        expect(f"{name}.sources", record["sources"], sources)
        for field, (source, source_field) in _RECORD_FIELDS.items():
            pointer = sources[source]
            if pointer is None:
                recorded = [] if field == ("evidence", "assumptions") else None
            else:
                recorded = _resolve(report, pointer).get(source_field)
            expect(f"{name}.{'.'.join(part for part in field if part)}", flat[field], recorded)
    return differences
