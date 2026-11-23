"""The claims.json schema (experimental, #11) stays as pinned here.

A change to these fields is a change to claims.json's schema: decide its schema_version
(CONTRIBUTING sections 4.2 and 4.3), then update the snapshot.
"""

from __future__ import annotations

from typing import Any

from safety_stats.schema.claims import ClaimsExport

_FIELDS = {
    "ClaimsExport": [
        "analysis",
        "claims",
        "decision",
        "experimental",
        "integrity_status",
        "package_version",
        "provenance",
        "report_schema_version",
        "schema_version",
    ],
    "ClaimBound": ["alpha", "estimate", "lower", "method", "multiplicity_family", "upper"],
    "ClaimEstimand": [
        "acceptable_upper",
        "cohort",
        "comparison",
        "group",
        "measure",
        "measure_type",
        "unacceptable_lower",
    ],
    "ClaimEvidence": [
        "assumptions",
        "coverage_mode",
        "coverage_scope",
        "eligibility",
        "family_alpha",
        "independence_profile",
        "interval_id",
        "interval_method",
        "refs",
        "test",
    ],
    "ClaimRecord": [
        "bound",
        "claim_id",
        "descriptive_status",
        "estimand",
        "evidence",
        "judge_corrected",
        "judge_correction_unavailable",
        "mandatory",
        "reasons",
        "sources",
        "status",
        "title",
    ],
    "ClaimSources": ["claim", "estimand", "evidence", "evidence_interval"],
    "ClaimsAnalysis": ["id", "intent", "plan_frozen_at", "plan_sha256"],
    "ClaimsProvenance": ["manifest_sha256", "sources"],
    "ClaimsSource": ["path", "sha256"],
}


def _fields(schema: dict[str, Any]) -> dict[str, list[str]]:
    fields = {"ClaimsExport": sorted(schema["properties"])}
    for name, definition in schema["$defs"].items():
        fields[name] = sorted(definition["properties"])
    return fields


def test_the_claims_schema_keeps_its_fields() -> None:
    assert _fields(ClaimsExport.model_json_schema()) == _FIELDS


def test_the_claims_schema_keeps_its_version() -> None:
    schema = ClaimsExport.model_json_schema()
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    assert schema["properties"]["experimental"]["const"] is True


def test_a_claim_record_requires_the_fields_the_exporter_always_writes() -> None:
    record = ClaimsExport.model_json_schema()["$defs"]["ClaimRecord"]

    assert sorted(record["required"]) == [
        "bound",
        "claim_id",
        "estimand",
        "evidence",
        "judge_corrected",
        "judge_correction_unavailable",
        "mandatory",
        "reasons",
        "sources",
        "status",
        "title",
    ]
