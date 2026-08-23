from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from inspect_safety_stats import certify, load_certification_plan
from inspect_safety_stats.schema.reports import (
    IntegritySummary,
    PromptEstimate,
    SafetyReport,
)

PLAN_HASH = "a" * 64


def _write_empirical_report(tmp_path: Path, *, intent: str = "confirmatory") -> Path:
    directory = tmp_path / "empirical"
    directory.mkdir()
    estimates = [
        PromptEstimate(
            cohort_id="candidate",
            prompt_id="p-safe",
            question_id="q-safe",
            task_id="t1",
            n_total=100,
            n_expected=100,
            n_valid=100,
            n_failure=0,
            n_safe=100,
            n_error=0,
            p_hat=0.0,
            lower=0.0,
            upper=0.1,
            method="clopper_pearson_one_sided",
            eligibility="certifying",
        ),
        PromptEstimate(
            cohort_id="candidate",
            prompt_id="p-unsafe",
            question_id="q-unsafe",
            task_id="t1",
            n_total=100,
            n_expected=100,
            n_valid=100,
            n_failure=100,
            n_safe=0,
            n_error=0,
            p_hat=1.0,
            lower=0.9,
            upper=1.0,
            method="clopper_pearson_one_sided",
            eligibility="certifying",
        ),
    ]
    report = SafetyReport(
        package_version="0.5.0",
        analysis={
            "id": "empirical-fixture",
            "title": "Empirical fixture",
            "intent": intent,
            "plan_frozen_at": "2026-01-01T00:00:00Z" if intent == "confirmatory" else None,
            "registration": None,
            "plan_sha256": PLAN_HASH,
        },
        provenance={"source_logs": [], "inspect_version_constraint": ">=0.3.259,<0.4"},
        scope={"design": "fixed", "randomness": "rollouts", "estimand": "system"},
        presentation={"include_decisions": False},
        integrity=IntegritySummary(status="valid"),
        design={
            "manifest": "design.csv",
            "manifest_sha256": "b" * 64,
            "hierarchy": ["task_id", "question_id", "prompt_id"],
            "prompt_count": 2,
            "groups": [{"id": "all", "title": "All", "prompt_count": 2}],
            "independence_profiles": {
                "rollout": {
                    "block_by": ["rollout"],
                    "comparison_coupling": "independent",
                },
                "prompt": {
                    "block_by": ["prompt_id"],
                    "comparison_coupling": "joint",
                },
            },
        },
        cohorts=[{"cohort_id": "candidate"}],
        prompt_estimates=estimates,
        group_risks=[],
        comparisons=[],
        multiplicity=[],
        claims=[],
        policy_events=[],
        decision="INCONCLUSIVE",
    )
    (directory / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "group_id": "all",
                "task_id": "t1",
                "question_id": "q-safe",
                "prompt_id": "p-safe",
                "weight": 0.5,
            },
            {
                "group_id": "all",
                "task_id": "t1",
                "question_id": "q-unsafe",
                "prompt_id": "p-unsafe",
                "weight": 0.5,
            },
        ]
    ).to_csv(directory / "design.resolved.csv", index=False)
    observation_rows = []
    for prompt_id, failure in (("p-safe", False), ("p-unsafe", True)):
        for epoch in range(1, 101):
            observation_rows.append(
                {
                    "observation_id": f"{prompt_id}-{epoch}",
                    "source_log": "test.eval",
                    "source_log_sha256": "",
                    "eval_id": "eval-candidate",
                    "run_id": "",
                    "sample_id": f"{prompt_id}-{epoch}",
                    "inspect_id": prompt_id,
                    "epoch": epoch,
                    "prompt_id": prompt_id,
                    "question_id": f"q-{prompt_id[2:]}",
                    "task_id": "t1",
                    "cohort_id": "candidate",
                    "score_name": "safety",
                    "score_raw": "FAIL" if failure else "PASS",
                    "failure": failure,
                    "outcome_status": "classified_failure" if failure else "classified_safe",
                    "error": "",
                    "error_stage": "",
                    "content_delivered": "",
                    "provider_error_code": "",
                    "retries": 0,
                    "model": "candidate",
                    "configuration_fingerprint": "candidate",
                    "collection_order": len(observation_rows) + 1,
                }
            )
    pd.DataFrame(observation_rows).to_csv(directory / "observations.csv", index=False)
    return directory


def _write_plan(tmp_path: Path) -> Path:
    plan = {
        "schema_version": "2.0",
        "certification": {
            "id": "cert-fixture",
            "title": "Certification fixture",
        },
        "source": {"require_plan_sha256": PLAN_HASH},
        "claims": [
            {
                "id": "safe-prompt",
                "title": "Safe prompt risk is below 10%",
                "target": {
                    "type": "prompt_failure_rate",
                    "cohort": "candidate",
                    "prompt_id": "p-safe",
                },
                "threshold": 0.1,
                "support": {"family": "support", "method": "clopper_pearson"},
                "rejection": {"family": "rejection", "method": "clopper_pearson"},
            },
            {
                "id": "unsafe-prompt",
                "title": "Unsafe prompt risk is below 10%",
                "target": {
                    "type": "prompt_failure_rate",
                    "cohort": "candidate",
                    "prompt_id": "p-unsafe",
                },
                "threshold": 0.1,
                "support": {"family": "support", "method": "clopper_pearson"},
                "rejection": {"family": "rejection", "method": "clopper_pearson"},
            },
            {
                "id": "group",
                "title": "Group mean risk is below 50%",
                "target": {"type": "group_mean", "cohort": "candidate", "group": "all"},
                "threshold": 0.5,
                "support": {
                    "family": "support",
                    "method": "weighted_empirical_bernstein",
                    "independence_profile": "rollout",
                },
                "rejection": {
                    "family": "rejection",
                    "method": "weighted_empirical_bernstein",
                    "independence_profile": "rollout",
                },
                "sensitivity": [
                    {"method": "weighted_hoeffding", "independence_profile": "rollout"},
                    {
                        "method": "weighted_empirical_bernstein",
                        "independence_profile": "prompt",
                    },
                ],
            },
            {
                "id": "tasks",
                "title": "Task mean risk is below 50%",
                "target": {
                    "type": "hierarchy_mean",
                    "cohort": "candidate",
                    "group": "all",
                    "level": "task_id",
                    "all_paths": True,
                },
                "threshold": 0.5,
                "support": {
                    "family": "support",
                    "method": "weighted_empirical_bernstein",
                    "independence_profile": "rollout",
                },
                "rejection": {
                    "family": "rejection",
                    "method": "weighted_empirical_bernstein",
                    "independence_profile": "rollout",
                },
            },
        ],
        "multiplicity": {
            "families": [
                {
                    "id": "support",
                    "purpose": "simultaneous_support",
                    "method": "bonferroni",
                    "alpha": 0.05,
                },
                {
                    "id": "rejection",
                    "purpose": "simultaneous_rejection",
                    "method": "bonferroni",
                    "alpha": 0.05,
                },
            ]
        },
        "report": {"decimal_places": 6},
    }
    path = tmp_path / "certification.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


def test_certification_end_to_end(tmp_path: Path) -> None:
    source = _write_empirical_report(tmp_path)
    plan = _write_plan(tmp_path)

    report = certify(empirical_report=source, plan=plan)

    assert report.summary == {
        "supported": 1,
        "rejected": 1,
        "inconclusive": 2,
        "total": 4,
    }
    assert {claim.evidence for claim in report.claims} == {
        "SUPPORTED",
        "REJECTED",
        "INCONCLUSIVE",
    }
    assert all(claim.support.local_alpha == pytest.approx(0.0125) for claim in report.claims)
    group = next(claim for claim in report.claims if claim.claim_id == "group")
    assert len(group.sensitivities) == 2
    assert {item.independence_profile for item in group.detectability} == {"rollout"}
    assert all(item.guarantee == "conservative" for item in group.detectability)
    prompt = next(claim for claim in report.claims if claim.claim_id == "safe-prompt")
    assert len(prompt.detectability) == 1
    assert prompt.detectability[0].guarantee == "exact"
    task = next(claim for claim in report.claims if claim.claim_id.startswith("tasks:task_id:"))
    assert task.target["path"] == {"task_id": "t1"}

    output = report.write(tmp_path / "certification-report")
    assert (output / "certification-report.json").exists()
    assert (output / "certification-report.md").exists()
    assert (output / "certification-report.html").exists()
    assert (output / "certification-claims.csv").exists()
    assert "Support adjusted p" in (output / "certification-report.md").read_text(encoding="utf-8")
    assert "certification-report.html" in (output / "checksums.json").read_text(encoding="utf-8")
    assert "Detectability at the evaluated horizon" in (
        output / "certification-report.html"
    ).read_text(encoding="utf-8")
    assert "Sensitivity evidence" not in (output / "certification-report.html").read_text(
        encoding="utf-8"
    )
    assert "Robustness</th>" not in (output / "certification-report.html").read_text(
        encoding="utf-8"
    )


def test_empirical_intent_does_not_change_numerical_evidence(tmp_path: Path) -> None:
    source = _write_empirical_report(tmp_path, intent="exploratory")
    plan = _write_plan(tmp_path)
    report = certify(empirical_report=source, plan=plan)
    assert report.summary == {
        "supported": 1,
        "rejected": 1,
        "inconclusive": 2,
        "total": 4,
    }
    assert {claim.evidence for claim in report.claims} == {
        "SUPPORTED",
        "REJECTED",
        "INCONCLUSIVE",
    }


def test_invalid_empirical_integrity_is_rejected(tmp_path: Path) -> None:
    source = _write_empirical_report(tmp_path)
    report_path = source / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["integrity"]["status"] = "invalid"
    report_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity is invalid"):
        certify(empirical_report=source, plan=_write_plan(tmp_path))


def test_weighted_bonferroni_allocates_over_expanded_claims(tmp_path: Path) -> None:
    source = _write_empirical_report(tmp_path)
    plan = load_certification_plan(_write_plan(tmp_path))
    for family in plan.multiplicity.families:
        family.method = "weighted_bonferroni"
    plan.claims[0].support.allocation_weight = 2.0
    plan.claims[0].rejection.allocation_weight = 2.0

    report = certify(empirical_report=source, plan=plan)

    safe = next(claim for claim in report.claims if claim.claim_id == "safe-prompt")
    others = [claim for claim in report.claims if claim.claim_id != "safe-prompt"]
    assert safe.support.local_alpha == pytest.approx(0.05 * 2 / 5)
    assert safe.rejection is not None
    assert safe.rejection.local_alpha == pytest.approx(0.05 * 2 / 5)
    assert all(claim.support.local_alpha == pytest.approx(0.05 / 5) for claim in others)
