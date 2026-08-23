from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault(
    "PYTENSOR_FLAGS",
    f"base_compiledir={Path(tempfile.gettempdir()) / 'inspect-safety-stats-pytensor'}",
)
pytest.importorskip("pymc")

from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.structural.fit import fit_structural_model


def _write_fixture(tmp_path: Path) -> tuple[AnalysisPlan, Path]:
    plan = AnalysisPlan.model_validate(
        {
            "analysis": {"id": "structural-smoke", "title": "Smoke", "intent": "exploratory"},
            "inputs": {
                "design_manifest": str(tmp_path / "manifest.csv"),
                "score": {
                    "name": "harmful",
                    "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                },
            },
            "cohorts": {"candidate": {"match": {"model": "model/test"}}},
            "groups": [{"id": "all"}],
            "outputs": {"risk_measures": [{"id": "mean", "type": "weighted_mean"}]},
            "multiplicity": {"families": []},
            "claims": [],
            "structural_model": {
                "enabled": True,
                "cohort": "candidate",
                "group": "all",
                "prompt_frailty": {
                    "distribution": "student_t",
                    "gaussian_sensitivity": True,
                },
                "audit_metrics": {
                    "vulnerable_thresholds": [0.1],
                    "concentration_mass": [0.5],
                    "credible_interval": 0.8,
                },
                "inference": {
                    "chains": 2,
                    "draws": 100,
                    "tune": 100,
                    "cores": 1,
                    "target_accept": 0.9,
                    "prior_predictive_draws": 100,
                    "cross_validation_folds": 2,
                    "cross_validation_draws": 100,
                    "cross_validation_tune": 100,
                },
            },
        }
    )
    report_dir = tmp_path / "empirical"
    report_dir.mkdir()
    report = {
        "schema_version": "2.0",
        "package_version": "0.5.0",
        "analysis": {"id": "structural-smoke"},
        "provenance": {},
        "scope": {},
        "presentation": {},
        "integrity": {"status": "valid", "diagnostics": []},
        "design": {"groups": [{"id": "all"}]},
        "cohorts": [{"id": "candidate"}],
        "prompt_estimates": [],
        "group_risks": [
            {
                "cohort_id": "candidate",
                "group_id": "all",
                "measure_id": "mean",
                "estimate": 0.175,
                "method": "weighted_mean",
                "prompt_count": 4,
                "valid_prompt_count": 4,
                "resolved_weight": 1.0,
                "unresolved_weight": 0.0,
                "eligibility": "diagnostic",
            }
        ],
        "comparisons": [],
        "multiplicity": [],
        "claims": [],
        "policy_events": [],
        "decision": "INCONCLUSIVE",
    }
    (report_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (report_dir / "design.resolved.csv").write_text(
        "group_id,task_id,question_id,prompt_id,weight\n"
        "all,t1,q1,p1,0.25\n"
        "all,t1,q2,p2,0.25\n"
        "all,t2,q3,p3,0.25\n"
        "all,t2,q4,p4,0.25\n",
        encoding="utf-8",
    )
    rows = ["prompt_id,cohort_id,failure,configuration_fingerprint"]
    failures = {"p1": 0, "p2": 1, "p3": 2, "p4": 4}
    for prompt_id, failure_count in failures.items():
        for rollout in range(10):
            rows.append(f"{prompt_id},candidate,{str(rollout < failure_count)},configuration-1")
    (report_dir / "observations.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return plan, report_dir


def test_pymc_structural_fit_and_artifacts(tmp_path: Path) -> None:
    plan, empirical = _write_fixture(tmp_path)
    report = fit_structural_model(
        plan=plan,
        empirical_report=empirical,
        run_cross_validation=True,
    )
    assert report.certification_eligible is False
    assert report.primary_fit.link == "cloglog"
    assert len(report.primary_fit.prompt_risks) == 4
    assert len(report.primary_fit.hierarchy_effects) == 2
    assert len(report.primary_fit.hierarchy_marginal_risks) == 2
    assert report.primary_fit.posterior_predictive.statistics
    assert report.primary_fit.cross_validation.enabled is True
    assert report.primary_fit.cross_validation.prompt_count == 4
    assert report.data["hierarchy"] == ["task_id", "question_id", "prompt_id"]
    assert report.data["effect_hierarchy"] == ["task_id"]
    assert report.data["prompt_frailty_levels"] == ["question_id", "prompt_id"]
    assert [fit.prompt_frailty for fit in report.sensitivity_fits] == ["normal"]
    output = report.write(tmp_path / "structural")
    assert (output / "structural-report.json").exists()
    assert (output / "structural-report.html").exists()
    assert (output / "posterior.nc").exists()
    assert (output / "posterior-sensitivity-normal-frailty.nc").exists()
    assert (output / "hierarchy-marginal-risks.csv").exists()
    assert "Non-certifying structural analysis" in (output / "structural-report.html").read_text(
        encoding="utf-8"
    )
    assert "Full design and model hierarchy" in (output / "structural-report.html").read_text(
        encoding="utf-8"
    )
    assert "Marginal posterior risk by hierarchy node" in (
        output / "structural-report.html"
    ).read_text(encoding="utf-8")
    assert (
        "prompt or response" not in (output / "model-frame.csv").read_text(encoding="utf-8").lower()
    )
