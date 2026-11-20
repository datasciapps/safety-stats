"""Every plan-validation diagnostic that validate_plan_references raises is named here (#171).

Each test builds a small plan that meets exactly the condition of one raising site and asserts
the code, the severity, the scope and a distinctive fragment of the message. No logs are read.
"""

from __future__ import annotations

from typing import Any

import pytest

from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.validate.plan import validate_plan_references

_FAMILY = {"id": "fam", "purpose": "simultaneous_intervals", "method": "bonferroni", "alpha": 0.05}
_MEASURE = {"id": "mean", "type": "weighted_mean"}
_PROFILES = {"a": {"block_by": ["task_id"]}, "b": {"block_by": ["question_id"]}}


def _comparison(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "cmp",
        "candidate": "cand",
        "baseline": "base",
        "group": "all",
        "measure": "mean",
        "family": "fam",
        "independence": {"primary": "a"},
        **overrides,
    }


def _plan(
    *,
    families: list[dict[str, Any]] | None = None,
    measures: list[dict[str, Any]] | None = None,
    groups: list[dict[str, Any]] | None = None,
    claims: list[dict[str, Any]] | None = None,
    comparisons: list[dict[str, Any]] | None = None,
    design: dict[str, Any] | None = None,
    prompt_bounds: dict[str, Any] | None = None,
    direct: dict[str, Any] | None = None,
) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "schema_version": "2.0",
            "analysis": {"id": "p", "title": "P", "intent": "exploratory"},
            "inputs": {
                "design_manifest": "design.csv",
                "score": {
                    "name": "harmful",
                    "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                },
            },
            "design": {"independence_profiles": _PROFILES, **(design or {})},
            "cohorts": {"cand": {"match": {"model": "m/c"}}, "base": {"match": {"model": "m/b"}}},
            "groups": groups if groups is not None else [{"id": "all", "filter": {"suite": "s"}}],
            "outputs": {
                "prompt_bounds": {"enabled": False, **(prompt_bounds or {})},
                "direct_mean_intervals": direct or {},
                "risk_measures": measures if measures is not None else [_MEASURE],
            },
            "comparisons": comparisons or [],
            "multiplicity": {"families": families if families is not None else [_FAMILY]},
            "claims": claims or [],
        }
    )


def _only(plan: AnalysisPlan, code: str) -> Diagnostic:
    (found,) = [item for item in validate_plan_references(plan) if item.code == code]
    return found


def _check(found: Diagnostic, severity: str, scope: str, fragment: str) -> None:
    assert found.severity == severity
    assert found.scope == scope
    assert fragment in found.message


def _claim(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "claim",
        "title": "C",
        "cohort": "cand",
        "group": "all",
        "measure": "mean",
        "family": "fam",
        "acceptable_upper": 0.01,
        "unacceptable_lower": 0.1,
        **overrides,
    }


def test_a_valid_plan_raises_none_of_these_codes() -> None:
    assert validate_plan_references(_plan()) == []


def test_diagnostic_only_measure_cannot_back_a_claim() -> None:
    plan = _plan(
        measures=[{**_MEASURE, "diagnostic_only": True}],
        claims=[_claim()],
    )
    found = _only(plan, "PLAN_DIAGNOSTIC_GATE")
    _check(found, "ERROR", "plan/claims/claim", "diagnostic-only measure 'mean'")


@pytest.mark.parametrize(
    ("kind", "build"),
    [
        ("group", lambda: _plan(groups=[{"id": "all", "filter": {}}, {"id": "all", "filter": {}}])),
        ("measure", lambda: _plan(measures=[_MEASURE, _MEASURE])),
        ("family", lambda: _plan(families=[_FAMILY, _FAMILY])),
        ("claim", lambda: _plan(claims=[_claim(), _claim()])),
    ],
)
def test_duplicate_identifiers_are_named_by_kind(kind: str, build: Any) -> None:
    plan = build()
    found = _only(plan, "PLAN_DUPLICATE_ID")
    value = {"group": "all", "measure": "mean", "family": "fam", "claim": "claim"}[kind]
    path = {
        "group": "groups",
        "measure": "outputs/risk_measures",
        "family": "multiplicity/families",
        "claim": "claims",
    }[kind]
    _check(found, "ERROR", f"plan/{path}/{value}", f"duplicate {kind} identifier {value!r}")


def test_the_same_cohort_in_both_arms_is_an_invalid_comparison() -> None:
    plan = _plan(comparisons=[_comparison(baseline="cand")])
    found = _only(plan, "PLAN_INVALID_COMPARISON")
    _check(found, "ERROR", "plan/comparisons/cmp", "uses the same cohort in both arms")


def test_a_comparison_needs_a_weighted_mean_measure() -> None:
    plan = _plan(
        measures=[_MEASURE, {"id": "worst", "type": "maximum"}],
        comparisons=[_comparison(measure="worst")],
    )
    found = _only(plan, "PLAN_INVALID_COMPARISON")
    _check(found, "ERROR", "plan/comparisons/cmp", "requires a weighted_mean measure")


@pytest.mark.parametrize("method", ["exact_mcnemar", "newcombe_paired", "fixed_cluster_bootstrap"])
def test_paired_diagnostics_need_a_pair_key(method: str) -> None:
    plan = _plan(comparisons=[_comparison(diagnostic_methods=[method])])
    found = _only(plan, "PLAN_INVALID_COMPARISON")
    _check(found, "ERROR", "plan/comparisons/cmp", "paired diagnostics for 'cmp' require pair_key")
    keyed = _plan(comparisons=[_comparison(diagnostic_methods=[method], pair_key=["prompt_id"])])
    assert not [i for i in validate_plan_references(keyed) if i.code == "PLAN_INVALID_COMPARISON"]


def test_simultaneous_comparisons_need_a_bonferroni_family() -> None:
    plan = _plan(
        families=[{**_FAMILY, "method": "intersection_union"}],
        comparisons=[_comparison()],
    )
    found = _only(plan, "PLAN_INVALID_MULTIPLICITY")
    _check(
        found,
        "ERROR",
        "plan/comparisons/cmp",
        "simultaneous direct comparisons require a bonferroni",
    )


def test_comparisons_sharing_a_family_must_share_a_coverage_mode() -> None:
    plan = _plan(
        comparisons=[
            _comparison(id="one"),
            _comparison(id="two", coverage_mode="per_direction"),
        ]
    )
    found = _only(plan, "PLAN_INVALID_MULTIPLICITY")
    _check(found, "ERROR", "plan/multiplicity/families/fam", "must use the same coverage_mode")


def test_comparisons_sharing_a_family_must_share_independence_profiles() -> None:
    plan = _plan(
        comparisons=[
            _comparison(id="one"),
            _comparison(id="two", independence={"primary": "b"}),
        ]
    )
    found = _only(plan, "PLAN_INVALID_MULTIPLICITY")
    _check(
        found,
        "ERROR",
        "plan/multiplicity/families/fam",
        "same primary and sensitivity independence profiles",
    )


def test_simultaneous_prompt_bounds_need_a_bonferroni_family() -> None:
    plan = _plan(
        families=[{**_FAMILY, "method": "intersection_union"}],
        prompt_bounds={"enabled": True, "family": "fam"},
    )
    found = _only(plan, "PLAN_INVALID_MULTIPLICITY")
    _check(found, "ERROR", "plan/outputs/prompt_bounds", "simultaneous prompt bounds require")


def test_simultaneous_direct_mean_intervals_need_a_bonferroni_family() -> None:
    # weighted_bonferroni passes for comparisons and prompt bounds, but not for this site.
    plan = _plan(
        families=[{**_FAMILY, "method": "weighted_bonferroni"}],
        direct={"enabled": True, "family": "fam", "independence": {"primary": "a"}},
    )
    found = _only(plan, "PLAN_INVALID_MULTIPLICITY")
    _check(found, "ERROR", "plan/outputs/direct_mean_intervals", "require a bonferroni family")


def test_direct_mean_intervals_without_a_weighted_mean_are_unused() -> None:
    plan = _plan(
        measures=[{"id": "worst", "type": "maximum"}],
        direct={"enabled": True, "family": "fam", "independence": {"primary": "a"}},
    )
    found = _only(plan, "PLAN_METHOD_UNUSED")
    _check(found, "WARNING", "plan/outputs/direct_mean_intervals", "no weighted mean is requested")


_SAMPLED = {"type": "sampled", "inclusion_probability_column": "pi"}
_DIRECT = {"enabled": True, "family": "fam", "independence": {"primary": "a"}}


def test_a_sampled_design_does_not_support_comparisons() -> None:
    plan = _plan(design=_SAMPLED, comparisons=[_comparison()], direct=_DIRECT)
    found = _only(plan, "PLAN_SAMPLED_COMPARISON_UNSUPPORTED")
    _check(found, "ERROR", "plan/comparisons", "not yet supported under a sampled design")
    fixed = _plan(comparisons=[_comparison()])
    codes = {item.code for item in validate_plan_references(fixed)}
    assert "PLAN_SAMPLED_COMPARISON_UNSUPPORTED" not in codes


def test_a_sampled_design_requires_direct_mean_intervals() -> None:
    plan = _plan(design=_SAMPLED)
    found = _only(plan, "PLAN_SAMPLED_REQUIRES_DIRECT_MEAN")
    _check(found, "ERROR", "plan/outputs/direct_mean_intervals", "requires outputs.direct_mean")
    enabled = _plan(design=_SAMPLED, direct=_DIRECT)
    codes = {item.code for item in validate_plan_references(enabled)}
    assert "PLAN_SAMPLED_REQUIRES_DIRECT_MEAN" not in codes


def test_simultaneous_prompt_bounds_under_a_sampled_design_warn() -> None:
    plan = _plan(design=_SAMPLED, direct=_DIRECT, prompt_bounds={"enabled": True, "family": "fam"})
    found = _only(plan, "PLAN_SAMPLED_PROMPT_BOUNDS")
    _check(found, "WARNING", "plan/outputs/prompt_bounds", "cover only the sampled frame items")
    pointwise = _plan(
        design=_SAMPLED,
        direct=_DIRECT,
        prompt_bounds={"enabled": True, "family": "fam", "simultaneous": False},
    )
    codes = {item.code for item in validate_plan_references(pointwise)}
    assert "PLAN_SAMPLED_PROMPT_BOUNDS" not in codes
