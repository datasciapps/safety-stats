from __future__ import annotations

from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.schema.reports import Diagnostic


def _duplicates(values: list[str]) -> set[str]:
    return {value for value in values if values.count(value) > 1}


def validate_plan_references(plan: AnalysisPlan) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    expanded_comparisons = plan.expanded_comparisons()
    group_ids = [group.id for group in plan.groups]
    measure_ids = [measure.id for measure in plan.outputs.risk_measures]
    family_ids = [family.id for family in plan.multiplicity.families]
    comparison_ids = [comparison.id for comparison in expanded_comparisons]
    claim_ids = [claim.id for claim in plan.claims]
    for kind, values in (
        ("group", group_ids),
        ("measure", measure_ids),
        ("family", family_ids),
        ("comparison", comparison_ids),
        ("claim", claim_ids),
    ):
        for value in sorted(_duplicates(values)):
            diagnostics.append(
                Diagnostic(
                    code="PLAN_DUPLICATE_ID",
                    severity="ERROR",
                    message=f"duplicate {kind} identifier {value!r}",
                    scope=value,
                )
            )
    groups = set(group_ids)
    measures = {measure.id: measure for measure in plan.outputs.risk_measures}
    families = {family.id: family for family in plan.multiplicity.families}
    comparisons = set(comparison_ids)
    cohorts = set(plan.cohorts)
    comparison_family_modes: dict[str, set[str]] = {}
    comparison_family_profiles: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
    for comparison in expanded_comparisons:
        if comparison.candidate not in cohorts or comparison.baseline not in cohorts:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_UNKNOWN_REFERENCE",
                    severity="ERROR",
                    message=f"comparison {comparison.id!r} references an unknown cohort",
                    scope=comparison.id,
                )
            )
        if comparison.candidate == comparison.baseline:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_INVALID_COMPARISON",
                    severity="ERROR",
                    message=f"comparison {comparison.id!r} uses the same cohort in both arms",
                    scope=comparison.id,
                )
            )
        if comparison.group not in groups or comparison.measure not in measures:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_UNKNOWN_REFERENCE",
                    severity="ERROR",
                    message=f"comparison {comparison.id!r} references an unknown group or measure",
                    scope=comparison.id,
                )
            )
        elif comparison.method != "fixed_design_bounds":
            if measures[comparison.measure].type != "weighted_mean":
                diagnostics.append(
                    Diagnostic(
                        code="PLAN_INVALID_COMPARISON",
                        severity="ERROR",
                        message=(
                            f"direct comparison {comparison.id!r} requires a weighted_mean "
                            "measure; use fixed_design_bounds for other measures"
                        ),
                        scope=comparison.id,
                    )
                )
            if comparison.family not in families:
                diagnostics.append(
                    Diagnostic(
                        code="PLAN_UNKNOWN_REFERENCE",
                        severity="ERROR",
                        message=(
                            f"direct comparison {comparison.id!r} requires a valid "
                            "multiplicity family"
                        ),
                        scope=comparison.id,
                    )
                )
            elif comparison.simultaneous and families[comparison.family].method not in {
                "bonferroni",
                "weighted_bonferroni",
            }:
                diagnostics.append(
                    Diagnostic(
                        code="PLAN_INVALID_MULTIPLICITY",
                        severity="ERROR",
                        message=(
                            "simultaneous direct comparisons require a bonferroni or "
                            "weighted_bonferroni family"
                        ),
                        scope=comparison.id,
                    )
                )
            if comparison.family:
                comparison_family_modes.setdefault(comparison.family, set()).add(
                    comparison.coverage_mode
                )
                assert comparison.independence is not None
                comparison_family_profiles.setdefault(comparison.family, set()).add(
                    (
                        comparison.independence.primary,
                        tuple(comparison.independence.sensitivities),
                    )
                )
        paired = {
            "exact_mcnemar",
            "newcombe_paired",
            "fixed_cluster_bootstrap",
        }.intersection(comparison.diagnostic_methods)
        if paired and comparison.pair_key is None:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_INVALID_COMPARISON",
                    severity="ERROR",
                    message=f"paired diagnostics for {comparison.id!r} require pair_key",
                    scope=comparison.id,
                )
            )
    for family_id, modes in comparison_family_modes.items():
        if len(modes) > 1:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_INVALID_MULTIPLICITY",
                    severity="ERROR",
                    message=(
                        "all direct comparisons sharing a multiplicity family must use the "
                        "same coverage_mode"
                    ),
                    scope=family_id,
                )
            )
        if len(comparison_family_profiles.get(family_id, set())) > 1:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_INVALID_MULTIPLICITY",
                    severity="ERROR",
                    message=(
                        "all direct comparisons sharing a multiplicity family must use the "
                        "same primary and sensitivity independence profiles"
                    ),
                    scope=family_id,
                )
            )
    for claim in plan.claims:
        if claim.family not in families:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_UNKNOWN_REFERENCE",
                    severity="ERROR",
                    message=f"claim {claim.id!r} references unknown family {claim.family!r}",
                    scope=claim.id,
                )
            )
        if claim.comparison:
            if claim.comparison not in comparisons:
                diagnostics.append(
                    Diagnostic(
                        code="PLAN_UNKNOWN_REFERENCE",
                        severity="ERROR",
                        message=f"claim {claim.id!r} references unknown comparison",
                        scope=claim.id,
                    )
                )
        else:
            if (
                claim.cohort not in cohorts
                or claim.group not in groups
                or claim.measure not in measures
            ):
                diagnostics.append(
                    Diagnostic(
                        code="PLAN_UNKNOWN_REFERENCE",
                        severity="ERROR",
                        message=f"claim {claim.id!r} has an unknown cohort, group, or measure",
                        scope=claim.id,
                    )
                )
            elif measures[claim.measure].diagnostic_only:
                diagnostics.append(
                    Diagnostic(
                        code="PLAN_DIAGNOSTIC_GATE",
                        severity="ERROR",
                        message=(
                            f"claim {claim.id!r} uses diagnostic-only measure {claim.measure!r}"
                        ),
                        scope=claim.id,
                    )
                )
    prompt_family = plan.outputs.prompt_bounds.family
    if plan.outputs.prompt_bounds.simultaneous and prompt_family not in families:
        diagnostics.append(
            Diagnostic(
                code="PLAN_UNKNOWN_REFERENCE",
                severity="ERROR",
                message="simultaneous prompt bounds require a valid multiplicity family",
                scope="outputs.prompt_bounds",
            )
        )
    elif plan.outputs.prompt_bounds.simultaneous and prompt_family in families:
        family_method = families[prompt_family].method
        valid_methods = {"bonferroni", "weighted_bonferroni"}
        if family_method not in valid_methods:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_INVALID_MULTIPLICITY",
                    severity="ERROR",
                    message=(
                        "simultaneous prompt bounds require a bonferroni or "
                        "weighted_bonferroni family"
                    ),
                    scope="outputs.prompt_bounds",
                )
            )
    direct = plan.outputs.direct_mean_intervals
    if direct.enabled:
        if direct.family not in families:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_UNKNOWN_REFERENCE",
                    severity="ERROR",
                    message="direct mean intervals require a valid multiplicity family",
                    scope="outputs.direct_mean_intervals",
                )
            )
        elif direct.simultaneous and families[direct.family].method != "bonferroni":
            diagnostics.append(
                Diagnostic(
                    code="PLAN_INVALID_MULTIPLICITY",
                    severity="ERROR",
                    message="simultaneous direct mean intervals require a bonferroni family",
                    scope="outputs.direct_mean_intervals",
                )
            )
        unknown_groups = sorted(set(direct.groups or []) - groups)
        if unknown_groups:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_UNKNOWN_REFERENCE",
                    severity="ERROR",
                    message=(
                        "direct mean intervals reference unknown groups: "
                        + ", ".join(unknown_groups)
                    ),
                    scope="outputs.direct_mean_intervals",
                )
            )
        if not any(measure.type == "weighted_mean" for measure in measures.values()):
            diagnostics.append(
                Diagnostic(
                    code="PLAN_METHOD_UNUSED",
                    severity="WARNING",
                    message="direct mean intervals are enabled but no weighted mean is requested",
                    scope="outputs.direct_mean_intervals",
                )
            )
    return diagnostics
