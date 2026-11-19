from __future__ import annotations

from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.schema.scopes import plan_scope


def _duplicates(values: list[str]) -> set[str]:
    return {value for value in values if values.count(value) > 1}


def noisy_valid_test_supported(plan: AnalysisPlan) -> bool:
    """Whether the plan's judge calibration can run a ``noisy_valid_test`` claim's test.

    The test needs a validated calibration whose labels are a simple random sample.
    """

    calibration = plan.inputs.judge_calibration
    return (
        calibration is not None
        and calibration.mode == "validated"
        and calibration.sampling == "simple_random"
    )


def validate_plan_references(plan: AnalysisPlan) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    expanded_comparisons = plan.expanded_comparisons()
    group_ids = [group.id for group in plan.groups]
    measure_ids = [measure.id for measure in plan.outputs.risk_measures]
    family_ids = [family.id for family in plan.multiplicity.families]
    comparison_ids = [comparison.id for comparison in expanded_comparisons]
    claim_ids = [claim.id for claim in plan.claims] + [item.id for item in plan.adaptive_claims]
    for kind, collection, values in (
        ("group", ("groups",), group_ids),
        ("measure", ("outputs", "risk_measures"), measure_ids),
        ("family", ("multiplicity", "families"), family_ids),
        ("comparison", ("comparisons",), comparison_ids),
        ("claim", ("claims",), claim_ids),
    ):
        for value in sorted(_duplicates(values)):
            diagnostics.append(
                Diagnostic(
                    code="PLAN_DUPLICATE_ID",
                    severity="ERROR",
                    message=f"duplicate {kind} identifier {value!r}",
                    scope=plan_scope(*collection, value),
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
                    scope=plan_scope("comparisons", comparison.id),
                )
            )
        if comparison.candidate == comparison.baseline:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_INVALID_COMPARISON",
                    severity="ERROR",
                    message=f"comparison {comparison.id!r} uses the same cohort in both arms",
                    scope=plan_scope("comparisons", comparison.id),
                )
            )
        if comparison.group not in groups or comparison.measure not in measures:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_UNKNOWN_REFERENCE",
                    severity="ERROR",
                    message=f"comparison {comparison.id!r} references an unknown group or measure",
                    scope=plan_scope("comparisons", comparison.id),
                )
            )
        else:
            if measures[comparison.measure].type != "weighted_mean":
                diagnostics.append(
                    Diagnostic(
                        code="PLAN_INVALID_COMPARISON",
                        severity="ERROR",
                        message=(
                            f"comparison {comparison.id!r} requires a weighted_mean measure"
                        ),
                        scope=plan_scope("comparisons", comparison.id),
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
                        scope=plan_scope("comparisons", comparison.id),
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
                        scope=plan_scope("comparisons", comparison.id),
                    )
                )
            if comparison.family:
                comparison_family_modes.setdefault(comparison.family, set()).add(
                    comparison.coverage_mode
                )
                independence = comparison.independence
                if independence is None:
                    raise ValueError("comparisons require an independence selection")
                comparison_family_profiles.setdefault(comparison.family, set()).add(
                    (independence.primary, tuple(independence.sensitivities))
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
                    scope=plan_scope("comparisons", comparison.id),
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
                    scope=plan_scope("multiplicity", "families", family_id),
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
                    scope=plan_scope("multiplicity", "families", family_id),
                )
            )
    for claim in plan.claims:
        if claim.method == "noisy_valid_test":
            if not noisy_valid_test_supported(plan):
                diagnostics.append(
                    Diagnostic(
                        code="PLAN_INVALID_CLAIM",
                        severity="ERROR",
                        message=(
                            f"claim {claim.id!r} uses noisy_valid_test, which requires "
                            "judge_calibration.mode: validated with sampling: simple_random"
                        ),
                        scope=plan_scope("claims", claim.id),
                    )
                )
        if claim.family not in families:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_UNKNOWN_REFERENCE",
                    severity="ERROR",
                    message=f"claim {claim.id!r} references unknown family {claim.family!r}",
                    scope=plan_scope("claims", claim.id),
                )
            )
        if claim.comparison:
            if claim.comparison not in comparisons:
                diagnostics.append(
                    Diagnostic(
                        code="PLAN_UNKNOWN_REFERENCE",
                        severity="ERROR",
                        message=f"claim {claim.id!r} references unknown comparison",
                        scope=plan_scope("claims", claim.id),
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
                        scope=plan_scope("claims", claim.id),
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
                        scope=plan_scope("claims", claim.id),
                    )
                )
    for adaptive in plan.adaptive_claims:
        unknown = [
            f"{kind} {value!r}"
            for kind, value, known in (
                ("cohort", adaptive.cohort, cohorts),
                ("group", adaptive.group, groups),
                ("family", adaptive.family, families),
            )
            if value not in known
        ]
        if unknown:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_UNKNOWN_REFERENCE",
                    severity="ERROR",
                    message=f"adaptive claim {adaptive.id!r} references unknown "
                    + ", ".join(unknown),
                    scope=plan_scope("adaptive_claims", adaptive.id),
                )
            )
    prompt_bounds = plan.outputs.prompt_bounds
    prompt_family = prompt_bounds.family
    # Prompt bounds are on by default, and take their alpha from a family, simultaneous or
    # not. Without one they are silently absent, so a minimal plan analysed nothing (#57).
    if prompt_bounds.enabled and prompt_family not in families:
        diagnostics.append(
            Diagnostic(
                code="PLAN_UNKNOWN_REFERENCE",
                severity="ERROR",
                message=(
                    "prompt bounds take their alpha from outputs.prompt_bounds.family, which "
                    + (
                        "is not set"
                        if prompt_family is None
                        else f"names {prompt_family!r}, not a declared multiplicity family"
                    )
                    + ". Declare one under multiplicity.families, for example "
                    "{id: displayed, purpose: simultaneous_intervals, method: bonferroni, "
                    "alpha: 0.05}, and set outputs.prompt_bounds.family to its id; or turn "
                    "prompt bounds off with outputs.prompt_bounds.enabled: false"
                ),
                scope=plan_scope("outputs", "prompt_bounds"),
            )
        )
    elif prompt_bounds.enabled and prompt_bounds.simultaneous and prompt_family in families:
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
                    scope=plan_scope("outputs", "prompt_bounds"),
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
                    scope=plan_scope("outputs", "direct_mean_intervals"),
                )
            )
        elif direct.simultaneous and families[direct.family].method != "bonferroni":
            diagnostics.append(
                Diagnostic(
                    code="PLAN_INVALID_MULTIPLICITY",
                    severity="ERROR",
                    message="simultaneous direct mean intervals require a bonferroni family",
                    scope=plan_scope("outputs", "direct_mean_intervals"),
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
                    scope=plan_scope("outputs", "direct_mean_intervals"),
                )
            )
        # A direct worst_of_k interval is calibrated in the direct-mean family, so the section
        # is in use even without a weighted mean.
        if not any(
            measure.type == "weighted_mean" or measure.direct_interval
            for measure in measures.values()
        ):
            diagnostics.append(
                Diagnostic(
                    code="PLAN_METHOD_UNUSED",
                    severity="WARNING",
                    message="direct mean intervals are enabled but no weighted mean is requested",
                    scope=plan_scope("outputs", "direct_mean_intervals"),
                )
            )
    if plan.design.type == "sampled":
        nonlinear = sorted(
            measure.id
            for measure in plan.outputs.risk_measures
            if measure.type != "weighted_mean"
        )
        if nonlinear:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_SAMPLED_MEASURE_UNSUPPORTED",
                    severity="ERROR",
                    message=(
                        "a sampled design supports only weighted_mean measures; every other "
                        "measure is a nonlinear functional of the whole frame's risk vector "
                        "and cannot be evaluated from a sample: " + ", ".join(nonlinear)
                    ),
                    scope=plan_scope("outputs", "risk_measures"),
                )
            )
        if plan.expanded_comparisons():
            diagnostics.append(
                Diagnostic(
                    code="PLAN_SAMPLED_COMPARISON_UNSUPPORTED",
                    severity="ERROR",
                    message=(
                        "comparisons are not yet supported under a sampled design; the arm "
                        "constructions assume every declared prompt was evaluated"
                    ),
                    scope=plan_scope("comparisons"),
                )
            )
        if not plan.outputs.direct_mean_intervals.enabled:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_SAMPLED_REQUIRES_DIRECT_MEAN",
                    severity="ERROR",
                    message=(
                        "a sampled design requires outputs.direct_mean_intervals.enabled; the "
                        "propagated prompt envelope has no bound for unsampled frame items"
                    ),
                    scope=plan_scope("outputs", "direct_mean_intervals"),
                )
            )
        sampled_independence = plan.outputs.direct_mean_intervals.independence
        if (
            plan.outputs.direct_mean_intervals.enabled
            and sampled_independence is not None
            and sampled_independence.sensitivities
        ):
            # The frame item is the independent block of the frame intervals, so a declared
            # sensitivity profile has no interval and no claim verdict.
            diagnostics.append(
                Diagnostic(
                    code="PLAN_SAMPLED_SENSITIVITY_PROFILES",
                    severity="WARNING",
                    message=(
                        "a sampled design takes the frame item as the independent block, so no "
                        "interval or claim verdict is reported under the declared sensitivity "
                        "profiles: " + ", ".join(sampled_independence.sensitivities)
                    ),
                    scope=plan_scope("outputs", "direct_mean_intervals", "independence"),
                )
            )
        if plan.outputs.prompt_bounds.enabled and plan.outputs.prompt_bounds.simultaneous:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_SAMPLED_PROMPT_BOUNDS",
                    severity="WARNING",
                    message=(
                        "simultaneous prompt bounds under a sampled design cover only the "
                        "sampled frame items, not the frame"
                    ),
                    scope=plan_scope("outputs", "prompt_bounds"),
                )
            )
    if plan.outputs.direct_mean_intervals.single_valued_bound:
        unsupported = []
        if plan.design.type == "sampled":
            unsupported.append(
                "a sampled design, whose frame bounds are not a family of node bounds"
            )
        if plan.inputs.judge_calibration is not None:
            unsupported.append(
                "judge calibration, because the bound would describe the judge's risk, "
                "not the corrected risk"
            )
        if unsupported:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_SINGLE_VALUED_BOUND_UNSUPPORTED",
                    severity="ERROR",
                    message=(
                        "outputs.direct_mean_intervals.single_valued_bound is not supported "
                        "with " + " or ".join(unsupported) + "; set it to false"
                    ),
                    scope=plan_scope("outputs", "direct_mean_intervals"),
                )
            )
    calibration = plan.inputs.judge_calibration
    scopes: list[tuple[str, str, list[str] | None]] = [
        (
            "inputs.judge_calibration",
            plan_scope("inputs", "judge_calibration"),
            calibration.groups if calibration else None,
        ),
        ("inputs.panel_groups", plan_scope("inputs", "panel_groups"), plan.inputs.panel_groups),
        *(
            (
                f"inputs.judges.{judge.id}.accuracy",
                plan_scope("inputs", "judges", judge.id, "accuracy"),
                judge.accuracy.groups,
            )
            for judge in plan.inputs.judges
            if judge.accuracy is not None
        ),
    ]
    for location, location_scope, scoped in scopes:
        unknown_groups = sorted(set(scoped or []) - groups)
        if unknown_groups:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_UNKNOWN_REFERENCE",
                    severity="ERROR",
                    message=f"{location} references unknown groups: " + ", ".join(unknown_groups),
                    scope=location_scope,
                )
            )
    for scenario in calibration.scenarios if calibration else []:
        unknown_cohorts = sorted(set(scenario.cohorts or {}) - set(plan.cohorts))
        if unknown_cohorts:
            diagnostics.append(
                Diagnostic(
                    code="PLAN_UNKNOWN_REFERENCE",
                    severity="ERROR",
                    message=(
                        f"inputs.judge_calibration scenario {scenario.id!r} references unknown "
                        "cohorts: " + ", ".join(unknown_cohorts)
                    ),
                    scope=plan_scope("inputs", "judge_calibration"),
                )
            )
    diagnostics.extend(_worst_of_k_interval_diagnostics(plan))
    diagnostics.extend(anytime_bound_problems(plan))
    if not plan.outputs.risk_measures:
        diagnostics.append(
            Diagnostic(
                code="PLAN_NO_RISK_MEASURES",
                severity="WARNING",
                message=(
                    "outputs.risk_measures declares no measure, so no group risk is reported. "
                    "Add one, for example {id: mean, type: weighted_mean}"
                ),
                scope=plan_scope("outputs", "risk_measures"),
            )
        )
    return diagnostics


def worst_of_k_interval_problems(plan: AnalysisPlan) -> list[str]:
    """Why the plan cannot have the experimental direct worst_of_k interval it asks for.

    Empty when no measure asks for the interval, or when the plan supports it.
    ``validate_plan_references`` reports each problem as a PLAN_WORST_OF_K_INTERVAL_UNSUPPORTED
    error, and ``analyze`` withholds the interval from every row while any remains.
    """

    if not any(measure.direct_interval for measure in plan.outputs.risk_measures):
        return []
    direct = plan.outputs.direct_mean_intervals
    problems: list[str] = []
    if plan.design.type != "fixed":
        problems.append(
            "it bounds the fixed prompt set only, so it needs design.type: fixed, not "
            f"{plan.design.type!r}"
        )
    if not direct.enabled:
        problems.append(
            "it is calibrated in the direct-mean family, so it needs "
            "outputs.direct_mean_intervals.enabled"
        )
    elif direct.hierarchy_calibration == "frontier_sidak":
        problems.append(
            "frontier_sidak calibration is not supported; use flat_bonferroni or "
            "simultaneous: false"
        )
    if direct.enabled and direct.independence is not None:
        profile = plan.design.independence_profiles.get(direct.independence.primary)
        if profile is not None and profile.block_by != ["rollout"]:
            problems.append(
                "it needs independent rollouts, so the primary independence profile of "
                "outputs.direct_mean_intervals must block by rollout, not "
                f"{profile.block_by!r}"
            )
    return problems


def _worst_of_k_interval_diagnostics(plan: AnalysisPlan) -> list[Diagnostic]:
    """What the experimental worst-of-k direct interval cannot do yet, each an error."""

    requested = sorted(
        measure.id for measure in plan.outputs.risk_measures if measure.direct_interval
    )
    return [
        Diagnostic(
            code="PLAN_WORST_OF_K_INTERVAL_UNSUPPORTED",
            severity="ERROR",
            message=f"the direct worst_of_k interval of {', '.join(requested)}: {problem}",
            scope=plan_scope("outputs", "risk_measures"),
        )
        for problem in worst_of_k_interval_problems(plan)
    ]


def anytime_bound_problems(plan: AnalysisPlan) -> list[Diagnostic]:
    """What keeps the experimental anytime-valid bounds (#126) from being computed."""

    spec = plan.outputs.anytime_bounds
    if spec is None:
        return []
    problems: list[str] = []
    if plan.design.type != "fixed":
        problems.append(
            f"they bound a fixed design's weighted mean, so they need design.type: fixed, not "
            f"{plan.design.type!r}"
        )
    if spec.family not in {family.id for family in plan.multiplicity.families}:
        problems.append(f"family {spec.family!r} is not declared in multiplicity.families")
    unknown = sorted(set(spec.groups or []) - {group.id for group in plan.groups})
    if unknown:
        problems.append(f"groups {unknown!r} are not declared")
    profile = plan.design.independence_profiles.get(spec.independence_profile)
    if profile is None:
        problems.append(
            f"independence profile {spec.independence_profile!r} is not declared in "
            "design.independence_profiles"
        )
    elif profile.block_by != ["rollout"] and "prompt_id" not in profile.block_by:
        problems.append(
            f"independence profile {spec.independence_profile!r} blocks by "
            f"{profile.block_by!r}: the bound needs independent items, so the profile must "
            "block by rollout or by prompt_id"
        )
    return [
        Diagnostic(
            code="PLAN_ANYTIME_BOUNDS_INVALID",
            severity="ERROR",
            message=f"outputs.anytime_bounds: {problem}",
            scope=plan_scope("outputs", "anytime_bounds"),
        )
        for problem in problems
    ]
