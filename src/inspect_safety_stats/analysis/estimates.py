"""Prompt-level estimates: each prompt's rate and its bounds, per cohort."""

from __future__ import annotations

from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.manifest import resolve_weights
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.schema.reports import Diagnostic, Eligibility, PromptEstimate
from inspect_safety_stats.stats.binomial import binomial_bounds, missingness_robust_binomial_bounds
from inspect_safety_stats.stats.independence import BlockIndex, block_index
from inspect_safety_stats.stats.multiplicity import bonferroni_allocate


def _alpha_by_prompt(
    plan: AnalysisPlan,
    data: NormalizedData,
    diagnostics: list[Diagnostic],
) -> dict[str, float]:
    if not plan.outputs.prompt_bounds.enabled:
        return {}
    family_id = plan.outputs.prompt_bounds.family
    families = {family.id: family for family in plan.multiplicity.families}
    if family_id not in families:
        return {}
    alpha = families[family_id].alpha
    directional_alpha = (
        alpha / 2 if plan.outputs.prompt_bounds.coverage_mode == "two_sided" else alpha
    )
    prompts = sorted(data.manifest["prompt_id"].astype(str).tolist())
    cohort_count = len(plan.cohorts)
    if not plan.outputs.prompt_bounds.simultaneous:
        return {prompt: directional_alpha for prompt in prompts}
    strategy = plan.outputs.prompt_bounds.alpha_allocation
    weights: list[float] | None = None
    if strategy == "weighted_bonferroni":
        try:
            frame = data.manifest.sort_values("prompt_id", kind="stable")
            weights = resolve_weights(frame, plan.design.weights).tolist()
            if any(weight <= 0 for weight in weights):
                raise ValueError("all alpha weights must be positive")
        except ValueError as exc:
            diagnostics.append(
                Diagnostic(
                    code="ALPHA_ALLOCATION_INVALID",
                    severity="ERROR",
                    message=str(exc),
                    scope=family_id,
                )
            )
            return {}
    elif strategy == "explicit":
        explicit = plan.outputs.prompt_bounds.explicit_alpha or {}
        manifest_prompts = set(prompts)
        if set(explicit) != manifest_prompts or sum(explicit.values()) > alpha + 1e-12:
            diagnostics.append(
                Diagnostic(
                    code="ALPHA_ALLOCATION_INVALID",
                    severity="ERROR",
                    message=(
                        "explicit_alpha must contain every manifest prompt exactly once "
                        "and its values must sum to no more than the family alpha"
                    ),
                    scope=family_id,
                )
            )
            return {}
        tail_factor = 0.5 if plan.outputs.prompt_bounds.coverage_mode == "two_sided" else 1.0
        return {prompt: explicit[prompt] * tail_factor / cohort_count for prompt in prompts}
    if weights is not None:
        weights = [weight / cohort_count for weight in weights]
        return {
            prompt: directional_alpha * weight
            for prompt, weight in zip(prompts, weights, strict=True)
        }
    endpoint_alpha = bonferroni_allocate(directional_alpha, len(prompts) * cohort_count)[0]
    return {prompt: endpoint_alpha for prompt in prompts}


def _block_index(data: NormalizedData) -> BlockIndex:
    """The block index for this analysis's observations, built on first use."""
    if data.block_index is None:
        data.block_index = block_index(data.observations, data.manifest)
    return data.block_index


def _prompt_blocked(diagnostics: list[Diagnostic], cohort: str, prompt: str) -> bool:
    global_codes = {
        "PLAN_DUPLICATE_ID",
        "PLAN_UNKNOWN_REFERENCE",
        "PLAN_DIAGNOSTIC_GATE",
        "PLAN_POSTHOC",
        "INGEST_FAILED",
        "COHORT_OVERLAP",
        "ALPHA_ALLOCATION_INVALID",
        "METHOD_UNAVAILABLE",
    }
    for diagnostic in diagnostics:
        if diagnostic.severity != "ERROR":
            continue
        if diagnostic.code in global_codes or diagnostic.scope is None:
            return True
        # An error about one prompt names it in its details, with its cohort when it concerns
        # only one. Scope strings are for people. Matching their prefixes let an error about
        # p10 block p1, and an error about group g1 block a prompt named g1.
        details = diagnostic.details
        if details.get("prompt_id") is not None:
            if details["prompt_id"] == prompt and details.get("cohort_id") in (None, cohort):
                return True
            continue
        if diagnostic.scope == cohort:
            return True
    return False


def _bounds_carry_unknown_outcome(plan: AnalysisPlan, row: Observation) -> bool:
    """Whether an unclassified rollout stays in its prompt's bounds as an unknown outcome.

    Under the other policies it is blocked or excluded, and leaves the prompt's denominator.
    """
    if row.failure is not None:
        return False
    if row.error is not None:
        return plan.inputs.score.errors in {"end_to_end", "conservative_bounds"}
    return plan.inputs.score.missing == "conservative_bounds"


def _prompt_estimates(
    plan: AnalysisPlan,
    data: NormalizedData,
    diagnostics: list[Diagnostic],
) -> list[PromptEstimate]:
    allocations = _alpha_by_prompt(plan, data, diagnostics)
    manifest = data.manifest.set_index("prompt_id", drop=False)
    by_key: dict[tuple[str, str], list[Observation]] = {}
    for observation in data.observations:
        by_key.setdefault((observation.cohort_id, observation.prompt_id), []).append(observation)
    results: list[PromptEstimate] = []
    for cohort_id in sorted(plan.cohorts):
        for prompt_id in sorted(manifest.index.astype(str)):
            rows = by_key.get((cohort_id, prompt_id), [])
            valid = [row for row in rows if row.failure is not None]
            failures = sum(bool(row.failure) for row in valid)
            bounded_unknown = [row for row in rows if _bounds_carry_unknown_outcome(plan, row)]
            blocked = _prompt_blocked(diagnostics, cohort_id, prompt_id)
            alpha = allocations.get(prompt_id)
            lower = upper = None
            eligibility: Eligibility = "invalid"
            warnings: list[str] = []
            bounds = None
            if alpha is not None and not blocked and bounded_unknown:
                bounds = missingness_robust_binomial_bounds(
                    failures,
                    len(valid),
                    len(valid) + len(bounded_unknown),
                    alpha_lower=alpha,
                    alpha_upper=alpha,
                )
                warnings.append(
                    f"Bounds cover all {len(bounded_unknown)} possible unclassified outcomes."
                )
            elif valid and alpha is not None and not blocked:
                bounds = binomial_bounds(
                    failures,
                    len(valid),
                    alpha_lower=alpha,
                    alpha_upper=alpha,
                )
            if bounds is not None:
                lower, upper = bounds.lower, bounds.upper
                eligibility = (
                    "certifying" if plan.outputs.prompt_bounds.simultaneous else "diagnostic"
                )
            elif valid:
                warnings.append(
                    "Point estimate is descriptive because certifying bounds are unavailable."
                )
            item = manifest.loc[prompt_id]
            expected = int(item["expected_rollouts"]) if "expected_rollouts" in item else None
            results.append(
                PromptEstimate(
                    cohort_id=cohort_id,
                    prompt_id=prompt_id,
                    question_id=str(item["question_id"]),
                    task_id=str(item["task_id"]),
                    n_total=len(rows),
                    n_expected=expected,
                    n_valid=len(valid),
                    n_failure=failures,
                    n_safe=len(valid) - failures,
                    n_error=sum(row.error is not None for row in rows),
                    n_unclassified=sum(row.failure is None for row in rows),
                    n_policy_blocked_safe=sum(
                        row.outcome_status == "target_policy_blocked_safe" for row in rows
                    ),
                    p_hat=failures / len(valid) if valid else None,
                    lower=lower,
                    upper=upper,
                    alpha_lower=alpha,
                    alpha_upper=alpha,
                    method=bounds.method if bounds is not None else None,
                    missingness_robust=bool(bounded_unknown and bounds is not None),
                    eligibility=eligibility,
                    warnings=warnings,
                )
            )
    return results
