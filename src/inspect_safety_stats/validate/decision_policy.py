from __future__ import annotations

from inspect_safety_stats.config.decision_policy import DecisionPolicy, inference_plan_hash
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.schema.scopes import DECISION_POLICY_ROOT, scope_path


def validate_decision_policy(plan: AnalysisPlan, policy: DecisionPolicy) -> list[Diagnostic]:
    """Errors that keep a decision policy from applying to this plan.

    A policy applies only to the plan whose inference hash it names, only to the plan's claims,
    and never to a ``noisy_valid_test`` claim, whose thresholds enter its test.
    """

    diagnostics: list[Diagnostic] = []
    expected = inference_plan_hash(plan)
    if policy.inference_plan_sha256 != expected:
        diagnostics.append(
            Diagnostic(
                code="DECISION_POLICY_MISMATCH",
                severity="ERROR",
                message=(
                    "the decision policy names inference plan "
                    f"{policy.inference_plan_sha256}, but this plan's inference hash is "
                    f"{expected}; the plan changed in something that shapes its bounds or its "
                    "claims, or the policy was written for another plan. Print the hash with "
                    "`inspect-safety-stats hash-plan --inference`"
                ),
                details={
                    "policy_inference_plan_sha256": policy.inference_plan_sha256,
                    "inference_plan_sha256": expected,
                },
            )
        )
    claims = {claim.id: claim for claim in plan.claims}
    for version in policy.versions:
        for claim_id in sorted(version.thresholds):
            claim = claims.get(claim_id)
            if claim is None:
                diagnostics.append(
                    Diagnostic(
                        code="DECISION_POLICY_UNKNOWN_CLAIM",
                        severity="ERROR",
                        message=(
                            f"decision policy version {version.id!r} sets thresholds for "
                            f"claim {claim_id!r}, which the plan does not declare"
                        ),
                        scope=scope_path(
                            DECISION_POLICY_ROOT, "versions", version.id, "thresholds", claim_id
                        ),
                    )
                )
            elif claim.method == "noisy_valid_test":
                diagnostics.append(
                    Diagnostic(
                        code="DECISION_POLICY_FIXED_CLAIM",
                        severity="ERROR",
                        message=(
                            f"decision policy version {version.id!r} sets thresholds for "
                            f"claim {claim_id!r}, a noisy_valid_test claim, whose test uses its "
                            "thresholds; they can be set only in the plan, before the run"
                        ),
                        scope=scope_path(
                            DECISION_POLICY_ROOT, "versions", version.id, "thresholds", claim_id
                        ),
                    )
                )
    return diagnostics
