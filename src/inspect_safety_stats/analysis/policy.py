"""Applying a decision policy's thresholds to a frozen plan, and recording it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from inspect_safety_stats.config.decision_policy import (
    DecisionPolicy,
    decision_policy_hash,
    inference_plan_hash,
    load_decision_policy,
    plan_with_thresholds,
    policy_thresholds,
)
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.schema.reports import (
    ClaimThresholdsRecord,
    DecisionPolicyRecord,
    DecisionPolicyVersionRecord,
    Diagnostic,
    PolicyThresholdsRecord,
)


@dataclass
class _PolicyApplication:
    """A decision policy, checked against the frozen plan, and the plan it decides with."""

    frozen: AnalysisPlan
    policy: DecisionPolicy
    applied: bool
    effective: AnalysisPlan


def _apply_decision_policy(
    plan: AnalysisPlan,
    policy: DecisionPolicy | str | Path | None,
    diagnostics: list[Diagnostic],
) -> _PolicyApplication | None:
    """Check a decision policy against the plan and give the plan with its thresholds.

    A policy that fails a check adds its errors to ``diagnostics``, which makes integrity
    invalid, and leaves the plan's thresholds in force.
    """

    if policy is None:
        return None
    from inspect_safety_stats.validate.decision_policy import validate_decision_policy

    resolved = policy if isinstance(policy, DecisionPolicy) else load_decision_policy(policy)
    problems = validate_decision_policy(plan, resolved)
    diagnostics.extend(problems)
    effective = plan if problems else plan_with_thresholds(plan, policy_thresholds(resolved))
    return _PolicyApplication(
        frozen=plan, policy=resolved, applied=not problems, effective=effective
    )


def _decision_policy_record(
    application: _PolicyApplication | None,
    first_started_at: datetime | None,
    diagnostics: list[Diagnostic],
) -> DecisionPolicyRecord | None:
    """The report's record of a decision policy, and a note when it was set after the run."""

    if application is None:
        return None
    policy = application.policy
    in_force = policy_thresholds(policy) if application.applied else {}
    versions = [
        DecisionPolicyVersionRecord(
            id=version.id,
            set_at=version.set_at.isoformat(),
            reason=version.reason,
            thresholds={
                claim_id: PolicyThresholdsRecord(
                    acceptable_upper=item.acceptable_upper,
                    unacceptable_lower=item.unacceptable_lower,
                )
                for claim_id, item in sorted(version.thresholds.items())
            },
            set_after_first_evaluation=(
                None if first_started_at is None else version.set_at > first_started_at
            ),
        )
        for version in policy.versions
    ]
    claims = [
        ClaimThresholdsRecord(
            claim_id=claim.id,
            acceptable_upper=(
                in_force[claim.id][0].acceptable_upper
                if claim.id in in_force
                else claim.acceptable_upper
            ),
            unacceptable_lower=(
                in_force[claim.id][0].unacceptable_lower
                if claim.id in in_force
                else claim.unacceptable_lower
            ),
            set_by=in_force[claim.id][1] if claim.id in in_force else "plan",
            plan_acceptable_upper=claim.acceptable_upper,
            plan_unacceptable_lower=claim.unacceptable_lower,
        )
        for claim in sorted(application.frozen.claims, key=lambda item: item.id)
    ]
    later = [version.id for version in versions if version.set_after_first_evaluation]
    if application.applied and later:
        diagnostics.append(
            Diagnostic(
                code="DECISION_POLICY_AFTER_GENERATION",
                severity="INFO",
                message=(
                    "decision policy version(s) "
                    + ", ".join(repr(item) for item in later)
                    + " were set after the first evaluation started. Thresholds do not enter "
                    "the bounds, so the verdicts at them remain certifying"
                ),
                details={"versions": later},
            )
        )
    return DecisionPolicyRecord(
        inference_plan_sha256=inference_plan_hash(application.frozen),
        policy_inference_plan_sha256=policy.inference_plan_sha256,
        policy_sha256=decision_policy_hash(policy),
        applied=application.applied,
        first_evaluation_started_at=(
            None if first_started_at is None else first_started_at.isoformat()
        ),
        versions=versions,
        claims=claims,
    )
