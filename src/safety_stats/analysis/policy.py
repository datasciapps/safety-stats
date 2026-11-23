"""Applying a decision policy's thresholds to a frozen plan, and recording it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from safety_stats.config.decision_policy import (
    DecisionPolicy,
    decision_policy_hash,
    inference_plan_hash,
    load_decision_policy,
    plan_with_thresholds,
    policy_thresholds,
)
from safety_stats.config.models import AnalysisPlan
from safety_stats.schema.reports import (
    ClaimThresholdsRecord,
    Decision,
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
    from safety_stats.validate.decision_policy import validate_decision_policy

    resolved = policy if isinstance(policy, DecisionPolicy) else load_decision_policy(policy)
    problems = validate_decision_policy(plan, resolved)
    diagnostics.extend(problems)
    effective = plan if problems else plan_with_thresholds(plan, policy_thresholds(resolved))
    return _PolicyApplication(
        frozen=plan, policy=resolved, applied=not problems, effective=effective
    )


def _policy_set_claims(application: _PolicyApplication | None) -> frozenset[str]:
    """The claims whose thresholds an applied policy sets."""

    if application is None or not application.applied:
        return frozenset()
    return frozenset(policy_thresholds(application.policy))


def _revised_mandatory_claims(application: _PolicyApplication | None) -> list[str]:
    """The mandatory claims whose ``acceptable_upper`` an applied policy moved from the plan's.

    Only ``acceptable_upper`` enters an overall PASS, which needs every mandatory claim's
    upper bound at or below it. A policy that restates the plan's value leaves the threshold
    the frozen plan fixed, so exact equality is intended here: both values are read from the
    same decimal text. Every version counts, whatever its ``set_at``: that time is the policy
    author's word, and nothing in the analysis can confirm it.

    Which claims a policy moves may itself have been chosen after seeing the bounds, so
    ``_policy_roll_up`` reads this list only as "some threshold may depend on the data", never
    as the only claims whose thresholds could.
    """

    if application is None or not application.applied:
        return []
    in_force = policy_thresholds(application.policy)
    return sorted(
        claim.id
        for claim in application.frozen.claims
        if claim.mandatory
        and claim.id in in_force
        and in_force[claim.id][0].acceptable_upper != claim.acceptable_upper
    )


def _policy_roll_up(
    application: _PolicyApplication | None,
    coverage_events: Mapping[str, str],
    decision: Decision,
    diagnostics: list[Diagnostic],
) -> Decision:
    """The overall decision, kept from PASS when a policy's thresholds leave it uncertified.

    With thresholds fixed before the data, an overall PASS is wrong only if some claim whose
    true risk is above its threshold passes, and which claims those are is fixed in advance:
    the chance is at most that one claim's error budget, with no correction (the
    intersection-union argument, docs/statistical-methods.md section 9.3). A threshold a
    policy sets may depend on the data, as one set at the upper bound it will be compared
    with does, and so may the choice of which claims to move. A wrong overall PASS then needs
    the bound of some mandatory claim, which one decided by the data, to miss. That chance is
    bounded by one error budget only when one coverage event covers every mandatory claim's
    bound; across separate events it can reach the sum of their budgets, even when the
    policy moves a single threshold: tightening the claim with the lowest upper bound to that
    bound picks the claim likeliest to have missed.

    So when a policy moves the acceptable_upper of any mandatory claim and the mandatory
    claims' bounds rest on more than one coverage event (``coverage_events``, from
    ``claims.coverage_event``), a ``DECISION_POLICY_OVERALL_UNCERTIFIED`` warning is added
    and a PASS becomes INCONCLUSIVE. A FAIL stays FAIL, the conservative direction, and every
    claim keeps its own status, whose error bound holds at any threshold.

    A mandatory ``noisy_valid_test`` claim counts with an event of its own, although a policy
    cannot move its thresholds and leaving it out would be sound: so the rule does not rest
    on that refusal (``validate_decision_policy``). Adaptive claims are not among
    ``frozen.claims``: they take no acceptable_upper and are never PASS, so a mandatory one
    keeps the overall decision from PASS already.
    """

    revised = _revised_mandatory_claims(application)
    if application is None or not revised:
        return decision
    events = {
        claim.id: coverage_events.get(claim.id, claim.id)
        for claim in application.frozen.claims
        if claim.mandatory
    }
    count = len(set(events.values()))
    if count <= 1:
        return decision
    diagnostics.append(
        Diagnostic(
            code="DECISION_POLICY_OVERALL_UNCERTIFIED",
            severity="WARNING",
            message=(
                "the decision policy moves acceptable_upper from the plan's for the mandatory "
                "claim(s) "
                + ", ".join(repr(item) for item in revised)
                + f", and the mandatory claims' bounds rest on {count} separate coverage "
                "events. A threshold set by a policy, and the choice of which claims to move, "
                "may depend on the data, and then the chance of a wrong overall PASS is "
                "bounded only by the sum of those events' error budgets, so the overall "
                "decision cannot certify a PASS: it is INCONCLUSIVE where every mandatory "
                "claim passes. Each claim's own verdict keeps its error bound"
            ),
            details={"claims": revised, "coverage_events": events},
        )
    )
    return "INCONCLUSIVE" if decision == "PASS" else decision


def _set_after_first_evaluation(
    set_at: datetime, first_started_at: datetime | None, start_unrecorded: bool
) -> bool | None:
    """Whether a version was set after the first evaluation started; None when unknown.

    A version set after the earliest recorded start was set after the first evaluation
    started, whatever the others' starts. One set at or before it is known to precede every
    evaluation only when every evaluation records its start: one that records none may have
    started earlier.
    """

    if first_started_at is None:
        return None
    if set_at > first_started_at:
        return True
    return None if start_unrecorded else False


def _decision_policy_record(
    application: _PolicyApplication | None,
    first_started_at: datetime | None,
    diagnostics: list[Diagnostic],
    *,
    start_unrecorded: bool = False,
) -> DecisionPolicyRecord | None:
    """The report's record of a decision policy, and a note when it was set after the run.

    ``first_started_at`` is the earliest start the selected evaluations record, and
    ``start_unrecorded`` whether some of them record none.
    """

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
            set_after_first_evaluation=_set_after_first_evaluation(
                version.set_at, first_started_at, start_unrecorded
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
    earlier = [
        version.id for version in versions if version.set_after_first_evaluation is False
    ]
    if application.applied and earlier:
        diagnostics.append(
            Diagnostic(
                code="DECISION_POLICY_SELF_DECLARED",
                severity="WARNING",
                message=(
                    "decision policy version(s) "
                    + ", ".join(repr(item) for item in earlier)
                    + " state a set_at before the first evaluation started. That time is the "
                    "policy author's statement, and nothing in the analysis confirms it, so "
                    "the report cannot show that these thresholds were fixed before the "
                    "outcomes were seen. Each claim's verdict keeps its error bound at any "
                    "threshold, and the overall decision does not rely on set_at"
                ),
                details={"versions": earlier},
            )
        )
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
                    "the bounds, so each claim's verdict keeps its error bound at them, but a "
                    "threshold set after the data may describe the bounds rather than a "
                    "requirement fixed in advance"
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
