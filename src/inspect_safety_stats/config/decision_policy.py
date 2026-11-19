"""Decision policies: claim thresholds kept apart from the analysis plan (experimental).

A claim's thresholds never enter its bounds. The decision rule only compares a finished bound
with them, so the confidence statement behind one claim's verdict holds at any thresholds. The
analysis plan freezes what shapes the bounds and the family of claims; a decision policy sets or
revises the thresholds of ``corrected_interval`` claims, possibly after the run, without
changing the plan or its hash. A policy names the plan it applies to by the plan's inference
hash, the plan's canonical hash with those thresholds left out (``inference_plan_hash``).

The overall decision's error bound needs more: once a policy moves a threshold, a wrong overall
PASS stays as rare as one wrong claim only when one coverage event covers every mandatory claim.
``analysis.policy._policy_roll_up`` holds back an overall PASS otherwise.

``noisy_valid_test`` claims use their thresholds inside the test, so their thresholds stay in
the inference hash and a policy cannot revise them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AwareDatetime, Field, model_validator

from .canonicalize import sha256_value
from .models import AnalysisPlan, PlanFloat, StrictModel

DECISION_POLICY_SCHEMA_VERSIONS = ("1.0",)


class PolicyThresholds(StrictModel):
    acceptable_upper: PlanFloat = Field(ge=-1, le=1)
    unacceptable_lower: PlanFloat = Field(ge=-1, le=1)

    @model_validator(mode="after")
    def ordered(self) -> PolicyThresholds:
        if self.acceptable_upper >= self.unacceptable_lower:
            raise ValueError("acceptable_upper must be less than unacceptable_lower")
        return self


class DecisionPolicyVersion(StrictModel):
    id: str = Field(min_length=1)
    # When this version was decided. A timezone is required, so that comparing it with the
    # evaluations' start times cannot depend on where the policy was read.
    set_at: AwareDatetime
    reason: str | None = None
    thresholds: dict[str, PolicyThresholds] = Field(min_length=1)


class DecisionPolicy(StrictModel):
    """A versioned decision policy, read with ``load_decision_policy``.

    Versions apply in the order given, which must be the order of their ``set_at`` times: the
    thresholds in force for a claim are those of the last version that sets them, or the plan's
    own when no version does. The fields are documented in docs/decision-policy.md.
    """

    schema_version: Literal["1.0"] = "1.0"
    inference_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    versions: list[DecisionPolicyVersion] = Field(min_length=1)

    @model_validator(mode="after")
    def versions_are_ordered_and_unique(self) -> DecisionPolicy:
        ids = [version.id for version in self.versions]
        repeated = sorted({value for value in ids if ids.count(value) > 1})
        if repeated:
            raise ValueError("decision policy version ids must be unique: " + ", ".join(repeated))
        for earlier, later in zip(self.versions, self.versions[1:], strict=False):
            if later.set_at <= earlier.set_at:
                raise ValueError(
                    f"decision policy version {later.id!r} is set at or before version "
                    f"{earlier.id!r}; list versions in the order they were set"
                )
        return self


def load_decision_policy(path: str | Path) -> DecisionPolicy:
    """Read and validate a decision policy from a YAML (``.yaml``, ``.yml``) or JSON file.

    Raises ``ValueError`` for any other suffix, a root that is not an object, or an
    unsupported schema version, and pydantic's ``ValidationError`` for an invalid policy.
    """

    source = Path(path)
    if source.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("decision policy must be YAML or JSON")
    if not isinstance(raw, dict):
        raise ValueError("decision policy root must be an object")
    declared = raw.get("schema_version", "1.0")
    if isinstance(declared, float):
        declared = raw["schema_version"] = str(declared)
    if declared not in DECISION_POLICY_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported decision policy schema_version {declared!r}; this release reads "
            + " and ".join(repr(version) for version in DECISION_POLICY_SCHEMA_VERSIONS)
        )
    return DecisionPolicy.model_validate(raw)


def decision_policy_hash(policy: DecisionPolicy) -> str:
    """The canonical SHA-256 of a decision policy, as a report records it."""

    return sha256_value(policy.model_dump(mode="json"))


def inference_plan_hash(plan: AnalysisPlan) -> str:
    """A plan's canonical SHA-256 with the thresholds a decision policy may set left out.

    Those are ``acceptable_upper`` and ``unacceptable_lower`` of every ``corrected_interval``
    claim. Everything else, ``noisy_valid_test`` thresholds included, is hashed as
    ``canonical_plan_hash`` hashes it, so two plans that differ only in those thresholds share
    an inference hash, and any other difference changes it.
    """

    from .loader import canonical_plan_value

    value = canonical_plan_value(plan)
    for claim in value["claims"]:
        if claim.get("method", "corrected_interval") == "corrected_interval":
            claim.pop("acceptable_upper", None)
            claim.pop("unacceptable_lower", None)
    return sha256_value(value)


def policy_thresholds(policy: DecisionPolicy) -> dict[str, tuple[PolicyThresholds, str]]:
    """For each claim a policy sets, the thresholds in force and the version that set them."""

    in_force: dict[str, tuple[PolicyThresholds, str]] = {}
    for version in policy.versions:
        for claim_id, thresholds in version.thresholds.items():
            in_force[claim_id] = (thresholds, version.id)
    return in_force


def plan_with_thresholds(
    plan: AnalysisPlan, thresholds: dict[str, tuple[PolicyThresholds, str]]
) -> AnalysisPlan:
    """A copy of the plan whose claims carry the given thresholds; the plan is not changed."""

    claims = [
        claim.model_copy(
            update={
                "acceptable_upper": thresholds[claim.id][0].acceptable_upper,
                "unacceptable_lower": thresholds[claim.id][0].unacceptable_lower,
            }
        )
        if claim.id in thresholds
        else claim
        for claim in plan.claims
    ]
    return plan.model_copy(update={"claims": claims})
