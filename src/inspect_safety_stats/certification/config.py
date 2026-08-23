from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from inspect_safety_stats.config.canonicalize import sha256_value


class CertificationConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class CertificationMetadata(CertificationConfigModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str | None = None


class CertificationSourceSpec(CertificationConfigModel):
    require_plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")


class CertificationTargetSpec(CertificationConfigModel):
    type: Literal["prompt_failure_rate", "group_mean", "hierarchy_mean"]
    cohort: str = Field(min_length=1)
    prompt_id: str | None = None
    group: str | None = None
    level: str | None = None
    path: dict[str, str] | None = None
    all_paths: bool = False

    @model_validator(mode="after")
    def validate_target(self) -> CertificationTargetSpec:
        if self.type == "prompt_failure_rate":
            if not self.prompt_id:
                raise ValueError("prompt_failure_rate requires prompt_id")
            if self.group or self.level or self.path or self.all_paths:
                raise ValueError("prompt_failure_rate accepts only cohort and prompt_id")
        elif self.type == "group_mean":
            if not self.group:
                raise ValueError("group_mean requires group")
            if self.prompt_id or self.level or self.path or self.all_paths:
                raise ValueError("group_mean accepts only cohort and group")
        else:
            if not self.group or not self.level:
                raise ValueError("hierarchy_mean requires group and level")
            if self.prompt_id:
                raise ValueError("hierarchy_mean does not accept prompt_id")
            if self.all_paths == (self.path is not None):
                raise ValueError("hierarchy_mean requires exactly one of path or all_paths: true")
            if self.path is not None and not self.path:
                raise ValueError("hierarchy_mean path cannot be empty")
        return self


MeanMethod = Literal[
    "weighted_empirical_bernstein",
    "weighted_hoeffding",
    "weighted_kl_chernoff",
]


class DirectionSpec(CertificationConfigModel):
    family: str = Field(min_length=1)
    method: MeanMethod | Literal["clopper_pearson"] = "weighted_empirical_bernstein"
    independence_profile: str | None = None
    allocation_weight: float = Field(default=1.0, gt=0)


class RejectionSpec(DirectionSpec):
    enabled: bool = True


class SensitivitySpec(CertificationConfigModel):
    method: MeanMethod
    independence_profile: str = Field(min_length=1)


class CertificationClaimSpec(CertificationConfigModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    target: CertificationTargetSpec
    threshold: float = Field(gt=0, lt=1)
    support: DirectionSpec
    rejection: RejectionSpec
    sensitivity: list[SensitivitySpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_methods(self) -> CertificationClaimSpec:
        prompt = self.target.type == "prompt_failure_rate"
        directions = [self.support, self.rejection]
        if prompt and any(item.method != "clopper_pearson" for item in directions):
            raise ValueError("prompt_failure_rate requires clopper_pearson")
        if not prompt and any(item.method == "clopper_pearson" for item in directions):
            raise ValueError("fixed-design mean targets require a weighted concentration method")
        if prompt and self.sensitivity:
            raise ValueError("prompt_failure_rate does not support mean sensitivities")
        if prompt and any(item.independence_profile is not None for item in directions):
            raise ValueError("prompt_failure_rate does not use an independence profile")
        if not prompt and any(item.independence_profile is None for item in directions):
            raise ValueError("fixed-design mean targets require an independence profile")
        if (
            not prompt
            and self.rejection.enabled
            and self.support.independence_profile != self.rejection.independence_profile
        ):
            raise ValueError("support and rejection must use the same primary independence profile")
        pairs = [(item.method, item.independence_profile) for item in self.sensitivity]
        if len(set(pairs)) != len(pairs):
            raise ValueError("claim sensitivities must be unique")
        return self


class CertificationFamilySpec(CertificationConfigModel):
    id: str = Field(min_length=1)
    purpose: Literal["simultaneous_support", "simultaneous_rejection"]
    method: Literal["bonferroni", "weighted_bonferroni"] = "bonferroni"
    alpha: float = Field(default=0.05, gt=0, lt=1)


class CertificationMultiplicitySpec(CertificationConfigModel):
    families: list[CertificationFamilySpec]

    @model_validator(mode="after")
    def unique_families(self) -> CertificationMultiplicitySpec:
        ids = [item.id for item in self.families]
        if len(set(ids)) != len(ids):
            raise ValueError("certification multiplicity family ids must be unique")
        return self


class CertificationReportSpec(CertificationConfigModel):
    decimal_places: int = Field(default=6, ge=0, le=15)
    detectable_power: float | None = Field(default=0.9, gt=0, lt=1)


class CertificationPlan(CertificationConfigModel):
    schema_version: Literal["2.0"] = "2.0"
    certification: CertificationMetadata
    source: CertificationSourceSpec = Field(default_factory=CertificationSourceSpec)
    claims: list[CertificationClaimSpec]
    multiplicity: CertificationMultiplicitySpec
    report: CertificationReportSpec = Field(default_factory=CertificationReportSpec)
    _source_path: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_references(self) -> CertificationPlan:
        if not self.claims:
            raise ValueError("certification plan requires at least one claim")
        claim_ids = [item.id for item in self.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("certification claim ids must be unique")
        families = {item.id: item for item in self.multiplicity.families}
        for claim in self.claims:
            support = families.get(claim.support.family)
            if support is None or support.purpose != "simultaneous_support":
                raise ValueError(f"claim {claim.id!r} requires a simultaneous_support family")
            rejection = families.get(claim.rejection.family)
            if claim.rejection.enabled and (
                rejection is None or rejection.purpose != "simultaneous_rejection"
            ):
                raise ValueError(f"claim {claim.id!r} requires a simultaneous_rejection family")
        return self

    def source_path(self) -> Path | None:
        return self._source_path


def load_certification_plan(path: str | Path) -> CertificationPlan:
    source = Path(path).resolve()
    if source.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("certification plan must be YAML or JSON")
    if not isinstance(raw, dict):
        raise ValueError("certification plan root must be an object")
    plan = CertificationPlan.model_validate(raw)
    plan._source_path = source
    return plan


def canonical_certification_plan_hash(plan: CertificationPlan) -> str:
    return sha256_value(plan.model_dump(mode="json"))
