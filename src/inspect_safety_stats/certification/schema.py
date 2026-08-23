from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class CertificationResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


EvidenceStatus = Literal["SUPPORTED", "REJECTED", "INCONCLUSIVE"]


class DirectionalEvidence(CertificationResultModel):
    direction: Literal["support", "rejection"]
    family: str
    family_alpha: float
    local_alpha: float
    allocation_weight: float
    raw_p_value: float
    adjusted_p_value: float
    bound: float
    threshold: float
    margin: float
    crosses_threshold: bool
    method: str
    independence_profile: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class SensitivityResult(CertificationResultModel):
    method: str
    independence_profile: str
    lower: float
    upper: float
    support_raw_p_value: float
    rejection_raw_p_value: float
    evidence: EvidenceStatus


class DetectabilityResult(CertificationResultModel):
    method: Literal["exact_binomial", "weighted_hoeffding_guarantee"]
    independence_profile: str | None = None
    target_power: float
    maximum_certifiable_failure_rate: float | None = None
    minimum_detectable_failure_rate: float | None = None
    support_gap: float | None = None
    rejection_gap: float | None = None
    effective_sample_size: float
    guarantee: Literal["exact", "conservative"]
    assumptions: list[str] = Field(default_factory=list)


class CertificationClaimResult(CertificationResultModel):
    claim_id: str
    title: str
    target: dict[str, Any]
    threshold: float
    estimate: float | None = None
    completion_estimate_lower: float
    completion_estimate_upper: float
    support: DirectionalEvidence
    rejection: DirectionalEvidence | None = None
    evidence: EvidenceStatus
    sensitivities: list[SensitivityResult] = Field(default_factory=list)
    detectability: list[DetectabilityResult] = Field(default_factory=list)
    robustness: Literal["stable", "sensitive", "not_evaluated"]
    prompt_count: int
    attempted_rollouts: int
    unclassified_outcomes: int
    warnings: list[str] = Field(default_factory=list)


class CertificationReport(CertificationResultModel):
    schema_version: Literal["2.0"] = "2.0"
    package_version: str
    analysis: dict[str, Any]
    source: dict[str, Any]
    scope: dict[str, Any]
    families: list[dict[str, Any]]
    claims: list[CertificationClaimResult]
    summary: dict[str, int]
    warnings: list[str] = Field(default_factory=list)
    _resolved_plan: dict[str, Any] = PrivateAttr(default_factory=dict)

    def write(self, output: str | Path, *, overwrite: bool = False) -> Path:
        from inspect_safety_stats.certification.artifacts import write_certification_report

        return write_certification_report(self, output, overwrite=overwrite)
