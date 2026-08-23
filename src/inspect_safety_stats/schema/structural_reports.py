from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class StructuralResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PosteriorSummary(StructuralResultModel):
    mean: float
    median: float
    sd: float
    lower: float
    upper: float
    credible_mass: float


class PromptRiskSummary(StructuralResultModel):
    prompt_id: str
    weight: float
    risk: PosteriorSummary
    exceedance_probability: dict[str, float] = Field(default_factory=dict)


class HierarchyEffectSummary(StructuralResultModel):
    level: str
    category: str
    log_intensity_effect: PosteriorSummary
    intensity_multiplier: PosteriorSummary


class HierarchyRiskSummary(StructuralResultModel):
    level: str
    category: str
    path: dict[str, str]
    design_weight: float
    prompt_count: int
    risk: PosteriorSummary


class VulnerableMassSummary(StructuralResultModel):
    threshold: float
    mass: PosteriorSummary
    expected_exceedance_mass: float


class ConcentrationSummary(StructuralResultModel):
    prompt_mass: float
    risk_share: PosteriorSummary | None = None
    absolute_tail_risk: PosteriorSummary
    near_zero_total_risk_probability: float


class PredictiveStatistic(StructuralResultModel):
    name: str
    observed: float
    replicated: PosteriorSummary
    posterior_predictive_p_lower: float
    posterior_predictive_p_upper: float


class PredictiveChecks(StructuralResultModel):
    statistics: list[PredictiveStatistic] = Field(default_factory=list)


class SamplerDiagnostics(StructuralResultModel):
    status: Literal["valid", "degraded", "invalid"]
    chains: int
    draws_per_chain: int
    divergences: int
    max_rhat: float | None = None
    min_bulk_ess: float | None = None
    min_tail_ess: float | None = None
    warnings: list[str] = Field(default_factory=list)


class CrossValidationSummary(StructuralResultModel):
    enabled: bool
    status: Literal["disabled", "valid", "degraded", "invalid"] = "disabled"
    folds: int
    prompt_count: int
    mean_log_predictive_density: float | None = None
    mean_brier_score: float | None = None
    weighted_calibration_error: float | None = None
    warnings: list[str] = Field(default_factory=list)


class StructuralFitSummary(StructuralResultModel):
    fit_id: str
    link: Literal["cloglog", "logit"]
    prompt_frailty: Literal["student_t", "normal"]
    diagnostics: SamplerDiagnostics
    weighted_mean_risk: PosteriorSummary
    prompt_frailty_scale: PosteriorSummary
    hierarchy_marginal_risks: list[HierarchyRiskSummary]
    hierarchy_effects: list[HierarchyEffectSummary]
    prompt_risks: list[PromptRiskSummary]
    vulnerable_mass: list[VulnerableMassSummary]
    concentration: list[ConcentrationSummary]
    prior_predictive: PredictiveChecks
    posterior_predictive: PredictiveChecks
    cross_validation: CrossValidationSummary


class StructuralReport(StructuralResultModel):
    schema_version: Literal["2.0"] = "2.0"
    package_version: str
    certification_eligible: Literal[False] = False
    status: Literal["valid", "degraded", "invalid"]
    analysis: dict[str, Any]
    scope: dict[str, Any]
    provenance: dict[str, Any]
    model: dict[str, Any]
    data: dict[str, Any]
    primary_fit: StructuralFitSummary
    sensitivity_fits: list[StructuralFitSummary] = Field(default_factory=list)
    consistency: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    _model_frame: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _posterior: Any = PrivateAttr(default=None)
    _sensitivity_posteriors: dict[str, Any] = PrivateAttr(default_factory=dict)
    _resolved_model: dict[str, Any] = PrivateAttr(default_factory=dict)

    def write(self, output: str | Path, *, overwrite: bool = False) -> Path:
        from inspect_safety_stats.structural.artifacts import write_structural_report

        return write_structural_report(self, output, overwrite=overwrite)
