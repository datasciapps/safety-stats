from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
from inspect_safety_stats.config.models import AnalysisPlan, StructuralModelSpec
from inspect_safety_stats.schema.structural_reports import (
    CrossValidationSummary,
    HierarchyEffectSummary,
    StructuralFitSummary,
    StructuralReport,
)
from inspect_safety_stats.structural.audit_metrics import (
    concentration_summaries,
    hierarchy_marginal_risk_summaries,
    posterior_summary,
    predictive_checks,
    prompt_risk_summaries,
    vulnerable_mass_summaries,
    weighted_mean_risk_draws,
)
from inspect_safety_stats.structural.backend import BackendFit
from inspect_safety_stats.structural.cross_validation import prompt_level_cross_validation
from inspect_safety_stats.structural.data import (
    ModelFrame,
    build_model_frame,
    model_frame_metadata,
)
from inspect_safety_stats.structural.pymc_backend import PyMCStructuralBackend, backend_versions
from inspect_safety_stats.version import __version__


def _coerce_plan(plan: AnalysisPlan | str | Path) -> AnalysisPlan:
    return plan if isinstance(plan, AnalysisPlan) else load_plan(plan)


def validate_structural_inputs(
    *, plan: AnalysisPlan | str | Path, empirical_report: str | Path
) -> dict[str, object]:
    return model_frame_metadata(build_model_frame(plan=plan, empirical_report=empirical_report))


def _fit_summary(
    *,
    fit_id: str,
    backend_fit: BackendFit,
    model_frame: ModelFrame,
    spec: StructuralModelSpec,
    frailty: Literal["student_t", "normal"],
    cross_validation: CrossValidationSummary,
) -> StructuralFitSummary:
    metrics = spec.audit_metrics
    prior_counts = backend_fit.prior_predictive_counts
    posterior_counts = backend_fit.posterior_predictive_counts
    if prior_counts is None or posterior_counts is None:
        raise RuntimeError("full structural fits require prior and posterior predictive draws")
    return StructuralFitSummary(
        fit_id=fit_id,
        link=spec.link,
        prompt_frailty=frailty,
        diagnostics=backend_fit.diagnostics,
        weighted_mean_risk=posterior_summary(
            weighted_mean_risk_draws(backend_fit.risk_draws, model_frame.weights),
            credible_mass=metrics.credible_interval,
        ),
        prompt_frailty_scale=posterior_summary(
            backend_fit.prompt_scale_draws,
            credible_mass=metrics.credible_interval,
        ),
        hierarchy_marginal_risks=hierarchy_marginal_risk_summaries(
            backend_fit.risk_draws,
            model_frame.weights,
            hierarchy=model_frame.hierarchy,
            hierarchy_values={
                level: model_frame.frame[level].astype(str) for level in model_frame.hierarchy
            },
            credible_mass=metrics.credible_interval,
        ),
        hierarchy_effects=[
            HierarchyEffectSummary(
                level=column,
                category=category,
                log_intensity_effect=posterior_summary(
                    backend_fit.hierarchy_effect_draws[column][:, category_index],
                    credible_mass=metrics.credible_interval,
                ),
                intensity_multiplier=posterior_summary(
                    np.exp(backend_fit.hierarchy_effect_draws[column][:, category_index]),
                    credible_mass=metrics.credible_interval,
                ),
            )
            for column in model_frame.hierarchy
            if column in backend_fit.hierarchy_effect_draws
            for category_index, category in enumerate(model_frame.category_levels[column])
        ],
        prompt_risks=prompt_risk_summaries(
            backend_fit.risk_draws,
            model_frame.prompt_ids,
            model_frame.weights,
            thresholds=metrics.vulnerable_thresholds,
            credible_mass=metrics.credible_interval,
        ),
        vulnerable_mass=vulnerable_mass_summaries(
            backend_fit.risk_draws,
            model_frame.weights,
            thresholds=metrics.vulnerable_thresholds,
            credible_mass=metrics.credible_interval,
        ),
        concentration=concentration_summaries(
            backend_fit.risk_draws,
            model_frame.weights,
            prompt_masses=metrics.concentration_mass,
            credible_mass=metrics.credible_interval,
            zero_risk_tolerance=metrics.zero_risk_tolerance,
        ),
        prior_predictive=predictive_checks(
            prior_counts,
            model_frame.failures,
            model_frame.trials,
            model_frame.weights,
            credible_mass=metrics.credible_interval,
        ),
        posterior_predictive=predictive_checks(
            posterior_counts,
            model_frame.failures,
            model_frame.trials,
            model_frame.weights,
            credible_mass=metrics.credible_interval,
        ),
        cross_validation=cross_validation,
    )


def _status(
    primary: StructuralFitSummary, sensitivities: list[StructuralFitSummary]
) -> Literal["valid", "degraded", "invalid"]:
    statuses = [primary.diagnostics.status, *(fit.diagnostics.status for fit in sensitivities)]
    if primary.cross_validation.status != "disabled":
        statuses.append(primary.cross_validation.status)
    if "invalid" in statuses:
        return "invalid"
    if "degraded" in statuses:
        return "degraded"
    return "valid"


def fit_structural_model(
    *,
    plan: AnalysisPlan | str | Path,
    empirical_report: str | Path,
    run_cross_validation: bool = True,
) -> StructuralReport:
    resolved_plan = _coerce_plan(plan)
    spec = resolved_plan.structural_model
    if spec is None or not spec.enabled:
        raise ValueError("the analysis plan does not enable structural_model")
    model_frame = build_model_frame(plan=resolved_plan, empirical_report=empirical_report)
    primary_frailty = spec.prompt_frailty.distribution
    primary_backend = PyMCStructuralBackend(model_frame, spec).fit(frailty=primary_frailty)
    cross_validation = (
        prompt_level_cross_validation(model_frame, spec)
        if run_cross_validation
        else CrossValidationSummary(
            enabled=False,
            status="disabled",
            folds=0,
            prompt_count=len(model_frame.frame),
            warnings=["Cross-validation was disabled by the caller."],
        )
    )
    primary = _fit_summary(
        fit_id=f"primary-{primary_frailty}",
        backend_fit=primary_backend,
        model_frame=model_frame,
        spec=spec,
        frailty=primary_frailty,
        cross_validation=cross_validation,
    )

    sensitivity_summaries: list[StructuralFitSummary] = []
    sensitivity_backends: dict[str, BackendFit] = {}
    if spec.prompt_frailty.gaussian_sensitivity and primary_frailty != "normal":
        sensitivity_backend = PyMCStructuralBackend(model_frame, spec).fit(
            frailty="normal", seed_offset=1_000_000
        )
        sensitivity_id = "sensitivity-normal-frailty"
        sensitivity_backends[sensitivity_id] = sensitivity_backend
        sensitivity_summaries.append(
            _fit_summary(
                fit_id=sensitivity_id,
                backend_fit=sensitivity_backend,
                model_frame=model_frame,
                spec=spec,
                frailty="normal",
                cross_validation=CrossValidationSummary(
                    enabled=False,
                    status="disabled",
                    folds=0,
                    prompt_count=len(model_frame.frame),
                    warnings=["Cross-validation is reported for the primary fit only."],
                ),
            )
        )

    primary_mean = primary.weighted_mean_risk
    empirical_mean = model_frame.empirical_weighted_mean
    consistency = {
        "role": "diagnostic_only",
        "empirical_weighted_mean": empirical_mean,
        "posterior_weighted_mean": primary_mean.model_dump(mode="json"),
        "empirical_estimate_inside_posterior_interval": (
            primary_mean.lower <= empirical_mean <= primary_mean.upper
            if empirical_mean is not None
            else None
        ),
        "absolute_difference": (
            abs(primary_mean.mean - empirical_mean) if empirical_mean is not None else None
        ),
    }
    warnings = [
        "Structural posterior intervals are model-conditional credible intervals, not "
        "finite-sample confidence bounds.",
        "Results describe the declared fixed evaluation design and do not establish coverage "
        "of an external prompt population.",
    ]
    report = StructuralReport(
        package_version=__version__,
        status=_status(primary, sensitivity_summaries),
        analysis={
            "id": resolved_plan.analysis.id,
            "title": f"{resolved_plan.analysis.title} — structural safety landscape",
            "plan_sha256": canonical_plan_hash(resolved_plan),
        },
        scope={
            "design": "fixed",
            "randomness": "stochastic rollouts within fixed prompts",
            "inferential_kind": "bayesian_structural_model",
            "interpretation": (
                "Posterior summaries characterize heterogeneity and concentration within the "
                "declared weighted design, conditional on the model and priors."
            ),
        },
        provenance={
            "empirical_report": str(model_frame.empirical_report_path),
            "empirical_report_sha256": model_frame.empirical_report_sha256,
            "backend": "pymc_nuts",
            "backend_versions": backend_versions(),
        },
        model=spec.model_dump(mode="json"),
        data=model_frame_metadata(model_frame),
        primary_fit=primary,
        sensitivity_fits=sensitivity_summaries,
        consistency=consistency,
        warnings=warnings,
    )
    report._model_frame = model_frame.records()
    report._posterior = primary_backend.inference_data
    report._sensitivity_posteriors = {
        fit_id: backend.inference_data for fit_id, backend in sensitivity_backends.items()
    }
    report._resolved_model = spec.model_dump(mode="json")
    return report
