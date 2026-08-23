from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

from inspect_safety_stats.config.models import StructuralModelSpec
from inspect_safety_stats.schema.structural_reports import CrossValidationSummary
from inspect_safety_stats.structural.data import ModelFrame
from inspect_safety_stats.structural.pymc_backend import PyMCStructuralBackend


def _flatten(variable: object) -> np.ndarray:
    values = np.asarray(variable, dtype=float)
    return values.reshape((-1, *values.shape[2:]))


def _subset(frame: ModelFrame, indices: np.ndarray) -> ModelFrame:
    selected = frame.frame.iloc[indices].reset_index(drop=True)
    return ModelFrame(
        frame=selected,
        cohort_id=frame.cohort_id,
        group_id=frame.group_id,
        declared_hierarchy=frame.declared_hierarchy,
        hierarchy=frame.hierarchy,
        prompt_frailty_levels=frame.prompt_frailty_levels,
        category_levels=frame.category_levels,
        category_codes={column: codes[indices] for column, codes in frame.category_codes.items()},
        empirical_report_path=frame.empirical_report_path,
        empirical_report_sha256=frame.empirical_report_sha256,
        empirical_weighted_mean=frame.empirical_weighted_mean,
    )


def _held_out_risk_draws(
    *,
    inference_data: object,
    full_frame: ModelFrame,
    test_indices: np.ndarray,
    spec: StructuralModelSpec,
    frailty: str,
    rng: np.random.Generator,
) -> np.ndarray:
    posterior = inference_data.posterior  # type: ignore[attr-defined]
    mu = _flatten(posterior["mu"]).reshape(-1)
    eta = np.repeat(mu[:, None], len(test_indices), axis=1)
    for index, column in enumerate(full_frame.hierarchy):
        if len(full_frame.category_levels[column]) < 2:
            continue
        effect = _flatten(posterior[f"effect_hierarchy_{index}"])
        eta += effect[:, full_frame.category_codes[column][test_indices]]
    sigma_prompt = _flatten(posterior["sigma_prompt"]).reshape(-1)
    if frailty == "student_t":
        raw = rng.standard_t(
            df=spec.prompt_frailty.degrees_of_freedom,
            size=(len(mu), len(test_indices)),
        )
    else:
        raw = rng.normal(size=(len(mu), len(test_indices)))
    eta += sigma_prompt[:, None] * raw
    eta = np.clip(eta, -40, 20)
    return np.asarray(-np.expm1(-np.exp(eta)), dtype=float)


def prompt_level_cross_validation(
    model_frame: ModelFrame,
    spec: StructuralModelSpec,
) -> CrossValidationSummary:
    folds = spec.inference.cross_validation_folds
    prompt_count = len(model_frame.frame)
    if folds == 0:
        return CrossValidationSummary(
            enabled=False, status="disabled", folds=0, prompt_count=prompt_count
        )
    if folds > prompt_count:
        raise ValueError("cross-validation folds cannot exceed the prompt count")

    rng = np.random.default_rng(spec.inference.seed + 100_000)
    partitions = np.array_split(rng.permutation(prompt_count), folds)
    predicted = np.zeros(prompt_count, dtype=float)
    log_predictive = np.zeros(prompt_count, dtype=float)
    warnings: list[str] = []
    statuses: list[str] = []
    all_indices = np.arange(prompt_count)
    frailty = spec.prompt_frailty.distribution
    for fold_index, test_indices in enumerate(partitions):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        for column in model_frame.hierarchy:
            train_levels = set(model_frame.category_codes[column][train_indices])
            unseen = set(model_frame.category_codes[column][test_indices]) - train_levels
            if unseen:
                warnings.append(
                    f"Fold {fold_index + 1} holds out {len(unseen)} previously unseen "
                    f"{column!r} levels; predictions for those effects are prior-driven."
                )
        training_frame = _subset(model_frame, train_indices)
        fit = PyMCStructuralBackend(training_frame, spec).fit(
            frailty=frailty,
            draws=spec.inference.cross_validation_draws,
            tune=spec.inference.cross_validation_tune,
            include_prior_predictive=False,
            include_posterior_predictive=False,
            seed_offset=10_000 + fold_index,
        )
        held_risks = _held_out_risk_draws(
            inference_data=fit.inference_data,
            full_frame=model_frame,
            test_indices=test_indices,
            spec=spec,
            frailty=frailty,
            rng=rng,
        )
        predicted[test_indices] = np.mean(held_risks, axis=0)
        for local_index, prompt_index in enumerate(test_indices):
            log_probabilities = np.asarray(
                binom.logpmf(
                    model_frame.failures[prompt_index],
                    model_frame.trials[prompt_index],
                    held_risks[:, local_index],
                ),
                dtype=float,
            )
            log_predictive[prompt_index] = float(
                logsumexp(log_probabilities) - math.log(len(log_probabilities))
            )
        if fit.diagnostics.status != "valid":
            warnings.append(
                f"Fold {fold_index + 1} sampler diagnostics were {fit.diagnostics.status}."
            )
        statuses.append(fit.diagnostics.status)

    observed_rates = model_frame.failures / model_frame.trials
    weights = model_frame.weights
    return CrossValidationSummary(
        enabled=True,
        status=(
            "invalid"
            if "invalid" in statuses
            else "degraded"
            if "degraded" in statuses
            else "valid"
        ),
        folds=folds,
        prompt_count=prompt_count,
        mean_log_predictive_density=float(np.mean(log_predictive)),
        mean_brier_score=float(np.dot(weights, (predicted - observed_rates) ** 2)),
        weighted_calibration_error=float(
            abs(np.dot(weights, predicted) - np.dot(weights, observed_rates))
        ),
        warnings=warnings,
    )
