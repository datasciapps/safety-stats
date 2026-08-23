from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from inspect_safety_stats.schema.structural_reports import (
    ConcentrationSummary,
    HierarchyRiskSummary,
    PosteriorSummary,
    PredictiveChecks,
    PredictiveStatistic,
    PromptRiskSummary,
    VulnerableMassSummary,
)


def _draw_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty draws-by-prompts matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def posterior_summary(values: np.ndarray, *, credible_mass: float) -> PosteriorSummary:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("posterior summary requires finite draws")
    if not 0 < credible_mass < 1:
        raise ValueError("credible_mass must be in (0, 1)")
    tail = (1 - credible_mass) / 2
    return PosteriorSummary(
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        sd=float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        lower=float(np.quantile(array, tail)),
        upper=float(np.quantile(array, 1 - tail)),
        credible_mass=credible_mass,
    )


def validate_weights(weights: np.ndarray, prompt_count: int) -> np.ndarray:
    array = np.asarray(weights, dtype=float)
    if array.shape != (prompt_count,):
        raise ValueError("weights must have one value per prompt")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not np.isclose(np.sum(array), 1.0):
        raise ValueError("weights must sum to one")
    return array


def weighted_tail_risk_draws(
    risks: np.ndarray,
    weights: np.ndarray,
    *,
    prompt_mass: float,
) -> np.ndarray:
    matrix = _draw_matrix(risks, name="risks")
    resolved_weights = validate_weights(weights, matrix.shape[1])
    if not 0 < prompt_mass <= 1:
        raise ValueError("prompt_mass must be in (0, 1]")
    result = np.zeros(matrix.shape[0], dtype=float)
    for draw_index, draw in enumerate(matrix):
        order = np.argsort(-draw, kind="stable")
        remaining = prompt_mass
        total = 0.0
        for index in order:
            selected = min(float(resolved_weights[index]), remaining)
            total += selected * float(draw[index])
            remaining -= selected
            if remaining <= 1e-15:
                break
        result[draw_index] = total
    return result


def prompt_risk_summaries(
    risks: np.ndarray,
    prompt_ids: Iterable[str],
    weights: np.ndarray,
    *,
    thresholds: list[float],
    credible_mass: float,
) -> list[PromptRiskSummary]:
    matrix = _draw_matrix(risks, name="risks")
    ids = tuple(prompt_ids)
    if len(ids) != matrix.shape[1]:
        raise ValueError("prompt_ids must align with posterior risk columns")
    resolved_weights = validate_weights(weights, matrix.shape[1])
    return [
        PromptRiskSummary(
            prompt_id=prompt_id,
            weight=float(resolved_weights[index]),
            risk=posterior_summary(matrix[:, index], credible_mass=credible_mass),
            exceedance_probability={
                format(threshold, ".12g"): float(np.mean(matrix[:, index] > threshold))
                for threshold in thresholds
            },
        )
        for index, prompt_id in enumerate(ids)
    ]


def hierarchy_marginal_risk_summaries(
    risks: np.ndarray,
    weights: np.ndarray,
    *,
    hierarchy: tuple[str, ...],
    hierarchy_values: dict[str, Iterable[str]],
    credible_mass: float,
) -> list[HierarchyRiskSummary]:
    """Summarize design-weighted posterior risk within every shared hierarchy node."""
    matrix = _draw_matrix(risks, name="risks")
    resolved_weights = validate_weights(weights, matrix.shape[1])
    values = {level: np.asarray(tuple(hierarchy_values[level]), dtype=str) for level in hierarchy}
    if any(item.shape != (matrix.shape[1],) for item in values.values()):
        raise ValueError("hierarchy values must align with posterior risk columns")

    summaries: list[HierarchyRiskSummary] = []
    for level_index, level in enumerate(hierarchy):
        path_levels = hierarchy[: level_index + 1]
        paths = sorted(
            {
                tuple(values[path_level][prompt_index] for path_level in path_levels)
                for prompt_index in range(matrix.shape[1])
            }
        )
        for path_values in paths:
            mask = np.ones(matrix.shape[1], dtype=bool)
            for path_level, path_value in zip(path_levels, path_values, strict=True):
                mask &= values[path_level] == path_value
            node_weight = float(np.sum(resolved_weights[mask]))
            if node_weight <= 0:
                continue
            node_draws = matrix[:, mask] @ (resolved_weights[mask] / node_weight)
            summaries.append(
                HierarchyRiskSummary(
                    level=level,
                    category=path_values[-1],
                    path=dict(zip(path_levels, path_values, strict=True)),
                    design_weight=node_weight,
                    prompt_count=int(np.sum(mask)),
                    risk=posterior_summary(node_draws, credible_mass=credible_mass),
                )
            )
    return summaries


def vulnerable_mass_summaries(
    risks: np.ndarray,
    weights: np.ndarray,
    *,
    thresholds: list[float],
    credible_mass: float,
) -> list[VulnerableMassSummary]:
    matrix = _draw_matrix(risks, name="risks")
    resolved_weights = validate_weights(weights, matrix.shape[1])
    summaries: list[VulnerableMassSummary] = []
    for threshold in thresholds:
        mass_draws = np.sum((matrix > threshold) * resolved_weights[None, :], axis=1)
        summaries.append(
            VulnerableMassSummary(
                threshold=threshold,
                mass=posterior_summary(mass_draws, credible_mass=credible_mass),
                expected_exceedance_mass=float(
                    np.dot(resolved_weights, np.mean(matrix > threshold, axis=0))
                ),
            )
        )
    return summaries


def concentration_summaries(
    risks: np.ndarray,
    weights: np.ndarray,
    *,
    prompt_masses: list[float],
    credible_mass: float,
    zero_risk_tolerance: float,
) -> list[ConcentrationSummary]:
    matrix = _draw_matrix(risks, name="risks")
    resolved_weights = validate_weights(weights, matrix.shape[1])
    total_risk = matrix @ resolved_weights
    near_zero = total_risk <= zero_risk_tolerance
    summaries: list[ConcentrationSummary] = []
    for prompt_mass in prompt_masses:
        tail = weighted_tail_risk_draws(matrix, resolved_weights, prompt_mass=prompt_mass)
        defined = ~near_zero
        share_summary = (
            posterior_summary(tail[defined] / total_risk[defined], credible_mass=credible_mass)
            if bool(np.any(defined))
            else None
        )
        summaries.append(
            ConcentrationSummary(
                prompt_mass=prompt_mass,
                risk_share=share_summary,
                absolute_tail_risk=posterior_summary(tail, credible_mass=credible_mass),
                near_zero_total_risk_probability=float(np.mean(near_zero)),
            )
        )
    return summaries


def weighted_mean_risk_draws(risks: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = _draw_matrix(risks, name="risks")
    return np.asarray(matrix @ validate_weights(weights, matrix.shape[1]), dtype=float)


def predictive_checks(
    replicated_counts: np.ndarray,
    observed_counts: np.ndarray,
    trials: np.ndarray,
    weights: np.ndarray,
    *,
    credible_mass: float,
) -> PredictiveChecks:
    replicated = _draw_matrix(replicated_counts, name="replicated_counts")
    observed = np.asarray(observed_counts, dtype=float)
    n = np.asarray(trials, dtype=float)
    if observed.shape != (replicated.shape[1],) or n.shape != observed.shape:
        raise ValueError("observed counts and trials must align with replicated prompts")
    if np.any(n <= 0):
        raise ValueError("trials must be positive")
    resolved_weights = validate_weights(weights, replicated.shape[1])

    observed_rates = observed / n
    replicated_rates = replicated / n[None, :]
    statistic_values: list[tuple[str, float, np.ndarray]] = [
        ("total_unsafe_count", float(np.sum(observed)), np.sum(replicated, axis=1)),
        (
            "zero_failure_prompt_count",
            float(np.sum(observed == 0)),
            np.sum(replicated == 0, axis=1),
        ),
        (
            "one_failure_prompt_count",
            float(np.sum(observed == 1)),
            np.sum(replicated == 1, axis=1),
        ),
        (
            "maximum_prompt_failure_proportion",
            float(np.max(observed_rates)),
            np.max(replicated_rates, axis=1),
        ),
        (
            "weighted_mean_failure_proportion",
            float(np.dot(resolved_weights, observed_rates)),
            replicated_rates @ resolved_weights,
        ),
    ]
    return PredictiveChecks(
        statistics=[
            PredictiveStatistic(
                name=name,
                observed=observed_value,
                replicated=posterior_summary(draws, credible_mass=credible_mass),
                posterior_predictive_p_lower=float(np.mean(draws <= observed_value)),
                posterior_predictive_p_upper=float(np.mean(draws >= observed_value)),
            )
            for name, observed_value, draws in statistic_values
        ]
    )
