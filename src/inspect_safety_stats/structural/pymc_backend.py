from __future__ import annotations

import inspect
import math
from importlib import import_module
from importlib.metadata import version
from typing import Any, Literal

import numpy as np

from inspect_safety_stats.config.models import StructuralModelSpec
from inspect_safety_stats.schema.structural_reports import SamplerDiagnostics
from inspect_safety_stats.structural.backend import BackendFit, StructuralBackendUnavailable
from inspect_safety_stats.structural.data import ModelFrame


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        az = import_module("arviz")
        pm = import_module("pymc")
        pt = import_module("pytensor.tensor")
    except ImportError as exc:
        raise StructuralBackendUnavailable(
            "hierarchical modelling dependencies are unavailable; install "
            "inspect-safety-stats[hierarchical]"
        ) from exc
    return pm, pt, az


def backend_versions() -> dict[str, str]:
    pm, pt, az = _dependencies()
    return {
        "pymc": str(pm.__version__),
        "pytensor": version("pytensor"),
        "arviz": str(az.__version__),
    }


def _matrix(variable: Any) -> np.ndarray:
    values = np.asarray(variable, dtype=float)
    if values.ndim < 2:
        raise ValueError("posterior variable is missing chain and draw dimensions")
    return values.reshape((-1, *values.shape[2:]))


def _dataset_extreme(dataset: Any, reducer: str) -> float | None:
    values: list[np.ndarray] = []
    for variable in dataset.data_vars.values():
        array = np.asarray(variable, dtype=float)
        values.append(array[np.isfinite(array)])
    populated = [item for item in values if item.size]
    if not populated:
        return None
    combined = np.concatenate(populated)
    return float(np.max(combined) if reducer == "max" else np.min(combined))


def _nested_centering_matrix(frame: ModelFrame, hierarchy_index: int) -> np.ndarray:
    column = frame.hierarchy[hierarchy_index]
    level_count = len(frame.category_levels[column])
    matrix = np.eye(level_count, dtype=float)
    if hierarchy_index == 0:
        return np.asarray(
            matrix - np.full((level_count, level_count), 1 / level_count), dtype=float
        )
    parent_column = frame.hierarchy[hierarchy_index - 1]
    child_codes = frame.category_codes[column]
    parent_codes = frame.category_codes[parent_column]
    child_to_parent: dict[int, int] = {}
    for child, parent in zip(child_codes, parent_codes, strict=True):
        child_to_parent[int(child)] = int(parent)
    for parent in sorted(set(child_to_parent.values())):
        children = sorted(
            child for child, resolved_parent in child_to_parent.items() if resolved_parent == parent
        )
        for child in children:
            matrix[child, children] -= 1 / len(children)
    return matrix


def _prompt_centering_matrix(frame: ModelFrame) -> np.ndarray:
    prompt_count = len(frame.frame)
    matrix = np.eye(prompt_count, dtype=float)
    parent_column = frame.hierarchy[-1]
    parent_codes = frame.category_codes[parent_column]
    for parent in sorted(set(parent_codes.tolist())):
        prompts = np.flatnonzero(parent_codes == parent)
        for prompt in prompts:
            matrix[prompt, prompts] -= 1 / len(prompts)
    return matrix


def _diagnostics(idata: Any, *, parameter_names: list[str], az: Any) -> SamplerDiagnostics:
    posterior = idata.posterior
    chains = int(posterior.sizes.get("chain", 0))
    draws = int(posterior.sizes.get("draw", 0))
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    max_rhat = _dataset_extreme(az.rhat(idata, var_names=parameter_names), "max")
    min_bulk = _dataset_extreme(az.ess(idata, var_names=parameter_names, method="bulk"), "min")
    min_tail = _dataset_extreme(az.ess(idata, var_names=parameter_names, method="tail"), "min")
    warnings: list[str] = []
    status: Literal["valid", "degraded", "invalid"] = "valid"
    if divergences:
        status = "invalid"
        warnings.append(f"{divergences} divergent transitions were observed.")
    if max_rhat is None or max_rhat >= 1.01:
        status = "invalid"
        warnings.append(f"Maximum rank-normalized R-hat is {max_rhat!r}, not below 1.01.")
    required_ess = min(400.0, max(100.0, chains * draws / 5))
    if min_bulk is None or min_tail is None or min(min_bulk, min_tail) < required_ess:
        if status == "valid":
            status = "degraded"
        warnings.append(
            "Minimum bulk/tail effective sample size is below the configured diagnostic floor "
            f"of {required_ess:g}."
        )
    return SamplerDiagnostics(
        status=status,
        chains=chains,
        draws_per_chain=draws,
        divergences=divergences,
        max_rhat=max_rhat,
        min_bulk_ess=min_bulk,
        min_tail_ess=min_tail,
        warnings=warnings,
    )


class PyMCStructuralBackend:
    def __init__(self, model_frame: ModelFrame, spec: StructuralModelSpec):
        self.model_frame = model_frame
        self.spec = spec

    def _build_model(self, frailty: Literal["student_t", "normal"]) -> tuple[Any, list[str]]:
        pm, pt, _ = _dependencies()
        frame = self.model_frame
        coords: dict[str, Any] = {"prompt": list(frame.prompt_ids)}
        for index, column in enumerate(frame.hierarchy):
            coords[f"hierarchy_{index}"] = list(frame.category_levels[column])
        parameter_names = ["mu", "sigma_prompt"]
        with pm.Model(coords=coords) as model:
            reference = self.spec.priors.reference_probability
            reference_eta = math.log(-math.log1p(-reference))
            mu = pm.Normal(
                "mu",
                mu=reference_eta,
                sigma=self.spec.priors.intercept_sd,
            )
            eta = pt.ones(len(frame.frame)) * mu
            for index, column in enumerate(frame.hierarchy):
                if len(frame.category_levels[column]) < 2:
                    continue
                coordinate = f"hierarchy_{index}"
                sigma_name = f"sigma_hierarchy_{index}"
                raw_name = f"raw_hierarchy_{index}"
                effect_name = f"effect_hierarchy_{index}"
                sigma = pm.HalfNormal(
                    sigma_name,
                    sigma=self.spec.priors.hierarchy_scale,
                )
                raw = pm.Normal(raw_name, mu=0, sigma=1, dims=coordinate)
                centered = pt.dot(_nested_centering_matrix(frame, index), raw)
                effect = pm.Deterministic(effect_name, sigma * centered, dims=coordinate)
                eta = eta + effect[frame.category_codes[column]]
                parameter_names.extend([sigma_name, effect_name])
            sigma_prompt = pm.HalfNormal("sigma_prompt", sigma=self.spec.priors.prompt_scale)
            if frailty == "student_t":
                raw_prompt = pm.StudentT(
                    "raw_prompt",
                    nu=self.spec.prompt_frailty.degrees_of_freedom,
                    mu=0,
                    sigma=1,
                    dims="prompt",
                )
            else:
                raw_prompt = pm.Normal("raw_prompt", mu=0, sigma=1, dims="prompt")
            centered_prompt = pt.dot(_prompt_centering_matrix(frame), raw_prompt)
            prompt_effect = pm.Deterministic(
                "prompt_effect", sigma_prompt * centered_prompt, dims="prompt"
            )
            eta = pm.Deterministic("eta", eta + prompt_effect, dims="prompt")
            risk = pm.Deterministic(
                "prompt_risk",
                1 - pt.exp(-pt.exp(eta)),
                dims="prompt",
            )
            # Prompt-level posterior summaries are first-class outputs of the
            # structural analysis. Include them in the convergence gate rather
            # than validating only the population-level hyperparameters.
            parameter_names.append("prompt_risk")
            pm.Binomial(
                "unsafe_count",
                n=frame.trials,
                p=risk,
                observed=frame.failures,
                dims="prompt",
            )
        return model, parameter_names

    def fit(
        self,
        *,
        frailty: Literal["student_t", "normal"],
        draws: int | None = None,
        tune: int | None = None,
        include_prior_predictive: bool = True,
        include_posterior_predictive: bool = True,
        seed_offset: int = 0,
    ) -> BackendFit:
        pm, _, az = _dependencies()
        model, parameter_names = self._build_model(frailty)
        inference = self.spec.inference
        seed = inference.seed + seed_offset
        prior = None
        with model:
            if include_prior_predictive:
                prior_draw_argument = (
                    "draws"
                    if "draws" in inspect.signature(pm.sample_prior_predictive).parameters
                    else "samples"
                )
                prior = pm.sample_prior_predictive(
                    random_seed=seed,
                    **{prior_draw_argument: inference.prior_predictive_draws},
                )
            idata = pm.sample(
                draws=draws or inference.draws,
                tune=tune or inference.tune,
                chains=inference.chains,
                cores=min(inference.cores, inference.chains),
                target_accept=inference.target_accept,
                random_seed=seed,
                return_inferencedata=True,
                progressbar=False,
            )
            posterior_predictive = (
                pm.sample_posterior_predictive(
                    idata,
                    var_names=["unsafe_count"],
                    random_seed=seed + 1,
                    return_inferencedata=True,
                    progressbar=False,
                )
                if include_posterior_predictive
                else None
            )
        prior_risk = _matrix(prior.prior["prompt_risk"]) if prior is not None else None
        prior_counts = (
            _matrix(prior.prior_predictive["unsafe_count"]) if prior is not None else None
        )
        posterior_counts = (
            _matrix(posterior_predictive.posterior_predictive["unsafe_count"])
            if posterior_predictive is not None
            else None
        )
        return BackendFit(
            inference_data=idata,
            risk_draws=_matrix(idata.posterior["prompt_risk"]),
            prompt_scale_draws=_matrix(idata.posterior["sigma_prompt"]).reshape(-1),
            hierarchy_effect_draws={
                column: _matrix(idata.posterior[f"effect_hierarchy_{index}"])
                for index, column in enumerate(self.model_frame.hierarchy)
                if len(self.model_frame.category_levels[column]) >= 2
            },
            prior_risk_draws=prior_risk,
            prior_predictive_counts=prior_counts,
            posterior_predictive_counts=posterior_counts,
            diagnostics=_diagnostics(idata, parameter_names=parameter_names, az=az),
        )
