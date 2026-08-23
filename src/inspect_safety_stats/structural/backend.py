from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from inspect_safety_stats.schema.structural_reports import SamplerDiagnostics


class StructuralBackendUnavailable(RuntimeError):
    pass


@dataclass
class BackendFit:
    inference_data: Any
    risk_draws: np.ndarray
    prompt_scale_draws: np.ndarray
    hierarchy_effect_draws: dict[str, np.ndarray]
    prior_risk_draws: np.ndarray | None
    prior_predictive_counts: np.ndarray | None
    posterior_predictive_counts: np.ndarray | None
    diagnostics: SamplerDiagnostics


class StructuralBackend(Protocol):
    def fit(
        self,
        *,
        frailty: Literal["student_t", "normal"],
        draws: int | None = None,
        tune: int | None = None,
        include_prior_predictive: bool = True,
        include_posterior_predictive: bool = True,
        seed_offset: int = 0,
    ) -> BackendFit: ...
