from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    source_log: str
    source_log_sha256: str | None = None
    eval_id: str
    run_id: str | None = None
    sample_id: str
    inspect_id: str
    epoch: int
    prompt_id: str
    question_id: str
    task_id: str
    cohort_id: str
    score_name: str
    score_raw: Any = None
    failure: bool | None = None
    harm: float | None = None
    outcome_status: Literal[
        "classified_failure",
        "classified_safe",
        "target_policy_blocked_safe",
        "unclassified",
    ] = "unclassified"
    error: Any = None
    error_stage: Literal["target_generation", "post_generation", "unknown"] | None = None
    content_delivered: bool | None = None
    provider_error_code: str | None = None
    retries: int = 0
    model: str
    configuration_fingerprint: str
    collection_order: int

    @model_validator(mode="after")
    def harm_follows_binary_failure(self) -> Observation:
        """Binary outcomes carry harm 0 or 1; graded outcomes set harm explicitly.

        ``failure`` is True whenever there is any harm, so zero-tolerance policies and
        judge strata keep working on graded scores; ``harm`` is the magnitude the
        interval engines bound.
        """

        if self.harm is None and self.failure is not None:
            object.__setattr__(self, "harm", 1.0 if self.failure else 0.0)
        if self.harm is not None:
            if not 0.0 <= self.harm <= 1.0:
                raise ValueError("harm must lie in [0, 1]")
            if self.failure is None:
                raise ValueError("an observation with a harm value must be classified")
        return self
