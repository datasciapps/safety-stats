from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


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
