from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# Report schema 2.0 named these fields after Inspect. A plan may still name them so in
# block_by, pair_key and cluster_key.
LEGACY_FIELD_NAMES = {
    "source_log": "source",
    "source_log_sha256": "source_sha256",
    "eval_id": "run_key",
    "sample_id": "rollout_id",
    "inspect_id": "item_id",
    "epoch": "repeat",
}


def observation_field(name: str) -> str:
    """The observation field a plan names, by its current or its report schema 2.0 name."""

    return LEGACY_FIELD_NAMES.get(name, name)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    # The file the rollout was read from, and its SHA-256 when it is a local file.
    source: str
    source_sha256: str | None = None
    run_key: str
    run_id: str | None = None
    rollout_id: str
    item_id: str
    repeat: int
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
