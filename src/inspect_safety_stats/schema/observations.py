from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

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
    # Whether the answer stopped at the generation token limit without a score or a sample
    # error; None when the plan's score.token_limit is "score", which does not look.
    token_limit_unscored: bool | None = None

    @model_serializer(mode="wrap")
    def leave_out_absent_token_limit(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # Left out unless the plan looks, so observations.csv keeps the bytes it had before.
        data: dict[str, Any] = handler(self)
        if self.token_limit_unscored is None:
            data.pop("token_limit_unscored", None)
        return data
