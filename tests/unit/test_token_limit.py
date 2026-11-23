"""Answers at the generation token limit: how Inspect records them, and how a plan counts them."""

from __future__ import annotations

from typing import Any

import pytest

from safety_stats.config.models import InputsSpec
from safety_stats.ingest.inspect_logs import _stop_of, _token_limit_stop_of
from safety_stats.ingest.records import StopContext


def _output(content: str = "", stop_reason: str = "stop") -> Any:
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

    choice = ChatCompletionChoice(
        message=ChatMessageAssistant(content=content),
        stop_reason=stop_reason,  # type: ignore[arg-type]
    )
    return ModelOutput(model="mockllm", choices=[choice])


def test_an_answer_that_ran_out_of_tokens_is_a_token_limit_stop() -> None:
    assert _token_limit_stop_of(_output("a long chain of", stop_reason="max_tokens")) == (
        StopContext(
            content_delivered=True,
            block_type=None,
            block_category=None,
            block_error=False,
            stop_reason="max_tokens",
        )
    )


@pytest.mark.parametrize(
    "output", [None, _output("answer"), _output(stop_reason="content_filter")]
)
def test_an_output_that_did_not_run_out_of_tokens_is_not_a_token_limit_stop(
    output: Any,
) -> None:
    assert _token_limit_stop_of(output) is None


def test_a_token_limit_stop_is_not_a_content_filter_stop() -> None:
    assert _stop_of(_output(stop_reason="max_tokens")) is None


def _inputs(**score: Any) -> dict[str, Any]:
    mapping = {"failure_values": ["FAIL"], "safe_values": ["PASS"]}
    return {"design_manifest": "design.csv", "score": {"name": "s", "mapping": mapping, **score}}


def test_the_default_is_left_out_of_the_plan_so_its_hash_is_unchanged() -> None:
    absent = InputsSpec.model_validate(_inputs())
    explicit = InputsSpec.model_validate(_inputs(token_limit="score"))
    assert "token_limit" not in absent.model_dump(mode="json")["score"]
    assert absent.model_dump(mode="json") == explicit.model_dump(mode="json")
    chosen = InputsSpec.model_validate(_inputs(token_limit="failure"))
    assert chosen.model_dump(mode="json")["score"]["token_limit"] == "failure"


def test_an_unknown_token_limit_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="token_limit"):
        InputsSpec.model_validate(_inputs(token_limit="count_as_failure"))
