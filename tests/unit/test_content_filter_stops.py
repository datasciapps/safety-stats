"""Content-filter stops: how Inspect records them, and how a plan declares their outcome."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from inspect_safety_stats.config.models import InputsSpec
from inspect_safety_stats.ingest.inspect_logs import _stop_of
from inspect_safety_stats.ingest.records import StopContext


def _output(
    content: str = "",
    stop_reason: str = "stop",
    category: str | None = None,
    error: str | None = None,
) -> Any:
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ModelOutput,
        StopDetails,
    )

    details = StopDetails(type="refusal", category=category) if category else None
    choice = ChatCompletionChoice(
        message=ChatMessageAssistant(content=content),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_details=details,
    )
    return ModelOutput(model="mockllm", choices=[choice], error=error)


def test_a_classifier_decline_without_content_is_a_stop_with_its_category() -> None:
    assert _stop_of(_output(stop_reason="content_filter", category="bio")) == StopContext(
        content_delivered=False, block_type="refusal", block_category="bio", block_error=False
    )


def test_a_decline_reported_as_an_api_error_delivered_nothing() -> None:
    # Inspect puts a stand-in apology in place of the content of a declined request.
    output = _output(
        "Sorry, but I am unable to help with that request.",
        stop_reason="content_filter",
        error="output blocked by content filtering policy",
    )
    assert _stop_of(output) == StopContext(
        content_delivered=False, block_type=None, block_category=None, block_error=True
    )


def test_content_before_a_decline_mid_answer_is_delivered() -> None:
    stop = _stop_of(_output("partial answer", stop_reason="content_filter", category="cyber"))
    assert stop is not None and stop.content_delivered is True


@pytest.mark.parametrize("output", [None, _output("answer"), _output(stop_reason="max_tokens")])
def test_an_output_no_content_filter_stopped_is_not_a_stop(output: Any) -> None:
    assert _stop_of(output) is None


def _inputs(**score: Any) -> dict[str, Any]:
    mapping = {"failure_values": ["FAIL"], "safe_values": ["PASS"]}
    return {"design_manifest": "design.csv", "score": {"name": "s", "mapping": mapping, **score}}


def test_the_default_is_left_out_of_the_plan_so_its_hash_is_unchanged() -> None:
    absent = InputsSpec.model_validate(_inputs())
    explicit = InputsSpec.model_validate(_inputs(content_filter_stops="score"))
    assert "content_filter_stops" not in absent.model_dump(mode="json")["score"]
    assert absent.model_dump(mode="json") == explicit.model_dump(mode="json")
    chosen = InputsSpec.model_validate(_inputs(content_filter_stops="safe"))
    assert chosen.model_dump(mode="json")["score"]["content_filter_stops"] == "safe"


def test_content_filter_stops_need_the_end_to_end_estimand() -> None:
    with pytest.raises(ValidationError, match="end_to_end_system"):
        InputsSpec.model_validate(
            {**_inputs(content_filter_stops="safe"), "estimand": "underlying_model"}
        )


def test_a_block_cannot_be_a_refusal_while_blocked_errors_count_as_safe() -> None:
    with pytest.raises(ValidationError, match="choose another score.errors"):
        InputsSpec.model_validate(_inputs(content_filter_stops="failure"))
    InputsSpec.model_validate(_inputs(content_filter_stops="failure", errors="count_as_failure"))
