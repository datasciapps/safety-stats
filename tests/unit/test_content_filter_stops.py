"""Content-filter stops: how Inspect records them, and how a plan declares their outcome."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from safety_stats.config.models import InputsSpec
from safety_stats.ingest.inspect_logs import _sample_stop, _stop_of
from safety_stats.ingest.records import StopContext


def _output(
    content: Any = "",
    stop_reason: str = "stop",
    category: str | None = None,
    error: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ModelOutput,
        StopDetails,
    )

    details = StopDetails(type="refusal", category=category) if category else None
    choice = ChatCompletionChoice(
        message=ChatMessageAssistant(content=content, tool_calls=tool_calls),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_details=details,
    )
    return ModelOutput(model="mockllm", choices=[choice], error=error)


def _tool_call() -> Any:
    from inspect_ai.tool import ToolCall

    return ToolCall(id="call-1", function="run", arguments={"command": "x"})


def _reasoning(text: str, *, redacted: bool = False, signature: str | None = None) -> Any:
    from inspect_ai.model import ContentReasoning

    return ContentReasoning(reasoning=text, redacted=redacted, signature=signature)


def _generation(output: Any, error: str | None = None) -> Any:
    from inspect_ai.event import ModelEvent
    from inspect_ai.model import GenerateConfig

    return ModelEvent(
        model="mockllm",
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=output,
        error=error,
    )


def _sample(output: Any, events: list[Any]) -> Any:
    from inspect_ai.log import EvalSample

    return EvalSample(id="p1", epoch=1, input="q", target="", output=output, events=events)


def _decline() -> Any:
    return _output(stop_reason="content_filter", category="cyber")


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
    assert stop.delivered_in == "stopped_output"


@pytest.mark.parametrize(
    "fields",
    [
        {"tool_calls": [_tool_call()]},
        {"content": [_reasoning("first, the precursor")]},
    ],
    ids=["tool call", "reasoning"],
)
def test_a_tool_call_or_readable_reasoning_before_a_decline_is_delivered(
    fields: dict[str, Any],
) -> None:
    # The stopped output has no text, but it called a tool or showed its reasoning.
    stop = _stop_of(_output(stop_reason="content_filter", category="cyber", **fields))
    assert stop is not None and (stop.content_delivered, stop.delivered_in) == (
        True,
        "stopped_output",
    )


@pytest.mark.parametrize(
    "part",
    [_reasoning("ENCRYPTED", redacted=True), _reasoning("", signature="signature")],
    ids=["redacted", "signature only"],
)
def test_reasoning_no_one_can_read_is_not_delivered(part: Any) -> None:
    stop = _stop_of(_output([part], stop_reason="content_filter", category="cyber"))
    assert stop is not None and stop.content_delivered is False


@pytest.mark.parametrize("output", [None, _output("answer"), _output(stop_reason="max_tokens")])
def test_an_output_no_content_filter_stopped_is_not_a_stop(output: Any) -> None:
    assert _stop_of(output) is None


@pytest.mark.parametrize(
    "earlier",
    [
        _output("HARMFUL first turn"),
        _output(tool_calls=[_tool_call()], stop_reason="tool_calls"),
        _output([_reasoning("worked out the steps")]),
    ],
    ids=["text", "tool call", "reasoning"],
)
def test_content_an_earlier_turn_delivered_is_not_undone_by_a_later_decline(
    earlier: Any,
) -> None:
    stopped = _decline()
    stop = _sample_stop(_sample(stopped, [_generation(earlier), _generation(stopped)]))
    assert stop is not None
    assert (stop.content_delivered, stop.delivered_in) == (True, "earlier_generation")
    assert (stop.block_type, stop.block_category) == ("refusal", "cyber")


@pytest.mark.parametrize(
    "earlier",
    [
        # A failed call, and a decline that Inspect reported as an API error with a stand-in.
        _generation(_output(), error="connection reset"),
        _generation(
            _output("Sorry, I cannot help.", stop_reason="content_filter", error="blocked")
        ),
        # An earlier decline that delivered nothing either.
        _generation(_decline()),
    ],
    ids=["failed call", "api error decline", "earlier decline"],
)
def test_earlier_generations_without_content_leave_a_decline_undelivered(earlier: Any) -> None:
    stopped = _decline()
    stop = _sample_stop(_sample(stopped, [earlier, _generation(stopped)]))
    assert stop is not None and (stop.content_delivered, stop.delivered_in) == (False, None)


def test_the_scorers_model_calls_after_the_decline_are_not_the_targets_content() -> None:
    stopped = _decline()
    grader = _generation(_output("GRADE: the answer is safe"))
    stop = _sample_stop(_sample(stopped, [_generation(stopped), grader]))
    assert stop is not None and stop.content_delivered is False


def test_content_between_two_records_of_the_stopped_message_counts() -> None:
    # A cached output can repeat a message id: the stopped generation is the last record of
    # it, so the content delivered after the first record came before the stop.
    stopped = _decline()
    events = [_generation(stopped), _generation(_output("HARMFUL turn")), _generation(stopped)]
    stop = _sample_stop(_sample(stopped, events))
    assert stop is not None
    assert (stop.content_delivered, stop.delivered_in) == (True, "earlier_generation")


@pytest.mark.parametrize("events", [[], [_generation(_output("unrelated"))]])
def test_a_decline_the_transcript_does_not_record_cannot_rule_content_out(
    events: list[Any],
) -> None:
    # Without the stopped generation in the transcript, nothing says what came before it.
    stop = _sample_stop(_sample(_decline(), events))
    assert stop is not None
    assert (stop.content_delivered, stop.delivered_in) == (True, "unverifiable")


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


@pytest.mark.parametrize("stops", ["safe", "failure"])
def test_a_judge_cannot_classify_content_filter_stops(stops: str) -> None:
    # Every judge's run takes the field from inputs.score, so a judge's own value is refused.
    mapping = {"failure_values": ["FAIL"], "safe_values": ["PASS"]}
    judge = {"id": "j", "score": {"name": "j", "mapping": mapping, "content_filter_stops": stops}}
    inputs = {**_inputs(errors="conservative_bounds"), "judges": [judge]}
    with pytest.raises(ValidationError, match="content_filter_stops applies to inputs.score only"):
        InputsSpec.model_validate(inputs)
    judge["score"]["content_filter_stops"] = "score"  # type: ignore[index]
    InputsSpec.model_validate(inputs)
