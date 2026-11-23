"""Conversation-shaped rewrites: kernels returning whole chat conversations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.dataset import json_dataset
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

import safety_stats.generative.runner as runner_module
from safety_stats.generative.execution import (
    execute_rewrites,
    load_seed_texts,
    read_prompt_store,
    write_prompt_store,
)
from safety_stats.generative.inspect_bridge import materialize_dataset
from safety_stats.generative.kernels import (
    ChatTurn,
    DeterministicFakeKernel,
    RewriteOutcome,
    conversation_problem,
    decode_conversation,
    encode_conversation,
    resolve_kernel_fingerprint,
)
from safety_stats.generative.ledger import prompt_sha256, read_rewrites
from safety_stats.generative.runner import materialize_run
from safety_stats.generative.sampling import draw_selections, plan_design
from safety_stats.generative.schema import load_generative_plan
from safety_stats.generative.taxonomy import compile_taxonomy
from tests.generative_fixtures import (
    SEEDS_WITH_TEXT,
    ConversationKernel,
    base_plan,
    materialize,
)

USER = ChatTurn("user", "first question")
ASSISTANT = ChatTurn("assistant", "an answer the generator wrote")
SYSTEM = ChatTurn("system", "You are a careful assistant.")


def conversation_plan(tmp_path: Path, total: int = 40, **rewrite: Any) -> Path:
    plan = base_plan()
    plan["sampling"] = {"method": "direct", "total_rewrites": total}
    plan["rewrite"]["output"] = "conversation"
    plan["rewrite"].update(rewrite)
    return materialize(tmp_path, plan, SEEDS_WITH_TEXT)


def prepare(path: Path) -> tuple[Any, ...]:
    parsed = load_generative_plan(path)
    taxonomy = compile_taxonomy(parsed, base_path=path.parent)
    selections = draw_selections(taxonomy, plan_design(taxonomy, parsed))
    fingerprint = resolve_kernel_fingerprint(parsed, base_path=path.parent)
    texts = load_seed_texts(parsed, base_path=path.parent)
    return parsed, selections, fingerprint, texts


@pytest.mark.parametrize(
    ("turns", "problem"),
    [
        ([], "no turns"),
        ([SYSTEM], "only a system turn"),
        ([USER, SYSTEM, USER], "may only open"),
        ([ASSISTANT, USER], "open with a user turn"),
        ([USER, USER], "two user turns"),
        ([USER, ASSISTANT], "end on a user turn"),
    ],
)
def test_malformed_conversations_are_named(turns: list[ChatTurn], problem: str) -> None:
    found = conversation_problem(turns)
    assert found is not None and problem in found


@pytest.mark.parametrize("turns", [[USER], [SYSTEM, USER], [SYSTEM, USER, ASSISTANT, USER]])
def test_well_formed_conversations_pass(turns: list[ChatTurn]) -> None:
    assert conversation_problem(turns) is None


def test_chat_turns_reject_unknown_roles_and_empty_text() -> None:
    with pytest.raises(ValueError, match="chat role"):
        ChatTurn("tool", "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must carry text"):
        ChatTurn("user", "   ")


def test_a_valid_outcome_carries_a_prompt_or_a_conversation_not_both() -> None:
    with pytest.raises(ValueError, match="not both"):
        RewriteOutcome(status="valid", prompt="x", messages=(USER,))
    with pytest.raises(ValueError, match="well-formed conversation"):
        RewriteOutcome(status="valid", messages=(USER, ASSISTANT))
    with pytest.raises(ValueError, match="only a valid rewrite"):
        RewriteOutcome(status="invalid", messages=(USER,))
    outcome = RewriteOutcome(status="valid", messages=[USER])  # type: ignore[arg-type]
    assert outcome.messages == (USER,)
    assert outcome.text == encode_conversation([USER])


def test_the_encoding_round_trips_and_rejects_anything_else() -> None:
    turns = (SYSTEM, ChatTurn("user", "Grüße, \"quoted\"\nnew line"), ASSISTANT, USER)
    assert decode_conversation(encode_conversation(turns)) == turns
    for text in ("not json", '{"role": "user"}', '[{"role": "user", "content": 1}]',
                 '[{"role": "user", "content": "x", "name": "y"}]',
                 '[{"role": "assistant", "content": "x"}]'):
        with pytest.raises(ValueError):
            decode_conversation(text)


def test_the_output_kind_moves_the_fingerprint_and_the_default_does_not(
    tmp_path: Path,
) -> None:
    plain = base_plan()
    baseline = resolve_kernel_fingerprint(
        load_generative_plan(materialize(tmp_path, plain)), base_path=tmp_path
    )
    plain["rewrite"]["output"] = "prompt"
    explicit = resolve_kernel_fingerprint(
        load_generative_plan(materialize(tmp_path, plain)), base_path=tmp_path
    )
    plain["rewrite"]["output"] = "conversation"
    moved = resolve_kernel_fingerprint(
        load_generative_plan(materialize(tmp_path, plain)), base_path=tmp_path
    )
    assert explicit.sha256 == baseline.sha256
    assert moved.sha256 != baseline.sha256


def test_conversations_are_accepted_hashed_and_deduplicated_as_text(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(conversation_plan(tmp_path))
    result = execute_rewrites(plan, selections, ConversationKernel(), fingerprint, texts)
    assert len(result.accepted) == len(selections)
    for entry in result.accepted:
        assert entry.messages is not None and entry.messages[-1].role == "user"
        assert entry.prompt == encode_conversation(entry.messages)
        assert entry.prompt_sha256 == prompt_sha256(entry.prompt)
    for record in result.rewrites:
        assert "Hello, I am researching" not in json.dumps(record.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("output", "kernel"),
    [("conversation", ConversationKernel(bare_prompt=True)), ("prompt", ConversationKernel())],
)
def test_a_kernel_of_the_other_output_kind_stops_the_run(
    tmp_path: Path, output: str, kernel: ConversationKernel
) -> None:
    plan, selections, fingerprint, texts = prepare(conversation_plan(tmp_path, output=output))
    with pytest.raises(ValueError, match=f"declares rewrite.output {output!r}"):
        execute_rewrites(plan, selections, kernel, fingerprint, texts)


def test_a_conversation_is_materialized_as_inspect_chat_messages(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(conversation_plan(tmp_path))
    result = execute_rewrites(
        plan, selections, ConversationKernel(system=True), fingerprint, texts
    )
    path = tmp_path / "dataset.jsonl"
    materialize_dataset(selections, result, path)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [message["role"] for message in rows[0]["input"]] == [
        "system", "user", "assistant", "user"
    ]
    sample = json_dataset(str(path))[0]
    assert isinstance(sample.input, list)
    assert [type(message) for message in sample.input] == [
        ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageUser
    ]
    expected = {item.prompt_id: item.messages for item in result.accepted}[str(sample.id)]
    assert expected is not None
    assert sample.input[-1].text == expected[-1].content


def test_single_turn_rewrites_are_still_materialized_as_strings(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(conversation_plan(tmp_path, output="prompt"))
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    path = tmp_path / "dataset.jsonl"
    materialize_dataset(selections, result, path)
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(first["input"], str)


def test_the_prompt_store_round_trips_conversations(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(conversation_plan(tmp_path))
    result = execute_rewrites(plan, selections, ConversationKernel(), fingerprint, texts)
    store = tmp_path / "prompts.csv"
    write_prompt_store(result.accepted, store)
    read = read_prompt_store(store, conversations=True)
    assert {unit: entry.messages for unit, entry in read.items()} == {
        entry.sampling_unit_id: entry.messages for entry in result.accepted
    }


def test_a_resumed_conversation_run_rebuilds_the_same_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = conversation_plan(tmp_path)
    monkeypatch.setattr(runner_module, "resolve_kernel", lambda spec: ConversationKernel())
    first = materialize_run(path, tmp_path / "run")
    dataset = (tmp_path / "run" / "datasets" / "dataset.jsonl").read_text(encoding="utf-8")

    resumed = materialize_run(path, tmp_path / "run", resume=True)
    assert resumed.resumed == first.selections
    assert (tmp_path / "run" / "datasets" / "dataset.jsonl").read_text(
        encoding="utf-8"
    ) == dataset
    assert len(read_rewrites(tmp_path / "run" / "rewrites.csv")) == first.selections


SYSTEM_TEXTS = {"plain": "You are a helpful assistant.", "roleplay": "Stay in character.",
                "nested": ""}


def system_prompt_plan(tmp_path: Path, output: str = "prompt", **spec: Any) -> Path:
    plan = base_plan()
    plan["sampling"] = {"method": "direct", "total_rewrites": 60}
    plan["rewrite"]["output"] = output
    plan["rewrite"]["system_prompts"] = {"level": "wrapper", "texts": dict(SYSTEM_TEXTS), **spec}
    return materialize(tmp_path, plan, SEEDS_WITH_TEXT)


def wrapper_of(node_id: str) -> str:
    return dict(part.split("=", 1) for part in node_id.split("|")[1:])["wrapper"]


@pytest.mark.parametrize("output", ["prompt", "conversation"])
def test_each_arm_value_gets_exactly_its_declared_system_prompt(
    tmp_path: Path, output: str
) -> None:
    plan, selections, fingerprint, texts = prepare(system_prompt_plan(tmp_path, output))
    kernel = DeterministicFakeKernel() if output == "prompt" else ConversationKernel()
    result = execute_rewrites(plan, selections, kernel, fingerprint, texts)
    by_unit = {item.sampling_unit_id: item for item in result.accepted}
    seen = set()
    for selection in selections:
        entry = by_unit[selection.sampling_unit_id]
        assert entry.messages is not None
        assert entry.prompt == encode_conversation(entry.messages)
        wrapper = wrapper_of(selection.node_id)
        seen.add(wrapper)
        if SYSTEM_TEXTS[wrapper]:
            assert entry.messages[0] == ChatTurn("system", SYSTEM_TEXTS[wrapper])
            assert [turn.role for turn in entry.messages[1:]].count("system") == 0
        else:
            assert entry.messages[0].role == "user"
    assert seen == set(SYSTEM_TEXTS)


def test_the_kernel_is_told_the_system_prompt(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(system_prompt_plan(tmp_path))
    told: dict[str, str | None] = {}

    class Recording(DeterministicFakeKernel):
        def rewrite(self, request):  # type: ignore[no-untyped-def]
            told[request.sampling_unit_id] = request.system_prompt
            return super().rewrite(request)

    execute_rewrites(plan, selections, Recording(), fingerprint, texts)
    for selection in selections:
        expected = SYSTEM_TEXTS[wrapper_of(selection.node_id)] or None
        assert told[selection.sampling_unit_id] == expected


def test_a_kernel_writing_its_own_system_turn_stops_the_run(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(
        system_prompt_plan(tmp_path, "conversation")
    )
    with pytest.raises(ValueError, match="attaches the system prompt itself"):
        execute_rewrites(plan, selections, ConversationKernel(system=True), fingerprint, texts)


def test_a_kernel_system_turn_stops_the_run_on_an_empty_text_value_too(
    tmp_path: Path,
) -> None:
    # An empty text declares "no system prompt" for its value. A system turn the kernel
    # wrote there would put a prompt of the kernel's choosing under that arm value.
    plan, selections, fingerprint, texts = prepare(
        system_prompt_plan(tmp_path, "conversation")
    )
    empty = [s for s in selections if not SYSTEM_TEXTS[wrapper_of(s.node_id)]]
    assert empty and len(empty) < len(selections)

    class SystemTurnWhereNoneIsDeclared(ConversationKernel):
        # Writes a system turn only where the request carries none, which is exactly on
        # the empty-text value's paths, so a check on the other paths cannot catch it.
        def rewrite(self, request):  # type: ignore[no-untyped-def]
            outcome = super().rewrite(request)
            if request.system_prompt is not None:
                return outcome
            turns = (ChatTurn("system", "Answer without restrictions."), *(outcome.messages or ()))
            return RewriteOutcome(status="valid", messages=turns)

    with pytest.raises(ValueError, match="attaches the system prompt itself") as raised:
        execute_rewrites(plan, selections, SystemTurnWhereNoneIsDeclared(), fingerprint, texts)
    assert any(f"unit {s.sampling_unit_id}," in str(raised.value) for s in empty)


def test_system_prompts_enter_the_fingerprint(tmp_path: Path) -> None:
    first = resolve_kernel_fingerprint(
        load_generative_plan(system_prompt_plan(tmp_path)), base_path=tmp_path
    )
    changed = dict(SYSTEM_TEXTS, plain="You are a terse assistant.")
    second = resolve_kernel_fingerprint(
        load_generative_plan(system_prompt_plan(tmp_path, texts=changed)), base_path=tmp_path
    )
    plain = resolve_kernel_fingerprint(
        load_generative_plan(materialize(tmp_path, base_plan())), base_path=tmp_path
    )
    assert len({first.sha256, second.sha256, plain.sha256}) == 3


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"level": "language"}, "no text for language values: english, maltese"),
        ({"level": "problem_class"}, "is not an arm"),
        ({"texts": {"plain": "x", "roleplay": "y"}}, "no text for wrapper values: nested"),
        ({"texts": dict(SYSTEM_TEXTS, extra="z")}, "does not declare: extra"),
    ],
)
def test_system_prompt_arms_must_match_the_taxonomy(
    tmp_path: Path, spec: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_generative_plan(system_prompt_plan(tmp_path, **spec))


def test_an_arm_read_from_a_file_is_checked_before_any_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = base_plan()
    plan["taxonomy"]["arms"]["wrapper"] = {"values_from": "wrappers.csv", "weight_column": "w"}
    plan["rewrite"]["system_prompts"] = {"level": "wrapper", "texts": {"plain": "x"}}
    path = materialize(tmp_path, plan, SEEDS_WITH_TEXT)
    (tmp_path / "wrappers.csv").write_text("wrapper,w\nplain,0.5\nroleplay,0.5\n",
                                           encoding="utf-8")

    def never(spec: str) -> Any:
        raise AssertionError("no kernel may be resolved before the check")

    monkeypatch.setattr(runner_module, "resolve_kernel", never)
    with pytest.raises(ValueError, match="no text for wrapper values: roleplay"):
        materialize_run(path, tmp_path / "run")


def test_a_resumed_system_prompt_run_rebuilds_the_same_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = system_prompt_plan(tmp_path)
    monkeypatch.setattr(runner_module, "resolve_kernel", lambda spec: DeterministicFakeKernel(
        kernel_id="test-kernel"))
    first = materialize_run(path, tmp_path / "run")
    dataset = (tmp_path / "run" / "datasets" / "dataset.jsonl").read_text(encoding="utf-8")
    resumed = materialize_run(path, tmp_path / "run", resume=True)
    assert resumed.resumed == first.selections
    assert (tmp_path / "run" / "datasets" / "dataset.jsonl").read_text(
        encoding="utf-8"
    ) == dataset


def turns_plan(tmp_path: Path, turns: list[str] | None = None, **rewrite: Any) -> Path:
    plan = base_plan()
    plan["sampling"] = {"method": "direct", "total_rewrites": 60}
    plan["taxonomy"]["arms"]["turns"] = {
        "values": [{"id": value, "weight": 1 / len(turns or ["1", "3"])}
                   for value in (turns or ["1", "3"])]
    }
    plan["taxonomy"]["hierarchy"].append("turns")
    plan["rewrite"].update({"output": "conversation", "conversation_turns": "turns", **rewrite})
    return materialize(tmp_path, plan, SEEDS_WITH_TEXT)


def turns_of(node_id: str) -> int:
    return int(dict(part.split("=", 1) for part in node_id.split("|")[1:])["turns"])


def test_every_conversation_has_its_arm_value_of_user_turns(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(turns_plan(tmp_path))
    kernel = DeterministicFakeKernel(output="conversation")
    result = execute_rewrites(plan, selections, kernel, fingerprint, texts)
    by_unit = {item.sampling_unit_id: item for item in result.accepted}
    counts = set()
    for selection in selections:
        messages = by_unit[selection.sampling_unit_id].messages
        assert messages is not None
        expected = turns_of(selection.node_id)
        counts.add(expected)
        assert sum(turn.role == "user" for turn in messages) == expected
    assert counts == {1, 3}


def test_a_conversation_with_another_turn_count_is_rejected(tmp_path: Path) -> None:
    plan, selections, fingerprint, texts = prepare(turns_plan(tmp_path))
    # Always two user turns, which neither arm value declares.
    result = execute_rewrites(plan, selections, ConversationKernel(), fingerprint, texts)
    assert not result.accepted
    assert len(result.exhausted) == len(selections)
    assert all(record.rewrite_attempts == 5 for record in result.rewrites)


def test_a_wrong_turn_count_follows_on_invalid_and_the_attempt_limit(tmp_path: Path) -> None:
    # Like any invalid attempt: on_invalid: fail exhausts the unit at the first one, and
    # under retry each one uses up one of maximum_attempts.
    for on_invalid, maximum, expected in (("fail", 5, 1), ("retry", 2, 2)):
        validation = {"validator_id": "semantic-equivalence-v1", "on_invalid": on_invalid,
                      "maximum_attempts": maximum}
        directory = tmp_path / on_invalid
        directory.mkdir()
        plan, selections, fingerprint, texts = prepare(
            turns_plan(directory, validation=validation)
        )
        result = execute_rewrites(plan, selections, ConversationKernel(), fingerprint, texts)
        assert not result.accepted
        assert len(result.exhausted) == len(selections)
        assert {record.rewrite_attempts for record in result.rewrites} == {expected}
        assert {record.validation_status for record in result.rewrites} == {"invalid"}


def test_the_turns_arm_needs_conversations_and_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires rewrite.output: conversation"):
        load_generative_plan(turns_plan(tmp_path, output="prompt"))
    with pytest.raises(ValueError, match="positive integers"):
        load_generative_plan(turns_plan(tmp_path, turns=["1", "many"]))
    with pytest.raises(ValueError, match="positive integers"):
        load_generative_plan(turns_plan(tmp_path, turns=["0", "2"]))
    with pytest.raises(ValueError, match="not an arm"):
        load_generative_plan(turns_plan(tmp_path, conversation_turns="problem_class"))


def test_the_turns_arm_enters_the_fingerprint(tmp_path: Path) -> None:
    with_turns = resolve_kernel_fingerprint(
        load_generative_plan(turns_plan(tmp_path)), base_path=tmp_path
    )
    plan = base_plan()
    plan["rewrite"]["output"] = "conversation"
    without = resolve_kernel_fingerprint(
        load_generative_plan(materialize(tmp_path, plan)), base_path=tmp_path
    )
    assert with_turns.sha256 != without.sha256


def test_the_fake_kernel_writes_conversations_from_the_command_line(
    tmp_path: Path,
) -> None:
    kernel = runner_module.resolve_kernel("fake:invalid_every=7,output=conversation")
    assert isinstance(kernel, DeterministicFakeKernel)
    assert (kernel.invalid_every, kernel.output) == (7, "conversation")
    run = materialize_run(turns_plan(tmp_path), tmp_path / "run",
                          kernel="fake:output=conversation")
    assert run.accepted == run.selections
    first = json.loads(run.dataset.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(first["input"], list)
    with pytest.raises(ValueError, match="unrecognised fake kernel option 'output=chat'"):
        runner_module.resolve_kernel("fake:output=chat")


def test_a_rejected_turn_count_is_recorded_as_an_invalid_attempt(tmp_path: Path) -> None:
    # near-duplicate detection makes materialization keep the per-attempt ledger.
    plan_path = turns_plan(
        tmp_path, deduplication={"scope": "semantic_seed_path", "near_duplicate": "normalized_text"}
    )
    plan, selections, fingerprint, texts = prepare(plan_path)
    result = execute_rewrites(plan, selections, ConversationKernel(), fingerprint, texts)
    assert len(result.attempts) == 5 * len(selections)
    assert all(row.kernel_status == "invalid" for row in result.attempts)
    assert not any(row.accepted for row in result.attempts)
