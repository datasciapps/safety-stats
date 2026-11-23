"""The frozen rewrite procedure and its configuration fingerprint.

The estimand is defined against the distribution a *complete* rewrite procedure
induces, including its validation, retry and deduplication rules. Anything that
could shift that distribution therefore belongs in the fingerprint, so a run whose
kernel changed cannot be silently pooled with one whose kernel did not.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from safety_stats.generative.schema import GenerativePlan

ValidationStatus = Literal["valid", "invalid", "error"]
ChatRole = Literal["system", "user", "assistant"]
CHAT_ROLES: tuple[ChatRole, ...] = ("system", "user", "assistant")


@dataclass(frozen=True)
class ChatTurn:
    """One message of a generated conversation.

    Assistant turns in a generated conversation are written by the generator, not by the
    target: the target model answers only the final user turn, conditioned on the rest.
    """

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in CHAT_ROLES:
            raise ValueError(f"chat role must be one of {', '.join(CHAT_ROLES)}; got {self.role!r}")
        if not self.content.strip():
            raise ValueError(f"a {self.role} turn must carry text")


def conversation_problem(turns: Sequence[ChatTurn]) -> str | None:
    """Why ``turns`` cannot be evaluated as a conversation, or ``None`` if it can.

    The target answers the conversation's last turn, so it must end on a user turn. A
    system turn may only open it, and user and assistant turns alternate, as chat
    templates expect. A kernel that parses a conversation out of model output can call
    this to report a malformed one as an invalid rewrite rather than raise.
    """

    if not turns:
        return "the conversation has no turns"
    body = list(turns[1:]) if turns[0].role == "system" else list(turns)
    if not body:
        return "the conversation has only a system turn"
    if any(turn.role == "system" for turn in body):
        return "a system turn may only open the conversation"
    if body[0].role != "user":
        return "the conversation must open with a user turn after any system turn"
    for earlier, later in zip(body, body[1:], strict=False):
        if earlier.role == later.role:
            return f"two {later.role} turns follow each other"
    if body[-1].role != "user":
        return "the conversation must end on a user turn, which the target answers"
    return None


def user_turns(turns: Sequence[ChatTurn]) -> int:
    return sum(1 for turn in turns if turn.role == "user")


def encode_conversation(turns: Sequence[ChatTurn]) -> str:
    """The canonical text of a conversation: what is hashed, deduplicated and stored."""

    return json.dumps(
        [{"role": turn.role, "content": turn.content} for turn in turns],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_conversation(text: str) -> tuple[ChatTurn, ...]:
    """Read back :func:`encode_conversation`'s output, rejecting anything else."""

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"stored conversation is not JSON: {exc}") from exc
    if not isinstance(document, list) or not all(isinstance(item, dict) for item in document):
        raise ValueError("stored conversation is not a list of messages")
    turns = []
    for item in document:
        if set(item) != {"role", "content"} or not isinstance(item["content"], str):
            raise ValueError("a stored message must have exactly a role and text content")
        turns.append(ChatTurn(role=item["role"], content=item["content"]))
    problem = conversation_problem(turns)
    if problem is not None:
        raise ValueError(f"stored conversation is malformed: {problem}")
    return tuple(turns)


@dataclass(frozen=True)
class KernelFingerprint:
    """The identity of a rewrite kernel's resolved configuration.

    Built by :func:`resolve_kernel_fingerprint`. ``sha256`` hashes the provider, model,
    sampling, validation and deduplication settings together with the prompt template's
    hash, which is also kept separately as ``prompt_template_sha256``.
    """

    kernel_id: str
    sha256: str
    prompt_template_sha256: str
    seed_replayable: bool


@dataclass(frozen=True)
class RewriteRequest:
    """One call to a :class:`RewriteKernel`: a seed prompt and the path to rewrite it for.

    ``path_values`` maps each level of the selected path to its value. ``attempt`` counts
    from 1 within a sampling unit, and ``selection_seed`` is the plan's selection seed, or a
    value derived from it and the rewrite salt when one is given.
    """

    sampling_unit_id: str
    path_hash: str
    path_values: Mapping[str, str]
    semantic_seed_id: str
    seed_text: str
    attempt: int
    selection_seed: int
    # The system prompt the procedure attaches to this rewrite, when the plan declares
    # rewrite.system_prompts: context for the kernel, which must not write one itself.
    # None where the arm value's text is empty; a system turn is refused there too.
    system_prompt: str | None = None
    # The number of user turns the conversation must have, when the plan declares
    # rewrite.conversation_turns. A conversation with any other count is rejected.
    conversation_turns: int | None = None


@dataclass(frozen=True)
class RewriteOutcome:
    """What a kernel returns for one attempt.

    A ``valid`` outcome carries either non-empty ``prompt`` text, a single user message, or
    ``messages``, a whole conversation, as the plan's ``rewrite.output`` declares. Never
    both, and no other status may carry either. ``invalid`` may be retried as the plan's
    validation settings allow; ``error`` ends the unit's attempts.
    """

    status: ValidationStatus
    prompt: str | None = None
    detail: str | None = None
    messages: tuple[ChatTurn, ...] | None = None

    def __post_init__(self) -> None:
        if self.messages is not None:
            # A list would make the frozen outcome unhashable and mutable from outside.
            object.__setattr__(self, "messages", tuple(self.messages))
        if self.status == "valid":
            if self.prompt is not None and self.messages is not None:
                raise ValueError("a valid rewrite carries prompt text or messages, not both")
            if self.messages is None and not self.prompt:
                raise ValueError("a valid rewrite must carry prompt text or messages")
            if self.messages is not None:
                problem = conversation_problem(self.messages)
                if problem is not None:
                    raise ValueError(
                        f"a valid rewrite must carry a well-formed conversation: {problem}"
                    )
        elif self.prompt is not None or self.messages is not None:
            raise ValueError("only a valid rewrite may carry prompt text or messages")

    @property
    def text(self) -> str:
        """The canonical text of a valid outcome: the prompt, or the encoded conversation."""

        if self.messages is not None:
            return encode_conversation(self.messages)
        if self.prompt is None:
            raise ValueError(f"a {self.status} rewrite has no text")
        return self.prompt


@runtime_checkable
class RewriteKernel(Protocol):
    """One frozen rewrite procedure.

    Implementations must be stateless with respect to outcomes: nothing a kernel
    observes about target-model failures may influence a later rewrite, or the
    declared path distribution no longer describes what was sampled.
    """

    kernel_id: str

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome: ...


def resolve_kernel_fingerprint(
    plan: GenerativePlan, *, base_path: str | Path = "."
) -> KernelFingerprint:
    """Hash the kernel's complete resolved configuration, template contents included."""

    spec = plan.rewrite
    template_path = Path(base_path) / spec.prompt_template
    template = template_path.read_bytes()
    template_hash = hashlib.sha256(template).hexdigest()
    payload = {
        "kernel_id": spec.kernel_id,
        "provider": spec.provider,
        "model": spec.model,
        "prompt_template_sha256": template_hash,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "seed_replayable": spec.seed_replayable,
        "validation": _settings(spec.validation),
        "deduplication": _settings(spec.deduplication),
    }
    # Keys for opt-in settings join the payload only when they are used, so a plan that
    # uses none of them keeps the fingerprint it had before they existed.
    if spec.output != "prompt":
        payload["output"] = spec.output
    if spec.system_prompts is not None:
        payload["system_prompts"] = spec.system_prompts.model_dump(mode="json")
    if spec.conversation_turns is not None:
        payload["conversation_turns"] = spec.conversation_turns
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return KernelFingerprint(
        kernel_id=spec.kernel_id,
        sha256=digest,
        prompt_template_sha256=template_hash,
        seed_replayable=spec.seed_replayable,
    )


# Settings added after the fingerprint was defined. Each is hashed only when it differs from
# its default, so a plan that does not use it keeps the fingerprint it always had, and a
# plan that does cannot be pooled with one that does not.
_LATER_SETTINGS: dict[str, object] = {"invoke": False, "near_duplicate": "off"}


def _settings(spec: BaseModel) -> dict[str, object]:
    dumped = spec.model_dump(mode="json")
    for key, default in _LATER_SETTINGS.items():
        if key in dumped and dumped[key] == default:
            del dumped[key]
    return dumped


@dataclass(frozen=True)
class ValidationVerdict:
    """A validator's judgement of one rewrite the kernel returned as valid."""

    status: ValidationStatus
    detail: str | None = None


@runtime_checkable
class RewriteValidator(Protocol):
    """The check ``rewrite.validation.validator_id`` names, invoked by the package.

    It sees the request and the rewrite the kernel returned, and says whether the rewrite
    is acceptable: ``invalid`` puts the unit into the retry path, ``error`` is an
    operational failure, as for a kernel. Like a kernel, it must be stateless with respect
    to outcomes, and ``validation.mode: deterministic`` asks that the same input always
    receive the same verdict.
    """

    validator_id: str

    def validate(self, request: RewriteRequest, prompt: str) -> ValidationVerdict: ...


@dataclass
class DeterministicFakeValidator:
    """A network-free validator for tests and dry runs.

    It rejects a rewrite when a hash of it falls in one of ``reject_every`` buckets, so its
    verdicts depend only on the request and the text. ``reject_every=0`` accepts everything.
    """

    validator_id: str = "fake-validator"
    reject_every: int = 0

    def validate(self, request: RewriteRequest, prompt: str) -> ValidationVerdict:
        token = hashlib.sha256(f"{request.sampling_unit_id}|{prompt}".encode()).hexdigest()
        if self.reject_every and int(token[:8], 16) % self.reject_every == 0:
            return ValidationVerdict(status="invalid", detail="synthetic rejection")
        return ValidationVerdict(status="valid")


@dataclass
class DeterministicFakeKernel:
    """A network-free kernel for tests and dry runs.

    Output depends only on the request, so a suite exercising the sampler and the
    ledger never needs a provider. ``invalid_every`` forces the retry path. With
    ``output="conversation"`` it returns a conversation of ``request.conversation_turns``
    user turns (one if unset), with an assistant turn between each pair.
    """

    kernel_id: str = "fake-rewriter"
    invalid_every: int = 0
    output: Literal["prompt", "conversation"] = "prompt"

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        token = hashlib.sha256(
            "|".join(
                [
                    request.path_hash,
                    request.sampling_unit_id,
                    str(request.attempt),
                    str(request.selection_seed),
                ]
            ).encode("utf-8")
        ).hexdigest()
        if self.invalid_every and int(token[:8], 16) % self.invalid_every == 0:
            return RewriteOutcome(status="invalid", detail="synthetic rejection")
        pairs = sorted(request.path_values.items())
        rendered = ", ".join(f"{level}={value}" for level, value in pairs)
        prompt = f"[{request.semantic_seed_id}] {request.seed_text} ({rendered}) #{token[:12]}"
        if self.output == "prompt":
            return RewriteOutcome(status="valid", prompt=prompt)
        turns: list[ChatTurn] = []
        for index in range(1, (request.conversation_turns or 1)):
            turns.append(ChatTurn("user", f"Turn {index} leading to {request.semantic_seed_id}"))
            turns.append(ChatTurn("assistant", f"Reply {index} #{token[:12]}"))
        turns.append(ChatTurn("user", prompt))
        return RewriteOutcome(status="valid", messages=tuple(turns))
