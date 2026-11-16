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

from inspect_safety_stats.generative.schema import GenerativePlan

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
    kernel_id: str
    sha256: str
    prompt_template_sha256: str
    seed_replayable: bool


@dataclass(frozen=True)
class RewriteRequest:
    sampling_unit_id: str
    path_hash: str
    path_values: Mapping[str, str]
    semantic_seed_id: str
    seed_text: str
    attempt: int
    selection_seed: int


@dataclass(frozen=True)
class RewriteOutcome:
    """What one rewrite attempt produced.

    A valid outcome carries either ``prompt``, a single user message, or ``messages``, a
    whole conversation, as the plan's ``rewrite.output`` declares. Never both.
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
        "validation": spec.validation.model_dump(mode="json"),
        "deduplication": spec.deduplication.model_dump(mode="json"),
    }
    # Keys for opt-in settings join the payload only when they are used, so a plan that
    # uses none of them keeps the fingerprint it had before they existed.
    if spec.output != "prompt":
        payload["output"] = spec.output
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return KernelFingerprint(
        kernel_id=spec.kernel_id,
        sha256=digest,
        prompt_template_sha256=template_hash,
        seed_replayable=spec.seed_replayable,
    )


@dataclass
class DeterministicFakeKernel:
    """A network-free kernel for tests and dry runs.

    Output depends only on the request, so a suite exercising the sampler and the
    ledger never needs a provider. ``invalid_every`` forces the retry path.
    """

    kernel_id: str = "fake-rewriter"
    invalid_every: int = 0

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
        return RewriteOutcome(
            status="valid",
            prompt=f"[{request.semantic_seed_id}] {request.seed_text} ({rendered}) #{token[:12]}",
        )
