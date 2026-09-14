"""The frozen rewrite procedure and its configuration fingerprint.

The estimand is defined against the distribution a *complete* rewrite procedure
induces, including its validation, retry and deduplication rules. Anything that
could shift that distribution therefore belongs in the fingerprint, so a run whose
kernel changed cannot be silently pooled with one whose kernel did not.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from inspect_safety_stats.generative.schema import GenerativePlan

ValidationStatus = Literal["valid", "invalid", "error"]


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
    status: ValidationStatus
    prompt: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status == "valid" and not self.prompt:
            raise ValueError("a valid rewrite must carry prompt text")
        if self.status != "valid" and self.prompt is not None:
            raise ValueError("only a valid rewrite may carry prompt text")


@runtime_checkable
class RewriteKernel(Protocol):
    """One frozen rewrite procedure.

    Implementations must be stateless with respect to outcomes: nothing a kernel
    observes about target-model failures may influence a later rewrite, or the
    declared path distribution no longer describes what was sampled.
    """

    kernel_id: str

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome: ...


VerdictStatus = Literal["equivalent", "not_equivalent", "error"]


@dataclass(frozen=True)
class ValidationVerdict:
    """Whether a candidate rewrite still makes its seed's request."""

    status: VerdictStatus
    detail: str | None = None


@runtime_checkable
class RewriteValidator(Protocol):
    """A check that a candidate rewrite preserves its seed's meaning.

    It sees the request the kernel was given -- the seed text and the arm values the rewrite
    was asked to realize -- so a check can accept a French rewrite of an English seed.
    ``not_equivalent`` sends the unit back through the retry path, like a kernel's own
    ``invalid``. ``error`` is operational and is retried on resume, like a kernel error.

    The kernel's statelessness applies here too: nothing a validator observes about
    target-model failures may influence a later verdict.
    """

    validator_id: str

    def validate(self, request: RewriteRequest, candidate: str) -> ValidationVerdict: ...


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
        # Unset validator settings are left out, so declaring none keeps an old fingerprint.
        "validation": spec.validation.model_dump(mode="json", exclude_none=True),
        "deduplication": spec.deduplication.model_dump(mode="json"),
    }
    # A model-graded validator's question is part of the procedure, so its contents are
    # hashed the way the kernel's template is.
    if spec.validation.prompt_template is not None:
        question = (Path(base_path) / spec.validation.prompt_template).read_bytes()
        payload["validator_prompt_template_sha256"] = hashlib.sha256(question).hexdigest()
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


@dataclass
class DeterministicFakeValidator:
    """A network-free validator for tests and dry runs.

    Its verdict depends only on the candidate text. ``reject_every`` rejects roughly one
    candidate in N, which exercises the retry path from the validator's side.
    """

    validator_id: str = "fake-validator"
    reject_every: int = 0

    def validate(self, request: RewriteRequest, candidate: str) -> ValidationVerdict:
        token = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        if self.reject_every and int(token[:8], 16) % self.reject_every == 0:
            return ValidationVerdict(status="not_equivalent", detail="synthetic rejection")
        return ValidationVerdict(status="equivalent")
