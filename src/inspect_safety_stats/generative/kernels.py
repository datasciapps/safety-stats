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

from pydantic import BaseModel

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
