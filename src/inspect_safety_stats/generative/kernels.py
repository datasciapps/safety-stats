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


@dataclass(frozen=True)
class RewriteOutcome:
    """What a kernel returns for one attempt.

    A ``valid`` outcome must carry non-empty ``prompt`` text and no other status may carry
    any. ``invalid`` may be retried as the plan's validation settings allow; ``error`` ends
    the unit's attempts.
    """

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
        "validation": spec.validation.model_dump(mode="json"),
        "deduplication": spec.deduplication.model_dump(mode="json"),
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
