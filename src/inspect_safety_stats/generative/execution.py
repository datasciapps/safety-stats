"""Run the frozen rewrite procedure over selected paths.

The estimand is defined against the *accepted* distribution the complete procedure
induces, so validation, retry and deduplication all happen here rather than being
applied afterwards. A unit whose retries are exhausted is recorded as an incomplete
sampling unit and kept: dropping it would silently alter the path distribution it was
drawn to represent.

Prompt text never enters the ledger. It is returned separately so callers can write it
to a restricted artifact, and only its hash is recorded for provenance.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from inspect_safety_stats.generative.kernels import (
    KernelFingerprint,
    RewriteKernel,
    RewriteRequest,
)
from inspect_safety_stats.generative.ledger import (
    RewriteRecord,
    SelectionRecord,
    prompt_sha256,
)
from inspect_safety_stats.generative.schema import GenerativePlan


@dataclass(frozen=True)
class AcceptedRewrite:
    sampling_unit_id: str
    prompt_id: str
    prompt: str
    prompt_sha256: str
    attempts: int


@dataclass(frozen=True)
class ExecutionResult:
    rewrites: tuple[RewriteRecord, ...]
    accepted: tuple[AcceptedRewrite, ...]
    duplicates_rejected: int
    exhausted: tuple[str, ...]
    errored: tuple[str, ...]

    @property
    def accepted_by_unit(self) -> dict[str, AcceptedRewrite]:
        return {item.sampling_unit_id: item for item in self.accepted}


def load_seed_texts(plan: GenerativePlan, *, base_path: str | Path = ".") -> dict[str, str]:
    """Read the fixed seed prompts the rewrite kernel transforms."""

    taxonomy = plan.taxonomy
    path = Path(base_path) / taxonomy.seed_manifest
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in (taxonomy.semantic_seed_level, taxonomy.seed_text_column):
        if column not in frame.columns:
            raise ValueError(f"seed manifest {path} is missing column {column!r}")
    texts: dict[str, str] = {}
    for record in frame.to_dict("records"):
        seed = str(record[taxonomy.semantic_seed_level]).strip()
        text = str(record[taxonomy.seed_text_column]).strip()
        if not text:
            raise ValueError(f"semantic seed {seed!r} has no text in {path}")
        texts[seed] = text
    return texts


def _dedup_key(plan: GenerativePlan, node_id: str, semantic_seed_id: str) -> str | None:
    scope = plan.rewrite.deduplication.scope
    if scope == "none":
        return None
    if scope == "structural_path":
        return node_id
    return semantic_seed_id


def _prompt_id(sampling_unit_id: str) -> str:
    return f"p-{sampling_unit_id}"


def execute_rewrites(
    plan: GenerativePlan,
    selections: list[SelectionRecord],
    kernel: RewriteKernel,
    fingerprint: KernelFingerprint,
    seed_texts: Mapping[str, str],
    *,
    rewrite_salt: str = "",
) -> ExecutionResult:
    """Draw one accepted rewrite per selected path, ledgering every attempt.

    ``rewrite_salt`` makes a cohort's rewrites independent of another's. Leave it empty
    for the shared-rewrite pairing that ``target.shared_rewrites`` describes.
    """

    validation = plan.rewrite.validation
    deduplication = plan.rewrite.deduplication
    seen: dict[str, set[str]] = {}
    records: list[RewriteRecord] = []
    accepted: list[AcceptedRewrite] = []
    duplicates = 0
    exhausted: list[str] = []
    errored: list[str] = []

    salt_seed = plan.evaluation.selection_seed
    if rewrite_salt:
        salt_digest = hashlib.sha256(f"{salt_seed}|{rewrite_salt}".encode()).digest()
        salt_seed = int.from_bytes(salt_digest[:6], "big")

    for record in selections:
        unit_id = record.sampling_unit_id
        node_id = record.node_id
        seed_id = record.semantic_seed_id
        if seed_id not in seed_texts:
            raise ValueError(f"semantic seed {seed_id!r} has no text supplied")

        path_values = dict(
            part.split("=", 1) for part in node_id.split("|")[1:] if "=" in part
        )
        key = _dedup_key(plan, node_id, seed_id)
        status = "invalid"
        prompt: str | None = None
        attempts = 0

        for attempt in range(1, validation.maximum_attempts + 1):
            attempts = attempt
            outcome = kernel.rewrite(
                RewriteRequest(
                    sampling_unit_id=unit_id,
                    path_hash=record.path_hash,
                    path_values=path_values,
                    semantic_seed_id=seed_id,
                    seed_text=seed_texts[seed_id],
                    attempt=attempt,
                    selection_seed=salt_seed,
                )
            )
            if outcome.status == "error":
                status = "error"
                break
            if outcome.status == "invalid":
                if validation.on_invalid == "fail":
                    break
                continue
            candidate = str(outcome.prompt)
            digest = prompt_sha256(candidate)
            if key is not None and digest in seen.get(key, set()):
                duplicates += 1
                if deduplication.on_duplicate == "retry":
                    continue
            if key is not None:
                seen.setdefault(key, set()).add(digest)
            status = "valid"
            prompt = candidate
            break

        if status == "valid" and prompt is not None:
            entry = AcceptedRewrite(
                sampling_unit_id=unit_id,
                prompt_id=_prompt_id(unit_id),
                prompt=prompt,
                prompt_sha256=prompt_sha256(prompt),
                attempts=attempts,
            )
            accepted.append(entry)
            records.append(
                RewriteRecord(
                    sampling_unit_id=unit_id,
                    rewrite_draw_id=f"{unit_id}-r1",
                    rewrite_kernel_id=fingerprint.kernel_id,
                    rewrite_kernel_sha256=fingerprint.sha256,
                    rewrite_attempts=attempts,
                    validation_status="valid",
                    prompt_id=entry.prompt_id,
                    prompt_sha256=entry.prompt_sha256,
                    source_semantic_seed_id=seed_id,
                )
            )
            continue

        if status == "error":
            errored.append(unit_id)
        else:
            exhausted.append(unit_id)
        records.append(
            RewriteRecord(
                sampling_unit_id=unit_id,
                rewrite_draw_id=f"{unit_id}-r1",
                rewrite_kernel_id=fingerprint.kernel_id,
                rewrite_kernel_sha256=fingerprint.sha256,
                rewrite_attempts=attempts,
                validation_status="error" if status == "error" else "invalid",
                source_semantic_seed_id=seed_id,
            )
        )

    return ExecutionResult(
        rewrites=tuple(records),
        accepted=tuple(accepted),
        duplicates_rejected=duplicates,
        exhausted=tuple(exhausted),
        errored=tuple(errored),
    )


def write_prompt_store(accepted: tuple[AcceptedRewrite, ...], path: str | Path) -> None:
    """Write the restricted prompt artifact.

    This is the only file the package writes that contains prompt text. It is separate
    from the ledger precisely so it can be handled under different access rules.
    """

    pd.DataFrame(
        [
            {
                "sampling_unit_id": item.sampling_unit_id,
                "prompt_id": item.prompt_id,
                "prompt": item.prompt,
                "prompt_sha256": item.prompt_sha256,
            }
            for item in accepted
        ],
        columns=["sampling_unit_id", "prompt_id", "prompt", "prompt_sha256"],
    ).to_csv(path, index=False)
