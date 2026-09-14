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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from inspect_safety_stats.generative.kernels import (
    KernelFingerprint,
    RewriteKernel,
    RewriteRequest,
    RewriteValidator,
)
from inspect_safety_stats.generative.ledger import (
    IncrementalCsvWriter,
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
    first_error: str | None = None
    kernel_rejected: int = 0
    validator_rejected: int = 0

    @property
    def accepted_by_unit(self) -> dict[str, AcceptedRewrite]:
        return {item.sampling_unit_id: item for item in self.accepted}


@dataclass(frozen=True)
class AcceptanceRow:
    """How hard the rewrite procedure worked for one value of one hierarchy level."""

    level: str
    value: str
    units: int
    accepted: int
    exhausted: int
    errored: int
    mean_attempts: float
    exhaustion_rate: float
    # Attempts rejected by each check, summed over the value's units. None when any of
    # those units comes from a ledger written before rejections were recorded by reason.
    kernel_rejections: int | None = None
    validator_rejections: int | None = None
    duplicate_rejections: int | None = None


def _recorded_total(counts: list[int | None]) -> int | None:
    """Sum per-unit rejection counts, or None if any unit predates their recording."""

    if any(count is None for count in counts):
        return None
    return sum(count for count in counts if count is not None)


def acceptance_by_level(
    selections: list[SelectionRecord], rewrites: list[RewriteRecord]
) -> list[AcceptanceRow]:
    """Aggregate rewrite effort and rejection across each level of the taxonomy.

    The estimand is defined against the distribution the *complete* procedure induces,
    including its validation and retry rules. If the kernel struggles on one arm and not
    another -- a low-resource language, an awkward wrapper -- then the accepted
    distribution is differentially shaped across the design. That is a property of what
    was measured rather than a defect, but nothing else makes it visible.

    The three rejection counts say which check did the rejecting: the kernel's own verdict,
    the validator, or deduplication. Exhaustion alone cannot tell a rewriter that drifts
    off meaning from one that repeats itself.
    """

    by_unit = {record.sampling_unit_id: record for record in rewrites}
    buckets: dict[tuple[str, str], list[RewriteRecord]] = {}
    for selection in selections:
        record = by_unit.get(selection.sampling_unit_id)
        if record is None:
            continue
        for part in selection.node_id.split("|")[1:]:
            if "=" not in part:
                continue
            level, value = part.split("=", 1)
            buckets.setdefault((level, value), []).append(record)

    rows = []
    for (level, value), records in sorted(buckets.items()):
        exhausted = sum(1 for item in records if item.validation_status == "invalid")
        errored = sum(1 for item in records if item.validation_status == "error")
        rows.append(
            AcceptanceRow(
                level=level,
                value=value,
                units=len(records),
                accepted=sum(1 for item in records if item.is_evaluable()),
                exhausted=exhausted,
                errored=errored,
                mean_attempts=sum(item.rewrite_attempts for item in records) / len(records),
                exhaustion_rate=exhausted / len(records),
                kernel_rejections=_recorded_total(
                    [item.kernel_rejections for item in records]
                ),
                validator_rejections=_recorded_total(
                    [item.validator_rejections for item in records]
                ),
                duplicate_rejections=_recorded_total(
                    [item.duplicate_rejections for item in records]
                ),
            )
        )
    return rows


def uneven_acceptance(rows: list[AcceptanceRow], *, tolerance: float = 0.05) -> list[str]:
    """Levels whose values are rejected at materially different rates.

    A spread wider than ``tolerance`` in exhaustion rate means the accepted distribution is
    shaped differently across that level, so the risk being estimated for one arm rests on
    a differently filtered set of prompts than another's.
    """

    by_level: dict[str, list[AcceptanceRow]] = {}
    for row in rows:
        by_level.setdefault(row.level, []).append(row)
    uneven = []
    for level, group in sorted(by_level.items()):
        if len(group) < 2:
            continue
        spread = max(r.exhaustion_rate for r in group) - min(r.exhaustion_rate for r in group)
        if spread > tolerance:
            uneven.append(level)
    return uneven


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
    validator: RewriteValidator | None = None,
    rewrite_salt: str = "",
    completed: Mapping[str, RewriteRecord] | None = None,
    already_accepted: Mapping[str, AcceptedRewrite] | None = None,
    on_unit: Callable[[RewriteRecord, AcceptedRewrite | None], None] | None = None,
) -> ExecutionResult:
    """Draw one accepted rewrite per selected path, ledgering every attempt.

    ``validator``, when given, checks every candidate that is not a duplicate against its
    seed's meaning. Duplicates are ruled out first because that is a hash comparison while
    a validator may be a paid model call; the accepted rewrite is the same in either order.
    Without a validator the kernel's own verdict is trusted.

    ``rewrite_salt`` makes a cohort's rewrites independent of another's. Leave it empty
    for the shared-rewrite pairing that ``target.shared_rewrites`` describes.

    ``completed`` carries units a previous run already finished, which are replayed rather
    than re-executed. A unit whose retries were exhausted belongs there: it is part of the
    accepted distribution the estimand is defined against, and re-running it would grant it
    more than ``maximum_attempts`` and quietly shift that distribution. A unit that
    *errored* does not belong there -- an operational failure says nothing about the
    kernel -- so the caller should omit it and let it run again.

    ``on_unit`` is called as each *newly executed* unit finishes, so a caller can persist it
    before the next one starts. Replayed units do not fire it: they are already on disk, and
    re-emitting them would mean deleting the record before rewriting it.
    """

    validation = plan.rewrite.validation
    deduplication = plan.rewrite.deduplication
    seen: dict[str, set[str]] = {}
    records: list[RewriteRecord] = []
    accepted: list[AcceptedRewrite] = []
    duplicates = 0
    kernel_rejected = 0
    validator_rejected = 0
    validator_id = validator.validator_id if validator is not None else None
    exhausted: list[str] = []
    errored: list[str] = []
    first_error: str | None = None

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

        prior = completed.get(unit_id) if completed else None
        if prior is not None:
            records.append(prior)
            carried: AcceptedRewrite | None = None
            if prior.is_evaluable():
                if already_accepted is None or unit_id not in already_accepted:
                    raise ValueError(
                        f"unit {unit_id} is recorded as accepted but its prompt text is "
                        "missing; the rewrite ledger and the prompt store disagree"
                    )
                carried = already_accepted[unit_id]
                accepted.append(carried)
                if key is not None:
                    seen.setdefault(key, set()).add(carried.prompt_sha256)
            elif prior.validation_status == "error":
                errored.append(unit_id)
            else:
                exhausted.append(unit_id)
            continue

        status = "invalid"
        prompt: str | None = None
        attempts = 0
        kernel_rejections = validator_rejections = duplicate_rejections = 0

        for attempt in range(1, validation.maximum_attempts + 1):
            attempts = attempt
            request = RewriteRequest(
                sampling_unit_id=unit_id,
                path_hash=record.path_hash,
                path_values=path_values,
                semantic_seed_id=seed_id,
                seed_text=seed_texts[seed_id],
                attempt=attempt,
                selection_seed=salt_seed,
            )
            outcome = kernel.rewrite(request)
            if outcome.status == "error":
                status = "error"
                if first_error is None:
                    first_error = outcome.detail
                break
            if outcome.status == "invalid":
                kernel_rejections += 1
                if validation.on_invalid == "fail":
                    break
                continue
            candidate = str(outcome.prompt)
            digest = prompt_sha256(candidate)
            if key is not None and digest in seen.get(key, set()):
                duplicates += 1
                if deduplication.on_duplicate == "retry":
                    duplicate_rejections += 1
                    continue
            if validator is not None:
                verdict = validator.validate(request, candidate)
                if verdict.status == "error":
                    status = "error"
                    if first_error is None:
                        first_error = f"validator: {verdict.detail}"
                    break
                if verdict.status == "not_equivalent":
                    validator_rejections += 1
                    if validation.on_invalid == "fail":
                        break
                    continue
            if key is not None:
                seen.setdefault(key, set()).add(digest)
            status = "valid"
            prompt = candidate
            break

        kernel_rejected += kernel_rejections
        validator_rejected += validator_rejections

        if status == "valid" and prompt is not None:
            entry = AcceptedRewrite(
                sampling_unit_id=unit_id,
                prompt_id=_prompt_id(unit_id),
                prompt=prompt,
                prompt_sha256=prompt_sha256(prompt),
                attempts=attempts,
            )
            accepted.append(entry)
            fresh = RewriteRecord(
                sampling_unit_id=unit_id,
                rewrite_draw_id=f"{unit_id}-r1",
                rewrite_kernel_id=fingerprint.kernel_id,
                rewrite_kernel_sha256=fingerprint.sha256,
                rewrite_attempts=attempts,
                validation_status="valid",
                prompt_id=entry.prompt_id,
                prompt_sha256=entry.prompt_sha256,
                source_semantic_seed_id=seed_id,
                validator_id=validator_id,
                kernel_rejections=kernel_rejections,
                validator_rejections=validator_rejections,
                duplicate_rejections=duplicate_rejections,
            )
            records.append(fresh)
            if on_unit is not None:
                on_unit(fresh, entry)
            continue

        if status == "error":
            errored.append(unit_id)
        else:
            exhausted.append(unit_id)
        unfinished = RewriteRecord(
            sampling_unit_id=unit_id,
            rewrite_draw_id=f"{unit_id}-r1",
            rewrite_kernel_id=fingerprint.kernel_id,
            rewrite_kernel_sha256=fingerprint.sha256,
            rewrite_attempts=attempts,
            validation_status="error" if status == "error" else "invalid",
            source_semantic_seed_id=seed_id,
            validator_id=validator_id,
            kernel_rejections=kernel_rejections,
            validator_rejections=validator_rejections,
            duplicate_rejections=duplicate_rejections,
        )
        records.append(unfinished)
        if on_unit is not None:
            on_unit(unfinished, None)

    return ExecutionResult(
        rewrites=tuple(records),
        accepted=tuple(accepted),
        duplicates_rejected=duplicates,
        exhausted=tuple(exhausted),
        errored=tuple(errored),
        first_error=first_error,
        kernel_rejected=kernel_rejected,
        validator_rejected=validator_rejected,
    )


def read_prompt_store(path: str | Path) -> dict[str, AcceptedRewrite]:
    """Read back the restricted prompt artifact, keyed by sampling unit.

    Resumption needs the prompt text of previously accepted rewrites, both to rebuild the
    Inspect dataset and to reconstruct deduplication state. The ledger carries only hashes,
    so this is the artifact that has to be read.
    """

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ("sampling_unit_id", "prompt_id", "prompt", "prompt_sha256")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"prompt store {path} is missing columns: {', '.join(missing)}")
    entries: dict[str, AcceptedRewrite] = {}
    for row in frame.to_dict("records"):
        unit_id = str(row["sampling_unit_id"])
        entries[unit_id] = AcceptedRewrite(
            sampling_unit_id=unit_id,
            prompt_id=str(row["prompt_id"]),
            prompt=str(row["prompt"]),
            prompt_sha256=str(row["prompt_sha256"]),
            attempts=0,
        )
    return entries


PROMPT_STORE_COLUMNS = ("sampling_unit_id", "prompt_id", "prompt", "prompt_sha256")


def prompt_store_writer(path: str | Path) -> IncrementalCsvWriter:
    return IncrementalCsvWriter(path, PROMPT_STORE_COLUMNS)


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
