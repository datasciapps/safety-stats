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
import unicodedata
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
    AttemptRecord,
    IncrementalCsvWriter,
    RewriteRecord,
    SelectionRecord,
    prompt_sha256,
)
from inspect_safety_stats.generative.schema import GenerativePlan


@dataclass(frozen=True)
class AcceptedRewrite:
    """The accepted prompt text for one sampling unit, kept out of the ledger.

    ``attempts`` is the number of kernel calls the unit took; it is 0 when the entry was
    read back from the prompt store, which does not record it.
    """

    sampling_unit_id: str
    prompt_id: str
    prompt: str
    prompt_sha256: str
    attempts: int


@dataclass(frozen=True)
class ExecutionResult:
    """Everything :func:`execute_rewrites` produced, replayed units included.

    ``rewrites`` holds one ledger record per selected unit. ``exhausted`` and ``errored``
    list the unit ids that ran out of attempts or hit a kernel error, and ``first_error``
    is the detail of the first error seen in this run, or None.
    """

    rewrites: tuple[RewriteRecord, ...]
    accepted: tuple[AcceptedRewrite, ...]
    duplicates_rejected: int
    exhausted: tuple[str, ...]
    errored: tuple[str, ...]
    first_error: str | None = None
    attempts: tuple[AttemptRecord, ...] = ()

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


def acceptance_by_level(
    selections: list[SelectionRecord], rewrites: list[RewriteRecord]
) -> list[AcceptanceRow]:
    """Aggregate rewrite effort and rejection across each level of the taxonomy.

    The estimand is defined against the distribution the *complete* procedure induces,
    including its validation and retry rules. If the kernel struggles on one arm and not
    another -- a low-resource language, an awkward wrapper -- then the accepted
    distribution is differentially shaped across the design. That is a property of what
    was measured rather than a defect, but nothing else makes it visible.
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


@dataclass(frozen=True)
class FidelityRow:
    """What became of every rewrite attempt for one value of one hierarchy level.

    Counts are of attempts, not units: a unit retried three times contributes three.
    ``validator_rejection_rate`` is the share of the rewrites the validator judged that it
    rejected, and is None where it judged none. ``retried_after_duplicate`` counts units
    whose kept rewrite came after one of their attempts was rejected as a duplicate: their
    prompt depends on rewrites accepted for earlier units.
    """

    level: str
    value: str
    units: int
    attempts: int
    kernel_invalid: int
    kernel_errors: int
    exact_duplicates: int
    near_duplicates: int
    validator_judged: int
    validator_rejected: int
    validator_errors: int
    validator_rejection_rate: float | None
    retried_after_duplicate: int


def _path_parts(node_id: str) -> list[tuple[str, str]]:
    parts = []
    for part in node_id.split("|")[1:]:
        if "=" in part:
            level, value = part.split("=", 1)
            parts.append((level, value))
    return parts


def units_retried_after_duplicate(attempts: list[AttemptRecord]) -> set[str]:
    """Units whose kept rewrite followed a duplicate rejection of their own."""

    rejected = {
        item.sampling_unit_id
        for item in attempts
        if item.duplicate != "none" and not item.accepted
    }
    kept = {item.sampling_unit_id for item in attempts if item.accepted}
    return rejected & kept


def fidelity_by_level(
    selections: list[SelectionRecord], attempts: list[AttemptRecord]
) -> list[FidelityRow]:
    """Aggregate the attempt ledger across each level of the taxonomy.

    Acceptance by level says how many units ran out of attempts. This says why attempts
    failed -- the kernel's own rejection, a duplicate, or the validator -- so an arm whose
    rewrites the validator rejects more often than another's is visible even when retries
    hide it from acceptance.
    """

    by_unit: dict[str, list[AttemptRecord]] = {}
    for item in attempts:
        by_unit.setdefault(item.sampling_unit_id, []).append(item)
    dependent = units_retried_after_duplicate(attempts)
    buckets: dict[tuple[str, str], list[str]] = {}
    for selection in selections:
        if selection.sampling_unit_id not in by_unit:
            continue
        for part in _path_parts(selection.node_id):
            buckets.setdefault(part, []).append(selection.sampling_unit_id)

    rows = []
    for (level, value), units in sorted(buckets.items()):
        made = [item for unit in units for item in by_unit[unit]]
        judged = [item for item in made if item.validator_status in {"valid", "invalid"}]
        rejected = sum(1 for item in judged if item.validator_status == "invalid")
        rows.append(
            FidelityRow(
                level=level,
                value=value,
                units=len(units),
                attempts=len(made),
                kernel_invalid=sum(1 for item in made if item.kernel_status == "invalid"),
                kernel_errors=sum(1 for item in made if item.kernel_status == "error"),
                exact_duplicates=sum(1 for item in made if item.duplicate == "exact"),
                near_duplicates=sum(1 for item in made if item.duplicate == "near"),
                validator_judged=len(judged),
                validator_rejected=rejected,
                validator_errors=sum(1 for item in made if item.validator_status == "error"),
                validator_rejection_rate=rejected / len(judged) if judged else None,
                retried_after_duplicate=sum(1 for unit in units if unit in dependent),
            )
        )
    return rows


def uneven_fidelity(rows: list[FidelityRow], *, tolerance: float = 0.05) -> list[str]:
    """Levels whose values the validator rejects at materially different rates."""

    by_level: dict[str, list[float]] = {}
    for row in rows:
        if row.validator_rejection_rate is not None:
            by_level.setdefault(row.level, []).append(row.validator_rejection_rate)
    return sorted(
        level
        for level, rates in by_level.items()
        if len(rates) >= 2 and max(rates) - min(rates) > tolerance
    )


def check_attempts(rewrites: list[RewriteRecord], attempts: list[AttemptRecord]) -> None:
    """Refuse an attempt ledger that does not describe the rewrite ledger beside it."""

    by_unit: dict[str, list[AttemptRecord]] = {}
    for item in attempts:
        by_unit.setdefault(item.sampling_unit_id, []).append(item)
    known = {record.sampling_unit_id for record in rewrites}
    stray = sorted(set(by_unit) - known)
    if stray:
        raise ValueError(
            f"the attempt ledger records units the rewrite ledger does not: {stray[:3]}"
        )
    for record in rewrites:
        rows = sorted(by_unit.get(record.sampling_unit_id, []), key=lambda item: item.attempt)
        numbers = [item.attempt for item in rows]
        kept = [item for item in rows if item.accepted]
        consistent = numbers == list(range(1, record.rewrite_attempts + 1)) and (
            [item.prompt_sha256 for item in kept] == [record.prompt_sha256]
            if record.is_evaluable()
            else not kept
        )
        if not consistent:
            raise ValueError(
                f"unit {record.sampling_unit_id}: the attempt ledger and the rewrite ledger "
                "disagree about its attempts"
            )


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


def normalized_text(prompt: str) -> str:
    """The form two rewrites are compared in under ``near_duplicate: normalized_text``.

    NFKC folds compatibility forms (full-width letters, ligatures), case folding removes
    case, and punctuation and symbols are dropped before whitespace is collapsed, so
    rewrites that differ only in those respects compare equal. Word order, wording and
    language still distinguish them.
    """

    folded = unicodedata.normalize("NFKC", prompt).casefold()
    kept = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in folded
    )
    return " ".join(kept.split())


def _near_digest(plan: GenerativePlan, prompt: str) -> str | None:
    if plan.rewrite.deduplication.near_duplicate == "off":
        return None
    return prompt_sha256(normalized_text(prompt))


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
    completed: Mapping[str, RewriteRecord] | None = None,
    already_accepted: Mapping[str, AcceptedRewrite] | None = None,
    on_unit: Callable[[RewriteRecord, AcceptedRewrite | None], None] | None = None,
    validator: RewriteValidator | None = None,
    on_attempt: Callable[[AttemptRecord], None] | None = None,
) -> ExecutionResult:
    """Draw one accepted rewrite per selected path, ledgering every attempt.

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

    ``validator`` is required when the plan sets ``rewrite.validation.invoke`` and refused
    otherwise, so a declared check is never silently skipped and an undeclared one never
    silently shapes the accepted distribution. It runs on every rewrite the kernel returns
    as valid that is not rejected as a duplicate. When the plan records attempts, each
    kernel call of a newly executed unit becomes an :class:`AttemptRecord`, passed to
    ``on_attempt`` as it is made; the attempts of replayed units are already on disk.
    """

    validation = plan.rewrite.validation
    if validation.invoke and validator is None:
        raise ValueError(
            f"the plan sets rewrite.validation.invoke, so the validator "
            f"{validation.validator_id!r} must be supplied"
        )
    if validator is not None and not validation.invoke:
        raise ValueError(
            "a validator was supplied but the plan does not set rewrite.validation.invoke; "
            "it would shape the accepted distribution without the plan saying so"
        )
    recording = plan.rewrite.records_attempts
    attempt_records: list[AttemptRecord] = []
    deduplication = plan.rewrite.deduplication
    seen: dict[str, set[str]] = {}
    # Normalized forms of the accepted rewrites, per deduplication key, when near
    # duplicates are detected.
    seen_near: dict[str, set[str]] = {}
    records: list[RewriteRecord] = []
    accepted: list[AcceptedRewrite] = []
    duplicates = 0
    exhausted: list[str] = []
    errored: list[str] = []
    first_error: str | None = None

    salt_seed = plan.evaluation.selection_seed
    if rewrite_salt:
        salt_digest = hashlib.sha256(f"{salt_seed}|{rewrite_salt}".encode()).digest()
        salt_seed = int.from_bytes(salt_digest[:6], "big")

    def note(row: dict[str, object]) -> None:
        if not recording:
            return
        entry = AttemptRecord.model_validate(row)
        attempt_records.append(entry)
        if on_attempt is not None:
            on_attempt(entry)

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
                    near = _near_digest(plan, carried.prompt)
                    if near is not None:
                        seen_near.setdefault(key, set()).add(near)
            elif prior.validation_status == "error":
                errored.append(unit_id)
            else:
                exhausted.append(unit_id)
            continue

        status = "invalid"
        prompt: str | None = None
        attempts = 0

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
            row: dict[str, object] = {
                "sampling_unit_id": unit_id,
                "attempt": attempt,
                "kernel_status": outcome.status,
            }
            if outcome.status == "error":
                status = "error"
                if first_error is None:
                    first_error = outcome.detail
                note(row)
                break
            if outcome.status == "invalid":
                note(row)
                if validation.on_invalid == "fail":
                    break
                continue
            candidate = str(outcome.prompt)
            digest = prompt_sha256(candidate)
            near = _near_digest(plan, candidate)
            row["prompt_sha256"] = digest
            if key is not None and digest in seen.get(key, set()):
                row["duplicate"] = "exact"
            elif key is not None and near is not None and near in seen_near.get(key, set()):
                row["duplicate"] = "near"
            if row.get("duplicate") is not None:
                duplicates += 1
                if deduplication.on_duplicate == "retry":
                    note(row)
                    continue
            if validator is not None:
                verdict = validator.validate(request, candidate)
                row["validator_id"] = validator.validator_id
                row["validator_status"] = verdict.status
                if verdict.status == "error":
                    status = "error"
                    if first_error is None:
                        first_error = f"validator: {verdict.detail}"
                    note(row)
                    break
                if verdict.status == "invalid":
                    note(row)
                    if validation.on_invalid == "fail":
                        break
                    continue
            if key is not None:
                seen.setdefault(key, set()).add(digest)
                if near is not None:
                    seen_near.setdefault(key, set()).add(near)
            row["accepted"] = True
            note(row)
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
        attempts=tuple(attempt_records),
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
    ).to_csv(path, index=False, lineterminator="\n")
