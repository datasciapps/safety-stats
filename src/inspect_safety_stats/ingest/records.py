"""The records ingestion produces and consumes, free of any evaluation tool.

A source adapter reads one tool's logs into :class:`RunRecord` and :class:`RolloutRecord`
objects, wrapped in a :class:`SourceData`; :func:`inspect_safety_stats.ingest.normalize.normalize`
turns them into the :class:`NormalizedData` the analysis reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd

    from inspect_safety_stats.ingest.manifest import ResolvedGroup
    from inspect_safety_stats.schema.observations import Observation
    from inspect_safety_stats.schema.reports import Diagnostic
    from inspect_safety_stats.stats.independence import BlockIndex


@dataclass
class NormalizedData:
    observations: list[Observation]
    manifest: pd.DataFrame
    groups: list[ResolvedGroup]
    diagnostics: list[Diagnostic]
    cohorts: list[dict[str, Any]]
    source_hashes: dict[str, str | None]
    judge_verdicts: dict[str, dict[str, bool | None]] = dataclass_field(default_factory=dict)
    # Built by the analysis on first use, from these observations and this manifest.
    block_index: BlockIndex | None = dataclass_field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True)
class ErrorContext:
    stage: Literal["target_generation", "post_generation", "unknown"]
    content_delivered: bool | None
    provider_error_code: str | None


@dataclass(frozen=True)
class RunRecord:
    """One evaluation run, as a source adapter reports it.

    ``attributes`` holds every field the source has for the run, verbatim, so that cohort
    ``match``, ``require_uniform`` and ``rollout_key`` can name any of them.
    """

    run_key: str
    attributes: Mapping[str, Any]
    # The raw status; the run counts as complete only when it equals "success".
    status: Any
    started_at: datetime | None
    # None when the source has no model field at all.
    model: str | None
    run_id: Any
    source: str


@dataclass(frozen=True)
class RolloutRecord:
    """One rollout of one item in one run, as a source adapter reports it."""

    run_key: str
    # The source's own label for the record, named in diagnostics that point at it.
    record_id: Any
    item_id: str
    # The raw repeat number (an epoch, a seed index), as the source wrote it.
    repeat: Any
    sample_id: str
    # The identity whose hash, with the score name, is the observation id.
    identity: Mapping[str, Any]
    # Every field the source has for the rollout, verbatim, for ``rollout_key``.
    attributes: Mapping[str, Any]
    # Raw score values by score name; a name absent here is missing for this rollout.
    scores: Mapping[str, Any]
    error: Any
    retries: Any
    source: str


@dataclass(frozen=True)
class SourceData:
    """What a source adapter hands to :func:`normalize`.

    ``rollouts`` are in the source's own order, which judge diagnostics follow;
    ``collection_rank`` gives the order rollouts were collected in, which sets
    ``collection_order`` and the order of rollout diagnostics.
    """

    runs: Sequence[RunRecord]
    rollouts: Sequence[RolloutRecord]
    collection_rank: Sequence[int]
    # Score names the source carries for any rollout; a judge or score outside it is absent.
    score_names: frozenset[str]
    # Run fields that make up a run's configuration, besides those a cohort requires uniform.
    configuration_fields: tuple[str, ...]
    error_context: Callable[[RolloutRecord], ErrorContext]
    source_sha256: Callable[[str], str | None]
