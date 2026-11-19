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
from typing import TYPE_CHECKING, Any, Literal, Protocol

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
    # What kind of source was read, as its adapter describes it, for the report's provenance.
    source_format: dict[str, Any] = dataclass_field(default_factory=dict)
    # The earliest start time the selected evaluations report, None when none reports one.
    first_started_at: datetime | None = None
    # Whether some selected evaluation reports no start time, so that it may have started
    # before ``first_started_at``.
    start_unrecorded: bool = False
    # Built by the analysis on first use, from these observations and this manifest.
    block_index: BlockIndex | None = dataclass_field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True)
class ErrorContext:
    stage: Literal["target_generation", "post_generation", "unknown"]
    content_delivered: bool | None
    provider_error_code: str | None
    # Why the transcript that would give the stage could not be read, when it could not, as
    # the name of the error; the stage is then "unknown" and normalize reports it as
    # TRANSCRIPT_UNREAD.
    unread_reason: str | None = None


@dataclass(frozen=True)
class StopContext:
    """How a rollout's output ended, for one that a provider's content filter or the
    generation token limit stopped."""

    # Whether the output delivered content before the stop. A stop reported as an API error
    # delivered none, whatever stand-in text the tool put in its place.
    content_delivered: bool
    # The provider's own type and category of the block, when it gave them; None and False
    # for a stop at the token limit.
    block_type: str | None
    block_category: str | None
    block_error: bool
    # Inspect's stop reason: "content_filter", or "max_tokens" for an answer that ran out of
    # generation tokens.
    stop_reason: Literal["content_filter", "max_tokens"] = "content_filter"


@dataclass(frozen=True)
class SampleOrder:
    """The sample order a run recorded, for analyses that read trials in draw order (#126).

    ``shuffle`` is the run's shuffle setting as the source recorded it: an integer seed, True
    for a shuffle without a recorded seed, or None when the samples were not shuffled.
    ``sample_ids`` lists the run's sample ids in the order they were scheduled, or is None
    when the source does not record it.
    """

    run_key: str
    source: str
    shuffle: Any
    shuffled: bool | None
    sample_ids: tuple[str, ...] | None


@dataclass(frozen=True)
class RunRecord:
    """One evaluation run, as a source adapter reports it.

    ``attributes`` holds every field the source has for the run, verbatim, so that cohort
    ``match``, ``require_uniform`` and ``rollout_key`` can name any of them.
    """

    run_key: str
    attributes: Mapping[str, Any]
    # The raw status; the run counts as complete only when it equals "success". None when
    # the source reports no status, which normalize assumes complete and reports.
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
    # None when the source has no sample id; normalize then refuses the rollout.
    sample_id: str | None
    # The identity whose hash, with the score name, is the observation id; None with the
    # sample id.
    identity: Mapping[str, Any] | None
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
    # What kind of source this is, recorded in the report's provenance: at least its format.
    source_format: Mapping[str, Any]
    # The content-filter or token-limit stop of a rollout, or None when its output was stopped
    # by neither. None in place of the function when the source cannot tell; a plan that asks
    # for such stops (score.content_filter_stops, score.token_limit) is then refused.
    stop_context: Callable[[RolloutRecord], StopContext | None] | None = None


class EvaluationSource(Protocol):
    """Where the runs and rollouts of an analysis come from: one tool's logs, read once."""

    def read(self) -> SourceData:
        """Read the runs and rollouts. Raise ``ValueError`` when there are none."""
        ...
