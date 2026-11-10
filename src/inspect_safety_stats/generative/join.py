"""Join evaluated samples back to the sampling units that produced them.

The join knows nothing of the evaluation tool: it takes :class:`EvaluatedSample` rows,
keyed by the sample id written at materialization, and the ledgers.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from inspect_safety_stats.generative.inference import UnitOutcome
from inspect_safety_stats.generative.ledger import RewriteRecord, SelectionRecord
from inspect_safety_stats.generative.schema import GenerativePlan
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.schema.reports import Diagnostic


@dataclass(frozen=True)
class EvaluatedSample:
    sample_id: str
    cohort_id: str
    epoch: int
    failure: bool | None


@dataclass(frozen=True)
class JoinResult:
    outcomes: tuple[UnitOutcome, ...]
    diagnostics: tuple[Diagnostic, ...]
    unevaluated: tuple[str, ...]

    @property
    def has_errors(self) -> bool:
        return any(item.severity == "ERROR" for item in self.diagnostics)


def samples_from_observations(observations: list[Observation]) -> list[EvaluatedSample]:
    """Adapt normalized Inspect observations to the join's input.

    The generative design has no prompt manifest, so the join keys on the sample id
    that materialization wrote, not on ``prompt_id``.
    """

    return [
        EvaluatedSample(
            sample_id=observation.sample_id,
            cohort_id=observation.cohort_id,
            epoch=observation.epoch,
            failure=observation.failure,
        )
        for observation in observations
    ]


def join_outcomes(
    plan: GenerativePlan,
    selections: list[SelectionRecord],
    rewrites: list[RewriteRecord],
    samples: list[EvaluatedSample],
    *,
    cohort_id: str,
) -> JoinResult:
    """Map evaluated samples back to sampling units, one row per selection.

    The rewrite ledger is the input rather than the in-memory execution result, because
    evaluation happens between the two: a second process reading the ledger from disk has
    everything the join needs.

    Every selection produces an outcome, including units whose rewrite never passed
    validation and units that were never evaluated. Both are completed as unclassified
    so the conservative completion in the interval carries them, rather than the
    realized design silently shrinking to whatever happened to run.
    """

    if cohort_id not in plan.target.cohorts:
        raise ValueError(f"cohort {cohort_id!r} is not declared in target.cohorts")
    rollouts = plan.target.rollouts_per_rewrite
    unit_of_prompt = {
        str(record.prompt_id): record.sampling_unit_id
        for record in rewrites
        if record.is_evaluable() and record.prompt_id is not None
    }
    accepted_units = set(unit_of_prompt.values())

    diagnostics: list[Diagnostic] = []
    seen: set[tuple[str, int]] = set()
    grouped: dict[str, list[EvaluatedSample]] = defaultdict(list)
    for sample in samples:
        if sample.cohort_id != cohort_id:
            continue
        unit = unit_of_prompt.get(sample.sample_id)
        if unit is None:
            diagnostics.append(
                Diagnostic(
                    code="GENERATIVE_SAMPLE_UNMATCHED",
                    severity="ERROR",
                    message=(
                        f"sample {sample.sample_id!r} does not correspond to any accepted "
                        "rewrite; the log and the ledger describe different runs"
                    ),
                    scope=cohort_id,
                )
            )
            continue
        key = (sample.sample_id, sample.epoch)
        if key in seen:
            diagnostics.append(
                Diagnostic(
                    code="GENERATIVE_SAMPLE_DUPLICATE",
                    severity="ERROR",
                    message=(
                        f"sample {sample.sample_id!r} epoch {sample.epoch} appears more than "
                        "once; a retried or re-ingested sample would be counted twice"
                    ),
                    scope=cohort_id,
                )
            )
            continue
        seen.add(key)
        grouped[unit].append(sample)

    outcomes: list[UnitOutcome] = []
    unevaluated: list[str] = []
    over_evaluated: list[str] = []
    for selection in selections:
        unit = selection.sampling_unit_id
        evaluated = grouped.get(unit, [])
        if unit not in accepted_units or not evaluated:
            unevaluated.append(unit)
            outcomes.append(
                UnitOutcome(unit, trials=rollouts, failures=0, unclassified=rollouts)
            )
            continue
        if len(evaluated) > rollouts:
            over_evaluated.append(unit)
        failures = sum(1 for sample in evaluated if sample.failure is True)
        classified = sum(1 for sample in evaluated if sample.failure is not None)
        trials = max(rollouts, len(evaluated))
        outcomes.append(
            UnitOutcome(
                unit,
                trials=trials,
                failures=failures,
                unclassified=trials - classified,
            )
        )

    if over_evaluated:
        diagnostics.append(
            Diagnostic(
                code="GENERATIVE_ROLLOUT_OVERRUN",
                severity="ERROR",
                message=(
                    f"{len(over_evaluated)} sampling units have more evaluated rollouts than "
                    f"target.rollouts_per_rewrite ({rollouts}); the declared design and the "
                    "run disagree"
                ),
                scope=cohort_id,
                details={"sampling_units": sorted(over_evaluated)[:20]},
            )
        )
    if len(unevaluated) == len(selections):
        diagnostics.append(
            Diagnostic(
                code="GENERATIVE_COHORT_NOT_EVALUATED",
                severity="ERROR",
                message=(
                    f"cohort {cohort_id!r} has no evaluated samples at all: no log matched "
                    f"the model it declares ({plan.target.cohorts[cohort_id].model!r}), so "
                    "every sampling unit is completed as unclassified"
                ),
                scope=cohort_id,
                details={"declared_model": plan.target.cohorts[cohort_id].model},
            )
        )
    elif unevaluated:
        diagnostics.append(
            Diagnostic(
                code="GENERATIVE_UNIT_UNEVALUATED",
                severity="WARNING",
                message=(
                    f"{len(unevaluated)} of {len(selections)} sampling units produced no "
                    "usable outcome and are completed as unclassified"
                ),
                scope=cohort_id,
                details={"count": len(unevaluated)},
            )
        )
    return JoinResult(
        outcomes=tuple(outcomes),
        diagnostics=tuple(diagnostics),
        unevaluated=tuple(unevaluated),
    )
