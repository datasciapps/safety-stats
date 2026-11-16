"""Orchestration for the two phases of a generative evaluation.

Materialization and analysis are separate processes because an Inspect run happens between
them. Everything the second phase needs is on disk: the selection and rewrite ledgers, the
plan, and the logs. The plan is re-compiled in both phases and the result checked against
the ledger, so a plan edited between the two is caught rather than silently changing the
estimand.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from inspect_safety_stats.generative.comparisons import ComparisonResult, compare_cohorts
from inspect_safety_stats.generative.execution import (
    AcceptedRewrite,
    acceptance_by_level,
    check_conversation_arms,
    execute_rewrites,
    load_seed_texts,
    prompt_store_writer,
    read_prompt_store,
    uneven_acceptance,
)
from inspect_safety_stats.generative.inference import (
    NodeEstimate,
    TreeEstimate,
    UnitOutcome,
    estimate_reweighted,
    estimate_slice,
    estimate_tree,
)
from inspect_safety_stats.generative.inspect_bridge import materialize_for_cohorts
from inspect_safety_stats.generative.join import join_outcomes
from inspect_safety_stats.generative.judge import (
    CorrectedEstimate,
    comparison_refusal,
    correct,
    declared_calibration,
    load_labels,
    measured_calibration,
)
from inspect_safety_stats.generative.kernels import (
    DeterministicFakeKernel,
    KernelFingerprint,
    RewriteKernel,
    resolve_kernel_fingerprint,
)
from inspect_safety_stats.generative.ledger import (
    RewriteRecord,
    SelectionRecord,
    read_rewrites,
    read_selections,
    rewrite_ledger_writer,
    write_selections,
)
from inspect_safety_stats.generative.logs import LogInput, read_evaluated_samples
from inspect_safety_stats.generative.report import GenerativeReport, plan_sha256
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import GenerativePlan, load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy
from inspect_safety_stats.schema.reports import Diagnostic

SELECTION_LEDGER = "selections.csv"
REWRITE_LEDGER = "rewrites.csv"
PROMPT_STORE = "prompts.restricted.csv"


@dataclass(frozen=True)
class MaterializedRun:
    output: Path
    dataset: Path
    selections: int
    accepted: int
    exhausted: int
    errored: int
    duplicates_rejected: int
    taxonomy_fingerprint: str
    kernel: KernelFingerprint
    kernel_mismatch: str | None
    resumed: int = 0
    first_error: str | None = None


def resolve_kernel(specification: str) -> RewriteKernel:
    """Resolve ``fake``, ``fake:invalid_every=N,output=conversation``, or ``module:attribute``.

    A provider-backed kernel is supplied as an import path so nothing in this package
    depends on a particular rewriter client. The attribute may be a kernel or a
    zero-argument factory returning one.
    """

    if specification == "fake" or specification.startswith("fake:"):
        options = specification[5:] if ":" in specification else ""
        invalid_every = 0
        output: Literal["prompt", "conversation"] = "prompt"
        for option in options.split(",") if options else []:
            key, _, value = option.partition("=")
            if key == "invalid_every" and value.isdigit():
                invalid_every = int(value)
            elif key == "output" and value in ("prompt", "conversation"):
                output = "conversation" if value == "conversation" else "prompt"
            else:
                raise ValueError(
                    f"unrecognised fake kernel option {option!r}; expected invalid_every=N "
                    "or output=prompt|conversation, separated by commas"
                )
        return DeterministicFakeKernel(invalid_every=invalid_every, output=output)

    # rpartition, so a Windows drive letter in a file path is not mistaken for the
    # separator.
    target, separator, attribute = specification.rpartition(":")
    if not separator or not attribute or not target:
        raise ValueError(
            f"kernel {specification!r} must be 'fake', 'fake:invalid_every=N', "
            "'module:attribute', or 'path/to/file.py:attribute'"
        )

    candidate = Path(target)
    if candidate.suffix == ".py":
        if not candidate.is_file():
            raise ValueError(f"kernel file {candidate} does not exist")
        loader = importlib.util.spec_from_file_location(candidate.stem, candidate)
        if loader is None or loader.loader is None:
            raise ValueError(f"kernel file {candidate} could not be loaded")
        module = importlib.util.module_from_spec(loader)
        # Registered before execution so dataclasses and typing inside it resolve.
        sys.modules[candidate.stem] = module
        loader.loader.exec_module(module)
    else:
        # A console script does not put the working directory on sys.path, but someone
        # naming a module on the command line means the one in their project.
        working = str(Path.cwd())
        if working not in sys.path:
            sys.path.insert(0, working)
        try:
            module = importlib.import_module(target)
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"kernel module {target!r} could not be imported ({exc}). Name an "
                "installed module, or point at the file directly as "
                f"'path/to/file.py:{attribute}'."
            ) from exc

    try:
        resolved = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"{target!r} has no attribute {attribute!r}") from exc
    candidate = resolved() if callable(resolved) and not hasattr(resolved, "rewrite") else resolved
    if not hasattr(candidate, "rewrite"):
        raise ValueError(f"{specification!r} does not provide a rewrite() method")
    return candidate  # type: ignore[no-any-return]


def _load(plan_path: str | Path) -> tuple[GenerativePlan, Path]:
    path = Path(plan_path)
    return load_generative_plan(path), path.parent


def _prior_work(
    directory: Path, selections: list[SelectionRecord], *, conversations: bool = False
) -> tuple[dict[str, RewriteRecord], dict[str, AcceptedRewrite]]:
    """Read what an earlier run finished, and check it describes this same design.

    Units recorded as ``valid`` or ``invalid`` are final. An exhausted unit is part of the
    accepted distribution the estimand is defined against, so re-running it would grant it
    more attempts than the plan allows. A unit recorded as ``error`` is not final: an
    operational failure says nothing about the kernel, so it is left out and runs again.
    """

    ledger = directory / REWRITE_LEDGER
    if not ledger.exists():
        return {}, {}

    recorded = read_selections(directory / SELECTION_LEDGER)
    if [(item.sampling_unit_id, item.path_hash) for item in recorded] != [
        (item.sampling_unit_id, item.path_hash) for item in selections
    ]:
        raise ValueError(
            "the plan no longer reproduces the selection ledger in this directory, so the "
            "partial run it holds belongs to a different design; resume is not safe"
        )

    store = directory / PROMPT_STORE
    prompts = read_prompt_store(store, conversations=conversations) if store.exists() else {}
    completed: dict[str, RewriteRecord] = {}
    accepted: dict[str, AcceptedRewrite] = {}
    for record in read_rewrites(ledger):
        if record.validation_status == "error":
            continue
        completed[record.sampling_unit_id] = record
        if record.is_evaluable():
            entry = prompts.get(record.sampling_unit_id)
            if entry is None:
                raise ValueError(
                    f"unit {record.sampling_unit_id} is recorded as accepted but the prompt "
                    "store has no text for it; the two artifacts disagree"
                )
            accepted[record.sampling_unit_id] = entry
    return completed, accepted


def materialize_run(
    plan_path: str | Path,
    output: str | Path,
    *,
    kernel: str = "fake",
    stage: str = "main",
    overwrite: bool = False,
    resume: bool = False,
    rewrite_salt: str = "",
) -> MaterializedRun:
    """Compile, select, run the rewrite procedure, and write everything an Inspect run needs.

    Ledgers are written as each unit completes rather than in one batch at the end, so a run
    killed partway leaves a usable record. ``resume`` picks that record up and executes only
    the units still outstanding; because selection is counter-based, the units it re-derives
    are identical to the ones the interrupted run would have gone on to draw.
    """

    plan, base = _load(plan_path)
    directory = Path(output)
    occupied = directory.exists() and any(directory.iterdir())
    if occupied and not (overwrite or resume):
        raise FileExistsError(
            f"{directory} is not empty; pass --overwrite to replace it or --resume to "
            "continue the run it holds"
        )
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in (SELECTION_LEDGER, REWRITE_LEDGER, PROMPT_STORE):
            (directory / name).unlink(missing_ok=True)

    taxonomy = compile_taxonomy(plan, base_path=base)
    check_conversation_arms(plan, taxonomy)
    fingerprint = resolve_kernel_fingerprint(plan, base_path=base)
    design = plan_design(taxonomy, plan)
    rewriter = resolve_kernel(kernel)
    # The fingerprint hashes the plan's declared configuration, not the object supplied
    # here, so nothing can verify that an arbitrary implementation honours it. Comparing
    # the identifiers at least catches a fake kernel run against a real plan.
    declared = plan.rewrite.kernel_id
    running = getattr(rewriter, "kernel_id", "")
    mismatch = (
        None
        if running == declared
        else f"plan declares kernel {declared!r} but {running!r} produced the rewrites"
    )
    if stage not in {"pilot", "main"}:
        raise ValueError("stage must be 'pilot' or 'main'")
    selections = draw_selections(taxonomy, design, stage_id=stage)  # type: ignore[arg-type]

    completed, accepted_before = (
        _prior_work(
            directory, selections, conversations=plan.rewrite.stores_conversations()
        )
        if resume
        else ({}, {})
    )
    resumed = len(completed)
    # Selections are known before any kernel call, so they are persisted first: an
    # interrupted run then still records what it set out to evaluate.
    write_selections(selections, directory / SELECTION_LEDGER)

    # Both artifacts are appended to, never rewritten. Resuming therefore cannot destroy
    # the work it is building on, even if the resumed run is itself interrupted. Rows land
    # in completion order rather than selection order; nothing downstream reads order.
    ledger = rewrite_ledger_writer(directory / REWRITE_LEDGER)
    prompts_out = prompt_store_writer(directory / PROMPT_STORE)
    with ledger, prompts_out:
        def persist(record: RewriteRecord, entry: AcceptedRewrite | None) -> None:
            ledger.append(record.model_dump(mode="json"))
            if entry is not None:
                prompts_out.append({
                    "sampling_unit_id": entry.sampling_unit_id,
                    "prompt_id": entry.prompt_id,
                    "prompt": entry.prompt,
                    "prompt_sha256": entry.prompt_sha256,
                })

        result = execute_rewrites(
            plan,
            selections,
            rewriter,
            fingerprint,
            load_seed_texts(plan, base_path=base),
            rewrite_salt=rewrite_salt,
            completed=completed,
            already_accepted=accepted_before,
            on_unit=persist,
        )

    datasets = materialize_for_cohorts(plan, selections, result, directory / "datasets")
    return MaterializedRun(
        output=directory,
        dataset=next(iter(datasets.values())),
        selections=len(selections),
        accepted=len(result.accepted),
        exhausted=len(result.exhausted),
        errored=len(result.errored),
        duplicates_rejected=result.duplicates_rejected,
        taxonomy_fingerprint=taxonomy.fingerprint,
        kernel=fingerprint,
        kernel_mismatch=mismatch,
        resumed=resumed,
        first_error=result.first_error,
    )



def _judge_correction(
    plan: GenerativePlan,
    base: Path,
    rewrites: list[RewriteRecord],
    outcomes: list[UnitOutcome],
    tree: TreeEstimate,
    slices: list[NodeEstimate],
    comparisons: list[ComparisonResult],
    diagnostics: list[Diagnostic],
    *,
    cohort_id: str,
) -> tuple[list[CorrectedEstimate], dict[str, Any]]:
    """Map the reported intervals onto true risk, if the plan asks for it.

    The root is corrected under every declared scenario, since sensitivity to the judge is
    mostly a root-level question. Nodes and slices are corrected under the primary scenario
    only, which keeps the artifact readable on a taxonomy with hundreds of nodes.
    """

    spec = plan.judge_calibration
    if spec is None:
        return [], {}

    tuples = [
        (item.sampling_unit_id, item.trials, item.failures, item.unclassified)
        for item in outcomes
    ]
    corrected: list[CorrectedEstimate] = []
    alpha = plan.inference.alpha

    if spec.mode == "assumed":
        for scenario in spec.scenarios:
            calibration = declared_calibration(plan, scenario.id)
            targets = [tree.root]
            if scenario.id == spec.primary:
                targets = [*tree.nodes, *slices]
            corrected.extend(correct(targets, calibration, interval_alpha=alpha))
        meta: dict[str, Any] = {
            "judge_calibration": {"mode": "assumed", "primary": spec.primary}
        }
    else:
        labels = load_labels(base / str(spec.validation_manifest))
        measured, judge_diagnostics = measured_calibration(
            plan, rewrites, tuples, labels, cohort_id=cohort_id
        )
        diagnostics.extend(judge_diagnostics)
        if measured is None:
            unavailable: dict[str, Any] = {
                "judge_calibration": {"mode": "validated", "available": False}
            }
            return [], unavailable
        corrected.extend(correct([*tree.nodes, *slices], measured, interval_alpha=alpha))
        meta = {
            "judge_calibration": {
                "mode": "validated",
                "available": True,
                "rho_1": [measured.rho_1_lower, measured.rho_1_upper],
                "rho_0": [measured.rho_0_lower, measured.rho_0_upper],
                "alpha_spent": measured.alpha,
            }
        }
        if measured.completed_strata:
            meta["judge_calibration"]["completed_strata"] = list(measured.completed_strata)

    for comparison in comparisons:
        diagnostics.append(
            comparison_refusal(
                f"{comparison.candidate_cohort}:{comparison.baseline_cohort}"
            )
        )
    return corrected, meta


def analyze_run(
    plan_path: str | Path,
    artifacts: str | Path,
    logs: LogInput,
    *,
    cohort: str | None = None,
    slices: tuple[str, ...] = (),
    comparisons: tuple[str, ...] = (),
    parallel: bool | int = False,
) -> GenerativeReport:
    """Join a completed Inspect run against the ledgers and estimate the design."""

    plan, base = _load(plan_path)
    taxonomy = compile_taxonomy(plan, base_path=base)
    design = plan_design(taxonomy, plan)
    ledger_path = Path(artifacts)
    selections = read_selections(ledger_path / SELECTION_LEDGER)
    rewrites = read_rewrites(ledger_path / REWRITE_LEDGER)

    diagnostics: list[Diagnostic] = []
    expected = draw_selections(taxonomy, design, stage_id=selections[0].stage_id)
    recorded = [(item.sampling_unit_id, item.path_hash) for item in selections]
    if [(item.sampling_unit_id, item.path_hash) for item in expected] != recorded:
        raise ValueError(
            "the plan no longer reproduces the recorded selection ledger; the plan was "
            "edited after materialization, so the declared design and the evaluated one "
            "differ"
        )

    samples, log_diagnostics = read_evaluated_samples(plan, logs, parallel=parallel)
    diagnostics.extend(log_diagnostics)

    primary = cohort or next(iter(plan.target.cohorts))
    if primary not in plan.target.cohorts:
        raise ValueError(f"cohort {primary!r} is not declared in target.cohorts")
    # Only the cohorts this analysis needs: joining the rest reports them as wholly
    # unevaluated, which reads like a failed run rather than a cohort nobody asked about.
    needed = {primary}
    for specification in comparisons:
        needed.update(part for part in specification.split(":") if part)
    joined = {}
    for cohort_id in sorted(needed):
        if cohort_id not in plan.target.cohorts:
            raise ValueError(f"cohort {cohort_id!r} is not declared in target.cohorts")
        result = join_outcomes(plan, selections, rewrites, samples, cohort_id=cohort_id)
        joined[cohort_id] = list(result.outcomes)
        diagnostics.extend(result.diagnostics)

    tree = estimate_tree(
        taxonomy,
        design,
        selections,
        joined[primary],
        alpha=plan.inference.alpha,
        coverage_mode=plan.inference.coverage_mode,
        simultaneous_tree=plan.inference.simultaneous_tree,
        report_all_nodes=plan.inference.report_all_nodes,
    )

    cut_results: list[NodeEstimate] = []
    for specification in slices:
        level, separator, value = specification.partition("=")
        if not separator:
            raise ValueError(f"slice {specification!r} must be written level=value")
        cut_results.append(
            estimate_slice(
                taxonomy,
                design,
                selections,
                joined[primary],
                level=level,
                value=value,
                alpha=plan.inference.alpha,
                coverage_mode=plan.inference.coverage_mode,
            )
        )

    comparison_results: list[ComparisonResult] = []
    for specification in comparisons:
        candidate_id, separator, baseline_id = specification.partition(":")
        if not separator:
            raise ValueError(f"comparison {specification!r} must be written candidate:baseline")
        for cohort_id in (candidate_id, baseline_id):
            if cohort_id not in plan.target.cohorts:
                raise ValueError(f"cohort {cohort_id!r} is not declared in target.cohorts")
        comparison_results.append(
            compare_cohorts(
                taxonomy,
                design,
                selections,
                joined[candidate_id],
                joined[baseline_id],
                candidate_cohort=candidate_id,
                baseline_cohort=baseline_id,
                alpha=plan.inference.alpha,
                coverage_mode=plan.inference.coverage_mode,
                paired=plan.target.shared_rewrites,
            )
        )

    acceptance = acceptance_by_level(selections, rewrites)
    for level in uneven_acceptance(acceptance):
        spread = [row for row in acceptance if row.level == level]
        diagnostics.append(
            Diagnostic(
                code="GENERATIVE_ACCEPTANCE_UNEVEN",
                severity="WARNING",
                message=(
                    f"the rewrite procedure rejects values of {level!r} at materially "
                    "different rates, so the accepted prompt distribution is shaped "
                    "differently across that level"
                ),
                scope=level,
                details={
                    row.value: round(row.exhaustion_rate, 4)
                    for row in sorted(spread, key=lambda item: -item.exhaustion_rate)
                },
            )
        )

    reweightings = [
        estimate_reweighted(
            taxonomy,
            design,
            selections,
            joined[primary],
            tree.root,
            identifier=reweighting.id,
            level=reweighting.level,
            weights=reweighting.weights,
            alpha=plan.inference.alpha,
            coverage_mode=plan.inference.coverage_mode,
        )
        for reweighting in plan.inference.design_sensitivity
    ]

    corrected, judge_meta = _judge_correction(
        plan,
        base,
        rewrites,
        joined[primary],
        tree,
        cut_results,
        comparison_results,
        diagnostics,
        cohort_id=primary,
    )

    return GenerativeReport(
        evaluation={
            "id": plan.evaluation.id,
            "title": plan.evaluation.title,
            "intent": plan.evaluation.intent,
            "plan_sha256": plan_sha256(plan_path),
            "taxonomy_fingerprint": taxonomy.fingerprint,
            "kernel_sha256": resolve_kernel_fingerprint(plan, base_path=base).sha256,
            "primary_cohort": primary,
            **judge_meta,
        },
        design={
            "method": design.method,
            "total_rewrites": design.total_rewrites,
            "selection_seed": design.selection_seed,
            "proposal_id": design.proposal_id,
            "strata": len(design.allocations),
            "allocation": [
                {
                    "stratum_id": item.stratum_id,
                    "stratum_weight": item.stratum_weight,
                    "draws": item.draws,
                }
                for item in design.allocations
            ],
            "terminal_paths": len(taxonomy.terminal_paths),
        },
        tree=tree,
        taxonomy=taxonomy,
        slices=cut_results,
        comparisons=comparison_results,
        corrected=corrected,
        acceptance=acceptance,
        reweightings=reweightings,
        diagnostics=diagnostics,
    )
