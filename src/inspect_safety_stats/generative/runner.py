"""Orchestration for the two phases of a generative evaluation.

Materialization and analysis are separate processes because an Inspect run happens between
them. Everything the second phase needs is on disk: the selection and rewrite ledgers, the
plan, and the logs. The plan is re-compiled in both phases and the result checked against
the ledger, so a plan edited between the two is caught rather than silently changing the
estimand.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from inspect_safety_stats.generative.comparisons import ComparisonResult, compare_cohorts
from inspect_safety_stats.generative.execution import (
    execute_rewrites,
    load_seed_texts,
    write_prompt_store,
)
from inspect_safety_stats.generative.inference import (
    NodeEstimate,
    estimate_slice,
    estimate_tree,
)
from inspect_safety_stats.generative.inspect_bridge import (
    join_outcomes,
    materialize_for_cohorts,
)
from inspect_safety_stats.generative.kernels import (
    DeterministicFakeKernel,
    KernelFingerprint,
    RewriteKernel,
    resolve_kernel_fingerprint,
)
from inspect_safety_stats.generative.ledger import (
    read_rewrites,
    read_selections,
    write_rewrites,
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


def resolve_kernel(specification: str) -> RewriteKernel:
    """Resolve ``fake``, ``fake:invalid_every=N``, or ``module:attribute``.

    A provider-backed kernel is supplied as an import path so nothing in this package
    depends on a particular rewriter client. The attribute may be a kernel or a
    zero-argument factory returning one.
    """

    if specification == "fake" or specification.startswith("fake:"):
        options = specification[5:] if ":" in specification else ""
        invalid_every = 0
        if options:
            key, _, value = options.partition("=")
            if key != "invalid_every" or not value.isdigit():
                raise ValueError(
                    f"unrecognised fake kernel option {options!r}; expected invalid_every=N"
                )
            invalid_every = int(value)
        return DeterministicFakeKernel(invalid_every=invalid_every)

    module_name, separator, attribute = specification.partition(":")
    if not separator or not attribute:
        raise ValueError(
            f"kernel {specification!r} must be 'fake', 'fake:invalid_every=N', or "
            "'module:attribute'"
        )
    module = importlib.import_module(module_name)
    resolved = getattr(module, attribute)
    candidate = resolved() if callable(resolved) and not hasattr(resolved, "rewrite") else resolved
    if not hasattr(candidate, "rewrite"):
        raise ValueError(f"{specification!r} does not provide a rewrite() method")
    return candidate  # type: ignore[no-any-return]


def _load(plan_path: str | Path) -> tuple[GenerativePlan, Path]:
    path = Path(plan_path)
    return load_generative_plan(path), path.parent


def materialize_run(
    plan_path: str | Path,
    output: str | Path,
    *,
    kernel: str = "fake",
    stage: str = "main",
    overwrite: bool = False,
    rewrite_salt: str = "",
) -> MaterializedRun:
    """Compile, select, run the rewrite procedure, and write everything an Inspect run needs."""

    plan, base = _load(plan_path)
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"{directory} is not empty; pass --overwrite to replace it")
    directory.mkdir(parents=True, exist_ok=True)

    taxonomy = compile_taxonomy(plan, base_path=base)
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
    result = execute_rewrites(
        plan,
        selections,
        rewriter,
        fingerprint,
        load_seed_texts(plan, base_path=base),
        rewrite_salt=rewrite_salt,
    )

    write_selections(selections, directory / SELECTION_LEDGER)
    write_rewrites(list(result.rewrites), directory / REWRITE_LEDGER)
    write_prompt_store(result.accepted, directory / PROMPT_STORE)
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
    )


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
    joined = {}
    for cohort_id in plan.target.cohorts:
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

    return GenerativeReport(
        evaluation={
            "id": plan.evaluation.id,
            "title": plan.evaluation.title,
            "intent": plan.evaluation.intent,
            "plan_sha256": plan_sha256(plan_path),
            "taxonomy_fingerprint": taxonomy.fingerprint,
            "kernel_sha256": resolve_kernel_fingerprint(plan, base_path=base).sha256,
            "primary_cohort": primary,
        },
        design={
            "method": design.method,
            "total_rewrites": design.total_rewrites,
            "selection_seed": design.selection_seed,
            "proposal_id": design.proposal_id,
            "strata": len(design.allocations),
            "terminal_paths": len(taxonomy.terminal_paths),
        },
        tree=tree,
        taxonomy=taxonomy,
        slices=cut_results,
        comparisons=comparison_results,
        diagnostics=diagnostics,
    )
