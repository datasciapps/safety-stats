from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

from inspect_safety_stats.version import __version__

if TYPE_CHECKING:
    from inspect_safety_stats.ingest.records import EvaluationSource

# Each command imports what it needs when it runs, so that `version`, `--help`, `plan` and
# `schema` do not load Inspect AI, pandas and the analysis pipeline.

app = typer.Typer(help="Fixed-design statistical risk analysis for Inspect AI evaluations.")
plan_app = typer.Typer(help="Prospective evaluation planning tools.")
schema_app = typer.Typer(help="Print machine-readable schemas.")
generative_app = typer.Typer(
    help=(
        "Generative-design evaluations, where prompts are sampled from a declared "
        "taxonomy rather than fixed in advance."
    )
)
app.add_typer(plan_app, name="plan")
app.add_typer(schema_app, name="schema")
app.add_typer(generative_app, name="generative")


RunsOption = Annotated[
    Path | None,
    typer.Option(
        "--runs",
        exists=True,
        dir_okay=False,
        help=(
            "Runs file of a neutral rollout source (JSONL or CSV); requires "
            "--rollouts. Experimental: see docs/rollout-files.md."
        ),
    ),
]
RolloutsOption = Annotated[
    Path | None,
    typer.Option(
        "--rollouts",
        exists=True,
        dir_okay=False,
        help=(
            "Rollouts file of a neutral rollout source (JSONL or CSV); requires "
            "--runs. Experimental: see docs/rollout-files.md."
        ),
    ),
]


def _source_arguments(
    logs: list[Path] | None, runs: Path | None, rollouts: Path | None
) -> tuple[str | list[str] | None, EvaluationSource | None]:
    """What to analyse, as ``logs`` and ``source``: Inspect logs or rollout files."""

    if runs is not None or rollouts is not None:
        if runs is None or rollouts is None:
            raise typer.BadParameter("--runs and --rollouts are given together")
        if logs:
            raise typer.BadParameter("pass either --logs or --runs and --rollouts, not both")
        from inspect_safety_stats.ingest.rollout_files import RolloutFiles

        return None, RolloutFiles(runs=runs, rollouts=rollouts)
    if not logs:
        raise typer.BadParameter("at least one --logs path is required")
    return (str(logs[0]) if len(logs) == 1 else [str(path) for path in logs]), None


@app.command("analyze")
def analyze_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    runs: RunsOption = None,
    rollouts: RolloutsOption = None,
    output: Annotated[Path, typer.Option("--output")] = Path("safety-stats-report"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
    decision_exit_codes: Annotated[
        bool, typer.Option("--decision-exit-codes/--no-decision-exit-codes")
    ] = False,
    allow_invalid: Annotated[bool, typer.Option("--allow-invalid")] = False,
) -> None:
    """Analyze completed Inspect logs or rollout files against a plan and write the report."""

    log_input, source = _source_arguments(logs, runs, rollouts)
    from inspect_safety_stats.api import analyze as run_analysis

    try:
        report = run_analysis(
            plan=plan, logs=log_input, parallel=parallel or False, source=source
        )
        path = report.write(output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Analysis failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"Analysis: {report.analysis['id']}")
    typer.echo(f"Plan SHA-256: {report.analysis['plan_sha256']}")
    typer.echo(f"Integrity: {report.integrity.status}")
    if report.presentation.get("include_decisions", False):
        for claim in report.claims:
            if claim.mandatory:
                typer.echo(f"{claim.status}: {claim.title}")
        typer.echo(f"Decision: {report.decision}")
    else:
        typer.echo(f"Risk measures: {len(report.group_risks)}")
    typer.echo(f"Report: {path}")
    # An invalid analysis supports no statistical claim, so it must not pass a CI gate by
    # exiting 0. The report is still written above so that its diagnostics can be audited.
    if report.integrity.status == "invalid" and not allow_invalid:
        typer.echo(
            "Integrity is invalid: see the report's diagnostics. "
            "Pass --allow-invalid to exit 0 anyway.",
            err=True,
        )
        raise typer.Exit(code=3)
    if decision_exit_codes:
        raise typer.Exit(code={"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[report.decision])


@app.command("validate")
def validate_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    runs: RunsOption = None,
    rollouts: RolloutsOption = None,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
) -> None:
    """Check logs or rollout files against a plan (manifest, cohorts, rollouts, scores)."""

    log_input, source = _source_arguments(logs, runs, rollouts)
    from inspect_safety_stats.api import validate as run_validation

    result = run_validation(plan=plan, logs=log_input, parallel=parallel or False, source=source)
    for diagnostic in result.diagnostics:
        scope = f" [{diagnostic.scope}]" if diagnostic.scope else ""
        typer.echo(f"{diagnostic.severity} {diagnostic.code}{scope}: {diagnostic.message}")
    typer.echo(f"Integrity: {result.status}")
    raise typer.Exit(code=3 if result.status == "invalid" else 0)


@app.command("visualize")
def visualize_command(
    report: Annotated[Path, typer.Option("--report", exists=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Build a self-contained interactive HTML visualization from a report."""

    from inspect_safety_stats.report.visualization import write_visualization

    try:
        path = write_visualization(report, output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Visualization failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"Visualization: {path}")


@plan_app.command("zero-failure")
def zero_failure_command(
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    family_size: Annotated[int, typer.Option("--family-size", min=1)] = 1,
) -> None:
    """Rollouts needed for zero observed failures to bound the rate below a threshold."""

    from inspect_safety_stats.stats.planning import zero_failure_sample_size

    local_alpha = alpha / family_size
    count = zero_failure_sample_size(threshold=threshold, alpha=local_alpha)
    typer.echo(f"Required rollouts: {count}")
    typer.echo(
        f"Decision rule: PASS only if all {count} rollouts are valid and zero failures occur"
    )
    if family_size > 1:
        typer.echo(
            f"Local alpha: {local_alpha:.8g} (Bonferroni share of {alpha:g} "
            f"across {family_size} simultaneous claims)"
        )
    typer.echo(f"One-sided confidence level: {(1 - local_alpha) * 100:g}%")


@plan_app.command("certification")
def certification_planning_command(
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    anticipated_rate: Annotated[float, typer.Option("--anticipated-rate", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
    family_size: Annotated[int, typer.Option("--family-size", min=1)] = 1,
) -> None:
    """Plan an exact one-sided binomial certification claim."""

    from inspect_safety_stats.stats.planning import (
        certification_power,
        certification_sample_size,
        maximum_certifiable_failure_rate,
        maximum_certifying_failures,
        minimum_detectable_failure_rate,
    )

    family_alpha = alpha
    alpha = alpha / family_size
    try:
        trials = certification_sample_size(
            true_rate=anticipated_rate,
            threshold=threshold,
            alpha=alpha,
            power=power,
        )
        cutoff = maximum_certifying_failures(trials=trials, threshold=threshold, alpha=alpha)
        achieved = certification_power(
            trials=trials,
            true_rate=anticipated_rate,
            threshold=threshold,
            alpha=alpha,
        )
        maximum_certifiable = maximum_certifiable_failure_rate(
            trials=trials,
            threshold=threshold,
            alpha=alpha,
            power=power,
        )
        minimum_detectable = minimum_detectable_failure_rate(
            trials=trials,
            threshold=threshold,
            alpha=alpha,
            power=power,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Required rollouts: {trials}")
    typer.echo(f"Maximum certifying failures: {cutoff}")
    typer.echo(f"Achieved power: {achieved:.6f}")
    typer.echo(
        "Maximum certifiable failure rate at target power: "
        + ("unattainable" if maximum_certifiable is None else f"{maximum_certifiable:.8g}")
    )
    typer.echo(
        "Minimum detectable failure rate at target power: "
        + ("unattainable" if minimum_detectable is None else f"{minimum_detectable:.8g}")
    )
    if family_size > 1:
        typer.echo(
            f"Local alpha: {alpha:.8g} (Bonferroni share of {family_alpha:g} "
            f"across {family_size} simultaneous claims)"
        )
    typer.echo(f"One-sided alpha: {alpha:g}")


@plan_app.command("detectable-rate")
def detectable_rate_command(
    trials: Annotated[int, typer.Option("--trials", min=1)],
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
    family_size: Annotated[int, typer.Option("--family-size", min=1)] = 1,
) -> None:
    """Invert exact one-sided tests at a fixed rollout horizon."""

    from inspect_safety_stats.stats.planning import (
        maximum_certifiable_failure_rate,
        minimum_detectable_failure_rate,
    )

    family_alpha = alpha
    alpha = alpha / family_size
    try:
        maximum_certifiable = maximum_certifiable_failure_rate(
            trials=trials,
            threshold=threshold,
            alpha=alpha,
            power=power,
        )
        minimum_detectable = minimum_detectable_failure_rate(
            trials=trials,
            threshold=threshold,
            alpha=alpha,
            power=power,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Rollouts: {trials}")
    typer.echo(f"Threshold: {threshold:.8g}")
    typer.echo(f"Target power: {power:.8g}")
    typer.echo(
        "Maximum certifiable failure rate: "
        + ("unattainable" if maximum_certifiable is None else f"{maximum_certifiable:.8g}")
    )
    typer.echo(
        "Minimum detectable failure rate: "
        + ("unattainable" if minimum_detectable is None else f"{minimum_detectable:.8g}")
    )
    if maximum_certifiable is not None:
        typer.echo(f"Support-side detectable gap: {threshold - maximum_certifiable:.8g}")
    if minimum_detectable is not None:
        typer.echo(f"Rejection-side detectable gap: {minimum_detectable - threshold:.8g}")
    if family_size > 1:
        typer.echo(
            f"Local alpha: {alpha:.8g} (Bonferroni share of {family_alpha:g} "
            f"across {family_size} simultaneous claims)"
        )
    typer.echo(f"One-sided alpha: {alpha:g}")


class WeightStrategyOption(str, Enum):
    equal_prompt = "equal_prompt"
    nested_equal = "nested_equal"
    explicit = "explicit"


class IndependenceUnitOption(str, Enum):
    rollout = "rollout"
    prompt_cluster = "prompt_cluster"


@plan_app.command("fixed-design")
def fixed_design_planning_command(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
    rollouts: Annotated[int | None, typer.Option("--rollouts", min=1)] = None,
    weights: Annotated[
        WeightStrategyOption, typer.Option("--weights")
    ] = WeightStrategyOption.nested_equal,
    hierarchy: Annotated[str, typer.Option("--hierarchy")] = "task_id,question_id,prompt_id",
    weight_column: Annotated[str | None, typer.Option("--weight-column")] = None,
    independence_unit: Annotated[
        IndependenceUnitOption, typer.Option("--independence-unit")
    ] = IndependenceUnitOption.rollout,
    family_size: Annotated[int, typer.Option("--family-size", min=1)] = 1,
) -> None:
    """Detectable rates for a declared weighted design under a Hoeffding guarantee.

    Reads the design manifest so planning sees the same prompts, weights, and
    expected rollout counts that `analyze` will resolve. Rollout counts come
    from the manifest's `expected_rollouts` column unless `--rollouts` supplies
    a uniform override.
    """

    from inspect_safety_stats.config.models import WeightSpec
    from inspect_safety_stats.ingest.manifest import load_manifest, resolve_weights
    from inspect_safety_stats.stats.planning import fixed_design_detectable_failure_rates

    # resolve_weights only reads the column for explicit weights; ignoring it silently would
    # plan under a weighting the user did not ask for.
    if weight_column is not None and weights is not WeightStrategyOption.explicit:
        raise typer.BadParameter("--weight-column applies only with --weights explicit")
    local_alpha = alpha / family_size
    unit: Literal["rollout", "prompt_cluster"] = (
        "rollout" if independence_unit is IndependenceUnitOption.rollout else "prompt_cluster"
    )
    try:
        frame = load_manifest(manifest)
        spec = WeightSpec.model_validate(
            {
                "strategy": weights.value,
                "hierarchy": [level.strip() for level in hierarchy.split(",") if level.strip()],
                "column": weight_column,
            }
        )
        weight_values = resolve_weights(frame, spec)
        if rollouts is not None:
            trials = [rollouts] * len(frame)
        elif "expected_rollouts" in frame:
            trials = [int(value) for value in frame["expected_rollouts"]]
        else:
            raise ValueError(
                "the design manifest has no expected_rollouts column; "
                "pass --rollouts for a uniform count"
            )
        rates = fixed_design_detectable_failure_rates(
            trials,
            weight_values,
            threshold=threshold,
            support_alpha=local_alpha,
            rejection_alpha=local_alpha,
            power=power,
            independence_unit=unit,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Prompts: {len(frame)}")
    typer.echo(f"Total rollouts: {sum(trials)}")
    typer.echo(f"Independence unit: {rates.independence_unit}")
    typer.echo(f"Effective sample size: {rates.effective_sample_size:.8g}")
    typer.echo(f"Threshold: {threshold:.8g}")
    typer.echo(f"Target power: {power:.8g}")
    typer.echo(
        "Maximum certifiable failure rate at target power: "
        + (
            "unattainable"
            if rates.maximum_certifiable_failure_rate is None
            else f"{rates.maximum_certifiable_failure_rate:.8g}"
        )
    )
    typer.echo(
        "Minimum detectable failure rate at target power: "
        + (
            "unattainable"
            if rates.minimum_detectable_failure_rate is None
            else f"{rates.minimum_detectable_failure_rate:.8g}"
        )
    )
    if family_size > 1:
        typer.echo(
            f"Local alpha: {local_alpha:.8g} (Bonferroni share of {alpha:g} "
            f"across {family_size} simultaneous claims)"
        )
    typer.echo(f"One-sided alpha per direction: {local_alpha:g}")


@schema_app.command("plan")
def plan_schema_command() -> None:
    """Print the JSON schema for analysis plans."""

    from inspect_safety_stats.config.models import AnalysisPlan

    typer.echo(json.dumps(AnalysisPlan.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("report")
def report_schema_command() -> None:
    """Print the JSON schema for reports."""

    from inspect_safety_stats.schema.reports import SafetyReport

    typer.echo(json.dumps(SafetyReport.model_json_schema(), indent=2, sort_keys=True))


@app.command("version")
def version_command() -> None:
    """Print the package version."""

    typer.echo(__version__)


@app.command("hash-plan")
def hash_plan_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
) -> None:
    """Print a plan's canonical SHA-256, for registering a frozen plan."""

    from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan

    typer.echo(canonical_plan_hash(load_plan(plan)))


@generative_app.command("compile")
def generative_compile_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
) -> None:
    """Compile a generative plan and report what the canonical taxonomy contains."""

    from inspect_safety_stats.generative.sampling import plan_design
    from inspect_safety_stats.generative.schema import load_generative_plan
    from inspect_safety_stats.generative.taxonomy import compile_taxonomy

    try:
        parsed = load_generative_plan(plan)
        taxonomy = compile_taxonomy(parsed, base_path=plan.parent)
        design = plan_design(taxonomy, parsed)
    except Exception as exc:
        typer.echo(f"Compilation failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    typer.echo(f"Evaluation: {parsed.evaluation.id}")
    typer.echo(f"Taxonomy fingerprint: {taxonomy.fingerprint}")
    typer.echo(f"Nodes: {len(taxonomy.nodes)}")
    typer.echo(f"Terminal paths: {len(taxonomy.terminal_paths)}")
    typer.echo(f"Pruned by constraint: {len(taxonomy.pruned_by_constraint)}")
    typer.echo(f"Pruned as zero weight: {len(taxonomy.pruned_zero_weight)}")
    typer.echo(f"Design: {design.method}, {design.total_rewrites} rewrites")
    if design.allocations:
        typer.echo(f"Strata: {len(design.allocations)}")
    typer.echo(f"Cohorts: {', '.join(parsed.target.cohorts)}")


@generative_app.command("materialize")
def generative_materialize_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("generative-run"),
    kernel: Annotated[str, typer.Option("--kernel")] = "fake",
    stage: Annotated[str, typer.Option("--stage")] = "main",
    salt: Annotated[str, typer.Option("--rewrite-salt")] = "",
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
) -> None:
    """Select paths, run the rewrite procedure, and write the dataset and ledgers.

    Ledgers are appended as each unit completes, so an interrupted run leaves a usable
    record. --resume continues from it, executing only the units still outstanding.
    """

    from inspect_safety_stats.generative.runner import materialize_run

    # A console script does not put the working directory on sys.path, as `python -m` does,
    # but someone naming a module on the command line means the one in their project. The
    # CLI owns the process, so it sets this; the library leaves sys.path to its caller.
    working = str(Path.cwd())
    if working not in sys.path:
        sys.path.insert(0, working)

    try:
        run = materialize_run(
            plan,
            output,
            kernel=kernel,
            stage=stage,
            overwrite=overwrite,
            resume=resume,
            rewrite_salt=salt,
        )
    except Exception as exc:
        typer.echo(f"Materialization failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    typer.echo(f"Taxonomy fingerprint: {run.taxonomy_fingerprint}")
    typer.echo(f"Kernel: {run.kernel.kernel_id} @ {run.kernel.sha256}")
    typer.echo(f"Selections: {run.selections}")
    if run.resumed:
        typer.echo(f"Resumed: {run.resumed} units carried over, "
                   f"{run.selections - run.resumed} executed")
    typer.echo(
        f"Rewrites: {run.accepted} accepted, {run.duplicates_rejected} duplicates rejected, "
        f"{run.exhausted} exhausted, {run.errored} errored"
    )
    if run.kernel_mismatch:
        typer.echo(f"Warning: {run.kernel_mismatch}", err=True)
    typer.echo(f"Dataset: {run.dataset}")
    typer.echo(f"Artifacts: {run.output}")
    if run.accepted == 0:
        if run.errored >= run.exhausted:
            typer.echo(
                "Every rewrite failed with a provider error, so nothing was generated. "
                "Check credentials, connectivity and the model name in the plan.",
                err=True,
            )
            if run.first_error:
                typer.echo(f"First error: {run.first_error}", err=True)
        else:
            typer.echo(
                "No rewrite passed validation; there is nothing to evaluate. Raise "
                "rewrite.validation.maximum_attempts, loosen rewrite.deduplication, or "
                "check that the kernel returns usable text.",
                err=True,
            )
        raise typer.Exit(code=3)


@generative_app.command("analyze")
def generative_analyze_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    artifacts: Annotated[Path, typer.Option("--artifacts", exists=True, file_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("generative-report"),
    cohort: Annotated[str | None, typer.Option("--cohort")] = None,
    slice_: Annotated[list[str] | None, typer.Option("--slice")] = None,
    compare: Annotated[list[str] | None, typer.Option("--compare")] = None,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Join a completed Inspect run against the ledgers and estimate the design."""

    from inspect_safety_stats.generative.runner import analyze_run

    if not logs:
        raise typer.BadParameter("at least one --logs path is required")
    log_input: str | list[str] = str(logs[0]) if len(logs) == 1 else [str(path) for path in logs]
    try:
        report = analyze_run(
            plan,
            artifacts,
            log_input,
            cohort=cohort,
            slices=tuple(slice_ or ()),
            comparisons=tuple(compare or ()),
            parallel=parallel or False,
        )
        path = report.write(output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Analysis failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    root = report.tree.root
    typer.echo(f"Evaluation: {report.evaluation['id']}")
    typer.echo(f"Taxonomy fingerprint: {report.evaluation['taxonomy_fingerprint']}")
    typer.echo(f"Primary cohort: {report.evaluation['primary_cohort']}")
    typer.echo(f"Draws: {root.draws}, failures: {root.failures}")
    typer.echo(f"Root risk upper bound: {root.upper:.5f} ({root.method})")
    for cut in report.slices:
        typer.echo(f"  {cut.node_id}: upper {cut.upper:.5f}")
    for comparison in report.comparisons:
        typer.echo(
            f"  {comparison.candidate_cohort} - {comparison.baseline_cohort}: "
            f"upper {comparison.upper:+.5f} ({comparison.method})"
        )
    typer.echo(f"Nodes reported: {len(report.tree.nodes)}, unsampled: "
               f"{len(report.tree.unsampled_nodes)}")
    for diagnostic in report.diagnostics:
        if diagnostic.severity == "ERROR":
            typer.echo(f"[ERROR] {diagnostic.code}: {diagnostic.message}", err=True)
    typer.echo(f"Report: {path}")
    if report.has_errors:
        raise typer.Exit(code=2)


@generative_app.command("budget")
def generative_budget_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    assumed_rate: Annotated[float, typer.Option("--assumed-rate", min=0.0, max=0.999)] = 0.03,
    budget: Annotated[list[int] | None, typer.Option("--budget", min=1)] = None,
    threshold: Annotated[float | None, typer.Option("--threshold", min=0.0, max=1.0)] = None,
    acceptance_rate: Annotated[
        float, typer.Option("--acceptance-rate", min=0.01, max=1.0)
    ] = 1.0,
    target_bound: Annotated[
        float | None, typer.Option("--target-bound", min=0.0, max=1.0)
    ] = None,
) -> None:
    """Project what a budget will buy, before generating or evaluating anything.

    Everything reported comes from the compiled taxonomy and one assumed failure rate.
    The projection assumes risk is uniform across the design; real heterogeneity moves
    individual nodes either way, so node figures show what the geometry permits rather
    than what will happen.
    """

    from inspect_safety_stats.generative.planning import forecast, rewrites_for_bound
    from inspect_safety_stats.generative.schema import load_generative_plan
    from inspect_safety_stats.generative.taxonomy import compile_taxonomy

    try:
        parsed = load_generative_plan(plan)
        taxonomy = compile_taxonomy(parsed, base_path=plan.parent)
    except Exception as exc:
        typer.echo(f"Planning failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    declared = parsed.sampling.total_rewrites
    budgets = sorted(set(budget)) if budget else sorted(
        {max(1, int(declared * factor)) for factor in (0.25, 1, 4, 16)}
    )

    typer.echo(f"Evaluation: {parsed.evaluation.id}")
    typer.echo(
        f"Taxonomy: {len(taxonomy.terminal_paths)} terminal paths, "
        f"{len(taxonomy.levels)} levels, declared budget {declared}"
    )
    typer.echo(
        f"Assumed failure rate {assumed_rate}, alpha {parsed.inference.alpha}"
        + (f", acceptance rate {acceptance_rate}" if acceptance_rate < 1.0 else "")
    )
    typer.echo("")

    header = f"{'budget':>9} {'root bound':>11} {'paths hit':>10} {'unsampled':>10}"
    header += f"{'rewrite calls':>15} {'rollouts':>10}"
    if threshold is not None:
        header += f"{'certif. power':>14}"
    typer.echo(header)
    projected = {
        total: forecast(
            taxonomy,
            parsed,
            total_rewrites=total,
            assumed_rate=assumed_rate,
            acceptance_rate=acceptance_rate,
            threshold=threshold,
        )
        for total in budgets
    }
    for total, item in projected.items():
        row = (
            f"{total:>9} {item.root_upper:>11.4f} {item.paths_with_a_draw:>9.0%} "
            f"{item.expected_unsampled_nodes:>10.0f}{item.rewrite_calls:>15.0f} "
            f"{item.target_rollouts:>9}"
        )
        if threshold is not None and item.certification_power is not None:
            row += f"{item.certification_power:>14.2f}"
        if item.starved_strata:
            row += "   infeasible: the declared stratum floor needs more"
        typer.echo(row)

    # Detail the budget the caller actually asked about.
    detail = max(budgets) if budget else declared
    chosen = projected.get(detail) or forecast(
        taxonomy,
        parsed,
        total_rewrites=detail,
        assumed_rate=assumed_rate,
        acceptance_rate=acceptance_rate,
        threshold=threshold,
    )
    typer.echo(f"\nAt a budget of {detail}, by level:")
    typer.echo(
        f"  {'level':<20} {'nodes':>6} {'median draws':>13} {'median bound':>13} "
        f"{'worst bound':>12} {'unsampled':>10}"
    )
    for level in chosen.levels:
        typer.echo(
            f"  {level.level:<20} {level.nodes:>6} {level.median_draws:>13.1f} "
            f"{level.median_upper:>13.4f} {level.worst_upper:>12.4f} "
            f"{level.expected_unsampled:>10.1f}"
        )

    notes: list[str] = []
    for level in chosen.levels:
        if level.median_upper > 4 * chosen.root_upper:
            notes.append(
                f"level {level.level!r} cannot be resolved at this budget: its median node "
                f"would report {level.median_upper:.3f} against a root bound of "
                f"{chosen.root_upper:.4f}"
            )
    if chosen.exhausted_units >= 1:
        notes.append(
            f"about {chosen.exhausted_units:.0f} units would exhaust their retries and be "
            "completed as unclassified"
        )
    if chosen.starved_strata:
        notes.append(f"{chosen.starved_strata} strata would receive no draws")
    elif chosen.coefficient_ratio > 1.0 and chosen.coefficient_ratio != float("inf"):
        notes.append(
            f"allocation is not exactly proportional (coefficient ratio "
            f"{chosen.coefficient_ratio:.4f}), so the sharp exact kernel is unavailable"
        )
    if threshold is not None and chosen.certification_power is not None:
        verdict = "would" if chosen.certification_power >= 0.8 else "would not"
        notes.append(
            f"certifying root risk below {threshold} {verdict} be reliable at {detail} "
            f"rewrites (power {chosen.certification_power:.2f})"
        )
    if target_bound is not None:
        needed = rewrites_for_bound(
            assumed_rate=assumed_rate, target_upper=target_bound, alpha=parsed.inference.alpha
        )
        notes.append(
            f"reaching a root bound of {target_bound} needs about {needed} rewrites"
            if needed
            else f"a root bound of {target_bound} is unreachable at the assumed rate"
        )
    if notes:
        typer.echo("")
        for note in notes:
            typer.echo(f"! {note}")


@schema_app.command("generative-plan")
def generative_plan_schema_command() -> None:
    """Print the JSON schema for a generative evaluation plan."""

    from inspect_safety_stats.generative.schema import GenerativePlan

    typer.echo(json.dumps(GenerativePlan.model_json_schema(), indent=2))


if __name__ == "__main__":
    app()
