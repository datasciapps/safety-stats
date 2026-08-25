from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from inspect_safety_stats.api import analyze as run_analysis
from inspect_safety_stats.api import validate as run_validation
from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.report.visualization import write_visualization
from inspect_safety_stats.schema.reports import SafetyReport
from inspect_safety_stats.stats.planning import (
    certification_power,
    certification_sample_size,
    maximum_certifiable_failure_rate,
    maximum_certifying_failures,
    minimum_detectable_failure_rate,
    zero_failure_sample_size,
)
from inspect_safety_stats.version import __version__

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


@app.command("analyze")
def analyze_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("safety-stats-report"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
    decision_exit_codes: Annotated[
        bool, typer.Option("--decision-exit-codes/--no-decision-exit-codes")
    ] = False,
) -> None:
    if not logs:
        raise typer.BadParameter("at least one --logs path is required")
    log_input: str | list[str] = str(logs[0]) if len(logs) == 1 else [str(path) for path in logs]
    try:
        report = run_analysis(plan=plan, logs=log_input, parallel=parallel or False)
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
    if decision_exit_codes:
        raise typer.Exit(code={"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[report.decision])


@app.command("validate")
def validate_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
) -> None:
    if not logs:
        raise typer.BadParameter("at least one --logs path is required")
    log_input: str | list[str] = str(logs[0]) if len(logs) == 1 else [str(path) for path in logs]
    result = run_validation(plan=plan, logs=log_input, parallel=parallel or False)
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
) -> None:
    count = zero_failure_sample_size(threshold=threshold, alpha=alpha)
    typer.echo(f"Required rollouts: {count}")
    typer.echo(
        f"Decision rule: PASS only if all {count} rollouts are valid and zero failures occur"
    )
    typer.echo(f"One-sided confidence level: {(1 - alpha) * 100:g}%")


@plan_app.command("certification")
def certification_planning_command(
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    anticipated_rate: Annotated[float, typer.Option("--anticipated-rate", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
) -> None:
    """Plan an exact one-sided binomial certification claim."""

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
    typer.echo(f"One-sided alpha: {alpha:g}")


@plan_app.command("detectable-rate")
def detectable_rate_command(
    trials: Annotated[int, typer.Option("--trials", min=1)],
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
) -> None:
    """Invert exact one-sided tests at a fixed rollout horizon."""

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
    typer.echo(f"One-sided alpha: {alpha:g}")


@schema_app.command("plan")
def plan_schema_command() -> None:
    typer.echo(json.dumps(AnalysisPlan.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("report")
def report_schema_command() -> None:
    typer.echo(json.dumps(SafetyReport.model_json_schema(), indent=2, sort_keys=True))


@app.command("version")
def version_command() -> None:
    typer.echo(__version__)


@app.command("hash-plan")
def hash_plan_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
) -> None:
    typer.echo(canonical_plan_hash(load_plan(plan)))


if __name__ == "__main__":
    app()


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
        typer.echo("No rewrite passed validation; there is nothing to evaluate.", err=True)
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


@schema_app.command("generative-plan")
def generative_plan_schema_command() -> None:
    """Print the JSON schema for a generative evaluation plan."""

    from inspect_safety_stats.generative.schema import GenerativePlan

    typer.echo(json.dumps(GenerativePlan.model_json_schema(), indent=2))
