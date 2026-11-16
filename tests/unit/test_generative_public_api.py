"""The generative package's public surface, guarded so it cannot drift silently.

A stability promise only means something if removing or renaming a public name is loud.
This test is the snapshot: adding to it is a deliberate act, and so is taking away.
"""

from __future__ import annotations

import inspect_safety_stats.generative as generative

SURFACE = {
    # The plan and the tree it compiles to.
    "GenerativePlan", "load_generative_plan",
    "CompiledNode", "CompiledTaxonomy", "compile_taxonomy",
    # Deciding a budget before anything is generated.
    "BudgetForecast", "LevelForecast", "forecast", "rewrites_for_bound",
    # The rewrite kernel, which the caller supplies.
    "RewriteKernel", "RewriteRequest", "RewriteOutcome", "KernelFingerprint",
    "DeterministicFakeKernel", "resolve_kernel_fingerprint",
    # The validator the plan declares, which the package invokes when asked to.
    "RewriteValidator", "ValidationVerdict", "DeterministicFakeValidator",
    # The two-phase workflow.
    "MaterializedRun", "materialize_run", "analyze_run",
    "GenerativeReport", "render_generative_report",
    # Results.
    "TreeEstimate", "NodeEstimate", "ComparisonResult", "ReweightedEstimate",
    "CorrectedEstimate", "AcceptanceRow",
    # Composing a flow by hand.
    "SamplingDesign", "StratumAllocation", "plan_design", "draw_selections",
    "execute_rewrites", "ExecutionResult", "AcceptedRewrite", "load_seed_texts",
    "acceptance_by_level", "materialize_dataset", "materialize_for_cohorts",
    "read_evaluated_samples", "samples_from_observations", "join_outcomes",
    "JoinResult", "EvaluatedSample", "UnitOutcome", "estimate_tree", "estimate_slice",
    "estimate_reweighted", "compare_cohorts",
    # Ledgers, for auditing a run.
    "SelectionRecord", "RewriteRecord", "read_selections", "read_rewrites",
    "AttemptRecord", "read_attempts",
}


def test_the_public_surface_is_exactly_what_is_declared() -> None:
    declared = set(generative.__all__)
    assert declared - SURFACE == set(), "new public names need adding to this snapshot"
    assert SURFACE - declared == set(), "public names were removed; that is a breaking change"


def test_every_exported_name_resolves() -> None:
    unresolved = [name for name in generative.__all__ if not hasattr(generative, name)]
    assert unresolved == []


def test_nothing_private_is_exported() -> None:
    assert [name for name in generative.__all__ if name.startswith("_")] == []


def test_internal_helpers_stay_internal() -> None:
    """Cross-module helpers the subpackage shares are not part of the promise."""

    for name in ("_bounds", "_raw_coefficients", "_point_estimate", "_ancestors"):
        assert name not in generative.__all__


def test_the_documented_workflow_is_reachable_from_the_package_root() -> None:
    """The docstring's example must keep working."""

    from inspect_safety_stats.generative import (  # noqa: F401
        analyze_run,
        compile_taxonomy,
        forecast,
        load_generative_plan,
        materialize_run,
    )


def test_a_kernel_can_be_written_against_the_protocol_alone() -> None:
    """A user implementing RewriteKernel needs nothing beyond the exported names."""

    from inspect_safety_stats.generative import RewriteKernel, RewriteOutcome, RewriteRequest

    class Mine:
        kernel_id = "mine"

        def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
            return RewriteOutcome(status="valid", prompt=request.seed_text.upper())

    kernel = Mine()
    assert isinstance(kernel, RewriteKernel)
    outcome = kernel.rewrite(
        RewriteRequest(
            sampling_unit_id="u-0",
            path_hash="a" * 64,
            path_values={"language": "english"},
            semantic_seed_id="s1",
            seed_text="a probe",
            attempt=1,
            selection_seed=1,
        )
    )
    assert outcome.prompt == "A PROBE"
