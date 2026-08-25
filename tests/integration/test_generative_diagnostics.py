"""Rewrite acceptance rates, and sensitivity to the declared design."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from inspect_safety_stats.generative.execution import (
    acceptance_by_level,
    execute_rewrites,
    load_seed_texts,
    uneven_acceptance,
)
from inspect_safety_stats.generative.inference import estimate_reweighted
from inspect_safety_stats.generative.kernels import (
    RewriteOutcome,
    RewriteRequest,
    resolve_kernel_fingerprint,
)
from inspect_safety_stats.generative.runner import analyze_run, materialize_run
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy
from tests.integration.test_generative_cli import stub_logs, write_plan


@dataclass
class ArmSensitiveKernel:
    """Struggles on one arm and not the others, the way a real rewriter does."""

    hard_value: str = "maltese"
    kernel_id: str = "fake-rewriter"

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        if request.path_values.get("language") == self.hard_value:
            return RewriteOutcome(status="invalid", detail="cannot hold the constraint")
        return RewriteOutcome(status="valid", prompt=f"rewrite of {request.sampling_unit_id}")


def test_acceptance_is_aggregated_by_level(tmp_path: Path) -> None:
    plan_path = write_plan(tmp_path, total=200)
    parsed = load_generative_plan(plan_path)
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    selections = draw_selections(taxonomy, plan_design(taxonomy, parsed))
    result = execute_rewrites(
        parsed,
        selections,
        ArmSensitiveKernel(),
        resolve_kernel_fingerprint(parsed, base_path=tmp_path),
        load_seed_texts(parsed, base_path=tmp_path),
    )

    rows = acceptance_by_level(selections, list(result.rewrites))
    languages = {row.value: row for row in rows if row.level == "language"}
    assert languages["maltese"].exhaustion_rate == 1.0
    assert languages["english"].exhaustion_rate == 0.0
    assert languages["maltese"].mean_attempts > languages["english"].mean_attempts
    assert sum(row.units for row in rows if row.level == "language") == len(selections)


def test_an_evenly_accepting_kernel_raises_nothing(tmp_path: Path) -> None:
    plan_path = write_plan(tmp_path, total=100)
    parsed = load_generative_plan(plan_path)
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    selections = draw_selections(taxonomy, plan_design(taxonomy, parsed))
    from inspect_safety_stats.generative.kernels import DeterministicFakeKernel

    result = execute_rewrites(
        parsed,
        selections,
        DeterministicFakeKernel(),
        resolve_kernel_fingerprint(parsed, base_path=tmp_path),
        load_seed_texts(parsed, base_path=tmp_path),
    )
    rows = acceptance_by_level(selections, list(result.rewrites))
    assert all(row.exhaustion_rate == 0.0 for row in rows)
    assert uneven_acceptance(rows) == []


def test_uneven_acceptance_is_reported_in_the_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import inspect_safety_stats.generative.runner as runner_module

    plan_path = write_plan(tmp_path, total=200)
    artifacts = tmp_path / "run"
    original = runner_module.resolve_kernel
    runner_module.resolve_kernel = lambda spec: ArmSensitiveKernel()  # type: ignore[assignment]
    try:
        materialize_run(plan_path, artifacts)
    finally:
        runner_module.resolve_kernel = original  # type: ignore[assignment]
    stub_logs(monkeypatch, artifacts)

    report = analyze_run(plan_path, artifacts, "logs", cohort="candidate")
    diagnostic = next(
        item for item in report.diagnostics if item.code == "GENERATIVE_ACCEPTANCE_UNEVEN"
    )
    assert diagnostic.scope == "language"
    assert diagnostic.details["maltese"] == 1.0
    assert report.acceptance

    path = report.write(tmp_path / "report")
    assert (path / "acceptance.csv").is_file()
    payload = json.loads((path / "report.json").read_text(encoding="utf-8"))
    assert payload["acceptance"]


def sensitivity_plan(tmp_path: Path, weights: dict[str, float]) -> Path:
    plan_path = write_plan(tmp_path, total=400)
    edited = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    edited["inference"] = {
        "alpha": 0.05,
        "design_sensitivity": [
            {"id": "languages_equal", "level": "language", "weights": weights}
        ],
    }
    plan_path.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")
    return plan_path


def test_reweighting_the_declared_design_is_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = sensitivity_plan(tmp_path, {"english": 0.5, "maltese": 0.5})
    artifacts = tmp_path / "run"
    materialize_run(plan_path, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=40)

    report = analyze_run(plan_path, artifacts, "logs", cohort="candidate")
    assert len(report.reweightings) == 1
    item = report.reweightings[0]
    assert item.eligibility == "diagnostic"
    assert item.id == "languages_equal"
    assert item.declared_upper == report.tree.root.upper
    assert item.draws == report.tree.root.draws


def test_an_unchanged_reweighting_reproduces_the_declared_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reweighting to the mix already declared must change nothing material."""

    plan_path = sensitivity_plan(tmp_path, {"english": 0.5, "maltese": 0.5})
    artifacts = tmp_path / "run"
    materialize_run(plan_path, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=40)

    report = analyze_run(plan_path, artifacts, "logs", cohort="candidate")
    item = report.reweightings[0]
    assert item.estimate == pytest.approx(report.tree.root.estimate, abs=1e-9)
    assert item.effective_sample_size == pytest.approx(item.draws)


def test_a_lopsided_reweighting_costs_effective_sample_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reweighting is importance sampling, and the sample was not allocated for it."""

    plan_path = sensitivity_plan(tmp_path, {"english": 0.95, "maltese": 0.05})
    artifacts = tmp_path / "run"
    materialize_run(plan_path, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=40)

    report = analyze_run(plan_path, artifacts, "logs", cohort="candidate")
    item = report.reweightings[0]
    assert item.effective_sample_size < 0.6 * item.draws
    assert item.maximum_coefficient > 1.0 / item.draws
    assert item.upper > report.tree.root.upper       # the alternative costs precision


def test_a_reweighting_must_cover_every_value_of_its_level(tmp_path: Path) -> None:
    plan_path = write_plan(tmp_path, total=100)
    parsed = load_generative_plan(plan_path)
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    design = plan_design(taxonomy, parsed)
    selections = draw_selections(taxonomy, design)
    from inspect_safety_stats.generative.inference import UnitOutcome, estimate_tree

    outcomes = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    root = estimate_tree(
        taxonomy, design, selections, outcomes, report_all_nodes=False
    ).root

    with pytest.raises(ValueError, match="omits values of level"):
        estimate_reweighted(
            taxonomy, design, selections, outcomes, root,
            identifier="partial", level="language", weights={"english": 1.0},
        )
    with pytest.raises(ValueError, match="values absent from level"):
        estimate_reweighted(
            taxonomy, design, selections, outcomes, root,
            identifier="invented", level="language",
            weights={"english": 0.5, "maltese": 0.25, "klingon": 0.25},
        )
    with pytest.raises(ValueError, match="not a level of this taxonomy"):
        estimate_reweighted(
            taxonomy, design, selections, outcomes, root,
            identifier="ghost", level="nonexistent", weights={"a": 1.0},
        )


def test_a_reweighting_that_is_not_a_distribution_is_rejected(tmp_path: Path) -> None:
    from pydantic import ValidationError

    plan_path = sensitivity_plan(tmp_path, {"english": 0.5, "maltese": 0.9})
    with pytest.raises(ValidationError, match="sum to 1.4, not 1"):
        load_generative_plan(plan_path)
