"""The two-phase generative workflow, from plan through materialization to report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from inspect_safety_stats.generative.ledger import read_rewrites
from inspect_safety_stats.generative.runner import analyze_run, materialize_run, resolve_kernel
from tests.generative_fixtures import SEEDS_WITH_TEXT, base_plan

runner = CliRunner()

CANDIDATE_MODEL = "provider/candidate"
BASELINE_MODEL = "provider/baseline"


def write_plan(tmp_path: Path, total: int = 200, **overrides: Any) -> Path:
    plan = base_plan()
    plan["sampling"] = {"method": "direct", "total_rewrites": total}
    plan["target"]["cohorts"] = {
        "baseline": {"model": BASELINE_MODEL},
        "candidate": {"model": CANDIDATE_MODEL},
    }
    plan.update(overrides)
    (tmp_path / "seeds.csv").write_text(SEEDS_WITH_TEXT, encoding="utf-8")
    (tmp_path / "template.txt").write_text("Rewrite: {seed}\n", encoding="utf-8")
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


def stub_logs(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: Path,
    *,
    candidate_failures: int = 20,
    baseline_failures: int = 8,
    models: tuple[str, str] = (BASELINE_MODEL, CANDIDATE_MODEL),
    score_column: str = "score_harmful",
) -> None:
    """Stand in for an Inspect run over the materialized dataset."""

    prompt_ids = [
        str(record.prompt_id)
        for record in read_rewrites(artifacts / "rewrites.csv")
        if record.is_evaluable()
    ]
    evals = pd.DataFrame(
        [
            {"eval_id": "eval-baseline", "model": models[0], "status": "success"},
            {"eval_id": "eval-candidate", "model": models[1], "status": "success"},
        ]
    )
    rows = []
    for eval_id, failures in (
        ("eval-baseline", baseline_failures),
        ("eval-candidate", candidate_failures),
    ):
        for index, prompt_id in enumerate(prompt_ids):
            rows.append(
                {
                    "eval_id": eval_id,
                    "id": prompt_id,
                    "epoch": 1,
                    score_column: "FAIL" if index < failures else "PASS",
                }
            )
    samples = pd.DataFrame(rows)
    monkeypatch.setattr("inspect_safety_stats.generative.logs.evals_df", lambda logs: evals)
    monkeypatch.setattr(
        "inspect_safety_stats.generative.logs.samples_df",
        lambda logs, parallel=False: samples,
    )


def test_compile_command_reports_the_taxonomy(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    result = runner.invoke(app, ["generative", "compile", "--plan", str(plan)])
    assert result.exit_code == 0
    # 3 seeds x 2 languages x 3 wrappers.
    assert "Terminal paths: 18" in result.stdout
    assert "Cohorts: baseline, candidate" in result.stdout


def test_materialize_writes_the_ledgers_and_dataset(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    output = tmp_path / "run"
    result = runner.invoke(
        app, ["generative", "materialize", "--plan", str(plan), "--output", str(output)]
    )
    assert result.exit_code == 0, result.stdout
    assert (output / "selections.csv").is_file()
    assert (output / "rewrites.csv").is_file()
    assert (output / "prompts.restricted.csv").is_file()
    dataset = output / "datasets" / "dataset.jsonl"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 200
    assert "How do I synthesize" not in (output / "rewrites.csv").read_text(encoding="utf-8")


def test_materialize_refuses_to_clobber_without_overwrite(tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    output = tmp_path / "run"
    assert materialize_run(plan, output).accepted == 200
    with pytest.raises(FileExistsError, match="not empty"):
        materialize_run(plan, output)
    assert materialize_run(plan, output, overwrite=True).accepted == 200


def test_analyze_produces_a_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts)

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    assert report.tree.root.draws == 200
    assert report.tree.root.failures == 20
    assert report.tree.root.estimate == pytest.approx(0.1)
    assert not report.has_errors

    path = report.write(tmp_path / "report")
    payload = json.loads((path / "report.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "generative-report/1.0"
    assert payload["evaluation"]["primary_cohort"] == "candidate"
    assert "not a probability over all possible prompts" in payload["estimand"]
    assert payload["root"]["draws"] == 200
    assert (path / "nodes.csv").is_file()
    assert (path / "taxonomy.csv").is_file()
    frame = pd.read_csv(path / "nodes.csv")
    assert set(frame["kind"]) == {"node"}


def test_text_artifacts_have_the_same_bytes_on_every_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Platform newline translation would make the dataset and report bytes differ by OS."""
    plan = write_plan(tmp_path)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts)
    path = analyze_run(plan, artifacts, "logs", cohort="candidate").write(tmp_path / "report")
    written = [
        artifacts / "selections.csv",
        *sorted((artifacts / "datasets").glob("*.jsonl")),
        *sorted(path.iterdir()),
    ]
    assert len(written) > 3
    for item in written:
        assert b"\r" not in item.read_bytes(), item.name


def test_analyze_command_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = write_plan(tmp_path)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts)

    result = runner.invoke(
        app,
        [
            "generative", "analyze",
            "--plan", str(plan),
            "--artifacts", str(artifacts),
            "--logs", str(artifacts),
            "--output", str(tmp_path / "report"),
            "--cohort", "candidate",
            "--slice", "language=maltese",
            "--compare", "candidate:baseline",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Root risk upper bound" in result.stdout
    assert "slice:language=maltese" in result.stdout
    assert "candidate - baseline" in result.stdout

    frame = pd.read_csv(tmp_path / "report" / "comparisons.csv")
    assert len(frame) == 1
    assert frame.loc[0, "method"] == "paired_binomial_cells"
    assert frame.loc[0, "estimate"] > 0


def test_a_plan_edited_between_the_phases_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ledger records what was evaluated; the plan must still reproduce it."""

    plan = write_plan(tmp_path)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts)

    edited = yaml.safe_load(plan.read_text(encoding="utf-8"))
    edited["taxonomy"]["arms"]["language"]["values"] = [
        {"id": "english", "weight": 0.9},
        {"id": "maltese", "weight": 0.1},
    ]
    plan.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer reproduces the recorded selection ledger"):
        analyze_run(plan, artifacts, "logs")


def test_an_unmatched_model_is_reported_not_silently_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = write_plan(tmp_path)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, models=(BASELINE_MODEL, "provider/something-else"))

    report = analyze_run(plan, artifacts, "logs", cohort="baseline")
    codes = [item.code for item in report.diagnostics]
    assert "GENERATIVE_EVAL_UNMATCHED" in codes
    # The candidate cohort is not part of this analysis, so it is not reported on at all.
    assert "GENERATIVE_COHORT_NOT_EVALUATED" not in codes
    assert report.tree.root.unclassified == 0


def test_a_requested_cohort_with_no_logs_is_named_plainly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Distinct from partial evaluation: nothing matched this cohort's model at all."""

    plan = write_plan(tmp_path)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, models=(BASELINE_MODEL, "provider/something-else"))

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    diagnostic = next(
        item for item in report.diagnostics if item.code == "GENERATIVE_COHORT_NOT_EVALUATED"
    )
    assert diagnostic.severity == "ERROR"
    assert CANDIDATE_MODEL in diagnostic.message
    assert report.has_errors


def test_a_missing_score_column_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = write_plan(tmp_path)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, score_column="score_something_else")

    report = analyze_run(plan, artifacts, "logs")
    assert report.has_errors
    assert any(item.code == "GENERATIVE_SCORE_ABSENT" for item in report.diagnostics)
    # Unclassified rather than dropped, so the bound still covers every draw.
    assert report.tree.root.unclassified == report.tree.root.trials


@pytest.mark.parametrize(
    ("specification", "expected"),
    [("fake", 0), ("fake:invalid_every=3", 3)],
)
def test_resolve_kernel_builds_the_fake(specification: str, expected: int) -> None:
    kernel = resolve_kernel(specification)
    assert kernel.kernel_id == "fake-rewriter"
    assert kernel.invalid_every == expected


def test_resolve_kernel_loads_an_import_path() -> None:
    kernel = resolve_kernel("inspect_safety_stats.generative.kernels:DeterministicFakeKernel")
    assert hasattr(kernel, "rewrite")


@pytest.mark.parametrize(
    "specification",
    ["nonsense", "fake:unknown=1", "inspect_safety_stats.generative.kernels:RewriteRequest"],
)
def test_resolve_kernel_rejects_what_it_cannot_use(specification: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        resolve_kernel(specification)


def test_generative_plan_schema_command() -> None:
    result = runner.invoke(app, ["schema", "generative-plan"])
    assert result.exit_code == 0
    assert '"GenerativePlan"' in result.stdout


def test_a_kernel_that_is_not_the_declared_one_is_flagged(tmp_path: Path) -> None:
    """The fingerprint hashes the plan, so the identifiers are the only available check."""

    plan = write_plan(tmp_path)
    run = materialize_run(plan, tmp_path / "run", kernel="fake")
    assert run.kernel_mismatch is not None
    assert "test-kernel" in run.kernel_mismatch
    assert "fake-rewriter" in run.kernel_mismatch

    result = runner.invoke(
        app,
        [
            "generative", "materialize",
            "--plan", str(plan),
            "--output", str(tmp_path / "run2"),
        ],
    )
    assert result.exit_code == 0
    assert "Warning: plan declares kernel" in result.output


def test_a_matching_kernel_is_not_flagged(tmp_path: Path) -> None:
    plan_path = write_plan(tmp_path)
    edited = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    edited["rewrite"]["kernel_id"] = "fake-rewriter"
    plan_path.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")
    run = materialize_run(plan_path, tmp_path / "run", kernel="fake")
    assert run.kernel_mismatch is None


def test_the_report_directory_carries_a_self_contained_html_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = write_plan(tmp_path)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts)

    report = analyze_run(
        plan,
        artifacts,
        "logs",
        cohort="candidate",
        slices=("language=maltese",),
        comparisons=("candidate:baseline",),
    )
    path = report.write(tmp_path / "report")
    html = (path / "report.html").read_text(encoding="utf-8")

    assert "__INSPECT_SAFETY_STATS_GENERATIVE_DATA__" not in html
    assert html.startswith("<!doctype html>")
    # Self-contained: nothing is fetched when the page is opened.
    for forbidden in ("<script src=", "<link rel=\"stylesheet\"", "http://", "https://"):
        assert forbidden not in html
    # The estimand is stated on the page itself, not only in the JSON.
    assert "not a probability over all possible prompts" in html
    assert report.evaluation["taxonomy_fingerprint"] in html


def test_the_html_page_escapes_data_that_could_close_its_script(tmp_path: Path) -> None:
    """Embedded JSON must not be able to terminate the script element."""

    from inspect_safety_stats.generative.report import GenerativeReport
    from inspect_safety_stats.generative.visualization import render_generative_report

    built = GenerativeReport(
        evaluation={"id": "x", "title": "</script><script>alert(1)</script>"},
        design={},
        tree=_minimal_tree(tmp_path),
        taxonomy=_minimal_taxonomy(tmp_path),
    )
    html = render_generative_report(built)
    assert "</script><script>alert(1)" not in html
    assert "\\u003c/script" in html or "\\u003cscript" in html


def _minimal_taxonomy(tmp_path: Path):
    from inspect_safety_stats.generative.schema import load_generative_plan
    from inspect_safety_stats.generative.taxonomy import compile_taxonomy

    plan_path = write_plan(tmp_path, total=20)
    return compile_taxonomy(load_generative_plan(plan_path), base_path=tmp_path)


def _minimal_tree(tmp_path: Path):
    from inspect_safety_stats.generative.inference import UnitOutcome, estimate_tree
    from inspect_safety_stats.generative.sampling import draw_selections, plan_design
    from inspect_safety_stats.generative.schema import load_generative_plan

    plan_path = write_plan(tmp_path, total=20)
    plan = load_generative_plan(plan_path)
    taxonomy = _minimal_taxonomy(tmp_path)
    design = plan_design(taxonomy, plan)
    selections = draw_selections(taxonomy, design)
    outcomes = [UnitOutcome(r.sampling_unit_id, 1, 0, 0) for r in selections]
    return estimate_tree(taxonomy, design, selections, outcomes, report_all_nodes=False)


def test_the_report_carries_allocation_ess_and_the_raw_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Section 25's tenth completion criterion names allocation and ESS explicitly."""

    plan_path = write_plan(tmp_path)
    edited = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    edited["sampling"] = {
        "method": "proportional_stratified",
        "total_rewrites": 200,
        "minimum_rewrites_per_stratum": 50,
    }
    plan_path.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")
    artifacts = tmp_path / "run"
    materialize_run(plan_path, artifacts)
    stub_logs(monkeypatch, artifacts)

    report = analyze_run(
        plan_path, artifacts, "logs", cohort="candidate", slices=("language=maltese",)
    )
    payload = report.to_dict()

    allocation = payload["design"]["allocation"]
    assert len(allocation) == 3
    assert sum(item["draws"] for item in allocation) == 200
    assert all(item["draws"] >= 50 for item in allocation)

    root = payload["root"]
    assert root["effective_sample_size"] < root["draws"]      # the floor concentrates weight
    assert root["raw_failure_rate"] is not None
    assert root["raw_failure_rate"] != pytest.approx(root["estimate"])

    html = (report.write(tmp_path / "report") / "report.html").read_text(encoding="utf-8")
    assert "Effective sample size" in html
    assert "raw sampled rate" in html
    assert "Draws if exactly proportional" in html


def test_budget_command_projects_before_anything_is_spent(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, total=1200)
    result = runner.invoke(
        app,
        ["generative", "budget", "--plan", str(plan),
         "--assumed-rate", "0.03", "--threshold", "0.05", "--target-bound", "0.04"],
    )
    assert result.exit_code == 0, result.stdout
    assert "root bound" in result.stdout
    assert "At a budget of 1200, by level:" in result.stdout
    assert "certifying root risk below 0.05" in result.stdout
    assert "needs about" in result.stdout
    # Nothing was generated: the command reads only the plan.
    assert not (tmp_path / "rewrites.csv").exists()
    assert not (tmp_path / "datasets").exists()


def test_budget_command_details_the_budget_the_caller_asked_for(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, total=1200)
    result = runner.invoke(
        app,
        ["generative", "budget", "--plan", str(plan), "--budget", "300", "--budget", "900"],
    )
    assert result.exit_code == 0, result.stdout
    assert "At a budget of 900, by level:" in result.stdout
    assert "1200" not in result.stdout.split("by level:")[1]


def test_budget_command_flags_an_unaffordable_stratum_floor(tmp_path: Path) -> None:
    plan_path = write_plan(tmp_path)
    edited = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    edited["sampling"] = {
        "method": "proportional_stratified",
        "total_rewrites": 600,
        "minimum_rewrites_per_stratum": 150,
    }
    plan_path.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")

    result = runner.invoke(
        app, ["generative", "budget", "--plan", str(plan_path), "--budget", "100"]
    )
    assert result.exit_code == 0, result.stdout
    assert "infeasible" in result.stdout


def test_a_kernel_can_be_named_by_file_path(tmp_path: Path) -> None:
    """The form a local kernel needs: no __init__.py, no PYTHONPATH."""

    module = tmp_path / "my_kernel.py"
    module.write_text(
        "from inspect_safety_stats.generative import RewriteOutcome\n\n\n"
        "class Mine:\n"
        "    kernel_id = 'mine'\n\n"
        "    def rewrite(self, request):\n"
        "        return RewriteOutcome(status='valid', prompt='from a file')\n\n\n"
        "def build():\n"
        "    return Mine()\n",
        encoding="utf-8",
    )
    kernel = resolve_kernel(f"{module}:build")
    assert kernel.kernel_id == "mine"


def test_the_shipped_live_example_kernel_resolves_as_documented() -> None:
    """Exactly the string the runbook tells a reader to type."""

    root = Path(__file__).resolve().parents[2]
    kernel = resolve_kernel(
        f"{root / 'examples' / 'generative_live' / 'rewriter.py'}:build_kernel"
    )
    assert hasattr(kernel, "rewrite")
    assert kernel.kernel_id == "inspect-paraphrase-v1"


def test_a_missing_kernel_file_or_attribute_says_which(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        resolve_kernel(f"{tmp_path / 'absent.py'}:build")

    module = tmp_path / "empty_kernel.py"
    module.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no attribute 'build'"):
        resolve_kernel(f"{module}:build")


def test_an_unimportable_module_suggests_the_file_form() -> None:
    with pytest.raises(ValueError, match="path/to/file.py:build"):
        resolve_kernel("no.such.module:build")


def test_provider_errors_are_reported_as_such_not_as_validation(tmp_path: Path) -> None:
    """A missing API key must not read as a validator problem."""

    import inspect_safety_stats.generative.runner as runner_module
    from inspect_safety_stats.generative import RewriteOutcome

    class Broken:
        kernel_id = "broken"

        def rewrite(self, request):
            return RewriteOutcome(status="error", detail="no credentials configured")

    plan = write_plan(tmp_path, total=20)
    original = runner_module.resolve_kernel
    runner_module.resolve_kernel = lambda spec: Broken()  # type: ignore[assignment]
    try:
        result = runner.invoke(
            app,
            ["generative", "materialize", "--plan", str(plan),
             "--output", str(tmp_path / "run")],
        )
    finally:
        runner_module.resolve_kernel = original  # type: ignore[assignment]

    assert result.exit_code == 3
    assert "provider error" in result.output
    assert "no credentials configured" in result.output
    assert "No rewrite passed validation" not in result.output
