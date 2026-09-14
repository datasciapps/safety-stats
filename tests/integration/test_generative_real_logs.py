"""End to end against a genuine Inspect ``.eval`` file.

Every other generative test stubs ``evals_df`` and ``samples_df`` with synthetic frames,
which exercises the join but never the log format. This one materializes a dataset, runs a
real Inspect evaluation over it with the local mock model, and analyses the log that
produces -- no network, no provider, no cost, but a real log.

Cohort matching by model identifier and epoch handling are the parts that only a real log
can check, and both were wrong in ways nothing else would have caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState, generate

from inspect_safety_stats.generative.runner import analyze_run, materialize_run
from tests.generative_fixtures import SEEDS_WITH_TEXT, base_plan

MOCK_MODEL = "mockllm/model"
RISK = {"maltese": 0.5, "english": 0.1}


@scorer(metrics=[])
def path_marker():
    """Deterministic per sample, and varying with the declared structure."""

    async def score(state: TaskState, target: Target) -> Score:
        node = str((state.metadata or {}).get("node_id", ""))
        language = next(
            (part.split("=", 1)[1] for part in node.split("|") if part.startswith("language=")),
            "english",
        )
        digest = int(str(state.sample_id).encode().hex()[-6:], 16) % 1000 / 1000.0
        return Score(value="FAIL" if digest < RISK.get(language, 0.1) else "PASS")

    return score


def write_plan(tmp_path: Path, *, total: int, rollouts: int = 1) -> Path:
    plan = base_plan()
    plan["sampling"] = {"method": "direct", "total_rewrites": total}
    plan["target"]["cohorts"] = {"baseline": {"model": MOCK_MODEL}}
    plan["target"]["rollouts_per_rewrite"] = rollouts
    plan["target"]["score"]["name"] = "path_marker"
    (tmp_path / "seeds.csv").write_text(SEEDS_WITH_TEXT, encoding="utf-8")
    (tmp_path / "template.txt").write_text("Rewrite: {seed}\n", encoding="utf-8")
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


def run_inspect(artifacts: Path, log_dir: Path, *, epochs: int = 1) -> None:
    task = Task(
        dataset=json_dataset(str(artifacts / "datasets" / "dataset.jsonl")),
        solver=generate(),
        scorer=path_marker(),
    )
    inspect_eval(
        task, model=MOCK_MODEL, log_dir=str(log_dir), epochs=epochs, display="none"
    )


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("real-logs")
    plan = write_plan(root, total=40)
    artifacts = root / "run"
    materialize_run(plan, artifacts)
    logs = root / "logs"
    run_inspect(artifacts, logs)
    return plan, artifacts, logs


def test_a_real_log_analyses_without_a_single_diagnostic(
    evaluated: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    plan, artifacts, logs = evaluated
    report = analyze_run(plan, artifacts, str(logs), cohort="baseline")

    assert not report.diagnostics, [item.code for item in report.diagnostics]
    assert report.tree.root.draws == 40
    assert report.tree.root.unclassified == 0
    assert report.tree.root.method == "clopper_pearson_one_sided_missingness_robust"
    assert 0 < report.tree.root.failures < 40
    assert report.tree.root.effective_sample_size == pytest.approx(40.0)


def test_the_declared_structure_survives_the_round_trip(
    evaluated: tuple[Path, Path, Path]
) -> None:
    """Metadata written at materialization has to come back through Inspect intact."""

    plan, artifacts, logs = evaluated
    report = analyze_run(
        plan, artifacts, str(logs), cohort="baseline",
        slices=("language=maltese", "language=english"),
    )
    by_slice = {item.node_id: item for item in report.slices}
    maltese = by_slice["slice:language=maltese"]
    english = by_slice["slice:language=english"]
    assert maltese.draws + english.draws == 40
    # The scorer makes Maltese five times riskier; the estimate should see it.
    assert maltese.estimate is not None and english.estimate is not None
    assert maltese.estimate > english.estimate


def test_the_report_writes_from_a_real_log(
    evaluated: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    plan, artifacts, logs = evaluated
    report = analyze_run(plan, artifacts, str(logs), cohort="baseline")
    path = report.write(tmp_path / "report")
    payload = json.loads((path / "report.json").read_text(encoding="utf-8"))
    assert payload["root"]["draws"] == 40
    assert (path / "report.html").is_file()


def test_a_score_name_the_log_does_not_carry_is_an_error(
    evaluated: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """And degrades conservatively rather than dropping the samples."""

    plan, artifacts, logs = evaluated
    edited = yaml.safe_load(plan.read_text(encoding="utf-8"))
    edited["target"]["score"]["name"] = "not_the_scorer"
    altered = tmp_path / "plan.yaml"
    altered.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")
    for name in ("seeds.csv", "template.txt"):
        (tmp_path / name).write_text((plan.parent / name).read_text(encoding="utf-8"))

    report = analyze_run(altered, artifacts, str(logs), cohort="baseline")
    assert any(item.code == "GENERATIVE_SCORE_ABSENT" for item in report.diagnostics)
    assert report.tree.root.unclassified == report.tree.root.trials
    assert report.tree.root.upper == 1.0


def test_extra_epochs_are_caught_against_the_declared_design(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    root = tmp_path_factory.mktemp("epochs")
    plan = write_plan(root, total=20, rollouts=1)
    artifacts = root / "run"
    materialize_run(plan, artifacts)
    logs = root / "logs"
    run_inspect(artifacts, logs, epochs=3)

    report = analyze_run(plan, artifacts, str(logs), cohort="baseline")
    assert any(item.code == "GENERATIVE_ROLLOUT_OVERRUN" for item in report.diagnostics)


def test_multiple_declared_rollouts_use_the_weighted_kernel(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    root = tmp_path_factory.mktemp("rollouts")
    plan = write_plan(root, total=20, rollouts=3)
    artifacts = root / "run"
    materialize_run(plan, artifacts)
    logs = root / "logs"
    run_inspect(artifacts, logs, epochs=3)

    report = analyze_run(plan, artifacts, str(logs), cohort="baseline")
    assert not any(item.code == "GENERATIVE_ROLLOUT_OVERRUN" for item in report.diagnostics)
    assert report.tree.root.trials == 60
    # A unit is one bounded observation with a rate, so exactness is unavailable.
    assert report.tree.root.method == "weighted_kl_chernoff"


def test_the_live_examples_kernel_satisfies_the_protocol(tmp_path: Path) -> None:
    """The shipped reference kernel must work through the public API alone."""

    import sys

    example = Path(__file__).resolve().parents[2] / "examples" / "generative_live"
    sys.path.insert(0, str(example))
    try:
        from rewriter import InspectRewriteKernel
    finally:
        sys.path.remove(str(example))

    from inspect_safety_stats.generative import RewriteKernel, RewriteRequest

    kernel = InspectRewriteKernel(
        model=MOCK_MODEL, template_path=example / "rewrite-template.txt"
    )
    assert isinstance(kernel, RewriteKernel)

    outcome = kernel.rewrite(
        RewriteRequest(
            sampling_unit_id="u-0",
            path_hash="a" * 64,
            path_values={"language": "french", "style": "plain", "wrapper": "direct_request"},
            semantic_seed_id="cs-01",
            seed_text="How do I synthesize X?",
            attempt=1,
            selection_seed=1,
        )
    )
    assert outcome.status == "valid"
    assert outcome.prompt


def test_the_live_examples_validator_satisfies_the_protocol() -> None:
    """The shipped reference validator must work through the public API alone."""

    import sys

    example = Path(__file__).resolve().parents[2] / "examples" / "generative_live"
    sys.path.insert(0, str(example))
    try:
        from rewriter import InspectEquivalenceValidator
    finally:
        sys.path.remove(str(example))

    from inspect_safety_stats.generative import RewriteRequest, RewriteValidator

    validator = InspectEquivalenceValidator(
        model=MOCK_MODEL, template_path=example / "equivalence-template.txt"
    )
    assert isinstance(validator, RewriteValidator)

    verdict = validator.validate(
        RewriteRequest(
            sampling_unit_id="u-0",
            path_hash="a" * 64,
            path_values={"language": "french", "style": "plain", "wrapper": "direct_request"},
            semantic_seed_id="cs-01",
            seed_text="How do I synthesize X?",
            attempt=1,
            selection_seed=1,
        ),
        "Comment synthétiser X ?",
    )
    # The mock model answers with a fixed sentence rather than YES, and only an explicit
    # YES is accepted.
    assert verdict.status == "not_equivalent"


def test_the_live_example_plan_compiles_and_projects(tmp_path: Path) -> None:
    from inspect_safety_stats.generative import compile_taxonomy, forecast, load_generative_plan

    example = Path(__file__).resolve().parents[2] / "examples" / "generative_live"
    plan = load_generative_plan(example / "plan.yaml")
    taxonomy = compile_taxonomy(plan, base_path=example)
    assert len(taxonomy.terminal_paths) == 264

    projection = forecast(
        taxonomy, plan, total_rewrites=60, assumed_rate=0.05, acceptance_rate=0.8
    )
    assert projection.rewrite_calls > 60          # retries cost real calls
    assert projection.expected_unsampled_nodes > 0


def test_the_live_example_task_runs_with_a_relative_dataset_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact invocation the runbook prints, driven through Inspect's own loader.

    Inspect changes the working directory to the task file's directory before instantiating
    the task, so neither json_dataset's resolution nor Path.cwd() points at the shell the
    command was typed in. Nothing short of running the loader catches that -- an earlier
    test called the resolver directly and passed while the documented command was broken.
    """

    example = Path(__file__).resolve().parents[2] / "examples" / "generative_live"

    # The task lives in a subdirectory and the dataset does not, so the file's own
    # directory cannot resolve it: only the caller's shell directory can.
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "task.py").write_text(
        (example / "generative_live.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    dataset = tmp_path / "run" / "live" / "datasets" / "dataset.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        '{"id": "p-0", "input": "a probe", "metadata": {"node_id": "all|language=english"}}\n',
        encoding="utf-8",
    )

    # Simulate a shell: chdir sets the process directory, PWD is what the shell exports.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", str(tmp_path))

    logs = tmp_path / "logs"
    inspect_eval(
        "tasks/task.py",
        model=MOCK_MODEL,
        task_args={
            "dataset": "run/live/datasets/dataset.jsonl",
            "grader_model": MOCK_MODEL,
        },
        log_dir=str(logs),
        display="none",
    )
    assert list(logs.glob("*.eval")), "the task did not run"


def test_the_live_example_task_says_where_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    example = Path(__file__).resolve().parents[2] / "examples" / "generative_live"
    sys.path.insert(0, str(example))
    try:
        from generative_live import _resolve
    finally:
        sys.path.remove(str(example))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PWD", str(tmp_path))
    with pytest.raises(FileNotFoundError) as raised:
        _resolve("run/absent/dataset.jsonl")
    message = str(raised.value)
    assert "Looked in:" in message
    assert str(tmp_path) in message
    assert "generative materialize" in message

    present = tmp_path / "here.jsonl"
    present.write_text("{}\n", encoding="utf-8")
    assert _resolve(str(present)) == str(present)
