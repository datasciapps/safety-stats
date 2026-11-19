"""No value of a run's configuration read from Inspect logs reaches an output.

The package fingerprints each cohort's run configuration to detect drift between its runs
(``configuration_fingerprints`` in report.json, ``configuration_fingerprint`` in
observations.csv, ``CONFIG_DRIFT``) and copies none of it. These tests plant a distinct secret
in every configuration field of real logs written offline (tests/planted_configuration.py), run
each command that reads them, and look for the secrets in every file it writes and every line
it prints. The generative workflow's test is in test_generative_real_logs.py.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from inspect_ai.log import read_eval_log, write_eval_log
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from inspect_safety_stats.ingest.inspect_logs import read_inspect_runs
from inspect_safety_stats.scaffold import scaffold_plan
from tests.mock_inspect_logs import write_mock_logs
from tests.planted_configuration import (
    PLANTED_FIELDS,
    leaks,
    plant_configuration,
    written_texts,
)


@pytest.fixture(autouse=True)
def _run_from_the_drive_that_holds_the_logs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work from the drive of pytest's temporary root, where these tests write real logs.

    On Windows, inspect-ai drops the drive letter from local log paths, so it only finds a
    log on the working directory's drive (#35). Every path below is absolute.
    """
    monkeypatch.chdir(tmp_path_factory.getbasetemp())


@dataclass(frozen=True)
class PlantedLogs:
    # The mock logs as Inspect wrote them, which the plans are written from.
    clean: Path
    # The same evaluations with a secret in every configuration field. Cohort alpha's
    # evaluation is split into one per epoch, each with its own secrets, so the cohort drifts.
    planted: Path
    secrets: tuple[str, ...]


@pytest.fixture(scope="module")
def planted_logs(tmp_path_factory: pytest.TempPathFactory) -> PlantedLogs:
    root = tmp_path_factory.mktemp("planted")
    clean = write_mock_logs(root / "clean", epochs=2)
    planted = root / "logs"
    planted.mkdir()
    secrets: list[str] = []
    for path in sorted(clean.glob("*.eval")):
        log = read_eval_log(str(path))
        model = log.eval.model.removeprefix("mockllm/")
        if model == "alpha":
            for epoch in (1, 2):
                part = log.model_copy(deep=True)
                part.eval.eval_id = f"{log.eval.eval_id}-{epoch}"
                part.samples = [sample for sample in log.samples or [] if sample.epoch == epoch]
                secrets += plant_configuration(part, f"alpha{epoch}")
                write_eval_log(part, str(planted / f"alpha-{epoch}.eval"))
        else:
            secrets += plant_configuration(log, model)
            write_eval_log(log, str(planted / f"{model}.eval"))
    return PlantedLogs(clean=clean, planted=planted, secrets=tuple(secrets))


def test_the_reader_returns_every_planted_secret(planted_logs: PlantedLogs) -> None:
    """Without this, a field Inspect redacted or left out would pass the tests below vacuously."""

    runs = read_inspect_runs(str(planted_logs.planted))
    assert len(runs) == 3
    recorded = json.dumps([run.attributes for run in runs], default=str)
    assert [secret for secret in planted_logs.secrets if secret not in recorded] == []
    for field in PLANTED_FIELDS:
        assert all("planted-" in str(run.attributes[field]) for run in runs), field


@dataclass(frozen=True)
class Analyzed:
    # report.json as written.
    report: dict[str, Any]
    # Every file analyze wrote and what analyze and validate printed, by name.
    texts: dict[str, str]


def _analyze(
    planted_logs: PlantedLogs, directory: Path, edit: Callable[[dict[str, Any]], None]
) -> Analyzed:
    """Run analyze and validate on the planted logs, with init's plan changed by ``edit``."""

    result = scaffold_plan(
        planted_logs.clean,
        directory / "plan",
        failure_values=["FAIL"],
        group_by="category",
        baseline="mockllm/alpha",
    )
    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    edit(plan)
    result.plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    logs = str(planted_logs.planted)
    output = directory / "report"
    runner = CliRunner()
    analyzed = runner.invoke(
        app,
        ["analyze", "--plan", str(result.plan_path), "--logs", logs, "--output", str(output),
         "--allow-invalid"],
    )
    assert analyzed.exit_code == 0, analyzed.output
    validated = runner.invoke(app, ["validate", "--plan", str(result.plan_path), "--logs", logs])
    texts = written_texts(output)
    assert {
        "report.json",
        "report.md",
        "report-full.md",
        "report.html",
        "observations.csv",
        "index.json",
        "index.md",
        "checksums.json",
    } <= {Path(name).name for name in texts}
    texts["analyze output"] = analyzed.output
    texts["validate output"] = validated.output
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    return Analyzed(report=report, texts=texts)


def _messages(analyzed: Analyzed, code: str) -> list[str]:
    return [
        item["message"]
        for item in analyzed.report["integrity"]["diagnostics"]
        if item["code"] == code
    ]


def _require_every_planted_field(plan: dict[str, Any]) -> None:
    # Every planted field enters each cohort's configuration, and a judge on the primary score
    # records each cohort's model roles.
    for cohort in plan["cohorts"].values():
        cohort["require_uniform"] = [*cohort["require_uniform"], *PLANTED_FIELDS]
    plan["inputs"]["judges"] = [
        {
            "id": "judge",
            "score": {
                "name": plan["inputs"]["score"]["name"],
                "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
            },
        }
    ]
    plan["outputs"]["judge_claims"] = True


def test_a_drifting_cohort_shows_fingerprints_and_no_planted_secret(
    planted_logs: PlantedLogs, tmp_path: Path
) -> None:
    analyzed = _analyze(planted_logs, tmp_path, _require_every_planted_field)

    assert leaks(analyzed.texts, planted_logs.secrets) == []
    assert _messages(analyzed, "CONFIG_DRIFT") == [
        "cohort 'alpha' contains incompatible evaluation configurations"
    ]
    assert "CONFIG_DRIFT" in analyzed.texts["validate output"]
    # What the report records instead identifies each configuration without showing it.
    fingerprints = {
        item["cohort_id"]: item["configuration_fingerprints"]
        for item in analyzed.report["cohorts"]
    }
    assert [len(fingerprints["alpha"]), len(fingerprints["beta"])] == [2, 1]
    (judge,) = analyzed.report["provenance"]["judge_panel"]["judges"]
    assert judge["configuration"]["beta"]["model_roles"] == [
        {"grader": {"model": "mockllm/grader"}}
    ]


def test_init_copies_no_planted_secret_into_the_plan_or_the_terminal(
    planted_logs: PlantedLogs, tmp_path: Path
) -> None:
    # One evaluation per model: init refuses two of the same model.
    logs = [planted_logs.planted / "alpha-1.eval", planted_logs.planted / "beta.eval"]
    result = CliRunner().invoke(
        app,
        ["init", *[argument for log in logs for argument in ("--logs", str(log))],
         "--failure-values", "FAIL", "--output", str(tmp_path / "plan")],
    )
    assert result.exit_code == 0, result.output
    texts = written_texts(tmp_path / "plan")
    assert {"safety-plan.yaml", "design.csv"} <= {Path(name).name for name in texts}
    texts["init output"] = result.output
    assert leaks(texts, planted_logs.secrets) == []
