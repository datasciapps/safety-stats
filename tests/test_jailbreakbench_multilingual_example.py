from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from safety_stats import analyze, load_plan

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "real_jailbreakbench"
sys.path.insert(0, str(EXAMPLE_DIR))

import run_multilingual_jailbreakbench as runner  # noqa: E402
from jailbreakbench_live import behavior_rows  # noqa: E402
from jailbreakbench_multilingual import (  # noqa: E402
    LANGUAGE_BY_ID,
    LANGUAGES,
    extend_translation_artifact,
    jailbreakbench_100_multilingual,
    load_translation_artifact,
    metadata_path,
    multilingual_samples,
    multilingual_task_name,
    write_translation_artifact,
)
from run_jailbreakbench import write_design  # noqa: E402


def _write_test_translations(destination: Path) -> Path:
    def translate(values: list[str], language_code: str) -> list[str]:
        return [f"{language_code}:{index}" for index, _ in enumerate(values)]

    return write_translation_artifact(
        destination,
        translate=translate,
        translation_model="test/translator",
        requested_revision="test-revision",
        resolved_revision="test-commit",
        generation_config={"do_sample": False, "num_beams": 1},
        batch_size=100,
    )


def test_frozen_translation_artifact_has_exact_language_prompt_cross_product(
    tmp_path: Path,
) -> None:
    artifact = _write_test_translations(tmp_path / "translations.csv")
    values, metadata = load_translation_artifact(artifact)

    assert set(values) == {language.id for language in LANGUAGES}
    assert {len(language_rows) for language_rows in values.values()} == {100}
    assert metadata["row_count"] == 400
    assert metadata["translation"]["resolved_revision"] == "test-commit"
    assert values["english"]["jbb-000"] == behavior_rows()[0]["Goal"]
    assert values["french"]["jbb-000"].startswith("fra_Latn:")
    assert values["german"]["jbb-000"].startswith("deu_Latn:")
    assert values["maltese"]["jbb-000"].startswith("mlt_Latn:")

    french = multilingual_samples(artifact, "french")
    assert len(french) == 100
    assert {str(sample.id) for sample in french} == {f"jbb-{index:03d}" for index in range(100)}
    assert {sample.metadata["language_id"] for sample in french} == {"french"}


def test_translation_artifact_checksum_detects_modification(tmp_path: Path) -> None:
    artifact = _write_test_translations(tmp_path / "translations.csv")
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        load_translation_artifact(artifact)


def test_language_tasks_have_distinct_names_and_common_semantic_ids(tmp_path: Path) -> None:
    artifact = _write_test_translations(tmp_path / "translations.csv")
    tasks = {
        language.id: jailbreakbench_100_multilingual(
            grader_model="mockllm/grader",
            translation_artifact=str(artifact),
            language_id=language.id,
        )
        for language in LANGUAGES
    }
    assert {task.name for task in tasks.values()} == {
        "jailbreakbench_100_english",
        "jailbreakbench_100_french",
        "jailbreakbench_100_german",
        "jailbreakbench_100_maltese",
    }
    assert all(len(task.dataset) == 100 for task in tasks.values())
    assert [str(sample.id) for sample in tasks["english"].dataset] == [
        str(sample.id) for sample in tasks["german"].dataset
    ]


def test_multilingual_plan_uses_one_model_and_all_pair_language_contrasts(
    tmp_path: Path,
) -> None:
    write_design(
        tmp_path / "design.csv",
        rollouts=8,
        attack_name="translated_direct_request",
    )
    runner.write_plan(
        tmp_path / "analysis-plan.yaml",
        target_model="model/target",
        run_id="jbb-multilingual-test",
        frozen_at="2026-08-22T12:00:00+00:00",
        translation_artifact_sha256="a" * 64,
    )
    plan = load_plan(tmp_path / "analysis-plan.yaml")

    assert plan.design.hierarchy == ["category_id", "prompt_id"]
    assert {cohort.match["model"] for cohort in plan.cohorts.values()} == {"model/target"}
    assert {cohort.match["task_name"] for cohort in plan.cohorts.values()} == {
        multilingual_task_name(language.id) for language in LANGUAGES
    }
    comparison_set = plan.comparison_sets[0]
    assert comparison_set.contrasts.type == "all_pairs"
    assert comparison_set.cohorts == ["english", "french", "german", "maltese"]
    assert len(plan.expanded_comparisons()) == 18
    assert all(item.hierarchy_nodes for item in plan.expanded_comparisons())
    assert plan.report.include_prompt_text is False
    assert plan.report.include_decisions is False


def test_log_selection_distinguishes_same_model_by_language_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_design(
        tmp_path / "design.csv",
        rollouts=2,
        attack_name="translated_direct_request",
    )
    runner.write_plan(
        tmp_path / "analysis-plan.yaml",
        target_model="model/target",
        run_id="jbb-multilingual-logs",
        frozen_at="2026-08-22T12:00:00+00:00",
        translation_artifact_sha256="b" * 64,
    )
    infos = [
        SimpleNamespace(name="english-old.eval", mtime=1.0),
        SimpleNamespace(name="english-new.eval", mtime=4.0),
        SimpleNamespace(name="french.eval", mtime=2.0),
        SimpleNamespace(name="german.eval", mtime=3.0),
        SimpleNamespace(name="maltese.eval", mtime=5.0),
    ]
    headers = {
        "english-old.eval": SimpleNamespace(
            status="success",
            eval=SimpleNamespace(model="model/target", task="jailbreakbench_100_english"),
        ),
        "english-new.eval": SimpleNamespace(
            status="success",
            eval=SimpleNamespace(model="model/target", task="jailbreakbench_100_english"),
        ),
        "french.eval": SimpleNamespace(
            status="success",
            eval=SimpleNamespace(model="model/target", task="jailbreakbench_100_french"),
        ),
        "german.eval": SimpleNamespace(
            status="success",
            eval=SimpleNamespace(model="model/target", task="jailbreakbench_100_german"),
        ),
        "maltese.eval": SimpleNamespace(
            status="success",
            eval=SimpleNamespace(model="model/target", task="jailbreakbench_100_maltese"),
        ),
    }
    monkeypatch.setattr(runner, "list_eval_logs", lambda *args, **kwargs: infos)
    monkeypatch.setattr(
        runner,
        "read_eval_log",
        lambda info, header_only=True: headers[info.name],
    )
    assert runner.latest_successful_logs(tmp_path) == [
        "english-new.eval",
        "french.eval",
        "german.eval",
        "maltese.eval",
    ]


def _offline_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluations = []
    sample_rows = []
    for language_id, divisor in (
        ("english", 4),
        ("french", 5),
        ("german", 10),
        ("maltese", 20),
    ):
        eval_id = f"eval-{language_id}"
        log_path = f"/not/a/local/{language_id}.eval"
        evaluations.append(
            {
                "eval_id": eval_id,
                "run_id": f"run-{language_id}",
                "log": log_path,
                "created": "2026-08-22T12:00:00Z",
                "status": "success",
                "task_name": multilingual_task_name(language_id),
                "task_version": "1.0",
                "model": "model/target",
                "model_generate_config": {"temperature": 1.0, "max_tokens": 512},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": f"jbb-{language_id}",
                "epochs": 2,
            }
        )
        for index in range(100):
            prompt_id = f"jbb-{index:03d}"
            failure = index % divisor == 0
            for epoch in (1, 2):
                sample_rows.append(
                    {
                        "sample_id": f"{language_id}-{prompt_id}-{epoch}",
                        "eval_id": eval_id,
                        "log": log_path,
                        "id": prompt_id,
                        "epoch": epoch,
                        "score_jbb_harmful_compliance": "FAIL" if failure else "PASS",
                        "error": None,
                        "retries": 0,
                    }
                )
    return pd.DataFrame(evaluations), pd.DataFrame(sample_rows)


def test_multilingual_offline_analysis_and_unified_visualizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_design(
        tmp_path / "design.csv",
        rollouts=2,
        attack_name="translated_direct_request",
    )
    runner.write_plan(
        tmp_path / "analysis-plan.yaml",
        target_model="model/target",
        run_id="jbb-multilingual-offline",
        frozen_at="2026-08-22T12:00:00+00:00",
        translation_artifact_sha256="c" * 64,
    )
    eval_frame, sample_frame = _offline_frames()
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.evals_df",
        lambda logs, **_: eval_frame,
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=tmp_path / "analysis-plan.yaml", logs="ignored")
    assert report.integrity.status == "valid"
    assert len(report.prompt_estimates) == 400
    assert len(report.group_risks) == 16
    assert len(report.hierarchy_risks) == 40
    assert len(report.comparisons) == 198
    assert report.comparison_sets[0].contrast_type == "all_pairs"
    assert set(report.comparison_sets[0].rank_intervals) == {
        "english",
        "french",
        "german",
        "maltese",
    }

    output = report.write(tmp_path / "report")
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "language_conditions" in html
    assert "english" in html and "french" in html and "german" in html
    assert "maltese" in html
    assert "Within configuration" in html and "Across configurations" in html
    first_goal = behavior_rows()[0]["Goal"]
    for name in ("report.html", "report.md", "report.json"):
        assert first_goal not in (output / name).read_text(encoding="utf-8")


def test_translation_sidecar_is_content_free(tmp_path: Path) -> None:
    artifact = _write_test_translations(tmp_path / "translations.csv")
    sidecar_text = metadata_path(artifact).read_text(encoding="utf-8")
    assert behavior_rows()[0]["Goal"] not in sidecar_text


def test_translation_artifact_can_append_maltese_without_changing_existing_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "three.csv"

    def translate(values: list[str], language_code: str) -> list[str]:
        return [f"{language_code}:{index}" for index, _ in enumerate(values)]

    write_translation_artifact(
        source,
        translate=translate,
        translation_model="test/translator",
        requested_revision="old",
        resolved_revision="old-commit",
        generation_config={"do_sample": False},
        languages=LANGUAGES[:3],
        batch_size=100,
    )
    original_rows = source.read_text(encoding="utf-8").splitlines()[1:]
    destination = tmp_path / "four.csv"
    extend_translation_artifact(
        source,
        destination,
        language=LANGUAGE_BY_ID["maltese"],
        translate=translate,
        translation_model="test/translator",
        requested_revision="new",
        resolved_revision="new-commit",
        generation_config={"do_sample": False},
        batch_size=100,
    )
    extended_rows = destination.read_text(encoding="utf-8").splitlines()[1:]
    _, metadata = load_translation_artifact(destination)
    assert extended_rows[:300] == original_rows
    assert len(extended_rows) == 400
    assert metadata["parent_artifact_sha256"] is not None
    assert metadata["languages"][-1]["id"] == "maltese"
