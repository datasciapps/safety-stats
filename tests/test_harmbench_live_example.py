import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

from inspect_safety_stats.config import load_plan
from inspect_safety_stats.ingest.manifest import load_manifest, resolve_groups

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "real_harmbench"
sys.path.insert(0, str(EXAMPLE_DIR))

from harmbench_live import (  # noqa: E402
    _behavior_rows,
    _samples,
    harmbench_direct_request,
    harmful_behavior_failure,
)
from run_openai import _write_design  # noqa: E402

OFFICIAL_TEXT_ALL_SHA256 = "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc"


def test_vendored_harmbench_data_matches_official_checksum() -> None:
    data = (EXAMPLE_DIR / "harmbench_behaviors_text_all.csv").read_bytes()
    assert sha256(data).hexdigest() == OFFICIAL_TEXT_ALL_SHA256


def test_harmbench_full_text_dataset_and_hierarchy() -> None:
    rows = _behavior_rows()
    samples = _samples()
    assert len(rows) == 400
    assert len(samples) == 400
    assert len({str(sample.id) for sample in samples}) == 400

    functional_categories = Counter(
        str(sample.metadata["functional_category"]) for sample in samples
    )
    assert functional_categories == {
        "standard": 200,
        "contextual": 100,
        "copyright": 100,
    }

    domains = [str(sample.metadata["semantic_category"]) for sample in samples]
    assert Counter(domains) == {
        "chemical_biological": 56,
        "copyright": 100,
        "cybercrime_intrusion": 67,
        "harassment_bullying": 25,
        "harmful": 22,
        "illegal": 65,
        "misinformation_disinformation": 65,
    }
    assert Counter(str(sample.metadata["source_split"]) for sample in samples) == {
        "text_test": 320,
        "text_val": 80,
    }
    assert len(harmbench_direct_request(grader_model="mockllm/grader").dataset) == 400


def test_contextual_direct_requests_match_harmbench_format() -> None:
    contextual = next(
        sample for sample in _samples() if sample.metadata["functional_category"] == "contextual"
    )
    standard = next(
        sample for sample in _samples() if sample.metadata["functional_category"] == "standard"
    )
    assert "\n\n---\n\n" in str(contextual.input)
    assert "\n\n---\n\n" not in str(standard.input)


def test_generated_design_has_one_question_per_behavior(tmp_path: Path) -> None:
    import csv

    destination = tmp_path / "design.csv"
    _write_design(destination, rollouts=3)
    with destination.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 400
    assert all(row["prompt_id"] == row["question_id"] for row in rows)
    assert Counter(row["task_id"] for row in rows) == {
        "harmbench-harmful-compliance": 300,
        "harmbench-copyright-reproduction": 100,
    }
    assert Counter(row["assessment_family"] for row in rows) == {
        "harmful_behavior": 300,
        "copyright": 100,
    }
    assert {row["expected_rollouts"] for row in rows} == {"3"}
    assert Counter(row["functional_category"] for row in rows) == {
        "standard": 200,
        "contextual": 100,
        "copyright": 100,
    }


def test_primary_safety_aggregate_balances_six_harm_domains() -> None:
    plan = load_plan(EXAMPLE_DIR / "live-plan-template.yaml")
    direct = plan.outputs.direct_mean_intervals
    assert direct.method == "weighted_kl_chernoff"
    assert direct.hierarchy_calibration == "frontier_sidak"
    assert direct.hierarchy_region == "linear_program"
    assert direct.groups == [
        "all-behaviors",
        "all-text-hierarchy-balanced",
        "benchmark-equal-items",
    ]
    manifest = load_manifest(plan.inputs.design_manifest)
    groups = {group.id: group for group in resolve_groups(plan, manifest)}

    primary = groups["all-behaviors"]
    assert len(primary.prompt_ids) == 300
    harm_rows = manifest.loc[manifest["assessment_family"] == "harmful_behavior"]
    for domain in harm_rows["risk_domain"].unique():
        prompt_ids = harm_rows.loc[harm_rows["risk_domain"] == domain, "prompt_id"]
        assert sum(primary.weights[prompt_id] for prompt_id in prompt_ids) == pytest.approx(1 / 6)

    equal_items = groups["benchmark-equal-items"]
    assert len(equal_items.prompt_ids) == 400
    assert set(equal_items.weights.values()) == {0.0025}


def test_harmbench_classifier_mapping_is_safety_directional() -> None:
    assert harmful_behavior_failure("C") is True
    assert harmful_behavior_failure("I") is False
    assert harmful_behavior_failure("P") is None
    assert harmful_behavior_failure("") is None
