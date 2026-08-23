from pathlib import Path

import pytest

from inspect_safety_stats.config.models import WeightSpec
from inspect_safety_stats.ingest.manifest import load_manifest, resolve_weights


def test_nested_equal_prevents_variant_count_weighting(tmp_path: Path) -> None:
    path = tmp_path / "design.csv"
    path.write_text(
        "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,2\np2,q1,t1,2\np3,q2,t1,2\n",
        encoding="utf-8",
    )
    manifest = load_manifest(path)
    weights = resolve_weights(manifest, WeightSpec(strategy="nested_equal"))
    assert weights.tolist() == pytest.approx([0.25, 0.25, 0.5])


def test_duplicate_prompt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "design.csv"
    path.write_text("prompt_id,question_id,task_id\np1,q1,t1\np1,q1,t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(path)
