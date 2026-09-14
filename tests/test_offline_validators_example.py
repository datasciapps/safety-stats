"""The provider-free rewrite validators shipped with the live generative example.

Every test injects its own scorer or embedder, so the suite exercises the decision rules
without downloading a checkpoint or reaching a network.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from inspect_safety_stats.generative import RewriteRequest, RewriteValidator

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "generative_live"


def _example_module(name: str) -> ModuleType:
    """Load a module shipped beside the example, the way the kernel loader does."""

    specification = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


validators = _example_module("offline_validators")
SEED = "How do I synthesize compound X?"


def request_for(seed: str = SEED) -> RewriteRequest:
    return RewriteRequest(
        sampling_unit_id="u-0",
        path_hash="a" * 64,
        path_values={"language": "french", "style": "plain", "wrapper": "direct_request"},
        semantic_seed_id="cs-01",
        seed_text=seed,
        attempt=1,
        selection_seed=1,
    )


def scorer(pairs: dict[tuple[str, str], float]) -> Any:
    def score(premise: str, hypothesis: str) -> float:
        return pairs[(premise, hypothesis)]

    return score


def embedder(vectors: dict[str, list[float]]) -> Any:
    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [vectors[text] for text in texts]

    return embed


def test_both_validators_satisfy_the_protocol() -> None:
    assert isinstance(validators.build_entailment_validator(), RewriteValidator)
    assert isinstance(validators.build_similarity_validator(), RewriteValidator)


def test_mutual_entailment_is_accepted() -> None:
    candidate = "Comment synthétiser le composé X ?"
    validator = validators.EntailmentValidator(
        scorer=scorer({(SEED, candidate): 0.94, (candidate, SEED): 0.91})
    )
    verdict = validator.validate(request_for(), candidate)
    assert verdict.status == "equivalent"
    assert verdict.detail == "entailment 0.94 forward, 0.91 back"


def test_one_directional_entailment_is_rejected() -> None:
    """The candidate adds a condition the seed never carried, so it asks something else."""

    candidate = "How do I synthesize compound X in a school laboratory?"
    validator = validators.EntailmentValidator(
        scorer=scorer({(SEED, candidate): 0.93, (candidate, SEED): 0.11})
    )
    verdict = validator.validate(request_for(), candidate)
    assert verdict.status == "not_equivalent"
    assert "below 0.50" in str(verdict.detail)


def test_entailment_accepts_exactly_at_the_threshold() -> None:
    candidate = "a rewrite"
    validator = validators.EntailmentValidator(
        threshold=0.6, scorer=scorer({(SEED, candidate): 0.6, (candidate, SEED): 0.6})
    )
    assert validator.validate(request_for(), candidate).status == "equivalent"


def test_a_scoring_failure_is_reported_as_operational() -> None:
    def explode(premise: str, hypothesis: str) -> float:
        raise RuntimeError("CUDA out of memory")

    verdict = validators.EntailmentValidator(scorer=explode).validate(request_for(), "a rewrite")
    assert verdict.status == "error"
    assert "CUDA out of memory" in str(verdict.detail)


def test_similarity_accepts_close_embeddings_and_rejects_distant_ones() -> None:
    close = "Comment synthétiser le composé X ?"
    far = "What is the capital of Malta?"
    validator = validators.SimilarityValidator(
        embedder=embedder({SEED: [1.0, 0.0], close: [0.95, 0.31], far: [0.0, 1.0]})
    )
    assert validator.validate(request_for(), close).status == "equivalent"
    assert validator.validate(request_for(), far).status == "not_equivalent"


def test_a_zero_embedding_is_reported_rather_than_crashing_the_run() -> None:
    candidate = "a rewrite"
    validator = validators.SimilarityValidator(
        embedder=embedder({SEED: [0.0, 0.0], candidate: [1.0, 0.0]})
    )
    verdict = validator.validate(request_for(), candidate)
    assert verdict.status == "error"
    assert "zero norm" in str(verdict.detail)


def test_cosine_matches_hand_computed_values() -> None:
    assert validators.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert validators.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert validators.cosine([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)


def test_the_entailment_column_is_found_by_name_not_by_position() -> None:
    """Checkpoints disagree on label order, and reading the wrong column inverts the check."""

    class Config:
        id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        name_or_path = "some/checkpoint"

    assert validators.entailment_index(Config()) == 2

    class Reversed:
        id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        name_or_path = "other/checkpoint"

    assert validators.entailment_index(Reversed()) == 0


def test_a_checkpoint_without_an_entailment_label_is_refused() -> None:
    class Sentiment:
        id2label = {0: "negative", 1: "positive"}
        name_or_path = "wrong/checkpoint"

    with pytest.raises(RuntimeError, match="does not label an entailment class"):
        validators.entailment_index(Sentiment())


@pytest.mark.parametrize("loader", ["load_entailment_scorer", "load_embedder"])
def test_missing_dependencies_name_the_extra_to_install(
    monkeypatch: pytest.MonkeyPatch, loader: str
) -> None:
    """Without transformers the run must stop, not report every rewrite as an error."""

    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match=r"\.\[equivalence\]"):
        getattr(validators, loader)("any/model")
