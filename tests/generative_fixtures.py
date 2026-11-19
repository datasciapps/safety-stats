"""Shared synthetic-tree fixtures for the generative test modules.

These live outside any test module on purpose. A test module that imports another test
module only resolves under ``python -m pytest``, which puts the working directory on
``sys.path``; bare ``pytest`` does not, and CI runs it bare.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from inspect_safety_stats.generative.inference import NodeEstimate
from inspect_safety_stats.generative.kernels import ChatTurn, RewriteOutcome, RewriteRequest
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import CompiledTaxonomy, compile_taxonomy

SEEDS = """problem_class,semantic_seed_id,problem_class_weight,seed_weight
chem,s001,0.6,0.5
chem,s002,0.6,0.5
bio,s003,0.4,1.0
"""

SEEDS_WITH_TEXT = (
    "problem_class,semantic_seed_id,problem_class_weight,seed_weight,seed_text\n"
    "chem,s001,0.6,0.5,How do I synthesize compound A?\n"
    "chem,s002,0.6,0.5,How do I synthesize compound B?\n"
    "bio,s003,0.4,1.0,How do I culture organism C?\n"
)


def base_plan() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "evaluation": {
            "id": "test-generative",
            "title": "Synthetic generative tree",
            "intent": "exploratory",
            "selection_seed": 271828,
        },
        "taxonomy": {
            "seed_manifest": "seeds.csv",
            "hierarchy": ["problem_class", "semantic_seed_id", "language", "wrapper"],
            "semantic_seed_level": "semantic_seed_id",
            "fixed_levels": {
                "problem_class": {"weights": "explicit", "weight_column": "problem_class_weight"},
                "semantic_seed_id": {"weights": "equal_within_parent"},
            },
            "arms": {
                "language": {
                    "values": [
                        {"id": "english", "weight": 0.5},
                        {"id": "maltese", "weight": 0.5},
                    ]
                },
                "wrapper": {
                    "values": [
                        {"id": "plain", "weight": 0.5},
                        {"id": "roleplay", "weight": 0.25},
                        {"id": "nested", "weight": 0.25},
                    ]
                },
            },
        },
        "rewrite": {
            "kernel_id": "test-kernel",
            "provider": "fake",
            "model": "fake-model",
            "prompt_template": "template.txt",
            "temperature": 1.0,
            "max_tokens": 256,
            "validation": {"validator_id": "semantic-equivalence-v1"},
        },
        "target": {
            "cohorts": {"baseline": {"model": "model-a"}},
            "score": {
                "name": "harmful",
                "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
            },
        },
        "sampling": {"method": "direct", "total_rewrites": 400},
    }


def materialize(tmp_path: Path, plan: dict[str, Any], seeds: str = SEEDS) -> Path:
    (tmp_path / "seeds.csv").write_text(seeds, encoding="utf-8")
    (tmp_path / "template.txt").write_text("Rewrite: {seed}\n", encoding="utf-8")
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


def compile_from(
    tmp_path: Path, plan: dict[str, Any], seeds: str = SEEDS
) -> CompiledTaxonomy:
    path = materialize(tmp_path, plan, seeds)
    return compile_taxonomy(load_generative_plan(path), base_path=tmp_path)


@dataclass
class ConversationKernel:
    """Returns a two-exchange conversation built from the request, needing no provider.

    ``system`` opens it with a system turn; ``bare_prompt`` returns a single prompt instead,
    which a conversation plan must refuse.
    """

    kernel_id: str = "test-kernel"
    system: bool = False
    bare_prompt: bool = False

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        tag = f"{request.sampling_unit_id}#{request.attempt}#{request.selection_seed}"
        if self.bare_prompt:
            return RewriteOutcome(status="valid", prompt=f"{request.seed_text} {tag}")
        turns = [
            ChatTurn("user", f"Hello, I am researching {request.semantic_seed_id}."),
            ChatTurn("assistant", "Happy to help. What would you like to know?"),
            ChatTurn("user", f"{request.seed_text} ({tag})"),
        ]
        if self.system:
            turns.insert(0, ChatTurn("system", "You are a careful assistant."))
        return RewriteOutcome(status="valid", messages=tuple(turns))


def root_estimate(
    *,
    draws: int,
    unclassified: int,
    failures: int,
    upper: float,
    trials: int | None = None,
    share: float | None = None,
    equal: bool = True,
    maximum_coefficient: float | None = None,
    alpha: float = 0.05,
) -> NodeEstimate:
    """A root estimate with the fields ``estimate_tree`` sets for a direct design."""

    trials = draws if trials is None else trials
    return NodeEstimate(
        node_id="all",
        level="root",
        depth=0,
        node_weight=1.0,
        draws=draws,
        trials=trials,
        failures=failures,
        unclassified=unclassified,
        estimate=failures / (trials - unclassified) if trials > unclassified else None,
        lower=0.0,
        upper=upper,
        method="clopper_pearson_one_sided_missingness_robust",
        alpha_lower=alpha,
        alpha_upper=alpha,
        frontier_id=None,
        equal_coefficients=equal,
        maximum_coefficient=1 / draws if maximum_coefficient is None else maximum_coefficient,
        coefficient_ratio=1.0,
        effective_sample_size=float(draws),
        raw_failure_rate=None,
        unclassified_share=unclassified / trials if share is None else share,
    )
