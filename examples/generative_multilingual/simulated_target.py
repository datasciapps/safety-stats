"""The simulated target model, and the ground truth it is simulated from.

Nothing here is a real model. This file exists so the example can be checked against a
known answer: because the risk of every structural path is declared below, the true root
risk is computable in closed form, and an interval the package reports can be compared
against it rather than merely inspected.

Swapping in a real evaluation means replacing :func:`evaluate` with an Inspect run over
the materialized dataset -- see the README. Nothing else in the example changes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from safety_stats.generative.execution import AcceptedRewrite
from safety_stats.generative.inspect_bridge import EvaluatedSample
from safety_stats.generative.taxonomy import CompiledTaxonomy

BASE_RATE = 0.010

# Multipliers chosen to make the design's structure matter: a low-resource language and a
# wrapper that buries the request are both well-documented jailbreak vectors, so the risk
# is concentrated in paths that carry only a fifth of the target weight.
PROBLEM_CLASS = {"chemical_synthesis": 1.0, "biological_agents": 1.5}
LANGUAGE = {"english": 1.0, "french": 1.4, "german": 1.3, "maltese": 4.0}
STYLE = {"plain": 1.0, "conversational": 1.2, "formal": 0.9}
WRAPPER = {"direct_request": 1.0, "roleplay": 2.2, "nested_task": 3.0}

# The candidate is uniformly worse than the baseline by this factor. COUPLING is the
# probability that both arms respond to the same latent prompt difficulty, which is what
# gives a paired comparison something to exploit.
CANDIDATE_RATIO = 1.35
COUPLING = 0.8


def path_values(node_id: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in node_id.split("|")[1:] if "=" in part)


def path_risk(node_id: str) -> float:
    """The declared true failure probability of one terminal structural path."""

    values = path_values(node_id)
    risk = BASE_RATE
    risk *= PROBLEM_CLASS[values["problem_class"]]
    risk *= LANGUAGE[values["language"]]
    risk *= STYLE[values["style"]]
    risk *= WRAPPER[values["wrapper"]]
    return min(risk, 1.0)


def true_node_risk(taxonomy: CompiledTaxonomy, node_id: str) -> float:
    """The declared-weight average of path risks below a node -- the node's estimand."""

    descendants = [
        node
        for node in taxonomy.terminal_paths
        if node.node_id == node_id or node.node_id.startswith(f"{node_id}|")
    ]
    if not descendants:
        raise KeyError(node_id)
    mass = sum(node.target_path_weight for node in descendants)
    return sum(node.target_path_weight * path_risk(node.node_id) for node in descendants) / mass


def evaluate(
    selections: Sequence[object],
    accepted: Sequence[AcceptedRewrite],
    rng: np.random.Generator,
    *,
    cohort_id: str = "baseline",
) -> list[EvaluatedSample]:
    """Stand in for an Inspect run: one Bernoulli rollout per accepted rewrite."""

    node_of_unit = {
        str(record.sampling_unit_id): str(record.node_id)
        for record in selections
    }
    return [
        EvaluatedSample(
            sample_id=entry.prompt_id,
            cohort_id=cohort_id,
            epoch=1,
            failure=bool(rng.random() < path_risk(node_of_unit[entry.sampling_unit_id])),
        )
        for entry in accepted
    ]


def evaluate_pair(
    selections: Sequence[object],
    accepted: Sequence[AcceptedRewrite],
    rng: np.random.Generator,
    *,
    ratio: float = CANDIDATE_RATIO,
    coupling: float = COUPLING,
) -> list[EvaluatedSample]:
    """Both cohorts on the *same* accepted rewrites, sharing prompt difficulty.

    A shared latent draw is what makes the arms concordant on easy and hard prompts alike.
    Without it the two arms would be independent given the prompt and pairing would have
    nothing to remove.
    """

    node_of_unit = {
        str(record.sampling_unit_id): str(record.node_id)
        for record in selections
    }
    shared = rng.random(len(accepted))
    independent = rng.random(len(accepted))
    couple = rng.random(len(accepted)) < coupling
    samples: list[EvaluatedSample] = []
    for index, entry in enumerate(accepted):
        risk = path_risk(node_of_unit[entry.sampling_unit_id])
        candidate_draw = shared[index] if couple[index] else independent[index]
        samples.append(
            EvaluatedSample(
                sample_id=entry.prompt_id,
                cohort_id="baseline",
                epoch=1,
                failure=bool(shared[index] < risk),
            )
        )
        samples.append(
            EvaluatedSample(
                sample_id=entry.prompt_id,
                cohort_id="candidate",
                epoch=1,
                failure=bool(candidate_draw < min(risk * ratio, 1.0)),
            )
        )
    return samples


def true_difference(taxonomy: CompiledTaxonomy, ratio: float = CANDIDATE_RATIO) -> float:
    """The declared risk difference at the root, candidate minus baseline."""

    return sum(
        node.target_path_weight
        * (min(path_risk(node.node_id) * ratio, 1.0) - path_risk(node.node_id))
        for node in taxonomy.terminal_paths
    )
