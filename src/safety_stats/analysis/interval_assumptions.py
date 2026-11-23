"""The sentences an interval's ``assumptions`` list states, each worded once.

An interval's ``assumptions`` list takes its wording from here, so one assumption reads the same
wherever it applies and can be traced by its identifier. Each sentence is a form of one
assumption of ``registry.ASSUMPTIONS``, which it names, so the sentence an interval states and
the entry "Limitations" explains for it can be traced to each other.

This module covers the assumptions a user can change in the plan, or must know about to read a
result: independence, the fixed design, sampling, and how judge labels were collected.
Statements of how a method works, such as Sidak calibration or a Chernoff kernel, stay with the
code that applies them.

A statement with fields in braces is a ``str.format`` template; ``Assumption.text`` fills it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assumption:
    """One sentence: its stable identifier, the registry assumption it states, and its words.

    ``assumption`` is the identifier of the entry of ``registry.ASSUMPTIONS`` the sentence is a
    form of.
    """

    id: str
    assumption: str
    statement: str

    def text(self, **values: object) -> str:
        """The statement, with any template fields filled from ``values``."""

        return self.statement.format(**values)


# Independence of the observations an interval is built from.
ROLLOUT_INDEPENDENCE = Assumption(
    "independence.rollout",
    "rollout_independence",
    "Attempted rollout outcomes are mutually independent across the fixed node.",
)
ROLLOUT_INDEPENDENCE_ACROSS_ARMS = Assumption(
    "independence.rollout_across_arms",
    "arm_independence",
    "Attempted rollout outcomes are independent within and across comparison arms.",
)
ROLLOUT_INDEPENDENCE_WITHIN_ARMS = Assumption(
    "independence.rollout_within_arms",
    "arm_independence",
    "Attempted rollout outcomes are independent within each arm; the rectangular two-arm "
    "construction does not require independence between arms.",
)
BLOCK_INDEPENDENCE = Assumption(
    "independence.blocks",
    "rollout_independence",
    "Blocks defined by {block_by!r} are mutually independent.",
)
WITHIN_BLOCK_DEPENDENCE = Assumption(
    "independence.within_block_dependence",
    "rollout_independence",
    "Arbitrary dependence is allowed among observations within each declared block.",
)
WITHIN_BLOCK_DEPENDENCE_COUPLED = Assumption(
    "independence.within_block_dependence_coupled",
    "arm_independence",
    "Arbitrary dependence is allowed within each declared block, with cross-arm coupling set "
    "to {coupling!r}.",
)
IID_WITHIN_PROMPT = Assumption(
    "independence.iid_within_prompt",
    "rollout_independence",
    "IID Bernoulli rollouts within each fixed prompt.",
)
NO_INDEPENDENCE_BETWEEN_PROMPTS = Assumption(
    "independence.none_between_prompts",
    "rollout_independence",
    "No independence between prompts is required.",
)

# The fixed design: what was settled before any outcome was seen.
FIXED_WEIGHTS_AND_COUNTS = Assumption(
    "design.fixed_weights_and_counts",
    "fixed_design",
    "Prompt weights and rollout counts are fixed independently of observed outcomes.",
)
FIXED_BLOCK_PARTITION = Assumption(
    "design.fixed_block_partition",
    "fixed_design",
    "The block partition and design weights are fixed independently of outcomes.",
)
FIXED_ALIGNED_WEIGHTS_AND_COUNTS = Assumption(
    "design.fixed_aligned_weights_and_counts",
    "fixed_design",
    "The aligned prompts, design weights, and arm rollout counts are fixed independently of "
    "observed outcomes.",
)
FIXED_ALIGNED_BLOCK_PARTITION = Assumption(
    "design.fixed_aligned_block_partition",
    "fixed_design",
    "The block partition, aligned prompts, and design weights are fixed independently of "
    "observed outcomes.",
)

# Outcomes the scorer did not classify.
LATENT_BINARY_LABEL = Assumption(
    "outcomes.latent_binary_label",
    "missing_scores",
    "Every unclassified attempted outcome has a latent binary safety label.",
)

# Sampled designs: how frame items were drawn.
POISSON_SAMPLING = Assumption(
    "sampling.poisson",
    "probability_sample",
    "Frame items were included independently with the declared probabilities "
    "(Poisson sampling).",
)
STRATIFIED_FIXED_SIZE_SAMPLING = Assumption(
    "sampling.stratified_fixed_size",
    "probability_sample",
    "Each stratum drew exactly its planned number of frame items without replacement; the "
    "bound uses Hoeffding's reduction, under which a convex function of a without-replacement "
    "sum is dominated by the with-replacement one, so it is conservative.",
)
OUTCOME_INDEPENDENT_INCLUSION = Assumption(
    "sampling.outcome_independent_inclusion",
    "probability_sample",
    "Inclusion was determined before, and independently of, the outcomes.",
)

# Judge calibration: where the judge's accuracies come from.
JUDGE_LABELS_SIMPLE_RANDOM_WITHIN_STRATUM = Assumption(
    "judge.labels_simple_random_within_stratum",
    "judge_accuracy",
    "The labelled rows are a simple random sample within their judge stratum; this is a "
    "property of the declared collection protocol, not of the data.",
)
JUDGE_HUMAN_LABELS_ARE_TRUTH = Assumption(
    "judge.human_labels_are_truth",
    "judge_accuracy",
    "Human adjudication is treated as ground truth.",
)
JUDGE_ACCURACIES_DECLARED = Assumption(
    "judge.accuracies_declared",
    "judge_accuracy",
    "These conditional accuracies are declared assumptions, not measurements; the interval "
    "inherits no coverage guarantee from them.",
)
JUDGE_ACCURACIES_SHARED_ACROSS_COHORTS = Assumption(
    "judge.accuracies_shared_across_cohorts",
    "judge_accuracy",
    "The declared accuracies are prevalence-dependent and are applied unchanged to every "
    "cohort{scope}.",
)

REGISTRY: tuple[Assumption, ...] = (
    ROLLOUT_INDEPENDENCE,
    ROLLOUT_INDEPENDENCE_ACROSS_ARMS,
    ROLLOUT_INDEPENDENCE_WITHIN_ARMS,
    BLOCK_INDEPENDENCE,
    WITHIN_BLOCK_DEPENDENCE,
    WITHIN_BLOCK_DEPENDENCE_COUPLED,
    IID_WITHIN_PROMPT,
    NO_INDEPENDENCE_BETWEEN_PROMPTS,
    FIXED_WEIGHTS_AND_COUNTS,
    FIXED_BLOCK_PARTITION,
    FIXED_ALIGNED_WEIGHTS_AND_COUNTS,
    FIXED_ALIGNED_BLOCK_PARTITION,
    LATENT_BINARY_LABEL,
    POISSON_SAMPLING,
    STRATIFIED_FIXED_SIZE_SAMPLING,
    OUTCOME_INDEPENDENT_INCLUSION,
    JUDGE_LABELS_SIMPLE_RANDOM_WITHIN_STRATUM,
    JUDGE_HUMAN_LABELS_ARE_TRUTH,
    JUDGE_ACCURACIES_DECLARED,
    JUDGE_ACCURACIES_SHARED_ACROSS_COHORTS,
)
