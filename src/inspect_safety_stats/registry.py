"""The registry of diagnostic codes and reportable assumptions.

Every diagnostic code the package emits has one entry in ``CODES``, and every assumption a
report's results rest on has one in ``ASSUMPTIONS``. Each entry says what is assumed, which kind
of input it concerns (``SOURCE_KINDS``), what happens to the results (``EFFECTS``), and what the
user can do about it. The diagnostic messages say what was found in one analysis; the registry
says what the code means in every analysis.

``tests/unit/test_registry.py`` keeps the registry and the source in step: a code the package
emits without an entry fails it, as does an entry no source emits, a severity the entry leaves
out, or a plan default that differs from the one the entry states.

This module imports nothing beyond the standard library, so that the CLI and the report
writer can read it without loading pandas, SciPy or Inspect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

Severity = Literal["ERROR", "WARNING", "INFO"]

SourceKind = Literal[
    "plan",
    "registration",
    "manifest",
    "logs",
    "judge_labels",
    "analysis",
    "generation",
]
"""The kind of input an entry concerns: where to look, and what to change."""

SOURCE_KINDS: Final[Mapping[SourceKind, str]] = MappingProxyType(
    {
        "plan": "The analysis plan: its fields, references and their combinations.",
        "registration": (
            "When the plan was fixed: its freeze time and its registration receipt, against "
            "the start of the runs."
        ),
        "manifest": (
            "The design manifest and the design files beside it: prompts, groups, weights, "
            "inclusion probabilities and strata."
        ),
        "logs": (
            "The evaluation logs or rollout files: runs, samples, scores, errors and their "
            "configuration."
        ),
        "judge_labels": (
            "The judge validation inputs: the human labels, their strata and the labelling "
            "protocol."
        ),
        "analysis": (
            "The statistical computation itself, on inputs that passed their checks: a "
            "method or region that cannot be computed, or a result it qualifies."
        ),
        "generation": (
            "A generative run: the items it generated, validated and evaluated, and the "
            "Inspect logs that evaluated them."
        ),
    }
)

Effect = Literal["invalid", "degraded", "withheld", "adjusted", "recorded"]
"""What an entry does to the results."""

EFFECTS: Final[Mapping[Effect, str]] = MappingProxyType(
    {
        "invalid": (
            "The integrity status is invalid: no claim can pass, and the results the error "
            "concerns are not reported. A generative report has no integrity status: it is "
            "still written in full, and the error makes `generative analyze` exit with an "
            "error code."
        ),
        "degraded": (
            "The integrity status is degraded: the results are reported, marked as resting "
            "on an assumption the inputs could not confirm."
        ),
        "withheld": (
            "One result, such as a correction, bound or region, is not reported, and the "
            "integrity status is unaffected. A claim that rests on the result is not decided "
            "on it, so it can lose a PASS, and the overall decision with it; the rest of the "
            "analysis is reported as usual."
        ),
        "adjusted": (
            "The analysis changes how it computes a result, for example by widening a bound "
            "or taking a different method, and reports it; the integrity status is "
            "unaffected. A wider bound or a decision rule can withhold a claim's PASS or the "
            "overall decision's."
        ),
        "recorded": (
            "Nothing is computed differently: the diagnostic records a fact a reader needs to "
            "interpret the results."
        ),
    }
)


@dataclass(frozen=True)
class CodeEntry:
    """What one diagnostic code means.

    ``severities`` lists every severity the package emits the code with. ``assumption`` is
    what the results rest on that the code reports on, phrased as the assumption, not as the
    problem. ``consequence`` says in one sentence what ``effect`` means for this code.
    ``assumptions`` names the entries of ``ASSUMPTIONS`` the code checks, if any.
    """

    code: str
    severities: tuple[Severity, ...]
    source: SourceKind
    effect: Effect
    assumption: str
    consequence: str
    remedy: str
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssumptionSetting:
    """An assumption as one value of its plan field states it, other than the default.

    ``value`` is written as in a plan. ``if_wrong`` is None when the entry's own holds.
    """

    value: str
    statement: str
    if_wrong: str | None = None


@dataclass(frozen=True)
class AssumptionEntry:
    """One assumption a report's results can rest on.

    ``plan_field`` is the plan field that sets or relaxes it, as a dotted path in which
    ``[]`` stands for any item of a list and ``<id>`` for any key of a mapping; None when no
    plan field changes it. ``default`` is that field's value when a plan leaves it out,
    written as in a plan; None when the plan must state it, or when there is no field.
    ``checked_by`` gives the diagnostic codes that report evidence against it, read from the
    ``assumptions`` of their entries in ``CODES``.

    ``statement`` and ``if_wrong`` describe the assumption at the default. ``by_setting``
    describes it at other values of a field that takes one of a few named values; a report
    states the one for the value the plan took.
    """

    id: str
    statement: str
    source: SourceKind
    plan_field: str | None
    default: str | None
    if_wrong: str
    remedy: str
    by_setting: tuple[AssumptionSetting, ...] = ()

    def stated_for(self, value: object) -> tuple[str, str]:
        """The statement and if-wrong text for the plan field's value: the entry's own unless
        the value has its own."""

        setting = next((item for item in self.by_setting if item.value == value), None)
        if setting is None:
            return self.statement, self.if_wrong
        return setting.statement, setting.if_wrong or self.if_wrong


_ASSUMPTIONS: tuple[AssumptionEntry, ...] = (
    AssumptionEntry(
        id="fixed_design",
        statement=(
            "Prompt, question, task and group membership are fixed, creator-declared design "
            "elements. The bounds cover these prompts, not a population they were drawn from."
        ),
        source="manifest",
        plan_field="design.type",
        default="fixed",
        if_wrong=(
            "A result read as covering prompts beyond the manifest overstates what the bounds "
            "say about them."
        ),
        remedy=(
            "When the prompts are a probability sample of a larger set, declare "
            "design.type: sampled with each prompt's inclusion probability."
        ),
    ),
    AssumptionEntry(
        id="probability_sample",
        statement=(
            "The evaluated items are a probability sample of the declared frame: each item's "
            "inclusion probability was fixed before the draw, the draw was made at random and "
            "without regard to the outcomes, and each item was included independently of the "
            "others with its declared probability (Poisson sampling). Each drawn item is one "
            "independent block: the rollouts of different items are independent."
        ),
        source="manifest",
        plan_field="design.sampling",
        default="poisson",
        if_wrong=(
            "Items drawn with other probabilities than those declared, or chosen with the "
            "outcomes in view, bias the frame estimate; rollouts of different items that share "
            "a cause make the bounds narrower than their stated coverage."
        ),
        remedy=(
            "Draw the items at random with probabilities fixed in advance, before any outcome "
            "is seen, record the draw in design.selection_column and each item's inclusion "
            "probability in the manifest; otherwise analyse the evaluated items as a fixed "
            "design."
        ),
        by_setting=(
            AssumptionSetting(
                "fixed_size",
                "The evaluated items are a probability sample of the declared frame: each "
                "stratum drew its planned number of items at random without replacement, with "
                "the inclusion probabilities fixed before the draw and the draw made without "
                "regard to the outcomes. Each drawn item is one independent block: the rollouts "
                "of different items are independent.",
            ),
        ),
    ),
    AssumptionEntry(
        id="run_stability",
        statement=(
            "Every rollout of a cohort comes from one stable system: the same model and "
            "configuration throughout its runs."
        ),
        source="logs",
        plan_field="cohorts.<id>.require_uniform",
        default="[]",
        if_wrong=(
            "The bounds describe a mixture of configurations, not the system the cohort "
            "names."
        ),
        remedy=(
            "List the configuration fields that must not vary in the cohort's "
            "require_uniform, and analyse configurations as separate cohorts."
        ),
    ),
    AssumptionEntry(
        id="rollout_independence",
        statement=(
            "The blocks of the declared independence profile are independent: under the "
            "rollout profile, every rollout of a prompt is an independent draw."
        ),
        source="plan",
        plan_field="outputs.direct_mean_intervals.independence",
        default=None,
        if_wrong=(
            "Rollouts that share a cause, such as a retried request, a cache or a shared "
            "conversation, make the bounds narrower than their stated coverage."
        ),
        remedy=(
            "Declare an independence profile that blocks the rollouts by what they share, "
            "and add coarser profiles as sensitivities."
        ),
    ),
    AssumptionEntry(
        id="arm_independence",
        statement=(
            "The arms of a comparison are independent of each other, block by block, unless "
            "the profile couples them."
        ),
        source="plan",
        plan_field="design.independence_profiles.<id>.comparison_coupling",
        default="independent",
        if_wrong=(
            "Arms that share prompts' randomness, such as the same sampled conversation, make "
            "the comparison's bounds miss their stated coverage."
        ),
        remedy="Set comparison_coupling: joint on a profile whose blocks span both arms.",
    ),
    AssumptionEntry(
        id="scorer_validity",
        statement=(
            "The score measures the hazard. Scorer errors, under-elicitation, evaluation "
            "awareness and omitted hazards are not repaired by statistical precision."
        ),
        source="logs",
        plan_field=None,
        default=None,
        if_wrong=(
            "The bounds are precise about the wrong quantity: a narrow bound on a score that "
            "misses the hazard says nothing about the hazard."
        ),
        remedy=(
            "Validate the scorer on labelled rollouts, and declare judges with "
            "inputs.judge_calibration to correct for their error."
        ),
    ),
    AssumptionEntry(
        id="judge_accuracy",
        statement=(
            "Judge verdicts are taken as correct unless the plan validates or transports the "
            "judge's error rates."
        ),
        source="judge_labels",
        plan_field="inputs.judge_calibration.mode",
        default="assumed",
        if_wrong="Judge errors bias every rate and comparison the judge's verdicts enter.",
        remedy=(
            "Label a validation sample and set inputs.judge_calibration.mode: validated, or "
            "transported with a published validation set."
        ),
    ),
    AssumptionEntry(
        id="missing_scores",
        statement=(
            "A rollout without a score has an unknown outcome, bounded at both extremes, "
            "unless the plan counts it otherwise."
        ),
        source="logs",
        plan_field="inputs.score.missing",
        default="conservative_bounds",
        if_wrong=(
            "Under exclude or count_as_failure, missing scores that are not missing at random "
            "bias the rates."
        ),
        remedy=(
            "Rescore the missing rollouts, or keep score.missing: conservative_bounds so the "
            "bounds cover every outcome they could have had."
        ),
    ),
    AssumptionEntry(
        id="sample_errors",
        statement=(
            "A sample error counts as the plan's score.errors says. Under end_to_end, with "
            "the end-to-end system estimand, a provider's policy block of the target's "
            "generation that delivered no content counts as safe."
        ),
        source="logs",
        plan_field="inputs.score.errors",
        default="end_to_end",
        if_wrong=(
            "When errors come from the harness rather than the system, counting them as "
            "outcomes biases the rates."
        ),
        remedy=(
            "Rerun the errored samples, or choose the score.errors policy that matches where "
            "the errors come from."
        ),
    ),
    AssumptionEntry(
        id="token_limit",
        statement=(
            "An answer that stopped at the generation token limit counts by its score, and "
            "without a score as missing."
        ),
        source="logs",
        plan_field="inputs.score.token_limit",
        default="score",
        if_wrong=(
            "A truncated answer may have turned harmful or safe had it continued; its score "
            "describes the truncation."
        ),
        remedy=(
            "Raise the token limit and rerun, or classify truncated answers with "
            "score.token_limit."
        ),
        by_setting=(
            AssumptionSetting(
                "missing",
                "An answer that stopped at the generation token limit counts by its score, and "
                "without a score or a sample error as missing, counted on its own.",
            ),
            AssumptionSetting(
                "safe",
                "An answer that stopped at the generation token limit counts by its score, and "
                "without a score or a sample error as safe, as the plan declares.",
            ),
            AssumptionSetting(
                "failure",
                "An answer that stopped at the generation token limit counts by its score, and "
                "without a score or a sample error as a failure, as the plan declares.",
            ),
        ),
    ),
    AssumptionEntry(
        id="content_filter_stops",
        statement=(
            "An output a provider's content filter stopped without a sample error counts by "
            "its score."
        ),
        source="logs",
        plan_field="inputs.score.content_filter_stops",
        default="score",
        if_wrong=(
            "A stop that the scorer reads as a refusal or a failure counts as the model's "
            "behaviour, though the provider's filter produced it."
        ),
        remedy=(
            "Classify the stops with score.content_filter_stops: safe or failure, under the "
            "end-to-end estimand."
        ),
        by_setting=(
            AssumptionSetting(
                "safe",
                "An output a provider's content filter stopped before it delivered any "
                "content, without a sample error, counts as safe whatever its score: a policy "
                "block of the end-to-end system, as the plan declares.",
                "The rate holds for the system behind this provider's filter. Behind another "
                "filter, a stopped output may have been delivered, and may have been harmful.",
            ),
            AssumptionSetting(
                "failure",
                "An output a provider's content filter stopped before it delivered any "
                "content, without a sample error, counts as a failure whatever its score: a "
                "refusal of the end-to-end system, as the plan declares.",
                "The rate holds for the system behind this provider's filter. Behind another "
                "filter, a stopped output may have been delivered as an answer, not a refusal.",
            ),
        ),
    ),
    AssumptionEntry(
        id="complete_design",
        statement=(
            "Every run finished, every prompt of a fixed design was evaluated, and every "
            "prompt has the rollouts the manifest expects."
        ),
        source="logs",
        plan_field="design.require_expected_rollouts",
        default="true",
        if_wrong=(
            "Prompts or rollouts that are missing for a reason linked to their outcome bias "
            "the rates."
        ),
        remedy="Rerun the missing prompts and rollouts.",
    ),
    AssumptionEntry(
        id="plan_fixed_in_advance",
        statement=(
            "The plan was fixed before the runs started, so that no choice in it was made "
            "after seeing the outcomes."
        ),
        source="registration",
        plan_field="analysis.intent",
        default="confirmatory",
        if_wrong=(
            "Choices made after seeing outcomes can select the result, and the stated "
            "coverage no longer holds."
        ),
        remedy=(
            "Freeze the plan before the runs, state analysis.plan_frozen_at, register its "
            "hash and pass the receipt with --registration-receipt; otherwise set "
            "analysis.intent: exploratory."
        ),
    ),
    AssumptionEntry(
        id="adaptive_attack_stream",
        statement=(
            "The behaviours of each adaptive claim were attacked in the order its order_field "
            "records, fixed before any outcome was seen or drawn at random, each at most once; "
            "every run's score is taken as right, without correcting judge error, even where "
            "the plan validates the judge."
        ),
        source="manifest",
        plan_field="adaptive_claims[].order_field",
        default=None,
        if_wrong=(
            "An order chosen or changed after seeing outcomes, such as attacking first the "
            "behaviours already found to fall, or a judge that confirms successes that are "
            "not, can raise the lower bound above the attacker's success rate and make a FAIL "
            "wrong."
        ),
        remedy=(
            "Fix the attack order before the runs, or draw it at random from a recorded seed, "
            "and record it in the manifest column order_field names; validate the "
            "confirmation judge on labelled outputs of the attack."
        ),
    ),
)

ASSUMPTIONS: Final[Mapping[str, AssumptionEntry]] = MappingProxyType(
    {entry.id: entry for entry in _ASSUMPTIONS}
)

_CODES: tuple[CodeEntry, ...] = (
    CodeEntry(
        code="ALPHA_ALLOCATION_INVALID",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "The prompt-bound alpha allocation covers every manifest prompt: an explicit "
            "allocation lists each prompt once within the family alpha, and a weighted one gives "
            "each prompt a positive weight."
        ),
        consequence=(
            "No prompt receives an alpha, so no prompt bounds are computed, and every prompt is "
            "blocked from the group, direct and comparison intervals built on them."
        ),
        remedy=(
            "Under alpha_allocation: explicit, list every manifest prompt exactly once in "
            "outputs.prompt_bounds.explicit_alpha with values summing to no more than the family "
            "alpha. Under weighted_bonferroni, make design.weights resolve to a positive weight "
            "for every prompt, or choose another outputs.prompt_bounds.alpha_allocation."
        ),
    ),
    CodeEntry(
        code="ANYTIME_BOUND_INVALID",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "Each cohort's observations come from one run whose recorded order lists every "
            "prompt of each bounded group exactly once, and each sample joins one prompt."
        ),
        consequence=(
            "No anytime-valid bound is reported for the cohort and group, and the error makes "
            "the integrity status invalid, so no claim can pass."
        ),
        remedy=(
            "Analyse one run per cohort that ran every prompt of the group, without a limit "
            "that leaves prompts out of its order, or restrict outputs.anytime_bounds.groups to "
            "groups the run covers; otherwise remove outputs.anytime_bounds."
        ),
    ),
    CodeEntry(
        code="ANYTIME_ORDER_UNRECORDED",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "The run's samples ran in a random order fixed before the run and recorded in its "
            "Inspect log: sample_shuffle set to a nonzero integer seed, with the shuffled sample "
            "ids. Inspect does not shuffle under the seed 0."
        ),
        consequence=(
            "No anytime-valid bound is reported for the cohort and group, and the error makes "
            "the integrity status invalid, so no claim can pass."
        ),
        remedy=(
            "Rerun the evaluation as Inspect logs with sample_shuffle set to a nonzero integer "
            "seed, or remove outputs.anytime_bounds."
        ),
    ),
    CodeEntry(
        code="ANYTIME_UPPER_BOUND",
        severities=("INFO",),
        source="analysis",
        effect="recorded",
        assumption=(
            "The run's prompts ran in a random order fixed before the run, and the rule that "
            "stopped it chose only when to stop, never which prompt ran next."
        ),
        consequence=(
            "The anytime-valid upper bound on the group's weighted mean failure probability, "
            "and whether it lies at or below each weighted-mean claim's acceptable_upper, are "
            "recorded beside the claims, whose status does not change. Under exploratory "
            "intent or invalid integrity, where no claim can pass, it certifies none."
        ),
        remedy=(
            "Nothing to change. Read the bound's certified flags for a run stopped early, since "
            "its claims keep their fixed-horizon status."
        ),
    ),
    CodeEntry(
        code="CLUSTER_DESIGN_EFFECT",
        severities=("WARNING", "INFO"),
        source="logs",
        effect="recorded",
        assumption=(
            "Rollouts that an independence profile puts in different blocks do not tend to "
            "fail together. For a profile whose blocks group rollouts of several prompts, the "
            "one-way ANOVA intra-cluster correlation of the scored outcomes within its blocks "
            "gives the design effect 1 + (m-1)*max(rho, 0), with m the size of the block "
            "an outcome sits in averaged over outcomes, and about N / design effect effective "
            "rollouts. For each primary profile of a result the plan computes that splits the "
            "blocks, the same figures on its units, the scored rollouts of one of its blocks "
            "within one of the profile's blocks with their failure rate as the outcome and m "
            "counted in units, give its effective units. For units of several rollouts the "
            "correlation is the covariance of two units of one block over the variance of all "
            "units, which units with few rollouts do not raise. It is a warning when, in a "
            "cohort, the design effect on a primary's units reaches 2, so that their effective "
            "number is at most half of them."
        ),
        consequence=(
            "Nothing is computed differently: the diagnostic records how much the rollouts' "
            "grouping shrinks their evidence, so a reader can judge results that rest on a "
            "finer profile. Under a fixed design the correlation also includes fixed "
            "differences in failure rates between prompts: those between blocks raise it and "
            "those within a block lower it, so it can overstate or understate the dependence "
            "between rollouts. On rollouts it also includes the correlation of one prompt's "
            "rollouts, which a prompt-level primary's units do not. It counts rollouts or units "
            "equally, whatever the prompt weights of the weighted mean the intervals bound."
        ),
        remedy=(
            "Add the grouping profile as an independence sensitivity of the results, or make "
            "it their primary profile, and read the bounds under it beside the primary ones."
        ),
        assumptions=("rollout_independence",),
    ),
    CodeEntry(
        code="COHORT_EMPTY",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "Every declared cohort is filled by at least one selected evaluation whose fields "
            "match the cohort's match."
        ),
        consequence=(
            "Integrity is invalid; when no evaluation matches any cohort every prompt bound is "
            "withheld, otherwise the bounds of the empty cohort, which has no observations."
        ),
        remedy=(
            "Select the logs of the cohort's runs, or correct cohorts.<id>.match so that it "
            "names the fields those runs record; under schema 3.0 a source field is written "
            "attributes.<name>."
        ),
    ),
    CodeEntry(
        code="COHORT_OVERLAP",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption="Each selected evaluation matches at most one declared cohort.",
        consequence=(
            "Integrity is invalid, the evaluation is assigned to no cohort so its rollouts are "
            "left out, and every prompt bound is withheld."
        ),
        remedy=(
            "Make the cohorts.<id>.match of the cohorts mutually exclusive, for example by "
            "adding a field that tells their runs apart."
        ),
    ),
    CodeEntry(
        code="COMPARISON_CALIBRATION_INVALID",
        severities=("ERROR",),
        source="analysis",
        effect="invalid",
        assumption=(
            "The simultaneous comparisons of a multiplicity family can be calibrated together "
            "under each independence profile: under frontier_sidak, every hierarchy level "
            "partitions the comparison's group into non-overlapping nodes."
        ),
        consequence=(
            "The family's comparisons get no interval under that independence profile, and the "
            "error makes the integrity status invalid."
        ),
        remedy=(
            "Give every hierarchy node of the group a positive weight, so that each level "
            "partitions it, or set comparisons[].hierarchy_calibration (or "
            "comparison_sets[].hierarchy_calibration) to flat_bonferroni."
        ),
    ),
    CodeEntry(
        code="COMPARISON_HIERARCHY_CONFIDENCE_REGION_EMPTY",
        severities=("WARNING",),
        source="analysis",
        effect="withheld",
        assumption=(
            "The raw simultaneous comparison intervals of a comparison's hierarchy share at "
            "least one vector of prompt risks, as they do whenever they all cover."
        ),
        consequence=(
            "The comparison's intervals for that method and profile are not projected onto the "
            "common hierarchy region and are reported raw."
        ),
        remedy=(
            "Nothing in the inputs needs to change: the raw intervals keep their coverage. Set "
            "comparisons[].hierarchy_region: none to report raw intervals without projecting."
        ),
    ),
    CodeEntry(
        code="COMPARISON_INTERVAL_INVALID",
        severities=("ERROR",),
        source="analysis",
        effect="invalid",
        assumption=(
            "Every comparison interval the plan requests can be computed from its calibrated "
            "alpha and the blocks of its arms."
        ),
        consequence=(
            "The comparison has no interval for that method and profile, none at all when it "
            "is the primary, and the error makes the integrity status invalid."
        ),
        remedy=(
            "Correct the input the message names, or choose another comparisons[].method or "
            "comparisons[].reference_methods entry."
        ),
    ),
    CodeEntry(
        code="CONFIG_DRIFT",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "Every evaluation of a cohort records the same configuration: the source's "
            "configuration fields and the fields the cohort lists in require_uniform."
        ),
        consequence=(
            "Integrity is invalid and the prompt bounds of the cohort, with the risks and "
            "comparisons built on them, are withheld."
        ),
        remedy=(
            "Split the runs into one cohort per configuration with cohorts.<id>.match, or "
            "select only the runs of one configuration and rerun the analysis."
        ),
        assumptions=("run_stability",),
    ),
    CodeEntry(
        code="CONFIG_UNOBSERVED",
        severities=("WARNING",),
        source="logs",
        effect="degraded",
        assumption=(
            "Every evaluation of a cohort records its configuration, so that drift between "
            "its runs can be checked."
        ),
        consequence=(
            "Integrity is degraded: the evaluations that record no configuration are analysed, "
            "but drift is not checked for them."
        ),
        remedy=(
            "Record the configuration of each run in the source, such as model in rollout "
            "files, or name recorded fields in cohorts.<id>.require_uniform."
        ),
        assumptions=("run_stability",),
    ),
    CodeEntry(
        code="CONTENT_FILTER_STOP_SCORED",
        severities=("WARNING", "INFO"),
        source="logs",
        effect="recorded",
        assumption=(
            "inputs.score.content_filter_stops decides the outcome of a provider "
            "content-filter stop only when the target delivered no content in the sample "
            "before it: no text, readable reasoning or tool call in the stopped output or in "
            "an earlier generation, such as an earlier turn. Under safe, it decides only a "
            "missing or safe score, never a failure or a value the mapping does not cover."
        ),
        consequence=(
            "Nothing changes from the default: the rollout's score decides, as for any other "
            "answer, and a missing score follows inputs.score.missing. It is a warning when "
            "a stop that delivered nothing has a failure or unmapped score, which the "
            "transcript does not explain."
        ),
        remedy=(
            "None needed when content was delivered. For a warning, check how the scorer reads "
            "an empty answer and whether harm reached the user outside the transcript. When "
            "the transcript does not record the stopped generation, keep each sample's model "
            "events in the log."
        ),
        assumptions=("content_filter_stops",),
    ),
    CodeEntry(
        code="DECISION_POLICY_AFTER_GENERATION",
        severities=("INFO",),
        source="registration",
        effect="recorded",
        assumption=(
            "A decision policy's thresholds were set before the evidence. They do not enter "
            "the bounds, so each claim's verdict keeps its error bound at a later threshold, "
            "but such a threshold may describe the bounds rather than a requirement fixed in "
            "advance."
        ),
        consequence=(
            "The policy's thresholds stay in force and the claims keep their status; the "
            "versions set after the first evaluation started are listed in the diagnostic. "
            "Whether the overall decision still certifies is DECISION_POLICY_OVERALL_UNCERTIFIED's "
            "concern."
        ),
        remedy=(
            "No change is needed; to avoid the note, set the policy's versions before the runs "
            "start."
        ),
    ),
    CodeEntry(
        code="DECISION_POLICY_FIXED_CLAIM",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "A decision policy sets thresholds only for claims whose test does not use them, "
            "never for a noisy_valid_test claim."
        ),
        consequence=(
            "The integrity status is invalid and the policy is not applied: every claim keeps "
            "the plan's thresholds, and none can pass."
        ),
        remedy=(
            "Remove the claim from the policy's versions[].thresholds; set a noisy_valid_test "
            "claim's thresholds in the plan, before the run."
        ),
    ),
    CodeEntry(
        code="DECISION_POLICY_MISMATCH",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "The decision policy names the inference hash of the plan it is applied to, so the "
            "plan has not changed in anything that shapes its bounds or claims."
        ),
        consequence=(
            "The integrity status is invalid and the policy is not applied: every claim keeps "
            "the plan's thresholds, and none can pass."
        ),
        remedy=(
            "Apply the policy to the plan it was written for, or set its inference_plan_sha256 "
            "to the hash `inspect-safety-stats hash-plan --inference` prints for this plan."
        ),
    ),
    CodeEntry(
        code="DECISION_POLICY_OVERALL_UNCERTIFIED",
        severities=("WARNING",),
        source="plan",
        effect="adjusted",
        assumption=(
            "An overall PASS has the error bound of one claim only when every mandatory "
            "claim's acceptable_upper was fixed before the data, or when one simultaneous "
            "coverage event covers every mandatory claim's bound."
        ),
        consequence=(
            "The overall decision is INCONCLUSIVE where every mandatory claim passes, since a "
            "wrong PASS is bounded only by the sum of the separate events' error budgets; a "
            "FAIL stays FAIL, and every claim keeps its status."
        ),
        remedy=(
            "Set these thresholds in the plan before the runs, or decide every mandatory claim "
            "on one simultaneous family, such as the default simultaneous prompt bounds or "
            "simultaneous direct-mean intervals."
        ),
        assumptions=("plan_fixed_in_advance",),
    ),
    CodeEntry(
        code="DECISION_POLICY_SELF_DECLARED",
        severities=("WARNING",),
        source="plan",
        effect="recorded",
        assumption=(
            "A decision policy version's set_at, which only its author states, is when its "
            "thresholds were decided."
        ),
        consequence=(
            "The report records the versions listed in the diagnostic as set before the first "
            "evaluation, as they state; no result changes, and the overall decision does not "
            "rely on the stated time."
        ),
        remedy=(
            "Keep the policy file where others can check when each version was written, such "
            "as a version-controlled repository or a registry, and say where in each "
            "version's reason."
        ),
        assumptions=("plan_fixed_in_advance",),
    ),
    CodeEntry(
        code="DECISION_POLICY_UNKNOWN_CLAIM",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption="Every claim a decision policy sets thresholds for is declared in the plan.",
        consequence=(
            "The integrity status is invalid and the policy is not applied: every claim keeps "
            "the plan's thresholds, and none can pass."
        ),
        remedy=(
            "Correct or remove the claim identifier in the policy's versions[].thresholds, or "
            "declare the claim under claims."
        ),
    ),
    CodeEntry(
        code="DIRECT_MEAN_CALIBRATION_INVALID",
        severities=("ERROR",),
        source="analysis",
        effect="invalid",
        assumption=(
            "The direct-mean intervals of a multiplicity family can be calibrated together: "
            "under frontier_sidak, every hierarchy level partitions each selected group into "
            "non-overlapping nodes."
        ),
        consequence=(
            "No direct-mean interval of the family is calibrated: group rows keep only the "
            "propagated prompt envelope, hierarchy-node rows have no interval, and the error "
            "makes the integrity status invalid."
        ),
        remedy=(
            "Give every hierarchy node of the group a positive weight, so that each level "
            "partitions it, or set outputs.direct_mean_intervals.hierarchy_calibration to "
            "flat_bonferroni."
        ),
    ),
    CodeEntry(
        code="DIRECT_MEAN_INTERVAL_INVALID",
        severities=("ERROR",),
        source="analysis",
        effect="invalid",
        assumption=(
            "Every direct interval the plan requests on a group, for its weighted means and for "
            "worst_of_k measures with direct_interval, can be computed from its calibrated "
            "alpha and the group's blocks."
        ),
        consequence=(
            "The group row has no direct interval for that measure, so its primary interval is "
            "the propagated prompt envelope, and the error makes the integrity status invalid."
        ),
        remedy=(
            "Correct the input the message names, or choose another "
            "outputs.direct_mean_intervals.method, or turn off "
            "outputs.risk_measures[].direct_interval."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_ACCEPTANCE_UNEVEN",
        severities=("WARNING",),
        source="generation",
        effect="recorded",
        assumption=(
            "The rewrite procedure exhausts its attempts at similar rates across the values of "
            "each level, so the accepted prompts are filtered alike across that level."
        ),
        consequence=(
            "Nothing is recomputed: the exhaustion rates by value are in the details and in "
            "acceptance.csv, and the exhausted units are completed as unclassified in the join."
        ),
        remedy=(
            "Read acceptance.csv for the values with high exhaustion rates; improve the kernel "
            "for them or raise rewrite.validation.maximum_attempts, and materialize the run "
            "again."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_COHORT_AMBIGUOUS",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption="Every cohort in target.cohorts declares a model no other cohort declares.",
        consequence=(
            "The samples of every evaluation that ran the shared model are left out of every "
            "cohort; the analysis still runs and writes its report, and generative analyze "
            "exits with code 3."
        ),
        remedy=(
            "Give each cohort its own target.cohorts.<id>.model, matching the model named in "
            "its logs, and analyze again."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_COHORT_NOT_EVALUATED",
        severities=("ERROR",),
        source="generation",
        effect="invalid",
        assumption=(
            "Every cohort the analysis joins, the primary and each compared one, has evaluated "
            "samples that match its accepted rewrites."
        ),
        consequence=(
            "Every sampling unit of the cohort is completed as unclassified, so its bounds are "
            "the conservative completion of no data; the report is still written, and "
            "generative analyze exits with code 3."
        ),
        remedy=(
            "Pass the cohort's logs with --logs, and make target.cohorts.<id>.model equal the "
            "model those logs ran; then analyze again."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_EVAL_UNMATCHED",
        severities=("WARNING",),
        source="generation",
        effect="recorded",
        assumption="Every evaluation passed to the analysis ran a model some cohort declares.",
        consequence=(
            "The evaluation's samples are ignored, and nothing else is computed differently."
        ),
        remedy=(
            "Leave the evaluation out of --logs, or, if it belongs to a cohort, set that "
            "cohort's target.cohorts.<id>.model to the model the log names."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_FIDELITY_UNEVEN",
        severities=("WARNING",),
        source="generation",
        effect="recorded",
        assumption=(
            "The validator rejects rewrites at similar rates across the values of each level, "
            "so arm fidelity is alike across that level."
        ),
        consequence=(
            "Nothing is recomputed: the validator's rejection rates by value are in the details "
            "and in fidelity.csv, beside the rewrites it judged for each value. The spread is "
            "not tested against chance, so the message names the fewest rewrites judged for "
            "one value."
        ),
        remedy=(
            "Read fidelity.csv for the values the validator rejects most, and how many "
            "rewrites it judged for them; if they are few, the spread can be chance. Otherwise "
            "improve the kernel for them, or review the validator, and materialize the run "
            "again."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_KERNEL_MISMATCH",
        severities=("WARNING",),
        source="generation",
        effect="recorded",
        assumption=(
            "Every rewrite the ledger records was produced under the rewrite procedure the "
            "plan declares: its rewrite section and prompt template, whose kernel fingerprint "
            "the report gives as kernel_sha256."
        ),
        consequence=(
            "Nothing is recomputed: the diagnostic counts the units the rewrite ledger records "
            "under another kernel fingerprint, as when the plan's rewrite section or prompt "
            "template changed after materialization or before a resume, and its details give "
            "the plan's fingerprint and those the ledger records."
        ),
        remedy=(
            "Analyse under the plan and prompt template the run was materialized with, or "
            "materialize again from the start (--overwrite or a new output directory, not "
            "--resume) under the plan as it stands."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_LOG_INCOMPLETE",
        severities=("ERROR",),
        source="generation",
        effect="invalid",
        assumption=(
            "Every evaluation matched to a cohort finished, with status success, so its log "
            "holds every sample it was given."
        ),
        consequence=(
            "The samples the log does hold are still joined, and units it lacks are completed "
            "as unclassified; the report is written, and generative analyze exits with code 3."
        ),
        remedy=(
            "Finish the evaluation in Inspect, by retrying or rerunning it, until its log has "
            "status success, and analyze with that log."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_RETRY_DEPENDENCE",
        severities=("INFO",),
        source="generation",
        effect="recorded",
        assumption=(
            "Each sampling unit's prompt depends only on its own draw, not on the rewrites "
            "accepted for earlier units."
        ),
        consequence=(
            "Nothing is recomputed: the diagnostic gives how many units kept a rewrite only "
            "after a duplicate rejection, out of the evaluable units."
        ),
        remedy=(
            "To remove the dependence, set rewrite.deduplication.on_duplicate: accept, or "
            "rewrite.deduplication.scope: none with near_duplicate: off, in the frozen plan "
            "and materialize again; otherwise report the count beside the intervals."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_ROLLOUT_OVERRUN",
        severities=("ERROR",),
        source="generation",
        effect="invalid",
        assumption=(
            "No sampling unit has more evaluated rollouts than target.rollouts_per_rewrite."
        ),
        consequence=(
            "Each such unit enters the estimates with all its evaluated rollouts as trials; "
            "the report is written, and generative analyze exits with code 3."
        ),
        remedy=(
            "Evaluate with as many epochs as target.rollouts_per_rewrite, and pass only one "
            "evaluation of each cohort with --logs."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_SAMPLE_DUPLICATE",
        severities=("ERROR",),
        source="generation",
        effect="invalid",
        assumption="The logs hold each sample and epoch of a cohort once.",
        consequence=(
            "Only the first occurrence is joined and the repeats are dropped; the report is "
            "written, and generative analyze exits with code 3."
        ),
        remedy=(
            "Pass with --logs only the logs that hold each sample once, leaving out retried, "
            "copied or re-ingested logs."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_SAMPLE_UNMATCHED",
        severities=("ERROR",),
        source="generation",
        effect="invalid",
        assumption=(
            "Every sample of a cohort's logs corresponds to an accepted rewrite in the rewrite "
            "ledger the analysis reads."
        ),
        consequence=(
            "The sample is left out of the join; the report is written, and generative analyze "
            "exits with code 3."
        ),
        remedy=(
            "Analyze with --artifacts pointing at the directory the evaluated dataset was "
            "materialized into, and with the logs of that dataset only."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_SCORE_ABSENT",
        severities=("ERROR",),
        source="generation",
        effect="invalid",
        assumption="The logs carry the score that target.score.name names.",
        consequence=(
            "Every sample is unclassified, so every unit is completed at both extremes; the "
            "report is written, and generative analyze exits with code 3."
        ),
        remedy=(
            "Set target.score.name to the scorer name in the logs, or score the logs with that "
            "scorer, and analyze again."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_SCORE_UNMAPPED",
        severities=("WARNING",),
        source="generation",
        effect="adjusted",
        assumption=(
            "Every score value in the logs is one of target.score.mapping.failure_values or "
            "target.score.mapping.safe_values."
        ),
        consequence=(
            "The samples with an unmapped score are treated as unclassified and completed at "
            "both extremes, which widens the bounds."
        ),
        remedy=(
            "Add the values to target.score.mapping.failure_values or "
            "target.score.mapping.safe_values, or rescore the samples, and analyze again."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_UNIT_UNEVALUATED",
        severities=("WARNING",),
        source="generation",
        effect="adjusted",
        assumption=(
            "Every selected sampling unit has an accepted rewrite and at least one evaluated "
            "sample in the cohort."
        ),
        consequence=(
            "The units without one, exhausted or errored rewrites and unevaluated samples "
            "alike, are completed as unclassified, which widens the bounds."
        ),
        remedy=(
            "Resume the materialization to retry errored rewrites, and evaluate every accepted "
            "rewrite for the cohort; units that exhausted their attempts stay unclassified."
        ),
    ),
    CodeEntry(
        code="GENERATIVE_VALIDATOR_MISMATCH",
        severities=("WARNING",),
        source="generation",
        effect="recorded",
        assumption=(
            "When the plan sets rewrite.validation.invoke, every kept rewrite was judged by "
            "the validator rewrite.validation.validator_id names, and no rewrite by another."
        ),
        consequence=(
            "Nothing is recomputed: the diagnostic names any other validator the attempt "
            "ledger records, and counts the kept rewrites no validator judged, as when "
            "validation.invoke was set after materialization or before a resume. A run in "
            "which no rewrite reached the validator and none was kept has nothing to compare, "
            "and gets no diagnostic."
        ),
        remedy=(
            "Materialize again from the start (--overwrite or a new output directory, not "
            "--resume) under the plan as it stands, with --validator naming an "
            "implementation whose validator_id equals rewrite.validation.validator_id."
        ),
    ),
    CodeEntry(
        code="HIERARCHY_CONFIDENCE_REGION_EMPTY",
        severities=("WARNING",),
        source="analysis",
        effect="withheld",
        assumption=(
            "The raw simultaneous direct intervals of a cohort's groups and hierarchy nodes "
            "share at least one vector of prompt risks, as they do whenever they all cover."
        ),
        consequence=(
            "The cohort's direct intervals for that method and profile are not projected onto "
            "the common hierarchy region and are reported raw."
        ),
        remedy=(
            "Nothing in the inputs needs to change: the raw intervals keep their coverage. Set "
            "outputs.direct_mean_intervals.hierarchy_region: none to report raw intervals "
            "without projecting."
        ),
    ),
    CodeEntry(
        code="HIERARCHY_MEAN_INTERVAL_INVALID",
        severities=("ERROR",),
        source="analysis",
        effect="invalid",
        assumption=(
            "Every hierarchy node's direct interval can be computed from its calibrated alpha "
            "and the node's blocks."
        ),
        consequence=(
            "The hierarchy node has no interval and its row is invalid, and the error makes the "
            "integrity status invalid."
        ),
        remedy=(
            "Correct the input the message names, choose another "
            "outputs.direct_mean_intervals.method, or set "
            "outputs.direct_mean_intervals.hierarchy_nodes: false."
        ),
    ),
    CodeEntry(
        code="HIERARCHY_NODE_UNSAMPLED",
        severities=("WARNING",),
        source="manifest",
        effect="withheld",
        assumption=(
            "Under a sampled design, every hierarchy node has at least one frame item drawn "
            "into the sample, so its conditional risk is identified."
        ),
        consequence=(
            "The nodes without a drawn item get no interval; the other nodes and the groups are "
            "reported as usual."
        ),
        remedy=(
            "Draw a sample that reaches every node, for example by stratifying the draw on the "
            "node and declaring it in design.stratum_column, or set "
            "outputs.direct_mean_intervals.hierarchy_nodes: false."
        ),
    ),
    CodeEntry(
        code="INGEST_FAILED",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "The design manifest, the logs or rollout files and any registration receipt can "
            "be read and normalized under the plan."
        ),
        consequence=(
            "Integrity is invalid and the report has no observations, estimates, risks or "
            "comparisons, with every claim INCONCLUSIVE."
        ),
        remedy=(
            "Fix the input the message names, such as an unreadable manifest, log or receipt, "
            "or a plan option the source cannot serve, and rerun."
        ),
    ),
    CodeEntry(
        code="JUDGE_CALIBRATION_STRATIFIED",
        severities=("INFO",),
        source="judge_labels",
        effect="adjusted",
        assumption=(
            "The human labels are a simple random sample without replacement within each "
            "declared fine stratum, of a size fixed independently of the labels."
        ),
        consequence=(
            "The validated calibration bounds each judge stratum's truly unsafe count with "
            "exact hypergeometric bounds over the declared fine strata, and its conditional "
            "accuracy at those counts with Clopper-Pearson bounds; the asymptotic delta-method "
            "reference interval is not reported."
        ),
        remedy=(
            "None needed when the labels were drawn by the declared strata; otherwise set "
            "inputs.judge_calibration.sampling to the design the labels were drawn by."
        ),
    ),
    CodeEntry(
        code="JUDGE_CALIBRATION_TRANSPORTED",
        severities=("INFO",),
        source="judge_labels",
        effect="adjusted",
        assumption=(
            "The same judge and rubric are at least as sensitive on every prompt of this run as "
            "on the published validation set."
        ),
        consequence=(
            "Judge-corrected rows use the sensitivity transported from the published set, with a "
            "lower end of zero, and are diagnostic sensitivity intervals that decide no claim."
        ),
        remedy=(
            "To correct with evidence that can decide a claim, label a validation sample of this "
            "run and set inputs.judge_calibration.mode: validated."
        ),
        assumptions=("judge_accuracy",),
    ),
    CodeEntry(
        code="JUDGE_CALIBRATION_UNINFORMATIVE",
        severities=("WARNING",),
        source="judge_labels",
        effect="withheld",
        assumption=(
            "The human labels establish a strictly informative judge: the lower bound on "
            "P(truly unsafe | judge said unsafe) exceeds the upper bound on P(truly unsafe | "
            "judge said safe)."
        ),
        consequence=(
            "No validated judge correction is reported for the cohort, or cohort and group, so "
            "its claims are INCONCLUSIVE under a validated calibration."
        ),
        remedy=(
            "Label more rollouts in both judge strata in inputs.judge_calibration."
            "validation_manifest (judge_calibration.validation_manifest in a generative plan), "
            "or use a more accurate judge, and rerun the analysis."
        ),
        assumptions=("judge_accuracy",),
    ),
    CodeEntry(
        code="JUDGE_CONFIGURATION_UNREADABLE",
        severities=("WARNING",),
        source="logs",
        effect="recorded",
        assumption=(
            "With outputs.judge_claims, every run records its scorers and model roles, when it "
            "records them at all, as JSON: a list of scorers and a mapping from each role to a "
            "configuration with a model id."
        ),
        consequence=(
            "Nothing is computed differently and integrity is unchanged; the judges' recorded "
            "configurations leave out the field of that run."
        ),
        remedy=(
            "Record the run's scorers and model_roles as Inspect logs record them, or leave "
            "the field out of the rollout files' runs."
        ),
    ),
    CodeEntry(
        code="JUDGE_CORRECTED_CLAIMS",
        severities=("INFO",),
        source="analysis",
        effect="adjusted",
        assumption=(
            "A threshold claim is about the model's true failure risk, which a certifying "
            "validated judge correction measures."
        ),
        consequence=(
            "The listed claims are decided on the validated judge-corrected bounds, at the union "
            "of the claim family's and the judge calibration's error budgets, with no point "
            "estimate."
        ),
        remedy="None needed: the diagnostic records which claims were retargeted to true risk.",
    ),
    CodeEntry(
        code="JUDGE_CORRECTION_POLICY_CLASSIFIED",
        severities=("INFO",),
        source="analysis",
        effect="adjusted",
        assumption=(
            "Every classified rollout of a judge-corrected row carries a judge verdict, which "
            "the judge's error rates describe."
        ),
        consequence=(
            "The judge corrections of rows holding rollouts a plan policy classified, the rho "
            "corrections and the panel-derived one, credit the judge with no false alarm at the "
            "upper bound and no missed failure at the lower, which widens them, and those rows "
            "have no asymptotic reference intervals."
        ),
        remedy=(
            "None needed: the policy's classification is the truth the estimand defines. To "
            "narrow the correction, rescore the rollouts or classify them by their score, "
            "with inputs.score.errors, missing, content_filter_stops or token_limit."
        ),
        assumptions=("judge_accuracy",),
    ),
    CodeEntry(
        code="JUDGE_CORRECTION_UNAVAILABLE",
        severities=("WARNING",),
        source="analysis",
        effect="withheld",
        assumption=(
            "Every claim's evidence has a certifying validated judge correction when the plan "
            "declares a validated calibration."
        ),
        consequence=(
            "The listed claims are INCONCLUSIVE rather than decided on judge-reported risk."
        ),
        remedy=(
            "Resolve the judge diagnostics that refused the calibration, list the claim's group "
            "in inputs.judge_calibration.groups when the calibration is scoped, and base the "
            "claim on a weighted_mean over a whole cohort or group that weights every classified "
            "rollout equally, in a fixed design."
        ),
        assumptions=("judge_accuracy",),
    ),
    CodeEntry(
        code="JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON",
        severities=("WARNING",),
        source="analysis",
        effect="withheld",
        assumption=(
            "Every comparison is judge-corrected when the plan declares a judge calibration."
        ),
        consequence=(
            "The named comparisons are not judge-corrected and stay on the judge-reported scale; "
            "on the generative path every comparison stays there by design."
        ),
        remedy=(
            "On the fixed-design path, use a comparison whose primary interval exposes both arm "
            "bounds (comparisons[].method: weighted_kl_chernoff, or a comparison_sets entry) and "
            "resolve any calibration refusal for either cohort. Generative comparisons, and "
            "comparisons under inputs.judge_calibration.mode: transported, have no correction."
        ),
    ),
    CodeEntry(
        code="JUDGE_CORRECTION_UNAVAILABLE_FOR_SLICE_FAMILY",
        severities=("WARNING",),
        source="analysis",
        effect="withheld",
        assumption=(
            "The slice family's members are read on the judge-reported scale: the judge's "
            "verdicts are taken as correct within them."
        ),
        consequence=(
            "No member of inference.slice_family is corrected for judge error; the root, nodes "
            "and --slice cuts are corrected as judge_calibration declares."
        ),
        remedy=(
            "Read the family as bounds on the judge-reported failure rate; for a corrected "
            "pointwise bound, also request the slice with --slice."
        ),
    ),
    CodeEntry(
        code="JUDGE_FINE_STRATUM_UNLABELLED_COMPLETED",
        severities=("WARNING",),
        source="judge_labels",
        effect="adjusted",
        assumption="Every declared fine stratum of a labelled judge stratum has a human label.",
        consequence=(
            "Each unlabelled fine stratum's unsafe count is completed from none to all of its "
            "rollouts, which widens the stratified judge correction."
        ),
        remedy=(
            "Label rollouts from the listed fine strata in inputs.judge_calibration."
            "validation_manifest and rerun the analysis."
        ),
    ),
    CodeEntry(
        code="JUDGE_FLIP_INVALID",
        severities=("ERROR",),
        source="judge_labels",
        effect="invalid",
        assumption=(
            "The judge flip manifest can be read and rescores judge-positive completions of the "
            "analysed run under exactly the declared wrappers, as inputs.judge_flip.sampling "
            "states."
        ),
        consequence=(
            "The integrity status is invalid, so no claim can pass, and no flip rate is reported."
        ),
        remedy=(
            "Fix the CSV named by inputs.judge_flip.manifest, or inputs.judge_flip.wrappers and "
            "inputs.judge_flip.sampling, so that they match the rescoring that was done."
        ),
    ),
    CodeEntry(
        code="JUDGE_FLIP_RATE",
        severities=("WARNING", "INFO"),
        source="judge_labels",
        effect="recorded",
        assumption=(
            "The primary judge keeps an unsafe verdict when the same content is shown inside a "
            "content-preserving wrapper."
        ),
        consequence=(
            "The flip rate and its Clopper-Pearson interval are reported, as a WARNING above "
            "inputs.judge_flip.warn_above and as INFO otherwise, and no claim status changes. "
            "The interval treats the rescored completions as independent; under sampling: all "
            "it is descriptive, since completions of one prompt can flip together."
        ),
        remedy=(
            "When the rate is high, validate the judge with inputs.judge_calibration.mode: "
            "validated, or use a judge that is less easily talked out of its verdicts."
        ),
        assumptions=("judge_accuracy",),
    ),
    CodeEntry(
        code="JUDGE_LABELLING_PROTOCOL_VIOLATED",
        severities=("ERROR",),
        source="judge_labels",
        effect="invalid",
        assumption=(
            "The declared labelling queue lists only classified rollouts of the analysed run, "
            "and the validation manifest labels exactly its first rows, as many as a fixed_size "
            "protocol declares."
        ),
        consequence=(
            "The integrity status is invalid, so no claim can pass, and no validated judge "
            "correction is reported."
        ),
        remedy=(
            "Label the queue named by inputs.judge_calibration.protocol.queue in order without "
            "skipping rows, write the queue from the classified rollouts of the logs analysed, "
            "or correct protocol.tier, protocol.stopping or protocol.size to match how the "
            "labels were collected."
        ),
    ),
    CodeEntry(
        code="JUDGE_LABELLING_QUEUE_PARTIAL",
        severities=("WARNING",),
        source="judge_labels",
        effect="recorded",
        assumption=(
            "The declared labelling queue lists every classified rollout of each stratum it "
            "labels, so that only its random order decides which rollouts are labelled. The "
            "strata are the calibration's judge strata, or its declared fine strata in a "
            "stratified design."
        ),
        consequence=(
            "The validated correction is reported as computed; that the queue's rows of the "
            "named strata are a random sample of them is declared, not checked."
        ),
        remedy=(
            "Write the queue as a seeded random permutation of every classified rollout of each "
            "stratum and label it from the front; a queue drawn at random from part of a "
            "stratum stays valid, and the warning records that its draw is declared."
        ),
    ),
    CodeEntry(
        code="JUDGE_LANG_REICZIGEL_UNAVAILABLE",
        severities=("INFO",),
        source="analysis",
        effect="withheld",
        assumption=(
            "The labels hold both truly unsafe and truly safe rollouts, every prompt of the row "
            "has a classified rollout and no rollout of the row was classified by a plan "
            "policy, so the judge's sensitivity and specificity can be estimated and applied."
        ),
        consequence=(
            "The diagnostic Lang-Reiczigel limits are not reported for the row; the exact "
            "validated correction is unaffected."
        ),
        remedy=(
            "None needed for the certified correction; to obtain the limits, label a simple "
            "random sample that includes rollouts of both true classes."
        ),
    ),
    CodeEntry(
        code="JUDGE_SCORE_MAPPING_MISMATCH",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "Every value a declared judge's score takes in the selected logs is covered by "
            "that judge's outcome mapping."
        ),
        consequence=(
            "Integrity is invalid; the error lists the values the judge's mapping does not "
            "cover and withholds no bound itself, and each verdict with such a value counts "
            "as missing, with its own JUDGE_SCORE_UNMAPPED."
        ),
        remedy=(
            "Add each listed value to inputs.judges[].score.mapping, minding that mappings "
            "are type-sensitive, or fix the scorer that produced them and rescore."
        ),
    ),
    CodeEntry(
        code="JUDGE_SCORE_MISSING",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption="The score column of every declared judge is present in the selected logs.",
        consequence=(
            "Integrity is invalid and every verdict of the judge counts as missing, so a "
            "composite that uses it sees a missing vote; no bound is withheld on its account."
        ),
        remedy=(
            "Correct inputs.judges[].score.name to the score the logs record, or rescore the "
            "logs with the judge's scorer."
        ),
    ),
    CodeEntry(
        code="JUDGE_SCORE_UNMAPPED",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "Every value a declared judge's score takes is covered by that judge's outcome "
            "mapping."
        ),
        consequence=(
            "Integrity is invalid and the judge's verdict on the rollout counts as missing; no "
            "bound is withheld on its account."
        ),
        remedy=(
            "Add the value to inputs.judges[].score.mapping, or fix the scorer that produced "
            "it and rescore."
        ),
    ),
    CodeEntry(
        code="JUDGE_STRATUM_UNLABELLED_COMPLETED",
        severities=("WARNING",),
        source="judge_labels",
        effect="adjusted",
        assumption="Both judge strata of a validated calibration have human labels.",
        consequence=(
            "The unlabelled judge stratum's error rate is completed as [0, 1], which widens the "
            "judge correction."
        ),
        remedy=(
            "Label rollouts from the named judge stratum in inputs.judge_calibration."
            "validation_manifest (judge_calibration.validation_manifest in a generative plan) "
            "and rerun the analysis."
        ),
    ),
    CodeEntry(
        code="JUDGE_STRESS_BOX_UNMATCHED",
        severities=("WARNING",),
        source="plan",
        effect="withheld",
        assumption=(
            "The declared accuracy box the judge-error stress uses describes the primary "
            "verdict: one judge with declared accuracy reads the primary score with the primary "
            "mapping, or the primary mapping reads a composite's verdict as it is."
        ),
        consequence=(
            "The judge-error stress reports its tipping lines but no verdict over a declared box."
        ),
        remedy=(
            "Declare accuracy on exactly one judge whose inputs.judges[].score has the primary "
            "score's name and mapping, or map the primary composite's true verdict to failure "
            "and its false verdict to safe."
        ),
    ),
    CodeEntry(
        code="JUDGE_VALIDATION_INVALID",
        severities=("ERROR",),
        source="judge_labels",
        effect="invalid",
        assumption=(
            "The validation manifest, and the validation strata file when declared, can be read "
            "and hold one well-formed row per observation."
        ),
        consequence=(
            "The integrity status is invalid, so no claim can pass, and no validated judge "
            "correction is reported."
        ),
        remedy=(
            "Fix the file named by inputs.judge_calibration.validation_manifest or "
            "inputs.judge_calibration.validation_strata: required columns, no blank or duplicate "
            "observation_id, and labels safe or unsafe."
        ),
    ),
    CodeEntry(
        code="JUDGE_VALIDATION_OUT_OF_SCOPE",
        severities=("WARNING",),
        source="judge_labels",
        effect="recorded",
        assumption=(
            "Every validation label is on a rollout of a group the judge calibration is scoped "
            "to."
        ),
        consequence="The labels outside the scoped groups calibrate nothing.",
        remedy=(
            "Add the labelled rollouts' groups to inputs.judge_calibration.groups, or remove the "
            "labels outside them from the validation manifest."
        ),
    ),
    CodeEntry(
        code="JUDGE_VALIDATION_POLICY_CLASSIFIED",
        severities=("WARNING",),
        source="judge_labels",
        effect="recorded",
        assumption=(
            "Every validation label, and every validation strata row, names a rollout the "
            "judge classified."
        ),
        consequence=(
            "The labels and strata rows of rollouts a plan policy classified calibrate nothing: "
            "such rollouts are in neither judge stratum."
        ),
        remedy=(
            "Draw the labels from the rollouts the judge classified only, leaving out those a "
            "policy classified, such as policy blocks counted safe or missing scores counted "
            "as failures."
        ),
    ),
    CodeEntry(
        code="JUDGE_VALIDATION_STRATA_INCOMPLETE",
        severities=("ERROR",),
        source="judge_labels",
        effect="invalid",
        assumption=(
            "The declared validation strata place every classified observation of the analysed "
            "cohorts."
        ),
        consequence=(
            "The integrity status is invalid, so no claim can pass, and no validated judge "
            "correction is reported."
        ),
        remedy=(
            "Give every classified observation a stratum in the file named by "
            "inputs.judge_calibration.validation_strata."
        ),
    ),
    CodeEntry(
        code="JUDGE_VALIDATION_STRATA_UNMATCHED",
        severities=("ERROR",),
        source="judge_labels",
        effect="invalid",
        assumption=(
            "Every row of the validation strata file names a classified observation of the "
            "analysed logs."
        ),
        consequence=(
            "The integrity status is invalid, so no claim can pass, and no validated judge "
            "correction is reported."
        ),
        remedy=(
            "Remove the rows naming absent or unclassified observations from the file named by "
            "inputs.judge_calibration.validation_strata, or analyse the logs it was written for."
        ),
    ),
    CodeEntry(
        code="JUDGE_VALIDATION_STRATUM_EMPTY",
        severities=("WARNING",),
        source="logs",
        effect="withheld",
        assumption=(
            "The judge returns both verdicts on the classified rollouts of every calibrated "
            "cohort, or cohort and group."
        ),
        consequence=(
            "No validated judge correction is reported for that cohort, or cohort and group, so "
            "its claims are INCONCLUSIVE under a validated calibration."
        ),
        remedy=(
            "Evaluate more rollouts so that both judge verdicts occur, or remove the group from "
            "inputs.judge_calibration.groups; declared scenarios under mode: assumed give a "
            "sensitivity analysis only."
        ),
    ),
    CodeEntry(
        code="JUDGE_VALIDATION_UNMATCHED",
        severities=("ERROR",),
        source="judge_labels",
        effect="invalid",
        assumption=(
            "Every validation label references a rollout of the analysed run whose outcome was "
            "classified."
        ),
        consequence=(
            "On the fixed-design path the integrity status is invalid and no claim can pass; on "
            "the generative path the report is written with the correction from the matched "
            "labels, and generative analyze exits with an error."
        ),
        remedy=(
            "Remove the labels for absent or unclassified rollouts from inputs.judge_calibration."
            "validation_manifest (judge_calibration.validation_manifest in a generative plan), "
            "or analyse the logs the labels were drawn from."
        ),
    ),
    CodeEntry(
        code="LOG_INCOMPLETE",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption="Every selected evaluation ran to completion, with status success.",
        consequence=(
            "Integrity is invalid, so no claim can pass; the run's rollouts still enter the "
            "observations and no bound is withheld on its account."
        ),
        remedy=(
            "Rerun or resume the evaluation until it completes, or leave its log out of the "
            "selected logs."
        ),
        assumptions=("complete_design",),
    ),
    CodeEntry(
        code="MULTI_ARM_CONFIDENCE_REGION_EMPTY",
        severities=("WARNING",),
        source="analysis",
        effect="withheld",
        assumption=(
            "The jointly calibrated arm intervals of a comparison set share, for each arm, at "
            "least one vector of prompt risks, as they do whenever they all cover."
        ),
        consequence=(
            "The comparison set's effects for that method and profile are not derived from the "
            "common arm region, and its rows keep the intervals they had."
        ),
        remedy=(
            "Nothing in the inputs needs to change: the unprojected intervals keep their "
            "coverage. Set comparison_sets[].hierarchy_region: none to skip the projection."
        ),
    ),
    CodeEntry(
        code="PAIR_BROKEN",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "The comparison's pair_key names fields the observations carry, identifies each "
            "rollout once within an arm, and every key appears in both arms."
        ),
        consequence=(
            "The comparison's paired diagnostics (McNemar, Newcombe and cluster bootstrap) are "
            "not reported, and the error makes the integrity status invalid."
        ),
        remedy=(
            "Rerun the missing rollouts so both arms hold the same keys, or correct "
            "comparisons[].pair_key, and give comparisons[].cluster_key when "
            "comparisons[].diagnostic_methods lists fixed_cluster_bootstrap."
        ),
    ),
    CodeEntry(
        code="PAIR_NEWCOMBE_UNAVAILABLE",
        severities=("WARNING",),
        source="analysis",
        effect="withheld",
        assumption=(
            "The completions of a comparison's unscored pair outcomes produce at most 50,000,000 "
            "distinct 2x2 tables, the most newcombe_paired evaluates; 668 pairs unscored in both "
            "arms produce more."
        ),
        consequence=(
            "The comparison's newcombe_paired diagnostic is not reported; its other paired "
            "diagnostics, its interval and the integrity status are unaffected."
        ),
        remedy=(
            "Rescore or rerun the unscored rollouts, or leave newcombe_paired out of "
            "comparisons[].diagnostic_methods: exact_mcnemar and fixed_cluster_bootstrap bound "
            "the completions of any number of unscored pairs."
        ),
    ),
    CodeEntry(
        code="PAIR_UNSCORED_COMPLETED",
        severities=("WARNING",),
        source="logs",
        effect="adjusted",
        assumption="Every pair of the comparison has a classified outcome in both arms.",
        consequence=(
            "The paired diagnostics bound every completion of the unscored outcomes and report "
            "estimate_lower and estimate_upper in place of a point estimate."
        ),
        remedy="Rescore or rerun the unscored rollouts in either arm.",
    ),
    CodeEntry(
        code="PANEL_CORRECTION_NO_COMMON_GROUP",
        severities=("WARNING",),
        source="plan",
        effect="withheld",
        assumption=(
            "The accuracy declarations of a composite's judges cover at least one group in "
            "common."
        ),
        consequence="No panel-derived judge correction is reported.",
        remedy=(
            "Scope inputs.judges[].accuracy.groups so that every judge of the composite covers "
            "a common group, or leave groups out to declare the accuracies for every rollout."
        ),
    ),
    CodeEntry(
        code="PANEL_CORRECTION_UNSUPPORTED_RULE",
        severities=("INFO",),
        source="plan",
        effect="withheld",
        assumption=(
            "The primary composite's rule has a closed-form accuracy box under the declared "
            "dependence blocks, as any and all do."
        ),
        consequence="No panel-derived judge correction is reported for the majority composite.",
        remedy=(
            "Validate the composite's own verdict with inputs.judge_calibration.mode: validated, "
            "or use inputs.composites[].rule: any or all."
        ),
    ),
    CodeEntry(
        code="PANEL_CORRECTION_VERDICT_UNMATCHED",
        severities=("WARNING",),
        source="plan",
        effect="withheld",
        assumption=(
            "The primary mapping reads the primary composite's verdict as it is: true as "
            "failure and false as safe, the verdict the derived accuracy box describes."
        ),
        consequence="No panel-derived judge correction is reported.",
        remedy=(
            "Map the composite's verdict with failure_values: [true] and safe_values: [false] "
            "in inputs.score.mapping, or validate the outcome the mapping defines with "
            "inputs.judge_calibration.mode: validated."
        ),
    ),
    CodeEntry(
        code="PLAN_ANYTIME_BOUNDS_INVALID",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "The anytime-valid bounds are requested on a fixed design, with a declared family, "
            "declared groups and a declared independence profile that blocks by rollout or by "
            "prompt_id, and on groups no judge calibration covers: the bounds are on the "
            "judge-reported failure probability."
        ),
        consequence=(
            "The integrity status is invalid, so no claim can pass, and no anytime-valid bound "
            "is computed."
        ),
        remedy=(
            "Fix outputs.anytime_bounds as the message says: name a family of "
            "multiplicity.families, groups of groups and a profile of "
            "design.independence_profiles that blocks by rollout or prompt_id, under "
            "design.type: fixed, and bound only groups that inputs.judge_calibration.groups "
            "leaves out; or remove outputs.anytime_bounds."
        ),
    ),
    CodeEntry(
        code="PLAN_DIAGNOSTIC_GATE",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption="Every claim rests on a measure that is not declared diagnostic-only.",
        consequence="The integrity status is invalid, so no claim can pass.",
        remedy=(
            "Point claims[].measure at a measure without diagnostic_only: true, or drop "
            "diagnostic_only from the measure in outputs.risk_measures if it is meant to gate."
        ),
    ),
    CodeEntry(
        code="PLAN_DUPLICATE_ID",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "Every group, measure, multiplicity family, comparison and claim identifier is "
            "unique within its kind; comparisons include those generated by comparison_sets, "
            "and claims include adaptive_claims."
        ),
        consequence=(
            "The integrity status is invalid, so no claim can pass, and the duplicated "
            "identifier does not name one result."
        ),
        remedy=(
            "Rename one of the entries that share the identifier named in the diagnostic's "
            "scope."
        ),
    ),
    CodeEntry(
        code="PLAN_EXPECTED_ROLLOUTS_MISSING",
        severities=("ERROR",),
        source="manifest",
        effect="invalid",
        assumption=(
            "When design.require_expected_rollouts is on, the design manifest states in an "
            "expected_rollouts column how many rollouts the evaluation ran for each prompt."
        ),
        consequence=(
            "Integrity is invalid and, since the error names no prompt, every prompt's bounds "
            "are withheld; no rollout count is checked against the manifest."
        ),
        remedy=(
            "Add an expected_rollouts column to the design manifest with each prompt's planned "
            "rollout count, or set design.require_expected_rollouts: false."
        ),
        assumptions=("complete_design",),
    ),
    CodeEntry(
        code="PLAN_INVALID_CLAIM",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "A noisy_valid_test claim has validated judge calibration from labels drawn by "
            "simple random sampling."
        ),
        consequence="The integrity status is invalid, so no claim can pass.",
        remedy=(
            "Set inputs.judge_calibration.mode: validated with "
            "inputs.judge_calibration.sampling: simple_random, or set claims[].method: "
            "corrected_interval."
        ),
    ),
    CodeEntry(
        code="PLAN_INVALID_COMPARISON",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "Every comparison sets two different cohorts against each other on a "
            "weighted_mean measure, and names a pair_key when it asks for paired diagnostics."
        ),
        consequence="The integrity status is invalid, so no claim can pass.",
        remedy=(
            "Give comparisons[].candidate and comparisons[].baseline different cohorts, point "
            "comparisons[].measure at a weighted_mean measure, and set comparisons[].pair_key "
            "when diagnostic_methods lists exact_mcnemar, newcombe_paired or "
            "fixed_cluster_bootstrap."
        ),
    ),
    CodeEntry(
        code="PLAN_INVALID_MULTIPLICITY",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "Simultaneous results take their alpha from a family whose method supports them, "
            "the direct comparisons sharing a family use one coverage_mode and one set of "
            "independence profiles, and no two adaptive claims share a family, since each "
            "takes the family's whole alpha."
        ),
        consequence="The integrity status is invalid, so no claim can pass.",
        remedy=(
            "Give simultaneous comparisons and prompt bounds a bonferroni or "
            "weighted_bonferroni family, and simultaneous outputs.direct_mean_intervals a "
            "bonferroni family, in multiplicity.families[].method; or align coverage_mode and "
            "independence across the comparisons sharing a family, or split them into "
            "separate families; give each adaptive claim a family of its own."
        ),
    ),
    CodeEntry(
        code="PLAN_METHOD_UNUSED",
        severities=("WARNING",),
        source="plan",
        effect="recorded",
        assumption=(
            "A plan that enables direct mean intervals declares a weighted_mean measure for "
            "them to bound."
        ),
        consequence=(
            "No direct mean interval is computed, because direct mean intervals apply only to "
            "weighted_mean measures; the rest of the analysis is unchanged."
        ),
        remedy=(
            "Add a weighted_mean measure to outputs.risk_measures, or set "
            "outputs.direct_mean_intervals.enabled: false."
        ),
    ),
    CodeEntry(
        code="PLAN_NO_RISK_MEASURES",
        severities=("WARNING",),
        source="plan",
        effect="recorded",
        assumption="The plan declares at least one risk measure, so that group risk is reported.",
        consequence="No group risk is reported; prompt-level results are unaffected.",
        remedy=(
            "Add a measure to outputs.risk_measures, for example "
            "{id: mean, type: weighted_mean}."
        ),
    ),
    CodeEntry(
        code="PLAN_POSTHOC",
        severities=("ERROR",),
        source="registration",
        effect="invalid",
        assumption=(
            "A confirmatory plan's analysis.plan_frozen_at is no later than the start of the "
            "earliest selected evaluation."
        ),
        consequence="Integrity is invalid and every prompt bound is withheld.",
        remedy=(
            "Mark the analysis exploratory with analysis.intent, or analyse runs started after "
            "the plan was frozen; moving plan_frozen_at earlier misstates when it was fixed."
        ),
        assumptions=("plan_fixed_in_advance",),
    ),
    CodeEntry(
        code="PLAN_RECEIPT_MISMATCH",
        severities=("ERROR",),
        source="registration",
        effect="invalid",
        assumption="The registration receipt records the canonical hash of this plan.",
        consequence=(
            "Integrity is invalid, every prompt bound is withheld, and the receipt is not used "
            "for the timing check."
        ),
        remedy=(
            "Pass the receipt that hash-plan --register wrote for this plan, or register the "
            "plan as it now reads before the runs and rerun them."
        ),
        assumptions=("plan_fixed_in_advance",),
    ),
    CodeEntry(
        code="PLAN_RECEIPT_POSTHOC",
        severities=("ERROR",),
        source="registration",
        effect="invalid",
        assumption=(
            "The registration receipt of a confirmatory plan is no later than the earliest "
            "start time of the selected evaluations."
        ),
        consequence=(
            "Integrity is invalid and every prompt's bounds are withheld, since the error "
            "blocks the whole analysis."
        ),
        remedy=(
            "Register the plan with hash-plan --register before the runs start and rerun "
            "them, or set analysis.intent: exploratory."
        ),
        assumptions=("plan_fixed_in_advance",),
    ),
    CodeEntry(
        code="PLAN_REGISTERED",
        severities=("INFO",),
        source="registration",
        effect="recorded",
        assumption=(
            "The registration receipt's time is anchored by an external timestamp a reader "
            "trusts, such as a signed commit or an RFC 3161 token."
        ),
        consequence=(
            "Nothing changes: the diagnostic records that the receipt matches the plan and is "
            "not later than the earliest observed start, in place of PLAN_SELF_DECLARED."
        ),
        remedy=(
            "No change is needed; a reader who relies on the timing checks the receipt's "
            "external anchor."
        ),
    ),
    CodeEntry(
        code="PLAN_SAMPLED_COMPARISON_UNSUPPORTED",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "Comparisons are made only under a fixed design, whose arms assume every declared "
            "prompt was evaluated."
        ),
        consequence="The integrity status is invalid, so no claim can pass.",
        remedy=(
            "Remove comparisons and comparison_sets from a plan with design.type: sampled, or "
            "analyse the prompts as a fixed design."
        ),
    ),
    CodeEntry(
        code="PLAN_SAMPLED_FRONTIER_UNSUPPORTED",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "Under frontier_sidak, the intervals of the hierarchy nodes of one level are "
            "independent, each node's drawn frame items forming blocks no other node shares. "
            "Under a sampled design a frame item that was not drawn forms no block, and a "
            "fixed-size draw makes the nodes that share a stratum dependent."
        ),
        consequence=(
            "The plan is refused: frontier_sidak calibration is not yet supported under a "
            "sampled design, and the error makes the integrity status invalid."
        ),
        remedy=(
            "Set outputs.direct_mean_intervals.hierarchy_calibration to flat_bonferroni, which "
            "holds whatever the dependence between nodes, or set hierarchy_nodes to false."
        ),
    ),
    CodeEntry(
        code="PLAN_SAMPLED_MEASURE_UNSUPPORTED",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "Under a sampled design every risk measure is a weighted_mean, the only measure a "
            "sample of the frame can estimate."
        ),
        consequence="The integrity status is invalid, so no claim can pass.",
        remedy=(
            "Keep only weighted_mean measures in outputs.risk_measures under design.type: "
            "sampled, or analyse the prompts as a fixed design."
        ),
    ),
    CodeEntry(
        code="PLAN_SAMPLED_PROMPT_BOUNDS",
        severities=("WARNING",),
        source="plan",
        effect="recorded",
        assumption=(
            "Simultaneous prompt bounds under a sampled design are read as covering the "
            "sampled frame items only, not the whole frame."
        ),
        consequence=(
            "The simultaneous prompt bounds are reported as computed, over the sampled items "
            "only."
        ),
        remedy=(
            "Read the prompt bounds as covering the sampled items, or set "
            "outputs.prompt_bounds.simultaneous: false or outputs.prompt_bounds.enabled: false."
        ),
    ),
    CodeEntry(
        code="PLAN_SAMPLED_REQUIRES_DIRECT_MEAN",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "A sampled design enables direct mean intervals, since the propagated prompt "
            "envelope has no bound for unsampled frame items."
        ),
        consequence="The integrity status is invalid, so no claim can pass.",
        remedy="Set outputs.direct_mean_intervals.enabled: true, with its family.",
    ),
    CodeEntry(
        code="PLAN_SAMPLED_SENSITIVITY_PROFILES",
        severities=("WARNING",),
        source="plan",
        effect="withheld",
        assumption=(
            "A sampled design's frame intervals take the frame item as the independent block, "
            "whatever independence profiles the plan declares."
        ),
        consequence=(
            "No interval and no claim verdict is reported under the declared sensitivity "
            "profiles; the frame intervals and the claims' statuses are unaffected."
        ),
        remedy=(
            "Remove outputs.direct_mean_intervals.independence.sensitivities under design.type: "
            "sampled, or analyse the prompts as a fixed design to see verdicts under them."
        ),
    ),
    CodeEntry(
        code="PLAN_SELF_DECLARED",
        severities=("WARNING",),
        source="registration",
        effect="recorded",
        assumption=(
            "The confirmatory plan's freeze time is backed by an external registration, not "
            "only by the time its author wrote in analysis.plan_frozen_at."
        ),
        consequence=(
            "Nothing is computed differently and integrity is unchanged; the report records "
            "that the plan's timing rests on its author's word."
        ),
        remedy=(
            "Register the plan with hash-plan --register before the runs, anchor the receipt, "
            "and pass it to analyze or validate with --registration-receipt."
        ),
        assumptions=("plan_fixed_in_advance",),
    ),
    CodeEntry(
        code="PLAN_SINGLE_VALUED_BOUND_UNSUPPORTED",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "A single-valued bound is requested only under a fixed design without judge "
            "calibration: a sampled design's frame bounds are not a family of node bounds, and "
            "with judge calibration the bound would describe the judge's risk, not the "
            "corrected risk."
        ),
        consequence="The integrity status is invalid, so no claim can pass.",
        remedy=(
            "Set outputs.direct_mean_intervals.single_valued_bound: false, or use it only with "
            "design.type: fixed and no inputs.judge_calibration."
        ),
    ),
    CodeEntry(
        code="PLAN_UNKNOWN_REFERENCE",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "Every cohort, group, measure, comparison and multiplicity family the plan "
            "references is declared."
        ),
        consequence=(
            "The integrity status is invalid, so no claim can pass; for a validated judge "
            "calibration without a declared family no judge correction is computed."
        ),
        remedy=(
            "Declare the identifier the diagnostic names, or correct the reference: the "
            "cohorts, group, measure and family of comparisons and comparison_sets, claims and "
            "adaptive_claims; outputs.prompt_bounds.family; outputs.direct_mean_intervals.family "
            "and groups; inputs.judge_calibration.family and groups; inputs.panel_groups; "
            "inputs.judges[].accuracy.groups."
        ),
    ),
    CodeEntry(
        code="PLAN_WORST_OF_K_INTERVAL_UNSUPPORTED",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "A direct worst_of_k interval is requested under a fixed design, with direct mean "
            "intervals enabled without frontier_sidak calibration and a primary independence "
            "profile that blocks by rollout."
        ),
        consequence=(
            "The integrity status is invalid, so no claim can pass. The measure's group rows "
            "get no direct interval: they keep the propagated prompt envelope as primary and "
            "warn why."
        ),
        remedy=(
            "Set design.type: fixed, outputs.direct_mean_intervals.enabled: true with "
            "hierarchy_calibration: flat_bonferroni and a primary profile blocking by rollout; "
            "or drop direct_interval from the worst_of_k measure in outputs.risk_measures."
        ),
    ),
    CodeEntry(
        code="PROMPT_MISSING",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption="Every manifest prompt of a fixed design has observations in every cohort.",
        consequence=(
            "Integrity is invalid and the missing prompt's bounds in that cohort are withheld, "
            "with any result that needs that prompt's complete rollout ledger."
        ),
        remedy=(
            "Run the missing prompt in that cohort and add its logs, or, when the prompt was "
            "never meant to run, remove it from the manifest or set "
            "design.require_complete_manifest: false."
        ),
        assumptions=("complete_design",),
    ),
    CodeEntry(
        code="PROMPT_UNEXPECTED",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption="Every prompt the selected logs evaluated is declared in the design manifest.",
        consequence=(
            "Integrity is invalid and the prompt's rollouts are left out of the observations; "
            "the bounds of the manifest prompts are not withheld on its account."
        ),
        remedy=(
            "Add the prompt to the design manifest, or leave the logs that evaluated it out of "
            "the selection or out of every cohort's match."
        ),
    ),
    CodeEntry(
        code="PROVENANCE_UNOBSERVED",
        severities=("WARNING",),
        source="logs",
        effect="recorded",
        assumption=(
            "Every evaluation of a cohort records each field the cohort lists in "
            "require_uniform."
        ),
        consequence=(
            "Nothing is computed differently and integrity is unchanged; a field absent from "
            "every run goes unchecked, and one absent from only some runs also raises "
            "CONFIG_DRIFT."
        ),
        remedy=(
            "Record the field in every run of the cohort, or correct its name in "
            "cohorts.<id>.require_uniform; under schema 3.0 a source field is written "
            "attributes.<name>."
        ),
        assumptions=("run_stability",),
    ),
    CodeEntry(
        code="RISK_MEASURE_INVALID",
        severities=("ERROR",),
        source="plan",
        effect="invalid",
        assumption=(
            "Every risk measure can be evaluated on every group: its threshold source exists in "
            "the manifest with finite values, top_k_mean has equal prompt weights, and k does "
            "not exceed the group's prompts."
        ),
        consequence=(
            "The measure's estimate or propagated bounds are not reported for that cohort and "
            "group, and the error makes the integrity status invalid."
        ),
        remedy=(
            "Correct the measure in outputs.risk_measures[] (its threshold, threshold_column or "
            "k), or give its groups equal prompt weights through groups[].weights for top_k_mean."
        ),
    ),
    CodeEntry(
        code="ROLLOUT_DUPLICATE",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "The fields of design.rollout_key identify each rollout uniquely across the "
            "selected logs."
        ),
        consequence=(
            "Integrity is invalid and the prompt's bounds are withheld in every cohort; the "
            "duplicate rollout stays in the observations."
        ),
        remedy=(
            "Remove logs selected twice or overlapping runs from the selection, or set "
            "design.rollout_key to fields that tell the rollouts apart."
        ),
    ),
    CodeEntry(
        code="ROLLOUT_EXCESS",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "Each prompt has no more rollouts in each cohort than the manifest's "
            "expected_rollouts states."
        ),
        consequence=(
            "Integrity is invalid and the prompt's bounds in that cohort are withheld, with "
            "any result that needs that prompt's complete rollout ledger."
        ),
        remedy=(
            "Leave the extra evaluations out of the input logs, or correct expected_rollouts "
            "in the manifest when it misstates the planned count."
        ),
        assumptions=("complete_design",),
    ),
    CodeEntry(
        code="ROLLOUT_SHORTFALL",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "Each prompt has at least the manifest's expected_rollouts rollouts in each cohort."
        ),
        consequence=(
            "Integrity is invalid and the prompt's bounds in that cohort are withheld, with "
            "any result that needs that prompt's complete rollout ledger."
        ),
        remedy=(
            "Rerun the missing rollouts and add their logs, or correct expected_rollouts in "
            "the manifest when it misstates the planned count."
        ),
        assumptions=("complete_design",),
    ),
    CodeEntry(
        code="RUN_STATUS_UNOBSERVED",
        severities=("WARNING",),
        source="logs",
        effect="degraded",
        assumption=(
            "Every selected evaluation reports its status, so that it can be checked to have "
            "completed."
        ),
        consequence=(
            "Integrity is degraded: the evaluation is assumed complete and its rollouts are "
            "analysed."
        ),
        remedy="Record each run's status in the source, such as status in rollout files.",
        assumptions=("complete_design",),
    ),
    CodeEntry(
        code="SAMPLE_DRAWS_MISMATCH",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "Under design.sampling: fixed_size, each cohort has observations for exactly the "
            "number of frame items the inclusion probabilities plan for each stratum."
        ),
        consequence=(
            "Integrity is invalid, so no claim can pass: the declared inclusion probabilities "
            "misstate the sample the estimate rests on."
        ),
        remedy=(
            "Supply the logs of every drawn frame item and no others, or correct the "
            "inclusion probability column to the design actually drawn."
        ),
        assumptions=("probability_sample",),
    ),
    CodeEntry(
        code="SAMPLE_DRAWS_NOT_INTEGER",
        severities=("ERROR",),
        source="manifest",
        effect="invalid",
        assumption=(
            "Under design.sampling: fixed_size, each stratum's inclusion probability times its "
            "number of frame items is a whole number of draws."
        ),
        consequence=(
            "Integrity is invalid, so no claim can pass, and the stratum's realized draws are "
            "not checked."
        ),
        remedy=(
            "Correct the inclusion probability column to draws divided by stratum size, or "
            "name the column that defines the strata in design.stratum_column."
        ),
        assumptions=("probability_sample",),
    ),
    CodeEntry(
        code="SAMPLE_EMPTY",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption="Each cohort of a sampled design has observations for some frame item.",
        consequence=(
            "Integrity is invalid and every bound of the empty cohort is withheld."
        ),
        remedy=(
            "Check that cohorts.<id>.match selects the cohort's evaluations and that their "
            "item ids match the manifest's prompt_id, or run the sample."
        ),
    ),
    CodeEntry(
        code="SAMPLE_ERROR",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption="No sample ends in an error when inputs.score.errors is block.",
        consequence=(
            "Integrity is invalid and the errored rollout's prompt has its bounds withheld in "
            "that cohort."
        ),
        remedy=(
            "Rerun the errored samples, or choose the inputs.score.errors policy that matches "
            "where the errors come from."
        ),
        assumptions=("sample_errors",),
    ),
    CodeEntry(
        code="SAMPLE_ERROR_EXCLUDED",
        severities=("WARNING",),
        source="logs",
        effect="degraded",
        assumption=(
            "A rollout that inputs.score.errors: exclude leaves out of the bounds fails as "
            "often as the classified rollouts of its prompt."
        ),
        consequence=(
            "The bounds cover only the classified rollouts; if the errors are linked to the "
            "outcome they are biased, and integrity is degraded."
        ),
        remedy=(
            "Rerun the errored samples, or set inputs.score.errors to conservative_bounds so "
            "that the bounds cover both outcomes of each errored rollout."
        ),
        assumptions=("sample_errors",),
    ),
    CodeEntry(
        code="SAMPLE_ERROR_UNCLASSIFIED",
        severities=("WARNING",),
        source="logs",
        effect="degraded",
        assumption=(
            "Every sample error has a known outcome: the analysis can tell what the system "
            "would have done had the error not occurred."
        ),
        consequence=(
            "The errored rollout's outcome is unknown and its prompt's bounds cover both "
            "outcomes, and integrity is degraded."
        ),
        remedy=(
            "Rerun the errored samples, or set inputs.score.errors to count_as_failure or "
            "exclude when that matches where the errors come from."
        ),
        assumptions=("sample_errors",),
    ),
    CodeEntry(
        code="SAMPLE_SELECTION_MISMATCH",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "With design.selection_column, each cohort has observations for exactly the frame "
            "items that column marks as selected by the draw."
        ),
        consequence=(
            "Integrity is invalid and every bound of the cohort is withheld: the items "
            "evaluated are not the recorded draw, so the inclusion probabilities do not "
            "describe them."
        ),
        remedy=(
            "Evaluate every selected frame item and no other, or correct the selection column "
            "to the draw actually made, at random and before any outcome was seen."
        ),
        assumptions=("probability_sample",),
    ),
    CodeEntry(
        code="SAMPLE_SIZE_IMPLAUSIBLE",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "The declared inclusion probabilities describe the draw: each cohort's number of "
            "observed frame items is within four standard deviations of their sum."
        ),
        consequence=(
            "Integrity is invalid and every bound of the cohort is withheld, since "
            "probabilities that misdescribe the draw bias the estimate."
        ),
        remedy=(
            "Correct the inclusion probability column to the probabilities the draw used, or "
            "supply the logs of every drawn frame item."
        ),
        assumptions=("probability_sample",),
    ),
    CodeEntry(
        code="SAMPLE_STRATUM_INCONSISTENT",
        severities=("ERROR",),
        source="manifest",
        effect="invalid",
        assumption=(
            "Under design.sampling: fixed_size, every frame item of a stratum has the same "
            "inclusion probability, its stratum's draws divided by its size."
        ),
        consequence=(
            "Integrity is invalid, so no claim can pass, and the stratum's realized draws are "
            "not checked."
        ),
        remedy=(
            "Name the column that defines the strata in design.stratum_column, correct the "
            "inclusion probability column, or use design.sampling: poisson when items were "
            "drawn independently."
        ),
        assumptions=("probability_sample",),
    ),
    CodeEntry(
        code="SCORE_MAPPING_MISMATCH",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "Every value the selected score takes in the selected logs is classified by "
            "inputs.score.mapping as a failure or as safe."
        ),
        consequence=(
            "Integrity is invalid; the error lists the values the mapping does not cover and "
            "withholds no bound itself, while each rollout with such a value has its own "
            "SCORE_UNMAPPED, which withholds its prompt's bounds in its cohort."
        ),
        remedy=(
            "Add each listed value to failure_values or safe_values of inputs.score.mapping, "
            "minding that mappings are type-sensitive, or fix the scorer that produced them."
        ),
    ),
    CodeEntry(
        code="SCORE_MISSING",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "The selected logs carry the score inputs.score.name names, and under "
            "inputs.score.missing: block every rollout has it."
        ),
        consequence=(
            "Integrity is invalid; a score absent from the logs withholds every prompt bound, "
            "and a rollout without a score under block withholds its prompt's bounds in its "
            "cohort."
        ),
        remedy=(
            "Correct inputs.score.name or rescore the rollouts without a score; otherwise "
            "choose another inputs.score.missing policy, such as conservative_bounds."
        ),
        assumptions=("missing_scores",),
    ),
    CodeEntry(
        code="SCORE_MISSING_BOUNDED",
        severities=("WARNING",),
        source="logs",
        effect="degraded",
        assumption="Every rollout has a value of the selected score.",
        consequence=(
            "The unscored rollout's outcome is unknown and its prompt's bounds cover both "
            "outcomes, and integrity is degraded."
        ),
        remedy=(
            "Rescore the unscored rollouts, or set inputs.score.missing to another policy "
            "when its assumption about the missing scores holds."
        ),
        assumptions=("missing_scores",),
    ),
    CodeEntry(
        code="SCORE_MISSING_EXCLUDED",
        severities=("WARNING",),
        source="logs",
        effect="degraded",
        assumption=(
            "A rollout that inputs.score.missing: exclude leaves out of the bounds fails as "
            "often as the classified rollouts of its prompt: its score is missing at random."
        ),
        consequence=(
            "The bounds cover only the classified rollouts; if scores are missing for a "
            "reason linked to the outcome they are biased, and integrity is degraded."
        ),
        remedy=(
            "Rescore the unscored rollouts, or set inputs.score.missing to "
            "conservative_bounds so that the bounds cover both outcomes of each of them."
        ),
        assumptions=("missing_scores",),
    ),
    CodeEntry(
        code="SCORE_UNMAPPED",
        severities=("ERROR",),
        source="logs",
        effect="invalid",
        assumption=(
            "Every value of the selected score is classified by inputs.score.mapping as a "
            "failure or as safe."
        ),
        consequence=(
            "Integrity is invalid and the rollout's prompt has its bounds withheld in that "
            "cohort."
        ),
        remedy=(
            "Extend inputs.score.mapping to cover the value, minding that mappings are "
            "type-sensitive, or fix the scorer that produced it."
        ),
    ),
    CodeEntry(
        code="START_TIME_UNOBSERVED",
        severities=("WARNING",),
        source="logs",
        effect="degraded",
        assumption=(
            "Every selected evaluation of a confirmatory analysis reports its start time, so "
            "that the plan can be checked to predate it."
        ),
        consequence=(
            "Integrity is degraded: the plan's freeze time is checked only against the "
            "evaluations that report a start time."
        ),
        remedy="Record each run's start time in the source, such as started_at in rollout files.",
        assumptions=("plan_fixed_in_advance",),
    ),
    CodeEntry(
        code="TAIL_MEASURES_UNINFORMATIVE",
        severities=("WARNING",),
        source="analysis",
        effect="recorded",
        assumption=(
            "Each prompt has enough rollouts that the tail and threshold measures built from "
            "the prompts' simultaneous bounds can lie within report.precision_target of their "
            "estimates, since their precision is set by the rollouts on each prompt and not by "
            "the total rollout count."
        ),
        consequence=(
            "The tail and threshold measures are reported as computed, but in the cohort and "
            "group named, at the lower median of the rollouts per prompt, even if no prompt "
            "failed the named measures' upper bounds would lie more than the precision target "
            "above their estimates, so their intervals can only be wide."
        ),
        remedy=(
            "Run about the rollouts per prompt the diagnostic names for each measure, or read "
            "the named measures as uninformative at this budget."
        ),
    ),
    CodeEntry(
        code="TAIL_MEASURE_SINGLE_ROLLOUT",
        severities=("WARNING",),
        source="logs",
        effect="degraded",
        assumption=(
            "A claim on a cvar, maximum, top_k_mean, entropic, threshold_exceedance or "
            "threshold_excess measure rests on repeated rollouts per prompt, since these "
            "measures are nonlinear in each prompt's failure probability and one rollout "
            "estimates it only as 0 or 1."
        ),
        consequence=(
            "Integrity is degraded: the claim is evaluated, but with one rollout for at least "
            "half the prompts of its cohort and group the measure is driven by single trials "
            "and is misleading or vacuous."
        ),
        remedy=(
            "Run repeated rollouts per prompt, or make the claim on a weighted_mean measure, "
            "whose interval handles one rollout per prompt."
        ),
    ),
    CodeEntry(
        code="TARGET_POLICY_BLOCKED_REFUSAL",
        severities=("INFO",),
        source="logs",
        effect="adjusted",
        assumption=(
            "A provider content-filter stop before the target delivered any content in the "
            "sample is a refusal of the end-to-end system, as "
            "inputs.score.content_filter_stops: failure declares."
        ),
        consequence=(
            "The rollout counts as a failure, a refusal under an over-refusal estimand, "
            "whatever its score."
        ),
        remedy=(
            "Set inputs.score.content_filter_stops: score to let the score decide such "
            "rollouts."
        ),
        assumptions=("content_filter_stops",),
    ),
    CodeEntry(
        code="TARGET_POLICY_BLOCKED_SAFE",
        severities=("INFO",),
        source="logs",
        effect="adjusted",
        assumption=(
            "A target-side provider policy block before the target delivered any content in "
            "the sample is a safe outcome of the end-to-end system."
        ),
        consequence=(
            "The rollout counts as safe: a policy-blocked sample error under "
            "inputs.score.errors: end_to_end, whatever its score, or a content-filter stop "
            "under inputs.score.content_filter_stops: safe whose score is missing or safe."
        ),
        remedy=(
            "Set inputs.estimand: underlying_model, or another inputs.score.errors policy, to "
            "keep blocked errors unclassified, and inputs.score.content_filter_stops: score "
            "to let the score decide filter stops."
        ),
        assumptions=("content_filter_stops", "sample_errors"),
    ),
    CodeEntry(
        code="TOKEN_LIMIT_REACHED",
        severities=("INFO",),
        source="logs",
        effect="adjusted",
        assumption=(
            "An unscored answer that stopped at the generation token limit counts as "
            "inputs.score.token_limit declares."
        ),
        consequence=(
            "The rollout counts as safe or a failure under token_limit: safe or failure, and "
            "as a missing score under token_limit: missing, and is counted in n_token_limit."
        ),
        remedy=(
            "Raise the token limit and rerun the truncated rollouts, or choose another "
            "inputs.score.token_limit classification."
        ),
        assumptions=("token_limit",),
    ),
    CodeEntry(
        code="TRANSCRIPT_UNREAD",
        severities=("WARNING",),
        source="logs",
        effect="recorded",
        assumption=(
            "The transcript of every log with sample errors can be read, so that each error's "
            "stage and provider error code are known."
        ),
        consequence=(
            "Nothing is computed differently: the log's sample errors have an unknown stage, "
            "so none counts as a policy block and each follows inputs.score.errors."
        ),
        remedy=(
            "Restore or repair the log file so that its transcript can be read, and rerun the "
            "analysis."
        ),
    ),
)

CODES: Final[Mapping[str, CodeEntry]] = MappingProxyType({entry.code: entry for entry in _CODES})


def checked_by(assumption: str) -> tuple[str, ...]:
    """The codes that report evidence against an assumption of ``ASSUMPTIONS``, sorted."""

    if assumption not in ASSUMPTIONS:
        raise KeyError(f"no assumption {assumption!r} in the registry")
    return tuple(sorted(code for code, entry in CODES.items() if assumption in entry.assumptions))

__all__ = [
    "ASSUMPTIONS",
    "CODES",
    "EFFECTS",
    "SOURCE_KINDS",
    "AssumptionEntry",
    "CodeEntry",
    "Effect",
    "Severity",
    "SourceKind",
    "checked_by",
]
