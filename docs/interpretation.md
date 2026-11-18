# Interpreting results

[What results establish](what-results-establish.md) is the one-page summary of what a result
shows and what it does not. This page gives the details.

## Start with the risk estimates

The default report presents estimates and uncertainty intervals without turning them into a
package-level safety verdict. Interpret each result against thresholds that were chosen before
examining the evaluation. The most important choices are the cohort, fixed design group, policy
weights, risk measure, and confidence level.

Threshold decisions are an optional presentation layer enabled with
`report.include_decisions: true`. The option hides decisions rather than withholding them:
`report.json` holds the overall decision and every claim's status either way. When enabled:

`PASS` means the predeclared upper bound is at or below the acceptable threshold and all required integrity gates passed.

`FAIL` means a predeclared lower bound establishes unacceptable risk, a mandatory claim failed, or a verified zero-tolerance event occurred.

`INCONCLUSIVE` means neither conclusion was established or the evidence did not satisfy the design contract. It is not a near-pass.

When a validated judge calibration is declared, these decisions are made against true failure
risk rather than judge-reported risk, and the claim records both intervals. See
[Judge misclassification](judge-misclassification.md).

### Exploratory analyses

An analysis with `intent: exploratory` certifies nothing, so every claim and the report's
decision are `INCONCLUSIVE`. When integrity is valid and a claim's bounds are certifying, the
claim also has a descriptive status: the status a confirmatory analysis of the same bounds
would give at the plan's thresholds. `report.json` records it as `descriptive_status`,
and, with `report.include_decisions: true`, `report-full.md` shows it beside the status, for example
`` `INCONCLUSIVE` (descriptive: `PASS`) ``, and `report.html` as a dashed badge. A claim without
certifying bounds, or a report whose integrity is invalid, has none (`null`).

A descriptive status is for reading the bounds. It is not a certification: an exploratory plan
need not precede the evaluation it analyses, so a descriptive `PASS` is not the predeclared
decision a confirmatory `PASS` is. To certify, freeze a confirmatory plan and evaluate again.

### Verdicts at other thresholds

Each claim also records `verdict_range`: the thresholds at which the same bounds would give
`PASS` or `FAIL`. The claim is `PASS` for every `acceptable_upper` at or above
`pass_acceptable_upper`, and `FAIL` for every `unacceptable_lower` at or below
`fail_unacceptable_lower`. When `pass_strict` or `fail_strict` is true, the boundary itself is
excluded. Every other pair of thresholds gives `INCONCLUSIVE`. A `null` boundary means that no
threshold the plan allows gives that verdict. For an interval claim the boundaries are the upper
and lower bounds. For the noisy-but-valid test they are the thresholds at which its critical
values meet the judge-reported bounds. The report's claims table states the same range, with
each boundary rounded so that the statement still holds.

In an exploratory analysis the claim is `INCONCLUSIVE` at every threshold. Its range, like its
descriptive status, is what a confirmatory analysis of the same bounds would give, and it
certifies nothing.

The range shows how far the verdict is from changing. The error rate of a verdict read off it
is bounded at any thresholds, because a wrong `PASS` or `FAIL` needs the true risk outside the
bounds. A threshold chosen after seeing the bounds cannot show, though, that the requirement
came before the evidence. To decide at revised thresholds and keep that visible, give the
analysis a [decision policy](decision-policy.md) (experimental): it records each version of the
thresholds with the time it was set, and leaves the plan and its hash as frozen.

### Verdicts under sensitivity profiles

A claim's status is decided under its evidence's primary independence profile. When the plan
declares sensitivity profiles, `profile_verdicts` in `report.json`, a table in `report-full.md` and
the HTML claim card also give the verdict each sensitivity profile's interval would reach at
the same thresholds. These verdicts never change the claim's status or the decision. A
`PASS` that is `INCONCLUSIVE` under a coarser profile, such as `prompt`, rests on the primary
profile's independence assumption. See
[Verdicts by profile](independence-profiles.md#verdicts-by-profile).

## Integrity states

`valid` means every required outcome was classified and no structural integrity error occurred.

`degraded` means one or more attempted outcomes are unclassified, but every missing binary label is
covered by conservative bounds. Degraded evidence may certify a claim only when the claim passes
under the adverse completion of all unknown outcomes.

Integrity is also `degraded` when the source gives no data for a safeguard, so that it cannot be
checked. Each case is a warning that names the evaluations concerned:

| Code | The source gives no | So this is not checked |
|---|---|---|
| `RUN_STATUS_UNOBSERVED` | run status | that the evaluation completed; it is assumed complete |
| `START_TIME_UNOBSERVED` | start time, under a confirmatory plan | that the plan was frozen before the evaluation started |
| `CONFIG_UNOBSERVED` | configuration field, nor any field the cohort requires uniform | that the evaluation's configuration matches the rest of its cohort |

Inspect logs always carry a status, a start time and a model, so these arise from other sources,
such as [rollout files](rollout-files.md) that leave the fields out.

Integrity is `degraded` too when a claim uses a `cvar`, `maximum` or `entropic` measure and the
median prompt of its cohort and group has one classified rollout: `TAIL_MEASURE_SINGLE_ROLLOUT`.
Every per-prompt rate is then 0 or 1, so the measure is misleading or vacuous. Run repeated
rollouts per prompt, or claim on a weighted mean.

`invalid` means a structural or strict-policy error prevents certification. Examples include an
incomplete design, an unmapped score, configuration drift, a post hoc confirmatory plan, or a
missing/error policy explicitly set to `block`.

Under the default end-to-end estimand, a target-side provider policy block with no delivered
content is a classified safe system outcome. A grader-side block is not: the target output already
exists, so its safety label remains unknown.

[Diagnostic codes and assumptions](diagnostics.md) lists every code a report can carry, with the
assumption it reports on, its effect on integrity and results, and the remedy.

## Prompt-level bounds

Each prompt's bound concerns repeated stochastic rollouts of that exact prompt under the recorded configuration. It does not concern paraphrases, unseen prompts, or deployment traffic.

Simultaneous prompt bounds are wider than individual bounds because they protect the aggregate calculation across all relevant prompts.

The default `coverage_mode: two_sided` divides the family error budget equally between tails. At
family alpha `0.05`, the complete lower-to-upper range has at least 95% simultaneous coverage,
while each direction separately has at least 97.5% coverage. The compatibility mode
`per_direction` spends alpha `0.05` on each side, giving 95% directional coverage but only a 90%
combined floor. Reports display both figures explicitly.

When attempted rollouts have missing labels, the lower endpoint completes every unknown as safe
and the upper endpoint completes every unknown as a failure before applying the exact binomial
bound. Aggregate fixed-design measures propagate those endpoint vectors through the declared
monotone risk function.

## Risk measures

### Two uncertainty constructions

For weighted means, the primary interval may be a direct variance-adaptive
weighted-empirical-Bernstein interval. It targets the declared weighted mean without requiring
simultaneous precision for every prompt. Its guarantee assumes independent attempted rollouts
across the complete fixed design. The report can show weighted Hoeffding as a prespecified
range-only reference; empirical Bernstein is not guaranteed to be shorter in every dataset.

The report retains the propagated exact prompt envelope as a reference. That construction is
usually wider, but it supports every coordinatewise monotone risk measure and does not require
independence between prompts. Expand **Compare interval methods** to see both ranges and their
assumptions.

When enabled, **Hierarchy marginal risks** reports the same conditional weighted-mean estimand at
every declared non-prompt node. The original group weights are normalized among the node's fixed
descendants. These rows describe the fixed design and do not estimate prevalence in a population
of unseen tasks or questions.

### Where effects sit

A headline row averages over the hierarchy, so an effect concentrated in one node can look
small there: a few points of difference overall can be a large difference in one category and
none in the others. When a report has hierarchy rows, `report-full.md` lists, right after the
claims:

- every hierarchy node of a comparison whose interval excludes no difference (0 for a risk
  difference or a relative risk reduction, 1 for a risk ratio), beside the comparison's overall
  effect;
- every hierarchy node whose estimate is above the acceptable limit of a claim on the same
  evidence, with whether the whole interval or only the estimate is above. An aggregate claim
  reaches the node rows only through a `weighted_mean` measure, the estimand they report; a
  comparison claim reaches its comparison's node rows. The list is hidden with the decisions.

These rows are pointers into the hierarchy tables, with the same intervals and coverage; they
are not further claims. A claim decided on a judge-corrected interval is compared with
judge-reported node rows. A claim that passes on the average while nodes sit above its limit
passes as declared: the list makes that visible, it does not change the decision.

The HTML report presents these results as a collapsible tree. Choose one **Independence profile**
for the whole report. Every interval glyph uses the same 0–100% risk axis. The strong blue band is
the displayed simultaneous interval and the faint gray band is its raw concentration interval
before hierarchy projection. The estimate marker is descriptive; the interval carries the
coverage statement. The calibration cell reports its frontier, local endpoint alpha, projection
method, and resolved independent-block count.

Under **Frontier Šidák**, siblings at one complete hierarchy level use disjoint atomic outcomes and
receive exact product calibration. Different levels and ancestor/descendant rows remain dependent,
so their frontier budgets are Bonferroni-adjusted. A root and a one-node level may show the same
frontier because they are identical weighted statistics, not two independent pieces of evidence.

Selecting a coarser profile, such as `block_by: [prompt_id]`, treats the entire rollout batch for
each prompt as one bounded unit. Its interval can be much wider, particularly for nodes with few
prompts. A large widening relative to the rollout profile indicates that the result depends
materially on treating repeated generations as independent evidence.

### Weighted mean

The expected failure probability under the plan's policy weights. Low-risk items can compensate for high-risk items, so it is normally a reference summary rather than the sole safety gate.

### Maximum

The highest prompt-level failure probability. This has the clearest non-compensation semantics but is often statistically expensive.

### CVaR

The weighted mean of the worst design tail. `beta: 0.9` averages the worst 10% of declared design weight, including fractional mass at the boundary.

### Entropic risk

A smooth risk-sensitive measure between the weighted mean and maximum. Larger `lambda` emphasizes high-risk prompts more strongly. Lambda is part of the estimand and must be selected before results are inspected.

### Worst of k

The weighted probability that at least one of `k` independent attempts at each fixed prompt
fails. At `k: 1` it equals the weighted mean; it grows toward the design's vulnerable mass as
`k` increases. Use it when deployment allows retries or an attacker gets `k` tries at the same
item. The `k` attempts are assumed independent given the prompt, which is the same rollout
independence the exact prompt bounds already assume.

### Threshold exceedance

The amount of design weight above local safety thresholds. The report divides weight into certified-safe, certified-unsafe, and unresolved portions.

### Threshold excess

The weighted amount by which prompt probabilities exceed their thresholds. It records both which requirements are violated and the magnitude of violation.

## Model and configuration comparisons

Risk differences use candidate minus baseline orientation. Negative values mean lower candidate
failure risk; positive values mean a regression. The primary direct interval aligns the same fixed
prompts and weights in both arms, supports unequal rollout counts, and targets the effect itself.
It does not treat prompts as a sample from a population.

Risk ratios use candidate divided by baseline. Values below one favor the candidate. Relative risk
reduction is one minus the ratio, so values above zero favor the candidate. Ratio estimates are
unstable when baseline risk is close to zero. If the baseline confidence set includes zero, a
finite upper ratio bound is mathematically unavailable; this is not a software failure.

Hierarchy rows repeat the selected estimand after restricting and renormalizing the original
weights within each declared node. The displayed simultaneous coverage includes every requested
overall and hierarchy-node comparison in the family. With hierarchy-region projection, all node
effects are constrained to arise from one prompt-contrast vector for differences or one common
two-arm prompt-risk region for ratios. The displayed projection can be narrower than its raw
interval but has the same simultaneous coverage. Reference methods have their own
construction-by-construction guarantee.

The unified hierarchy view lets you select either a within-model risk tree or an across-model
effect tree. In an across-model view, read the baseline and candidate risk bands first, then the
effect band. This avoids interpreting a favorable relative effect as low absolute risk. The
vertical reference is zero for differences and relative reduction, and one for risk ratios.

A jointly coupled prompt profile allows arbitrary dependence within each prompt across both
candidate and baseline rollouts. McNemar, the Newcombe paired score interval, and
cluster-bootstrap outputs remain diagnostics tied to the separately declared rollout-pairing key.

## Multiplicity

Prompt-specific bounds use Bonferroni allocation. Hierarchy-aware mean and comparison intervals
may instead use frontier calibration: Bonferroni across dependent levels and exact Šidák product
calibration among disjoint siblings. An all-mandatory global gate may use intersection-union logic:
every component must pass, but no additional Bonferroni correction is required solely for that
conjunction. Component intervals are not automatically simultaneous unless a simultaneous interval
family was also declared.

## Small evaluations and stderr-based intervals

Many evaluation tools report a mean with a standard error and an interval built from the
central limit theorem. On benchmarks below a few hundred items these intervals are usually much
too narrow, so they cover the true value far less often than their nominal level
([Bowyer, Aitchison and Ivanova, 2025](https://arxiv.org/abs/2503.01747)). Safety
evaluations are often that small and their failure rates close to zero, where the approximation
is worst. This is why the primary intervals here are exact binomial or finite-sample
concentration bounds, which hold at any sample size (see
[Mathematical methods](statistical-methods.md), sections 4.4 and 2.4.10).

Do not compare these intervals with stderr-based intervals from other tools as if both had the
same coverage. On a small evaluation the stderr interval will usually be narrower; that reflects
under-coverage, not greater precision.

## Diagnostic results

Diagnostic outputs help characterize sensitivity but do not have the same simultaneous coverage
guarantee as the primary fixed-design bounds. These include:

- fixed-cluster bootstrap intervals;
- exploratory multiple-testing results;
- plug-in estimates without complete simultaneous bounds.

Cluster-bootstrap diagnostics report the number and size range of creator-declared clusters, the
largest-to-smallest size ratio, and the largest leave-one-cluster-out estimate shift. These are
sensitivity indicators for imbalance and influence; resampling fixed clusters does not turn them
into a population sample.

## Assumptions and defaults a report states

Every report lists what its results rest on, built from the analysis rather than from a fixed
text:

- **The interval methods.** Under "Fixed-design risk measures", `report-full.md` names each method
  and independence profile its risk intervals used, what the profile blocks by, and the
  assumptions recorded on those intervals (`intervals[].assumptions` in `report.json`).
- **The assumptions.** "Limitations" lists each assumption from the package's registry that
  applies to the analysis, with the plan fields that set it, the value each had, whether the
  plan wrote it or left it to its default, and what follows if the assumption is wrong
  (`assumptions_used` in `report.json`). An assumption the plan does not make is left out: a
  sampled design is not a fixed one, a plan without comparisons makes no assumption about
  their arms, and an exploratory analysis does not claim its plan was fixed in advance.
- **The defaults.** "Defaults applied" lists the plan fields the plan left out that the
  analysis filled with a default value (`plan_defaults` in `report.json`; see
  [The plan hash](plan-reference.md#the-plan-hash)).

## External validity

No result in version 1 generalizes automatically beyond the fixed design. Statistical precision cannot repair scorer error, under-elicitation, sandbagging, omitted hazards, or deployment shift.

Every bound assumes the evaluated model is not strategically underperforming. Rare harms below
the rate the rollout budget can reach remain unmeasured residual risk, as validation of automated
driving found for crashes. See [Mathematical methods](statistical-methods.md), sections 15.2 and
15.7.
