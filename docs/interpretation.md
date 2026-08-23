# Interpreting results

## Start with the risk estimates

The default report presents estimates and uncertainty intervals without turning them into a
package-level safety verdict. Interpret each result against thresholds that were chosen before
examining the evaluation. The most important choices are the cohort, fixed design group, policy
weights, risk measure, and confidence level.

Threshold decisions are an optional presentation layer enabled with
`report.include_decisions: true`. When enabled:

`PASS` means the predeclared upper bound is at or below the acceptable threshold and all required integrity gates passed.

`FAIL` means a predeclared lower bound establishes unacceptable risk, a mandatory claim failed, or a verified zero-tolerance event occurred.

`INCONCLUSIVE` means neither conclusion was established or the evidence did not satisfy the design contract. It is not a near-pass.

## Integrity states

`valid` means every required outcome was classified and no structural integrity error occurred.

`degraded` means one or more attempted outcomes are unclassified, but every missing binary label is
covered by conservative bounds. Degraded evidence may certify a claim only when the claim passes
under the adverse completion of all unknown outcomes.

`invalid` means a structural or strict-policy error prevents certification. Examples include an
incomplete design, an unmapped score, configuration drift, a post hoc confirmatory plan, or a
missing/error policy explicitly set to `block`.

Under the default end-to-end estimand, a target-side provider policy block with no delivered
content is a classified safe system outcome. A grader-side block is not: the target output already
exists, so its safety label remains unknown.

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

## Diagnostic results

Diagnostic outputs help characterize sensitivity but do not have the same simultaneous coverage
guarantee as the primary fixed-design bounds. These include:

- fixed-cluster bootstrap intervals;
- power means;
- design quantiles;
- spectral risk measures;
- exploratory multiple-testing results;
- plug-in estimates without complete simultaneous bounds.

Cluster-bootstrap diagnostics report the number and size range of creator-declared clusters, the
largest-to-smallest size ratio, and the largest leave-one-cluster-out estimate shift. These are
sensitivity indicators for imbalance and influence; resampling fixed clusters does not turn them
into a population sample.

## External validity

No result in version 1 generalizes automatically beyond the fixed design. Statistical precision cannot repair scorer error, under-elicitation, sandbagging, omitted hazards, or deployment shift.
