# What results establish

This page says what a result of this package shows, what it does not, and where in the report
to check each condition a result rests on. The details are in
[Interpreting results](interpretation.md) and [Mathematical methods](statistical-methods.md).
Entries such as "L2" refer to the [limitations log](limitations.md), which records where each
point was found on a real evaluation.

## What a result is about

A result is about the evaluation that was run, not about a population of prompts or a
deployment.

- **A fixed design** (the default) covers the declared prompts and the stochastic rollout process
  that produced the logs: the model, its configuration and the scaffold, as recorded. The only
  randomness is in the rollouts. The prompts are not treated as a sample, so a result says
  nothing about paraphrases, unseen prompts or deployment traffic (L11).
- **A [sampled design](design-based-sampling.md)** covers the declared item frame, from which the
  evaluated items were drawn with recorded inclusion probabilities. It says nothing about items
  outside the frame.
- **The estimand** is end-to-end system safety by default: a target-side provider block with no
  delivered content counts as safe. With `inputs.estimand: underlying_model` it stays unknown.

The report states its scope under "Scope", with the estimand.

## What an interval guarantees

An interval covers the true risk of that design with the stated coverage, at any sample size, as
long as the assumptions it rests on hold. The report lists them:

- "Fixed-design risk measures" names each interval method and independence profile used.
- "Limitations" lists each assumption the analysis made, the plan field that sets it, its value,
  and what follows if it is wrong. "Defaults applied" lists the plan fields left to their
  defaults.
- The diagnostics name each check that found evidence against an assumption. The
  [diagnostic catalogue](diagnostics.md) gives every code's assumption, effect and remedy.
- "Data integrity" says whether the evidence met the design: `degraded` evidence keeps
  conservative bounds, `invalid` evidence certifies nothing.

Unclassified outcomes are completed adversely: the upper bound counts every unknown as a
failure. A wide interval therefore can come from missing labels as well as from few rollouts.

## Which independence the result rests on

Under the rollout profile each rollout of a prompt is an independent draw. For a fixed prompt
set this follows from how rollouts are generated, so the intervals can be narrow. Under a
coarser profile such as `prompt`, each prompt's batch of rollouts counts as one unit, and a few
hundred prompts decide few claims at thresholds of 5% to 10% (L2). Comparison intervals can then
be close to vacuous: about ±0.26 overall in one real study (L19). This width is a property of the
design, not a fault in the analysis.

When the plan declares several profiles, the report gives each claim's verdict under each of
them ([verdicts by profile](independence-profiles.md#verdicts-by-profile)). A `PASS` that is
`INCONCLUSIVE` under a coarser profile rests on the finer profile's independence assumption.

## What the judge contributes

Unless a judge calibration is declared, every risk is the risk of a failure **as the judge
reports it**. An interval covers that rate, not the true failure rate.

- A verdict close to its threshold can rest on the judge's error rate. In one study a claim
  passed at an upper bound of 4.98% against a 5% limit. Re-judged with a more sensitive
  configuration of the same judge, the bound rose to 6.46% and the claim would have been
  `INCONCLUSIVE` (L18). The experimental
  [judge-error stress](judge-misclassification.md#stressing-a-threshold-verdict-experimental)
  says how much judge error a verdict survives.
- A judge's error rates measured on a published validation set do not carry over to another
  model's outputs (L23). A certifying correction needs human labels on the run's own outputs
  ([validated mode](judge-misclassification.md#validated-mode-measured-accuracies)).
- Two models each judged by a different judge are not comparable: the difference can be mostly
  judge (L46). Compare models judged by one judge
  ([one judge for every arm](model-comparisons.md#one-judge-for-every-arm)).

## How to read verdicts and comparisons

- `INCONCLUSIVE` means neither `PASS` nor `FAIL` was established. It is not a near-pass, and an
  interval that cannot tell two arms apart is not evidence that they are equal.
- Each claim records the thresholds at which the same bounds would pass or fail
  (`verdict_range`). A threshold chosen after seeing the bounds does not show that the
  requirement came before the evidence.
- An exploratory analysis certifies nothing. Its descriptive status says what a confirmatory
  analysis of the same bounds would give.
- Group intervals and a comparison of the same groups answer different questions. When
  simultaneous group intervals do not overlap, they order the two groups at their joint
  coverage, even if the comparison's interval includes no difference. In one study the
  comparison's method fell back to a wider bound, so the two readings disagreed (L52). Neither
  contradicts the other. The comparison is the reading the plan declared, and the report does
  not yet say which reading is stronger.
- A headline average can hide a large effect in one category. "Where effects sit" in
  `report.md` points to the hierarchy rows that matter.

## What no result establishes

Statistical precision cannot repair problems outside the rollout process. No result
establishes:

- **Behaviour beyond the evaluated set**: other prompts, languages, tools, users or deployment
  conditions, or hazards the evaluation left out.
- **Absence of sandbagging.** Every bound assumes the model is not strategically
  underperforming, for example because it recognises the evaluation. The package does not
  check this ([section 15.2](statistical-methods.md#152-evaluation-awareness-and-sandbagging)).
- **Absence of capability.** A scaffold, prompt or budget that fails to elicit a behaviour
  gives a low rate that is not evidence of safety.
- **Safety from rare harms.** An upper bound of θ at 95% needs about 3/θ failure-free
  rollouts. Harms rarer than the rollout budget can reach are not shown to be rare, only
  unmeasured. Claims about them need other evidence, such as monitoring in deployment
  ([section 15.7](statistical-methods.md#157-residual-risk-of-rare-harms)).
- **Uncontaminated items.** A model that has seen the benchmark may behave differently on it.

Intervals from other tools built from a mean and standard error are usually much narrower on
small evaluations because they under-cover, not because they are more precise. Do not compare
them with these intervals as if both had the same coverage
([small evaluations](interpretation.md#small-evaluations-and-stderr-based-intervals)).
