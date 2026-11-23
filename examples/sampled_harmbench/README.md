# Sampled-frame example: HarmBench from a probability sample

This example estimates the **same** domain-balanced mean harmful-compliance risk that the
[HarmBench census](../real_harmbench/README.md) already measured, but from a probability
sample of the 300-item frame instead of every item. Because the census answer is known, the
sampled interval can be checked against it.

That check is the point. A simulation can show a design-based bound covers a mean drawn from
an assumed distribution; only a real census can show it covers the mean this model and this
judge actually produced, with the real heterogeneity across behaviours.

## Precisely what this measures

- The frame is the **300 standard and contextual harmful-behaviour items**
  (`assessment_family: harmful_behavior`), the same set the census `all-behaviors` group
  covers. The 100 copyright items are a separate assessment task and are excluded.
- The estimand is the frame-wide **domain-balanced weighted mean**, using the same
  `nested_equal` weighting over `task_id → risk_domain → question_id → prompt_id` as the
  census. It is not the mean over the drawn items.
- Two sampling designs are declared, differing only in how the frame is drawn:
  `plan-poisson.yaml` includes each item independently with probability 0.25;
  `plan-stratified.yaml` draws a fixed count from each of the six risk domains without
  replacement.

## The example is retrospective, and says so

Both plans declare `intent: exploratory`. The sample is drawn from an evaluation that has
already run, so this cannot be a confirmatory safety claim, and the package refuses to let it
pretend otherwise — a confirmatory plan frozen after its evaluation is rejected with
`PLAN_POSTHOC`.

The draw uses only the frame and a declared seed, never an outcome, which is what makes the
comparison meaningful: the sample is chosen exactly as it would have been beforehand, and
then the already-recorded outcomes for those items are kept.

## Running it

Draw a sample and subset a completed census log to it:

```bash
python examples/sampled_harmbench/draw_sample.py \
  logs/2026-08-19T08-51-23-00-00_harmbench-direct-request_cyiCrFsAMFMTYAHjgUXuUv.eval \
  run/logs-stratified/sample.eval \
  --stratified
```

Then analyse the subset against the frame:

```bash
safety-stats analyze \
  --plan examples/sampled_harmbench/plan-stratified.yaml \
  --logs run/logs-stratified \
  --output run/report-stratified
```

Check the interval against the census, and see what a smaller budget buys:

```bash
python examples/sampled_harmbench/validate_coverage.py \
  examples/real_harmbench/live-runs/RUN_ID/report-e2e-empirical-hierarchy
```

Before committing budget, ask where the uncertainty would come from:

```bash
python examples/sampled_harmbench/variance_budget.py \
  examples/real_harmbench/live-runs/RUN_ID/report-e2e-empirical-hierarchy
```

It splits the estimator's variance into the part caused by *which items were drawn* and
the part caused by *rollout noise inside them*, then shows what interval width a given
number of items actually buys. Run it against a pilot to plan a real study.

## What it shows

Against a census answer of **0.13437**:

| design | items | rollouts | interval | width | covers census |
|---|---:|---:|---|---:|---|
| census | 300 | 6,400 | [0.10578, 0.16296] | 0.057 | — |
| Poisson, π=0.25 | 82 | 1,312 | [0.05624, 0.49386] | 0.438 | yes |
| stratified fixed-size | 75 | 1,200 | [0.03158, 0.38613] | 0.354 | yes |

Repeated redraws against the census give 1.000 empirical coverage for both designs, which is
what a finite-sample concentration bound should do — it is conservative by construction.

**Stratifying on the variable the weights balance removes the inverse-probability penalty
entirely.** Poisson sampling reports `inflation = 4.00`: every endpoint is widened fourfold
before any data is considered. The stratified design reports `inflation = 1.00`, because
`nested_equal` makes weights equal *within* each risk domain and the strata *are* the risk
domains, so the estimator reduces to a plain sample mean. It gives a tighter interval from
fewer items.

## Two costs worth understanding before using this

**Extra rollouts per item buy nothing.** The design-based bound reads each sampled item's
failure *rate* as one bounded observation in `[0, 1]`; it never reads how many rollouts
produced that rate. Sixteen rollouts on 75 items gives the same interval as one rollout on 75
items. Under a sampled design, budget belongs in **more items, not more rollouts** — the
opposite of the advice for a census.

That is a property of the design, not a gap in the estimator. On this frame at sixteen
rollouts and π=0.25, within-item rollout noise is only **2% of the estimator's variance**;
the other 98% is item-sampling variance driven by how much the per-item risks differ. A
two-stage decomposition that removed the rollout term entirely would narrow the interval by
about 1%. Run `variance_budget.py` to see the split for your own design.

The sampled intervals are wider than the census because sampling a quarter of a heterogeneous
frame genuinely carries about **fourteen times** the standard error of a census — 0.0298
against 0.0021. That cost is irreducible by any estimator; it is reduced by drawing more
items, or by stratifying on something that predicts risk.

**Do not request hierarchy nodes you cannot identify.** These plans set
`hierarchy_nodes: false`. With 300 question-level nodes and only 75 items drawn, most nodes
have no descendant in the sample: they report no interval, emit `HIERARCHY_NODE_UNSAMPLED`,
and — if requested — consume family alpha that the root then does without. Requesting them
here widened the root interval from 0.35 to 0.61.

## Interpretation boundary

At 300 items the frame is already small, and the table shows there is little slack to give
away. This example demonstrates that the machinery recovers a known answer; it is not an
argument that subsampling HarmBench is a good idea. The feature earns its keep when a frame
is large enough that a census is genuinely out of reach, and the cost curve from
`validate_coverage.py` is what tells you whether the trade is worth taking at your budget.

Everything the [HarmBench example](../real_harmbench/README.md) says about scope still
applies: these are DirectRequest results under one target configuration and one judge, and
they establish nothing about jailbreak robustness, omitted behaviours, or deployment.
