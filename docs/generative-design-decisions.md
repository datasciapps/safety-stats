# Generative evaluation: design decisions

The [original proposal](archive/generative-hierarchy-evaluation-plan.md) described a
seven-slice roadmap ending in cross-entropy adaptive importance sampling. What was built is
smaller, and the difference was decided by measurement rather than judgement. This document
records what was measured, what it closed, and the decisions taken where the proposal left a
choice open.

Every measurement here is reproducible from `research/`.

## Variance reduction does not improve the reported bound

Four of the seven slices -- hierarchical stratification, cost-aware allocation, model-assisted
estimation and cross-entropy importance sampling -- are variance-reduction techniques. Each was
compared against plain direct sampling at equal budget, scored not on estimator RMSE but on the
width of the finite-sample bound this package actually publishes.

### Importance sampling and Neyman allocation

`python research/generative_sampling_tradeoff.py`

| design | RMSE | bound width |
|---|---:|---:|
| direct target sampling | 1.00x | 1.00x |
| stratified, proportional | 0.98x | 1.00x |
| stratified, Neyman | 0.63x | **1.60x** |
| oracle importance sampling | 0.54x | **1.40x** |

The two scores move in opposite directions, and the gap widens exactly where importance
sampling is supposed to excel. At a 10,000-fold concentration of risk the estimator's variance
falls almost fourfold while the reported interval gets half again as wide. Importance sampling
was handed the second-moment-optimal proposal for free, so this bounds anything a learner could
reach.

**Why.** Every bound here is a bounded-weight concentration inequality, and those charge for
coefficient inequality. Direct sampling gives every draw the coefficient \(1/T\), which is
what makes the sharp Bernoulli-KL kernel available. Neyman allocation and inverse-probability
weighting deliberately create unequal coefficients -- that is *how* they reduce variance -- and
the bound bills for it.

Proportional stratification is free rather than harmful, so stratifying to guarantee coverage
of declared strata costs nothing even though it buys nothing.

### Model-assisted estimation

`python research/generative_model_assisted.py`

Model-assisted estimation -- GREG in survey sampling, two-level MLMC in Giles's formulation --
does *not* distort the coefficients, so the argument above does not condemn it. It was left
open for that reason, measured, and closed anyway.

A **prompt-level** predictor is the true two-level construction. At equal budget, where \(k\) is
how many predictor calls fit in one target evaluation:

| predictor | k=1 | k=3 | k=10 | k=100 |
|---|---:|---:|---:|---:|
| perfect | 0.68x | 0.84x | 0.95x | 1.07x |
| good | 0.58x | 0.70x | 0.79x | 0.87x |

A *perfect* predictor does not break even until the cheap level is a hundred times cheaper, and
that ratio is unreachable: evaluating the predictor on a prompt requires generating the prompt,
so both levels pay the rewrite cost.

A **path-level** predictor makes level 0 exact -- \(\mathbb E[f]=\sum_h w(h)f(h)\) is a
weighted sum over the compiled taxonomy, with no sampling and no alpha spent. Given the oracle
\(f(h)=p_h\), it still loses at **0.76x**, removing only 2.3% of the variance: with failures
rare, \(p(1-p)\) dominates and no function of the path reaches the rest.

**Why, and it is a different reason.** MLMC's own accounting is
\(\operatorname{Var}=V_0/n_0+V_1/n_1\), where a low-variance term contributes almost nothing.
Bound accounting is not additive that way: the widths add **and each level needs its own share
of alpha**. Even a perfect predictor leaves a floor of about \(\ln(4/\alpha)/n_1\) on the
residual, because the residual must be *proved* zero rather than merely be zero.

## Two failure modes, and the one technique that avoids both

| technique | effect on the bound's inputs | verdict |
|---|---|---|
| importance sampling, Neyman allocation | distorts the coefficient geometry | closed |
| model-assisted, MLMC | fragments the alpha budget across levels | closed |
| sequential Monte Carlo | spends the independence the bound is built on | closed |
| paired comparisons | none of the three: same coefficients, one bound | **kept** |

`python research/generative_paired_comparison.py`

Pairing on shared rewrites gives 2.0x--9.8x tighter root difference bounds when the arms
respond alike to prompt difficulty, 1.24x--1.44x when half coupled, and 1.02x when they are
independent given the prompt. It never widens the bound.

It survives because it is not really variance reduction. It changes *what is bounded*: from two
moderately common rates, whose bounds both enter the difference, to two rare discordance rates.
A concordant unit contributes exactly zero.

## Budget allocation has a constant answer

`python research/generative_budget_allocation.py`

The proposal's section 11 specified pilot-driven adaptive allocation to choose between more
paths, more rewrites and more rollouts. Additional paths are not a separate lever -- the design
draws paths in proportion to declared weight -- which leaves rewrites against rollouts, at a
fixed number of target evaluations:

| rewrites | rollouts each | kernel | root bound |
|---:|---:|---|---:|
| 1,200 | 1 | Clopper--Pearson | 1.00x |
| 600 | 2 | weighted KL/Chernoff | 1.28x |
| 150 | 8 | weighted KL/Chernoff | 1.86x |

Monotone, for two compounding reasons: a unit enters the bound as one bounded observation
however many rollouts produced it, and only single-rollout units keep outcomes binary. The
choice does not vary, so an adaptive allocator would be optimising a constant.

## Decisions the proposal left open

**A separate top-level plan schema.** The fixed analysis plan conditions on a materialized
design; a generative plan declares the process that produces prompts. One file carrying both
would let an author combine them into an estimand neither engine can state.

**Selection is always reproducible; the kernel is only frozen and fingerprinted.** Selection is
ours, so it replays exactly from `evaluation.selection_seed`. An LLM rewrite kernel cannot be
forced to replay, so the floor is a hash over its complete resolved configuration -- template
contents, decoding, validation and deduplication rules -- with `rewrite.seed_replayable` as an
optional declared property.

That fingerprint covers the plan's *declared* configuration, not the object actually supplied,
so nothing can verify an arbitrary implementation honours it. Materialization compares the
running kernel's identifier against `rewrite.kernel_id` and warns on a mismatch, which catches
a fake kernel run against a real plan without pretending to a check it cannot perform.

**Root and node intervals are not reconciled.** The root is computed from all \(T\) draws
directly, and no LP projection forces agreement with the aggregated node intervals. The root is
the primary estimand and deserves the tightest available bound; projecting would widen it to
buy a coherence the report does not claim.

**`simultaneous_tree` is off by default.** Turning it on puts the root in its own frontier and
splits alpha across depths by Bonferroni, which widens it. A reader who wants the tightest root
should leave it off and read the root alone.

**Log ingestion is deliberately thin.** It does not reuse `ingest_inspect_logs`, which resolves
groups, independence profiles and a prompt manifest from an `AnalysisPlan`. A generative run has
none of those: the sample identifier written at materialization is the whole join key, and the
cohort follows from the model. The score mapper is shared; nothing else is.

**Ledgers are append-only.** Materialization writes each unit as it completes, and resume never
rewrites what is there. An earlier draft deleted the ledger and re-emitted every unit, which
would have destroyed the first run's work had the resumed run also died. Rows therefore land in
completion order rather than selection order, which nothing downstream reads.

## Where the implementation departs from the proposal's structure

- Eight modules rather than fifteen. `balance`, `variance`, `allocation`, `model_assisted`,
  `cross_entropy` and the proposal's separate `comparisons` design are not created.
- `node_type` has no `rewrite_kernel` value. One kernel serves every path, so terminal nodes
  carry `rewrite_kernel_id` and "every positive-weight node has a valid route to a kernel"
  reduces to "every non-terminal node retained a child".
- Section 7.2's "transition table" for context-specific weights is an inline `context_weights`
  list on the arm.
- The selection ledger drops `round` and renames `allocation_stratum` to `stratum`.
  `proposal_path_probability` and `proposal_id` are kept even though direct sampling always
  sets the proposal equal to the target, so an auditor can verify that rather than assume it.
- `|` and `=` are reserved in every identifier, because canonical node ids are built from them.
- Cycles and orphans are not runtime checks: the hierarchy is a declared ordered list and every
  node id contains its parent's id as a prefix, so neither can be expressed.

## Reconciliation against the proposal's completion criteria

Section 25 of the proposal listed ten. All are disposed of:

| # | criterion | status |
|---|---|---|
| 1 | define a deep hierarchy without enumerating prompts | met |
| 2 | one auditable target distribution over valid paths | met |
| 3 | materialize, resume, pair, and join to a ledger | met |
| 4 | estimators recover known node risks with validated intervals | met |
| 5 | data-driven choice between paths, rewrites and rollouts | answered, not built |
| 6 | incomplete schedules used only with known probabilities | vacuous |
| 7 | model-assisted estimation | closed by measurement |
| 8 | cross-entropy importance sampling | closed by measurement |
| 9 | artifacts distinguish target risk from raw sampled rate | met |
| 10 | visualization shows risk, uncertainty, allocation, ESS, comparisons | met |

Criterion 9's "discovery metrics" have no referent: there is no discovery mode.

## Budget planning follows from the same measurements

Every measurement above ended in some form of "you could not have afforded that", and none
of it was available to a user in advance. `generative budget` closes that: from the compiled
taxonomy and one assumed rate it projects the root bound, which hierarchy levels the budget
can resolve, how many nodes will go unsampled, the retry cost of an imperfect validator, and
whether a threshold claim would be reliable -- before a single rewrite is generated.

It is a projection, not a prediction. It assumes risk is uniform across the design at the
declared rate, so real heterogeneity moves individual nodes either way. What it reports
exactly is the *geometry*: a level whose median node would receive four draws cannot support
an interval regardless of what the risk turns out to be.

## How the log-reading path is verified

Most generative tests stub `evals_df` and `samples_df` with synthetic frames, following the
fixed-design suite. That exercises the join but never the log format, and cohort matching by
model identifier and epoch handling can only be checked against a real log.

`tests/integration/test_generative_real_logs.py` therefore materializes a dataset, runs a
genuine Inspect evaluation over it with the local mock model, and analyses the `.eval` file
that produces -- no network, no provider, no cost, in about six seconds. It asserts a clean
run reports **no diagnostics at all**, that the structural metadata written at
materialization survives the round trip, that a score name the log does not carry degrades
conservatively rather than dropping samples, and that extra epochs are caught against the
declared `rollouts_per_rewrite`.

Two defects surfaced the first time it ran, neither reachable from synthetic frames. Analysis
joined *every* declared cohort rather than the ones in use, so a cohort nobody asked about
reported "120 of 120 sampling units produced no usable outcome" -- which reads like a failed
run. And a cohort with no matching logs at all was reported with the same code as one that
was partly evaluated, when the two are different situations with different fixes.

## Limits on all of this

Every measurement above uses one synthetic hierarchy in the sparse-failure regime, where
Bernoulli noise is 97.8% of the variance of a single draw. An evaluation with common failures,
or a qualitatively different risk structure, would shift that split. Before building any of the
closed slices, re-run the relevant script on a tree matching the intended evaluation; if the
sign holds, the conclusions hold.

The model-assisted measurement additionally assumed a **binary** predictor, which is what lets
the residual use the exact two-cell construction. A calibrated probabilistic predictor gives
continuous residuals and loses it, so it should be expected to do worse rather than better.

## Sequential Monte Carlo

**Status:** measured, 25 August 2026. Reproduce with `python research/generative_smc.py`.

SMC splits into two halves that fail differently.

**Without resampling** it is sequential importance sampling with a factorized proposal over
the tree -- exactly the cross-entropy slice, already closed by slice 0. The likelihood
ratios become unequal coefficients and the bound charges for that.

**With resampling** -- the half that makes it SMC -- particles interact. Whether particle
\(i\) survives depends on particle \(j\)'s weight, so the estimator stops being a sum of
independent bounded terms, which is the only object the engines here can bound. It is not a
worse estimator; it is outside the class.

There is one honest way to bound it: run \(K\) independent replicates and treat their
estimates as \(K\) independent observations in \([0,1]\). That is valid, and it is the trade
this measures. On 200 paths with one arm carrying a hundred-thousand-fold risk ratio, at a
budget of 1,200:

| design | RMSE | bound |
|---|---:|---:|
| direct sampling | 1.00x | 1.00x |
| SMC, 400 replicates x 3 | 0.11x | 3.78x |
| SMC, 100 replicates x 12 | 0.59x | 4.96x |
| SMC, 25 replicates x 48 | 1.03x | 11.83x |
| SMC, 8 replicates x 150 | 1.06x | **23.12x** |

RMSE only reaches parity at the fewest, largest replicates, where the bound is already an
order of magnitude wider than direct sampling's. At eight replicates the bound is 0.68 for a
root risk of 0.021 -- arithmetically valid and completely useless.

**The mechanism is a third distinct one.** Importance sampling distorts the coefficient
geometry; model-assisted estimation fragments the alpha budget; SMC *spends the
independence* the bound is built on, and buying it back through replication makes the bound
rest on the replicate count rather than the budget.

That penalty does not depend on how good the estimator is. An SMC with half the RMSE would
still be bounded by \(K\) observations. Better focusing does not buy back independence.

### Where it does belong

Not in the estimator -- in the design. Section 5 of the original plan forbids turning
risk-directed discovery into root-probability observations *without a valid sampling
probability*, which rules out feeding SMC into the estimate. It does not rule out discovery.

A two-stage separation works cleanly: run SMC, adaptive splitting or evolutionary mutation
to find where a model actually breaks; use what it finds to write a **new declared
taxonomy**; then sample that design directly with equal coefficients and exact bounds. The
discovery informs what is declared, and the declaration is sampled honestly. That is more
durable than a variance reduction, because a better taxonomy improves every subsequent run.

### Limits on this one

The resampling step measured here acts on a single-level draw, so it tests the coupling
faithfully but barely tests sequential adaptation; a real SMC would descend the hierarchy
with a proposal that adapts as it goes. The conclusion does not rest on that, for the reason
above -- the bound's dependence on \(K\) is independent of estimator quality -- but a claim
that *adaptation cannot help the estimate* would need the multi-level version.
