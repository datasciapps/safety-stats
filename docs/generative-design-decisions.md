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
| paired comparisons | neither: same coefficients, one bound | **kept** |

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

## Limits on all of this

Every measurement above uses one synthetic hierarchy in the sparse-failure regime, where
Bernoulli noise is 97.8% of the variance of a single draw. An evaluation with common failures,
or a qualitatively different risk structure, would shift that split. Before building any of the
closed slices, re-run the relevant script on a tree matching the intended evaluation; if the
sign holds, the conclusions hold.

The model-assisted measurement additionally assumed a **binary** predictor, which is what lets
the residual use the exact two-cell construction. A calibrated probabilistic predictor gives
continuous residuals and loses it, so it should be expected to do worse rather than better.
