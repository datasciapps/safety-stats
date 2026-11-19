# Anytime-valid confidence sequences: phase-0 design

**Status:** phase 0 of #10, a design; phase 1 is not built. One piece of it is: the one-sided
bound for runs that optstop stopped (#126, experimental), in
[Runs that optstop stopped](#runs-that-optstop-stopped-126). This document fixes what phase 1
builds and what it must refuse. The mathematics is in
[Statistical methods](statistical-methods.md) sections 6, 11, 12 and 16.7, and is not repeated
here.

The fixed-horizon analysis stays the default. For a rollout count declared in advance and one
final analysis, the exact bounds of sections 4 and 5 are tighter and simpler to audit. A
confidence sequence is worth its cost only when observed outcomes decide when sampling stops.

## Phases

| phase | scope | issue |
|---|---|---|
| 0 | this document | #10 |
| 1 | the Hoeffding-spending reference sequence and a betting or empirical-Bernstein sequence for draws with replacement, added to `stats/sequential.py` beside the bound of #126; `analyze --sequential` over a growing log directory; no Inspect integration | #10 |
| 2 | optional integration with Inspect's early stopping, after shadow mode (below) | separate issue when reached |

Phase 1 is accepted when a simulation shows time-uniform coverage under continuous monitoring
(section 16.7), the pass and fail stopping rules of section 6.5 control their error, and a
peeking test shows the fixed-sample bound failing where the sequence holds.

## Estimand

The estimand is the one the fixed-horizon analysis already reports: the design mean

\[
\theta=\sum_i w_i\,p_i ,
\]

where \(p_i\) is the failure probability of item \(i\) and \(w_i\) its declared weight. A
confidence sequence changes when the analysis may look at the data, not what it estimates.

## The stream

A confidence sequence needs an ordered stream \(X_1, X_2, \ldots\) with a known conditional mean
given the past. The items of a fixed design are heterogeneous, so a stream that walks through
them in a fixed order has a conditional mean that changes from step to step. This is why
`weighted_betting` (#3) is fixed-horizon only: it runs over the blocks in a fixed order.

### Draws with known probabilities

Draw the item for step \(t\) at random, independently of the past, with
\(\Pr(I_t=i)=w_i\), and let \(X_t\) be the outcome of that draw. Then

\[
\mathbb E[X_t\mid\mathcal F_{t-1}]=\sum_i w_i\,p_i=\theta
\]

for every \(t\), although the items differ. The identity also rests on three assumptions about
the evaluation, which the draws cannot supply:

- **Fixed failure probabilities.** Each \(p_i\) stays the same while the stream runs. The
  model, its configuration, the scaffold and the judge do not change between draws, and
  nothing in the system learns from earlier rollouts.
- **Fresh rollouts.** Each draw runs its rollouts anew, sharing no state, cache or conversation
  with earlier draws. An item drawn again is run again, not replayed.
- **A hidden draw sequence.** The selection seed is fixed before the run and kept from the
  system under evaluation, so that its behaviour on a draw cannot depend on which items come
  next.

Hsu and Shekhar prove the general form: for any predictable querying rule \(q_t\) with
\(q_t(i)>0\) for every item, their inverse-probability betting factor is a conditional e-value
(*Efficient Sequential Evaluation of Large Language Models*, arXiv:2607.17409, Proposition
4.2). Drawing in proportion to the weights, \(q_t=w\), makes the inverse-probability correction
cancel. Each observation is then the plain outcome in \([0,1]\), so the weighting costs nothing
in the range of admissible bets.

Two limits follow from that proposition:

- **With replacement only.** It covers independent draws from \(q_t\), so an item can be drawn
  again. A seeded permutation, drawing without replacement, is not covered: the conditional
  mean shifts as items are used up. The bound for #126 below covers a permutation by a
  different route, for an upper bound.
- **Every weight positive.** An item of weight zero is outside the estimand and is never drawn.

### Several rollouts per draw

When each draw runs \(r\) rollouts of its item, those rollouts are correlated, and updating after
each one breaks the conditional-mean condition (section 6.7). The unit of the stream is the
draw: the sequence updates once all \(r\) rollouts of a draw are complete, with
\(X_t\in[0,1]\) the draw's failure fraction. Its conditional mean is still \(\theta\).

### Where phase 1 starts: generative `direct` sampling

The generative path already draws this way. Under `sampling.method: direct`, draw \(t\) picks a
terminal node in proportion to its target weight, from a hash of the selection seed and \(t\)
alone (`draw_selections` in `src/inspect_safety_stats/generative/sampling.py`). The draw does not
depend on `sampling.total_rewrites`, so the first \(t\) draws of a long run are the same as those
of a short one, and the stream is well defined before its length is known.

Phase 1 refuses `sampling.method: proportional_stratified`. Its allocation across strata comes
from a largest-remainder split of `total_rewrites`, so the draws depend on the total.

The manifest-based Inspect path does not give this stream, with or without a recorded seed.
Inspect runs samples in dataset order unless `sample_shuffle` is set, and it is off by default.
With `sample_shuffle` set to a seed, the order is a seeded permutation: each sample appears once in the order, so
the draws are without replacement and outside the proposition above (with replacement only).
Phase 1's sequences refuse that path. Its only anytime-valid route is the upper bound of #126
([below](#runs-that-optstop-stopped-126)), which covers a permutation, and which in turn refuses
a run whose seed and shuffled order the log does not record (`ANYTIME_ORDER_UNRECORDED`).

### Draw order, not completion order

Phase 1 reads trials in draw order (`selection_order`), never in the order they finished.
Concurrent samples finish in an order correlated with their outcome: a refusal is short, so safe
outcomes tend to arrive first, and a sequence fed in completion order would look safer than the
system is. At each look the analysis uses one of:

- the longest prefix of draws that are all complete; or
- every draw issued so far, with unfinished draws counted as failures, for the upper bound only.

The second option rests on the argument of section 4.5 for unclassified rollouts: a bound that
never falls when one observation rises stays valid when unknown outcomes are set to their
largest value. The Hoeffding-spending reference sequence has that property, and so does a
betting mixture over fractions fixed in advance, such as the bound of #126: a larger outcome
lowers its own factor of the capital and raises none of the others. A sequence that fits its
bets or its variance estimate to earlier outcomes, as a betting sequence with predictable
plug-in bets or an empirical-Bernstein sequence does, need not have it. A larger early outcome
changes every later bet, and the bound can fall. Phase 1 offers the second option only for a
sequence that has the property, and uses the longest complete prefix for any other.

## Several claims

Section 11 applies unchanged. Each claim \(i\) gets its own \(\alpha_i\), with
\(\sum_i\alpha_i\) at most the family's \(\alpha\). Two mechanisms share the work:

- **Looks.** Each claim's sequence is time-uniform at its \(\alpha_i\): the chance that it ever
  excludes the true value, at any look, is at most \(\alpha_i\). Looks cost no alpha, however
  many there are and whenever they are taken.
- **Claims.** The union bound over claims then gives at most \(\sum_i\alpha_i\) for any claim
  ever being wrong.

The allocation is frozen before monitoring starts. A claim added during monitoring needs alpha
reserved for it in advance, or a new family.

## State between looks

The logs are the source of truth. Each look recomputes the sequence from the draw-ordered prefix,
so a resumed analysis and an uninterrupted one agree by construction, which is the replay and
resume test of section 16.7. The state kept between looks holds only what the logs cannot
supply:

- the plan hash, the selection seed and the maximum number of draws;
- the frozen alpha allocation per claim;
- each decision taken, with the look and prefix length it was taken at.

The record of decisions is what makes a stop auditable. A decision is never re-derived from a
later, longer prefix.

## Runs stopped early in the fixed-horizon path

A run that stopped because of its outcomes must not be certified by the fixed-horizon bound.
Section 12 says such a run is reported as adaptively collected, and so `INCONCLUSIVE`. The
package detects this today only in part.

A two-prompt fixed design was analysed with ten rollouts planned per prompt and three observed,
all safe, the way a run that stopped early looks (`analyze`, with the Inspect log reader
replaced by fixed data frames):

| `expected_rollouts` column | `design.require_expected_rollouts` | Inspect status | result |
|---|---|---|---|
| present | `true` (default) | `success` | `ROLLOUT_SHORTFALL`, integrity invalid |
| absent | `true` (default) | `success` | `PLAN_EXPECTED_ROLLOUTS_MISSING`, integrity invalid |
| present or absent | `false` | `success` | integrity valid, no diagnostic; upper bound computed on 3 rollouts |
| any | any | `cancelled` | `LOG_INCOMPLETE`, integrity invalid |

So:

- **Without an `expected_rollouts` column, a fixed design stopped early raises no
  `ROLLOUT_SHORTFALL`.** Under the default the analysis is still refused, but by
  `PLAN_EXPECTED_ROLLOUTS_MISSING`. That error says the manifest lacks the column, not that the
  run stopped early, and a complete run gets the same error.
- **With `require_expected_rollouts: false`, a run stopped early is certified.** No diagnostic
  is raised and the bound is computed on the rollouts that exist. Nothing compares them with the
  epochs the Inspect log declares. Section 4.5 says a prospective rollout shortfall remains an
  integrity error, which holds only with the column present and the flag on.
- **Inspect's own early stopping looks like a complete run.** When an `EarlyStopping` manager
  skips a sample (Inspect 0.3.290), the sample ends as completed, not as an error. The skipped
  sample and epoch pairs are listed in the log's `results.early_stopping`, and the run's status
  can still be `success`. The package reads no early-stopping field.

The checks are in `src/inspect_safety_stats/ingest/normalize.py`, after the observations are
counted. Closing these gaps changes output bytes, so it is a class B change and needs its own
issue (CONTRIBUTING section 3). It is a prerequisite of phase 1: a run that a sequence may stop
must not also be certifiable by the fixed-horizon gate. What that change needs to do:

- treat a non-empty `results.early_stopping` as adaptive collection and mark the run
  `INCONCLUSIVE` under the fixed-horizon gate, whatever the plan's flags;
- without an `expected_rollouts` column, compare the observed rollouts per item with the epochs
  the log declares.

## Shadow mode before anything stops an evaluation

Phase 1 is offline: it reads logs and stops nothing. Before phase 2 lets the sequence stop an
evaluation, it runs in shadow mode. The evaluation runs to its fixed horizon, and the sequence
is computed alongside and reports where it would have stopped and with what decision. The
fixed-horizon result at the end is what counts. Comparing the two over real runs is the evidence
phase 2 needs.

## Cost

Anytime validity costs width, and with it draws. The figures below are for one-sided
\(\alpha=0.05\), a pass threshold of 0.05 and at most 1,200 draws with replacement. The
sequence is a betting mixture: the capital of step 1 of #126 without its finite-population
correction, mixed over the fixed bet fractions of the betting bounds (#3), with the running
intersection over draws. A run passes at the first draw whose upper bound is at or below the
threshold. A fixed design is one-sided Clopper–Pearson at a number of draws fixed in advance.
`python research/sequential_cost.py` (about 75 s, 20,000 runs per rate, 1,000 per width cell)
gives every figure except Clopper–Pearson re-checked after every draw, which
`python research/sequential_peeking.py` gives from 200,000 runs.

| quantity | sequence | fixed design |
|---|---|---|
| upper bound minus estimate after 100, 300 or 1,200 draws, at true rates 0.005 to 0.05: median ratio to Clopper–Pearson's | 1.13–1.89× | 1 |
| draws to pass at a true rate of 0.005 | median 67, mean 101; every run passes | 59 draws pass 0.744 of runs, 93 pass 0.90, 153 pass 0.99 |
| probability of passing at a true rate of 0.03 | 0.708 by 1,200 draws; a run uses 745 draws on average | 0.970 with 1,200 draws; 482 draws pass 0.708 |
| false-pass rate at a true rate of 0.05, checked after every draw | 0.040 (s.e. 0.0014) | Clopper–Pearson: 0.242 (s.e. 0.0010); a single look at 1,200 draws: 0.045 |

The sequence saves the most draws where the failure rate is well below the threshold and not
known in advance. At 0.005 it passes after a mean of 101 draws, where a fixed design needs 153 to pass
0.99 of the time; the zero-failure design of 59 draws is smaller but passes only 0.744 of the
time. Near the threshold the sequence costs draws: at 0.03 a fixed design of 482 draws passes as
often as the sequence, which uses 745 on average. The Hoeffding-spending reference passes none
of these runs. After 1,200 draws without a failure its bound is still 0.085, and its margin over
the estimate is 4.7 to 17 times Clopper–Pearson's. It is a reference for tests, not a sequence
to run.

Re-checking Clopper–Pearson after every draw multiplies the false-pass rate about five-fold,
while the sequence, checked as often, stays within the nominal 0.05. This is the peeking
demonstration phase 1 must turn into a test.

The review of #10, tracked in #128, reported figures of the same kind from a sequence it did not
name: 1.1–1.5× for the width, 72 draws at 0.005, 0.76 at 0.03 and 0.247 for Clopper–Pearson
re-checked. It set its 72 draws against the 1,200-draw horizon, not against a fixed design. Its
0.247 is about five of this table's Monte Carlo standard errors from the 0.242 reproduced here,
but the review did not report how many runs it used, so the gap may be its own Monte Carlo
error. The table above replaces those figures. Phase 1
measures them again for the sequence it builds.

## Related work

[Statistical methods](statistical-methods.md) section 6.8 compares the prior work: Hsu and
Shekhar (reference 20) and CELEUS (reference 39) build anytime-valid intervals for LLM
evaluation, and Optstop (reference 21) is the Bayesian counterpart, with no frequentist
false-pass guarantee (section 12). Neither of the first two offers one-sided threshold
decisions, a weighted estimand, several rollouts per prompt, judge correction or multiplicity
across claims. Those five are what this package already provides at the fixed horizon, and the
sequence must keep them.

## Runs that optstop stopped (#126)

Optstop stops an Inspect run once a Bayesian credible interval is narrow enough. Its paper
reports that this targets precision, not coverage (section 12). `outputs.anytime_bounds`
(experimental, [plan reference](plan-reference.md#anytime-valid-bounds-for-runs-stopped-early-experimental))
reports a one-sided upper bound \(U_t\) with

\[
\Pr\bigl(\exists t:\ \theta > U_t\bigr)\le\alpha ,
\]

so \(U_\tau\) is valid at the stopping time \(\tau\), whatever rule chose it. The code is
`src/inspect_safety_stats/stats/sequential.py`.

### The stream

Inspect's `sample_shuffle` runs the \(N\) items in a uniformly random order, a permutation, fixed
before the run. Item \(i\) has weight \(w_i\) and outcome \(Y_i\in[0,1]\), its failure share
over its first \(r\) epochs, where \(r\) is declared in advance. Epochs after the first \(r\)
are ignored, so a rule that stops an item's later epochs on its results cannot bias \(Y_i\).

### Two steps, each at \(\alpha/2\)

1. **The realized design mean \(S=\sum_i w_iY_i\).** Given the outcomes, the order is a
   uniform permutation, so the prefix is a sample without replacement from the finite
   population of the \(N\) values \(Z_i=w_iY_i/\max_j w_j\in[0,1]\). For a candidate population
   mean \(\mu\) of the \(Z_i\), the next draw's conditional mean is
   \(m_t(\mu)=(N\mu-\sum_{s<t}Z_s)/(N-t+1)\), which the past determines. For a bet fraction
   \(f\in[0,1)\), the capital \(\prod_t\bigl(1-f+f(1-Z_t)/(1-m_t(\mu))\bigr)\) is then a
   nonnegative martingale with mean one at the true \(\mu\), and so is the mixture over the
   fixed fractions of the betting bounds (#3). Ville's inequality bounds the chance that it
   ever reaches \(2/\alpha\). The candidates it has reached that level at, by any prefix, are
   rejected; the capital rises with \(\mu\), so they form an upper ray.
2. **From \(S\) to \(\theta=\sum_iw_iE[Y_i]\).** The \(Y_i\) are independent across items, so the
   fixed-horizon KL/Chernoff bound turns an upper bound on \(S\) into one on \(\theta\). It rises
   with \(S\), so step 1's bound can stand in for \(S\). Under a `rollout` profile the \(Nr\)
   rollouts count as independent blocks, as they do at the fixed horizon.

At \(t=N\) step 1 is exact, and only step 2 remains. The package's labelling protocol uses a
related sequence, `judge_accuracy_confidence_sequence`, a binomial prior-posterior ratio for
binary labels without weights (statistical methods, judge calibration); this one takes
weighted outcomes in \([0,1]\).

Replacing an outcome by a larger value lowers every factor of the capital, and the fractions
are fixed in advance, so the bound stays valid when unfinished or unclassified epochs before
the last observed item count as failures. Leaving such an item out instead would not be valid.
Draws are read in the recorded order, never in completion order.

### Coverage where optstop's rule stops

`python research/optstop_coverage.py` (about 35 s) simulates 200 heterogeneous items with
10 rollouts each, item failure probabilities fixed per row from a Beta distribution with
concentration 4. The run stops once the 97% Beta(1, 1) interval of the pooled rate is at most
0.05 wide, after at least 5 items: optstop's validation settings, on the pooled rate rather
than its hierarchical model. 2,000 runs per row, one-sided \(\alpha=0.05\):

| \(\theta\) | items at stop (median) | anytime bound misses | Clopper–Pearson at the stop misses | Clopper–Pearson, full run, misses | median anytime bound | median item-level KL bound at the stop |
|---|---|---|---|---|---|---|
| 0.0073 | 9 | 0.000 | 0.000 | 0.036 | 0.388 | 0.283 |
| 0.0447 | 30 | 0.000 | 0.208 | 0.025 | 0.184 | 0.188 |
| 0.3089 | 161 | 0.002 | 0.061 | 0.028 | 0.370 | 0.402 |
| 0.4877 | 188 | 0.002 | 0.042 | 0.033 | 0.538 | 0.576 |

The anytime bound misses at most \(\alpha\) in every row. Clopper–Pearson at the stopping time,
the bound a run analysed as if its length had been fixed gets, misses 0.208 near
\(\theta=0.045\). Near 0.007 neither misses: the rule stops after 9 items while Clopper–Pearson is
still conservative there. The anytime bound is close to the fixed-horizon KL bound with items
as the unit at the same stop, so most of its width over pooled Clopper–Pearson is item
heterogeneity, which pooled Clopper–Pearson ignores, rather than the price of anytime
validity. A run stopped after 30 of 200 heterogeneous items says little about the other 170.

### Optstop's released shadow-mode data

Optstop's repository (MIT licence) releases its shadow-mode results as
`paper/paper_results/data/*_shadow.json`. They are run summaries: the trial count at which
optstop would have stopped and the sample ids, but no per-trial outcomes. The bound at the stop
cannot be computed from them. Their runs also record `sample_shuffle: null`, so the analysis
would refuse them for having no recorded order. A shadow-mode replay needs the run's Inspect
log, run with `sample_shuffle` set to a seed.

## Work that builds on this

- #126: certify Inspect runs that optstop stopped early with an anytime-valid bound; see
  [Runs that optstop stopped](#runs-that-optstop-stopped-126).
- #125: FAIL-only lower bounds for attackers that adapt across behaviours. Implemented as
  `adaptive_claims` (experimental), with its own lower bound (statistical methods section
  11.1); it does not wait for phase 1.
