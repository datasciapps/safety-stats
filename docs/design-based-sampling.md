# Design-based inference for sampled item frames

Everything else in this package conditions on a **census** of the declared design: every
fixed prompt is evaluated, so only rollouts are stochastic. This module covers the case where
the items themselves were drawn from a larger frame with **known inclusion probabilities**.

That is a different claim from the superpopulation story the package declines elsewhere. The
package refuses estimands whose randomization mechanism is imagined; it does not refuse
randomization that actually happened and was recorded. A recorded probability sample from a
real frame is exactly the second kind.

## Estimator

Let the frame contain items \(j=1,\ldots,N\) with declared policy weights \(w_j\) summing
to one over the **whole frame**, and let item \(j\) be included with known probability
\(\pi_j>0\). With inclusion indicator \(I_j\) and observed rate \(\widehat p_j\), the
Horvitz--Thompson estimator of \(R=\sum_j w_j p_j\) is

\[
\widehat R=\sum_j I_j\frac{w_j}{\pi_j}\widehat p_j=\sum_j c_jY_j,
\qquad c_j=\frac{w_j}{\pi_j},\qquad Y_j=I_j\widehat p_j\in[0,1].
\]

\(c_j\) is deterministic, because \(\pi_j\) is known for every frame item -- including the
ones that were not drawn. Dividing by \(C=\sum_jc_j\) leaves an ordinary bounded weighted
mean whose weights sum to one, so the existing weighted KL/Chernoff, empirical-Bernstein, and
Hoeffding engines apply unchanged, and the resulting bound on \(R/C\) is rescaled by
\(C\).

No new concentration mathematics is introduced. The reduction is the whole contribution.

## What it costs

Two quantities are reported rather than buried, because they govern whether the result is
usable at all.

`inflation` is \(C=\sum_j w_j/\pi_j\). It is at least one, equals one exactly under a
census, and is the factor by which inverse-probability weighting widens the interval. At
\(\pi=0.2\) throughout, \(C=5\) and every interval is five times wider before any data is
considered.

`maximum_weight` is the largest normalized coefficient \(c_j/C\). A single rarely-sampled
item can dominate the bound even when the average inclusion probability looks comfortable.
In one worked case a single item at \(\pi=0.02\) among ninety-nine at \(\pi=0.5\) cut the
effective sample size from 100 to 21.

Empirical KL/Chernoff is markedly the tightest of the three engines in the sparse-failure
regime this package targets, and should be the prespecified primary.

## Assumptions

- **Inclusion is independent across frame items** (Poisson sampling). Fixed-size designs --
  simple random sampling without replacement, or fixed stratum allocations -- induce
  negatively correlated indicators and are not covered. Supporting them needs a separate
  argument, not a wider interval.
- **The sample was drawn without reference to outcomes.** Inclusion probabilities must be
  fixed before evaluation, exactly as design weights are.
- Unsampled items contribute an exact zero. Their share of the estimand is carried by the
  sampling randomness inside the bound, not by outcome completion.
- Within a sampled item, unclassified rollouts are completed as safe for the lower endpoint
  and as failures for the upper, as everywhere else.

## Status

**This is an estimator core with no plan surface yet.** Nothing in the analysis pipeline calls
it: `design.type` remains `fixed`, and wiring it needs a sampling-design block in the plan,
manifest columns for inclusion probabilities, and changes to weight resolution and prompt
aggregation.

That makes it, for now, public API the pipeline does not use -- the same condition that got
`holm_adjust` and `benjamini_hochberg` deleted earlier. It is kept because it has three named
consumers rather than none:

1. **Subsampled benchmarks.** Evaluating 500 of 14,000 items is common and currently has no
   correct treatment: reporting the sample mean ignores the design, and a superpopulation
   standard error assumes a frame that was never sampled.
2. **Judge-validation labelling at scale.** Non-uniform label allocation with recorded
   probabilities needs exactly this estimator rather than a within-stratum binomial.
3. **The generative-hierarchy extension**, whose entire inference layer is bounded
   inverse-probability weighting under a defensive proposal.

If none of those is pursued, this module should be deleted rather than left standing.
