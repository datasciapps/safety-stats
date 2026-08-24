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

## Declaring a sampled design

```yaml
design:
  type: sampled
  inclusion_probability_column: inclusion_probability
```

The design manifest becomes the **frame**. Every frame item needs a row and a positive
inclusion probability; items that were never drawn simply have no observations, and are no
longer reported as `PROMPT_MISSING`. Design weights are declared over the whole frame and are
never renormalized to the realized sample.

Enable `outputs.direct_mean_intervals` and the group and hierarchy weighted means are
estimated by Horvitz--Thompson, reported as `horvitz_thompson_<method>` intervals with
`independence_profile: frame_item`.

### What a sampled design refuses

Four checks reject configurations that a sample cannot support:

- `PLAN_SAMPLED_MEASURE_UNSUPPORTED` -- only `weighted_mean` is available. Maximum, CVaR,
  top-k, entropic, and the threshold measures are nonlinear functionals of the whole frame's
  risk vector and cannot be evaluated from a sample.
- `PLAN_SAMPLED_COMPARISON_UNSUPPORTED` -- comparisons assume every declared prompt was
  evaluated in both arms.
- `PLAN_SAMPLED_REQUIRES_DIRECT_MEAN` -- the propagated prompt envelope has no bound for an
  item that was never drawn.
- `PLAN_SAMPLED_PROMPT_BOUNDS` (warning) -- simultaneous prompt bounds cover the sampled
  items, not the frame.

### Nodes with nothing drawn

A hierarchy node all of whose descendants went unsampled has no identified risk. Those nodes
report no interval and carry a row-level warning, and one aggregated
`HIERARCHY_NODE_UNSAMPLED` diagnostic reports how many there were. This is expected on a deep
hierarchy: in a 200-item frame at \(\pi=0.3\) with one question per four prompts, most
question nodes will be empty.

### Catching a truncated run

The most dangerous failure is a partial evaluation that looks like a smaller sample: the
declared \(\pi\) would then overstate how much of the frame each observed item represents,
biasing the estimate with no visible symptom. The realized sample size is therefore compared
against its Poisson-binomial distribution over the frame, and a deviation beyond four
standard deviations raises `SAMPLE_SIZE_IMPLAUSIBLE` as an error.

## Reading the reported diagnostics

`inflation` and `maximum_weight` are recorded in each interval's assumptions. Two other
fields deserve care:

- `independent_block_count` is the **realized** sample size -- how many frame items were
  actually drawn.
- `effective_sample_size` is the concentration geometry \(1/\sum_j(c_j/C)^2\) over the
  **whole frame**, because the estimator is a sum over every frame term, most of which
  realized as zero. Under equal weights and equal \(\pi\) it equals the frame size. It
  measures weight concentration, not information; precision is governed by `inflation`.

## Not yet supported

Fixed-size designs -- simple random sampling without replacement, or fixed stratum
allocations -- induce negatively correlated inclusion indicators and are not covered.
Supporting them needs a separate argument, not a wider interval. Comparisons across cohorts
on a sampled frame are also unsupported; correcting them means giving each arm its own
Horvitz--Thompson bounds and recombining, which is tractable but is a separate change.
