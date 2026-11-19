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
mean whose weights sum to one, so the existing weighted KL/Chernoff, empirical-Bernstein, betting, and
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

- **Inclusion is independent across frame items** (Poisson sampling). Fixed-size designs
  break that and are handled separately, by Hoeffding's reduction -- see below.
- **The sample was drawn without reference to outcomes.** Inclusion probabilities must be
  fixed before evaluation, exactly as design weights are.
- Unsampled items contribute an exact zero. Their share of the estimand is carried by the
  sampling randomness inside the bound, not by outcome completion.
- Within a sampled item, unclassified rollouts are completed as safe for the lower endpoint
  and as failures for the upper, as everywhere else.
- Each sampled item enters as **one bounded observation**, its failure rate. The bound never
  reads how many rollouts produced that rate, so extra rollouts per item do not tighten it:
  under a sampled design, budget belongs in more items rather than more rollouts. Separating
  between-item from within-item variance would recover only the rollout share of the
  estimator's variance, which is small whenever the frame is at all heterogeneous -- 2% at
  sixteen rollouts and \(\pi=0.25\) on the worked HarmBench frame, worth about 1% of the
  interval width. It is deliberately not implemented.

A worked demonstration against a known answer is in
[the sampled HarmBench example](../examples/sampled_harmbench/README.md).

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

Four checks reject configurations that a sample cannot support, and two warn:

- `PLAN_SAMPLED_MEASURE_UNSUPPORTED` -- only `weighted_mean` is available. Maximum, CVaR,
  top-k, entropic, and the threshold measures are nonlinear functionals of the whole frame's
  risk vector and cannot be evaluated from a sample.
- `PLAN_SAMPLED_COMPARISON_UNSUPPORTED` -- comparisons assume every declared prompt was
  evaluated in both arms.
- `PLAN_SAMPLED_REQUIRES_DIRECT_MEAN` -- the propagated prompt envelope has no bound for an
  item that was never drawn.
- `PLAN_SAMPLED_PROMPT_BOUNDS` (warning) -- simultaneous prompt bounds cover the sampled
  items, not the frame.
- `PLAN_SAMPLED_SENSITIVITY_PROFILES` (warning) -- the frame item is the independent block
  whatever `independence` declares, so a declared sensitivity profile gets no interval and
  claims get no verdict under it. A claim's `profile_verdicts` hold only its `frame_item` row.

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

The report's scope statement follows the design. A sampled design's `scope` in `report.json`
has `design: sampled` and says that results apply to the declared item frame, from which the
evaluated items were drawn with the recorded inclusion probabilities, and not to prompts
outside the frame. `report-full.md` and `report.html` print it above the claims. The rest of
the report follows it: `report.md` counts frame items rather than fixed prompts and says that
its intervals allow for both the draw and the rollouts, with each drawn item one observation,
and `report-full.md` and `report.html` head the intervals "Sampled-design risk measures".

## Fixed-size designs

```yaml
design:
  type: sampled
  sampling: fixed_size
  stratum_column: sampling_stratum      # omit for plain SRS without replacement
  inclusion_probability_column: inclusion_probability
```

Drawing exactly \(n_h\) of stratum \(h\)'s \(N_h\) frame items makes the inclusion
indicators **negatively correlated** -- they must sum to \(n_h\) -- so the independence the
concentration engines assume does not hold.

It is recovered by **Hoeffding's reduction** (1963, Theorem 4): for any continuous convex
\(f\), a without-replacement sum is dominated by the with-replacement one,

\[
\mathbb E\left[f\!\left(\textstyle\sum_{i\in S}v_i\right)\right]
\le
\mathbb E\left[f\!\left(\textstyle\sum_{i=1}^{n}v_i'\right)\right].
\]

Every engine here bounds a moment generating function, which is convex, so the bound each
computes under independence remains valid without replacement -- and is conservative, since
sampling without replacement has strictly smaller variance.

The estimator rearranges the same way as the Poisson case. Within stratum \(h\), letting
\(A_h\) be the largest target weight in that stratum, each drawn item carries the
deterministic coefficient \(N_hA_h/n_h\) and the bounded value \(a_j\widehat p_j/A_h\).
Those coefficients sum to \(K=\sum_hN_hA_h\) whichever items were drawn, so normalising by
\(K\) again gives an ordinary bounded weighted mean.

With equal design weights **there is no inflation at all**: \(A_h=1/N\) gives \(K=1\), and
the estimator reduces to a plain sample mean. Inflation reappears only for a target confined
to part of the frame -- a hierarchy node consumes stratum draws that land outside it, and
\(K\) reports exactly that cost.

### Declaring the plan

The draw counts are inferred from the same inclusion-probability column, since a fixed-size
design has \(\pi_j=n_h/N_h\) constant inside a stratum. Three checks enforce that reading:

- `SAMPLE_STRATUM_INCONSISTENT` -- inclusion probabilities vary inside a stratum.
- `SAMPLE_DRAWS_NOT_INTEGER` -- \(\pi_hN_h\) is not a whole number of draws.
- `SAMPLE_DRAWS_MISMATCH` -- the run realized a different number of draws than planned.
  Unlike the Poisson case, where a four-sigma deviation is the trigger, a fixed-size design
  must match its plan **exactly**; any shortfall means the declared probabilities misstate
  the sample.

## Drawing a random pilot

A pilot estimates what the full study will cost and roughly what its rates will be. Its
means estimate those quantities without bias only if the pilot prompts are a random sample.
The first prompts of each category are not one: identifiers are often assigned in an order
that tracks length, difficulty or source.

`plan pilot` (experimental) draws a pilot at random within strata of a design manifest:

```bash
inspect-safety-stats plan pilot --manifest design.csv --per-stratum 5 --seed 20261116 \
    --output pilot.csv
```

It draws `--per-stratum` prompts from each combination of the `--strata` columns
(`task_id` by default; an empty value makes the whole manifest one stratum) by simple random
sampling without replacement, and writes the whole manifest with three added columns:

| Column | Meaning |
|---|---|
| `pilot_stratum` | The stratum's values, joined by `/` |
| `pilot_inclusion_probability` | Draws over stratum size, \(n_h/N_h\) |
| `pilot_selected` | `True` for the prompts to run in the pilot |

Each prompt's rank is a SHA-256 hash of the seed and its `prompt_id`, so the draw:

- uses only the manifest and the seed, never an outcome;
- does not depend on the order of the manifest's rows or on the platform;
- is nested: a pilot drawn with a larger `--per-stratum` and the same seed contains the
  smaller one, so a pilot can be extended without discarding the prompts already run.

Record the seed with the pilot. The command refuses a stratum with fewer prompts than
`--per-stratum`, a missing value in a strata column, strata values that would share a label
once joined, a manifest that already has the added columns, and an `--output` that would
overwrite the manifest. Values are written back as they appear in the manifest.

The pilot frame is the frame of a fixed-size sampled design, so the pilot's rates can be
analysed as a sample of the whole manifest without editing any column:

```yaml
design:
  type: sampled
  sampling: fixed_size
  stratum_column: pilot_stratum
  inclusion_probability_column: pilot_inclusion_probability
```

In Python, `draw_pilot(manifest, per_stratum=..., strata=[...], seed=...)` in
`inspect_safety_stats.ingest.pilot` returns the same frame as a DataFrame.

## Not yet supported

Comparisons across cohorts on a sampled frame. Correcting them means giving each arm its own
design-based bounds and recombining, which is tractable but is a separate change.
