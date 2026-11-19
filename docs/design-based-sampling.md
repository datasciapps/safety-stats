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
`independence_profile: frame_item`. The experimental `weighted_betting` method is accepted
too. As on a fixed design, its rows carry the experimental betting assumption and
`chernoff_kernel: common_bet_mixture_jensen_exponent`, and a fixed-size design's row states
why the without-replacement reduction covers the betting capital.

### What a sampled design refuses

Three checks reject configurations that a sample cannot support, and two warn:

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

The Hoeffding, empirical-Bernstein and KL/Chernoff engines bound a moment generating function,
which is convex. The experimental betting engine bounds none. It bounds the capital of each
fixed betting fraction, a product of per-item wealth factors \(b_j\) raised to an exponent
\(c\le1\) that is common within a stratum. The item outcomes are independent across items and
of the draw, so averaging over them first and applying Jensen's inequality gives

\[
\mathbb E\Big[\textstyle\prod_{j\in S}b_j^{\,c}\;\Big|\;S\Big]
\le
\exp\Big(c\textstyle\sum_{j\in S}\log\mathbb E\,b_j\Big),
\]

a convex function of a without-replacement sum of fixed numbers. The reduction applies to it,
and so to the fixed mixture over fractions, and the with-replacement capital is bounded as in
the [betting interval](statistical-methods.md#betting-interval-from-a-fixed-mixture-of-common-bets-experimental).

So the bound each engine computes under independence remains valid without replacement -- and
is conservative, since sampling without replacement has strictly smaller variance.

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

### Checking which items were evaluated (experimental)

The checks above count items; they do not check which items were evaluated. A run on the
first twelve prompts of each stratum by identifier matches a draw of twelve per stratum in
every count and passes them, although it is not a random sample and the declared inclusion
probabilities do not describe it. When the frame records which items the draw selected,
name that column:

```yaml
design:
  type: sampled
  sampling: fixed_size
  stratum_column: sampling_stratum
  inclusion_probability_column: inclusion_probability
  selection_column: drawn        # true for each item the draw selected
```

`analyze` and `validate` then report `SAMPLE_SELECTION_MISMATCH`, an error that makes
integrity invalid and withholds the cohort's bounds, when a cohort has observations for an
item the column does not select or none for an item it selects. The column holds `true` or
`false` (or `yes`/`no`, `1`/`0`) for every frame item; a missing or other value fails
ingestion with a message that names the column. The check works under Poisson sampling too.
It compares the run with the recorded draw, not the draw with chance: that the selection was
random and made before any outcome was seen remains an assumption.

## Drawing a random pilot

A pilot estimates what the full study will cost and roughly what its rates will be. It can
do so only if its prompts are a random sample, weighted as they were drawn (see
[Estimating from a pilot](#estimating-from-a-pilot)). The first prompts of each category are
not a random sample: identifiers are often assigned in an order that tracks length,
difficulty or source.

`plan pilot` (experimental) draws a pilot at random within strata of a design manifest:

```bash
inspect-safety-stats plan pilot --manifest design.csv --per-stratum 5 --seed 20261116 \
    --output pilot.csv
```

It draws `--per-stratum` prompts from each combination of the `--strata` columns
(`task_id` by default; an empty value makes the whole manifest one stratum) by simple random
sampling without replacement, and writes the whole manifest with five added columns:

| Column | Meaning |
|---|---|
| `pilot_stratum` | The stratum's values, joined by `/` |
| `pilot_inclusion_probability` | Draws over stratum size, \(n_h/N_h\) |
| `pilot_selected` | `True` for the prompts to run in the pilot |
| `pilot_seed` | `--seed`, on every row |
| `pilot_per_stratum` | `--per-stratum`, on every row |

Each prompt's rank is a SHA-256 hash of the seed and its `prompt_id`, so the draw:

- uses only the manifest and the seed, never an outcome;
- does not depend on the order of the manifest's rows or on the platform;
- is nested: a pilot drawn with a larger `--per-stratum` and the same seed contains the
  smaller one, so a pilot can be enlarged for planning without discarding the prompts
  already run. An enlarged pilot certifies nothing; see
  [What a pilot can show](#what-a-pilot-can-show).

The frame records the seed and the per-stratum count, so the frame alone repeats the draw,
and a report whose design manifest is the frame covers them with the manifest's hash. The
command refuses a stratum with fewer prompts than `--per-stratum`, a missing value in a
strata column, strata values that would share a label once joined, a manifest that already
has the added columns, and an `--output` that would overwrite the manifest. Values are
written back as they appear in the manifest. The draw reads strata as text, but `analyze`
reads the pilot frame with pandas' type inference, so the command also refuses a stratum
label that pandas reads as missing (`N/A`, `None`, `NA`) and labels that it reads as one
value (`1` and `01`, `true` and `True`): `analyze` would otherwise check the draw against
strata it did not use.

In Python, `draw_pilot(manifest, per_stratum=..., strata=[...], seed=...)` in
`inspect_safety_stats.ingest.pilot` returns the same frame as a DataFrame. Read the manifest
as the command does, with `pd.read_csv(path, dtype=str, keep_default_na=False)`: the draw
hashes each `prompt_id` as written, and pandas' defaults would read an identifier such as
`0001` as the number 1. `draw_pilot` refuses a `prompt_id` that is not text or is blank.

### Estimating from a pilot

Every stratum gives the same number of prompts, so a prompt in a small stratum is more
likely to be drawn than one in a large stratum, and the pilot over-represents small strata.
In the [HarmBench example](../examples/real_harmbench/design.csv), `--per-stratum 2` with
the default `--strata task_id` draws two of the 100 copyright prompts and two of the 300
others: copyright is half of the pilot and a quarter of the manifest. The pilot's plain
mean, of failures or of answer length, is therefore biased for the manifest's whenever
strata differ in size and in that quantity.

Weight each pilot prompt by `1 / pilot_inclusion_probability`. Over the drawn prompts \(S\)
of a manifest of \(N\), \(\frac1N\sum_{j\in S}y_j/\pi_j\), which scales each stratum's pilot
mean by its share of the manifest, is unbiased for the manifest mean, and
\(\sum_{j\in S}y_j/\pi_j\) for the manifest total, such as the tokens one rollout of every
prompt would use. Analysing the pilot frame as a sampled design does the weighting and bounds
the result.

The pilot frame is the frame of a fixed-size sampled design, so the pilot's rates can be
analysed as a sample of the whole manifest without editing any column. Name
`pilot_selected` as the selection column, so that the analysis checks that the prompts run
are the ones drawn and not only that their number per stratum is right:

```yaml
design:
  type: sampled
  sampling: fixed_size
  stratum_column: pilot_stratum
  inclusion_probability_column: pilot_inclusion_probability
  selection_column: pilot_selected
```

### What a pilot can show

Analyse a pilot with `analysis.intent: exploratory`. An exploratory analysis certifies
nothing, and its descriptive status says what a confirmatory analysis of the same bounds would
give. A pilot's interval holds for a draw whose size was fixed before the pilot ran, but a
pilot exists to inform choices made after its outcomes are seen. Enlarging it after seeing
them and analysing it again until a claim passes is the repeated peeking of
[Statistical methods, section 6.1](statistical-methods.md#61-why-fixed-sample-bounds-cannot-be-repeatedly-peeked-at):
each look at a size fixed in advance passes a false claim with probability at most
\(\alpha\), but stopping at the first pass does so more often. In a simulation with KL/Chernoff
bounds at a one-sided \(\alpha\) of 0.05, a frame of 2,000 items whose mean risk equals the
claim's limit of 0.05 was drawn at 20, 40, ..., 600 items, nested as `plan pilot` nests its
draws. Each size alone passed with probability at most 0.045, but stopping at the first size
that passed did so with probability 0.068 (Monte Carlo standard error 0.003). An extension
decided after the outcomes is therefore not a fixed-size confirmatory design, whatever the
frame and the plan say.

A confirmatory analysis uses fresh rollouts under a plan frozen before they ran: of the whole
manifest as a fixed design, or of a new draw, with a new seed, whose size the frozen plan
fixes. Do not enlarge the pilot's own draw for it: with the same seed, the larger sample
would contain the pilot's prompts, whose outcomes already shaped the decision to run it and
its size.

## Not yet supported

Comparisons across cohorts on a sampled frame. Correcting them means giving each arm its own
design-based bounds and recombining, which is tractable but is a separate change.
