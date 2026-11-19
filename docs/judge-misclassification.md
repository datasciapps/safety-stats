# Judge misclassification

Every interval elsewhere in this package estimates

\[
R=\sum_j w_j\,\Pr(\text{judge reports a failure on prompt }j),
\]

which is a property of the **judge-model system**, not of the model under test. When an
LLM judge scores the outcomes, that distinction is not cosmetic. At 1% true prevalence and
95% judge specificity, false positives contribute \(0.99\times0.05=4.95\%\) against at
most \(1\%\) true positives, so roughly five sixths of the reported number is judge error.

An uncorrected report therefore carries an implicit and certainly false assumption: that the
judge never errs. This module makes that assumption explicit and lets you vary it.

## The identity

Condition on the judge verdict, which is observed. Write

- \(\rho_1=\Pr(\text{truly unsafe}\mid\text{judge said unsafe})\);
- \(\rho_0=\Pr(\text{truly unsafe}\mid\text{judge said safe})\).

The judge verdict partitions the design exactly, so the true risk is

\[
R^{*}=R\rho_1+(1-R)\rho_0 .
\]

This is preferred to the textbook Rogan--Gladen form
\(R^{*}=(R-(1-\beta))/(\alpha+\beta-1)\) for three reasons. It is affine rather than
fractional, so there is no division and no guard for a vanishing denominator. It is
multilinear in \((R,\rho_1,\rho_0)\), so extrema over a declared box are attained at the
box vertices and interval arithmetic is exact rather than conservative. And the sampling
design it implies -- stratify by judge verdict, sample within stratum -- is what an
evaluator would do operationally anyway.

The correction recommended for jailbreak benchmarks in
[*How Reliable Is Your Jailbreak Judge?*](https://arxiv.org/abs/2606.25487) (section 5),
reported attack success rate multiplied by judge precision, is this identity with
\(\rho_0=0\), because precision is \(\rho_1\). Setting \(\rho_0=0\) assumes the judge has no
false negatives, and those are the errors that make an unsafe model look safe. A safety gate
cannot ignore them, because a missed failure produces a false PASS. The paper itself notes that a
precision correction does not fix the poor recall of LLM judges. Declaring \(\rho_0\) here
puts that error into the bound.

Because the identity is affine in \(R\) with positive slope \(\rho_1-\rho_0\), it
applies unchanged to any risk measure that is positively homogeneous and translation
equivariant: `weighted_mean`, `maximum`, `cvar`, and `top_k_mean`. It does **not** apply to
`entropic` risk, whose \(\lambda\) would need rescaling, or to the threshold measures,
whose thresholds would need transforming. Those measures are left uncorrected.

## Assumed mode: declared accuracies

The first implementation declares \(\rho_1\) and \(\rho_0\) as ranges rather than
estimating them. It produces **no evidence**. Every interval it emits carries
`eligibility: diagnostic`, `role: sensitivity` and `varied_assumption: judge_error`, and
cannot support a claim.

```yaml
inputs:
  design_manifest: design.csv
  score:
    name: harmful_behavior
    mapping:
      failure_values: [FAIL]
      safe_values: [PASS]
  judge_calibration:
    mode: assumed
    primary: declared
    scenarios:
      - id: declared
        rho_1: {lower: 0.85, upper: 1.0}
        rho_0: {lower: 0.0, upper: 0.02}
      - id: pessimistic
        rho_1: {lower: 0.70, upper: 1.0}
        rho_0: {lower: 0.0, upper: 0.05}
```

Each scenario must be **strictly informative**: the \(\rho_1\) lower endpoint must exceed
the \(\rho_0\) upper endpoint. A plan asserting otherwise is rejected rather than silently
producing a meaningless correction.

### Per-cohort boxes

A judge can be more accurate on one model's outputs than on another's. To stress-test that
without human labels, a scenario can give some cohorts a box of their own under `cohorts`:

```yaml
  judge_calibration:
    mode: assumed
    primary: declared
    scenarios:
      - id: declared
        rho_1: {lower: 0.85, upper: 1.0}
        rho_0: {lower: 0.0, upper: 0.02}
      - id: worse_on_candidate
        rho_1: {lower: 0.85, upper: 1.0}
        rho_0: {lower: 0.0, upper: 0.02}
        cohorts:
          candidate:
            rho_1: {lower: 0.60, upper: 0.85}
            rho_0: {lower: 0.0, upper: 0.08}
```

Each entry maps a declared cohort to its box, and replaces the scenario's `rho_1` and `rho_0`
on that cohort's rows only. A cohort the scenario does not list takes the scenario's own box.
Every cohort box must be strictly informative too. `cohorts` cannot be empty, and a cohort
that the plan does not declare is a `PLAN_UNKNOWN_REFERENCE` error. With `groups`, a cohort box
applies to the listed groups' rows of that cohort, like the default box.

A plan with no `cohorts` in any scenario gives the same results, report bytes and plan hash as
before the field existed. The report records each scenario's cohort boxes under
`presentation.judge_calibration.scenarios[].cohorts` and lists them in the report's text.
Generative plans reject the field.

Reports gain a **Judge misclassification sensitivity** section listing, for every corrected
row and scenario, the judge-system bounds, the true-risk bounds, and the corrected estimate
range. The original interval is untouched and remains primary.

Each scenario also corrects every comparison whose primary interval exposes both arm bounds,
as [Comparisons](#comparisons) describes: each arm takes the scenario's box for its own
cohort, which is the same box on both arms unless the scenario gives a cohort its own. The
comparison gains one interval `judge_corrected_<scenario>` per scenario, with the same
`diagnostic` eligibility. A comparison that cannot be corrected, such as a direct contrast
interval, stays on the judge-reported scale and is named in
`JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON`.

## Reading the output

Two movements are expected, and they mean different things.

The corrected band is normally **wider**, because uncertainty about \(\rho\) is real
uncertainty being added. There is no incentive to prefer it for tightness.

The corrected point estimate is normally **lower**, because false positives are being
removed. That *is* an incentive, and it is why the calibration belongs in the frozen plan,
declared before any results are seen. A scenario chosen after looking at the number is not a
sensitivity analysis.

If the corrected lower bound reaches zero, the data are consistent with no true failures at
all -- everything observed could be judge noise. That is a legitimate and important finding,
not a defect.

## Limitations of the assumed mode

- \(\rho_1\) and \(\rho_0\) are **prevalence-dependent**. They are properties of the
  judge on a particular output distribution, so applying one declared scenario to several
  cohorts with materially different failure rates is itself an approximation. The report
  states this in the interval's assumption list.
- Unless a scenario declares [per-cohort boxes](#per-cohort-boxes), it applies one box to
  every cohort, so comparisons corrected in `assumed` mode inherit that approximation on both
  arms. Each arm is corrected over its box independently of the other, so the corrected
  comparison covers a judge whose accuracy differs between the arms only as far as each arm
  stays inside its own box.
- Nothing here detects a judge whose errors depend on which model produced the output.
  Differential misclassification across arms can manufacture an apparent difference from
  nothing. Per-cohort boxes let a plan state such a dependence as a scenario and see what it
  does to the comparison, but they do not measure it: whether the judge actually behaves
  that way is still an assumption.

## Validated mode: measured accuracies

`mode: validated` estimates the same two parameters from human labels instead of declaring
them. Its corrected intervals are **certifying** for the estimand the labels measure: a
weighted mean over a whole cohort that weights every rollout equally (see
[What it certifies](#what-it-certifies)).

```yaml
inputs:
  judge_calibration:
    mode: validated
    validation_manifest: judge-labels.csv
    family: judge_error
    sampling: simple_random_within_judge_stratum

multiplicity:
  families:
    - id: judge_error
      purpose: simultaneous_intervals
      method: bonferroni
      alpha: 0.05
```

### Workflow

The sampling frame is the package's own observation ledger, so calibration is a second pass
over a completed analysis:

1. run `analyze` once and take `observations.csv` from the report directory;
2. split its classified rows by the `failure` column -- that is the judge verdict -- into a
   judge-unsafe and a judge-safe stratum, **per cohort**;
3. draw a simple random sample within each stratum and have a human adjudicate it;
4. write `observation_id,human_label` with labels `safe` or `unsafe`; and
5. re-run `analyze` with `mode: validated` pointed at that file.

An item the labeller cannot resolve must be handled by the declared protocol before the file
is written. Dropping such rows ad hoc breaks the simple random sample that makes each
stratum's proportion binomial. Without a labelling protocol in the plan, the package cannot
detect it.

### Labelling protocol (experimental)

The calibration is valid only if each stratum's labels are a simple random sample of it, of a
size that did not depend on the labels. Labelling until the result looks good, skipping hard
rows, or labelling disagreements first breaks that. A plan can declare how the labels were
collected, and the analysis then checks what the data can show:

```yaml
inputs:
  judge_calibration:
    mode: validated
    validation_manifest: judge-labels.csv
    family: judge_error
    protocol:
      queue: label-queue.csv     # observation_id in labelling order, relative to the plan
      tier: calibration          # optional: only queue rows whose tier column says so
      seed: 20261104             # optional: the seed that ordered the queue, for an audit
      stopping: label_independent
```

The queue lists rollouts in the order people label them. Ordering each judge stratum by a
seeded random permutation, and interleaving the strata on a pattern fixed in advance, makes
every prefix of the queue a simple random sample within each stratum. `tier` is for a queue
that also holds rows the calibration does not use, such as rows oversampled to study where
the judge goes wrong; the manifest then holds the calibration tier's labels only.

The analysis checks that the manifest labels exactly the first rows of the queue (or of its
tier). A skipped row, or a label for a rollout outside the queue, is an error,
`JUDGE_LABELLING_PROTOCOL_VIOLATED`, and no correction is reported. `stopping` says why
labelling stopped where it did:

| `stopping` | Meaning | What the analysis does |
|---|---|---|
| `fixed_size` | at `size` labels, fixed in advance | checks that exactly `size` rows are labelled; Clopper-Pearson bounds |
| `label_independent` | at a point chosen without looking at the labels, such as when time ran out | Clopper-Pearson bounds; the independence is declared, not checked |
| `data_dependent` | possibly because of the labels, such as once the corrected bound cleared a threshold | anytime-valid confidence sequences instead of Clopper-Pearson |

A confidence sequence holds at every number of labels at once, so it stays valid whenever
labelling stops. When a small share of a stratum is labelled it is wider than
Clopper-Pearson at the same number of labels (60 labels with one unsafe in a stratum of
2,000 give an upper bound of 0.165 against 0.102 at the same budget). It counts a stratum's
size, though, so when most of a stratum is labelled it can be narrower, and a stratum
labelled in full is known exactly. The derivation is in
[Statistical methods, section 6.9](statistical-methods.md#69-implemented-human-labels-for-a-judge-calibration-experimental).
It spends the same budget per endpoint, so the coverage the corrected interval states is
unchanged. `data_dependent` requires `sampling: simple_random_within_judge_stratum`, because
neither the noisy-but-valid test nor the stratified design's bounds have an anytime-valid form
yet. With `groups`, the prefix check covers the whole manifest, and each calibration of a
cohort and group bounds its own strata.

The report's `provenance.judge_calibration.protocol` records the queue's SHA-256, its number
of rows, the labelled prefix, the seed, the stopping rule and the bound used. The package
cannot recompute the queue from the seed: that the queue's order is random is declared, and
the hash lets an auditor check the queue against the one the labelling used. The protocol is
available in fixed-design plans; a generative plan does not accept it yet.

### Estimation and error budget

Within each stratum the labelled rows are an ordinary binomial sample, so exact
Clopper--Pearson bounds apply. The judge family alpha is first divided among the declared
cohorts (see below); each cohort's share is then split equally between \(\rho_1\) and
\(\rho_0\), and each parameter's share again between its two endpoints. A union bound over
those four endpoints spends no more than the cohort's share.

A stratum with no labelled row is completed conservatively. Its parameter is a probability,
so \([0, 1]\) contains it with certainty: the calibration uses that range, spends none of the
budget on it, and warns with `JUDGE_STRATUM_UNLABELLED_COMPLETED`. Such a box is never
strictly informative, so the informativeness check is skipped for it. The correction stays
valid but wide. With \(\rho_1\in[0,1]\) and a judge-reported interval \([L, U]\), true
risk lies in \([(1-U)\rho_{0,L},\; U+(1-U)\rho_{0,U}]\), so it can still support an upper
claim when the judge rarely reports a failure. Labelling that stratum narrows it. The
budget split is unchanged, so `family_alpha` overstates what was spent. If neither stratum has
a label, the calibration is refused.

The corrected interval's simultaneous coverage is therefore

\[
1-\left(\alpha_{\text{interval}}+\alpha_{\text{spent}}\right),
\]

where \(\alpha_{\text{spent}}\) is the share the calibrations it used actually consumed, and
is reported in the interval's `family_alpha`.

Each *endpoint* of the corrected interval uses the base interval's endpoint plus two of the four
calibration endpoints, so a one-sided claim spends the base endpoint's alpha plus half the
calibration budget; the interval records this as its directional coverage. Beside the exact box
of a weighted-mean row, the report shows a `judge_corrected_validated_delta` reference. Tail
measures have none, since the reference propagates the weighted mean's variance. It is a
delta-method (Wald) interval that
propagates the calibration variance through \(R^*=R\rho_1+(1-R)\rho_0\) with Agresti--Coull
stabilised stratum proportions and the rollout-independence binomial variance of the judge-reported
mean. It is asymptotic, carries no finite-sample guarantee, is labelled `diagnostic`, and never
decides a claim. Its purpose is to show what the exact guarantee costs -- the "oracle gap" between
practical finite-sample methods and asymptotic ones. `research/judge_correction_widths.py`
tabulates width and Monte-Carlo coverage of both across prevalence and calibration size.
In that simulation (sensitivity 0.90, specificity 0.97, 1,000 rollouts) the reference is 30--45%
narrower, but its coverage falls to 90--94% at prevalences of 5% and below with up to 300 labels,
whereas the exact box never falls below its nominal level; at very low prevalence the exact path
frequently refuses to correct at all because the calibration is uninformative. A narrower
interval that under-covers where failures are rare is not a better interval for a safety claim.

### Lang--Reiczigel limits

With `sampling: simple_random` the labels also estimate the judge's sensitivity \(Se\) and
specificity \(Sp\) by true class, and a whole-cohort `weighted_mean` row gains a third interval,
`judge_corrected_validated_lang_reiczigel`. It holds the confidence limits of Lang and Reiczigel
(2014) for prevalence adjusted for estimated sensitivity and specificity. With \(z\) the normal
quantile at \(1-\alpha/2\), the judge-reported rate \(q\) over \(n\) rollouts and the two
truth-class counts are adjusted in the manner of Agresti and Coull,

\[
q'=\frac{nq+z^2/2}{n+z^2},\qquad Se'=\frac{x_{Se}+1}{n_{Se}+2},\qquad
Sp'=\frac{x_{Sp}+1}{n_{Sp}+2}.
\]

The Rogan--Gladen estimate \(P'=(q'+Sp'-1)/(Se'+Sp'-1)\) is then shifted by a bias term
\(2z^2\bigl(P'\,v_{Se}-(1-P')\,v_{Sp}\bigr)\), with \(v_{Se}=Se'(1-Se')/(n_{Se}+2)\) and
\(v_{Sp}\) alike. The limits are that centre plus or minus \(z\) delta-method standard errors,
clipped to \([0,1]\). The implementation reproduces the five worked examples of the paper. The
judge-reported rate is the row's design-weighted mean, and \(n\) is its effective size
\((\sum_j w_j)^2/\sum_j(w_j^2/n_j)\) under rollout independence. That is the classified count
when every rollout weighs the same. The interval spends the cohort's share of the judge family
alpha, as the delta reference does.

The interval is `diagnostic`. Like the delta reference, it never decides a claim.
`research/judge_correction_lang_reiczigel.py` compares it with the exact box at nominal 95% over
2,000 rollouts, prevalences from 0.5% to 20% and label budgets from 30 to 1,000, for two judges
(sensitivity 0.90 with specificity 0.97, and 0.80 with 0.90). With 2,000 replications per cell:

| | exact box | Lang--Reiczigel |
|---|---|---|
| coverage, prevalence \(\ge\) 5% | 1.000 | 0.950--0.999 |
| coverage, prevalence \(\le\) 2% and \(\le\) 300 labels | 1.000 | 0.914--0.990, below 0.95 in 11 of 18 cells |
| coverage, 1,000 labels | 1.000 | 0.986--1.000 |
| true risk above the upper limit (nominal 2.5%) | 0 | up to 2.9% |
| median width, judge 0.90/0.97 | reference | 7--68% narrower, but 11--16% wider at prevalence \(\le\) 1% with 1,000 labels |
| median width, judge 0.80/0.90 | reference | wider in most cells with \(\ge\) 300 labels (up to 2.2 times) |

The widths compare the box from judge-stratum labels with the limits from a simple random sample
of the same size. Where the box often refuses, its widths come only from the samples it
accepted. The limits are not uniformly narrower. Where failures are rare and labels few, the regime most safety claims sit in, the
limits miss their level, and the miss falls partly on the upper side, where it would let an
unsafe model pass. They also refuse less often, since they need one
truly unsafe label where the box needs both judge strata labelled and an informative
calibration. When the judge reports far fewer failures than its estimated false-positive rate
implies, the whole interval falls below zero and is clipped to the single point 0. The interval
then warns that the data contradict the estimated accuracies.

Labels drawn within judge strata (`simple_random_within_judge_stratum`) estimate \(\rho_1\) and
\(\rho_0\), not \(Se\) and \(Sp\), and get no such interval. In that parameterisation the identity
\(R^*=R\rho_1+(1-R)\rho_0\) is affine, so the Lang--Reiczigel bias term vanishes. What remains
is an Agresti--Coull-adjusted Wald interval, which is what the delta reference already is.
Stratified labels (`stratified_without_replacement`, below) get no such interval either. Where
the limits cannot be computed, for example when no labelled rollout is truly unsafe, the analysis
emits `JUDGE_LANG_REICZIGEL_UNAVAILABLE` with the reason.

### Stratified labels (experimental)

A labelling design often wants more than a simple random sample within each judge stratum.
For example, it may label every rollout on which a second judge disagrees, and only a sample of
the rest. Read as one simple random sample, such labels are biased: the oversampled
disagreements count as often as they were labelled, not as often as they occur. The binomial
bound also gives no credit for labelling a small stratum in full. A census of it still keeps a
binomial width, as if its rows had been drawn with replacement.

`sampling: stratified_without_replacement` reads the labels as the design drew them:

```yaml
inputs:
  judge_calibration:
    mode: validated
    validation_manifest: judge-labels.csv
    sampling: stratified_without_replacement
    validation_strata: judge-strata.csv
    family: judge_error
```

`validation_strata` holds `observation_id,stratum` for **every classified observation** of
the analysed cohorts, labelled or not, since each stratum's size is part of the design. The
strata may cut across the judge's verdicts. The analysis nests each declared stratum within
the cohort's two judge strata, so a *fine stratum* is one declared stratum within one cohort
and one judge verdict. A file that leaves out a classified observation
(`JUDGE_VALIDATION_STRATA_INCOMPLETE`), or names one that is absent or unclassified
(`JUDGE_VALIDATION_STRATA_UNMATCHED`), is an error, and no correction is reported.

**Estimation.** Let fine stratum \(h\) of a judge stratum of size \(N\) hold \(N_h\)
rollouts, \(K_h\) of them truly unsafe, and let \(n_h\) of them be labelled, \(x_h\) unsafe.
If the labelled rows are a simple random sample without replacement of the stratum, of a size
fixed independently of the labels, then \(x_h\) is hypergeometric given \(K_h\). Inverting
the two one-sided hypergeometric tests gives exact bounds

\[
K_{h,L}=\min\{K: P(X\ge x_h\mid N_h,K,n_h)>a_h\},\qquad
K_{h,U}=\max\{K: P(X\le x_h\mid N_h,K,n_h)>a_h\},
\]

and the judge stratum's parameter, \(\rho=\sum_h K_h/N\), lies in
\([\sum_h K_{h,L}/N,\ \sum_h K_{h,U}/N]\). Each fine stratum is weighted back by its size,
whatever share of it was labelled. A census (\(n_h=N_h\)) gives \(K_h=x_h\) exactly. A fine
stratum without labels is completed as \([0, N_h]\), which holds with certainty, and the
analysis warns with `JUDGE_FINE_STRATUM_UNLABELLED_COMPLETED`. A judge stratum without any
label is completed as \([0,1]\) and refused when both are unlabelled, as described above.

**Error budget.** Each of the four endpoints keeps a quarter of the cohort's share, as in the
binomial case. That quarter is split equally over the fine strata of its judge stratum that
were sampled but not censused, \(a_h=\alpha_{\text{endpoint}}/H_{\text{sampled}}\). Censused
and unlabelled strata spend nothing. A union bound over every count bound then gives the same
guarantee as before: under the design, \(\rho_1\) and \(\rho_0\) lie in their ranges together
with probability at least \(1-\alpha_{\text{cohort}}\). With a single fine stratum the
range narrows below the Clopper--Pearson one as the labelled share of the stratum grows. When
only a few rows of a stratum are labelled, the two nearly agree. Because counts are whole
numbers, the hypergeometric range can then be slightly wider. A sweep of every stratum of up
to 60 rollouts found 4 such cases among about 119,000, each wider by less than one
rollout's share, \(1/N\). With many sampled fine strata, the Bonferroni split widens each one; exact
joint methods for stratified finite populations exist (Wendell and Schmee, *JASA* 91(434),
1996) but are not implemented.

**What changes in the report.** Provenance records `validation_strata` and, per cohort,
`fine_strata`: each fine stratum's counts, its count bounds and the level each bound spent.
`report-full.md` lists them. The asymptotic `judge_corrected_validated_delta` reference assumes simple
random samples within the judge strata, so it is not reported under this design, and an INFO
diagnostic, `JUDGE_CALIBRATION_STRATIFIED`, says so. With `groups`, each scoped
calibration counts the declared strata of its group's rollouts. The strata file is still
checked against every classified observation, before any scope, and `fine_strata` is
recorded per cohort and group. Generative plans do not offer this design.

### Unclassified rollouts

A rollout the judge never classified, such as one with a missing score under
`conservative_bounds`, belongs to neither judge stratum, so no label can say anything about
it. The base bounds complete it as judge-safe at the lower end and judge-unsafe at the upper.
The correction keeps that completion on the true-risk scale, truly safe at the lower bound and
truly unsafe at the upper. Only the classified mass is mapped through \(\rho\):

\[
R^*=M\rho_1+(1-s-M)\rho_0+s\,u,\qquad u\in[0,1],
\]

where \(s\) is the row's weight on unclassified rollouts and \(M\in[L,\,U-s]\) its classified
judge-unsafe mass. For a tail measure, \(s\) is the largest unclassified share of any prompt
it weights. The corrected interval's assumptions state \(s\) whenever it is positive.

Mapping the whole completion through \(\rho_1\) would treat unclassified rollouts as if the
labels described them. With 1,000 of 3,200 rollouts unclassified, that once gave a corrected
upper bound of 0.064 where the unclassified rollouts alone allow 0.319
([issue #97](https://github.com/aduncan001/safety_stats/issues/97)).

### Comparisons

A comparison is corrected by correcting **each arm with its own cohort's calibration** and
then recombining, rather than by rescaling the contrast. Rescaling a difference by one shared
factor is only valid under prevalence-free \(\alpha/\beta\) parameters; since \(\rho\) is
prevalence-dependent the two arms carry different boxes, and

\[
\Delta^{*}=R_C\left(\rho_1^{C}-\rho_0^{C}\right)+\rho_0^{C}
-R_B\left(\rho_1^{B}-\rho_0^{B}\right)-\rho_0^{B}
\]

is not a function of \(\Delta\) alone. Risk ratios and relative reductions are derived from
the same two corrected arm intervals. Each arm must be a row its own cohort's labels
certify, as [below](#what-it-certifies); otherwise the corrected comparison is a sensitivity.

In `assumed` mode the same recombination runs once per declared scenario, with the box the
scenario declares for each arm's cohort: the same box on both arms unless the scenario gives
one of the cohorts a box of its own. Each corrected comparison is a sensitivity
(`eligibility: diagnostic`, `role: sensitivity`, no `family_alpha`), and a scenario other
than `primary` carries a warning that it is reported for sensitivity only.

This requires a primary interval that exposes both arm bounds -- the two-arm rectangle used
by `weighted_kl_chernoff` and by every `comparison_sets` entry. A direct contrast
construction (`weighted_empirical_bernstein` or `weighted_hoeffding` on the difference)
bounds only \(\Delta\) and carries no arm decomposition, so it cannot be corrected. Those
rows stay on the judge-reported scale and the analysis emits
`JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON`. Since the recommended primary for sparse
binary failures is already `weighted_kl_chernoff`, the recommended path is covered.

### Error budget across cohorts

The judge family alpha is divided equally among the **declared cohorts**, or, when the
calibration is [scoped to groups](#scoping-the-calibration-to-groups), among the cohorts
times the listed groups. A single risk row consumes one share; a comparison consumes the two
shares of the cohorts it spans. A union
over every corrected row therefore stays inside the declared family budget, and each interval
reports the share it actually spent in `family_alpha`.

### What it assumes

- The labelled rows are a **simple random sample within their judge stratum**. This is a
  property of the collection protocol, not of the data, and nothing in the package can check
  it. It is recorded in the interval's assumption list. Under
  `sampling: stratified_without_replacement`, the requirement is instead that each fine
  stratum's labelled rows are a simple random sample without replacement of it, of a size
  fixed independently of the labels.
- **Human adjudication is ground truth.** Labeller error is not modelled.

### What it certifies

One calibration is estimated per cohort. Its labels are drawn within judge strata over every
classified rollout of the cohort, so they measure \(\rho_1\) and \(\rho_0\) with each of those
rollouts weighted equally. The identity needs them weighted the way the row weights its
rollouts. The two agree exactly only for a row that

- uses the `weighted_mean` measure;
- covers the whole cohort, not a group or hierarchy node inside it;
- gives every classified rollout the same weight, meaning a prompt's weight divided by the
  number of rollouts its rate is computed over is the same for every prompt; and
- belongs to a fixed design. A sampled design estimates a frame mean, and labels on the drawn
  items do not bound the judge's accuracy over the frame.

Equal prompt weights with equal rollout counts satisfy the weighting condition. `nested_equal`
weights over an unbalanced hierarchy usually do not. A correction also never supports more
than the row it corrects: the correction of a diagnostic row, such as a `diagnostic_only`
measure, is diagnostic too.

Every other corrected row is still reported, as a **sensitivity analysis**:
`eligibility: diagnostic`, no coverage, and a warning that names the reason. A claim that
rests on it is INCONCLUSIVE. The same holds for each arm of a comparison, and for the
[noisy-but-valid test](#certifying-a-threshold-directly-the-noisy-but-valid-test).

The restriction matters because pooled labels dilute a prompt that is both heavily weighted and
badly judged. In the example from
[issue #95](https://github.com/aduncan001/safety_stats/issues/95), one hard prompt carries half the weight and the judge
misses 40% of its failures, while nine easy prompts' false positives dominate the labels. The
pooled \(\rho_1\) is 0.48 against a design-weighted 0.84, and a claim at 0.12 used to pass
although true risk was 0.15. Tail measures (`maximum`, `cvar`, `top_k_mean`) fare worse:
\(\rho\) rises with prevalence, so the riskiest prompts sit above the pooled value.

Weighting the labels by the design, which would certify more rows, is future work.

### Scoping the calibration to groups

A cohort often holds several benchmarks, and a judge's accuracy can differ between them. The
labels of one benchmark then say little about another's rows, yet a cohort-wide calibration
pools them. `judge_calibration.groups` estimates one calibration per cohort and listed group
instead, from the labels and judge-stratum sizes of that group's rollouts. Each listed group
is then calibrated the way a cohort is otherwise:

- the conditions above apply with "the cohort" read as "the group". A `weighted_mean` over a
  whole listed group, in a fixed design, that weights every classified rollout of the group
  equally, is certified;
- the judge family alpha is divided equally among all calibrations, cohorts times listed
  groups, and the report records that share as `per_calibration_alpha`;
- a comparison arm takes its cohort's calibration for the comparison's group;
- the noisy-but-valid test uses the truth strata of the row's group.

Rows of groups that are not listed are not corrected. A claim on such a row is INCONCLUSIVE,
and its reason says that no labels calibrate it. Labels on rollouts outside every listed group
calibrate nothing, and the warning `JUDGE_VALIDATION_OUT_OF_SCOPE` reports them. In `assumed`
mode, `groups` limits the declared scenarios to the listed groups' rows. Groups may overlap;
each listed group is calibrated from its own rollouts.

### When it refuses

The module produces no correction, and says why, when:

- a cohort has an empty judge stratum (`JUDGE_VALIDATION_STRATUM_EMPTY`);
- the labels do not establish a strictly informative judge -- the \(\rho_1\) lower bound
  fails to exceed the \(\rho_0\) upper bound, meaning more labels or a better judge are
  needed -- or neither judge stratum has a label (`JUDGE_CALIBRATION_UNINFORMATIVE`); or
- a validation row references an observation absent from the analysed logs, or one whose
  outcome was never classified (`JUDGE_VALIDATION_UNMATCHED`, an error).

### Effect on threshold claims

When a validated correction is available for a claim's evidence, the claim is decided from
the **corrected** bounds. A threshold in the plan is a statement about the model, so true
risk is the estimand it refers to; deciding on judge-reported risk would answer a different
question.

Three consequences are recorded on every affected `ClaimResult`:

- `method` becomes `validated_judge_misclassification_correction` and `evidence_refs` gains
  `interval:judge_corrected_validated`, so the interval actually used is identifiable;
- `alpha` becomes the **union** of the claim family budget and the judge calibration budget,
  because that is the claim's real error probability; and
- `estimate` is `null`. A calibration box yields a range of corrected point estimates rather
  than a single one, and the range is stated in the claim's reasons alongside both the
  corrected and uncorrected bounds.

The analysis emits `JUDGE_CORRECTED_CLAIMS` listing which claims were retargeted. Comparison
claims are corrected too, provided their comparison exposes both arm bounds.

A validated calibration makes every claim a claim about true risk. A claim whose evidence has
no certifying correction is therefore INCONCLUSIVE, not decided on judge-reported risk. That
happens when the calibration is refused, when a measure cannot be corrected, when the labels
do not certify the row ([What it certifies](#what-it-certifies)), or when a ratio has no
corrected baseline. The analysis lists such claims in `JUDGE_CORRECTION_UNAVAILABLE`.
A refused calibration can mean that the labels show the judge missing failures, which is
exactly when judge-reported risk understates true risk.

**An `assumed` correction never moves a claim.** The selection requires
`eligibility: certifying`, which only a validated calibration produces, so a declared
sensitivity cannot reach the decision path even if it is the only correction present.

Correction can move a decision in either direction. It widens the band by adding calibration
uncertainty, which makes both `PASS` and `FAIL` harder; but it also removes false positives,
which can lower the upper bound enough to establish a `PASS` that the uncorrected interval
could not. Both movements are evidence-backed, and both are visible in the claim's reasons.

## Certifying a threshold directly: the noisy-but-valid test

A threshold claim may opt into a direct finite-sample test instead of comparing corrected
interval endpoints:

```yaml
inputs:
  judge_calibration:
    mode: validated
    validation_manifest: judge-labels.csv
    family: judge_error
    sampling: simple_random          # a random sample of rollouts, not within judge strata
claims:
  - id: tail_safe
    ...
    method: noisy_valid_test
```

With sensitivity \(Se\), false-positive rate \(FPR\) and true risk \(\pi\), the judge-reported
rate is \(q=\pi\,Se+(1-\pi)\,FPR\), increasing in \(\pi\) for an informative judge. Under
\(H_0:\pi\ge\tau\) it follows that \(q\ge\tau\,Se+(1-\tau)\,FPR\), and with exact lower confidence
bounds on \(Se\) and \(FPR\) from the labelled sample the right-hand side is at least the **critical
threshold** \(c=\tau\,Se_L+(1-\tau)\,FPR_L\). PASS rejects \(H_0\) when the judge-reported upper
bound falls below \(c\); by the union bound over the three exact endpoints the Type-I error is at
most the judge-reported endpoint's alpha plus half the calibration budget. FAIL mirrors the
construction with the upper calibration endpoints. The same box gives the identified true-risk set,
so the decision and the interval agree by construction, and when the calibration cannot establish
an informative judge the outcome is INCONCLUSIVE.

Two things differ from the corrected-interval path. The parameters are prevalence-free, which is
why they require labels drawn as a **simple random sample of the design** (or by true class):
labels drawn within judge strata over-represent the judge's own positives and would bias
sensitivity upward, so the plan validator refuses `noisy_valid_test` under
`simple_random_within_judge_stratum`. And the test needs both true classes among the labels; with
very rare failures that may require many labels, in which case the honest outcome is INCONCLUSIVE.
The sensitivity and false-positive rate are pooled over the cohort's rollouts with equal weight, so
the test certifies only the rows the corrected interval does ([What it certifies](#what-it-certifies)).
Elsewhere the claim is INCONCLUSIVE, with the reason.
The construction follows the structure of Feng et al.'s "Noisy but Valid" framework (a calibration
set for the judge's error rates and a variance-corrected critical threshold on the judge-labelled
set, with finite-sample validity); it is realised here with the package's exact Clopper--Pearson
endpoints and a union bound rather than as a reproduction of that paper's threshold.

Power is the trade-off to understand before choosing this path.
`research/noisy_valid_power.py` simulates both decisions from the *same* random label sample
(sensitivity 0.90, specificity 0.97, 2,000 rollouts, threshold 10%): at 2% true prevalence with
300 labels the noisy-but-valid test passes 28% of the time against 93% for the corrected-interval
decision, and with 100 labels almost never against 22%; both have zero false passes at the
boundary. The reason is label efficiency: a random sample at low prevalence contains very few
truly unsafe items from which to bound sensitivity, whereas labels drawn within judge strata are
enriched for exactly those items. Prefer the corrected interval when labels are scarce; prefer this
test when the judge's error rates must be prevalence-free and transportable, or when sensitivity
can be bounded from seeded known-unsafe items rather than from a random sample.

## Transported mode: a published sensitivity (experimental)

Published human-labelled judge validation sets, such as HarmBench's classifier validation set
or XSTest's annotated completions, measure a judge on outputs that other models produced.
`mode: transported` uses such a set without human labels on the run's own outputs. It is a
**sensitivity analysis**: every interval it emits is `eligibility: diagnostic`,
`role: sensitivity` and `varied_assumption: judge_error`, and it never decides a claim.

```yaml
inputs:
  judge_calibration:
    mode: transported
    published_set:
      name: HarmBench classifier validation set
      source: https://github.com/centerforaisafety/HarmBench
      judge: HarmBench-Llama-2-13b-cls
      rubric: HarmBench behaviour rubric
      human_unsafe: 120          # items human annotators labelled unsafe
      judge_flagged_unsafe: 100  # of those, the items the judge flagged
      sensitivity_alpha: 0.05    # optional, default 0.05
```

`name`, `source`, `judge` and `rubric` are required. They record the provenance the transport
assumption rests on, and the report repeats them. The counts above are placeholders: take them
from the set's own publication for the judge and rubric the run actually used.

### Why \(\rho\) cannot be transported

\(\rho_1\) and \(\rho_0\) are predictive values, so they depend on prevalence. A published set
chosen to be roughly half unsafe gives a judge a \(\rho_1\) its run will not have. For example,
with a sensitivity of 0.974 and a specificity of about 0.84 (figures derived for the HarmBench
classifier in arXiv:2606.25487), \(\rho_1\approx0.83\) on a set that is 45% unsafe. At a true
prevalence of 1% the same judge has \(\rho_1\approx0.06\): 94% of its unsafe verdicts are false,
and it reports a rate of about 17%. Only the prevalence-free parameters, sensitivity \(Se\) and
specificity \(Sp\), can even be assumed to transport.

### Why only the sensitivity is transported

Specificity does not transport well either. The false positives a judge makes depend on what
the outputs look like, and a published set's safe items are often chosen to be hard. In one
open-weights evaluation, a judge flagged 4.9% of HarmBench's human-safe validation items, yet
the run's own judged failure rate had an upper bound of 1.06%. Its false-positive rate on
those outputs was therefore at most about 1.1%, so the published specificity was far too
pessimistic. Plugged into the Rogan--Gladen inversion
\(\pi=(q-(1-Sp))/(Se+Sp-1)\) (Rogan and Gladen, *American Journal of Epidemiology* 107(1),
71--76, 1978), it would have pushed the corrected rate below zero.

So the mode transports the sensitivity alone, as an exact one-sided Clopper--Pearson lower
bound \(Se_L\) at `sensitivity_alpha` from `judge_flagged_unsafe` of `human_unsafe`. The run's
false positives are bounded only by its own judged failures: the judged rate is
\(q=\pi Se+(1-\pi)FPR\), and \((1-\pi)FPR\le q\). Then

- the **lower bound is zero**, because every judged failure may be a false positive; and
- the **upper bound** is \(\pi\le q_U/Se_L\), capped at one, where \(q_U\) is the ordinary
  judge-system upper bound.

The upper bound is the Rogan--Gladen inversion at its maximising corner \(Sp=1\). The inversion
increases with \(Sp\) whenever \(q\le Se\), so declaring any specificity range that reaches one
would give the same upper bound. When \(Se_L=0\), a judge that may catch nothing, the bound is
\([0,1]\). A perfectly sensitive judge leaves the upper bound unchanged. Rollouts the judge
never classified stay completed as failures. The bound is monotone and positively homogeneous
in the prompt rates, so it applies to `weighted_mean`, `maximum`, `cvar` and `top_k_mean`, to
every hierarchy node, but not to `entropic` or the threshold measures. If both bounds hold,
coverage is at least one minus the sum of the row's own alpha and `sensitivity_alpha`.

### What it assumes

- **Transport.** The same judge and rubric, applied to this run's outputs, is at least as
  sensitive on every prompt as on the published set. Nothing in the run checks this, and a
  judge under attack can fail it.
- Human adjudication on the published set is the ground truth.

Each interval lists these assumptions, together with the published set and the sensitivity
bound. `report.json` adds a `judge_corrected_transported` interval on every correctable row,
with `method: transported_judge_sensitivity_correction`. Its provenance records the published
set and `sensitivity_lower`, and the analysis emits the `INFO` diagnostic
`JUDGE_CALIBRATION_TRANSPORTED`. `report-full.md` describes the published set in the judge
misclassification section.

`mode: transported` takes only `published_set`. It cannot be combined with scenarios, a
validation manifest or a judge family, and only fixed-design analyses offer it: a generative
plan that declares it is refused.

## On the generative path

The same correction is available for
[generative evaluations](generative-evaluation.md), where it is reported for the root, every
hierarchy node and every cross-cutting slice. A `validated` calibration certifies only the
root, and only when every unit carries the same weight in it, for the reason in
[What it certifies](#what-it-certifies): the labels pool the cohort's classified units with
equal weight, and \(\rho\) depends on prevalence, so a node or slice where the judge misses
more failures than average would get a corrected upper bound below its true risk. A root with
unequal unit weights, for example under `minimum_rewrites_per_stratum`, has the same problem.
Every other corrected estimate is reported as a sensitivity analysis: `eligibility:
diagnostic`, with a warning that names the reason.

Units with no judge verdict keep their completion through the correction, as on the
fixed-design path: each estimate reports the design weight on them as
`unclassified_share`, and only the classified mass is mapped through \(\rho\).

Two further differences: human labels key on the sample identifier materialization wrote
rather than an observation id, and comparisons are **not** corrected -- doing so would either
discard the paired construction that makes them tight or assume a prevalence-free judge.

## Declaring a judge panel

When several judges score every rollout, declare them under `inputs.judges` and, crucially,
declare which of them are one opinion. Judges built on related models share training data,
prompts and blind spots; a panel that agrees is not a panel that is right. Each judge carries a
dependence `block`, and a composite's **effective votes** is the Kish count over blocks: three
judges in two blocks of sizes two and one carry \(3^2/(2^2+1^2)=1.8\) votes, not three.

Composites (`any`, `all`, `majority`) become score columns and can be the primary outcome. They
are also just judges: the single-judge correction on this page applies to a composite verbatim,
and a `validated` calibration stratified by the *composite's* verdict needs no assumption at all
about dependence among the judges it combines, because it measures the composite's own accuracy
directly. That is the recommended route to evidence for a panel. The panel section of the report
adds pairwise agreement and Cohen's kappa as diagnostics; neither carries a coverage guarantee and
neither identifies the true failure rate on its own.

To see what each judge alone would decide, set `outputs.judge_claims: true` (experimental). The
report's judge panel section then gains each judge's claim rows, decided as the plan would decide
them with that judge as its primary outcome, and the configuration its runs record. The rows bound
each judge's *reported* risk: no correction on this page applies to them, and under a `validated`
calibration every row is `INCONCLUSIVE`. They compare judges and do not enter the decision. See
[Claims by judge](plan-reference.md#claims-by-judge-experimental).

### Deriving a panel's accuracy box

If the primary outcome is an `any` or `all` composite and every member judge declares
prevalence-free accuracy ranges,

```yaml
judges:
  - id: gpt_judge
    score: {...}
    block: openai
    accuracy: {sensitivity: {lower: 0.80, upper: 0.95}, specificity: {lower: 0.90, upper: 0.99}}
```

the report adds a `judge_corrected_panel_derived` sensitivity interval. The composite's
sensitivity and specificity box follows from the members' boxes by interval arithmetic: within a
declared dependence block the judges' errors may be arbitrarily dependent, so the block's joint
behaviour is bounded by Fréchet inequalities; across blocks the judges are independent, so block
probabilities multiply. For `any`, sensitivity is one minus the probability that every judge
clears an unsafe item and specificity is the probability that every judge clears a safe one; `all`
is the mirror image. `majority` has no closed form under partial dependence and is refused --
calibrate the composite's own verdict instead.

True risk is then the Rogan--Gladen inversion \(\pi=(q-(1-Sp))/(Se+Sp-1)\) evaluated over the box;
it is separately monotone in each argument, so the extremes sit at the vertices. **When the box
admits \(Se+Sp\le1\), nothing identifies the true rate and the reported set is \([0,1]\).** That is
the identification result of the partial-identification literature made concrete: agreement among
dependent judges cannot substitute for calibration. Because declared boxes are assumptions, this
interval is `diagnostic` and can never move a claim; the certifying route for a panel remains a
`validated` calibration of the composite's own verdict strata.

A declaration can be scoped with `accuracy.groups` when a judge's ranges are known for some
benchmarks only. The composite's box then holds only in the groups that every member's
declaration covers, and only those groups' rows get the derived interval. If the members
share no group, there is no derived interval, and the warning
`PANEL_CORRECTION_NO_COMMON_GROUP` says so.

### Agreement per group

Agreement over a cohort that holds several benchmarks mixes them. A benchmark on which two
judges' scores are copies of each other agrees perfectly and pulls the cohort's kappa up, even
when the judges disagree on the benchmark that matters. `inputs.panel_groups` adds the same
counts, agreement, kappa and McNemar figures for each listed group's rollouts, per cohort,
beside the cohort-wide ones. They are diagnostics like the cohort-wide figures.

## Stressing a threshold verdict (experimental)

Without a validated calibration, a threshold claim is decided on judge-reported risk. Its
verdict then says nothing about how good the judge must be for the same verdict to hold for
true risk. A PASS whose upper bound sits just below the threshold may need a nearly perfect
judge. The opt-in stress analysis reports that requirement beside each claim:

```yaml
report:
  include_decisions: true
  judge_error_stress: true
```

It is experimental: its fields and wording may change in a minor release. It is diagnostic
only and never changes a claim's status.

**Tipping lines.** Take the prevalence-free model of the
[noisy-but-valid test](#certifying-a-threshold-directly-the-noisy-but-valid-test),
\(q=\pi\,Se+(1-\pi)\,FPR\) with \(FPR=1-Sp\), applied unchanged to every prompt. Like the
correction, it carries over to `weighted_mean`, `maximum`, `cvar` and `top_k_mean`. Write
\([L,U]\) for the judge-reported bounds, \(a\) for `acceptable_upper`, \(b\) for
`unacceptable_lower`, and \(s\) for the share of the row's weight on rollouts the judge never
classified. The bounds complete those rollouts as judge-safe at the lower end and judge-unsafe
at the upper. The stress analysis keeps the same completion on the true-risk scale and inverts
only the classified mass. Then

- the PASS (\(U\le a\)) holds for true risk if and only if \((a-s)\,Se+(1-a)\,FPR\ge U-s\);
- the FAIL (\(L\ge b\)) holds for true risk if and only if \(b\,Se+(1-s-b)\,FPR\le L\).

Each condition is a line in the \((Se,Sp)\) square. The report gives where each line meets the
two perfect edges:

| Field | Meaning |
|---|---|
| `pass_sensitivity_at_perfect_specificity` | With no false alarms, PASS holds iff \(Se\ge (U-s)/(a-s)\) |
| `pass_specificity_at_perfect_sensitivity` | With no misses, PASS holds iff \(Sp\le (1-U)/(1-a)\) |
| `fail_sensitivity_at_perfect_specificity` | With no false alarms, FAIL holds iff \(Se\le L/b\) |
| `fail_specificity_at_perfect_sensitivity` | With no misses, FAIL holds iff \(Sp\ge 1-(L-b)/(1-s-b)\) |

A value above 1 or below 0 means the condition holds along the whole edge or nowhere on it.
`null` means the line never meets that edge. For example, a PASS with \(U=4.98\%\) against
\(a=5\%\) and no unclassified rollouts needs \(Se\ge0.996\) even at perfect specificity. A
false negative rate above 0.4% is enough to put true risk over the threshold.

The two directions differ. Missed failures (low \(Se\)) undermine a PASS, because true risk is
higher than reported. False alarms (low \(Sp\)) undermine a FAIL, because part of the reported
risk is judge noise. A PASS survives any specificity at perfect sensitivity, and a FAIL any
sensitivity at perfect specificity.

**Declared box.** When the primary outcome's judge declares `accuracy` ranges, the record also
gives the true-risk bounds over that box, the verdict they give (`stressed_status`), and
whether it matches the claim's status (`robust`). That judge is either a judge in
`inputs.judges` whose score is the primary score, or an `any` or `all` composite whose members
all declare ranges, with its box [derived as above](#deriving-a-panels-accuracy-box).
`accuracy_source` names it, as `judge:<id>` or `composite:<id>`. Both inversions are separately
monotone in \(Se\) and \(FPR\), so the bounds over the box are attained at its vertices. When the
box admits \(Se+Sp\le1\), or no accuracy in it can produce the observed bounds, true risk is not
identified: the bounds are \([0,1]\), `identified` is false, and the stressed verdict is
INCONCLUSIVE. Without a declared box these fields are `null`, and the lines are still reported.

**Where it does not apply.** Every claim gets a `judge_error_stress` record. Where the analysis
does not apply, the record has `applicable: false` and a `reason`. That is the case for:

- comparison claims, because the judge's accuracy may differ between arms;
- claims without complete bounds;
- claims not decided on their bounds: non-certifying evidence, an exploratory analysis, or
  invalid data integrity;
- `noisy_valid_test` claims, and claims decided on a validated correction, which already
  account for judge error;
- claims that a validated calibration leaves INCONCLUSIVE;
- measures other than `weighted_mean`, `maximum`, `cvar` and `top_k_mean`.

`report-full.md` gains a **Judge-error stress (experimental)** section after the claims, with one row
per claim, the declared box of each source, and the reasons for claims that were not stressed.
When `report.include_decisions` is false, the section only points to `report.json`. With the
switch off, the field is left out of the plan's dump and of every claim, so plans that do not
opt in keep their hash and every artifact byte for byte.

Two limits apply, as for the assumed mode. One \((Se,Sp)\) is applied to every prompt, so a
judge whose accuracy varies with the prompt or with the model under test is not covered. And a
declared box is an assumption, not evidence. The certifying route remains a `validated`
calibration.

## Judge flip rate under content-preserving wrappers (experimental)

The corrections above need the judge's error rates, from human labels or from a declared
assumption. A flip rate needs neither. It measures one way the judge errs. Each judge-positive
completion is rescored by the same judge after it is wrapped in text that leaves the harmful
content verbatim, such as a refusal prefix, benign text before and after, an educational frame
or a story frame. A verdict that becomes safe is a flip. Gao (arXiv:2606.25487) reports flips
on 57-100% of true positives for three LLM judges and on 3.4% for the HarmBench classifier.

The direction matters for a PASS claim. A flip is a false negative, and false negatives
understate risk. A judge that flips often under wrappers is likely to miss unsafe completions
that are framed like this in the run itself. So a PASS certified with that judge rests on its
sensitivity, which a high flip rate puts in doubt.

`inputs.judge_flip` reads the rescored verdicts and reports a `JUDGE_FLIP_RATE` diagnostic: the
share of rescored completions that flip under at least one wrapper, and under each wrapper,
each with an exact Clopper-Pearson interval (see
[the plan reference](plan-reference.md#judge-flip-rate-experimental)). With `sampling: all`
the interval is about the run's judge-positive completions. With `simple_random` it is about
the same population, through the sample. Above the declared `warn_above` the diagnostic is a
warning. The package does not rescore: [`examples/judge_flip`](../examples/judge_flip/README.md)
has an Inspect task that rescores a run's judge-positive completions under four wrappers and
writes the manifest, offline with simulated models or live with the run's judge.

What it assumes:

- **The wrappers preserve content.** A wrapper that weakens the content turns a correct
  "safe" into a flip. Gao's human audit found all 80 flipped responses still harmful, but
  that holds for their wrappers. Check new ones the same way.
- **Judge-positive is not truly unsafe.** The rate is over completions the judge called
  unsafe, and some of those are false positives. A flip of a false positive is a correction.
  With a precise judge the difference is small. With an imprecise one the rate overstates the
  judge's misses.

The flip rate is a screen, not a correction. It never changes a claim, an interval or the
decision. It does not estimate the judge's sensitivity on the run's own outputs, because the
run's false negatives need not look like wrapped positives. To bound risk under judge error,
use a `validated` calibration.

## Getting to evidence

`mode: validated` supplies the machinery; the labels are the remaining cost. An anchor must
have error that can be characterised without reference to the judge. Another LLM judge does not qualify: its errors correlate with
the first judge's on precisely the ambiguous cases, and its own error rates would then need
establishing.

In rough order of value:

1. **Make the outcome verifiable rather than judged.** Sandboxed execution, parsers,
   hash-matching against a reference corpus, tool-call traces. This removes the problem at
   source and leaves the statistics untouched.
2. **Carve the verifiable slice out of each stratum**, and human-label only the remainder.
   If a reliable refusal detector settles 80% of the judge-safe stratum, the human labelling
   budget falls by roughly the same proportion.
3. **Amortise.** Calibration is a property of `(judge, rubric, output distribution)`, not of
   the model under test. Freeze the judge and reuse one calibration across evaluations --
   though this requires the prevalence-free parameterisation, and with it the assumption that
   the judge is equally accurate on every arm.

Note that **targeted stratification does not reduce a conservative bound.** Sampling where
misses are likely sharpens the point estimate, but a bound must still prove the large-weight
stratum is clean, and that costs labels in proportion to the weight it carries.

Constructed cases, self-consistency checks, and cross-judge agreement can all *falsify* a bad
judge cheaply. None can certify a good one.
