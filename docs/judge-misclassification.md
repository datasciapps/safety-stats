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

Because the identity is affine in \(R\) with positive slope \(\rho_1-\rho_0\), it
applies unchanged to any risk measure that is positively homogeneous and translation
equivariant: `weighted_mean`, `maximum`, `cvar`, and `top_k_mean`. It does **not** apply to
`entropic` risk, whose \(\lambda\) would need rescaling, or to the threshold measures,
whose thresholds would need transforming. Those measures are left uncorrected.

## Assumed mode: declared accuracies

The first implementation declares \(\rho_1\) and \(\rho_0\) as ranges rather than
estimating them. It produces **no evidence**. Every interval it emits carries
`eligibility: diagnostic` and `role: sensitivity`, and cannot support a claim.

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

Reports gain a **Judge misclassification sensitivity** section listing, for every corrected
row and scenario, the judge-system bounds, the true-risk bounds, and the corrected estimate
range. The original interval is untouched and remains primary.

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
- A single declared scenario is applied to every cohort, so comparisons corrected in
  `assumed` mode inherit that approximation on both arms.
- Nothing here detects a judge whose errors depend on which model produced the output.
  Differential misclassification across arms can manufacture an apparent difference from
  nothing, and no declared scenario will reveal it.

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
stratum's proportion binomial, and the package cannot detect it.

### Estimation and error budget

Within each stratum the labelled rows are an ordinary binomial sample, so exact
Clopper--Pearson bounds apply. The judge family alpha is first divided among the declared
cohorts (see below); each cohort's share is then split equally between \(\rho_1\) and
\(\rho_0\), and each parameter's share again between its two endpoints. A union bound over
those four endpoints spends no more than the cohort's share.

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

This requires a primary interval that exposes both arm bounds -- the two-arm rectangle used
by `weighted_kl_chernoff` and by every `comparison_sets` entry. A direct contrast
construction (`weighted_empirical_bernstein` or `weighted_hoeffding` on the difference)
bounds only \(\Delta\) and carries no arm decomposition, so it cannot be corrected. Those
rows stay on the judge-reported scale and the analysis emits
`JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON`. Since the recommended primary for sparse
binary failures is already `weighted_kl_chernoff`, the recommended path is covered.

### Error budget across cohorts

The judge family alpha is divided equally among the **declared cohorts**. A single risk row
consumes one share; a comparison consumes the two shares of the cohorts it spans. A union
over every corrected row therefore stays inside the declared family budget, and each interval
reports the share it actually spent in `family_alpha`.

### What it assumes

- The labelled rows are a **simple random sample within their judge stratum**. This is a
  property of the collection protocol, not of the data, and nothing in the package can check
  it. It is recorded in the interval's assumption list.
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
weights over an unbalanced hierarchy usually do not.

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

### When it refuses

The module produces no correction, and says why, when:

- a cohort has an empty judge stratum (`JUDGE_VALIDATION_STRATUM_EMPTY`);
- the labels do not establish a strictly informative judge -- the \(\rho_1\) lower bound
  fails to exceed the \(\rho_0\) upper bound, meaning more labels or a better judge are
  needed (`JUDGE_CALIBRATION_UNINFORMATIVE`); or
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

## On the generative path

The same correction is available for
[generative evaluations](generative-evaluation.md), where it applies to the root, every
hierarchy node and every cross-cutting slice. Two differences: human labels key on the
sample identifier materialization wrote rather than an observation id, and comparisons are
**not** corrected -- doing so would either discard the paired construction that makes them
tight or assume a prevalence-free judge.

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
