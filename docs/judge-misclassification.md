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
- Comparisons are not yet corrected. Under a shared calibration the arms must be corrected
  individually and then differenced; the shortcut of correcting the difference directly is
  only valid under prevalence-free \(\alpha/\beta\) parameters.
- Nothing here detects a judge whose errors depend on which model produced the output.
  Differential misclassification across arms can manufacture an apparent difference from
  nothing, and no declared scenario will reveal it.

## Validated mode: measured accuracies

`mode: validated` estimates the same two parameters from human labels instead of declaring
them. Its corrected intervals are **certifying**.

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
Clopper--Pearson bounds apply. The family alpha is split equally between \(\rho_1\) and
\(\rho_0\), and each parameter's share is split again between its two endpoints, so a union
bound over the four endpoints spends no more than the declared family alpha.

The corrected interval's simultaneous coverage is therefore

\[
1-\left(\alpha_{\text{interval}}+\alpha_{\text{judge}}\right),
\]

the union of the underlying interval's own budget and the calibration budget. With both at
0.05 the corrected rows report 90% simultaneous coverage, and the report states this.

### What it assumes

- The labelled rows are a **simple random sample within their judge stratum**. This is a
  property of the collection protocol, not of the data, and nothing in the package can check
  it. It is recorded in the interval's assumption list.
- **Human adjudication is ground truth.** Labeller error is not modelled.
- One calibration is estimated **per cohort** and shared by every group within it. Since
  \(\rho\) is prevalence-dependent, a cohort is the right granularity; sharing across
  groups inside a cohort is an approximation the report states explicitly.

### When it refuses

The module produces no correction, and says why, when:

- a cohort has an empty judge stratum (`JUDGE_VALIDATION_STRATUM_EMPTY`);
- the labels do not establish a strictly informative judge -- the \(\rho_1\) lower bound
  fails to exceed the \(\rho_0\) upper bound, meaning more labels or a better judge are
  needed (`JUDGE_CALIBRATION_UNINFORMATIVE`); or
- a validation row references an observation absent from the analysed logs, or one whose
  outcome was never classified (`JUDGE_VALIDATION_UNMATCHED`, an error).

### Known gap

Threshold claims are still evaluated against the **uncorrected** judge-system interval. When
a validated correction is present the analysis emits
`JUDGE_CORRECTION_NOT_USED_BY_CLAIMS`, and the corrected bounds must be compared against
thresholds manually. Wiring the decision path to prefer the corrected interval changes
decision semantics and is deliberately left as a separate step.

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
