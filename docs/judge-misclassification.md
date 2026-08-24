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

## Assumed mode

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

## Getting to evidence

Replacing assumption with measurement requires an anchor whose error can be characterised
without reference to the judge. Another LLM judge does not qualify: its errors correlate with
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
