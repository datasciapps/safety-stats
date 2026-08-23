# One-sided statistical certification

The certification module evaluates specified claims of the form

\[
R \le \tau,
\]

where \(R\) is a failure probability or fixed-design weighted mean and \(\tau\) is the evaluator's
safety threshold. It is a separate second-stage analysis: the empirical report remains
threshold-neutral, while the certification plan records thresholds and one-sided error budgets.

Mean claims reference named independence profiles already frozen in the empirical analysis plan.
Certification reads `observations.csv` and rebuilds the selected blocks at its one-sided local
alpha, so arbitrary observation or manifest block keys are supported. Exact single-prompt claims
do not use an independence profile.

## Directional evidence

Support tests \(H_0:R\ge\tau\) against \(R<\tau\). A claim is supported when its one-sided upper
bound satisfies \(U\le\tau\). Rejection tests \(H_0:R\le\tau\) against \(R>\tau\), and rejects the
claim when its one-sided lower bound satisfies \(L>\tau\). Otherwise the result is inconclusive.

Support and rejection have separate error families. A support alpha of 0.05 controls false
support at 5% under the unsafe null. A rejection alpha of 0.05 separately controls false rejection
under the safe null. These statements do not imply a 95% two-sided interval.

For one fixed prompt with \(X\sim\operatorname{Binomial}(n,p)\), the exact p-values at threshold
\(\tau\) are

\[
p_S=\Pr_{\tau}(X\le x),\qquad p_R=\Pr_{\tau}(X\ge x).
\]

They are dual to one-sided Clopper–Pearson bounds. Unknown attempted outcomes are counted as
failures for support and as safe for rejection, so missing classification cannot help either
direction.

Group and hierarchy means use weighted KL/Chernoff, empirical-Bernstein, or weighted-Hoeffding
constructions. Their p-values are conservative inversions of the corresponding nested one-sided
confidence bounds. A claim names one analysis-plan independence profile; observations sharing its
`block_by` key are treated as one bounded block, permitting arbitrary dependence inside the block.

## Plan

```yaml
schema_version: "2.0"

certification:
  id: candidate-certification
  title: Candidate fixed-design statistical certification

source:
  require_plan_sha256: replace-with-empirical-plan-hash

claims:
  - id: overall-risk
    title: Fixed-design mean failure risk is at most 5%
    target:
      type: group_mean
      cohort: candidate
      group: all
    threshold: 0.05
    support:
      family: support-family
      method: weighted_kl_chernoff
      independence_profile: rollout
    rejection:
      enabled: true
      family: rejection-family
      method: weighted_kl_chernoff
      independence_profile: rollout
    sensitivity:
      - method: weighted_hoeffding
        independence_profile: rollout
      - method: weighted_empirical_bernstein
        independence_profile: rollout
      - method: weighted_kl_chernoff
        independence_profile: prompt

multiplicity:
  families:
    - id: support-family
      purpose: simultaneous_support
      method: bonferroni
      alpha: 0.05
    - id: rejection-family
      purpose: simultaneous_rejection
      method: bonferroni
      alpha: 0.05

report:
  decimal_places: 6
  detectable_power: 0.90
```

Targets are:

- `prompt_failure_rate`, requiring `cohort` and `prompt_id` and using `clopper_pearson`;
- `group_mean`, requiring `cohort` and `group`;
- `hierarchy_mean`, requiring `cohort`, `group`, `level`, and either an exact `path` or
  `all_paths: true`.

For a hierarchy target, the empirical group's original design weights are restricted to the node
and renormalized. `all_paths: true` expands into one claim per node at that level before
multiplicity allocation. `allocation_weight` on a direction is interpreted per expanded claim.

`bonferroni` gives equal local alpha. `weighted_bonferroni` uses the positive
`allocation_weight` values declared on each direction. Support and rejection references must point
to families with the corresponding purpose.

## Running certification

```bash
inspect-safety-stats certify \
  --report reports/candidate \
  --plan certification-plan.yaml \
  --output reports/candidate/certification
```

The output directory contains:

```text
certification-report.json
certification-report.md
certification-report.html
certification-claims.csv
certification-plan.resolved.yaml
checksums.json
```

The default successful command exit code is zero regardless of evidence direction. Opt into
`--evidence-exit-codes` for `0` when every claim is supported, `1` when any claim is rejected, and
`2` when none is rejected but at least one is inconclusive.

## Numerical validity and provenance

The module reports only numerical evidence and does not judge when or why the analyst selected a
threshold. `SUPPORTED`, `REJECTED`, and `INCONCLUSIVE` are determined directly from the configured
one-sided bounds, p-values, and multiplicity allocation.

An empirical report with invalid integrity is refused because its counts or design cannot support
the requested calculation. The module checks `report.json` and `design.resolved.csv` against the
empirical checksum manifest when present. `source.require_plan_sha256` is an optional
reproducibility check that ensures the intended empirical design was supplied. The module never
reads or displays prompt or response content.

## Prospective planning

Plan an exact one-sided prompt certification study with:

```bash
inspect-safety-stats plan certification \
  --threshold 0.01 \
  --anticipated-rate 0.002 \
  --alpha 0.05 \
  --power 0.90
```

This reports the minimum rollout count, the largest observed failure count that would still
support certification, exact power under the anticipated true failure rate, the maximum true
failure rate certifiable at the requested power, and the minimum unsafe failure rate detectable
at that power. Use the multiplicity-adjusted local alpha when planning a claim that belongs to a
family.

For a fixed rollout horizon, invert both one-sided tests directly:

```bash
inspect-safety-stats plan detectable-rate \
  --trials 500 \
  --threshold 0.05 \
  --alpha 0.05 \
  --power 0.90
```

The maximum certifiable rate is the largest \(q<\tau\) for which the support test has the requested
power. The minimum detectable rate is the smallest \(q>\tau\) for which the rejection test has the
requested power. The command also reports their distances from the threshold. These are exact
binomial calculations for an independent homogeneous Bernoulli stream.

Certification reports also include detectability at the evaluated horizon when
`report.detectable_power` is set. Individual-prompt claims use exact binomial inversion. Weighted
group and hierarchy claims use a conservative fixed-design Hoeffding guarantee based on the actual
weights and rollout counts. For concentration geometry \(S\), the reported support and rejection
gaps are

\[
\sqrt{\frac{S}{2}\log\frac{1}{\alpha}}
+
\sqrt{\frac{S}{2}\log\frac{1}{1-\text{power}}}.
\]

For a general configured profile, \(S=\sum_b a_b^2\), where \(a_b\) is the total fixed-design
weight carried by independent block \(b\). This reduces to \(\sum_j w_j^2/n_j\) for rollout blocks
and \(\sum_j w_j^2\) for prompt blocks. Repeated rollouts therefore add precision only when the
chosen profile declares them independent. Set `detectable_power: null` to omit these diagnostics.

## Fixed-design rollout planning

Plan an existing manifest and hierarchy before running the evaluation:

```bash
inspect-safety-stats plan fixed-design \
  --analysis-plan safety-plan.yaml \
  --certification-plan certification-plan.yaml \
  --anticipated-rate 0.02 \
  --power 0.90 \
  --output rollout-plan
```

The planner expands hierarchy claims, applies their multiplicity-adjusted local alpha values, and
uses the actual fixed-design weights. For a support claim with threshold \(\tau\), anticipated rate
\(q<\tau\), and power \(1-\beta\), it requires

\[
\sum_j \frac{w_j^2}{n_j}
\le
\frac{2(\tau-q)^2}
{\left(\sqrt{\log(1/\alpha)}+\sqrt{\log(1/\beta)}\right)^2}.
\]

The same expression applies to rejection planning with \(q>\tau\). Individual-prompt claims use
exact binomial power calculations. The planner reports both the minimum common rollout count and a
weight-adaptive integer allocation that is reduced while preserving every claim constraint.

Use `--maximum-rollouts` to test a per-prompt budget. A report can be numerically infeasible under
that budget; the required uncapped allocation is still emitted so the deficit is visible. The
planner assumes independent rollouts and permits heterogeneous prompt probabilities.

The output directory contains:

```text
planning-report.json
planning-report.html
planning-report.md
planning-claims.csv
rollout-allocation.csv
analysis-plan.resolved.yaml
certification-plan.resolved.yaml
checksums.json
```

## Interpretation limitations

- Evidence applies to the declared fixed design and rollout process, not an unseen prompt
  population.
- Adjusted p-values are not posterior probabilities that a claim is true.
- Each sensitivity method has its own numerical guarantee. Selecting among several methods changes
  the procedure whose error rate must be interpreted.
- More repeated rollouts tighten rollout-independent evidence but do not increase the number of
  independent prompt clusters.
