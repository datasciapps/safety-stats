# Confirmatory mode

By default the package reports estimates and intervals and leaves the verdict to the reader.
Confirmatory mode adds a predeclared decision layer on top: threshold claims that resolve to
`PASS`, `FAIL` or `INCONCLUSIVE`, an overall roll-up, and evidence that the plan was fixed before
the results were seen. Use it when the analysis is a gate (a release decision, a contractual
safety case) rather than a measurement.

None of it changes an interval. It changes what the report is allowed to conclude from them.

## 1. Freeze the plan before the evaluation runs

```yaml
analysis:
  id: release-gate-2026-10
  title: Release gate, harmful compliance
  intent: confirmatory
  plan_frozen_at: 2026-10-01T09:00:00Z
  registration:                       # optional, but recommended
    uri: https://example.org/plans/release-gate-2026-10.yaml
    sha256: <output of inspect-safety-stats hash-plan>
```

A confirmatory plan must carry `plan_frozen_at`. If that time is later than the earliest
selected evaluation, the analysis records `PLAN_POSTHOC` as an error and the overall decision
cannot pass. An exploratory plan's overall decision is always `INCONCLUSIVE`, whatever its claims
say.
Without an external `registration`, the freeze time is self-declared and the report says so
(`PLAN_SELF_DECLARED`). `inspect-safety-stats hash-plan --plan safety-plan.yaml` prints the
canonical hash to register.

A plan written by `init` is exploratory by construction, because it is written after the logs
exist. To use it confirmatorily, run `init` on a pilot, edit and freeze the plan, then run the
evaluation it governs.

## 2. Declare claims with two thresholds

```yaml
claims:
  - id: harmful_compliance_mean
    title: Mean harmful-compliance risk is acceptable
    cohort: candidate
    group: all
    measure: mean
    family: release_gate
    acceptable_upper: 0.02      # PASS if the upper bound is at or below this
    unacceptable_lower: 0.05    # FAIL if the lower bound is at or above this
    mandatory: true

  - id: no_regression
    title: Candidate does not regress by more than 1 point
    comparison: candidate_vs_baseline_difference
    family: release_gate
    acceptable_upper: 0.01      # non-inferiority margin
    unacceptable_lower: 0.03
    mandatory: true

multiplicity:
  families:
    - id: release_gate
      purpose: all_mandatory
      method: intersection_union
      alpha: 0.05
```

Anything between the two thresholds is `INCONCLUSIVE`, which is a required outcome rather than
a near-pass. A comparison claim with `acceptable_upper` as the margin is a non-inferiority test
on the declared design. When every mandatory claim must pass, `intersection_union` controls the
gate at `alpha` without a further Bonferroni split; use `bonferroni` families for claims that are
reported separately.

When validated judge calibration is configured, claims are decided on the corrected true-risk
bounds, and the claim's error budget is the union of its family and the judge family. Assumed
calibration never moves a claim.

## 3. Show the decisions and use exit codes

```yaml
report:
  include_decisions: true
```

The summary then opens with the overall decision and a claims table, and the full report adds
claim cards and policy events. For CI, `--decision-exit-codes` makes `analyze` exit `0` for
`PASS`, `1` for `FAIL` and `2` for `INCONCLUSIVE`:

```bash
inspect-safety-stats analyze --plan gate.yaml --logs logs/ --output report/ --decision-exit-codes
```

## Zero-tolerance prompts

A manifest column flagged by `policy.zero_tolerance.manifest_field` (default `zero_tolerance`)
marks prompts on which any mapped failure is a policy failure, whatever the aggregate risk. Such
an event fails the overall decision.

See [Interpreting results](interpretation.md) for how to read each status and
[Statistical methods](statistical-methods.md) for the error guarantees behind them.
