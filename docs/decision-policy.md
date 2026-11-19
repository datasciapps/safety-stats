# Decision policies (experimental)

A claim's thresholds, `acceptable_upper` and `unacceptable_lower`, never enter its bounds. The
analysis computes each bound from the data, the methods, the weights, the groups, the
independence profiles and the error budget of the claim's family, and only then compares it
with the thresholds. A decision policy keeps those thresholds apart from the plan. The plan
freezes everything that shapes the bounds and the family of claims. The policy sets the
thresholds, and it may be written or revised after the run without changing the plan, its hash,
or the certification of its verdicts.

Decision policies are experimental. An analysis uses one only when it is given one, and without
one nothing changes: the plan's own thresholds decide, and the report has no `decision_policy`
section.

## Why a later threshold keeps the error bound

An interval claim is `PASS` when its upper bound `U` is at or below `acceptable_upper`, and
`FAIL` when its lower bound `L` is at or above `unacceptable_lower`. A wrong `PASS` needs a
true risk `R` above `acceptable_upper` and so above `U`. A wrong `FAIL` needs `R` below
`unacceptable_lower` and so below `L`. The bounds hold, in each direction, with the
probability the claim's family allots, so the chance of a wrong verdict is bounded by the same
error budget at any thresholds, including thresholds chosen after the bounds were seen.

What a later threshold cannot show is that the requirement existed before the evidence. A
threshold set just above a known upper bound passes by construction. The report therefore
records every version of the policy with the time it was set, and says which versions came after
the first evaluation started. A reader can then tell a threshold that was fixed in advance from
one that describes the bounds.

The `noisy_valid_test` method is the exception: it uses the thresholds inside its test, so its
thresholds stay in the plan, and a policy cannot set them.

## The inference hash

A policy names the plan it decides for by the plan's inference hash: the plan's
[canonical hash](plan-reference.md#the-plan-hash) with `acceptable_upper` and `unacceptable_lower` of every `corrected_interval` claim left out.
Any other change to the plan changes the inference hash, and an analysis refuses a policy whose
hash does not match.

```bash
inspect-safety-stats hash-plan --plan plan.yaml --inference
```

`hash-plan` without `--inference` still prints the full hash, which is what
`analysis.plan_sha256` in the report and a plan registration record.

## The policy file

A policy is a YAML or JSON file:

```yaml
schema_version: "1.0"
inference_plan_sha256: 0dddf7e605c115312f8ef40d4359fc6d360afc849ad64115ec50f551aedb2bf7
versions:
  - id: release-gate
    set_at: 2026-11-01T09:00:00Z
    reason: Limits agreed for the release review
    thresholds:
      harm_safe: {acceptable_upper: 0.05, unacceptable_lower: 0.10}
  - id: stricter-misinformation
    set_at: 2026-11-16T14:00:00+01:00
    reason: The review board lowered the limit after the main run
    thresholds:
      harm_safe: {acceptable_upper: 0.03, unacceptable_lower: 0.10}
      overrefusal_safe: {acceptable_upper: 0.10, unacceptable_lower: 0.20}
```

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | no | `"1.0"`, the only version |
| `inference_plan_sha256` | yes | The plan's inference hash, as `hash-plan --inference` prints it |
| `versions` | yes | At least one version, in the order they were set |
| `versions[].id` | yes | A unique name for the version |
| `versions[].set_at` | yes | When the version was decided, with a timezone |
| `versions[].reason` | no | Why it was set |
| `versions[].thresholds` | yes | For each claim it sets, by claim id, `acceptable_upper` below `unacceptable_lower`, both in [-1, 1] |

Versions apply in order, and their `set_at` times must increase. A claim is decided at the
thresholds of the last version that sets it, or at the plan's when no version does. Keep the
earlier versions in the file when revising it: the report records only the versions the file
holds.

`inspect-safety-stats schema decision-policy` prints the JSON schema.

## Running an analysis with a policy

```bash
inspect-safety-stats analyze --plan plan.yaml --logs logs/ --decision-policy policy.yaml
inspect-safety-stats validate --plan plan.yaml --logs logs/ --decision-policy policy.yaml
```

From Python, pass `decision_policy=` to `analyze` or `validate`, as a path or as a
`DecisionPolicy` from `inspect_safety_stats.config.decision_policy`.

A policy that fails a check is not applied. Its error makes integrity `invalid`, so every claim
is `INCONCLUSIVE`, and `analyze` exits `3`:

| Code | Meaning |
|---|---|
| `DECISION_POLICY_MISMATCH` | The policy's `inference_plan_sha256` is not the plan's inference hash |
| `DECISION_POLICY_UNKNOWN_CLAIM` | A version sets a claim the plan does not declare |
| `DECISION_POLICY_FIXED_CLAIM` | A version sets a `noisy_valid_test` claim |

A file that is not a valid policy, for example one whose versions are out of order, is refused
before the analysis starts, as an invalid plan is.

When an applied version was set after the earliest start time of the selected evaluations, the
report has the `INFO` diagnostic `DECISION_POLICY_AFTER_GENERATION`. It does not change
integrity.

## What the report records

The claims carry the thresholds they were decided at, and their status, `verdict_range` and
judge-error stress follow from those. `analysis.plan_sha256` and `plan.resolved.yaml` are the
plan's as frozen. The `decision_policy` section of `report.json` holds:

| Field | Meaning |
|---|---|
| `inference_plan_sha256` | The plan's inference hash |
| `policy_inference_plan_sha256` | The hash the policy names |
| `policy_sha256` | The canonical hash of the policy |
| `applied` | Whether the policy passed its checks and set the thresholds |
| `first_evaluation_started_at` | The earliest start time of the selected evaluations, or `null` |
| `versions` | Every version, with `set_after_first_evaluation` (`null` when no start time is known) |
| `claims` | For every claim, the thresholds it was decided at, `set_by` (`plan` or a version id) and the plan's thresholds |

`report.md` shows the same in a "Decision policy" section. With
[verdict ranges](interpretation.md#verdicts-at-other-thresholds), a reader can also see the
verdict at any other thresholds without a policy.
