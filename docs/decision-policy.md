# Decision policies (experimental)

A claim's thresholds, `acceptable_upper` and `unacceptable_lower`, never enter its bounds. The
analysis computes each bound from the data, the methods, the weights, the groups, the
independence profiles and the error budget of the claim's family, and only then compares it
with the thresholds. A decision policy keeps those thresholds apart from the plan. The plan
freezes everything that shapes the bounds and the family of claims. The policy sets the
thresholds, and it may be written or revised after the run without changing the plan or its
hash. Each claim's verdict keeps its error bound at the policy's thresholds. The overall
decision keeps its own only on the condition described in
[When the overall decision certifies](#when-the-overall-decision-certifies).

Decision policies are experimental. An analysis uses one only when it is given one, and without
one nothing changes: the plan's own thresholds decide, and the report has no `decision_policy`
section.

## Why a later threshold keeps a claim's error bound

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
one that describes the bounds, as far as the stated times can be trusted.

Those times are the policy author's statement. Nothing in the analysis confirms a `set_at`:
unlike a plan, a policy has no registration receipt. A version that states a time before the
first evaluation started therefore gets the `WARNING` diagnostic
`DECISION_POLICY_SELF_DECLARED`, and the overall decision never relies on a stated time. To make
the times checkable, keep the policy file where others can see when each version was written,
such as a version-controlled repository or a registry, and say where in each version's
`reason`.

The `noisy_valid_test` method is the exception: it uses the thresholds inside its test, so its
thresholds stay in the plan, and a policy cannot set them.

## When the overall decision certifies

The overall decision is `PASS` only when every mandatory claim passes. With every
`acceptable_upper` fixed before the data, a wrong overall `PASS` needs one claim whose true risk
is above its threshold to pass, and which claims those are does not depend on the data. The
chance is at most that one claim's error budget, with no correction for the number of claims
(the intersection-union argument in
[statistical methods, section 9.3](statistical-methods.md#93-the-all-mandatory-claims-exception)).

A threshold a policy sets may depend on the data. Set every `acceptable_upper` just above its
own upper bound, and every claim passes. The overall `PASS` is then wrong whenever any one of
those bounds misses its true risk. When the bounds rest on separate coverage events, for example
direct-mean intervals with `simultaneous: false`, or means on the direct-mean family and maxima
on the prompt family, that chance can reach the sum of their error budgets. In a simulation with
ten such claims at `alpha` 0.05, 200 rollouts each and a true risk of 0.10, it was about 0.27.

Moving one threshold is enough, because which claim to move can be chosen from the data too.
Tighten the claim with the lowest upper bound to that bound, leaving the others at the plan's
thresholds: that claim is the likeliest to have missed its true risk. In the same simulation,
where every claim has the same true risk, such an overall `PASS` is wrong exactly when some
bound misses, so the share of wrong ones was the same, about 0.27.

So the overall decision is held back when a policy moves `acceptable_upper` away from the plan's
for any mandatory claim while the mandatory claims' bounds rest on more than one coverage event.
A `PASS` becomes `INCONCLUSIVE`, with the `WARNING` diagnostic
`DECISION_POLICY_OVERALL_UNCERTIFIED`, which lists the claims the policy moves and the coverage
event of every mandatory claim. A `FAIL` stays `FAIL`, and every claim keeps its own status,
whose error bound holds at any threshold. Integrity does not change.

It applies to every version of the policy, whatever its `set_at`: that time is the policy
author's statement, and nothing in the analysis confirms it. It does not apply when:

- the policy moves no mandatory claim's `acceptable_upper`: it restates the plan's values, moves
  only `unacceptable_lower`, which no `PASS` depends on, or sets only optional claims. The
  thresholds an overall `PASS` depends on are then the plan's, fixed in advance;
- one simultaneous family decides every mandatory claim: the default simultaneous prompt
  bounds, simultaneous direct-mean intervals of one independence profile, simultaneous
  sampled-design intervals, or one simultaneous comparison family. One coverage event then
  covers every bound, and a wrong overall `PASS` needs that event to fail, whichever thresholds
  were moved. Pointwise intervals, with `simultaneous: false`, are each an event of their own.

A claim decided on a judge-corrected interval counts as a coverage event of its own, since its
correction also needs its calibration to cover.

So does a `noisy_valid_test` claim, decided by a test rather than on an interval. A policy cannot
set such a claim's thresholds, so they are fixed in advance, and leaving the claim out of the
count would still bound a wrong overall `PASS` by one error budget. It counts all the same, the
conservative reading: the rule then does not rest on that refusal, which a later change to what
a policy may set could lift without anyone revisiting the roll-up. The cost falls only on plans
that mix a mandatory `noisy_valid_test` claim with other mandatory claims under a policy that
moves a threshold.

Adaptive claims (`adaptive_claims`) are not counted. They take no `acceptable_upper`, so a policy
cannot set one, since it names only claims under `claims`, and they are never `PASS`. A mandatory
adaptive claim therefore already keeps the overall decision from `PASS`, and an optional one
does not enter it.

## The inference hash

A policy names the plan it decides for by the plan's inference hash: the plan's
[canonical hash](plan-reference.md#the-plan-hash) with `acceptable_upper` and `unacceptable_lower` of every `corrected_interval` claim left out.
Any other change to the plan changes the inference hash, and an analysis refuses a policy whose
hash does not match.

```bash
safety-stats hash-plan --plan plan.yaml --inference
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
| `versions[].set_at` | yes | When the version was decided, with a timezone, as its author states it |
| `versions[].reason` | no | Why it was set |
| `versions[].thresholds` | yes | For each claim it sets, by claim id, `acceptable_upper` below `unacceptable_lower`, both in [-1, 1] |

Versions apply in order, and their `set_at` times must increase. A claim is decided at the
thresholds of the last version that sets it, or at the plan's when no version does. Keep the
earlier versions in the file when revising it: the report records only the versions the file
holds.

`safety-stats schema decision-policy` prints the JSON schema.

## Running an analysis with a policy

```bash
safety-stats analyze --plan plan.yaml --logs logs/ --decision-policy policy.yaml
safety-stats validate --plan plan.yaml --logs logs/ --decision-policy policy.yaml
```

From Python, pass `decision_policy=` to `analyze` or `validate`, as a path or as a
`DecisionPolicy` from `safety_stats.config.decision_policy`.

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
report has the `INFO` diagnostic `DECISION_POLICY_AFTER_GENERATION`. When one states a time
before it, the report has the `WARNING` diagnostic `DECISION_POLICY_SELF_DECLARED`, since nothing
checks that time. Neither changes integrity. When the policy's thresholds leave the overall
decision uncertified, the report has the `WARNING` diagnostic
`DECISION_POLICY_OVERALL_UNCERTIFIED`, and an overall `PASS` becomes `INCONCLUSIVE` (see
[When the overall decision certifies](#when-the-overall-decision-certifies)). It does not change
integrity either.

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
| `first_evaluation_started_at` | The earliest start time the selected evaluations report, or `null` when none reports one |
| `versions` | Every version, with `set_after_first_evaluation`: `null` when it cannot be told, because no evaluation reports a start time, or because the version is set at or before the earliest reported start while some evaluation reports none |
| `claims` | For every claim, the thresholds it was decided at, `set_by` (`plan` or a version id) and the plan's thresholds |

`report-full.md` shows the same in a "Decision policy" section, and `report.html` in a section
of its own. `report.md` says below its claims table which claims' thresholds the policy set and
which versions state a time after the first evaluation started. With
[verdict ranges](interpretation.md#verdicts-at-other-thresholds), a reader can also see the
verdict at any other thresholds without a policy.
