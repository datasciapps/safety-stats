# Analysis plan reference

Plans are strict YAML or JSON documents. Unknown fields are errors, and integer fields reject `true` and `false` rather than reading them as 1 and 0. Relative manifest paths resolve against the plan file.

## Schema versions

`schema_version` is `"2.0"` (the default when it is left out) or `"3.0"`, and an unquoted `2.0`
or `3.0` reads the same. A plan that declares `1.0` is refused with what changed in 2.0 (see
[Configuring hierarchy and independence](independence-profiles.md)). The two differ only in
how a plan names the fields of runs and rollouts, in `design.rollout_key`, `cohorts.*.match` and
`cohorts.*.require_uniform`:

| | Schema 2.0 | Schema 3.0 |
|---|---|---|
| Field names | The source's own: for Inspect logs, the columns of `evals_df` and `samples_df` | A field every source has, or a source field written `attributes.<name>` |
| Run fields | `model`, `task_version`, `task_arg_*`, ... | `run_key`, `model`, `status`, `run_id`, `attributes.<name>` |
| Rollout fields | `eval_id`, `id`, `epoch`, ... | `run_key`, `item_id`, `repeat`, `rollout_id`, `attributes.<name>` |
| `design.rollout_key` default | `[eval_id, id, epoch]` | `[run_key, item_id, repeat]` |
| `inputs.sample_join` | Allowed, with its only values | Removed |

Schema 3.0 works the same with every source, Inspect logs or [rollout files](rollout-files.md),
and is the version to use for new plans. A field name that is neither neutral nor written
`attributes.<name>` is an error. `attributes.<name>` names the source's field verbatim: for an
Inspect run, `attributes.task_version` is the `task_version` column. For an Inspect log, a run's
`run_key` is its `eval_id`, and a rollout's `item_id`, `repeat` and `rollout_id` are its sample
`id`, `epoch` and `sample_id`.

To move a 2.0 plan to 3.0, set `schema_version: "3.0"`, delete `inputs.sample_join`, and rename
fields: `eval_id` to `run_key`, `id` to `item_id`, `epoch` to `repeat`, `sample_id` to
`rollout_id`, and any other source field `f` to `attributes.f` (`model` keeps its name). The
plan's hash changes with its text. A 2.0 plan keeps its meaning and its hash.

## `analysis`

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Stable analysis identifier |
| `title` | yes | Report title |
| `description` | no | Human context |
| `intent` | yes | `confirmatory` or `exploratory` |
| `plan_frozen_at` | confirmatory | Timestamp preceding selected evaluations |
| `registration` | no | External URI and SHA-256 of a preregistered plan |

Exploratory analyses cannot produce `PASS`. Their claims are `INCONCLUSIVE`, and each records
as `descriptive_status` what a confirmatory analysis of the same bounds would give. See
[Exploratory analyses](interpretation.md#exploratory-analyses).

## `inputs`

`design_manifest` points to the fixed-design CSV. Each rollout joins the manifest's `prompt_id`
by its item: the Inspect sample `id`, or the `item_id` of rollout files. Schema 2.0 plans may
state this as `sample_join: {inspect_field: id, manifest_field: prompt_id}`, its only values;
schema 3.0 removes the field.

`estimand` is `end_to_end_system` by default. In this mode, an explicit provider policy block is
safe only if it occurred during target generation and no content was delivered. Set
`underlying_model` to keep such blocks unclassified because the wrapper prevented observation of
the underlying model behavior.

`score.name` selects an Inspect score. Mapping is categorical:

```yaml
mapping:
  failure_values: ["FAIL"]
  safe_values: ["PASS"]
```

or numeric:

```yaml
mapping:
  failure_if:
    operator: greater_than_or_equal
    value: 0.5
```

Mappings are type-sensitive: `true`, `1`, `1.0`, and `"1"` are distinct.

`missing` accepts `conservative_bounds`, `block`, `count_as_failure`, or `exclude`.
`conservative_bounds` is the default: integrity becomes `degraded`, and exact bounds cover both
the all-safe and all-failure completions of every attempted but unclassified rollout.

`errors` additionally accepts `end_to_end`, which is the default. It recognizes only explicit
target-side provider policy blocks with no delivered content as safe for the end-to-end estimand.
Post-generation, grader, network, quota, timeout, and unrecognized errors remain unclassified and
use conservative bounds. `block` retains the original strict behavior and makes integrity invalid.

```yaml
inputs:
  design_manifest: design.csv
  estimand: end_to_end_system
  score:
    name: harmful_behavior
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
    missing: conservative_bounds
    errors: end_to_end
```

`judge_calibration` is optional. It declares assumed conditional judge accuracies and emits
diagnostic true-risk sensitivity intervals beside the ordinary judge-system intervals:

```yaml
inputs:
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

`rho_1` is P(truly unsafe | judge said unsafe) and `rho_0` is P(truly unsafe | judge said
safe). Every scenario must satisfy `rho_1.lower > rho_0.upper`. In `assumed` mode the
resulting intervals are always `diagnostic`: a declaration carries no coverage guarantee.

`mode: validated` estimates the same two parameters from human labels instead, and its
intervals are `certifying`:

```yaml
inputs:
  judge_calibration:
    mode: validated
    validation_manifest: judge-labels.csv
    family: judge_error
    sampling: simple_random_within_judge_stratum
```

The manifest holds `observation_id,human_label` with labels `safe` or `unsafe`; identifiers
come from the `observations.csv` of a prior run. `sampling` is
`simple_random_within_judge_stratum` (the default) or `simple_random`; only the latter also
estimates sensitivity and the false-positive rate by true class, which a claim with
`method: noisy_valid_test` requires (the validator emits `PLAN_INVALID_CLAIM` otherwise). Threshold claims whose evidence carries a
validated correction are decided from the corrected bounds; an `assumed` correction never
affects a claim. One calibration is estimated per cohort by
exact Clopper-Pearson bounds within each judge stratum, and the corrected interval's coverage
is the union of its own family budget and the judge family budget. The two modes are mutually
exclusive.

Corrections are applied to `weighted_mean`, `maximum`, `cvar`, and `top_k_mean` measures, to
every hierarchy node, and to comparisons whose primary interval exposes both arm bounds
(`weighted_kl_chernoff` and any `comparison_sets` entry). A validated correction certifies
only a `weighted_mean` over a whole cohort, in a fixed design, that weights every classified
rollout equally. Every other corrected row is a sensitivity, and a claim on it is
INCONCLUSIVE. See [Judge misclassification](judge-misclassification.md).

### Judge panels

`inputs.judges` declares additional binary judges scored on every rollout, each with its own
Inspect score column and outcome mapping. `block` names the judge's dependence block: judges
sharing a block are one opinion, not independent votes (two judges built on related models
belong in one block). It defaults to the judge's own id, so a panel is independent only where the
plan says so. `kind: monitor` marks a chain-of-thought or trajectory monitor.

```yaml
inputs:
  score:
    name: any_judge                       # a composite may serve as the primary outcome
    mapping: {failure_values: [true], safe_values: [false]}
  judges:
    - id: gpt_judge
      score: {name: harmful_gpt, mapping: {failure_values: [FAIL], safe_values: [PASS]}}
      block: openai
    - id: claude_judge
      score: {name: harmful_claude, mapping: {failure_values: [FAIL], safe_values: [PASS]}}
    - id: cot_monitor
      score: {name: cot_flag, mapping: {failure_values: [FLAG], safe_values: [CLEAR]}}
      block: openai
      kind: monitor
  composites:
    - {id: any_judge, rule: any, judges: [gpt_judge, claude_judge, cot_monitor]}
    - {id: majority, rule: majority, judges: [gpt_judge, claude_judge, cot_monitor]}
```

`composites` derive a verdict from declared judges under `any`, `all`, or `majority` and are
materialised as score columns (`score_<id>`, boolean), so any downstream construction can use
them. A composite is undecided -- and flows through the ordinary unclassified-outcome handling --
when a missing vote could still change it or when `majority` ties. The report gains a **Judge
panel** section with per-judge counts, each composite's Kish **effective votes** over the declared
blocks, and pairwise agreement with Cohen's kappa. Agreement is a diagnostic, not evidence: judges
that share a blind spot agree and are both wrong. Missing judge columns and unmappable judge
values are errors (`JUDGE_SCORE_MISSING`, `JUDGE_SCORE_UNMAPPED`).

A judge may declare prevalence-free accuracy ranges, `accuracy: {sensitivity: {lower, upper},
specificity: {lower, upper}}`. When the primary outcome is an `any` or `all` composite whose judges
all declare them, the report adds a diagnostic `judge_corrected_panel_derived` interval: the
composite's box is derived by Fréchet bounds within blocks and products across blocks, and true
risk is partially identified by inverting Rogan–Gladen over it (the whole unit interval when the
box admits an uninformative panel). See [Judge misclassification](judge-misclassification.md).

## `design`

| Field | Meaning |
|---|---|
| `type` | `fixed` (census of the declared design) or `sampled` (items drawn from a frame) |
| `inclusion_probability_column` | Required with `type: sampled`; names a manifest column holding each frame item's inclusion probability in (0, 1] |
| `sampling` | `poisson` (independent inclusion, default) or `fixed_size` (exact draws per stratum, without replacement) |
| `stratum_column` | Optional with `sampling: fixed_size`; names the manifest column defining sampling strata |
| `hierarchy` | Creator-declared hierarchy ending in `prompt_id` |
| `rollout_key` | Fields uniquely identifying one terminal rollout; see [Schema versions](#schema-versions) for the names and default |
| `require_complete_manifest` | Require every fixed prompt in every cohort |
| `require_expected_rollouts` | Require prospective rollout counts, from the manifest's `expected_rollouts` column (on by default) |
| `weights` | Default group weight strategy |
| `independence_profiles` | Named independent-block partitions; see [Configuring hierarchy and independence](independence-profiles.md) |

Example:

With `type: sampled` the manifest is the sampling frame rather than a census, and only
`weighted_mean` measures are available. See
[Design-based inference for sampled item frames](design-based-sampling.md).

```yaml
design:
  hierarchy: [task_id, question_id, prompt_id]
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
    prompt: {block_by: [prompt_id], comparison_coupling: joint}
```

### Prompt variants in fixed designs

A fixed design can ask one base prompt in several wordings, such as the original, a
paraphrase or a role-play wrapper. Model this with the hierarchy `[task_id, question_id,
prompt_id]`: the base prompt is the `question_id`, and each wording is a `prompt_id` beneath it.
Each wording is its own manifest row and its own Inspect sample. Its `id` matches the row's
`prompt_id`.

```text
prompt_id,question_id,task_id,variant,expected_rollouts
cyber-credentials-original,cyber-credentials,cyber-safety,original,20
cyber-credentials-paraphrase,cyber-credentials,cyber-safety,paraphrase,20
cyber-credentials-roleplay,cyber-credentials,cyber-safety,roleplay,20
cyber-auth-bypass-original,cyber-auth-bypass,cyber-safety,original,20
```

The `variant` column only describes the wording, and the analysis does not read it unless the
plan names it. A group can filter on it, for example with `filter: {variant: original}` to
cover only the original wordings.

**Weights.** With `nested_equal` weights over the same hierarchy, each base prompt of a task
gets the same weight whatever its number of wordings, and its wordings share that weight
equally. In the rows above, the two base prompts weigh 1/2 each within the task. Each
credential-theft wording weighs 1/6 and the authentication-bypass prompt weighs 1/2.
`equal_prompt` would instead give each row 1/4, so the base prompt with more wordings would
count three times as much.

**Independence.** Wordings of one base prompt can fail together, so a profile that blocks by
base prompt is a natural sensitivity:

```yaml
design:
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
    prompt: {block_by: [prompt_id], comparison_coupling: joint}
    question: {block_by: [question_id], comparison_coupling: joint}
outputs:
  direct_mean_intervals:
    independence:
      primary: rollout
      sensitivities: [prompt, question]
```

The `question` profile puts every rollout of every wording of one base prompt in one block, so
its block count is the number of base prompts. Its effective sample size,
$\left(\sum_g A_g^2\right)^{-1}$ over the base-prompt weights $A_g$ (see
[Statistical construction](independence-profiles.md#statistical-construction)), is no larger
than the `prompt` profile's. With few base prompts its intervals can reach the whole range
$[0, 1]$. Set `hierarchy_nodes: true` under `direct_mean_intervals` to report an interval for
each base prompt as well as each task.

[`examples/fixed_design`](../examples/fixed_design/README.md#prompt-variants-of-one-base-prompt)
has a complete template: `variant-design.csv`, `variant-plan.yaml` and the Inspect task
`variant_eval.py`.

## `cohorts`

Cohorts use exact evaluation-level field matches:

```yaml
cohorts:
  candidate:
    match:
      model: openai/example-candidate
    require_uniform:
      - task_version
      - model_generate_config
```

In schema 3.0 the same cohort reads:

```yaml
cohorts:
  candidate:
    match:
      model: openai/example-candidate
    require_uniform:
      - attributes.task_version
      - attributes.model_generate_config
```

Every selected evaluation must match at most one cohort.

## `groups`

Groups use exact manifest filters and one weight strategy:

- `equal_prompt`;
- `nested_equal`;
- `explicit`, with a manifest `column`.

```yaml
groups:
  - id: cyber
    filter: {risk_domain: cyber}
    weights:
      strategy: nested_equal
      hierarchy: [task_id, question_id, prompt_id]
```

## `outputs`

`prompt_bounds` declares the simultaneous-bound family and alpha allocation. Prompt bounds are on by default and take their alpha from `family`, simultaneous or not, and `family` has no default: name a declared multiplicity family, or set `enabled: false`. `risk_measures` contains named measures. A plan without one reports no group risk, and the analysis warns (`PLAN_NO_RISK_MEASURES`).

```yaml
outputs:
  prompt_bounds:
    enabled: true
    simultaneous: true
    family: displayed
    coverage_mode: two_sided
    alpha_allocation: equal_bonferroni
  risk_measures:
    - {id: mean, type: weighted_mean}
    - {id: tail, type: cvar, beta: 0.90}
    - {id: entropic, type: entropic, lambda: 10}
    - {id: worst, type: maximum}
    - {id: worst_of_4, type: worst_of_k, k: 4}
```

`worst_of_k` reports the weighted probability that at least one of `k` independent attempts at
each fixed prompt fails, `sum_j w_j (1 - (1 - p_j)^k)`. It is the repeated-attempt analogue of
the weighted mean (`k: 1` is exactly the mean) and targets deployment or attack settings where an
item is retried `k` times. `k` is part of the estimand and must be chosen before results are
inspected. The per-item transform is coordinatewise monotone, so the measure inherits the
propagated simultaneous prompt envelope. Judge-misclassification correction does not apply to it.
The point estimate is the unbiased U-statistic `sum_j w_j (1 - C(n_j - f_j, k) / C(n_j, k))`,
from `f_j` failures in `n_j` classified rollouts. Plugging the failure rate into the transform
would understate the risk, because the transform is concave. With fewer than `k` classified
rollouts on some prompt, no estimate is reported, but the bounds still are.

`coverage_mode: two_sided` is the default. It interprets the multiplicity-family alpha as the
total error budget for the displayed lower-to-upper range and divides it equally between tails.
Thus family alpha `0.05` produces a simultaneous two-sided coverage floor of 95%, with 97.5%
coverage in each direction. `coverage_mode: per_direction` instead spends the full family alpha
on each direction; at alpha `0.05`, each direction has 95% coverage but the combined range has
only a 90% floor.

`hierarchy_region: linear_program` keeps the simultaneous floor but not the per-direction level.
A projected endpoint depends on the other direction's raw endpoints too, so projected intervals
report the simultaneous floor in each direction (see
[hierarchy-simultaneous-inference.md](hierarchy-simultaneous-inference.md)).

For `explicit` allocation, provide one prospective total alpha budget per manifest prompt:

```yaml
prompt_bounds:
  enabled: true
  simultaneous: true
  family: displayed
  coverage_mode: two_sided
  alpha_allocation: explicit
  explicit_alpha:
    prompt-001: 0.01
    prompt-002: 0.04
```

The entries must cover the manifest exactly and sum to no more than the family alpha. In
`two_sided` mode each entry is split equally between tails. When there are multiple cohorts, each
tail budget is also divided equally across its cohort-specific endpoints, so the declared family
budget still covers every reported prompt range.

### Direct fixed-design mean intervals

`direct_mean_intervals` adds direct concentration intervals for selected `weighted_mean` outputs
and, optionally, every non-prompt node in the declared hierarchy:

```yaml
outputs:
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    reference_methods: [weighted_empirical_bernstein, weighted_hoeffding]
    family: displayed_direct_means
    groups: [all, cyber]
    simultaneous: true
    coverage_mode: two_sided
    independence:
      primary: rollout
      sensitivities: [prompt]
    hierarchy_nodes: true
    hierarchy_calibration: frontier_sidak
    hierarchy_region: linear_program
```

If `groups` is omitted, the method applies to every declared group. With `simultaneous: true`,
the primary intervals are simultaneous across all selected fixed-design estimands. With
`simultaneous: false`, each interval is pointwise.

`hierarchy_calibration` controls how the directional family error budget is spent:

- `flat_bonferroni` (default) divides alpha over every group root, hierarchy node, and cohort;
- `frontier_sidak` divides alpha by Bonferroni over potentially dependent hierarchy frontiers,
  then uses exact Šidák product calibration among pairwise-disjoint nodes within each frontier.

`frontier_sidak` requires `hierarchy_nodes: true`. It assumes independence across blocks in the
selected named profile. It does not assume that failure probabilities are identical. The compiler
checks actual block overlap: siblings receive Šidák calibration only when their block supports are
disjoint. Ancestors, descendants, levels, cohorts, and overlapping groups are conservatively
treated as dependent.
Repeated identical targets—for example a root and a hierarchy level containing one node with the
same weights—are aliases and do not spend alpha twice.

`weighted_kl_chernoff` is a finite-sample Chernoff inversion optimized for sparse binary failures.
It uses the Bernoulli-KL exponent when atomic weights are equal and a rigorous weighted
Poisson-Chernoff fallback when they are not. `weighted_empirical_bernstein` is variance-adaptive;
`weighted_hoeffding` uses only boundedness. `reference_methods` reports prespecified alternatives
without replacing the primary method. Each method's coverage guarantee is construction-specific;
the report is not permission to choose the shortest interval after seeing the results.

Each direct construction assumes independence across the blocks in its selected profile and
allows arbitrary dependence inside a block. Unclassified attempted outcomes are completed as safe
for the lower endpoint and as failures for the upper endpoint.

With `hierarchy_nodes: true`, node weights are inherited from the enclosing declared group and
renormalized within the node. `hierarchy_region: linear_program` (the default) treats prompt risks
as shared primitive parameters, places every raw direct group and node interval into one linear
confidence region, and reports the coordinate projection of that common region for each node.
Use `hierarchy_region: none` to retain the unprojected intervals. The `independence` selector names
one primary profile and optional prespecified sensitivity profiles. See
[Configuring hierarchy and independence](independence-profiles.md).

## `comparison_sets`

Use a comparison set for two or more cohorts:

```yaml
comparison_sets:
  - id: models_vs_reference
    cohorts: [reference, candidate_a, candidate_b]
    contrasts:
      type: versus_reference
      reference: reference
    estimands: [risk_difference, risk_ratio]
    group: all
    measure: mean
    method: weighted_kl_chernoff
    reference_methods: [weighted_empirical_bernstein, weighted_hoeffding]
    family: model_comparisons
    coverage_mode: two_sided
    independence:
      primary: rollout
      sensitivities: [prompt]
    hierarchy_nodes: true
    hierarchy_calibration: frontier_sidak
    hierarchy_region: linear_program
```

At least two unique cohort identifiers are required. A two-arm set gives exactly the estimates and
bounds of the single `weighted_kl_chernoff` comparison with the set's settings, at the root and at
every node. `contrasts.type` is either
`versus_reference`, which requires a reference member, or `all_pairs`. Generated comparisons are
calibrated together and projected from common arm-risk regions. `all_pairs` plus
`risk_difference` also produces simultaneous rank sets. The remaining interval, hierarchy, and
independence fields have the same meanings as direct comparisons below.

## `comparisons`

```yaml
comparisons:
  - id: candidate_vs_baseline
    candidate: candidate
    baseline: baseline
    group: all
    measure: mean
    estimand: risk_difference
    method: weighted_kl_chernoff
    reference_methods: [weighted_empirical_bernstein, weighted_hoeffding]
    family: model_comparisons
    simultaneous: true
    coverage_mode: two_sided
    independence:
      primary: rollout
      sensitivities: [prompt]
    hierarchy_nodes: true
    hierarchy_calibration: frontier_sidak
    hierarchy_region: linear_program
    allocation_weight: 1
    pair_key: [prompt_id, repeat]
    diagnostic_methods: [exact_mcnemar, newcombe_paired, fixed_cluster_bootstrap]
    cluster_key: question_id
    bootstrap_replicates: 2000
    bootstrap_seed: 1234
```

`risk_difference` is candidate minus baseline, so negative values favor the candidate.
`risk_ratio` is candidate divided by baseline. `relative_risk_reduction` is one minus that ratio,
so positive values favor the candidate. Direct weighted KL/Chernoff, empirical-Bernstein, and
Hoeffding inference requires a `weighted_mean` measure. Every comparison names a multiplicity
family and an independence selection.

Each direct comparison names a simultaneous-interval family. With `flat_bonferroni`, error is
allocated over every requested overall and hierarchy-node result. With `frontier_sidak`, error is
Bonferroni-allocated over dependent hierarchy frontiers and exact Šidák product calibration is
used among disjoint sibling nodes. All comparison specifications sharing the family are calibrated
together. A `weighted_bonferroni` family uses each specification's positive `allocation_weight` as
its frontier allocation weight; ordinary `bonferroni` allocation is equal. Reference methods reuse
the same local levels and provide construction-by-construction coverage rather than joint coverage
across methods.

Compatible weighted-KL difference, ratio, and relative-reduction specifications over the same two
arms and hierarchy reuse one two-arm confidence region. Those raw regions are calibrated aliases,
so deterministic effect projections do not consume duplicate family alpha.

`hierarchy_nodes: true` repeats the same fixed-design comparison at every creator-declared node,
renormalizing the original group weights within that node. Named sensitivity profiles provide
alternative independent-block partitions. `comparison_coupling: joint` permits arbitrary
dependence between candidate and baseline observations sharing a block key.

`hierarchy_region: linear_program` places every raw node interval into one common region. Risk
differences use a prompt-contrast vector in `[-1, 1]`; ratios use a two-arm region and
linear-fractional projection. The displayed intervals therefore respect exact parent/child
aggregation and cannot be wider than their raw intervals. Set it to `none` to retain raw calibrated
intervals only. `frontier_sidak` requires `hierarchy_nodes: true`.

Pair and cluster keys for McNemar, Newcombe, and bootstrap diagnostics are supplied by the creator;
the package never infers them. These diagnostics remain separate from the primary direct interval.

See [Fixed-design model and configuration comparisons](model-comparisons.md) for the complete
mathematical construction, assumptions, output fields, and visualization guide.

## `multiplicity`

Families specify `purpose`, `method`, and `alpha`. Supported method names are:

- `intersection_union`;
- `bonferroni`;
- `weighted_bonferroni`.

The simultaneous prompt-bound family must use a Bonferroni method in v1.

## `claims`

A group-risk claim names cohort, group, and measure. A comparison claim names only its comparison.

```yaml
claims:
  - id: tail_safe
    title: Upper-tail risk is acceptable
    cohort: candidate
    group: cyber
    measure: tail
    family: deployment_gate
    acceptable_upper: 0.01
    unacceptable_lower: 0.03
    mandatory: true
```

`acceptable_upper` must be lower than `unacceptable_lower`.

## `policy`

`zero_tolerance` selects a Boolean manifest field. Any mapped failure on a flagged prompt creates a policy failure event.

## `report`

| Field | Default | Meaning |
|---|---:|---|
| `include_decisions` | false | Show overall decision, threshold-claim cards, and policy-event roll-up in human-facing reports |
| `include_prompt_text` | false | Reserved; raw content remains omitted in v1 |
| `include_observations` | true | Write normalized observation CSV |
| `decimal_places` | 6 | Markdown display precision; JSON retains full precision |

## Generate JSON schemas

```bash
inspect-safety-stats schema plan > plan.schema.json
inspect-safety-stats schema report > report.schema.json
```
