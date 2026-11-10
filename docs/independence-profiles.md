# Configuring hierarchy and independence

The evaluation hierarchy and the stochastic independence structure are separate declarations.
The hierarchy says which fixed items are aggregated and displayed. An independence profile says
which observed generations may be treated as independent for concentration inference. The
library never derives one from the other.

## Plan interface

Profiles are named once under `design.independence_profiles`:

```yaml
design:
  hierarchy: [task_id, risk_domain, question_id, prompt_id]
  independence_profiles:
    rollout:
      block_by: [rollout]
      comparison_coupling: independent
    prompt:
      block_by: [prompt_id]
      comparison_coupling: joint
    question:
      block_by: [question_id]
      comparison_coupling: joint
```

`block_by` accepts:

- `[rollout]`, a reserved key meaning one independent block per normalized attempted rollout;
- normalized observation fields such as `prompt_id`, `question_id`, `task_id`, `repeat`,
  `run_key`, or `configuration_fingerprint`. The names observations had before report schema
  3.0 still work: `epoch` for `repeat`, `eval_id` for `run_key`, `sample_id` for `rollout_id`,
  `inspect_id` for `item_id`, `source_log` for `source` and `source_log_sha256` for
  `source_sha256`; and
- any column in the design manifest, including evaluation-specific columns such as
  `risk_domain`, `category_id`, or `attack_family`.

Multiple fields form a composite key. For example, `[question_id, repeat]` groups the same
repeat across all prompts belonging to one question. `rollout` cannot be combined with another field.
Unknown fields, duplicate fields, and empty block lists are rejected. So is a manifest column
used as a block key that has no value for some prompt: a missing value identifies no block. Give
a prompt that forms its own block a unique value, such as its `prompt_id`.
If a name exists in both schemas, the normalized observation field takes precedence; use unique
names for evaluation-specific manifest columns.
Outcome-dependent observation fields such as `failure`, `score_raw`, `error`, and `retries` are
not eligible block keys: the independence partition must be fixed without looking at outcomes.

An analysis output then selects one primary profile and zero or more prespecified sensitivity
profiles:

```yaml
outputs:
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    reference_methods: [weighted_empirical_bernstein, weighted_hoeffding]
    family: displayed_direct_means
    independence:
      primary: rollout
      sensitivities: [prompt, question]
```

Comparisons use the same selector:

```yaml
comparisons:
  - id: candidate_vs_baseline
    candidate: candidate
    baseline: baseline
    group: all
    measure: mean
    estimand: risk_difference
    method: weighted_kl_chernoff
    family: model_comparisons
    independence:
      primary: rollout
      sensitivities: [prompt, question]
```

This interface was introduced in analysis schema `2.0` and package version `0.7.0`
and remains current in `0.15.0`. The removed
`independence_unit` and `cluster_sensitivity` fields are deliberately not translated. A schema
`1.0` or legacy-field plan fails strict validation so its assumptions cannot silently change.
Existing Inspect logs remain reusable: write a schema `2.0` plan and rerun only
`inspect-safety-stats analyze`; model generations do not need to be repeated.

## Statistical construction

For a target node, prompt $j$ has fixed design weight $w_j$, $n_j$ attempted rollouts, and
binary outcome $X_{jr}$. Each rollout contributes

\[
c_{jr}X_{jr},\qquad c_{jr}=w_j/n_j,
\]

to the fixed-design mean. A profile partitions those contributions into blocks $B_g$. Define

\[
A_g=\sum_{(j,r)\in B_g}c_{jr},\qquad
Y_g=A_g^{-1}\sum_{(j,r)\in B_g}c_{jr}X_{jr}.
\]

Then $0\le Y_g\le1$, $\sum_g A_g=1$, and the target is exactly

\[
\mu=\sum_g A_g\,\mathbb E[Y_g].
\]

The declared assumption is independence across the $Y_g$. Dependence inside a block is
unrestricted. The same weighted Hoeffding, empirical-Bernstein, and KL/Chernoff engines can
therefore operate on `(Y_g, A_g)` regardless of how the evaluation creator named or nested the
items. Missing labels are completed as zero for a lower endpoint and one for an upper endpoint
before block aggregation.

The effective concentration sample size reported for a profile is

\[
n_{\mathrm{eff}}=\left(\sum_g A_g^2\right)^{-1}.
\]

It is a property of the declared block geometry, not the number of prompts or raw rollouts.
Coarser blocks normally reduce it and widen intervals.

## Cross-model coupling

`comparison_coupling` controls whether blocks with the same key in different cohorts are kept
separate or combined:

- `independent` prefixes the cohort identifier to every comparison block. It asserts independence
  across model/configuration arms as well as across the declared within-arm blocks.
- `joint` puts candidate and baseline observations with the same block key into one block. It
  permits arbitrary dependence across arms within that block and only assumes independence across
  distinct block keys.

For risk differences, the implementation applies the block partition directly to
$Q=(1+\mu_C-\mu_B)/2$, whose block contributions remain bounded in `[0,1]`. Sparse
KL/Chernoff differences and all ratios use simultaneous candidate/baseline mean regions; these do
not require cross-arm independence, but the chosen partition still controls within-arm block
independence and interval width.

Use `joint` for paired prompts, shared decoding seeds, common perturbation batches, or any other
mechanism that can couple arms. Use `independent` only when the two cohorts' generation randomness
is genuinely separate.

## Interaction with hierarchy calibration

At each displayed hierarchy level, the compiler calculates the actual block support of every
sibling node under the selected profile.

- If siblings use pairwise-disjoint blocks, their interval events are functions of disjoint
  independent variables and may use Šidák calibration within that frontier.
- If a declared block crosses siblings, those siblings are not treated as independent. The
  compiler splits the frontier into conservative singleton frontiers and allocates error between
  them by Bonferroni.

For example, `block_by: [risk_domain]` produces disjoint blocks at a risk-domain level but a block
can cross question nodes below it. The risk-domain siblings may use Šidák; the affected question
nodes do not. This check is based on resolved data, not on names or hierarchy position.

Every independence profile receives a separate coverage construction. Sensitivity profiles are
not pooled into one confidence family and cannot be selected post hoc while retaining the primary
profile's guarantee.

## Choosing profiles

A useful default for ordinary temperature sampling is:

```yaml
rollout:
  block_by: [rollout]
  comparison_coupling: independent
```

Add coarser profiles when shared state, correlated scoring, batched attacks, prompt families, or
unmodelled provider behavior make rollout independence questionable. The profile should name the
smallest blocks that are defensibly independent. Making blocks coarser is not automatically more
correct: it changes the assumption from independence of individual generations to independence of
whole batches, and very few blocks can make meaningful finite-sample inference impossible.

The HTML report has one global independence-profile selector. It displays only the selected
profile's intervals in risk charts, hierarchy trees, and comparison summaries. JSON and CSV retain
all prespecified profiles, their independent-block counts, effective sample sizes, calibration
details, and explicit assumptions.
