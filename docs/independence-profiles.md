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

Each interval under a sensitivity profile has `role: sensitivity` and
`varied_assumption: independence` in `report.json`, and `report-full.md` and `report.html` label it
`sensitivity (independence)`. A judge-misclassification sensitivity has
`varied_assumption: judge_error` instead, so the two cannot be mistaken for each other.

### Verdicts by profile

Each claim in `report.json` lists its verdict under every profile in `profile_verdicts`. The
first row is the primary profile's and repeats the claim's status and bounds. Each sensitivity
row decides the same `acceptable_upper` and `unacceptable_lower` on that profile's interval,
computed with the primary interval's method, under the same integrity and confirmatory
conditions. A row also gives the interval id, the independent block count and the effective
sample size, so a reader can see how many independent units each verdict rests on.

The claim's `status`, and the report's decision, are always the primary profile's verdict.
Sensitivity verdicts show how far that verdict depends on the primary independence assumption;
they are never selected in its place. A `PASS` under `rollout` that is `INCONCLUSIVE` under
`prompt` holds only if rollouts of the same prompt are independent.

`report-full.md` adds a "Verdicts by independence profile" table under the mandatory claims when
any of them has a sensitivity profile, and the HTML claim card lists each profile's verdict
and bounds. Claims decided on a judge-corrected interval or the noisy-valid test have no
per-profile counterpart and no rows; when their evidence has sensitivity profiles, a reason
says so.

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
unrestricted. The same weighted Hoeffding, empirical-Bernstein, KL/Chernoff, and betting engines can
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

When a design asks one base prompt in several wordings, a `block_by: [question_id]` profile
treats all wordings of a base prompt as one block. See
[Prompt variants in fixed designs](plan-reference.md#prompt-variants-in-fixed-designs).

### How much the grouping matters: `CLUSTER_DESIGN_EFFECT`

Many benchmarks put many items into a few shared contexts: hundreds of questions asked inside ten
conversations, or several secrets graded in one response. Items in one context can fail together.
A profile at the item level (`[rollout]` or `[prompt_id]`) then counts every item as independent
evidence. A profile that blocks by the context column counts only the contexts, and with very few
blocks its bounds can be close to vacuous. `validate` and `analyze` both report how far apart
these two choices are in the data.

For every profile whose blocks hold rollouts of more than one prompt, the diagnostic
`CLUSTER_DESIGN_EFFECT` gives, for each cohort:

- the scored rollouts (failure or safe) and the number of blocks they fall in;
- the one-way ANOVA estimate of the intra-cluster correlation $\rho$ of their outcomes within
  the blocks;
- the design effect $1 + (m - 1)\max(\rho, 0)$, where $m = \sum_g m_g^2 / N$ is the size of
  the block an outcome sits in, averaged over the $N$ outcomes;
- the effective number of rollouts, $N$ divided by the design effect.

To get it for a column, declare a profile that blocks by that column, for example as a
sensitivity of the item-level primary:

```yaml
design:
  independence_profiles:
    item: {block_by: [prompt_id]}
    conversation: {block_by: [conversation]}
outputs:
  direct_mean_intervals:
    independence: {primary: item, sensitivities: [conversation]}
```

The diagnostic is a warning when, in some cohort, the design effect is at least 2 and a primary
profile of the plan puts rollouts of one block in different blocks of its own. It is
informational otherwise, for example when the grouping profile is itself the primary. Profiles
whose blocks never hold more than one prompt get no diagnostic. It changes no bound: it is a
measure of how much the independence choice matters, to read beside the sensitivity bounds.

The figures describe the data, and they have limits:

- Under a fixed design, the correlation also includes fixed differences in failure rates between
  the prompts of different blocks. It can therefore overstate the dependence between rollouts.
  A near-zero correlation is still a sign that the grouping matters little.
- The correlation is undefined when every rollout falls in one block, when no block holds two
  rollouts, or when all rollouts have the same outcome. The diagnostic then says so, and gives no
  design effect.
- The estimate counts rollouts equally, whatever the plan's prompt weights.

The HTML report has one global independence-profile selector. It displays only the selected
profile's intervals in risk charts, hierarchy trees, and comparison summaries. JSON and CSV retain
all prespecified profiles, their independent-block counts, effective sample sizes, calibration
details, and explicit assumptions.
