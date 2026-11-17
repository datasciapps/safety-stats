# Fixed-design model and configuration comparisons

This module compares two or more model or system configurations on the same creator-declared fixed
evaluation design. It reports the risk in each arm and the requested effects at the design root
and every requested hierarchy node. Its confidence statements concern stochastic rollout
uncertainty on those fixed prompts. They do not treat the prompts as an IID sample from a larger
population.

## Multi-arm comparison sets

For two or more configurations, use `comparison_sets` rather than manually enumerating pairs:

```yaml
comparison_sets:
  - id: model_family
    cohorts: [reference, candidate_a, candidate_b, candidate_c]
    contrasts:
      type: versus_reference
      reference: reference
    estimands: [risk_difference, risk_ratio, relative_risk_reduction]
    group: all
    measure: mean
    method: weighted_kl_chernoff
    reference_methods: [weighted_empirical_bernstein, weighted_hoeffding]
    family: model_comparisons
    coverage_mode: two_sided
    independence:
      primary: rollout
      sensitivities: [prompt, category]
    hierarchy_nodes: true
    hierarchy_calibration: frontier_sidak
    hierarchy_region: linear_program
```

`versus_reference` creates one contrast from every non-reference cohort to the declared reference.
`all_pairs` creates each unordered pair once, orienting the later cohort in `cohorts` as the
candidate. All estimands for all generated pairs share one simultaneous family construction.

Each pair initially supplies calibrated lower and upper constraints for both arm risks. Constraints
for a shared arm are intersected, and every arm is projected through its fixed-design hierarchy
once. All differences, ratios, and relative reductions are then derived from these common arm
projections. Consequently the report cannot display different risk intervals for the same arm in
different comparisons. The construction is conservative because the family calibration protects
all generated pair constraints, but it is finite-sample and does not rely on a multivariate normal
approximation.

For an `all_pairs` set containing risk differences, the report also supplies simultaneous rank
sets. An arm is declared definitely better than another only when the corresponding simultaneous
risk-difference interval lies wholly below zero. If (b_k) arms are definitely better than arm
(k), and arm (k) is definitely better than (w_k) arms, its reported rank range is

\[
[1+b_k,\ K-w_k].
\]

Rank one means lowest harmful-compliance risk. Overlapping rank sets honestly retain ordering
uncertainty; the software does not force a total ranking from point estimates.

## Estimands

For prompt \(j\), let \(p_{Cj}\) and \(p_{Bj}\) be the candidate and baseline harmful-compliance
probabilities under their recorded generation configurations. Let \(w_j\ge 0\) be fixed design
weights with \(\sum_jw_j=1\). The arm risks are

\[
R_C=\sum_jw_jp_{Cj},\qquad R_B=\sum_jw_jp_{Bj}.
\]

The available comparison estimands are:

\[
\Delta=R_C-R_B,
\qquad
RR=\frac{R_C}{R_B},
\qquad
RRR=1-RR.
\]

Negative \(\Delta\), \(RR<1\), and \(RRR>0\) favor the candidate. A ratio is intrinsically
poorly determined when baseline risk is near zero. If the confidence region allows \(R_B\) to
approach zero while \(R_C>0\), the upper risk-ratio endpoint is mathematically unbounded and is
reported as `null`, not replaced by an arbitrary cap.

At a hierarchy node \(v\), the original design weights are restricted to its descendant prompts
and renormalized:

\[
w_{vj}=\frac{w_j\mathbf 1\{j\in v\}}
{\sum_{k\in v}w_k}.
\]

The node effect is the same estimand with \(w_j\) replaced by \(w_{vj}\). This is a conditional
summary over fixed descendants, not a population estimate for unseen categories or prompts.

## Primary interval methods

The recommended primary method for sparse binary failures is `weighted_kl_chernoff`. Two
prespecified references are available:

- `weighted_empirical_bernstein`, which adapts to a finite-sample upper bound on variance; and
- `weighted_hoeffding`, which uses boundedness alone.

Each construction has its own coverage guarantee. Showing all three is valid when they are
prespecified; selecting the shortest one after observing the results does not preserve the stated
coverage without another selection adjustment.

### Risk-difference intervals

For empirical Bernstein and Hoeffding, the implementation directly rewrites

\[
Q=\frac{1+\Delta}{2}
=\frac12\sum_jw_jp_{Cj}
+\frac12\sum_jw_j(1-p_{Bj}).
\]

Candidate failures and baseline safe outcomes are therefore bounded independent contributions to
one weighted mean. The selected concentration engine bounds \(Q\), and the result is transformed
back by \(\Delta=2Q-1\). This supports unequal candidate and baseline rollout counts.

Empirical Bernstein estimates the variance of \(Q\) about centers fixed by arm, and uses
whichever of a few fixed choices gives the smallest estimate: on the rollout unit, each arm's
rollouts about 0 or about 1; on the prompt-cluster unit, about 1/2, 0 or 1. Two sparse arms then
give a small variance and a correspondingly narrow interval, and so do arms that fail often. See
[Statistical methods](statistical-methods.md) for the derivation.

Weighted KL/Chernoff deliberately uses a different sparse-risk construction. Applying a generic
Bernoulli-KL inversion to \(Q\), whose mean is ordinarily near one half, discards the low-variance
structure of two near-zero harmful-compliance rates. The KL method instead constructs a joint
two-arm fixed-design rectangle and reports

\[
\Delta\in[L_C-U_B,\;U_C-L_B].
\]

Each directional endpoint splits its local budget between the two required arm endpoints. This is
usually much tighter in sparse safety evaluations and supplies the same two-arm region used by
ratio and relative-reduction projections.

For unclassified attempted outcomes, the lower endpoint treats candidate unknowns as safe and
baseline unknowns as failures. The upper endpoint reverses those completions. Unknown labels
therefore cannot make either direction look artificially favorable.

### Risk-ratio and relative-reduction intervals

For ratios, each node first constructs a simultaneous rectangular confidence set for its two arm
means. A directional ratio endpoint uses two arm endpoints, so its local error budget is split by
Bonferroni without assuming independence between arms:

\[
RR\in\left[\frac{L_C}{U_B},\frac{U_C}{L_B}\right].
\]

The relative-risk-reduction interval is the monotone transformation

\[
[L_{RRR},U_{RRR}]=[1-U_{RR},1-L_{RR}].
\]

## Hierarchy-wide simultaneous calibration

Set `hierarchy_nodes: true` to include every non-prompt level. With
`hierarchy_calibration: frontier_sidak`, the compiler forms one frontier for the root and one for
each complete hierarchy level. Nodes within a level have disjoint prompt blocks. Under the chosen
atomic independence assumption, their comparison statistics are independent even though their
Bernoulli probabilities need not be identical.

For a directional family budget \(\alpha_d\), potentially dependent frontiers receive Bonferroni
budgets \(\alpha_F\). Within a frontier of \(m_F\) nodes, the local endpoint level is

\[
\alpha_v=1-(1-\alpha_F)^{1/m_F}.
\]

Ancestor and descendant frontiers reuse data and are never treated as independent. Identical
weighted statistics, such as a design root and a one-node top level, are aliases and spend error
budget once. All simultaneous comparison specifications sharing a multiplicity family are
calibrated together. Compatible KL difference, ratio, and relative-reduction specifications over
the same arms, design, weights, hierarchy, and coverage settings are projections of the same
two-arm confidence region. Their identical raw regions are aliases and spend family error once;
any number of deterministic projections inherits that region's simultaneous coverage.

`flat_bonferroni` is the conservative compatibility option. It treats every displayed comparison
target as a separate dependent endpoint.

### Independence assumptions

Each comparison selects a named independence profile. The profile's `block_by` fields partition
weighted observations, and arbitrary dependence is allowed inside a block. `comparison_coupling:
independent` keeps cohort blocks separate; `joint` combines matching candidate and baseline block
keys and permits cross-arm dependence within them. KL two-arm regions do not require cross-arm
independence, but the selected within-arm block partition still determines their concentration.
For frontier Šidák calibration, sibling nodes must have disjoint resolved block supports;
otherwise the compiler uses conservative between-node allocation. See
[Configuring hierarchy and independence](independence-profiles.md).

## Common hierarchy confidence regions

Raw node intervals are simultaneous but do not automatically look coherent: independently
computed child and parent bounds can fail to reflect their exact aggregation relationship. Set
`hierarchy_region: linear_program` to project every raw interval from one common region.

### Risk difference

Let \(d_j=p_{Cj}-p_{Bj}\in[-1,1]\), and let \([L_v^0,U_v^0]\) be the raw simultaneous interval
for node \(v\). The contrast region is

\[
\mathcal C_\Delta=\left\{d\in[-1,1]^J:
L_v^0\le w_v^\top d\le U_v^0\ \text{for every }v\right\}.
\]

For empirical-Bernstein and Hoeffding direct contrasts, the displayed endpoints are exact
linear-program projections

\[
L_v=\min_{d\in\mathcal C_\Delta}w_v^\top d,
\qquad
U_v=\max_{d\in\mathcal C_\Delta}w_v^\top d.
\]

The primary sparse KL difference instead uses the two-arm region below. Its objective
\(w_v^\top c-w_v^\top b\) is linear and is optimized directly over that common region.

### Risk ratio

For ratios, the common primitive parameters are both prompt-risk vectors
\(c,b\in[0,1]^J\). Every candidate and baseline arm interval contributes a linear constraint to
one two-arm region \(\mathcal C_{CB}\). The node endpoints solve

\[
\min_{(c,b)\in\mathcal C_{CB}}\frac{w_v^\top c}{w_v^\top b},
\qquad
\max_{(c,b)\in\mathcal C_{CB}}\frac{w_v^\top c}{w_v^\top b}.
\]

These linear-fractional programs are solved with the Charnes--Cooper transformation and HiGHS.
Relative-reduction intervals are transformed from the corresponding ratio projection. Projection
uses only constraints that already hold on the simultaneous event, so it preserves coverage and
cannot widen a raw interval.

The projection is performed separately for each interval method and independence profile.
Regions from different methods or profiles are not intersected with one another.

## Plan configuration

```yaml
comparisons:
  - id: candidate_vs_baseline_difference
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
    pair_key: [prompt_id, epoch]
    diagnostic_methods: [exact_mcnemar, newcombe_paired, fixed_cluster_bootstrap]
    cluster_key: prompt_id

  - id: candidate_vs_baseline_ratio
    candidate: candidate
    baseline: baseline
    group: all
    measure: mean
    estimand: risk_ratio
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

multiplicity:
  families:
    - id: model_comparisons
      purpose: simultaneous_intervals
      method: bonferroni
      alpha: 0.05
```

`two_sided` splits family alpha equally between lower and upper directions. The reported complete
interval collection then has a familywise coverage floor of \(1-\alpha\). `per_direction` spends
the full family alpha in each direction, so the two-sided floor is only \(1-2\alpha\).

## Report and visualization

`report.json` and `comparisons.csv` contain one row per comparison target. Important fields are:

- `candidate_estimate`, `baseline_estimate`, and `estimate`;
- displayed `lower` and `upper`;
- `raw_lower` and `raw_upper` before common-region projection;
- `interval_method`, the named independence profile, independent-block count, and
  `chernoff_kernel`;
- `calibration_frontier`, `calibration_frontier_alpha`, and local `alpha_lower`/`alpha_upper`;
- `projection_method` and `projection_constraint_count`; and
- for ratio constructions, candidate and baseline arm endpoints inside each interval record.

The single HTML hierarchy view offers both `Within model` and `Across models` selections. An
across-model row displays baseline risk, candidate risk, and the selected effect on the same tree
node. A global independence-profile selector shows one profile at a time. Blue bands are projected
intervals and faint bands are raw intervals. The compact comparison table below the tree retains
only overall effects, alternative methods for the selected profile, and paired diagnostics.

Generate the report with:

```bash
inspect-safety-stats analyze \
  --plan analysis-plan.yaml \
  --logs logs/ \
  --output report

inspect-safety-stats visualize \
  --report report/report.json \
  --output report/model-comparison.html
```

## Interpretation boundaries

The comparison interval does not establish behavioral coverage beyond the fixed design. It also
does not account for scorer misclassification, model-version drift, a changed attack template, or
unrecorded generation differences. The two cohorts must represent the intended configurations,
the design weights must be fixed independently of outcomes, and any diagnostic pairing or cluster
keys must be declared by the evaluator rather than inferred by the package.
