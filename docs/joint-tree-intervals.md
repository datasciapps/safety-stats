# Joint summary-tree confidence intervals

`inspect-safety-stats` can report a dependency-aware, tree-wide bootstrap interval next to
its finite-sample KL/Chernoff, empirical-Bernstein, Hoeffding, and exact-binomial intervals.
The bootstrap interval is an additional numerical view; it does not replace the existing
finite-sample interval.

## Estimand and resampling unit

The evaluation prompts and their weights remain fixed. For prompt `j` in cohort `a`, the
observed rollout count is `n_aj`, the failure count is `x_aj`, and the fixed-design node risk
is

\[
\theta_{av}=\sum_j w_{vj}p_{aj}.
\]

The method never resamples prompts. It generates Bernoulli rollout counts independently
within each fixed prompt/cohort stratum. Boundary strata use the Jeffreys-smoothed
probability

\[
\widetilde p_{aj}=\frac{x_{aj}+1/2}{n_{aj}+1}.
\]

Smoothing is used only to generate bootstrap draws. Point estimates remain the direct
Monte Carlo estimates `x_aj / n_aj`. This prevents an all-safe prompt with a small rollout
count from producing a degenerate zero-width bootstrap distribution.

The first implementation requires a declared rollout independence profile and complete
binary outcomes. If a report contains unclassified outcomes, the joint interval is skipped
and the missingness-robust finite-sample intervals remain available.

## Joint max-T calibration

Every requested root, internal hierarchy node, cohort, and declared risk-difference target
is recomputed under the same bootstrap draw. If `s_k` is the bootstrap standard error of
target `k`, the two-sided statistic is

\[
M^*=\max_k\left|\frac{\widehat\theta_k^*-E^*(\widehat\theta_k^*)}{s_k}\right|.
\]

The empirical `1 - alpha` quantile supplies one critical value for the complete family.
Unlike node-wise Bonferroni calibration, this retains the observed dependence caused by
overlapping parent/child nodes and shared configurations. Duplicate or perfectly correlated
targets do not create an additional maximum-statistic penalty.

This is bootstrap-calibrated inference, not a finite-sample coverage guarantee. It is most
useful for benchmark roots, categories, and matched configuration differences. Sparse
prompt-level inference should continue to use the exact or KL/Chernoff intervals.

### One-sided upper calibration

When `include_one_sided_upper` is true, the same bootstrap draws also calibrate

\[
M_+^*=\max_k
\frac{\widehat\theta_k^*-E^*(\widehat\theta_k^*)}{s_k}.
\]

If `c_+` is its empirical `1 - alpha` quantile, the raw upper confidence bound is

\[
U_k=\widehat\theta_k+c_+s_k.
\]

The resulting family has the approximate directional guarantee

\[
P\!\left(\theta_k\leq U_k\text{ for every selected }k\right)\approx1-\alpha.
\]

Because `M_+^* <= M^*` for every bootstrap draw, its critical value is no larger than the
corresponding two-sided max-absolute critical value. The upper bound is therefore no wider before
parameter clipping and hierarchy projection. It is the appropriate max-T result when only an upper
safety-risk limit is required.

The two-sided interval and one-sided upper bound are separately calibrated families. Do not combine
the lower endpoint of the two-sided interval with the tighter one-sided upper endpoint and describe
that hybrid as a `1 - alpha` two-sided interval.

## Common hierarchy and multi-arm region

Raw max-T bands are converted into simultaneous linear constraints over every
prompt-by-cohort probability. The library intersects those constraints with

- `0 <= p_aj <= 1`;
- every declared hierarchy weighting identity;
- all included model/language risk-difference constraints.

It then projects the common region back onto each displayed target with linear programming.
Consequently parent risks, child risks, arms, and pairwise differences cannot report an
algebraically incompatible collection of intervals. Ratios and relative-risk reductions
continue to use the existing finite-sample multi-arm region in this version.

## Configuration

```yaml
outputs:
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    reference_methods: [weighted_empirical_bernstein, weighted_hoeffding]
    family: displayed_direct_means
    groups: [all]
    independence:
      primary: rollout
    hierarchy_nodes: true
    hierarchy_calibration: frontier_sidak
    hierarchy_region: linear_program

  joint_tree_intervals:
    enabled: true
    method: stratified_bernoulli_max_t
    family: model_comparisons
    groups: [all]
    coverage_mode: two_sided
    include_one_sided_upper: true
    independence_profile: rollout
    include_comparisons: true
    replicates: 5000
    seed: 1729
    smoothing: jeffreys
    hierarchy_region: linear_program
```

The referenced profile must be declared as:

```yaml
design:
  independence_profiles:
    rollout:
      block_by: [rollout]
      comparison_coupling: independent
```

The multiplicity family contributes its `alpha`. The family covers all joint targets selected
by `groups`, across all cohorts, plus every declared risk-difference comparison when
`include_comparisons` is true.

## Report and visualisation

The JSON two-sided interval has method `stratified_bernoulli_max_t`. The separately calibrated
upper bound has method `stratified_bernoulli_max_t_upper`, coverage mode `upper_one_sided`, and no
lower endpoint. Both have role `reference` and eligibility `diagnostic`. They record the bootstrap
seed, replicate count, critical values, number of jointly calibrated targets, variance
regularisation, and hierarchy projection.

In the hierarchy visualisation:

- blue is the existing finite-sample primary interval;
- purple is the joint two-sided max-T reference interval;
- orange is the one-sided joint max-T upper endpoint;
- grey is the raw interval before hierarchy projection.

The exact numerical endpoints for both methods remain available in the interval-method
tables and comparison disclosures.

## Interpretation with few rollouts

Joint calibration can reduce multiplicity conservatism at roots and internal nodes because
those summaries share many outcomes. It cannot recover prompt-level information absent from
the experiment. With only a few rollouts, the finite-sample interval should remain the main
small-sample reference, particularly for prompts with zero failures. Increasing the number
of bootstrap replicates reduces Monte Carlo error in the bootstrap quantile; it does not
increase the evaluation sample size.
