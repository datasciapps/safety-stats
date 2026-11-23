# Hierarchy-aware simultaneous inference

This document describes the finite-sample hierarchy calibration implemented by
`safety-stats`. It applies to fixed, creator-declared evaluation trees; it does not treat
questions or prompts as IID samples from a larger population.

## Statistical target

Let the primitive fixed-prompt failure probabilities be

\[
\theta=(p_1,\ldots,p_J)\in[0,1]^J.
\]

Every reported group root or hierarchy node is a declared weighted mean

\[
\mu_v=a_v^\top\theta,
\]

where $a_v$ contains the node's group weights, restricted to the node's descendants and
renormalized to sum to one. These weights come from the analysis manifest. Observed rollout counts
do not determine them.

A named independence profile partitions rollout contributions into blocks. Block $g$ has total
coefficient $A_g$ and a normalized value $Y_g\in[0,1]$, so the target remains
$\mu_v=\sum_g A_g\mathbb E[Y_g]$. Independence is assumed across blocks and arbitrary dependence
is allowed within a block. The package can construct a marginal finite-sample interval for every
$\mu_v$ using weighted KL/Chernoff, empirical Bernstein, or Hoeffding concentration.

The hierarchy problem is to make all selected node intervals simultaneous without treating every
node as independent or spending a separate Bonferroni share when the tree proves more structure.

## Dependence structure

Two facts determine the calibration:

- An ancestor and descendant reuse atomic outcomes, so their interval events are dependent.
- Sibling nodes in a complete hierarchy level may or may not contain disjoint profile blocks. When
  their resolved block supports are disjoint, their interval events are mutually independent.

The second fact does not require identical distributions. A chemical-risk node and a cyber-risk
node may have different probabilities, weights, and rollout counts. Their statistics are
independent because they are deterministic functions of disjoint independent atoms.

## Independent frontiers

An independent frontier $F$ is a set of targets whose atomic identifier sets are pairwise
disjoint. For a one-direction frontier budget $\alpha_F$, equal Šidák allocation gives each of
$m_F$ targets

\[
\alpha_v=1-(1-\alpha_F)^{1/m_F}.
\]

If target $v$ has endpoint error probability at most $\alpha_v$, then

\[
\Pr\left(\bigcap_{v\in F}\{\text{endpoint }v\text{ covers}\}\right)
\ge \prod_{v\in F}(1-\alpha_v)
=1-\alpha_F.
\]

The equality is a calibration equality. Individual concentration bounds may be conservative, so
actual coverage can exceed it.

Different hierarchy levels are overlapping frontiers. The implementation therefore allocates the
directional family budget by Bonferroni across the $K$ compiled frontiers:

\[
\alpha_F=\frac{\alpha_d}{K}.
\]

No cross-level independence is assumed. By the union bound,

\[
\Pr\{\text{every endpoint in the compiled tree covers}\}
\ge 1-\sum_F\alpha_F
\ge 1-\alpha_d.
\]

For a family alpha $\alpha$ in `two_sided` mode, $\alpha_d=\alpha/2$ is used separately for lower
and upper endpoints. A final union bound gives simultaneous two-sided coverage at least
$1-\alpha$.

## Hierarchy compilation

For every selected cohort and group, the compiler:

1. creates a frontier for the group root;
2. creates one frontier for each declared non-prompt hierarchy level;
3. verifies that the level's nodes are nonempty and cover the group, then resolves their selected
   independence-profile block supports;
4. compares weighted target signatures and merges exact duplicates as aliases; and
5. uses Šidák only for pairwise-disjoint block supports, splits crossing-block siblings into
   conservative singleton frontiers, and treats all remaining cohorts, groups, and levels as
   potentially dependent.

The frontiers are those of the plan, not of the rollouts that arrived. In a fixed design, a cohort
that has no rows of some prompt, because it matches no evaluation (`COHORT_EMPTY`) or misses the
prompt (`PROMPT_MISSING`), gets no interval for the targets that hold that prompt: every interval
needs a complete attempted-rollout ledger. Those targets keep their places and their alpha in
their frontiers all the same, with the block supports a complete ledger would give them. Under
the `rollout` profile each has blocks of its own. Under a profile that blocks only by manifest
columns, `prompt_id`, `question_id` or `task_id`, the manifest rows of its prompts name its
blocks. Every other cohort and group is then calibrated as with a complete ledger. Under any other
profile, such as `[run_key]` or `[repeat]`, the blocks depend on rollouts that are not there, so
the frontier is split into singleton frontiers as if its siblings crossed blocks. That spends the
most alpha: the other intervals under such a profile can be wider than a complete ledger would
make them, never narrower. Comparisons are compiled the same way, with both arms' ledgers.

An alias is important when a level has one node. In the HarmBench harmful-behavior group, the group
root and the sole harmful-compliance task are exactly the same weighted statistic. Reporting both
is useful for navigation, but charging alpha twice would not add a coverage guarantee. They share
one frontier target and one local alpha.

Overlapping groups are not merged merely because their prompt identifiers overlap. They are either
exact weighted aliases or distinct dependent frontiers. This is conservative and avoids an
unjustified independence claim.

## Common linear confidence region

For one concentration method and independence profile, let the calibrated raw interval for target
$v$ be $[L_v^0,U_v^0]$. The package forms

\[
\mathcal C=\left\{
\theta\in[0,1]^J:
L_v^0\le a_v^\top\theta\le U_v^0\quad\text{for every }v
\right\}.
\]

The displayed interval is the coordinate projection

\[
L_v=\min_{\theta\in\mathcal C}a_v^\top\theta,
\qquad
U_v=\max_{\theta\in\mathcal C}a_v^\top\theta.
\]

These are linear programs. On the raw simultaneous-coverage event, the true $\theta$ is in
$\mathcal C$, so every projected interval covers simultaneously. Projection never widens a raw
interval and enforces exact parent/child coherence through the shared primitive vector.

That guarantee is joint, and projection does not keep a separate one for each direction. A
projected upper endpoint optimises over a region that the raw lower endpoints also shape, so a
raw lower endpoint that misses can pull it below the truth.

For example, take a parent mean $(\theta_1+\theta_2)/2\in[0.2,0.5]$ with children
$\theta_1\in[0.05,0.8]$ and $\theta_2\in[0.9,1]$, and the truth $(0.5,0.1)$:
- every raw upper endpoint covers, and only $\theta_2$'s lower endpoint misses;
- yet the projected upper endpoint for $\theta_1$ is $0.1$, below the true $0.5$.

Projected intervals therefore report the simultaneous floor as their coverage in each
direction: 95% at family alpha 0.05 in `two_sided` mode, not 97.5%.

Projection is run separately for weighted KL/Chernoff, empirical Bernstein, and Hoeffding, and
separately for every named independence profile. The package does not intersect
different methods and then claim their original coverage. Each reference construction retains its
own guarantee.

## HarmBench allocation example

The included completed-run analysis selects one 300-behavior harmful-compliance group with the
hierarchy

```text
group root
  -> task (1 node; an alias of the root)
     -> risk domain (6 nodes)
        -> question (300 nodes)
           -> prompt (not reported as a direct hierarchy mean)
```

At family alpha 0.05 in two-sided mode, each direction receives 0.025. After merging the root/task
alias there are three dependent frontiers, so each receives

\[
\alpha_F=0.025/3=0.008333\ldots.
\]

The local endpoint alphas are therefore:

| Frontier | Distinct targets | Local endpoint alpha |
|---|---:|---:|
| root + task alias | 1 | 0.0083333333 |
| risk domains | 6 | 0.0013937361 |
| questions | 300 | 0.0000278938 |

The 300-node value is slightly larger than flat Bonferroni within that frontier because it uses
the exact independent product. The larger practical gain is that the root/task duplicate is not
charged twice and the error budget is structured by distinct levels instead of treating 307
dependent display rows as one unstructured list.

## Configuration

```yaml
outputs:
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    reference_methods: [weighted_empirical_bernstein, weighted_hoeffding]
    family: displayed_direct_means
    groups: [all-behaviors]
    simultaneous: true
    coverage_mode: two_sided
    independence:
      primary: rollout
      sensitivities: [prompt]
    hierarchy_nodes: true
    hierarchy_calibration: frontier_sidak
    hierarchy_region: linear_program
```

The profiles themselves are declared under `design.independence_profiles`; see
[Configuring hierarchy and independence](independence-profiles.md). Use
`hierarchy_calibration: flat_bonferroni` for the simpler target-by-target allocation. Use
`hierarchy_region: none` to display calibrated raw intervals without linear projection.

## Across-model hierarchy regions

The same calibration applies to direct model/configuration comparisons. A comparison sibling is a
joint block containing the candidate and baseline outcomes for that node's prompts. Disjoint
sibling blocks are independent under rollout independence even though the two arms and all prompt
probabilities need not be identically distributed.

Empirical-Bernstein and Hoeffding risk-difference nodes are projected from one prompt-contrast
region with primitive coordinates in `[-1, 1]`. The sparse KL primary uses a common
candidate/baseline region for differences, ratios, and relative reduction: difference is a linear
projection, ratio uses exact linear-fractional optimization, and relative reduction is the
monotone transform `1 - ratio`. The comparison report therefore preserves the same hierarchy and
coverage semantics as the within-model report.

All comparison specifications in one multiplicity family share its budget. Compatible KL
difference, ratio, and relative-reduction hierarchies are deterministic projections of one
two-arm confidence region, so the compiler aliases that shared region instead of charging it three
times. See
[Fixed-design model and configuration comparisons](model-comparisons.md) for the equations,
configuration, assumptions, and output fields.

## Report fields

Every direct interval records:

- `calibration_method`;
- `calibration_frontier`;
- `calibration_frontier_alpha`;
- `calibration_frontier_size`;
- `alpha_lower` and `alpha_upper` for the local endpoint;
- `raw_lower` and `raw_upper` before projection;
- `lower` and `upper` after projection;
- `projection_method` and `projection_constraint_count`; and
- the named independence profile, independent-block count, and explicit assumptions.

The HTML report displays one collapsible hierarchy selector for both within-model and across-model
views. Within-model bands use a common 0–100% risk axis. Across-model views show both arm risks and
an effect axis with the no-change reference at zero (difference/reduction) or one (ratio). Blue
bands are projected simultaneous intervals and gray bands are raw intervals. A global selector
shows one named independence profile at a time. The CSV and JSON artifacts retain every
prespecified profile for downstream work.

## What this does not claim

This implementation is not hierarchical gatekeeping and does not stop inference at failed parent
nodes. Every selected node is reported in one simultaneous family.

It is also not a fully joint ancestor/descendant Chernoff maximum-statistic region. Such a method
would need a finite-sample critical value for overlapping weighted sums over heterogeneous
Bernoulli atoms. The current method uses exact independence only where disjointness proves it and
uses a union bound elsewhere. It therefore provides a rigorous production method without
mislabeling simulation or an asymptotic covariance approximation as exact.

Finally, the intervals quantify stochastic rollout uncertainty on the fixed evaluation design.
They do not establish coverage of omitted behaviors, new prompts, new attacks, deployment drift,
or scorer error.
