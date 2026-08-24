# Generative hierarchy evaluation planner: implementation plan

**Status:** slice G1 implemented; slices 3, 4 and 7 closed by the slice 0 measurement  
**Target package:** `inspect-safety-stats`  
**Relationship to the current package:** separate generative-design module that reuses ingestion,
outcome mapping, hierarchy identifiers, and report components  
**Scope of this plan:** as written, implementation through cross-entropy adaptive importance
sampling; as amended, direct target-distribution sampling with guaranteed stratum coverage.
SMC, subset simulation, evolutionary prompt mutation, and anytime-valid live stopping were
never in scope  
**Last updated:** 24 August 2026  
**Amendments:** *Slice 0 result* -- variance reduction does not improve the finite-sample
bound this package reports, which closes slices 3, 4 and 7. *Slice G1 as built* -- the
reduced roadmap, the decisions taken, and where the implementation departs from sections
6 to 8. Both are at the end of this document and supersede the plan above where they
disagree.

## 1. Objective

The extension will estimate a failure probability at any node of a deep, evaluator-declared
hierarchy without enumerating or evaluating every possible terminal prompt.

The hierarchy begins with fixed safety-problem classes and fixed semantic seeds. Below each seed,
finite structural arms describe semantically controlled variations such as language, style,
directness, intent framing, and wrapper. A terminal rewrite node is not an enumerated list of
prompts. It is a frozen stochastic LLM rewrite procedure that generates prompts conditional on the
path above it.

The primary root estimand is the failure probability obtained by:

1. selecting a structural path using declared target weights;
2. sampling a valid rewrite from that path's declared LLM rewrite procedure;
3. sampling a target-model rollout under a fixed configuration; and
4. applying the declared binary outcome mapping.

The extension should make partial evaluation efficient by adding, in order:

1. direct target-distribution Monte Carlo;
2. hierarchically balanced stratified sampling;
3. nested variance estimation and cost-aware allocation;
4. randomized balanced incomplete path assignments where their inclusion probabilities are
   known;
5. paired evaluation across models and configurations;
6. model-assisted/control-variate estimation; and
7. cross-entropy adaptive importance sampling over the structured hierarchy.

The extension must preserve the interpretation of the root probability as the risk under the
declared generative design. It must not interpret the taxonomy as a claim that all possible safety
hazards or prompt transformations have been covered.

## 2. Why this is a separate module

The current empirical engine conditions on a fixed, materialized prompt design. It treats prompt
weights as fixed and rollout outcomes as the principal stochastic observations. The proposed
extension introduces additional randomization:

- selection of structural paths;
- sampling of LLM rewrites;
- potentially unequal and adaptive selection probabilities; and
- pilot-dependent allocation of later evaluation effort.

Conditioning on the generated prompts and passing them directly to the fixed-design engine would
estimate risk over the realized generated prompts. It would not, by itself, account for selection
and rewrite-generation uncertainty in the generative root estimand.

The extension therefore needs separate planning, sampling-ledger, estimator, and result schemas.
It should reuse the following current components where their semantics remain correct:

- Inspect log ingestion and outcome normalization;
- end-to-end versus underlying-model outcome semantics;
- configuration fingerprints and source checksums;
- hierarchy path identifiers;
- missing-outcome completion logic;
- model/configuration cohort alignment;
- report serialization and visual tree components; and
- lower-level bounded-mean numerical primitives where their assumptions match the new sampling
  design.

The fixed-design report may still be produced as a conditional audit of a materialized batch. The
generative report must remain visibly distinct and must identify path selection and rewrite
generation as sources of uncertainty.

## 3. Statistical object

### 3.1 Generative tree

Let the finite structural hierarchy have terminal structural paths \(h\in\mathcal H\). A path may
contain, for example,

```text
root
  -> problem class
     -> semantic seed
        -> language
           -> style
              -> directness / intent framing
                 -> wrapper
                    -> LLM rewrite kernel
                       -> target-model rollout
```

The evaluator declares a conditional target weight on every structural edge. The target weight of
a complete structural path is

\[
w(h)=\prod_{e\in\operatorname{path}(h)}w(e\mid\operatorname{parent}(e)),
\qquad
\sum_{h\in\mathcal H}w(h)=1.
\]

Zero-weight branches are outside the estimand. Every positive-weight branch must remain reachable
by any sampling policy used for estimation.

### 3.2 Rewrite kernel

For path \(h\), let \(K_h\) be the distribution induced by the frozen rewrite procedure. It
includes:

- rewrite model and immutable model identifier where available;
- system and user rewrite instructions;
- decoding configuration;
- language, style, and other path instructions;
- semantic-equivalence validation;
- invalid-rewrite rejection and retry rules;
- deduplication rules; and
- normalization or post-processing applied before evaluation.

If invalid outputs are rejected until a valid rewrite is produced, \(K_h\) denotes the accepted
distribution induced by that complete procedure. Operational retry exhaustion must be recorded as
an incomplete sampling unit; it must not silently alter the path distribution.

### 3.3 Node risks

Let \(X\sim K_h\) be a valid rewrite, let \(Y\sim M(\cdot\mid X)\) be the target-model rollout,
and let \(F(X,Y)\in\{0,1\}\) be the failure indicator after outcome normalization. The terminal
structural-path risk is

\[
p_h=\mathbb E_{X\sim K_h}\mathbb E_{Y\sim M(\cdot\mid X)}[F(X,Y)].
\]

For any internal node \(v\), let \(D(v)\) be its descendant structural paths and

\[
W_v=\sum_{h\in D(v)}w(h).
\]

The node risk is

\[
R_v=\frac{1}{W_v}\sum_{h\in D(v)}w(h)p_h.
\]

The root has \(W_{\mathrm{root}}=1\), so

\[
R_{\mathrm{root}}=\sum_h w(h)p_h.
\]

Every reported ancestor must be a deterministic aggregation of the same descendant estimands.
No independently fitted parent estimate should be allowed to disagree with its declared child
weights except for displayed unprojected numerical diagnostics.

### 3.4 Fixed semantic seeds

Semantic seeds are fixed evaluator-selected design objects. Unless a separate seed-sampling design
is explicitly supplied, the extension does not treat them as IID observations from a population
of possible safety questions.

The recommended default is:

- include every semantic seed in the estimation design;
- treat semantic seed as a mandatory stratification frontier;
- assign seed weights prospectively, commonly equally within a problem class; and
- sample only the descendant transformation and rewrite processes.

If the seed set is too large to enumerate, sampling seeds is permitted only with recorded target
and selection probabilities. The report must then distinguish uncertainty from seed selection,
rewrite generation, and rollouts.

### 3.5 Scope of the probability

The resulting probability refers to the declared structural weights and frozen rewrite kernels.
It is not a probability over all conceivable semantically equivalent prompts, all future attacks,
or a deployment distribution unless those objects have separately been identified with the
declared generative design.

## 4. Statistical assumptions

The first production implementation should use the following default assumptions:

1. Structural target weights are fixed independently of outcomes.
2. A sampling unit's selection probability is known before its outcome is observed.
3. Rewrite draws are independent conditional on their structural path and frozen kernel.
4. Target-model rollouts are independent conditional on the generated prompt and fixed target
   configuration.
5. Outcome mapping and semantic validation are stable over the run.
6. Model, rewriter, validator, scorer, and decoding configurations are fingerprinted.
7. Reusing one generated rewrite for several target-model rollouts creates a rewrite cluster.
8. Reusing the same rewrite across model cohorts creates deliberate pairing, not independent
   prompt generation.

Alternative independence profiles may group rewrite batches, provider batches, or other declared
shared state. They must be defined prospectively from identifiers that do not depend on outcomes.

## 5. Non-goals through the cross-entropy slice

The implementation described here will not:

- infer the taxonomy from prompt text;
- assert coverage of a broader threat space;
- use unconstrained LLM mutation inside the primary importance-sampling estimator;
- estimate likelihood ratios between arbitrary free-form LLM mutation kernels;
- use SMC, adaptive multilevel splitting, or subset simulation;
- implement outcome-dependent live stopping;
- use Bayesian hierarchical shrinkage as the primary root estimator;
- model scorer or judge error;
- estimate deployment prevalence; or
- turn risk-directed discovery results into root-probability observations without a valid
  sampling probability.

## 6. Proposed package architecture

Add a new `generative` package rather than expanding `certification` or `structural`:

```text
src/inspect_safety_stats/
  generative/
    __init__.py
    schema.py              # strict plan and result schemas
    taxonomy.py            # compile and validate the structural tree
    weights.py             # target edge/path/node weights
    kernels.py             # rewrite-kernel protocol and fingerprints
    ledger.py              # selection, generation, and evaluation ledger
    sampling.py            # direct and stratified path samplers
    balance.py             # recursive allocation and incomplete-block schedules
    variance.py            # nested variance components and cost models
    allocation.py          # pilot-based constrained allocation
    model_assisted.py      # cross-fitted control-variate estimators
    cross_entropy.py       # factorized CE proposal learning
    inference.py           # node estimators and confidence bounds
    comparisons.py         # paired multi-cohort effects
    inspect_bridge.py      # materialize Inspect datasets and join logs
    artifacts.py           # JSON, CSV, Markdown, and HTML output
```

Add separate result schemas under:

```text
src/inspect_safety_stats/schema/
  generative_reports.py
```

The module should depend only on existing core dependencies: NumPy, pandas, SciPy, Pydantic, and
Inspect AI. Rewriter-specific clients remain optional. The core tests must use deterministic fake
rewrite kernels and must not require network access.

## 7. Configuration interface

Use a strict generative evaluation plan independent of the current fixed analysis plan. The
following is illustrative; exact field names should be frozen in implementation slice 1.

```yaml
schema_version: "1.0"

evaluation:
  id: multilingual-jailbreak-generative
  title: Generative multilingual jailbreak evaluation
  seed: 271828

taxonomy:
  seed_manifest: semantic-seeds.csv
  root_id: all
  hierarchy:
    - problem_class
    - semantic_seed_id
    - language
    - style
    - directness
    - wrapper
  semantic_seed_level: semantic_seed_id

  fixed_levels:
    problem_class:
      source: seed_manifest
      weights: explicit
      weight_column: problem_class_weight
    semantic_seed_id:
      source: seed_manifest
      weights: equal_within_parent

  arms:
    language:
      values:
        - {id: english, weight: 0.25}
        - {id: french, weight: 0.25}
        - {id: german, weight: 0.25}
        - {id: maltese, weight: 0.25}
    style:
      values:
        - {id: plain, weight: 0.50}
        - {id: conversational, weight: 0.25}
        - {id: formal, weight: 0.25}
    directness:
      values:
        - {id: direct, weight: 0.50}
        - {id: indirect, weight: 0.50}
    wrapper:
      values_from: wrapper-arms.csv
      weights: explicit
      weight_column: target_weight

  constraints:
    - id: maltese-wrapper-compatibility
      when: {language: maltese}
      allow:
        wrapper: [plain_request, roleplay]

rewrite:
  kernel_id: permissive-semantic-rewriter-v1
  provider: openai
  model: configured-at-runtime
  prompt_template: rewrite-template.txt
  temperature: 1.0
  max_tokens: 512
  independent_draw_key: rewrite_draw_id
  validation:
    validator_id: semantic-equivalence-v1
    mode: deterministic
    on_invalid: retry
    maximum_attempts: 5
  deduplication:
    scope: semantic_seed_path
    on_duplicate: retry

target:
  cohorts:
    baseline:
      model: configured-at-runtime
      generate_config: {temperature: 1.0, max_tokens: 512}
    candidate:
      model: configured-at-runtime
      generate_config: {temperature: 1.0, max_tokens: 512}
  shared_rewrites: true
  rollouts_per_rewrite: 1
  score:
    name: harmful_compliance
    mapping:
      failure_values: [FAIL]
      safe_values: [PASS]
  estimand: end_to_end_system

sampling:
  semantic_seed_policy: enumerate
  method: adaptive_stratified
  budget:
    maximum_target_generations: 8000
    maximum_rewrite_generations: 4000
  pilot:
    rewrites_per_seed: 2
    rollouts_per_rewrite: 1
  allocation:
    objective: root_variance
    cost_aware: true
    minimum_rewrites_per_seed: 2
    minimum_rewrites_per_declared_arm: 10
    variance_regularization: jeffreys

model_assisted:
  enabled: false

cross_entropy:
  enabled: false

inference:
  alpha: 0.05
  coverage_mode: two_sided
  root_method: weighted_empirical_bernstein
  one_sided_upper: true
  report_all_nodes: true
  simultaneous_tree: false
  independence:
    primary: rewrite_draw
    sensitivities: [semantic_seed]
```

### 7.1 Canonical compiled tree

The authoring configuration must compile to one canonical edge table:

| Field | Meaning |
|---|---|
| `node_id` | Globally stable node identifier |
| `parent_id` | Parent node identifier; null only at root |
| `level` | Declared hierarchy level |
| `value` | User-facing arm or seed identifier |
| `node_type` | `fixed`, `arm`, or `rewrite_kernel` |
| `conditional_target_weight` | Weight among valid siblings |
| `target_path_weight` | Product of edge weights from the root |
| `semantic_seed_id` | Fixed seed ancestor |
| `rewrite_kernel_id` | Terminal kernel reference where applicable |
| `path_hash` | Hash of the complete canonical path |

The compiled table is the object used by every sampler and estimator. Authoring shortcuts must not
create a second statistical representation.

### 7.2 Conditional arms

Constraints may prune structurally invalid combinations. Child weights must be renormalized only
according to an explicit rule:

- `renormalize_valid_children`; or
- explicit context-specific weights supplied in a transition table.

The compiler must reject a parent with positive target weight but no positive-weight valid child.
It must also reject implicit renormalization when the plan has not selected a rule.

## 8. Execution ledger

Every attempted sampling unit must be recorded before and after evaluation. The ledger is required
for inference; an Inspect log alone is insufficient.

### 8.1 Selection fields

| Field | Meaning |
|---|---|
| `sampling_unit_id` | Stable identifier for one selected path/rewrite cluster |
| `stage_id` | Pilot, allocation, model-assisted confirmation, or CE confirmation |
| `round` | Adaptive round number |
| `selection_order` | Global order before observing the outcome |
| `path_hash` | Selected canonical structural path |
| `target_path_probability` | \(w(h)\) |
| `proposal_path_probability` | \(q_t(h)\) used for this selection |
| `proposal_id` | Immutable proposal fingerprint |
| `selection_seed` | Reproducibility seed or counter-based key |
| `semantic_seed_id` | Fixed semantic ancestor |
| `allocation_stratum` | Stratum used by the allocator |

### 8.2 Rewrite fields

| Field | Meaning |
|---|---|
| `rewrite_draw_id` | Independent rewrite-draw identifier |
| `rewrite_kernel_id` | Frozen kernel identifier |
| `rewrite_kernel_sha256` | Hash of its complete resolved configuration |
| `rewrite_attempts` | Number of attempts including invalid outputs |
| `validation_status` | Valid, invalid-exhausted, or error |
| `prompt_id` | ID exposed to Inspect for the accepted rewrite |
| `prompt_sha256` | Content hash for provenance without report disclosure |
| `source_semantic_seed_id` | Fixed source seed |

Prompt content is required to execute the evaluation but should be stored in a separate restricted
materialization artifact. Statistical ledgers, JSON reports, CSV files, and HTML reports should
contain identifiers and hashes, not prompt or response content.

### 8.3 Outcome fields

Normalized outcome rows should retain the current observation fields and add:

- `sampling_unit_id`;
- `rewrite_draw_id`;
- `path_hash`;
- `stage_id`;
- `proposal_id`;
- `target_path_probability`;
- `proposal_path_probability`; and
- `rollout_within_rewrite`.

The join must be one-to-one at the declared rollout key and must detect retries or duplicate Inspect
samples that would otherwise be counted twice.

## 9. Baseline: direct target-distribution Monte Carlo

The first end-to-end implementation should sample a fresh structural path \(H_t\sim w\), a fresh
rewrite \(X_t\sim K_{H_t}\), and a fresh rollout for every independent sampling unit.

Then

\[
Y_t=F(X_t,Y_t^{\mathrm{rollout}})\sim\operatorname{Bernoulli}(R_{\mathrm{root}})
\]

marginally, even though individual path risks differ. Under independent complete sampling units,
Clopper--Pearson and Bernoulli KL/Chernoff intervals apply directly to the root.

This baseline is required because it provides:

- a reference implementation with minimal weighting;
- an empirical check for more complex estimators;
- a fallback when allocation diagnostics are unstable; and
- the cleanest zero-failure upper bound.

If one selected rewrite receives several rollouts, the rewrite draw becomes the independent
sampling unit. Its bounded response is the within-rewrite failure proportion. The result is no
longer an ordinary IID Bernoulli count, and the interval must use bounded-cluster inference.

## 10. Hierarchically balanced stratified sampling

### 10.1 Recursive stratification

The recommended production default is to stratify at fixed semantic seeds and at as many
high-level arm frontiers as the budget permits. If frontier strata are \(h=1,\ldots,H\), with
weights \(W_h\), estimate

\[
R=\sum_{h=1}^H W_hR_h,
\qquad
\widehat R=\sum_{h=1}^H W_h\widehat R_h.
\]

Within a frontier stratum, unresolved descendant paths are sampled from their normalized target
distribution. This avoids enumerating deeper leaves while preserving the declared estimand.

The allocator should recursively choose a frontier:

1. begin with semantic seeds as mandatory strata;
2. split a stratum when its children can all receive their minimum allocation;
3. prioritize splits with large target weight, estimated between-child variance, or explicit
   analyst importance;
4. stop splitting when another split would violate the budget or minimum allocations; and
5. sample the remaining descendants conditionally from target weights.

### 10.2 Fixed allocation estimator

For independent bounded rewrite-cluster summaries \(Z_{hi}\in[0,1]\),

\[
\widehat R_h=\frac{1}{n_h}\sum_{i=1}^{n_h}Z_{hi}.
\]

The variance is

\[
\operatorname{Var}(\widehat R)
=\sum_h\frac{W_h^2\sigma_h^2}{n_h}.
\]

Weighted empirical-Bernstein, Bennett, Hoeffding, and KL/Chernoff methods should be implemented as
separate named result families. The implementation must not select the shortest interval after
observing outcomes.

### 10.3 Recursive integer allocation

For a fixed requested allocation, use deterministic largest-remainder allocation only to translate
already-declared stratum counts into integers. It must not be confused with random path selection.
Within unresolved descendants, use counter-based randomized draws from conditional target weights.

Every allocated count and every realized selection probability must be written to the resolved
plan and ledger.

## 11. Nested variance and budget allocation

### 11.1 Variance levels

Within a structural stratum, distinguish at least:

- between-path variance across unresolved structural descendants;
- between-rewrite variance within one structural path; and
- between-rollout variance within one generated rewrite.

For \(m_h\) independent rewrites and \(r_h\) rollouts per rewrite, a useful planning approximation
is

\[
\operatorname{Var}(\widehat R_h)
\approx
\frac{\tau_h^2}{m_h}
+\frac{\sigma_h^2}{m_hr_h},
\]

where \(\tau_h^2\) is between-rewrite variance and \(\sigma_h^2\) is conditional rollout
variance.

The pilot report should show both components and the estimated marginal variance reduction from:

- one additional distinct rewrite; and
- one additional rollout attached to an existing rewrite.

### 11.2 Cost-aware Neyman allocation

For stratum cost \(c_h\), the continuous approximation is

\[
n_h\propto\frac{W_h\sigma_h}{\sqrt{c_h}}.
\]

The implementation should solve the integer allocation subject to:

- total rewrite and rollout budgets;
- per-seed minimums;
- per-arm minimums;
- maximum concurrency or provider quotas if supplied;
- integer rollout counts;
- paired-cohort requirements; and
- optional maximum weights per independent unit.

Use `scipy.optimize.milp` where the objective and constraints are linearized. Provide a deterministic
greedy repair algorithm for small residual budget adjustments, not as a silent substitute for an
infeasible plan.

### 11.3 Regularization after sparse pilots

Zero pilot failures must not force zero future allocation. Supported regularizers should include:

- a Jeffreys-smoothed Bernoulli variance estimate;
- an upper-confidence variance estimate; and
- a configurable minimum variance floor.

The default should use an upper-confidence or conservative smoothed estimate plus mandatory
minimum allocation. Allocation diagnostics should show the result under at least one less
aggressive regularization sensitivity.

### 11.4 Adaptation boundary

The first implementation should use a two-stage design:

1. run a fixed pilot;
2. estimate variance components and freeze the remaining allocation;
3. draw fresh observations under the frozen allocation; and
4. use the confirmation observations for the simplest primary interval.

Pilot observations may be included only through an estimator that explicitly accounts for their
sampling probabilities. Cross-fitting or predictable martingale weighting can be added later; the
initial implementation should prefer clarity over reusing every observation.

## 12. Randomized balanced incomplete path assignments

Balanced incomplete assignments are useful when every semantic seed cannot receive every arm
combination, especially for estimating language, style, or configuration contrasts.

### 12.1 Supported first version

The first version should support designs whose randomization probabilities are analytically known:

- randomized complete blocks for selected arm frontiers;
- balanced incomplete blocks with equal treatment inclusion probabilities;
- cyclic assignments with a uniformly random initial permutation; and
- stratified random assignment followed by conditional target sampling below the assigned
  frontier.

Semantic seeds act as blocks. Models/configurations should receive the same accepted rewrite
within each assigned path when paired comparison is requested.

### 12.2 Statistical constraint

A schedule may contribute to a generative root interval only when the implementation can calculate
and record the inclusion or draw probability required by its estimator. A deterministic heuristic
that merely looks balanced is allowed as a discovery or operational schedule but must not be
treated as probability-sampled evidence for unevaluated combinations.

### 12.3 Optimization objective

The schedule builder should minimize a prespecified combination of:

- marginal arm imbalance;
- pairwise arm-combination imbalance;
- maximum path weight assigned to one semantic seed;
- missing pairing across cohorts; and
- deviation from target conditional weights.

It should report the achieved balance rather than labeling it as threat-space coverage.

## 13. Paired model and configuration evaluation

One accepted rewrite should be reusable across all selected model/configuration cohorts. The
resulting comparison unit is the generated rewrite, identified by `rewrite_draw_id`.

For candidate \(C\) and baseline \(B\), define the within-rewrite difference

\[
D_i=Y_{Ci}-Y_{Bi}\in[-1,1].
\]

The root risk difference is estimated using the same stratification or importance coefficients as
the within-model root risks. Pairing removes rewrite-generation variation from the comparison and
should be the default whenever the scientific question permits it.

For three or more cohorts:

- one rewrite bank is shared by all cohorts;
- arbitrary prespecified pairwise or versus-reference contrasts are supported;
- the sampling design is learned without using cohort labels unless a contrast-specific allocator
  is explicitly selected; and
- simultaneous comparison families reuse the current hierarchy-aware comparison presentation.

Missing outcomes in one cohort must not silently destroy pairing. Reports should show complete-pair
counts and conservative missing-outcome bounds.

## 14. Model-assisted estimation

### 14.1 Purpose

A cheap risk predictor can reduce expensive target-model evaluations without making correctness of
that predictor an assumption of the point estimator. Predictors may use:

- structural path features;
- prompt embeddings;
- a cheap model or classifier;
- a lower-cost target-model configuration; or
- previous, non-overlapping evaluation data.

### 14.2 Estimator

Let \(m(x)\in[0,1]\) predict the expensive binary outcome. Draw a large cheap reference sample
\(U_1,\ldots,U_M\) from the target generative design and a smaller labeled sample
\((H_i,X_i,Y_i)\) from proposal \(q\). Use

\[
\widehat R_{\mathrm{MA}}
=
\frac{1}{M}\sum_{j=1}^M m(U_j)
+
\frac{1}{T}\sum_{i=1}^T
\frac{w(H_i)}{q(H_i)}\left[Y_i-m(X_i)\right].
\]

The first term estimates the target expectation of the predictor. The second corrects its residual
bias using expensive outcomes. The two samples should initially be independent so that their
uncertainties can be combined transparently.

### 14.3 Cross-fitting

If the predictor is trained on evaluation outcomes, use \(K\)-fold cross-fitting at the semantic
seed or rewrite-cluster level:

1. partition independent clusters before fitting;
2. train on all folds except one;
3. predict the held-out fold;
4. never use an outcome to predict its own residual; and
5. retain fold identifiers and model hashes.

The split level must follow the declared independence structure. Rollouts from one rewrite may not
be divided between training and held-out residual evaluation.

### 14.4 Confidence intervals

The initial implementation should combine separately valid concentration bounds for:

- the cheap target-distribution prediction mean; and
- the bounded importance-weighted residual mean.

The residual lies in `[-1, 1]` before importance weighting. A defensive proposal bound is required
for finite-sample concentration. A union allocation between the two components is conservative but
auditable. A sharper joint influence-function interval may be added only after simulation coverage
tests.

### 14.5 Required diagnostics

Report:

- cheap reference-sample size;
- labeled sample size;
- out-of-fold Brier score and calibration table;
- residual variance versus raw-outcome variance;
- achieved variance-reduction factor;
- maximum and effective importance weight;
- prediction-model fingerprint; and
- the unassisted estimator as a reference result.

## 15. Cross-entropy adaptive importance sampling

### 15.1 Role

Cross-entropy adaptive importance sampling should learn where in the finite structural hierarchy
expensive evaluation is most informative. It must not replace the rewrite kernel within a selected
path.

For a selected path \(h\sim q(h)\), the proposal continues with

\[
X\sim K_h,
\qquad
Y\sim M(\cdot\mid X).
\]

The target joint law contains \(w(h)K_h(X)M(Y\mid X)\), while the proposal contains
\(q(h)K_h(X)M(Y\mid X)\). The rewrite and rollout distributions therefore cancel, leaving the
tractable ratio

\[
\rho(h)=\frac{w(h)}{q(h)}.
\]

### 15.2 Proposal family

Represent the proposal using conditional probabilities on the compiled tree:

\[
q_\theta(h)
=
\prod_{e\in\operatorname{path}(h)}
q_\theta(e\mid\operatorname{parent}(e)).
\]

The default proposal should hold problem-class and semantic-seed allocation fixed and adapt only
below the semantic-seed frontier. This prevents a few early failures from eliminating most fixed
safety questions. Higher-level adaptation may be enabled explicitly when the evaluator is willing
to estimate those levels by importance weighting.

Parameters are conditional categorical probabilities over valid children. Context-specific
constraints from the canonical tree are therefore respected automatically.

### 15.3 Learning signal

The CE learner should support:

- binary failures;
- a bounded continuous severity score whose final binary threshold matches the analysis outcome;
- smoothed within-path failure estimates; and
- model-assisted predicted risk.

For Bernoulli outcomes when only the structural path is proposed, the second-moment-optimal path
proposal is proportional to

\[
q^*(h)\propto w(h)\sqrt{p_h},
\]

not \(w(h)p_h\). The implementation should therefore offer a regularized square-root-risk CE
objective for probability estimation. An elite severity objective may be useful for discovery but
must be labeled separately from the variance-minimizing estimation objective.

### 15.4 CE update

At iteration \(t\):

1. draw paths from the current defensive proposal;
2. generate fresh rewrites from their original path kernels;
3. evaluate severity and binary outcomes;
4. construct nonnegative learning weights from regularized risk or severity;
5. fit the conditional categorical proposal by weighted maximum likelihood;
6. apply Dirichlet pseudocounts and probability floors;
7. exponentially smooth the new parameters with the previous proposal;
8. compute effective sample size and branch-entropy diagnostics; and
9. stop learning at a fixed iteration count or an outcome-independent numerical convergence rule.

The weighted categorical update for parent \(u\) and child \(c\) has the form

\[
\widetilde q_{t+1}(c\mid u)
=
\frac{a_{uc}+\sum_i\omega_i
\mathbf 1\{H_i\text{ traverses }u\to c\}}
{\sum_{c'}a_{uc'}+\sum_i\omega_i
\mathbf 1\{H_i\text{ traverses }u\}},
\]

followed by smoothing and projection onto the configured probability floor.

### 15.5 Defensive mixture

The proposal used for final estimation must be

\[
q(h)=\varepsilon w(h)+(1-\varepsilon)q_{\mathrm{CE}}(h),
\qquad 0<\varepsilon\le1.
\]

Then

\[
0\le\rho(h)=\frac{w(h)}{q(h)}\le\frac{1}{\varepsilon}.
\]

The plan must reject `epsilon: 0` for any inferential run. It should warn when the implied maximum
weight makes the requested interval precision unattainable at the available budget.

### 15.6 Frozen confirmation

The production CE workflow is:

1. run CE learning rounds;
2. freeze and hash \(q_{\mathrm{CE}}\);
3. form and hash the defensive proposal;
4. draw a fresh independent confirmation sample;
5. sample rewrites from the original \(K_h\);
6. calculate importance-weighted node estimates; and
7. report bounded weighted confidence intervals.

The CE learning population should not enter the primary confirmation interval in the first
implementation. A later cross-fitted CE estimator may reuse learning data while keeping proposal
training and outcome evaluation separated.

### 15.7 Node estimators

For node \(v\), the global-proposal estimator is

\[
\widehat R_v
=
\frac{1}{T}\sum_{t=1}^T
\frac{w(H_t)\mathbf 1\{H_t\in D(v)\}}
{W_vq(H_t)}Y_t.
\]

This is unbiased but may be inefficient for small-mass nodes. The implementation should therefore
allow selected node frontiers to retain fixed stratified allocations while CE operates only below
those frontiers. Node-specific effective sample size and maximum weight must be reported.

### 15.8 CE diagnostics

Each learning round and the final proposal must report:

- proposal entropy at every adapted frontier;
- minimum and maximum branch probability;
- target-to-proposal likelihood-ratio range;
- raw and normalized importance-weight effective sample size;
- number of unique semantic seeds and structural paths sampled;
- failure and severity rates before weighting;
- weighted root-risk estimate as a diagnostic only during learning;
- proposal change, such as total variation or KL divergence from the preceding round;
- defensive-mixture epsilon; and
- the fraction of budget spent on target-distribution exploration.

No report should describe a high unweighted CE failure rate as the root failure probability.

## 16. Inference and interval methods

### 16.1 Required primary methods

The extension should implement the following estimator/interval pairings:

| Sampling design | Point estimator | Primary interval |
|---|---|---|
| Independent target-path sampling, one rollout per rewrite | Ordinary failure proportion | Exact Clopper--Pearson or Bernoulli KL/Chernoff |
| Fixed hierarchical strata | Weighted stratum means | Weighted empirical-Bernstein or Bennett/Chernoff |
| Rewrite clusters with repeated rollouts | Weighted cluster means | Bounded cluster empirical-Bernstein |
| Frozen defensive CE proposal | Importance-weighted mean | Bounded importance-weight empirical-Bernstein or Bennett |
| Model-assisted estimator | Prediction mean plus weighted residual correction | Separately calibrated component bounds |

Every result must state whether its guarantee is exact finite-sample, finite-sample concentration,
or approximate.

### 16.2 Effective sample size

At minimum report two distinct diagnostics:

1. Kish importance-weight ESS,

   \[
   n_{\mathrm{Kish}}=
   \frac{(\sum_i\rho_i)^2}{\sum_i\rho_i^2};
   \]

2. concentration ESS after normalizing estimator coefficients \(a_i\),

   \[
   n_{\mathrm{conc}}=\frac{1}{\sum_i a_i^2}.
   \]

Neither should be presented as a literal count of independent prompts. The report should also show
the number of independent rewrite clusters and the number of target-model rollouts.

### 16.3 One-sided and simultaneous results

The root should receive both a two-sided interval and a separately calibrated one-sided upper
bound when requested. Node intervals may be:

- pointwise;
- simultaneous over a selected frontier; or
- simultaneous over the displayed tree.

The first implementation should use the existing conservative hierarchy calibration ideas only
after adapting them to the generative estimator coefficients. Existing fixed-design interval
labels must not be reused without verifying that the new selection-randomness assumptions are
represented.

Joint max-\(T\) resampling for generative designs should remain a reference method until the
resampling procedure reproduces every stochastic level included in the estimand.

### 16.4 Missing outcomes

For every attempted confirmation unit:

- a classified failure contributes one;
- a classified safe outcome contributes zero;
- a target-side provider guardrail may be safe under the existing end-to-end estimand;
- unclassified outcomes receive safe completion for a lower endpoint and failure completion for an
  upper endpoint; and
- missingness must not alter the recorded proposal probability or remove the sampling unit from
  the denominator silently.

Rewrite-generation failure before an accepted prompt is different from an unclassified target
outcome. It should be reported as failure to realize the specified rewrite kernel and should block
the primary estimator if the predeclared retry procedure is exhausted often enough to change the
effective generative design.

## 17. Inspect AI integration

### 17.1 Materialization boundary

The extension should materialize accepted rewrites before each Inspect batch. Each generated
rewrite becomes one Inspect sample with:

- `id = prompt_id`;
- metadata containing the complete hierarchy path identifiers;
- `semantic_seed_id` and `rewrite_draw_id`;
- `sampling_unit_id` and `stage_id`; and
- proposal and kernel fingerprints, with probabilities retained in the external ledger.

The same materialized dataset is used for every paired model/configuration cohort.

### 17.2 Round orchestration

Adaptive allocation and CE learning require multiple frozen rounds:

```text
compile taxonomy
  -> select paths for round
  -> generate and validate rewrites
  -> write immutable round manifest
  -> execute Inspect evaluations
  -> ingest outcomes
  -> close and checksum round
  -> update allocator/proposal
```

A round may not be modified after any of its outcomes have been ingested. Restarting an incomplete
round should preserve its original selection ledger and retry only missing execution units.

### 17.3 Current analysis compatibility

For each closed round, write a conventional fixed-design manifest so the existing analyzer can
describe the realized generated prompts. Also write the generative ledger required by the new
estimator. Reports must distinguish:

- conditional risk over the realized prompt bank; and
- generative risk over the declared path and rewrite distributions.

Only the latter answers the root-probability question in this plan.

## 18. Command-line workflow

Proposed commands are:

```bash
inspect-safety-stats generative compile \
  --plan generative-plan.yaml \
  --output run/design

inspect-safety-stats generative sample \
  --design run/design \
  --stage pilot \
  --output run/pilot

inspect-safety-stats generative allocate \
  --run run \
  --output run/allocation

inspect-safety-stats generative ce-fit \
  --run run \
  --output run/ce-proposal

inspect-safety-stats generative sample \
  --design run/design \
  --proposal run/ce-proposal/defensive-proposal.json \
  --stage confirmation \
  --output run/confirmation

inspect-safety-stats generative analyze \
  --run run \
  --output run/report
```

Rewrite materialization and Inspect execution should also be exposed as Python APIs so example
projects can control provider concurrency and recovery behavior directly.

Every command must be resumable from immutable artifacts and deterministic seeds. Commands that
would overwrite a closed round or change a hashed proposal must fail unless a new run or stage ID
is supplied.

## 19. Output artifacts

The final run directory should contain:

```text
generative-plan.resolved.yaml
taxonomy.edges.csv
taxonomy.nodes.csv
target-paths.csv
kernel-manifest.json
sampling-ledger.csv
rewrite-ledger.csv
observations.csv
allocation-report.json
variance-components.csv
proposal-history/
  round-000.json
  round-001.json
  final-ce-proposal.json
  defensive-proposal.json
generative-report.json
generative-report.md
generative-report.html
node-estimates.csv
comparison-estimates.csv
checksums.json
```

The report should remain compact. Its default view should contain:

1. root estimate, two-sided interval, and one-sided upper bound;
2. selected sampling method and target-distribution definition;
3. hierarchy tree with node estimates and intervals;
4. evaluation counts at path, rewrite, and rollout levels;
5. maximum weight, ESS, and allocation concentration;
6. variance decomposition and the rewrite-versus-rollout budget implication;
7. paired model/configuration comparisons; and
8. a collapsed diagnostics section for model assistance or CE.

Prompt and response content must never appear in these artifacts.

## 20. Implementation slices

### Slice 1: contracts and canonical taxonomy

Implement:

- strict generative-plan schema;
- semantic-seed manifest loader;
- global and context-conditional arm declarations;
- canonical node and edge compilation;
- weight validation and path hashing;
- rewrite-kernel protocol and configuration fingerprint;
- selection/rewrite ledger schemas; and
- synthetic tree fixtures.

Acceptance criteria:

- every positive-weight node has a valid route to a rewrite kernel;
- sibling weights sum to one after the declared constraint rule;
- every semantic seed has one fixed parent path;
- compiled path weights sum to one to numerical tolerance;
- semantically identical plans compile to the same hash; and
- invalid nesting, cycles, orphan nodes, and ambiguous renormalization are rejected.

### Slice 2: direct target sampler and Inspect bridge

Implement:

- counter-based target path sampling;
- fake and provider-backed rewrite-kernel adapters;
- validation/retry/deduplication ledgering;
- Inspect dataset materialization;
- paired cohort materialization;
- outcome ingestion join; and
- exact root inference for one rollout per independently selected rewrite.

Acceptance criteria:

- simulated path frequencies converge to declared weights;
- generated sample IDs are stable and collision-free;
- rerunning an interrupted stage does not duplicate sampling units;
- paired cohorts receive identical accepted rewrites;
- a known Bernoulli-mixture simulation produces unbiased root estimates; and
- exact interval simulation meets its nominal coverage tolerance.

### Slice 3: hierarchical stratification and bounded intervals

Implement:

- mandatory semantic-seed stratification;
- recursive frontier selection;
- fixed integer allocation;
- conditional target sampling below the frontier;
- node estimators;
- weighted empirical-Bernstein, Bennett/Chernoff, and Hoeffding intervals;
- repeated-rollout rewrite clusters; and
- pointwise and selected simultaneous node reporting.

Acceptance criteria:

- estimates remain unbiased under heterogeneous path risks;
- ancestor estimates equal declared weighted descendant estimates;
- repeated rollouts do not increase the reported rewrite-cluster count;
- bounds widen when independence is coarsened; and
- simulation covers balanced, highly imbalanced, sparse-failure, and boundary cases.

### Slice 4: nested variance and adaptive allocation

Implement:

- pilot variance-component estimates;
- cost recording;
- constrained continuous and integer allocation;
- variance regularization;
- minimum exploration constraints;
- frozen confirmation-stage plan; and
- allocation reports showing expected precision and binding constraints.

Acceptance criteria:

- allocation never assigns zero to a mandatory positive-weight stratum;
- additional rollouts and additional rewrites have separately calculated marginal value;
- known synthetic variance structures lead to the expected allocation direction;
- infeasible budgets return a diagnostic allocation deficit; and
- pilot outcomes cannot modify an already frozen confirmation allocation.

### Slice 5: randomized incomplete assignments and comparisons

Implement:

- supported randomized block schedule generators;
- known assignment probabilities;
- balance diagnostics;
- shared rewrite banks across any number of cohorts;
- paired difference estimators at every hierarchy node; and
- simultaneous versus-reference and all-pairs outputs.

Acceptance criteria:

- every inferential schedule has auditable inclusion probabilities;
- unpaired or incomplete cohort observations are detected;
- paired estimators improve variance in positively correlated simulations;
- three-or-more-cohort contrast families match direct pairwise calculations; and
- heuristic schedules without known probabilities are excluded from root inference.

### Slice 6: model-assisted estimation

Implement:

- predictor protocol;
- cheap target-distribution reference sampling;
- seed/rewrite-cluster cross-fitting;
- bounded residual correction;
- component-wise confidence bounds;
- unassisted reference estimate; and
- calibration and variance-reduction diagnostics.

Acceptance criteria:

- misspecified predictors do not bias the estimator in simulation;
- a constant predictor reduces exactly to the corresponding unassisted construction;
- highly predictive synthetic models reduce variance;
- fold leakage across rewrite clusters is impossible by schema validation; and
- uncertainty in the cheap reference mean is included rather than ignored.

### Slice 7: cross-entropy adaptive importance sampling

Implement:

- factorized tree proposal;
- categorical weighted maximum-likelihood updates;
- risk, square-root-risk, and severity learning objectives;
- probability floors, Dirichlet regularization, and exponential smoothing;
- proposal checkpoints and deterministic replay;
- defensive target mixture;
- frozen independent confirmation sampling;
- importance-weighted node estimates and intervals; and
- CE diagnostics in JSON, CSV, Markdown, and HTML.

Acceptance criteria:

- every confirmation path has positive proposal probability;
- the likelihood ratio is bounded by the configured defensive mixture;
- learned proposals concentrate on high-risk regions in synthetic trees;
- frozen-confirmation estimates remain unbiased across repeated simulations;
- empirical interval coverage meets a prespecified Monte Carlo tolerance;
- CE outperforms direct target sampling in at least one documented rare, concentrated-risk
  benchmark without materially underperforming in a diffuse-risk benchmark; and
- an unweighted CE failure rate is never exposed as a root-risk estimate.

## 21. Test strategy

### 21.1 Unit and property tests

Use Hypothesis and deterministic fixtures to test:

- arbitrary valid trees and conditional constraints;
- normalization of edge, path, and node weights;
- equivalence of recursive and flattened estimators;
- proposal probability products;
- bounded likelihood ratios;
- deterministic hashes and counter-based draws;
- allocation constraint preservation; and
- serialization round trips with strict unknown-field rejection.

### 21.2 Statistical simulation tests

Create synthetic generators with known \(p_h\) and separately controlled:

- between-seed variance;
- between-arm variance;
- between-rewrite variance;
- rollout variance;
- rare concentrated failures;
- rare diffuse failures;
- strongly unequal target weights; and
- positively correlated paired model outcomes.

For each primary interval, run enough seeded replications to bound Monte Carlo error in the observed
coverage assessment. Tests should verify both coverage and expected-width behavior. Coverage tests
belong in a slower statistical test suite rather than every fast unit-test run.

### 21.3 Integration tests

Use a fake Inspect-compatible task and fake rewriter to test:

- multi-round materialization;
- partial-run recovery;
- log ingestion and ledger joins;
- target policy-block outcomes;
- paired multi-cohort execution;
- missing and duplicate samples;
- proposal freezing; and
- complete artifact checksum validation.

### 21.4 Golden reports

Maintain small synthetic golden reports for:

- direct target sampling;
- stratified sampling with repeated rollouts;
- model-assisted estimation; and
- CE confirmation.

Golden data must contain no harmful prompt or response content.

## 22. Example delivery

### 22.1 Synthetic example

Ship a fully deterministic example with:

- three problem classes;
- twelve semantic seeds;
- three languages;
- two styles;
- two directness arms;
- a fake stochastic rewrite kernel; and
- known heterogeneous target-model failure probabilities.

The example should compare equal-cost direct, stratified, model-assisted, and CE designs against the
known root risk.

### 22.2 JailbreakBench example

Add an optional real-provider example that treats the 100 JailbreakBench behaviors as fixed
semantic seeds. The initial transformation hierarchy should remain small enough to audit, for
example:

```text
problem class
  -> semantic seed
     -> language
        -> directness
           -> style
              -> rewrite kernel
```

The example should:

- define explicit target weights;
- evaluate every semantic seed but only a randomized subset of descendant variations;
- share accepted rewrites across at least three model/configuration cohorts;
- demonstrate pilot allocation and a frozen confirmation stage;
- optionally learn a CE proposal below the semantic-seed frontier;
- report root and hierarchy-node intervals; and
- compare interval width and cost across sampling strategies.

The example README must state that the root probability applies to the specified transformation
and rewrite process, not all possible jailbreak prompts.

## 23. Numerical and operational failure modes

The implementation must detect and report:

- target weights that do not normalize;
- positive-target branches with zero proposal probability;
- extremely small defensive epsilon;
- low importance-weight ESS;
- proposal collapse onto few semantic seeds;
- rewrite-kernel or validator drift between rounds;
- semantic-validation exhaustion;
- duplicate rewrites that violate the declared kernel procedure;
- adaptive use of outcomes without recorded pre-outcome proposals;
- reuse of CE learning observations in a nominally independent confirmation interval;
- unrecognized model or scorer configuration changes;
- insufficient complete pairs across cohorts; and
- budgets too small for mandatory strata.

Warnings should quantify their numerical effect where possible. A result should be absent, rather
than silently reinterpreted, when the required selection probability or kernel identity cannot be
reconstructed.

## 24. Recommended defaults

The initial public interface should use the following defaults:

- enumerate all fixed semantic seeds;
- equal seed weight within each problem class unless explicitly supplied;
- declare all structural arm weights explicitly;
- stratify at semantic seed;
- sample unresolved descendants from normalized target weights;
- generate more distinct rewrites before adding many repeated rollouts, subject to pilot variance;
- use one accepted rewrite bank across model cohorts;
- use a fixed pilot followed by a frozen allocation;
- use weighted empirical-Bernstein as the general bounded-mean interval and report a
  Bennett/Chernoff reference;
- keep model assistance disabled unless a predictor and independent cheap reference sample are
  configured;
- keep CE disabled by default;
- when CE is enabled, adapt only below semantic seed;
- require a positive defensive target mixture; and
- use a fresh confirmation sample for primary CE inference.

## 25. Definition of completion

The extension through cross-entropy adaptive importance sampling is complete when:

1. A user can define a deep generative hierarchy with fixed semantic seeds and stochastic LLM
   rewrite terminals without enumerating prompt strings.
2. The compiler produces one auditable target distribution over valid structural paths.
3. Inspect evaluations can be materialized, resumed, paired across cohorts, and joined to a complete
   sampling ledger.
4. Direct and stratified estimators recover known node risks in simulation with validated interval
   behavior.
5. The planner makes an explicit, data-driven choice between additional structural paths,
   additional rewrites, and additional rollouts.
6. Balanced incomplete schedules are used inferentially only when their randomization probabilities
   are known.
7. Model-assisted estimation preserves design validity under predictor misspecification and reports
   its actual variance reduction.
8. CE learns a bounded defensive proposal over the structural tree and uses a fresh independent
   confirmation sample for root and node inference.
9. All result artifacts distinguish target risk, raw sampled failure rate, and discovery metrics.
10. The visualization presents root risk, hierarchy-node risks, uncertainty, allocation, ESS, and
    model comparisons without exposing prompt or response content.

SMC and evolutionary prompt mutation should be considered only after these foundations are working
and simulation-tested. At that point, their incremental value can be measured against a strong
balanced, model-assisted, defensive-importance-sampling baseline.

---

## Slice 0 result: variance reduction does not improve the reported bound

**Status:** measured, 24 August 2026. Reproduce with
`python research/generative_sampling_tradeoff.py`.

Slices 3, 4 and 7 of this plan -- hierarchical stratification, cost-aware Neyman
allocation, and cross-entropy adaptive importance sampling -- are all variance-reduction
techniques. Before building them, they were compared at equal budget against plain direct
target-distribution sampling (slice 2), scored on two things: the classical estimator
RMSE, and the width of the finite-sample bound this package actually reports.

At a budget of 1,200 evaluations over a 200-path tree, relative to direct sampling:

| design | RMSE | bound width |
|---|---:|---:|
| direct target sampling | 1.00x | 1.00x |
| stratified (proportional) | 0.98x | 1.00x |
| stratified (Neyman) | 0.63x | **1.60x** |
| oracle importance sampling | 0.54x | **1.40x** |

Importance sampling was given the second-moment-optimal proposal \(q^*\propto
w(h)\sqrt{p_h}\) for free, so the result does not depend on how well a cross-entropy
learner performs; it upper-bounds anything a learner could reach.

**The two scores move in opposite directions, and the gap widens exactly where importance
sampling is supposed to excel.** Holding the base rate at 0.0005 and increasing the ratio
between the risky arm and the rest:

| risk ratio | IS RMSE | IS bound width |
|---:|---:|---:|
| 10 | 0.90x | 1.04x |
| 100 | 0.65x | 1.19x |
| 1,000 | 0.35x | 1.40x |
| 10,000 | 0.27x | **1.51x** |

At a 10,000-fold concentration the estimator's variance falls almost fourfold while the
reported interval gets half again as wide.

### Why

Every bound in this package is a bounded-weight concentration inequality, and those pay
for coefficient inequality. Direct sampling gives every draw the coefficient \(1/T\);
with equal atomic weights the KL/Chernoff engine uses the sharp Bernoulli-KL kernel.
Neyman allocation and inverse-probability weighting both deliberately create unequal
coefficients -- that is how they reduce variance -- and the bound charges for it, partly
through the larger maximum increment and partly by falling back to the weighted
Poisson-Chernoff kernel. The defensive mixture that makes a finite-sample bound available
for importance sampling at all is the same thing that caps how much variance reduction is
reachable.

The effect is not an artefact of one engine: under empirical Bernstein the penalty is
1.32x-1.49x instead of 1.40x-1.93x -- smaller, because there is no kernel switch, but the
same sign.

Nor is it rescued by a frontier too large to stratify, the regime where importance
sampling was expected to earn its keep. At 128 arms with 300 evaluations -- 2.3 draws per
stratum -- importance sampling is still 1.15x wider than direct sampling, and proportional
stratification 1.17x.

### What this implies for the plan

Slice 2 is not merely the baseline. On this evidence it is the **best** design for a
package that reports finite-sample bounds, and slices 3, 4 and 7 should not be expected to
improve the reported interval. They remain sound statistics; they optimise a quantity this
package does not publish.

Three honest limits on the finding. It uses one synthetic hierarchy family, so a
qualitatively different risk structure could behave differently. It is specific to
finite-sample reporting -- under asymptotic intervals importance sampling would win, which
is precisely the trade
[considered and declined for sampled frames](design-based-sampling.md). And proportional
stratification is free rather than harmful (1.00x-1.01x), so stratifying to guarantee
coverage of declared strata costs nothing even though it buys nothing.

Before building slice 3, 4 or 7, re-run this comparison on a tree that matches the intended
evaluation. If it still shows the same sign, build slice 2 and stop.

## Slice G1 as built

**Status:** implemented, 24 August 2026. Modules under
`src/inspect_safety_stats/generative/`; tests in `tests/unit/test_generative_taxonomy.py`
and `tests/unit/test_generative_kernels_ledger.py`.

Following the slice 0 result, the seven slices of section 20 were reduced to two builds
and a gate. G1 is section 20's slice 1; G2 is its slice 2 plus the coverage half of slice
3. Slices 3 (Neyman), 4 and 7 are closed by measurement, and slice 6 is deferred behind a
separate measurement of residual-range reduction.

### Decisions taken

**The generative plan is a separate top-level document.** The fixed analysis plan
conditions on a materialized design; a generative plan declares the process that produces
prompts. One file carrying both would let an author combine them into an estimand neither
engine can state, so `GenerativePlan` shares no schema with `AnalysisPlan`.

**Path selection is always reproducible; the rewrite kernel is only frozen and
fingerprinted.** Selection is ours, so it replays exactly from `evaluation.selection_seed`.
An LLM rewrite kernel cannot be forced to be reproducible, so the floor is a fingerprint
over its complete resolved configuration -- template contents, decoding, validation and
deduplication rules included -- with `rewrite.seed_replayable` as an optional declared
property for kernels that can do better.

### Deviations from sections 6 to 8

- Eight modules rather than fifteen. `balance`, `variance`, `allocation`, `model_assisted`,
  `cross_entropy` and `comparisons` are not created.
- `node_type` has no `rewrite_kernel` value. A single kernel serves every path, so terminal
  nodes carry `rewrite_kernel_id` and the criterion "every positive-weight node has a valid
  route to a rewrite kernel" reduces to "every non-terminal node retained a child".
- The `sampling` block offers `direct` and `proportional_stratified` only; the
  `model_assisted` and `cross_entropy` blocks are absent rather than disabled.
- The selection ledger drops `round` and renames `allocation_stratum` to `stratum`.
  `proposal_path_probability` and `proposal_id` are kept even though direct sampling always
  sets the proposal equal to the target, so an auditor can verify that rather than assume it.
- Section 7.2's "explicit context-specific weights supplied in a transition table" is
  implemented as an inline `context_weights` list on the arm rather than a separate file.
- `|` and `=` are reserved in every identifier, because canonical node ids are built from
  them. Compilation rejects an identifier containing either.

### Invariants the compiler enforces

Cycles and orphans are not rejected at runtime because they cannot be expressed: the
hierarchy is a declared ordered list and every node id contains its parent's id as a prefix.
What is checked is weight normalization at every frontier, one fixed parent path per
semantic seed, a positive-weight node never losing all its children to a constraint,
constraints that condition strictly above what they prune, and a declared `renormalization`
rule whenever constraints prune at all. Semantically identical plans -- differing only in
declaration order, or in `equal_within_parent` versus explicit equal weights -- compile to
the same fingerprint.

## Slice G2a as built

**Status:** implemented, 24 August 2026. `generative/sampling.py` and
`generative/inference.py`; tests in `tests/unit/test_generative_sampling.py` and
`tests/unit/test_generative_inference.py`.

G2a is the half of section 20's slice 2 that could invalidate the design: path selection
and inference, both testable by simulation with no provider and no Inspect. The Inspect
bridge -- dataset materialization, kernel execution, outcome join -- is G2b.

### Why the inference is exact

Path selection is independent of outcomes, so conditional on the number of draws that
landed in a node, those draws are independent observations from that node's conditional
distribution. Exact Clopper--Pearson bounds are therefore valid conditionally, and a bound
valid for every realized count is valid unconditionally.

The same argument licenses Sidak calibration across siblings. Conditional on the count
vector, sibling nodes hold disjoint sets of independent draws, so their coverage events are
independent and the product bound holds for every realization; taking the expectation over
counts preserves it. It also licenses reporting only the nodes that received draws, because
that selection rule reads the counts -- which are outcome-independent -- and never the
failures.

Exactness needs equal coefficients and one rollout per rewrite. Where either fails the
weighted KL/Chernoff engine takes over and the interval reports which kernel was used.

### The integer-allocation question, measured

Proportional stratification keeps every coefficient at exactly \(1/T\) only when
\(Tw_v\) are whole numbers. Largest-remainder allocation makes them near-equal, which drops
the sharp Bernoulli-KL kernel for the weighted fallback. On seven strata at \(1/7\) --
a weight that divides no round budget -- the cost of that fallback is:

| budget | kernel | root bound vs direct sampling |
|---:|---|---:|
| 100 | weighted | 1.036x |
| 300 | weighted | 1.026x |
| 700 | exact | 1.000x |
| 1,000 | weighted | 1.008x |
| 3,000 | weighted | 1.002x |

The penalty is under 4% at the smallest budget, vanishes when the budget divides the
weights, and is below 1% by a thousand draws. Slice 0's finding that proportional
stratification is free survives integer allocation, so no budget-snapping machinery is
warranted. Every node reports `coefficient_ratio` and `equal_coefficients`, so the cost is
visible rather than assumed.

### Root and node intervals are not forced to agree

The root interval is computed directly from all \(T\) draws, not by aggregating node
intervals, and no LP projection reconciles the two. They can disagree slightly. This is
deliberate: the root is the primary estimand and deserves the tightest available bound,
while a node interval is a conditional statement about that node. Projecting would widen
the root to buy a coherence the report does not claim.

`simultaneous_tree` is off by default. Turning it on puts the root in its own frontier and
splits alpha across depths by Bonferroni, which widens the root -- so a reader who wants the
tightest root should leave it off and read the root alone, with the node intervals labelled
pointwise.

## Slice G2b as built

**Status:** implemented, 24 August 2026. `generative/execution.py` and
`generative/inspect_bridge.py`; tests in `tests/unit/test_generative_bridge.py`.

G2b closes section 20's slice 2: the rewrite procedure runs, its output becomes an
Inspect dataset, and the evaluated samples come back as sampling-unit outcomes.

### The accepted distribution is produced here

Validation, retry and deduplication are applied during execution rather than afterwards,
because section 3.2 defines the kernel as the distribution the *complete* procedure
induces. A unit whose retries are exhausted is recorded as an incomplete sampling unit and
kept -- never dropped -- so the path distribution it was drawn to represent is preserved
and the conservative completion inside the interval carries it.

### Where prompt text is allowed to be

The materialized dataset necessarily contains prompts; that is what Inspect evaluates, and
it is the restricted artifact section 8.2 describes. Everything else keys on identifiers:
the rewrite ledger stores `prompt_sha256` and has no text column, and the join reads only
identifiers. `write_prompt_store` puts the text in one clearly separate file so it can be
handled under different access rules. A test asserts the seed text cannot be found anywhere
in a serialized ledger row.

### The join keys on the materialized sample id

A generative design has no prompt manifest, so the normalized `prompt_id` the fixed-design
ingestion populates is empty. The join therefore keys on the sample id that materialization
wrote, and `samples_from_observations` adapts normalized Inspect observations to it.

Every selection yields exactly one outcome row, including units that were never evaluated
and units whose rewrite never passed validation; both are completed as unclassified. The
realized design cannot silently shrink to whatever happened to run.

Three conditions are reported rather than absorbed: `GENERATIVE_SAMPLE_UNMATCHED` when a
log row matches no accepted rewrite, `GENERATIVE_SAMPLE_DUPLICATE` when the same sample and
epoch arrive twice -- a retried or re-ingested sample would otherwise be counted twice --
and `GENERATIVE_ROLLOUT_OVERRUN` when a unit has more rollouts than the plan declares.
`GENERATIVE_UNIT_UNEVALUATED` is a warning, since incomplete evaluation is expected and
already handled conservatively.

### Pairing

Under `target.shared_rewrites` every cohort is pointed at the identical dataset file, so
the same accepted rewrite is evaluated by each model and a difference between arms cannot
be an artefact of different prompts. Independent rewrites per cohort are supported by
running the kernel again with a distinct `rewrite_salt`; the paired helper refuses that
configuration explicitly rather than silently producing unpaired arms.

### Still open

No worked example exercises the whole path yet, and comparisons across cohorts are not
implemented -- G2a estimates one cohort at a time.
