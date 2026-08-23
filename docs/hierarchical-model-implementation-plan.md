# Hierarchical safety-landscape model: assessment and implementation plan

**Status:** first vertical slice implemented in package version 0.5.0; later stages remain planned  
**Target package:** `inspect-safety-stats`  
**Relationship to the fixed-design analysis:** optional, model-based, and non-certifying

## 1. Outcome

The proposed hierarchical model is a valuable extension of the package, provided it remains a
separate structural analysis rather than becoming a replacement confidence interval for the
fixed-design risk estimates.

The extension should answer:

- how heterogeneous the prompt-level failure probabilities appear to be;
- whether risk is diffuse or concentrated in a small part of the declared design;
- which declared descriptors are associated with elevated risk;
- whether selected descriptor effects are materially non-separable;
- how much prompt-level variation remains unexplained by the declared taxonomy; and
- which evaluated prompts appear unusually vulnerable relative to their descriptors.

It must not answer:

- whether the model is safe beyond the fixed evaluation design;
- how common the evaluated prompt types are in deployment;
- whether the descriptor taxonomy covers all relevant hazards;
- whether a scorer or judge is correct; or
- whether a Bayesian credible interval is equivalent to a finite-sample confidence bound.

The implementation should therefore preserve two visibly different result families:

| Layer | Primary object | Uncertainty | Permitted role |
|---|---|---|---|
| Empirical fixed-design analysis | Declared functions of the finite vector of prompt failure probabilities | Exact/concentration confidence bounds under stated rollout assumptions | Primary risk reporting and, where enabled, certification |
| Structural model | Posterior distribution of latent prompt risks and descriptor effects | Bayesian credible distributions conditional on the model, priors, and descriptor space | Explanation, diagnosis, prioritization, and sensitivity analysis |

## 2. Assessment of the proposed statistical model

### 2.1 Elements that should be retained

The following parts of the specification are statistically and architecturally well aligned with
the package:

1. **Prompt-level binomial likelihood.** Aggregating identical-condition rollouts to
   `(k_unsafe, n)` preserves the likelihood while avoiding unnecessary sampler cost.
2. **Complementary-log-log default link.** This supplies a useful multiplicative
   unsafe-intensity interpretation and behaves approximately log-linearly for rare failures.
3. **Logit sensitivity fit.** Conclusions that depend heavily on the link should be shown as
   model uncertainty.
4. **Partial pooling across a declared taxonomy.** Sparse categories should borrow information
   only through relationships explicitly supplied by the evaluation creator.
5. **Regularized, predeclared interactions.** Task-by-attack and selected varying effects are more
   interpretable than unrestricted interaction discovery.
6. **Prompt-specific frailty.** A residual prompt effect directly addresses unexplained
   vulnerability after accounting for known descriptors.
7. **Heavy-tail sensitivity.** Comparing Student-t and Gaussian prompt effects tests whether tail
   conclusions depend on the frailty distribution.
8. **Poststratification with fixed-design weights.** The likelihood describes the observed
   rollouts; declared weights define summaries over the fixed design.
9. **Prompt-level validation.** Replicates from one prompt must never be split across training and
   validation folds.
10. **Posterior predictive checks and synthetic recovery tests.** Aggregate fit alone is
    insufficient for a model whose purpose is to characterize heterogeneity and tails.

### 2.2 Clarifications required before production use

The implementation should make the following points more explicit than the source specification.

#### Fixed prompts and exchangeability

The prompts remain fixed design items. A hierarchical prior over prompt effects is a regularizing
model for the finite collection; it does not turn those prompts into an IID sample from an
unobserved prompt population. Model-derived quantities therefore refer to the declared weighted
design unless a future, separately justified sampling design is introduced.

#### Missing or unclassified outcomes

The model must not silently discard unclassified rollouts. Version 1 should fit only when all
attempted outcomes used by the model are classified, including target-side policy blocks that are
classified safe under the selected end-to-end estimand.

If unclassified outcomes remain, the structural fit should be blocked by default. A later
diagnostic mode may fit explicit all-safe and all-failure completion sensitivities, but neither fit
should be presented as certifying evidence.

#### Conditional independence

The binomial likelihood assumes exchangeable, conditionally independent rollouts within a prompt
under an identical model and generation configuration. The product likelihood also requires the
declared rollout processes not to share unmodelled stochastic shocks. Configuration fingerprints,
seeds, caches, batch effects, tool environments, and evaluator versions must be checked or
documented where relevant.

#### Taxonomy validation

A hierarchy such as `class -> task -> topic` must be a valid nesting in the manifest. Every child
must have exactly one parent. Reused labels must be converted to unique nested identifiers before
design matrices are built.

#### Continuous and ordinal descriptors

The plan must declare descriptor types. Column names such as `difficulty` must not determine
whether a variable is continuous, ordinal, monotone, or categorical.

#### Interpretation of prompt frailty

The frailty scale measures residual variation conditional on the specified model and priors. It is
not a model-free measure of taxonomy completeness. A large value is evidence that the current
model does not explain the observed prompt variation; possible causes include omitted descriptors,
interactions, judge noise, rollout dependence, or link misspecification.

#### Held-out prediction

For a held-out prompt, validation predictions must integrate over a new prompt effect drawn from
the fitted frailty distribution. They must not condition on an estimated effect for the held-out
prompt. Two distinct tasks should be reported:

- prediction for another rollout of an already observed prompt; and
- prediction for a held-out prompt using only its descriptors.

Only the second evaluates whether the descriptor model explains risk beyond prompt identity.

#### Risk concentration near zero total risk

For posterior draw `s`, the concentration share

\[
C^{(s)}(q)=
\frac{\text{risk in the highest-risk }q\text{ prompt mass}}
{\sum_i w_i p_i^{(s)}}
\]

is unstable when the denominator is close to zero. Reports should also show the absolute tail-risk
numerator and the posterior probability that total risk is below a declared numerical tolerance.
No arbitrary zero-denominator substitution should be hidden.

#### Weighted tail construction

The highest-risk `q` fraction must be defined by cumulative design weight, with fractional weight
at the boundary. Selecting a fixed number of rows is valid only for equal prompt weights.

#### Threshold summaries

`T(tau)` should be calculated for every posterior draw and summarized as a posterior distribution.
Because it is discontinuous and prior-sensitive when prompt risks lie near `tau`, the report should
also show the posterior expected exceedance mass

\[
\sum_i w_i P(p_i>\tau\mid D).
\]

The two have the same posterior mean but communicate prompt-level uncertainty more clearly.

## 3. Architectural decision

### 3.1 Use a parallel structural report

Do not add posterior credible intervals to `GroupRiskResult.intervals`. That type currently carries
coverage and certification semantics for empirical confidence procedures. Mixing posterior and
frequentist intervals in that collection would make eligibility, multiplicity, and report labels
too easy to misinterpret.

Introduce a separate `StructuralReport` object with:

- `certification_eligible: false`, fixed by the schema;
- its own model specification and prior provenance;
- convergence, predictive, calibration, and sensitivity diagnostics;
- posterior audit metrics;
- references to posterior-draw artifacts; and
- an optional consistency comparison with the empirical weighted mean.

The human-facing report may display the structural section after the empirical section, but the
JSON schemas and result objects should remain separate.

### 3.2 Proposed package layout

```text
src/inspect_safety_stats/
  structural/
    __init__.py
    data.py              # prompt-level aggregation and descriptor validation
    design_matrix.py     # coding, nesting, splines, and interaction matrices
    specification.py     # backend-neutral resolved model specification
    backend.py           # fit-result protocol
    pymc_backend.py      # initial NUTS implementation
    fit.py               # orchestration and deterministic seeds
    diagnostics.py       # convergence, PPC, calibration, and sensitivity checks
    cross_validation.py  # prompt-, topic-, and interaction-block holdouts
    audit_metrics.py     # posterior metrics independent of sampler objects
    artifacts.py         # posterior and summary serialization
  schema/
    structural_reports.py
  report/
    structural_writer.py
```

The statistical metric functions must consume ordinary arrays or a backend-neutral posterior draw
object. They must not import PyMC, ArviZ, or sampler-specific trace classes.

### 3.3 Optional dependency boundary

Bayesian modelling dependencies should be installed through an optional extra rather than added to
the lightweight empirical core:

```toml
[project.optional-dependencies]
hierarchical = [
  "pymc",
  "arviz",
  "xarray",
]
```

The CLI should give an actionable installation message when structural modelling is requested
without this extra.

## 4. Configuration design

Add an optional top-level section named `structural_model`. Omitting or disabling it must preserve
all current behaviour.

The structural model must inherit `design.hierarchy`; it must not define a competing hierarchy.
Shared levels receive partial-pooling effects, `prompt_id` is represented by prompt frailty, and
terminal levels that are one-to-one with prompts are recorded as aliases of that same frailty.

An illustrative configuration is:

```yaml
structural_model:
  enabled: true
  cohort: candidate
  link: cloglog
  likelihood: binomial
  descriptors:
    attack_method: {type: categorical}
    difficulty:
      type: continuous
      effect: spline
      basis_dimension: 6
      monotonic: false
  interactions:
    - {left: task_id, right: attack_method, type: categorical_categorical}
    - {left: task_id, right: difficulty, type: varying_slope}
  prompt_frailty:
    distribution: student_t
    degrees_of_freedom: 4
  priors:
    reference_probability: 0.01
    intercept_sd: 2.5
    main_effect_scale: 1.0
    interaction_scale: 0.5
    prompt_scale: 1.5
  audit_metrics:
    vulnerable_thresholds: [0.01, 0.10, 0.50]
    concentration_mass: [0.01, 0.05, 0.10]
  inference:
    chains: 4
    draws: 2000
    tune: 2000
    target_accept: 0.95
    seed: 87241
  validation:
    prompt_folds: 10
    holdout_levels: [topic]
  sensitivities:
    links: [cloglog, logit]
    frailty_distributions: [student_t, normal]
    interaction_scale_multipliers: [0.5, 2.0]
```

Configuration validation must reject:

- descriptor or hierarchy columns absent from the manifest;
- non-nested hierarchy declarations;
- interactions involving undeclared descriptors;
- spline effects applied to nonnumeric columns;
- monotonic effects without an explicit declaration;
- empty or unsupported interaction cells;
- multiple cohorts in version 1;
- incompatible configuration fingerprints within the selected cohort;
- any prompt without a classified rollout;
- any remaining unclassified attempted outcome; and
- thresholds or concentration masses outside their valid ranges.

## 5. Data contract

### 5.1 Prompt-level model frame

Create one row for every prompt in the selected fixed design:

| Field | Meaning |
|---|---|
| `prompt_id` | Stable identifier; no prompt content required |
| `k_unsafe` | Classified unsafe rollout count |
| `n` | Classified rollout count |
| `weight` | Resolved design weight for the selected audit group |
| hierarchy columns | Creator-declared nested identifiers |
| descriptor columns | Creator-declared categorical, ordinal, or continuous values |
| `configuration_fingerprint` | Identity of the rollout-generating conditions |

Raw prompts and responses must not be loaded into the structural model or written to its output.

### 5.2 Replicate aggregation rule

Aggregation is permitted only when all included observations for a prompt have the same declared
generation and scoring conditions. If conditions differ, the analysis must either:

1. reject the aggregation;
2. select a homogeneous cohort; or
3. in a later release, include the changing condition as a declared predictor.

### 5.3 Weight resolution

The structural likelihood remains unweighted. The selected group's resolved fixed-design weights
enter only when posterior draws are converted into audit metrics. The structural report must store
the exact resolved prompt weights and their hierarchy-derived allocation summary.

## 6. Version-1 statistical model

For prompt `i`:

\[
K_i\mid p_i\sim\operatorname{Binomial}(n_i,p_i),
\]

\[
p_i=1-\exp[-\exp(\eta_i)]
\]

under the default complementary-log-log link, and

\[
\eta_i=\mu+F_{\text{hierarchy}}(x_i)+F_{\text{descriptor}}(x_i)
+F_{\text{interaction}}(x_i)+u_i.
\]

Version 1 should initially support:

- a global intercept;
- nested categorical hierarchy effects;
- categorical main effects;
- one regularized continuous spline;
- categorical-by-categorical interactions;
- categorical varying slopes for a continuous descriptor; and
- Student-t or Gaussian prompt frailty.

The initial backend should use non-centred parameterizations. Hierarchical and categorical effects
should be centered for interpretability; interaction blocks must be constrained or coded so they
cannot reproduce their main effects.

## 7. Model-derived audit metrics

For every metric, computation occurs independently on each retained posterior draw.

### 7.1 Latent prompt risk

For every prompt, store posterior mean, median, standard deviation, selected quantiles, and the
probability of exceeding each configured vulnerability threshold. Prompt identifiers may be
included; prompt or response content must not be included.

### 7.2 Vulnerable-prompt mass

\[
T(\tau)=\sum_iw_i\mathbf 1\{p_i>\tau\}.
\]

Report its posterior median and interval, together with
`sum_i w_i P(p_i > tau | D)`.

### 7.3 Risk concentration

For each draw, sort prompts by latent risk, accumulate their fixed-design weights, and use a
fractional boundary item to form the highest-risk mass `q`. Report:

- the fraction of total risk attributable to that mass;
- the absolute risk numerator within that mass; and
- a warning or suppressed ratio if total risk is numerically indistinguishable from zero.

### 7.4 Descriptor-standardized risks

Standardization must use an explicitly named reference composition:

- `observed_joint`: retain the observed combinations of all other descriptors;
- `group_design`: use the fixed-design weights of a declared group; or
- a later explicit poststratification table.

The implementation must not generate structurally impossible combinations. Results are
associational descriptive risks, not causal effects.

### 7.5 Residual vulnerability

For every prompt, summarize

\[
H_i=\exp(u_i)
\]

and configured exceedance probabilities such as `P(H_i > 5 | D)`. Reporting should emphasize
posterior uncertainty and avoid treating a rank ordering as known exactly.

### 7.6 Interaction summaries

Report posterior interaction scales, probability-scale contrasts, and changes in prompt-level
held-out prediction. Coefficients alone are insufficient.

### 7.7 Empirical consistency check

For every posterior draw calculate

\[
R_{\mathrm{model}}=\sum_iw_ip_i.
\]

Compare its posterior distribution with the direct empirical weighted-mean estimate and confidence
interval. This comparison is a diagnostic. It does not replace the empirical result and does not
grant certification eligibility to the model result.

## 8. Diagnostics and validity gates

### 8.1 Sampler diagnostics

A production structural report requires:

- multiple chains;
- no unresolved divergent transitions;
- rank-normalized split `R-hat < 1.01` for monitored quantities;
- adequate bulk and tail effective sample sizes;
- no severe energy or tree-depth warnings; and
- finite posterior summaries.

Failure of a gate makes the structural analysis `invalid` or `degraded`; it must not change the
integrity or decision of the separate empirical report.

### 8.2 Prior predictive checks

Generate prior draws before observing outcomes and summarize:

- weighted mean risk;
- fraction of prompts near zero and near one;
- maximum prompt risk;
- vulnerable-prompt mass; and
- interaction and frailty multipliers.

The CLI should support a prior-only check so evaluators can review priors before fitting outcomes.

### 8.3 Posterior predictive checks

At minimum compare observed and replicated:

- total unsafe count;
- unsafe counts by major descriptor groups;
- number of zero-, one-, and high-failure prompts;
- distribution of prompt failure proportions;
- maximum prompt failure proportion; and
- weighted upper-tail summaries.

### 8.4 Cross-validation

The baseline validation unit is the prompt. Also support complete holdout of declared taxonomy
levels or interaction cells when data permit. Folds should be stratified only using pre-outcome
design metadata.

Report predictive log score, Brier score, and calibration summaries with uncertainty. Evaluation
of held-out prompts must integrate over new prompt frailty.

### 8.5 Sensitivity analysis

Production status requires the configured link, frailty, interaction-prior, and intercept-prior
sensitivities. The report should identify audit conclusions that change qualitatively, rather than
only placing multiple fit tables side by side.

## 9. Artifacts and report schema

A structural-model output directory should contain:

```text
structural-model/
  structural-report.json
  structural-report.md
  structural-report.html
  model-frame.csv
  prompt-posterior.csv
  audit-metrics.csv
  descriptor-risks.csv
  diagnostics.json
  posterior.nc
  model.resolved.yaml
  checksums.json
```

`posterior.nc` should use a standard labelled array format. The compact JSON report should contain
only summaries needed by downstream tools and the visualizer.

The structural schema should contain:

- package, schema, model, and backend versions;
- hashes for the empirical report, model frame, plan, and design manifest;
- resolved descriptors, hierarchy, interactions, weights, and priors;
- sampler settings and deterministic seeds;
- convergence and predictive status;
- prompt-risk summaries;
- audit-metric posterior summaries;
- standardized descriptor risks;
- sensitivity comparisons;
- empirical consistency diagnostics; and
- `certification_eligible: false`.

## 10. API and CLI

### 10.1 Python API

```python
from inspect_safety_stats.structural import fit_structural_model

structural = fit_structural_model(
    plan="safety-plan.yaml",
    empirical_report="report/report.json",
)
structural.write("report/structural-model")
```

The input empirical report ensures that classification, cohort selection, configuration checking,
and design-weight resolution are reused rather than independently reimplemented.

### 10.2 CLI

```bash
inspect-safety-stats model validate \
  --plan safety-plan.yaml \
  --report report/

inspect-safety-stats model prior-check \
  --plan safety-plan.yaml \
  --report report/ \
  --output prior-check/

inspect-safety-stats model fit \
  --plan safety-plan.yaml \
  --report report/ \
  --output report/structural-model

inspect-safety-stats model visualize \
  --report report/structural-model
```

Keeping model fitting as an explicit command preserves the speed and dependency footprint of the
existing empirical `analyze` command.

## 11. Staged delivery plan

### Stage 0: specification hardening

Deliverables:

- finalized configuration schema;
- explicit fixed-design interpretation;
- missing-data rule;
- hierarchy and descriptor validation rules;
- weighted concentration definition; and
- structural report schema draft.

Acceptance criteria:

- every estimand states whether it applies to fixed evaluated prompts or a hypothetical new prompt;
- no model result can enter the empirical decision engine; and
- all defaults can be examined with a prior predictive command before outcomes are fitted.

### Stage 1: model-frame and audit-metric core

Deliverables:

- prompt-level aggregation from an existing empirical report;
- descriptor schema and hierarchy validation;
- backend-neutral posterior-draw container;
- pure functions for `T(tau)`, weighted `C(q)`, absolute tail risk, model mean, and residual
  vulnerability; and
- unit and property tests for weighted boundary cases.

Acceptance criteria:

- no prompt or response content enters generated artifacts;
- all design weights resolve identically to the empirical report; and
- metric functions pass exact hand-calculated and randomized invariance tests.

### Stage 2: baseline Bayesian model (M0)

Deliverables:

- binomial likelihood;
- cloglog and logit links;
- nested categorical effects;
- categorical main effects;
- Student-t and Gaussian prompt frailty;
- configurable priors;
- non-centred parameterization; and
- posterior serialization.

Acceptance criteria:

- synthetic homogeneous and pure-main-effect cases recover their generating structure within
  simulation-calibrated tolerances;
- zero-failure data retain meaningful posterior uncertainty;
- all sampler validity gates are represented in the schema; and
- model results remain explicitly non-certifying.

### Stage 3: flexible descriptors and interactions (M1)

Deliverables:

- regularized spline effects;
- categorical-by-categorical interactions;
- categorical varying slopes;
- interaction-scale summaries; and
- probability-scale descriptor profiles and surfaces.

Acceptance criteria:

- known synthetic interactions are recovered;
- null interactions shrink toward zero;
- unsupported cells are not silently extrapolated; and
- effect coding and centering tests demonstrate identifiability.

### Stage 4: diagnostics and validation

Deliverables:

- prior predictive workflow;
- posterior predictive checks;
- prompt-level cross-validation;
- taxonomy-level and interaction-cell holdouts;
- calibration summaries; and
- automated sensitivity comparison.

Acceptance criteria:

- replicate-level leakage tests fail if a fold splitter puts one prompt in multiple folds;
- held-out prompt prediction integrates over new frailty;
- synthetic misspecification cases produce degraded adequacy status; and
- material sensitivity is surfaced prominently.

### Stage 5: reporting and Inspect-oriented example

Deliverables:

- structural JSON, Markdown, and static interactive HTML outputs;
- hierarchy-aware exploration of prompt-risk distributions, concentration, descriptor profiles,
  interactions, and diagnostics;
- empirical-versus-model consistency view; and
- one synthetic example plus one real safety-evaluation example whose manifest contains useful
  attack and difficulty descriptors.

Acceptance criteria:

- the empirical report appears first and remains unchanged;
- credible intervals are always labelled `posterior credible interval`;
- confidence bounds are always labelled `frequentist confidence bound`;
- report generation works offline without prompt or response content; and
- the example documentation explains model dependence and lack of coverage generalization.

## 12. Test strategy

### Unit tests

- link and inverse-link stability at extreme predictors;
- hierarchy nesting and coding;
- spline basis construction;
- interaction constraints;
- weighted tail selection with fractional boundary weight;
- zero-total-risk handling;
- posterior metric summaries; and
- artifact schema round trips.

### Property tests

- vulnerable mass lies in `[0, 1]`;
- concentration share lies in `[0, 1]` whenever defined;
- `C(1) = 1` for positive total risk;
- increasing any prompt risk cannot decrease vulnerable mass at fixed threshold;
- design-row order does not change any result;
- splitting one prompt's weight into identical duplicate representations does not change weighted
  summaries when hierarchy semantics preserve its parent mass; and
- relabelling category identifiers does not change posterior predictions under the same seed and
  coding structure, within Monte Carlo tolerance.

### Simulation tests

- homogeneous risk;
- pure hierarchical main effects;
- known sparse interaction;
- Gaussian frailty;
- heavy-tailed frailty;
- sparse and zero failures;
- omitted descriptor variation;
- severe link misspecification; and
- correlated rollouts as a deliberate negative-control case.

Simulation evaluation should assess interval calibration for model-derived quantities under the
generating model, rank recovery, predictive calibration, false interaction discovery, and
sensitivity of tail metrics.

### Integration and golden tests

- empirical report to structural model-frame conversion;
- complete CLI run on a small deterministic synthetic fixture;
- optional-dependency error behavior;
- posterior artifact checksums and provenance;
- report rendering without network access; and
- explicit verification that prompt and response text are absent.

Sampler-based golden tests should compare stable summaries within tolerances rather than requiring
byte-identical posterior draws across platforms.

## 13. Current-project integration map

| Existing component | Change |
|---|---|
| `config/models.py` | Add optional structural configuration models while preserving existing plans |
| `ingest/inspect_logs.py` | Reuse normalized classifications and configuration fingerprints; do not add a second Inspect parser |
| `ingest/manifest.py` | Reuse resolved group weights and add taxonomy-nesting validation |
| `schema/reports.py` | Keep empirical report semantics unchanged; add a separate structural report schema module |
| `api.py` | No posterior calculations; expose only a handoff to the structural subsystem if desired |
| `decision/engine.py` | No change; structural outputs never drive `PASS`, `FAIL`, or `INCONCLUSIVE` |
| `report/writer.py` | Optionally link to a structural sidecar without merging interval semantics |
| `report/visualization.py` | Add a separately labelled structural section or standalone structural visualizer |
| `pyproject.toml` | Add an optional hierarchical-modelling dependency extra |
| `tests/simulation/` | Add synthetic recovery, calibration, misspecification, and leakage tests |

## 14. Recommended first implementation slice

The smallest useful vertical slice is deliberately narrower than the complete source
specification:

1. one selected cohort and one declared audit group;
2. complete classified outcomes only;
3. binomial likelihood with cloglog link;
4. nested categorical hierarchy;
5. Student-t prompt frailty with Gaussian sensitivity;
6. latent prompt-risk summaries;
7. vulnerable-prompt mass and weighted risk concentration;
8. prior and posterior predictive checks;
9. prompt-level cross-validation; and
10. a standalone non-certifying structural report.

Spline effects and interactions should follow only after this slice passes synthetic recovery and
held-out prediction tests. This gives users the novel safety value—heterogeneity and concentration
analysis—without making the initial sampler and report architecture depend on the most complex
parts of the model.

## 15. Final recommendation

Proceed with the extension, but treat it as a new **safety-landscape modelling subsystem**, not as
another confidence-interval method.

The initial engineering milestone should implement the model-frame contract, fixed-design
posterior audit metrics, and a minimal hierarchical binomial model. The current HarmBench report
can exercise hierarchy and frailty, but it is not sufficient to validate difficulty or attack
effects because those descriptors are not present in the current design. Those features require a
synthetic fixture and a real evaluation manifest with prospectively declared descriptors.

The extension is successful if it makes concentration, heterogeneity, interactions, and unexplained
vulnerability auditable while leaving the empirical fixed-design risk estimates—and their coverage
claims—fully intact.
