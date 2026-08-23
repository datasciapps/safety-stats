# Hierarchical safety-landscape modelling

The optional structural subsystem models how unsafe-response probability varies across the prompts
in a fixed evaluation design. It supplements the empirical confidence-bound analysis; it never
changes the empirical report's integrity status or safety decision.

## What the first implementation provides

- Prompt-level binomial likelihood using classified rollout counts.
- Complementary-log-log link and multiplicative unsafe-intensity interpretation.
- Partial pooling over a creator-declared nested categorical hierarchy.
- Student-t prompt frailty with a mandatory-by-default Gaussian sensitivity fit.
- Posterior latent-risk summaries for every fixed prompt.
- Weighted vulnerable-prompt mass at declared risk thresholds.
- Weighted highest-risk concentration with fractional boundary weights.
- Prior and posterior predictive checks.
- Prompt-level K-fold cross-validation without replicate leakage.
- A diagnostic comparison with the direct empirical weighted mean.
- JSON, Markdown, CSV, NetCDF, checksum, and standalone HTML artifacts.

The implementation does not yet provide splines, descriptor interactions, model comparisons,
prompt transformations, judge-error models, or adaptive sampling.

## Installation

Install the optional modelling dependencies:

```bash
python -m pip install -e '.[hierarchical]'
```

For development:

```bash
python -m pip install -e '.[hierarchical,dev]'
```

The empirical analyzer remains usable without these dependencies.

## Add the model to the analysis plan

The empirical analysis and structural fit must use the same plan. Add `structural_model` before
running or regenerating the empirical report:

```yaml
structural_model:
  enabled: true
  cohort: candidate
  group: all-behaviors
  link: cloglog
  prompt_frailty:
    distribution: student_t
    degrees_of_freedom: 4
    gaussian_sensitivity: true
  priors:
    reference_probability: 0.01
    intercept_sd: 2.5
    hierarchy_scale: 1.0
    prompt_scale: 1.5
  audit_metrics:
    vulnerable_thresholds: [0.01, 0.10, 0.50]
    concentration_mass: [0.01, 0.05, 0.10]
    credible_interval: 0.95
    zero_risk_tolerance: 1.0e-12
  inference:
    chains: 4
    draws: 1000
    tune: 1000
    cores: 1
    target_accept: 0.95
    seed: 87241
    prior_predictive_draws: 1000
    cross_validation_folds: 5
    cross_validation_draws: 500
    cross_validation_tune: 500
```

The structural model inherits the complete `design.hierarchy`; it cannot declare a different
hierarchy. Levels shared by multiple prompts receive partial-pooling effects. For the current
HarmBench design, `question_id` is one-to-one with `prompt_id`, so separate question and prompt
variance components are not identifiable. Both levels remain in the declared hierarchy and are
represented by one terminal prompt-frailty term. The report records this aliasing explicitly.

Nested-equal weighting must use this same declared hierarchy. Weights define the fixed-design audit
distribution, while the matching hierarchy defines the model's pooling structure; the structural
likelihood itself remains unweighted.

## Prepare the empirical report

The structural subsystem deliberately reuses the empirical classifier, cohort selection,
configuration checks, and resolved weights. Run the ordinary analysis first:

```bash
inspect-safety-stats analyze \
  --plan safety-plan.yaml \
  --logs logs/ \
  --output report
```

`report.include_observations` must remain `true`. If the empirical report predates the
`structural_model` plan section, regenerate it after adding that section so the plan hashes match.

## Validate without fitting

```bash
inspect-safety-stats model validate \
  --plan safety-plan.yaml \
  --report report
```

Validation checks:

- the selected cohort and group exist;
- the hierarchy columns exist and are genuinely nested;
- terminal hierarchy levels that uniquely identify prompts form a valid alias block represented
  by prompt frailty;
- resolved weights are nonnegative and sum to one;
- every selected prompt has rollouts;
- every attempted rollout is classified;
- all selected rollouts share one configuration fingerprint; and
- the empirical report and structural plan hashes agree.

Provider policy blocks already classified safe under the end-to-end system estimand are classified
outcomes. Other missing, grader-error, quota-error, or otherwise unclassified outcomes block the
first structural implementation rather than being silently discarded.

## Fit the model

```bash
inspect-safety-stats model fit \
  --plan safety-plan.yaml \
  --report report \
  --output report/structural-model
```

The primary fit, Gaussian sensitivity, and cross-validation fits can be computationally expensive.
For a preliminary engineering smoke test, cross-validation can be omitted:

```bash
inspect-safety-stats model fit \
  --plan safety-plan.yaml \
  --report report \
  --output report/structural-model-smoke \
  --no-cross-validation
```

Disabling cross-validation should be disclosed and is not recommended for a completed structural
assessment.

## Outputs

```text
structural-model/
  structural-report.json
  structural-report.md
  structural-report.html
  model-frame.csv
  prompt-posterior.csv
  audit-metrics.csv
  diagnostics.json
  hierarchy-effects.csv
  hierarchy-marginal-risks.csv
  posterior.nc
  posterior-sensitivity-normal-frailty.nc
  model.resolved.yaml
  checksums.json
```

No prompt text or response content is loaded into the model frame or written to these artifacts.
Prompt identifiers are retained so findings can be joined back to the evaluator's controlled
qualitative-review process.

Regenerate the standalone visualization with:

```bash
inspect-safety-stats model visualize \
  --report report/structural-model \
  --overwrite
```

## Interpretation

### Latent prompt risk

Each prompt receives a posterior distribution for its repeated-generation unsafe probability.
Partial pooling prevents a small number of rollouts from being interpreted as a perfectly known
prompt probability.

### Marginal posterior risk by hierarchy node

For hierarchy node (g), the report computes the following functional on every joint posterior
draw (s):

\[
R_g^{(s)}=
\frac{\sum_{j\in g}w_j p_j^{(s)}}{\sum_{j\in g}w_j}.
\]

The reported mean and credible interval therefore live directly on the unsafe-probability scale
and preserve posterior dependence among prompts. They are conditional fixed-design summaries, not
external-population estimates. Conditional hierarchy-effect multipliers remain available as model
diagnostics but are not substitutes for these marginal risks.

### Vulnerable-prompt mass

For threshold `tau`, the report computes on every posterior draw:

\[
T(\tau)=\sum_i w_i\mathbf 1\{p_i>\tau\}.
\]

This is the weighted fraction of the fixed design whose latent unsafe probability exceeds the
threshold. It is not an estimate of prevalence in an external prompt population.

### Risk concentration

For prompt mass `q`, the report orders prompts by latent risk separately on each posterior draw and
calculates how much total risk arises in the highest-risk `q` design mass. Fractional weight is used
at the boundary. The absolute tail-risk numerator is also reported because a concentration ratio is
unstable when total risk is nearly zero.

### Prompt frailty

Prompt frailty is the residual log-intensity variation after the declared hierarchy. A large scale
means the hierarchy does not explain much of the prompt-to-prompt variation. It does not prove that
the taxonomy alone is deficient: scorer noise, omitted interactions, rollout dependence, or model
misspecification can produce the same pattern.

### Empirical consistency

The model-derived weighted mean is compared with the direct empirical weighted mean. A discrepancy
is a model diagnostic. The posterior result never replaces the direct estimate or its confidence
bounds.

## Validity and status

The structural report is always `certification_eligible: false`. Its own status is based on NUTS
convergence and held-out fit diagnostics. In particular, production-quality fits require no
divergences, maximum rank-normalized R-hat below 1.01, and adequate bulk and tail effective sample
sizes.

Posterior credible intervals are conditional on:

- the binomial rollout model;
- conditional rollout independence;
- the complementary-log-log link;
- the hierarchy and frailty specification;
- the declared priors;
- the fixed evaluated descriptor space; and
- the outcome classifier.

They are not finite-sample confidence bounds and do not establish safety outside the evaluated
design.

## Python API

```python
from inspect_safety_stats import fit_structural_model, validate_structural_inputs

metadata = validate_structural_inputs(
    plan="safety-plan.yaml",
    empirical_report="report",
)

structural = fit_structural_model(
    plan="safety-plan.yaml",
    empirical_report="report",
)
structural.write("report/structural-model")
```
