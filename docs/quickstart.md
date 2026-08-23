# Quickstart

This walkthrough runs the example fixed-design safety evaluation and analyzes its completed Inspect logs.

## 1. Review the design

The example manifest is [design.csv](../examples/fixed_design/design.csv). It declares prompt, question, task, risk-domain, rollout count, and zero-tolerance metadata. These relationships are never inferred from prompt text.

The example plan is [safety-plan.yaml](../examples/fixed_design/safety-plan.yaml). Before using it, update:

- the cohort model identifier;
- `analysis.plan_frozen_at`;
- the optional registration information;
- thresholds and rollout counts appropriate to the real safety case.

## 2. Run the Inspect task

From the repository root:

```bash
inspect eval examples/fixed_design/safety_eval.py \
  --model openai/gpt-4o-mini \
  --epochs 20 \
  --no-epochs-reducer \
  --log-dir logs/fixed-design-example
```

This command requires the selected model provider's credentials. The statistical package does not.

## 3. Validate first

```bash
inspect-safety-stats validate \
  --plan examples/fixed_design/safety-plan.yaml \
  --logs logs/fixed-design-example
```

Resolve every `ERROR` before interpreting safety claims. Warnings remain visible in the final report.

## 4. Analyze

```bash
inspect-safety-stats analyze \
  --plan examples/fixed_design/safety-plan.yaml \
  --logs logs/fixed-design-example \
  --output reports/fixed-design-example
```

Open `reports/fixed-design-example/report.html` for the interactive hierarchy explorer, or
`report.md` for the compact review document.

## 5. Read the result

The report separates:

- data integrity;
- prompt-level exact bounds;
- finite-design aggregate risk;
- simultaneous conditional mean-risk intervals at every declared hierarchy node;
- one selected independence profile at a time, with any configured sensitivity profiles;
- comparisons and diagnostics;
- provenance and audit information.

Human-facing reports are metrics-first by default. If a project wants the package to evaluate
predeclared thresholds, set `report.include_decisions: true` in the plan.

A zero observed failure count does not imply zero risk. If a prompt has zero failures in (n) independent rollouts, its one-sided upper bound is

\[
1-\alpha^{1/n}.
\]

Risk measures are calculated from prompt-level estimates and simultaneous bounds. They are not calculated by pretending prompts were IID samples.

## 6. Use a fresh output directory

Reports are write-once by default. Choose a new output directory for a new analysis. `--overwrite` is available only for deliberate replacement of files in a known report directory.
