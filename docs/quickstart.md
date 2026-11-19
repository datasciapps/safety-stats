# Quickstart

This walkthrough first analyzes a completed Inspect log that ships with the repository, once with
a correct plan and once with a broken one. Neither run needs provider credentials or network
access. It then runs the example fixed-design safety evaluation against a real model and analyzes
its logs.

Install the package with the `inspect` extra, as [Installation](installation.md) describes, and run
every command from the repository root.

To watch a whole evaluation run in Inspect without credentials, including a comparison of two
configurations, there is also the [offline demo](../examples/demo/README.md):
`python examples/demo/run_demo.py`.

## Offline: analyze a bundled log

[examples/real_simpleqa](../examples/real_simpleqa/README.md) holds eight real samples of a
SimpleQA evaluation of `openai/gpt-4o`, the design manifest that declares them, and a frozen plan
that counts an incorrect answer as a failure.

### Validate

```bash
inspect-safety-stats validate \
  --plan examples/real_simpleqa/safety-plan.yaml \
  --logs examples/real_simpleqa/simpleqa-8-real.eval
```

The command prints `Integrity: valid` and exits `0`. It also prints one warning,
`PLAN_SELF_DECLARED`: the plan states when it was frozen, but no external registration backs that
date, so the timing rests on the plan author's word. Warnings do not invalidate an analysis. They
stay visible in the report.

### Analyze

```bash
inspect-safety-stats analyze \
  --plan examples/real_simpleqa/safety-plan.yaml \
  --logs examples/real_simpleqa/simpleqa-8-real.eval \
  --output reports/quickstart/valid
```

Open `reports/quickstart/valid/report.md` for the one-page summary, `report-full.md` for the
full report, or `report.html` for the interactive hierarchy explorer.
Four of the eight answers are incorrect, so the estimated mean failure rate is 0.5. Its 95%
interval runs from about 0.09 to about 0.91, because each item was answered only once. More
rollouts of the same eight items would narrow the interval. They would not support a claim about
questions outside the eight.

### See an invalid run

[wrong-scorer-plan.yaml](../examples/real_simpleqa/wrong-scorer-plan.yaml) is the same plan with
one change: it reads the score of a scorer named `match`, which this log does not contain.

<!-- quickstart-test: exit 3 -->
```bash
inspect-safety-stats validate \
  --plan examples/real_simpleqa/wrong-scorer-plan.yaml \
  --logs examples/real_simpleqa/simpleqa-8-real.eval
```

The command prints a `SCORE_MISSING` error for the absent score column and one for each of the
eight rollouts, then `Integrity: invalid`, and exits `3`. A command that runs to the end has not
necessarily passed: scripts and CI jobs should check the exit code.

`analyze` also exits `3` on this input. It still writes the report, so the errors can be read in
context:

<!-- quickstart-test: exit 3 -->
```bash
inspect-safety-stats analyze \
  --plan examples/real_simpleqa/wrong-scorer-plan.yaml \
  --logs examples/real_simpleqa/simpleqa-8-real.eval \
  --output reports/quickstart/invalid
```

`reports/quickstart/invalid/report.md` lists the errors under "Warnings", with the assumption
each one checks. `report-full.md` lists them under "Data integrity", and every risk measure in
it is `unavailable`. The exit codes are listed in the
[API reference](api-reference.md).

Reports are write-once, so a second run into the same directory is refused. Delete
`reports/quickstart` or choose another output directory before running these steps again.

## Live: run the fixed-design example

### 1. Review the design

The example manifest is [design.csv](../examples/fixed_design/design.csv). It declares prompt, question, task, risk-domain, rollout count, and zero-tolerance metadata. These relationships are never inferred from prompt text.

The example plan is [safety-plan.yaml](../examples/fixed_design/safety-plan.yaml). Before using it, update:

- the cohort model identifier;
- `analysis.plan_frozen_at`;
- the optional registration information;
- thresholds and rollout counts appropriate to the real safety case.

### 2. Run the Inspect task

From the repository root:

```bash
inspect eval examples/fixed_design/safety_eval.py \
  --model openai/gpt-4o-mini \
  --epochs 20 \
  --no-epochs-reducer \
  --log-dir logs/fixed-design-example
```

This command requires the selected model provider's credentials. The statistical package does not.

### 3. Validate first

<!-- quickstart-test: needs live logs -->
```bash
inspect-safety-stats validate \
  --plan examples/fixed_design/safety-plan.yaml \
  --logs logs/fixed-design-example
```

Resolve every `ERROR` before interpreting safety claims. Warnings remain visible in the final report.

### 4. Analyze

<!-- quickstart-test: needs live logs -->
```bash
inspect-safety-stats analyze \
  --plan examples/fixed_design/safety-plan.yaml \
  --logs logs/fixed-design-example \
  --output reports/fixed-design-example
```

Open `reports/fixed-design-example/report.html` for the interactive hierarchy explorer, or
`report.md` for the one-page summary, or `report-full.md` for the full review document.

### 5. Read the result

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

### 6. Use a fresh output directory

Reports are write-once by default. Choose a new output directory for a new analysis. `--overwrite` is available only for deliberate replacement of files in a known report directory.
