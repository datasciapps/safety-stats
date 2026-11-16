# Fixed-design example

This directory contains a minimal Inspect task, its creator-declared design manifest, and a
confirmatory analysis plan.

1. Update the model ID and freeze timestamp in `safety-plan.yaml`.
2. Run 20 epochs for every sample:

   ```bash
   inspect eval examples/fixed_design/safety_eval.py \
     --model openai/gpt-4o-mini \
     --epochs 20 \
     --no-epochs-reducer \
     --log-dir logs/fixed-design-example
   ```

3. Validate and analyze:

   ```bash
   inspect-safety-stats validate \
     --plan examples/fixed_design/safety-plan.yaml \
     --logs logs/fixed-design-example

   inspect-safety-stats analyze \
     --plan examples/fixed_design/safety-plan.yaml \
     --logs logs/fixed-design-example \
     --output reports/fixed-design-example \
     --no-decision-exit-codes
   ```

The example scorer is deliberately simple and exists only to demonstrate integration. It is not
a validated safety scorer. Replace it, the hazards, rollout counts, thresholds, and model metadata
before using the workflow for a real safety decision.

## Compare two models or configurations

`comparison-plan.yaml` shows the first-class comparison workflow. Update its candidate and
baseline cohort filters, run both configurations over the same fixed design, and point one analysis
at the directory containing both sets of Inspect logs:

```bash
inspect-safety-stats analyze \
  --plan examples/fixed_design/comparison-plan.yaml \
  --logs logs/fixed-design-comparison \
  --output reports/fixed-design-comparison
```

The report contains the two arm risks, direct candidate-minus-baseline interval, risk-ratio and
relative-risk-reduction summaries, hierarchy-node contrasts under every configured independence
profile, and optional paired diagnostics. The primary interval permits unequal rollout counts in
the two arms; the diagnostic `pair_key` requires exact matched rollouts and is evaluated separately.

## Prompt variants of one base prompt

`variant-plan.yaml`, `variant-design.csv` and `variant_eval.py` show a fixed design in which
several prompts are wordings of one base prompt: the original, a paraphrase, a role-play
wrapper. The base prompt is the `question_id` and each wording is a `prompt_id` beneath it; the
`variant` column names the kind of wording and is not used by the analysis.

- `nested_equal` weights over `[task_id, question_id, prompt_id]` give each base prompt of a task
  the same weight whatever its number of variants, and split that weight equally among its
  variants. In this design the three credential-theft wordings get 1/12 each and the single
  authentication-bypass prompt gets 1/4.
- The `question` independence profile, `block_by: [question_id]`, puts every rollout of every
  variant of one base prompt into one block. It is reported as a sensitivity beside the
  `rollout` and `prompt` profiles.
- `hierarchy_nodes: true` reports an interval for each base prompt as well as each task.

```bash
inspect eval examples/fixed_design/variant_eval.py \
  --model openai/gpt-4o-mini \
  --epochs 20 \
  --no-epochs-reducer \
  --log-dir logs/fixed-design-variants

inspect-safety-stats analyze \
  --plan examples/fixed_design/variant-plan.yaml \
  --logs logs/fixed-design-variants \
  --output reports/fixed-design-variants \
  --no-decision-exit-codes
```

[Prompt variants in fixed designs](../../docs/plan-reference.md#prompt-variants-in-fixed-designs)
explains the choices.
