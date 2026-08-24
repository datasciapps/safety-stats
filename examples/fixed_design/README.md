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
