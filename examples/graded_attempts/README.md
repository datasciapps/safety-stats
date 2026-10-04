# Worked example: graded scores and repeated attempts

This example runs entirely offline. Two mock models answer twelve fixed requests in three
categories, eight times each, and a judge grades every response on a 1–5 harm rubric (5 is the
most harmful). Most responses are harmless; a few prompts sometimes elicit serious harm; the
second model is somewhat worse. The grades come from a fixed, seeded harm profile in
`rubric_task.py`, so anyone can reproduce the run without credentials.

```bash
python examples/graded_attempts/run_example.py
```

The script:

1. runs the Inspect task for `mockllm/model-a` and `mockllm/model-b` with 8 epochs;
2. writes a starter plan with `init`'s graded option, equivalent to

   ```bash
   inspect-safety-stats init --logs examples/graded_attempts/run/logs \
     --harm-scale 1 5 --group-by category --baseline mockllm/model-a \
     --output examples/graded_attempts/run/analysis
   ```

3. adds a worst-of-4 attempt measure in its own multiplicity family:

   ```yaml
   outputs:
     risk_measures:
       - {id: worst_of_4, type: k_attempt_risk, k: 4, family: attempts}
   multiplicity:
     families:
       - {id: attempts, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
   ```

4. analyzes and writes `run/report/`.

## Reading `run/report/report.md`

**Mean harm.** Each grade is placed on a 0–100% scale (grade 1 is 0%, grade 5 is 100%), and
the report bounds the design-weighted mean. Model A averages about 3% of maximum harm and model B
about 6%.

**Risk over repeated attempts.** With four attempts at each prompt, the expected worst outcome
is roughly three times the single-attempt mean (about 9% and 17%). Harm concentrates on a few
prompts, and an adversary who retries finds it. This is the number to report when an attacker
can resample.

**Precision.** Every interval is marked: 8 rollouts on 12 prompts cannot resolve rates this
low, let alone a 3-point difference between the models. The Precision section turns each mark
into a budget. Before spending it, `plan comparison` sizes the run that would resolve the
difference:

```bash
inspect-safety-stats plan comparison --baseline-rate 0.03 --candidate-rate 0.06 --prompts 12
```

The interactive `run/report/report.html` labels the measures by meaning ("Mean harm",
"Expected worst of 4 attempts") and explains each interval's construction.
