# Inspect Safety Stats

[![CI](https://github.com/aduncan001/safety_stats/actions/workflows/ci.yml/badge.svg)](https://github.com/aduncan001/safety_stats/actions/workflows/ci.yml)

`inspect-safety-stats` puts honest error bars on safety evaluations run with
[Inspect AI](https://inspect.aisi.org.uk/). Given the logs of a finished evaluation it reports
how often each model fails, with finite-sample confidence bounds that hold at any sample size
(including zero observed failures), whether one model fails more than another, where the
failures are, and how far an LLM judge's errors could move those numbers.

It is built for hand-curated safety evaluations, where the prompts are fixed in advance rather
than drawn from a population. The randomness it accounts for is the model's: which of its
possible responses each rollout produced. It never treats a benchmark as a random sample of
prompts, so its intervals mean what they say about the prompts you ran and make no claim about
prompts you did not.

## Quick start

```bash
pip install inspect-safety-stats

inspect-safety-stats init --logs logs/ --failure-values FAIL --group-by category --output analysis/
inspect-safety-stats analyze --plan analysis/safety-plan.yaml --logs logs/ --output report/
```

`init` writes a starter design manifest and plan from the logs. It never guesses which score
values are failures; you name them. `analyze` prints the headline rates and writes `report/`,
whose `report.md` is a one-page summary:

```text
| Model       | Failure rate | Interval      | Prompt-level sensitivity |
| gpt-4o      | 4.8%         | [2.7%, 7.5%]  | [0.7%, 14.6%]            |
| gpt-4o-mini | 6.5%         | [4.1%, 9.6%]  | [1.4%, 17.2%]            |

| Candidate vs baseline | Difference | Interval             | Reading                  |
| gpt-4o-mini vs gpt-4o | +1.8 pts   | [-3.4 pts, +6.9 pts] | no difference resolved † |
```

Intervals marked † are too wide to act on, and the summary says roughly how many more rollouts
would resolve each one. The [quickstart](docs/quickstart.md) walks through the options.

## What it does

- **Failure-rate bounds** for a fixed set of prompts: a weighted KL/Chernoff interval on the
  mean failure rate (exact Clopper–Pearson at prompt level), with a sensitivity that allows
  arbitrary dependence among one prompt's rollouts. Unclassified outcomes widen the bounds
  rather than being dropped.
- **Graded scores**, for rubric or probability judges: a declared scale maps each score onto
  harm in [0, 1], and every bound becomes a bound on mean harm.
- **Risk over repeated attempts**: the chance that an adversary with `k` tries at a prompt
  succeeds at least once, estimated without bias and bounded by the same engines.
- **Model comparisons** on the same prompts: risk differences and ratios with simultaneous
  coverage across any number of models, and simultaneous rank sets for all-pairs comparisons.
- **Breakdowns** by a declared hierarchy (task, category, …), calibrated so every node's
  interval holds simultaneously and parent and child intervals stay consistent.
- **Judge-error correction.** Every bound on judge-scored outcomes is a bound on how often the
  *judge reports* a failure. Declared judge-accuracy ranges give a sensitivity analysis; human
  labels drawn within each judge stratum give a certifying correction onto true risk.
  `judge sample` draws that labelling sample, blinded, and `plan judge-labels` sizes it.
- **Planning before you spend.** Zero-failure sizing, certification power, detectable rates,
  comparison power (a difference or a non-inferiority margin), and judge-label budgets, all
  exact.
- **Confirmatory mode** for release gates: frozen, hashable plans, two-threshold
  `PASS`/`FAIL`/`INCONCLUSIVE` claims, non-inferiority margins and CI exit codes. Off by default.
- **Auditable output**: deterministic JSON, Markdown, a self-contained interactive HTML
  explorer, CSVs and checksums.

Tail measures (CVaR, worst prompt, entropic and threshold risk) are available for plans that
declare them. They are built from per-prompt bounds and need many rollouts per prompt to be
informative; the analysis warns when they cannot be.

## Plan before you run

```bash
inspect-safety-stats plan zero-failure --threshold 0.01 --alpha 0.05
inspect-safety-stats plan certification --threshold 0.05 --anticipated-rate 0.02 --power 0.9
inspect-safety-stats plan detectable-rate --trials 800 --threshold 0.05 --power 0.9
inspect-safety-stats plan comparison --baseline-rate 0.05 --candidate-rate 0.05 --margin 0.02 --prompts 100
inspect-safety-stats plan judge-labels --judge-risk-upper 0.05 --target 0.08 --cohorts 2
```

## Python API

```python
from inspect_safety_stats import analyze

report = analyze(plan="analysis/safety-plan.yaml", logs="logs/")
report.write("report/")

for risk in report.group_risks:
    print(risk.cohort_id, risk.measure_id, risk.estimate, risk.lower, risk.upper)
```

The statistical functions are public too:

```python
from inspect_safety_stats.stats import binomial_upper_bound, zero_failure_sample_size

n = zero_failure_sample_size(threshold=0.01, alpha=0.05)   # 299
upper = binomial_upper_bound(0, n, alpha=0.05)              # just under 0.01
```

## Inputs and estimand

An analysis uses completed Inspect `.eval` logs, a CSV manifest with one row per fixed prompt,
and a strict YAML plan; `init` writes the last two. The score mapping is always explicit: the
package never assumes whether `C`, `I`, `PASS`, `FAIL`, `0` or `1` means a safety failure.

The default estimand is end-to-end system safety. A provider policy block during target
generation, with no content delivered, counts as a safe outcome; grader errors and unparseable
scores remain unknown and widen the bounds. Set `inputs.estimand: underlying_model` to keep
provider blocks unknown instead.

Every report states its scope:

> Results apply to the declared fixed evaluation set and specified stochastic rollout process. They do not estimate failure prevalence over an external population of prompts or establish safety for omitted hazards or deployment conditions.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Interpreting results](docs/interpretation.md)
- [Judge misclassification](docs/judge-misclassification.md)
- [Confirmatory mode](docs/confirmatory-mode.md)
- [Analysis plan reference](docs/plan-reference.md)
- [Python and command-line API reference](docs/api-reference.md)
- [Model and configuration comparisons](docs/model-comparisons.md)
- [Configuring hierarchy and independence](docs/independence-profiles.md)
- [Hierarchy-aware simultaneous inference](docs/hierarchy-simultaneous-inference.md)
- [Statistical methods and justification](docs/statistical-methods.md)
- [Installation](docs/installation.md)
- [Changelog](CHANGELOG.md) and [releasing](RELEASING.md)

Worked examples on real evaluations:

- [JailbreakBench-100: comparing models](examples/real_jailbreakbench/README.md)
- [JailbreakBench-100 in English, French, German and Maltese](examples/real_jailbreakbench/MULTILINGUAL.md)
- [HarmBench direct requests with repeated rollouts](examples/real_harmbench/README.md)
- [XSTest over-refusal](examples/real_xstest/README.md)
- [A hand-written fixed-design plan](examples/fixed_design/README.md)
- [Graded scores and repeated attempts, offline](examples/graded_attempts/README.md)

## License

MIT
