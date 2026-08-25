# Live generative example: real providers end to end

Every other generative artifact in this repository uses a deterministic fake kernel and a
simulated target, which is what makes the
[multilingual example](../generative_multilingual/README.md) checkable against a known
answer. This one has no known answer. It exists to be **run against real providers**, so
that the path from a declared taxonomy to a reported bound is exercised with a real
rewriter, a real target model and a real model-graded judge.

You supply the API access. Nothing here runs without it.

## What is real, and what you write

| | |
|---|---|
| [`rewriter.py`](rewriter.py) | a `RewriteKernel` backed by any model Inspect can reach — **the one piece a real evaluation requires you to write**, kept here as a reference |
| [`generative_live.py`](generative_live.py) | the Inspect task: reads the materialized dataset, generates, grades with a model judge |
| [`plan.yaml`](plan.yaml) | a small budget — 60 rewrites — configured for `openai/gpt-4o` and `openai/gpt-4o-mini` |

`--kernel` takes either `module:attribute` for something importable, or
`path/to/file.py:attribute` for a local file. The file form is what you want for a kernel
that lives in your project rather than in an installed package — it needs no `__init__.py`
and no `PYTHONPATH`.

The kernel demonstrates two things the package cannot do for you.

**It performs validation rather than declaring it.** `rewrite.validation.validator_id` is
fingerprinted by the package but never invoked — nothing checks that a rewrite preserves its
seed's meaning. Set `validator` and this kernel runs a second, model-graded equivalence check
and reports a failure as `invalid`, which puts the unit into the retry path and keeps the
accepted distribution honest. Leave it unset and the run is roughly half the cost, but
nothing verifies the rewrites still mean what the seeds mean.

**It distinguishes error from invalidity.** A provider failure is `error` — operational,
retried on `--resume`, and saying nothing about the kernel. A rewrite that came back and was
rejected is `invalid`, and is part of the accepted distribution the estimand is defined
against. Getting this backwards would silently change what you measured.

## Running it

Decide the budget first — this costs nothing:

```bash
inspect-safety-stats generative budget --plan examples/generative_live/plan.yaml --assumed-rate 0.05 --acceptance-rate 0.8
```

Generate the prompts. This is where the spend starts:

```bash
inspect-safety-stats generative materialize \
  --plan examples/generative_live/plan.yaml \
  --output run/live \
  --kernel examples/generative_live/rewriter.py:build_kernel
```

Evaluate them, then analyse:

```bash
inspect eval examples/generative_live/generative_live.py \
  --model openai/gpt-4o-mini \
  -T dataset=run/live/datasets/dataset.jsonl \
  --log-dir run/live/logs

inspect-safety-stats generative analyze \
  --plan examples/generative_live/plan.yaml \
  --artifacts run/live --logs run/live/logs --output run/live/report \
  --slice language=maltese
```

`-T dataset=` is read relative to your working directory. Inspect itself resolves a
relative dataset path against the task file, so the task converts it first; an absolute path
always works.

If materialization is interrupted — a rate limit, a dropped connection — add `--resume`.
Ledgers are appended per unit, so only the outstanding rewrites run again.

## What it costs, and how long

At the declared budget of 60 rewrites: about 60 rewrite calls plus 60 validator calls, then
60 target rollouts and 60 judge calls. Roughly 240 model calls, most of them short.

**The kernel protocol is synchronous**, so materialization is sequential — one rewrite at a
time. At around two seconds per call that is a few minutes for 60 units, and it scales
linearly. A production-sized run wants concurrency, which the protocol does not currently
offer; that is a known limitation rather than a tuning problem.

## Two things to check before trusting the number

**Acceptance rates.** `acceptance.csv` and the report's rewrite-acceptance section show how
hard the kernel worked for each arm. A low-diversity rewriter interacts badly with
deduplication: in a dry run against a mock model that returns the same string every time,
`scope: semantic_seed_path` rejected 156 candidates and 52 of 60 units exhausted their
attempts, leaving 8 accepted.

```
  language          german              12         0         100%           3.00
  language          english             24         4          83%           2.67
```

If your real rewriter produces low-variety paraphrases you will see the same shape, and the
accepted prompts will be a thin and unrepresentative slice of what you declared. Raise
`maximum_attempts`, loosen `deduplication.scope`, or raise the rewriter's temperature.

**Whether the arms actually bite.** Group the accepted prompts by seed and read them. On a
first run the template named each constraint rather than defining it, and `gpt-4o` echoed the
labels — prompts opening `direct_request:` or `Rollenspiel:` — while `roleplay` produced no
roleplay and `plain` and `conversational` were near-indistinguishable. The paraphrases were
semantically faithful and the equivalence check passed them all; the *arms* were the problem.

That matters statistically: the plan assigns 30% of the design's weight to `roleplay` and 20%
to `nested_task`. If the kernel is not producing those variations, the hierarchy claims
structure the prompts do not carry, and any node or slice split on `wrapper` measures noise.
The current template defines each value's meaning and forbids naming it; check the output
anyway, because only reading the prompts tells you.

**Whether the judge dominates.** The plan declares `judge_calibration` scenarios, so the
report carries a sensitivity section. At low prevalence most of a judge-reported number can
be judge error, and these bounds are declared rather than measured — they are diagnostics,
not evidence. See [judge misclassification](../../docs/judge-misclassification.md) for what
it would take to make them certifying.

## What this example cannot tell you

There is no ground truth here. The multilingual example exists to check coverage against a
known answer; this one exists to check that the pipeline survives contact with real
providers. Read the numbers it produces as a demonstration that the machinery works, not as
a measurement of any model's safety.
