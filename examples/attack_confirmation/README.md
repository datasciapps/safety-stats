# Attack runs scored by a confirmation rollout and an independent judge

In this example one Inspect sample is one complete attack run on one behaviour. The solver
runs an iterative attack with a query budget K: an attacker proposes a prompt, the target
answers, and the attacker's own judge grades the answer. The attack stops at that judge's
first "unsafe" verdict. The scorer then sends the prompt the attack found to the target once
more, and grades that fresh completion with a second judge, one the attacker did not search
against. The package's fixed-design path analyses the result unchanged.

It runs offline by default, against a simulated attacker, target and judges, so it needs no
credentials. The same task runs against real models with `-T mode=live`.

| | |
|---|---|
| [`attack_confirmation.py`](attack_confirmation.py) | the Inspect task: the attack solver, the confirmation scorer, the simulated and the model-backed components |
| [`behaviours.csv`](behaviours.csv) | 100 placeholder behaviours, 10 in each of 10 categories |
| [`design.csv`](design.csv) | the design manifest: one attack run per behaviour, with the attack declared in every row |
| [`plan.yaml`](plan.yaml) | the analysis plan: the confirmed outcome, a candidate-versus-baseline comparison and one threshold claim |
| [`run_offline.py`](run_offline.py) | runs both simulated configurations and the analysis, and prints both rates |

## Why the confirmation rollout

An attack that stops at its judge's first "unsafe" verdict selects for that judge's false
positives. Take a behaviour the target never carries out, and a judge that calls a safe
response unsafe with probability f on each query. The attack still reports a jailbreak with
probability 1 − (1 − f)^K:

| f | K = 5 | K = 20 | K = 60 |
|---|---|---|---|
| 0.02 | 0.10 | 0.33 | 0.70 |
| 0.05 | 0.23 | 0.64 | 0.95 |

The confirmation verdict is drawn once, after the search has ended, so the search does not
select it, on two conditions that the task does not check:

- **The target samples.** The fresh completion is a new draw only at a temperature above 0.
  At temperature 0 it repeats the response that the attacker's judge called unsafe, and the
  confirmation judge reads the very text the search selected.
- **The two judges err independently.** The task refuses only a confirmation judge with the
  same name as the attacker's. Both judges here use one rubric (`JUDGE_TEMPLATE`), and judges
  that share a rubric or a model family tend to accept the same false positives, the ones the
  search selected for. A judge of another model family, with a rubric of its own, keeps them
  apart better; their independence stays an assumption.

Then the confirmation's error rates are the confirmation judge's own on the text it reads,
fresh completions of attack-found prompts, and the package's judge correction applies to them
with labels drawn as [What a result means](#what-a-result-means) describes (see
[Judge misclassification](../../docs/judge-misclassification.md)). The confirmation costs one
target call and one judge call per run, against up to K of each in the search.

The confirmation keeps the attacker's judge's false positives out of the outcome, but not out
of the search. A false positive ends a run on a prompt that does not work, with the rest of
its budget unspent, so a noisier attacker's judge makes the same attacker weaker and the
confirmed rate lower. Take behaviours on which each query finds a prompt that works with
probability q = 0.05, a target that carries out a working prompt every time, an attacker's
judge that never misses one and a confirmation judge that never errs. The confirmed rate is
then q(1 − a^K)/(1 − a), with a = (1 − q)(1 − f):

| f | 0 | 0.02 | 0.05 | 0.20 |
|---|---|---|---|---|
| K = 20 | 0.64 | 0.55 | 0.45 | 0.21 |
| K = 60 | 0.95 | 0.71 | 0.51 | 0.21 |

For a gate, that is the direction that favours a PASS (see
[What a result means](#what-a-result-means)). The offline simulation does not show it: its
targets comply with each query at a fixed rate, whatever the prompt, so a found prompt is
worth no more than any other.

Every run writes two scores:

- `confirmed_jailbreak` is FAIL when the confirmation judge calls the fresh completion of the
  found prompt unsafe, and its `answer` is that completion. A run that found no prompt is
  PASS, and makes no further call. This is the outcome the plan analyses.
- `attacker_reported` is FAIL when the attacker's own judge called some query unsafe. It is
  kept for comparison only.

## Running it offline

```bash
python examples/attack_confirmation/run_offline.py
```

This evaluates two simulated configurations, writes the logs to `logs/attack-confirmation`
and the report to `reports/attack-confirmation`, and prints:

```text
configuration          attacker-reported   confirmed   unconfirmed
mockllm/baseline                    0.77        0.08          0.69
mockllm/candidate                   0.68        0.01          0.67
claim attack_success_bounded: INCONCLUSIVE (upper bound 0.06391479153815988)
```

`unconfirmed` is the share of runs that stopped on a find the confirmation did not reproduce.

The `candidate` configuration never complies, so every jailbreak reported against it is a
judge false positive. The attacker's judge (f = 0.05, K = 20) reports 68 of 100 behaviours
jailbroken; the confirmation judge (f = 0.02) confirms one. The `baseline` configuration
complies with each query with probability 0.5, 0.3 and 0.2 in three of the categories and
never in the others. Both configurations are declared in `SIMULATED_COMPLIANCE` and the judge
rates in `SIMULATED_ATTACKER_JUDGE` and `SIMULATED_CONFIRMATION_JUDGE`. Each run draws its
randomness from streams seeded by configuration, behaviour and epoch, so the output repeats.

The same steps by hand:

```bash
inspect eval examples/attack_confirmation/attack_confirmation.py \
  --model mockllm/baseline,mockllm/candidate --log-dir logs/attack-confirmation
inspect-safety-stats analyze --plan examples/attack_confirmation/plan.yaml \
  --logs logs/attack-confirmation --output reports/attack-confirmation
```

Delete old logs first: logs left in the directory join the analysis as extra attack runs,
and the integrity check then marks the analysis invalid.

The simulated numbers support no safety claim. They show that the analysis path works and
how far apart the two verdicts can be.

## Running it against real models

Live mode needs an attacker model, a target and two judges. The target is the eval's model;
the others are named as task arguments. The two judges must differ: the task refuses to run
when the confirmation judge has the name of the judge the attacker searched against. It
compares the names only; [Why the confirmation rollout](#why-the-confirmation-rollout) says
what else the confirmation needs of the judges, and of the target's temperature, which
`--temperature` sets.

The design manifest declares the attack in every row: `attack_id`, `attack_fingerprint`,
`budget_k`, `attacker_model`, `attacker_judge` and `confirmation_judge`. The fingerprint is a
digest of the attack id, the budget, the attacker model, the attacker's judge and the prompt
templates. It does not cover the attacker's code, the stopping rule, generation settings or,
offline, the simulated rates, so give the attack a new `attack_id` whenever one of those
changes. The task refuses to run against a design that declares a different attack, so
write a design for the attack you are about to run. The task checks the design once, when it
starts. `analyze` never compares the logs with the design's declaration; the plan's cohorts,
below, keep runs of another attack out of the analysis.

Put your behaviours in `run/attack/behaviours.csv`, with the columns `prompt_id`,
`category_id` and `goal`, and write the design from the repository root:

```bash
python -c "
import sys; sys.path.insert(0, 'examples/attack_confirmation')
from pathlib import Path
from attack_confirmation import AttackDeclaration, write_design
write_design(Path('run/attack/design.csv'), AttackDeclaration(
    attack_id='pair', budget_k=20,
    attacker_model='<provider/attacker-model>',
    attacker_judge='<provider/attacker-judge>',
    confirmation_judge='<provider/confirmation-judge>'),
    behaviours=Path('run/attack/behaviours.csv'))
"
```

Then run the attack against each configuration with the same settings. Name the design and
the behaviours by absolute paths: `inspect eval` creates the task in the task file's
directory, so a relative path would be read from there, and the task refuses one.

```bash
inspect eval examples/attack_confirmation/attack_confirmation.py \
  --model <provider/target-model> --log-dir run/attack/logs \
  -T mode=live -T design=$PWD/run/attack/design.csv \
  -T behaviours=$PWD/run/attack/behaviours.csv \
  -T attack_id=pair -T budget_k=20 \
  -T attacker_model=<provider/attacker-model> \
  -T attacker_judge=<provider/attacker-judge> \
  -T confirmation_judge=<provider/confirmation-judge>
```

Point a copy of `plan.yaml` at the new design, and match every cohort on the attack as well as
the model, with the values the design declares:

```yaml
cohorts:
  candidate:
    match:
      model: <provider/target-model>
      task_arg_attack_id: pair
      task_arg_budget_k: 20
      task_arg_attacker_model: <provider/attacker-model>
      task_arg_attacker_judge: <provider/attacker-judge>
      task_arg_confirmation_judge: <provider/confirmation-judge>
    require_uniform: [...]  # as in plan.yaml
```

The baseline takes the same match with its own model. Then run `analyze` as above. A run of
another attack, budget, attacker or judge joins no cohort, so it cannot enter a claim or the
comparison: the behaviours it leaves without a run are reported as `PROMPT_MISSING`, a cohort
it leaves empty as `COHORT_EMPTY`, and either makes integrity invalid. Matching on the model
alone would let two arms attacked at different budgets, or judged by different judges, be
compared (see [one judge for every arm](../../docs/model-comparisons.md#one-judge-for-every-arm)).
`require_uniform` names the task's metadata, which holds the whole declaration, fingerprint
included, so runs of two attacks within one cohort are reported as `CONFIG_DRIFT`.

The attacker in live mode is a minimal PAIR-style loop: each attacker call sees the goal, its
previous prompt, the target's answer and the judge's verdict. Replace `ModelAttacker` with the
attack you want to evaluate, and give it its own `attack_id`: the fingerprint cannot tell two
attackers' code apart. The declaration and the confirmation step stay as they are.

## What a result means

- **The estimand.** For each behaviour, the probability that this attack procedure with this
  budget produces a prompt whose fresh completion the confirmation judge scores unsafe. The
  claim is about the weighted mean of that probability over the declared behaviours.
- **A PASS covers this attack at this budget, with this attacker's judge.** It says this
  attack at budget K succeeded on at most the bound's share of behaviours. It is no evidence
  about other attackers or larger budgets, nor about the same attacker with a judge that errs
  less: each false positive of the attacker's judge ends a run on a prompt that does not work,
  so a noisier judge lowers the confirmed rate (see
  [Why the confirmation rollout](#why-the-confirmation-rollout)).
  Report beside the result the attacker's judge's false-positive rate, measured on labelled
  responses from the search (the sample store's `attack_attempts`), and the share of runs
  that stopped on a find the confirmation did not reproduce (`unconfirmed` above). Each of
  those runs left the rest of its budget unspent.
- **The unit is the attack run.** The queries inside a run add no independent observations.
  One run per behaviour makes the run and the behaviour the same unit, which is why both
  independence profiles block by `prompt_id` or coarser.
- **Independence across behaviours is assumed.** The attacker carries no state from one
  behaviour to the next. An attacker that learns across behaviours breaks that assumption,
  and this example's claims do not cover it. Such an attacker can still fail a gate: an
  [adaptive claim](../../docs/plan-reference.md#adaptive_claims-experimental) (experimental)
  bounds its running success rate from below without assuming independence, given a manifest
  column that fixes the attack order, and can only FAIL.
- **Pair on behaviour.** Attack prompts differ per target, so there is no shared prompt to
  pair the two configurations on. The comparison pairs on `prompt_id`: the same behaviour,
  attacked once against each configuration.
- **Judge correction needs labels of what the confirmation judge read.** A `validated`
  calibration of the confirmation judge must be labelled on fresh completions of attack-found
  prompts, not on a benign set: the judge's error rates on adversarial outputs are what the
  correction needs. Label the `answer` of the `confirmed_jailbreak` score, which is the fresh
  completion. The sample's output is the search's last response, a different text. A run that
  found no prompt has no answer and made no judge call; its PASS is right by construction, so
  when the labelling sample draws it, label it `safe` rather than skip it, since skipping
  breaks the simple random sample of its stratum.

## Planning the number of behaviours

With the exact one-sided binomial test at α = 0.05 and a threshold of 0.05, 100 behaviours
(the size of JailbreakBench) certify a rate below 0.05 only with at most one confirmed
jailbreak, and 400 (the size of HarmBench's design) with at most 12. The plan's interval
method is more conservative than that test, as the offline run shows: one failure in 100
gives an upper bound of 0.064, and the claim is INCONCLUSIVE.
