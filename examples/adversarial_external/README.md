# Externally generated adversarial prompt sets

Someone else's attack produced the prompts; this example turns them into a frozen fixed design
and bounds the failure rate on that set.

It is written against [JailbreakBench artifacts](https://github.com/JailbreakBench/artifacts)
(MIT), whose published runs cover PAIR, GCG, JBC, DSN and random search against several target
models. The same script shape fits any generator that can produce a prompt with its anchor,
including DART, once its output exists.

## Why an adversarial set enters here and not through the generative path

The generative path estimates risk under a *declared rewrite distribution*, which requires the
rewriter to be blind to the target's outcomes. An attack is the opposite: it searches against
the target's failures. That rules it out as a rewrite kernel.

The fixed path has no such requirement, because it makes no claim that the prompts sample
anything. The declared set is the estimand. The honest sentence is:

> Risk on this frozen, adversarially constructed set is at most X.

Not available, and worth refusing explicitly:

> The worst case over all prompts near each anchor is at most X.

Attack search gives a lower bound reached by one optimiser. It cannot bound the worst case from
above, and the artifacts' own papers say the optimal success rate is unknown.

## What the script does

```bash
python build_design.py \
  --behaviors ../real_jailbreakbench/jbb_behaviors_harmful.csv \
  --artifact path/to/PAIR/black_box/vicuna-13b-v1.5.json \
  --artifact path/to/GCG/white_box/vicuna-13b-v1.5.json \
  --output-dir run/
```

It writes `design.csv`, the manifest the fixed path reads, and `prompts.jsonl`, the text for an
Inspect task. Prompt text stays out of the manifest so reports can omit it.

Each attacked prompt sits under the anchor it attacks (`question_id`), so an anchor and its
attacked variants are siblings in the hierarchy, and the unmodified behaviour is one more prompt
under the same anchor. Every row carries `perturbation`, `attack_type`, `generator_model`,
`artifact` and `artifact_sha256`.

Checks worth knowing about:

- **Anchor alignment is verified, not assumed, and it does drift.** Every artifact item repeats
  the behaviour's `goal`. On the published PAIR and GCG artifacts for `vicuna-13b-v1.5`, five of
  the hundred goals differ from the revision this repository pins: the artifact wording is
  longer. For example, behaviour 3 gains "based on the premise that they are less capable or
  efficient". Those anchors are dropped by default and listed in the run summary, because
  keeping them would compare an attacked prompt against a differently worded original.
  `--allow-goal-drift` keeps them instead, using the artifact's wording for the unmodified arm
  so that both arms ask the same question, and marks the rows `goal_drift`. Anchors where two
  artifacts disagree with each other are always dropped.
- **Items with no prompt are skipped and counted**, not silently dropped.
- **Their verdicts are discarded.** Artifacts carry `response` and `jailbroken` from their own
  run, against their own target, under their own judge. The prompts are re-run against a target
  you choose and scored by your grader.

## The comparison trap

A set generated against model A cannot fairly compare A with B: it had A's failures to aim at.
That is why `generator_model` is a declared column. Options, in order of cost: report a single
arm only, generate a set per arm, or generate against a third model neither arm shares.

Comparing *models* over one frozen set is first class, through cohorts. Comparing *groups*
(unmodified against attacked) is a within-design contrast: read it off the two simultaneous
intervals rather than as a paired comparison.

## What this example does not do

- **It does not check that an attacked prompt still asks the anchor's question.** For PAIR-style
  rewrites that is a real question; for suffix attacks such as GCG the anchor text is usually
  present verbatim with noise appended. Measuring it needs the rewrite validator, which is not
  on `main` yet.
- **It does not run the evaluation.** Point an Inspect task at `prompts.jsonl`, then `validate`
  and `analyze` against `safety-plan.yaml` as in the other fixed-design examples.
- **It ships no artifacts.** Download them from the upstream repository; the licence is theirs.
