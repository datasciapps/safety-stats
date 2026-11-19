# Analysis plan reference

Plans are strict YAML or JSON documents. Unknown fields are errors, and numeric fields, including the values of a mapping such as `explicit_alpha`, reject `true` and `false` rather than reading them as 1 and 0. Relative manifest paths resolve against the plan file.

## Schema versions

`schema_version` is `"2.0"` (the default when it is left out) or `"3.0"`, and an unquoted `2.0`
or `3.0` reads the same. A plan that declares `1.0` is refused with what changed in 2.0 (see
[Configuring hierarchy and independence](independence-profiles.md)). The two differ only in
how a plan names the fields of runs and rollouts, in `design.rollout_key`, `cohorts.*.match` and
`cohorts.*.require_uniform`:

| | Schema 2.0 | Schema 3.0 |
|---|---|---|
| Field names | The source's own: for Inspect logs, the columns of `evals_df` and `samples_df` | A field every source has, or a source field written `attributes.<name>` |
| Run fields | `model`, `task_version`, `task_arg_*`, ... | `run_key`, `model`, `status`, `run_id`, `attributes.<name>` |
| Rollout fields | `eval_id`, `id`, `epoch`, ... | `run_key`, `item_id`, `repeat`, `rollout_id`, `attributes.<name>` |
| `design.rollout_key` default | `[eval_id, id, epoch]` | `[run_key, item_id, repeat]` |
| `inputs.sample_join` | Allowed, with its only values | Removed |

Schema 3.0 works the same with every source, Inspect logs or [rollout files](rollout-files.md),
and is the version to use for new plans. A field name that is neither neutral nor written
`attributes.<name>` is an error. `attributes.<name>` names the source's field verbatim: for an
Inspect run, `attributes.task_version` is the `task_version` column. For an Inspect log, a run's
`run_key` is its `eval_id`, and a rollout's `item_id`, `repeat` and `rollout_id` are its sample
`id`, `epoch` and `sample_id`.

To move a 2.0 plan to 3.0, set `schema_version: "3.0"`, delete `inputs.sample_join`, and rename
fields: `eval_id` to `run_key`, `id` to `item_id`, `epoch` to `repeat`, `sample_id` to
`rollout_id`, and any other source field `f` to `attributes.f` (`model` keeps its name). The
plan's hash changes with its text. A 2.0 plan keeps its meaning and its hash.

### The plan hash

`hash-plan` prints, and every report records as `plan_sha256`, the SHA-256 of what the plan's
author wrote: the fields present in the file, in their validated form, with input paths relative
to the plan's directory. A field the plan leaves out is not hashed, whatever its default, so a
release that adds a field or changes a default leaves the hash of an unchanged plan file as it
was (#56). A field written at its default value is hashed: the plan states it, and its meaning
does not follow a later default. Writing out a default therefore changes the hash, as any other
edit does.

What the hash leaves out, the report records. `plan_defaults` in `report.json` lists every field
the plan left out, with the value its default gave the analysis, as `{"path", "value"}` entries
such as `{"path": "inputs.score.token_limit", "value": "score"}`. A path writes a list item as
`[i]` and a mapping entry by its key, and a section left out whole has an entry for each of its
fields. Experimental
fields are listed too, although `plan.resolved.yaml` leaves them out while they are at their
defaults. A field that does not exist in the plan, such as `inputs.sample_join` under schema
3.0, has no entry. The
"Defaults applied" section of `report-full.md` lists the defaults that set a value, and counts those
that leave an optional feature off or a parameter unused (null, false or empty).

## `analysis`

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Stable analysis identifier |
| `title` | yes | Report title |
| `description` | no | Human context |
| `intent` | yes | `confirmatory` or `exploratory` |
| `plan_frozen_at` | confirmatory | Timestamp preceding selected evaluations |
| `registration` | no | External URI and SHA-256 of a preregistered plan |

Exploratory analyses cannot produce `PASS`. Their claims are `INCONCLUSIVE`, and each records
as `descriptive_status` what a confirmatory analysis of the same bounds would give. See
[Exploratory analyses](interpretation.md#exploratory-analyses).

### Registration receipts (experimental)

`plan_frozen_at` is a time the author writes, so a confirmatory report labels its timing
self-declared (`PLAN_SELF_DECLARED`). A registration receipt records when the plan's hash was
registered, so that a reader can check it against the start of generation:

```bash
inspect-safety-stats hash-plan --plan plan.yaml --register plan.receipt.json
```

The command prints the hash as before, then writes `plan.receipt.json`, which holds the plan's
canonical SHA-256 and the current UTC time, rounded up to the whole second so that it never
precedes the registration. Inspect records a run's start truncated to the second, so a receipt
written less than a second before generation starts can read as later than it. The command
never overwrites an existing receipt. The time comes from the local clock, so a receipt proves
nothing until it is anchored somewhere a reader trusts, before generation starts. For example,
commit it in a signed commit and push it, or have a timestamping service sign its hash
(`openssl ts -query -data plan.receipt.json -sha256 -cert -out plan.receipt.tsq`, sent to an RFC 3161 server).

Pass the receipt to `analyze` or `validate` with `--registration-receipt plan.receipt.json`,
or `registration_receipt=` in Python. They check two things:

| Code | Severity | When |
|---|---|---|
| `PLAN_RECEIPT_MISMATCH` | error | The receipt's hash is not this plan's hash. Any plan edit changes the hash. |
| `PLAN_RECEIPT_POSTHOC` | error | Confirmatory plan: the receipt is later than the earliest selected evaluation start. |
| `PLAN_REGISTERED` | info | Confirmatory plan: the receipt matches and is not later than the earliest start. Replaces `PLAN_SELF_DECLARED`. |

Both errors make integrity `invalid`. The checks do not verify the external anchor: the
`PLAN_REGISTERED` message says so, and a reader who relies on the timing checks the signed
commit or timestamp token themselves. The receipt is given to the command rather than named
in the plan, so that the plan's hash does not depend on it.

The receipt records the canonical hash, the one `hash-plan` prints without `--inference`;
`--register` cannot be combined with `--inference`. A [decision policy](decision-policy.md)
leaves the registered plan and its hash as they were, so a receipt still matches when the
analysis uses one.

## `inputs`

`design_manifest` points to the fixed-design CSV. Each rollout joins the manifest's `prompt_id`
by its item: the Inspect sample `id`, or the `item_id` of rollout files. Schema 2.0 plans may
state this as `sample_join: {inspect_field: id, manifest_field: prompt_id}`, its only values;
schema 3.0 removes the field.

The manifest always needs a `prompt_id` column. It needs `question_id` and `task_id` only when
the plan names them: in `design.hierarchy`, whose default `[task_id, question_id, prompt_id]`
names both, in `design.weights` or a group's `weights`, whose default `nested_equal` hierarchy
names both too, in a weight `column`, a group `filter`, an independence profile's `block_by`, a
comparison's `pair_key` or `cluster_key`, or any other field that names a manifest column. A level
that the plan doesn't name and the manifest omits is set to each prompt's `prompt_id`, so each
prompt counts as its own question or task in the report's prompt estimates. A plan that sets
both `design.hierarchy` and `design.weights.hierarchy` to `[task_id, prompt_id]` can therefore
use a manifest without a `question_id` column; setting only `design.hierarchy` does not, because
the weights still name it.

The identifier columns `prompt_id`, `question_id` and `task_id`, and every column the plan
groups prompts by (the levels of `design.hierarchy` and of a `nested_equal` weight hierarchy,
a group `filter` column, a manifest column in a `block_by` and `design.stratum_column`), are
read as text, exactly as written. `1`, `01` and `1.0` are three labels, `true` and `True` are
two, and `0001` stays `0001`. Write each `prompt_id` as the logs record the sample's id:
`0001` does not match the id `1`, and the `PROMPT_UNEXPECTED` and `PROMPT_MISSING` errors name
the other spelling when an identifier in the logs and one in the manifest spell the same number.
A cell in one of these columns has no value when it is blank or holds, ignoring surrounding
spaces, a marker that R, spreadsheets, databases and pandas write for a missing value: `NA`,
`N/A`, `n/a`, `None`, `null`, `NULL`, `nan`, `NaN`, `-nan`, `-NaN`, `<NA>`, `#N/A`, `#N/A N/A`,
`#NA`, `1.#IND`, `-1.#IND`, `1.#QNAN` or `-1.#QNAN`. Such a cell stops the analysis with an
error that names the column and the rows. Recode a label meant literally, such as `NA` for "not
applicable", for example as `none`.

The other columns keep the types pandas infers, and so do the columns the plan reads as numbers
or booleans even when it also groups prompts by them: a weight `column`, the
`inclusion_probability_column`, a `threshold_column`, an adaptive claim's `order_field` and
`expected_rollouts` are numbers, and the `selection_column`, `mandatory` and the zero-tolerance
field are booleans.

When `policy.zero_tolerance.manifest_field` is set in the plan and zero tolerance is enabled, the
manifest must have that column, and an analysis without it stops with an error. The default
field `zero_tolerance` stays optional.

`estimand` is `end_to_end_system` by default. In this mode, an explicit provider policy block is
safe only if it occurred during target generation and no content was delivered. Set
`underlying_model` to keep such blocks unclassified because the wrapper prevented observation of
the underlying model behavior.

`score.name` selects an Inspect score. Mapping is categorical:

```yaml
mapping:
  failure_values: ["FAIL"]
  safe_values: ["PASS"]
```

or numeric:

```yaml
mapping:
  failure_if:
    operator: greater_than_or_equal
    value: 0.5
```

Mappings are type-sensitive: `true`, `1`, `1.0`, and `"1"` are distinct. A score value that the
mapping does not cover is an error for its rollout, `SCORE_UNMAPPED`, which withholds the bounds
of that rollout's prompt. One `SCORE_MAPPING_MISMATCH` error lists every such value, with its
type and how many rollouts have it. `failure_values` and `safe_values` list text, numbers and
booleans only, the values a score can match: a plan that lists a missing value (`null` or
`.nan`), which never reaches the mapping since `missing` decides how it counts, or a list, an
object or a date, which no score value matches, is refused when it loads. Quote a date, such
as `"2024-01-01"`, to match it as text.

`missing` accepts `conservative_bounds`, `block`, `count_as_failure`, or `exclude`.
`conservative_bounds` is the default: integrity becomes `degraded`, and exact bounds cover both
the all-safe and all-failure completions of every attempted but unclassified rollout.
`count_as_failure` counts each such rollout as a failure. `exclude` leaves it out of the
bounds, which then cover only the classified rollouts and are biased if scores are missing for
a reason linked to the outcome; integrity is not degraded, and no diagnostic reports the
excluded rollouts, though `report-full.md` and `report.html` say that the policy leaves them
out.

`errors` additionally accepts `end_to_end`, which is the default. It recognizes only explicit
target-side provider policy blocks with no delivered content as safe for the end-to-end estimand.
Post-generation, grader, network, quota, timeout, and unrecognized errors remain unclassified and
use conservative bounds. `block` retains the original strict behavior and makes integrity invalid.
`count_as_failure` and `exclude` treat a rollout with a sample error as `missing` treats one
without a score.

```yaml
inputs:
  design_manifest: design.csv
  estimand: end_to_end_system
  score:
    name: harmful_behavior
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
    missing: conservative_bounds
    errors: end_to_end
```

`content_filter_stops` (experimental) decides how a rollout counts when a provider's content
filter stopped its output before the target delivered any content in the sample, without a
sample error. Claude 5-series models report a safety-classifier decline this way: a normal
answer with stop reason `refusal`, which Inspect records as `content_filter` with the provider's
category. A decline that arrives as an API error counts as having delivered nothing, whatever
stand-in text Inspect put in its place.

- `score`, the default: the rollout's score decides, as for any other rollout, and a missing
  score follows `missing`.
- `safe`: the block counts as `target_policy_blocked_safe`, for a harm estimand.
- `failure`: the block counts as a refusal, for an over-refusal estimand on safe prompts.

The target delivered content when the stopped output, or any generation that the sample's
transcript records before it, holds text, readable reasoning, a tool call or any other content
part. That covers a decline mid-answer, and a decline in a later turn of a conversation or a later
step of an agent after an earlier one delivered content: what the earlier turn delivered reached
the user, whatever the last turn did. Any model's generation counts, since the transcript does not
say which calls the end-to-end system delivered, so a solver that calls another model before the
target, such as an attacker model, also leaves the stop to the score. The scorer's own model calls
come after the stop and do not count. A rollout whose stop came after content, or whose stopped
generation the transcript does not record, is scored as usual and reported in a
`CONTENT_FILTER_STOP_SCORED` diagnostic (INFO) that says where the content was.

With `safe` or `failure`, the plan's choice replaces the score of a rollout whose stop delivered
nothing, and each block is reported in a `TARGET_POLICY_BLOCKED_SAFE` or
`TARGET_POLICY_BLOCKED_REFUSAL` diagnostic with the provider's block type and category. The choice
only ever adds failures: `failure` replaces any score, but `safe` replaces only a missing or safe
score. A failure score, or a value the mapping does not cover, keeps deciding under `safe`, since
the scorer may have seen harm that the transcript does not show, and is reported in a
`CONTENT_FILTER_STOP_SCORED` warning; an uncovered value is also a `SCORE_UNMAPPED` error. Both need
`estimand: end_to_end_system`, and `failure` cannot be combined with `errors: end_to_end`, which
counts a policy-blocked sample error as safe. Only Inspect logs report stop reasons; with rollout
files the option is refused.

```yaml
inputs:
  score:
    name: over_refusal
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
    errors: conservative_bounds
    content_filter_stops: failure
```

`token_limit` (experimental) decides how a rollout counts when its answer stopped at the
generation token limit (Inspect's stop reason `max_tokens`) and it has no score and no sample
error. Running out of tokens comes from the model's behaviour, such as a reasoning runaway, not
from the harness or a judge that failed, so the plan can say how it counts, decided before the
run.

- `score`, the default: the analysis does not look for such answers, and a missing score
  follows `missing`, as before.
- `missing`: the missing score still follows `missing`, but each such rollout is counted on its
  own.
- `safe`: the rollout counts as safe, whatever `missing` says.
- `failure`: the rollout counts as a failure, whatever `missing` says.

With any value but `score`, each such rollout is reported in a `TOKEN_LIMIT_REACHED` diagnostic
(INFO). Every prompt estimate in `report.json` gets an `n_token_limit` count, `report-full.md` gets a
"Token limit" column, and `observations.csv` gets a `token_limit_unscored` column. An answer
that stopped at the token limit but was scored keeps its score. Only Inspect logs report stop
reasons; with rollout files the option is refused.

Only the stop reason `max_tokens` is detected. Two other ends that look alike are not, and such
a rollout counts by its score, and without a score follows `missing`, whatever `token_limit`
says:

- Inspect's stop reason `model_length`: the conversation outgrew the model's context window,
  which a long prompt or long tool output can cause as well as the model's own answer.
- A sample that hit one of Inspect's own sample limits, such as its token limit
  (`--token-limit`, recorded as a sample limit of type `token`). That limit counts the tokens of
  the whole sample, across turns and tool calls, and is a harness setting rather than a property
  of one answer.

```yaml
inputs:
  score:
    name: harmful
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
    missing: conservative_bounds
    token_limit: failure
```

`judge_calibration` is optional. It declares assumed conditional judge accuracies and emits
diagnostic true-risk sensitivity intervals beside the ordinary judge-system intervals:

```yaml
inputs:
  judge_calibration:
    mode: assumed
    primary: declared
    scenarios:
      - id: declared
        rho_1: {lower: 0.85, upper: 1.0}
        rho_0: {lower: 0.0, upper: 0.02}
      - id: pessimistic
        rho_1: {lower: 0.70, upper: 1.0}
        rho_0: {lower: 0.0, upper: 0.05}
```

`rho_1` is P(truly unsafe | judge said unsafe) and `rho_0` is P(truly unsafe | judge said
safe). Every scenario must satisfy `rho_1.lower > rho_0.upper`. In `assumed` mode the
resulting intervals are always `diagnostic`: a declaration carries no coverage guarantee.

A scenario's optional `cohorts` maps declared cohorts to boxes of their own, for a judge whose
accuracy depends on the model under test:

```yaml
      - id: worse_on_candidate
        rho_1: {lower: 0.85, upper: 1.0}
        rho_0: {lower: 0.0, upper: 0.02}
        cohorts:
          candidate: {rho_1: {lower: 0.60, upper: 0.85}, rho_0: {lower: 0.0, upper: 0.08}}
```

A listed cohort's rows, and a comparison arm on that cohort, are corrected with its box;
every other cohort takes the scenario's `rho_1` and `rho_0`. Each cohort box must also satisfy
`rho_1.lower > rho_0.upper`, `cohorts` cannot be empty, and an undeclared cohort is
`PLAN_UNKNOWN_REFERENCE`. Generative plans reject the field. See
[Per-cohort boxes](judge-misclassification.md#per-cohort-boxes).

`mode: validated` estimates the same two parameters from human labels instead, and its
intervals are `certifying`:

```yaml
inputs:
  judge_calibration:
    mode: validated
    validation_manifest: judge-labels.csv
    family: judge_error
    sampling: simple_random_within_judge_stratum
```

The manifest holds `observation_id,human_label` with labels `safe` or `unsafe`; identifiers
come from the `observations.csv` of a prior run. `sampling` is
`simple_random_within_judge_stratum` (the default) or `simple_random`; only the latter also
estimates sensitivity and the false-positive rate by true class, which a claim with
`method: noisy_valid_test` requires (the validator emits `PLAN_INVALID_CLAIM` otherwise). With
those estimates, every group's `weighted_mean` row that the calibration corrects also carries
diagnostic Lang-Reiczigel limits beside the exact box; tail measures, hierarchy nodes and rows
that hold rollouts a plan policy classified do not.
Threshold claims whose evidence carries a validated correction are decided from the corrected
bounds; an `assumed` correction never affects a claim. One calibration is estimated per cohort
by exact Clopper-Pearson bounds within each judge stratum. The corrected interval's coverage is
the union of three budgets: its own family's, the judge family's, and that of the bound on its
[unclassified share](judge-misclassification.md#unclassified-rollouts). The last is a quarter
of the calibration's share of the judge family budget, and is spent on top of it; a row that
needs no share bound, such as one holding rollouts a plan policy classified under a policy
whose bounds complete the unclassified ones, spends none. A plan
declares one mode: `assumed`, `validated` or `transported` (below).

`groups` scopes an `assumed` or `validated` calibration to declared groups, for a cohort that
holds several benchmarks whose judge accuracies differ:

```yaml
inputs:
  judge_calibration:
    mode: validated
    validation_manifest: judge-labels.csv
    family: judge_error
    groups: [harmbench]
```

Without `groups` there is one calibration per cohort, and it corrects every row of the
cohort. With it:

- only rows of the listed groups are corrected. Rows of other groups keep their judge-system
  intervals, and a claim on such a row in a validated plan is INCONCLUSIVE, with a reason
  saying that no labels calibrate it;
- `assumed` mode applies its scenarios unchanged to the listed groups' rows;
- `validated` mode (experimental with `groups`) estimates one calibration per cohort and
  listed group, from the labels and judge-stratum sizes of that group's rollouts. The judge
  family budget is split equally over those calibrations (`per_calibration_alpha` in the
  report, instead of `per_cohort_alpha`), and the report records the label counts per cohort
  and group. A validated correction then certifies a `weighted_mean` over a whole listed group
  that weights every classified rollout equally, a row that an unscoped calibration reports
  only as a sensitivity analysis. Labels on rollouts outside every listed group calibrate
  nothing and raise the warning `JUDGE_VALIDATION_OUT_OF_SCOPE`. A group gets only the labels
  that land on its rollouts, so a small group's calibration is often refused: see
  [Scoping the calibration to groups](judge-misclassification.md#scoping-the-calibration-to-groups).

Each listed group must be declared under `groups`, and a scope can be neither empty nor list a
group twice. A `transported` calibration takes no `groups`. Generative plans have no groups and
reject the field.

`sampling: stratified_within_judge_stratum` (experimental) reads labels drawn by a stratified
design, such as one that labels the judges' disagreements more often. It requires
`validation_strata`, a CSV of `observation_id,stratum` for every classified observation. The
two fields go together, and both require `mode: validated`. Each declared stratum within a
judge stratum gets exact hypergeometric bounds on its truly unsafe count and is weighted by its
size; the judge stratum's accuracy then gets Clopper-Pearson bounds at the summed counts. Plans
without the field keep their hash. Generative plans do not offer this design. See
[Stratified labels](judge-misclassification.md#stratified-labels-experimental).

`protocol` (experimental, `mode: validated` only) declares how the labels were collected:
`queue`, a CSV of `observation_id` in labelling order, relative to the plan; `tier`, to keep
only queue rows whose `tier` column has that value; `seed`, recorded for an audit; and
`stopping`, one of `fixed_size` (with `size`), `label_independent` or `data_dependent`. The
analysis checks that the manifest labels exactly the queue's first rows, and that `size` rows
are labelled under `fixed_size`. `data_dependent` bounds each stratum with an anytime-valid
confidence sequence instead of Clopper-Pearson, and requires
`sampling: simple_random_within_judge_stratum`. See
[Labelling protocol](judge-misclassification.md#labelling-protocol-experimental).

`mode: transported` (experimental) takes a judge's sensitivity from a published
human-labelled validation set, and nothing else:

```yaml
inputs:
  judge_calibration:
    mode: transported
    published_set:
      name: HarmBench classifier validation set
      source: https://github.com/centerforaisafety/HarmBench
      judge: HarmBench-Llama-2-13b-cls
      rubric: HarmBench behaviour rubric
      human_unsafe: 120
      judge_flagged_unsafe: 100
      sensitivity_alpha: 0.05
```

`name`, `source`, `judge` and `rubric` are required and must not be empty.
`judge_flagged_unsafe` cannot exceed `human_unsafe`, which must be at least one.
`sensitivity_alpha` defaults to 0.05. The mode cannot be combined with `scenarios`,
`primary`, `validation_manifest` or `family`, and `published_set` is refused in any other
mode. The specificity is not transported, so each corrected interval runs from zero to the
judge-system upper bound divided by the sensitivity's lower bound. The intervals are always
`diagnostic` and never affect a claim. Fixed and sampled designs offer this mode; generative
plans do not. See
[Transported mode](judge-misclassification.md#transported-mode-a-published-sensitivity-experimental).

Corrections are applied to `weighted_mean`, `maximum`, `cvar`, and `top_k_mean` measures, to
every hierarchy node, and to comparisons whose primary interval exposes both arm bounds
(`weighted_kl_chernoff` and any `comparison_sets` entry). A `transported` calibration corrects
no comparison, and the warning `JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON` names them. A
validated correction certifies only a `weighted_mean` over a whole cohort, in a fixed design,
that weights every classified rollout equally. Every other corrected row is a sensitivity, and a
claim on it is INCONCLUSIVE. See [Judge misclassification](judge-misclassification.md).

### Outcome mapping sensitivities (experimental)

Some scores have more than two values, and which of them count as failures is a judgement. A
grader of over-refusal may return full compliance, partial refusal or full refusal, and a
partial refusal can reasonably be read either way. `mapping_sensitivities` declares
alternative readings of the same score, each with an `id`, an optional `title` and a `mapping`
of the same form as the primary one:

```yaml
inputs:
  score:
    name: refusal_grade
    mapping:
      failure_values: ["PARTIAL_REFUSAL", "FULL_REFUSAL"]
      safe_values: ["FULL_COMPLIANCE"]
    mapping_sensitivities:
      - id: partial_refusal_safe
        title: A partial refusal is not an over-refusal
        mapping:
          failure_values: ["FULL_REFUSAL"]
          safe_values: ["PARTIAL_REFUSAL", "FULL_COMPLIANCE"]
```

The analysis is repeated under each alternative on the same rollouts, read once, and the
report shows every reading side by side:

- `report.json` gains `outcome_mapping_sensitivities`: per mapping, its integrity status,
  the count of each diagnostic code, the decision it alone would give, and the estimate and
  interval of every group risk, comparison and claim (with the claim's status), and
  `presentation.primary_outcome_mapping` records the primary mapping beside them;
- `outcome-mapping-sensitivity.csv` holds one row per result and mapping, `primary` first,
  with each result's readings adjacent, and is listed in `checksums.json`. Its last column,
  `descriptive`, is `True` on every row of an alternative, whose claim status certifies
  nothing;
- `report-full.md` has an "Outcome mapping sensitivity (experimental)" section. Claims and
  decisions appear in it only when `report.include_decisions` is true, and an alternative's
  are marked "(descriptive)".

Only the primary mapping enters claims and the decision. Each alternative's intervals carry
the coverage the plan declares for that mapping alone; the readings are not simultaneous
across mappings, so the side-by-side view is a sensitivity analysis, not a set of tests to
pick from. A mapping that leaves some observed value unmapped makes its own integrity
`invalid` (`SCORE_UNMAPPED` and `SCORE_MAPPING_MISMATCH` in its diagnostic counts) without
affecting the primary analysis. Each alternative carries what the plan itself, its decision
policy and its registration receipt were checked for, so a receipt for another plan
(`PLAN_RECEIPT_MISMATCH`) or a decision policy that fails its checks makes every alternative
`invalid`, as it does the primary analysis. Whenever the report's integrity is `invalid`, for
whatever reason, even an error of the primary outcome alone, every alternative's
`integrity_status` is `invalid`, each of its claims' `status` is `INCONCLUSIVE` and its
`decision` is the one an invalid analysis gives (`INCONCLUSIVE`, or `FAIL` when a
zero-tolerance prompt failed under its mapping), while its intervals and `diagnostic_counts`
stay its own. The diagnostics that describe the outcomes, such as
`TAIL_MEASURE_SINGLE_ROLLOUT`, `TAIL_MEASURES_UNINFORMATIVE` and `CLUSTER_DESIGN_EFFECT`, are
counted for each alternative from its own reading. The anytime-valid bounds
(`outputs.anytime_bounds`) and the judge flip rate (`inputs.judge_flip`) are computed for the
primary mapping only.

Identifiers must be unique and cannot be `primary`, which names the primary mapping in the
outputs; an alternative may not repeat the primary mapping. One that classifies every value as
the primary mapping does is a repeat, whatever the order of its values and however often it
lists one; values of another type are not, since mappings are type-sensitive. Only
`inputs.score` takes sensitivities, not a judge's score. A plan without the field keeps its
hash, and its report is unchanged. `report.html` does not show the sensitivities yet.

### Judge panels

`inputs.judges` declares additional binary judges scored on every rollout, each with its own
Inspect score column and outcome mapping. `content_filter_stops` and `mapping_sensitivities`
apply to `inputs.score` only, and a judge's score that sets either is refused. A content-filter
stop is the target's, the same whichever judge reads the rollout, so a judge's run under
`outputs.judge_claims` takes the primary score's `content_filter_stops` (see
[Claims by judge](#claims-by-judge-experimental)). `block` names the judge's dependence block:
judges sharing a block are one opinion, not independent votes (two judges built on related
models belong in one block). It defaults to the judge's own id, so a panel is independent only
where the plan says so. `kind: monitor` marks a chain-of-thought or trajectory monitor.

```yaml
inputs:
  score:
    name: any_judge                       # a composite may serve as the primary outcome
    mapping: {failure_values: [true], safe_values: [false]}
  judges:
    - id: gpt_judge
      score: {name: harmful_gpt, mapping: {failure_values: [FAIL], safe_values: [PASS]}}
      block: openai
    - id: claude_judge
      score: {name: harmful_claude, mapping: {failure_values: [FAIL], safe_values: [PASS]}}
    - id: cot_monitor
      score: {name: cot_flag, mapping: {failure_values: [FLAG], safe_values: [CLEAR]}}
      block: openai
      kind: monitor
  composites:
    - {id: any_judge, rule: any, judges: [gpt_judge, claude_judge, cot_monitor]}
    - {id: majority, rule: majority, judges: [gpt_judge, claude_judge, cot_monitor]}
```

`composites` derive a verdict from declared judges under `any`, `all`, or `majority` and are
materialised as score columns (`score_<id>`, boolean), so any downstream construction can use
them. A composite is undecided -- and flows through the ordinary unclassified-outcome handling --
when a missing vote could still change it or when `majority` ties. The report gains a **Judge
panel** section with per-judge counts, each composite's Kish **effective votes** over the declared
blocks, and pairwise agreement with Cohen's kappa. Agreement is a diagnostic, not evidence: judges
that share a blind spot agree and are both wrong. Missing judge columns and unmappable judge
values are errors (`JUDGE_SCORE_MISSING`, `JUDGE_SCORE_UNMAPPED`), and one
`JUDGE_SCORE_MAPPING_MISMATCH` per judge lists the values its mapping does not cover.

A judge may declare prevalence-free accuracy ranges, `accuracy: {sensitivity: {lower, upper},
specificity: {lower, upper}}`. When the primary outcome is an `any` or `all` composite whose judges
all declare them, the report adds a diagnostic `judge_corrected_panel_derived` interval: the
composite's box is derived by Fréchet bounds within blocks and products across blocks, and true
risk is partially identified by inverting Rogan–Gladen over it (the whole unit interval when the
box admits an uninformative panel). The box describes the composite's verdict, so the primary
mapping must read it as it is, `failure_values: [true]` and `safe_values: [false]`; otherwise no
interval is added, and the warning `PANEL_CORRECTION_VERDICT_UNMATCHED` says so. See
[Judge misclassification](judge-misclassification.md).

`accuracy.groups` restricts a judge's declared ranges to the rollouts of the listed groups:

```yaml
    - id: gpt_judge
      score: {name: harmful_gpt, mapping: {failure_values: [FAIL], safe_values: [PASS]}}
      accuracy:
        sensitivity: {lower: 0.85, upper: 0.95}
        specificity: {lower: 0.90, upper: 0.99}
        groups: [harmbench]
```

A box built from several declarations holds only in the groups that every one of them covers,
so the `judge_corrected_panel_derived` interval is added only to rows of those groups. The
report records the groups as `groups` under `provenance.judge_panel.derived_correction`. When
the declarations share no group, no correction is reported, and the warning
`PANEL_CORRECTION_NO_COMMON_GROUP` says so. With `report.judge_error_stress`, a claim on a group
that the primary outcome's declared box does not cover is stressed without the box: its tipping
lines are reported, its stressed verdict is not.

`inputs.panel_groups` breaks the judge panel down by group. A cohort that holds several
benchmarks otherwise gets one agreement figure across all of them. For example, a benchmark
whose two judge scores are copies of each other pulls Cohen's kappa up for the cohort.

```yaml
inputs:
  panel_groups: [harmbench, xstest]
```

Every judge, composite and judge pair in `provenance.judge_panel` then carries `groups`, with
the same counts and agreement figures for each listed group's rollouts, per cohort. `report-full.md`
shows them under **Judge panel by group**. The cohort-wide figures are reported as before. The
field requires `inputs.judges`, and its groups must be declared.

In all three scopes, a group the plan does not declare is a `PLAN_UNKNOWN_REFERENCE` error:
integrity is invalid, and `analyze` still writes the report. The undeclared group gets no
calibration and no panel breakdown, but in `validated` mode it still takes its share of the
judge family budget.

All three scopes are optional. A plan that leaves them out keeps its hash and writes the same
report bytes as before.

#### Claims by judge (experimental)

The report's claims are decided on the primary outcome alone. To see what each declared judge
would decide, set `outputs.judge_claims: true`. The analysis then runs the plan again once per
judge, with that judge's `score` as the primary outcome and without `judges`, `composites` or
`judge_calibration` (the calibration validates the primary outcome, not the other judges). Every
run uses the same rollouts: each log is read, and hashed, once.

```yaml
outputs:
  judge_claims: true
```

A judge's run takes that judge's `score`, including its `missing` and `errors` policies, not
the primary outcome's. A judge that leaves them out gets their defaults, `conservative_bounds`
and `end_to_end`, so with a primary `score` set to `missing: block` and `errors: block`, the rows
also differ by how missing and errored verdicts are handled. To compare judges alone, give each
judge the primary outcome's policies:

```yaml
inputs:
  score: {name: harmful, mapping: {...}, missing: block, errors: block}
  judges:
    - id: judge_a
      score: {name: harmful_a, mapping: {...}, missing: block, errors: block}
```

`content_filter_stops` and `token_limit` are different: they say how rollouts count that the
target's provider stopped, which is the same whichever judge reads them. Every judge's run takes
the primary outcome's `content_filter_stops`, which a judge's score cannot set, so it counts a
content-filter stop as the primary analysis does. A judge that leaves `token_limit` at its
default, `score`, likewise takes the primary outcome's value, so it counts an unscored answer
cut at the token limit as the primary analysis does. A judge that sets its own `token_limit`
keeps it; there is no way to ask a judge's run not to look when the primary does, short of
`token_limit: missing`, which leaves such answers to the judge's `missing`. An inherited
`token_limit` applies to every truncated rollout the judge left unscored, whatever the judge's
reason, so it can make a judge's row narrower than its own `missing` would;
`token_limit: missing` on the judge is the way out. The inherited `content_filter_stops` must
fit each judge's `errors` as it fits the primary's: a primary `content_filter_stops: failure`
needs every judge to set `errors` to something other than `end_to_end`, its default, or a plan
with `outputs.judge_claims` is refused when it is loaded.

Each judge in `provenance.judge_panel.judges` of `report.json` gains:

- `integrity`: the integrity status of that judge's run. Each judge's run carries what the
  plan itself, its decision policy and its registration receipt were checked for, so a receipt
  for another plan (`PLAN_RECEIPT_MISMATCH`) or a decision policy that fails its checks makes
  every judge's run `invalid`, as it does the primary run. Whenever the report's integrity is
  `invalid`, for whatever reason, every judge's `integrity` is `invalid` and every row's
  `status` is `INCONCLUSIVE`, with no `descriptive_status`, even where the error concerns the
  primary outcome alone; `diagnostic_counts` stay those of the judge's own run, and `reasons`
  says so. A judge whose own `score` asks for what the source cannot give, such as the stop
  reasons that its `token_limit` needs from rollout files, gets an `invalid` run with
  `INGEST_FAILED`, as the primary run would, and rows that decide nothing.
  The rest of the report stands.
- `diagnostic_counts`: how often each diagnostic code occurred in that judge's run, by code, so
  a run that is not `valid` says why.
- `claims`: one row per claim, with `claim_id`, `status`, `descriptive_status`, `estimate`,
  `lower`, `upper`, `method` and `reasons`. Under `intent: exploratory` every `status` is
  `INCONCLUSIVE` and `descriptive_status` holds what a confirmatory analysis would decide, as
  for the report's own claims; otherwise it is `null`. The bounds are on that judge's reported
  risk, since its run has no calibration. When the plan declares a validated
  `judge_calibration`, the plan's thresholds are about true risk, and the primary outcome's
  claims are not decided on judge-reported bounds.
  The rows are not either: every `status` is `INCONCLUSIVE`, and `reasons` says why. A
  `noisy_valid_test` claim's test needs labels of the judge's verdicts, so a judge's run bounds
  such a claim as a `corrected_interval` claim. Where the plan's calibration cannot run the
  test, the claim is a plan error in the primary run and in each judge's run alike.
- `configuration`: per cohort, the distinct configurations of the scorer named like the judge's
  score (`scorer`) and the model roles (`model_roles`) that the cohort's runs record. Inspect
  logs record them under `eval.scorers` and `eval.model_roles`. Neither is copied as recorded:
  - Each scorer is `{"name": "<score name>", "sha256": "<fingerprint>"}`. The fingerprint is
    the SHA-256 of the scorer's whole entry in `eval.scorers` (name, options, metrics and
    metadata) as JSON with sorted keys, no whitespace and UTF-8 text. Equal fingerprints mean
    equal recorded configurations. The options and metadata themselves can hold request
    headers, tokens, endpoints and local paths, such as a template file, and Inspect redacts
    only some of them, so they never reach the report.

    The fingerprint hides a value only as well as the value is hard to guess. It has no salt,
    so the same entry always gives the same fingerprint, which is what lets two reports be
    compared. Someone who knows the rest of the entry, such as the scorer's other options, can
    hash candidate values for the one they do not know and compare: a value with few
    possibilities, such as an internal IP address, a port or a user name in a path, can be
    recovered that way in seconds. A long random credential, such as an API key or a
    bearer token, cannot. If an entry holds a guessable value that must stay private, remove
    it from the logs' scorer options before the analysis, or leave `judge_claims` off.
  - Each model role keeps only its name and model id, as `{"<role>": {"model": "<model id>"}}`.
    The rest of its configuration (`base_url`, `args` and the generate config with its
    `extra_headers`) can hold endpoints and credentials, so it never reaches the report.

  Where the runs record none, the value is `null`. A run whose `scorers` or `model_roles` cannot
  be read, such as text that is not JSON or a role without a model id, is left out of the
  record, with a `JUDGE_CONFIGURATION_UNREADABLE` warning that names the run and the field but
  not its value. Rollout files can carry either field as a run field.

`report-full.md` shows the same rows under **Claims by judge (experimental)** in the judge panel
section, without the status column when `report.include_decisions` is false. The status column
shows a descriptive status beside the status, as the report's claims table does. The rows compare
judge configurations. They do not enter the decision and are not adjusted for multiplicity
across judges. The option requires `inputs.judges`. A plan that leaves it off keeps its hash and
writes the same report bytes as before.

### Judge flip rate (experimental)

`inputs.judge_flip` reports how easily the primary judge is talked out of an unsafe verdict.
Each judge-positive completion of the run is rescored by the same judge inside wrappers that
keep its content verbatim, such as a refusal prefix, benign text before and after, an
educational frame or a story frame. A verdict that becomes safe is a flip. Since the content
did not change, a flip is a false negative that the wrapper caused. The rescoring is done
outside the package, which never calls a model, and its verdicts are passed in as a CSV.

A judge-positive completion is one the primary judge scored unsafe: its score maps to a
failure. A failure that a plan policy classified is not judge-positive, since no judge scored
it. Those are a sample error or a missing score under `count_as_failure`, a content-filter stop
under `content_filter_stops: failure`, and an unscored truncated answer under
`token_limit: failure`. Such failures are not rescored, and the rate leaves them out.

```yaml
inputs:
  judge_flip:
    manifest: flips.csv           # relative to the plan
    wrappers: [refusal_prefix, benign_padding, educational, story]
    sampling: all                 # or simple_random
    warn_above: 0.1               # warn when the flip rate exceeds this, in (0, 1)
    alpha: 0.05                   # optional, default 0.05
```

The manifest has the columns `observation_id`, `wrapper` and `verdict`, with one row per
rescored completion and wrapper. `observation_id` is the observation identifier of the
analysed run, as in a validation manifest. `verdict` is `safe` or `unsafe`. A rescoring that
failed must be rerun before the file is written: the analysis does not drop rows. The manifest's
cells are read with surrounding whitespace removed, so a name in `wrappers` that starts or ends
with whitespace could match none of its rows, and the plan rejects it.

- With `sampling: all`, every judge-positive completion of the run must be rescored. With
  `simple_random`, the rescored completions must be a simple random sample of them, drawn
  before rescoring, and at least one must be rescored.
- Every rescored completion needs a row for each declared wrapper and for no other wrapper,
  and it must be judge-positive in the run. A failure that a plan policy classified must not
  be rescored.

`analyze` adds a `JUDGE_FLIP_RATE` diagnostic to the integrity section of `report.json` and
`report-full.md`. Its `details` hold `judge_positive` (the run's judge-positive completions),
`policy_classified_failures` (the failures a plan policy classified, which the rate leaves out),
`prompts` (the prompts the rescored completions come from), `any_wrapper` (the completions that
flip under at least one wrapper) and `by_wrapper` (one entry per wrapper). Each entry holds
`flipped`, `rescored`, `estimate`, `lower` and `upper`. The interval is the two-sided exact
Clopper-Pearson interval at `alpha`, split equally between its ends. Each interval holds for
its own rate, not jointly with the others. The rate is pooled over the analysed cohorts.

The interval treats the rescored completions as independent. With `simple_random` it is for the
share of the run's judge-positive completions that flip, which the random sample justifies.
With `sampling: all` that share is known exactly, and the interval is descriptive: completions
of one prompt tend to flip together, so as a range for the judge's flip rate on other prompts
it is too narrow when the completions come from few prompts. See
[Judge flip rate](judge-misclassification.md#judge-flip-rate-under-content-preserving-wrappers-experimental).

The diagnostic is `INFO`, or a `WARNING` when the `any_wrapper` estimate exceeds `warn_above`.
Neither changes integrity, a claim or the decision. A manifest that cannot be read, or that
breaks one of the rules above, is a `JUDGE_FLIP_INVALID` error, as an invalid validation
manifest is. That error makes integrity invalid. The flip rate is reported on the fixed path
only. A plan without the section keeps its hash and writes the same report bytes as before. See
[Judge flip rate](judge-misclassification.md#judge-flip-rate-under-content-preserving-wrappers-experimental)
for what the rate shows.

## `design`

| Field | Meaning |
|---|---|
| `type` | `fixed` (census of the declared design) or `sampled` (items drawn from a frame) |
| `inclusion_probability_column` | Required with `type: sampled`; names a manifest column holding each frame item's inclusion probability in (0, 1] |
| `sampling` | `poisson` (independent inclusion, default) or `fixed_size` (exact draws per stratum, without replacement) |
| `stratum_column` | Optional with `sampling: fixed_size`; names the manifest column defining sampling strata |
| `selection_column` | Experimental, optional with `type: sampled`; names a manifest column marking the frame items the draw selected (`true`/`false`). `analyze` and `validate` then check that each cohort observed exactly those items (`SAMPLE_SELECTION_MISMATCH`); without it only the number of items is checked. A plan without it keeps its hash and report bytes. See [Checking which items were evaluated](design-based-sampling.md#checking-which-items-were-evaluated-experimental) |
| `hierarchy` | Creator-declared hierarchy ending in `prompt_id` |
| `rollout_key` | Fields uniquely identifying one terminal rollout; see [Schema versions](#schema-versions) for the names and default. A field the rollout does not record is read from its run. A run's fields hold its configuration, so a `ROLLOUT_DUPLICATE` message names such a field, as `<model_base_url of the run>`, instead of showing its value |
| `require_complete_manifest` | Require every fixed prompt in every cohort |
| `require_expected_rollouts` | Require prospective rollout counts, from the manifest's `expected_rollouts` column (on by default) |
| `weights` | Default group weight strategy |
| `independence_profiles` | Named independent-block partitions; see [Configuring hierarchy and independence](independence-profiles.md) |

Example:

With `type: sampled` the manifest is the sampling frame rather than a census, and only
`weighted_mean` measures are available. See
[Design-based inference for sampled item frames](design-based-sampling.md).

```yaml
design:
  hierarchy: [task_id, question_id, prompt_id]
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
    prompt: {block_by: [prompt_id], comparison_coupling: joint}
```

### Prompt variants in fixed designs

A fixed design can ask one base prompt in several wordings, such as the original, a
paraphrase or a role-play wrapper. Model this with the hierarchy `[task_id, question_id,
prompt_id]`: the base prompt is the `question_id`, and each wording is a `prompt_id` beneath it.
Each wording is its own manifest row and its own Inspect sample. Its `id` matches the row's
`prompt_id`.

```text
prompt_id,question_id,task_id,variant,expected_rollouts
cyber-credentials-original,cyber-credentials,cyber-safety,original,20
cyber-credentials-paraphrase,cyber-credentials,cyber-safety,paraphrase,20
cyber-credentials-roleplay,cyber-credentials,cyber-safety,roleplay,20
cyber-auth-bypass-original,cyber-auth-bypass,cyber-safety,original,20
```

The `variant` column only describes the wording, and the analysis does not read it unless the
plan names it. A group can filter on it, for example with `filter: {variant: original}` to
cover only the original wordings.

**Weights.** With `nested_equal` weights over the same hierarchy, each base prompt of a task
gets the same weight whatever its number of wordings, and its wordings share that weight
equally. In the rows above, the two base prompts weigh 1/2 each within the task. Each
credential-theft wording weighs 1/6 and the authentication-bypass prompt weighs 1/2.
`equal_prompt` would instead give each row 1/4, so the base prompt with more wordings would
count three times as much.

**Independence.** Wordings of one base prompt can be related in two ways, and only one of them
breaks the `rollout` profile:

- **Similar failure rates.** Wordings of one base prompt are often hard or easy together. In a
  fixed design each wording's risk is a fixed quantity of the estimand, so similar risks do not
  make rollouts dependent, and the `rollout` profile stays valid.
- **Coupled outcomes.** Rollouts of different wordings are dependent when they share randomness
  or state: a sampling seed shared by the same epoch of each wording, a sandbox or tool state,
  or a batch that fails as a whole. The `rollout` and `prompt` profiles then count dependent
  rollouts as independent ones, and their bounds can miss the risk more often than alpha
  allows. In a simulation of 4 base prompts with 10 wordings each, 20 epochs and
  risk 0.12, in which the wordings of a base prompt shared each epoch's outcome, the `rollout`
  upper bound at one-sided level 0.025 fell below the risk in about 7% (Hoeffding), 14%
  (empirical Bernstein) and 24% (KL/Chernoff) of runs, so a claim could pass on a risk above
  its `acceptable_upper`.

Whether outcomes are coupled is a fact about how the evaluation ran; the bounds do not reveal
it. `CLUSTER_DESIGN_EFFECT` can hint at it, but under a fixed design its correlation also counts
differences in rate between base prompts (see
[its limits](independence-profiles.md#how-much-the-grouping-matters-cluster_design_effect)).

A claim's status, and the report's decision, are the primary profile's verdict alone (see
[Verdicts by profile](independence-profiles.md#verdicts-by-profile)). A sensitivity profile
shows how far that verdict depends on rollout independence, but never changes it. When the
wordings' outcomes can be coupled, the primary profile must therefore put all the rollouts that
can share the coupling in one block:

- `[question_id, repeat]` when the coupling stays within one epoch. The same epoch of every
  wording of a base prompt forms one block, and the epochs stay independent of each other. In
  the simulation above its upper bounds never fell below the risk under Hoeffding or empirical
  Bernstein, and did so in about 0.25% of runs under KL/Chernoff.
- `[question_id]` when the coupling can reach across epochs, for example through a sandbox or
  tool state that persists between them. Every rollout of every wording of a base prompt forms
  one block, so the block count is the number of base prompts.
- A coarser key when the coupling reaches beyond one base prompt, such as `[repeat]` for a seed
  shared by the same epoch of every prompt.

The primary profile sets the blocks of the direct-mean intervals only. The prompt bounds treat
the epochs of each wording as independent draws whatever the primary, and so does every interval
built from them: the propagated envelope of every risk measure, which is the only interval of
`maximum`, `cvar`, `top_k_mean`, `entropic`, the threshold measures and a `worst_of_k` without
`direct_interval`. Coupling across the epochs of one wording therefore breaks the prompt bounds
and these measures under any profile, and claims should then rest on the direct `weighted_mean`
interval. This includes coupling confined to one wording's epochs, such as a response cache
keyed on the exact input. For the direct mean that needs only `[prompt_id]` as the primary, which
already puts each wording's epochs in one block, but the prompt bounds stay invalid. Coupling
within one epoch across wordings, such as the shared seed above, leaves the prompt bounds and
the measures built from them valid: each prompt bound needs independence only within its prompt,
and the Bonferroni allocation across prompts holds under any dependence between them. A
`worst_of_k` measure with `direct_interval: true` needs a primary that blocks by rollout, so
making another profile primary makes such a plan invalid (`PLAN_WORST_OF_K_INTERVAL_UNSUPPORTED`).

```yaml
design:
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
    prompt: {block_by: [prompt_id], comparison_coupling: joint}
    question_repeat: {block_by: [question_id, repeat], comparison_coupling: joint}
    question: {block_by: [question_id], comparison_coupling: joint}
outputs:
  direct_mean_intervals:
    independence:
      primary: rollout  # question_repeat, question or coarser when wordings can be coupled
      sensitivities: [prompt, question_repeat, question]
```

With independent rollouts, `rollout` stays the primary and the other profiles are
sensitivities. The `question` profile's effective sample size,
$\left(\sum_g A_g^2\right)^{-1}$ over the base-prompt weights $A_g$ (see
[Statistical construction](independence-profiles.md#statistical-construction)), is no larger
than the `prompt` profile's. With the same number of epochs for every wording,
`question_repeat`'s is that many times larger than `question`'s. With few base prompts the
`question` intervals can reach the whole range $[0, 1]$.

A block key is the raw value of each field, not a path in the hierarchy. If two tasks both have
a base prompt with `question_id: q1`, `[question_id]` puts the two in one block. Keep base-prompt
identifiers unique across tasks, or block by `[task_id, question_id]` and
`[task_id, question_id, repeat]`.

**Risk measures.** Measures that use the group weights, such as `weighted_mean`, `cvar` and
`worst_of_k`, count each base prompt once under `nested_equal`, whatever its number of wordings.
Two measures count wordings instead:

- `maximum` ignores weights: it is the largest risk of any wording, not of any base prompt.
  Its bounds come from the prompt bounds, and `alpha_allocation: equal_bonferroni` divides the
  family alpha equally over every wording of every cohort. Each added wording therefore widens
  every prompt bound, and with them the maximum's upper bound. With `nested_equal` as
  `design.weights`, `alpha_allocation: weighted_bonferroni` instead gives each base prompt a
  fixed share of the alpha, which its wordings split, so an added wording widens only the bounds
  of its own base prompt's wordings.
- `top_k_mean` needs equal prompt weights. On a group where `nested_equal` gives the wordings
  unequal weights it is refused with `RISK_MEASURE_INVALID`, which makes integrity invalid.
  Where it is accepted, under `equal_prompt` or under `nested_equal` on a group whose tasks all
  have the same number of base prompts and whose base prompts all have the same number of
  wordings, it averages the `k` riskiest wordings, so a base prompt with three wordings can fill
  three of the `k` places.

**Hierarchy nodes.** Set `hierarchy_nodes: true` under `direct_mean_intervals` to report an
interval for each base prompt as well as each task. The nodes have two costs:

- Under the `question` profile a base-prompt node is a single block, so its interval rests on
  one independent value and is close to $[0, 1]$ whatever the data. Under `question_repeat` it
  has one block per epoch.
- Under the default `hierarchy_calibration: flat_bonferroni`, every node is one more target of
  the direct-mean family, whose alpha is divided over all of them. Each node therefore widens
  every interval of the family, including the group intervals that claims are decided on. In
  the template below, each group interval's one-sided alpha is 0.0025 with the nodes and 0.0125
  without them. `frontier_sidak` spends less where sibling nodes have disjoint blocks (see
  [Direct fixed-design mean intervals](#direct-fixed-design-mean-intervals)).

[`examples/fixed_design`](../examples/fixed_design/README.md#prompt-variants-of-one-base-prompt)
has a complete template: `variant-design.csv`, `variant-plan.yaml` and the Inspect task
`variant_eval.py`.

## `cohorts`

Cohorts use exact evaluation-level field matches:

```yaml
cohorts:
  candidate:
    match:
      model: openai/example-candidate
    require_uniform:
      - task_version
      - model_generate_config
```

In schema 3.0 the same cohort reads:

```yaml
cohorts:
  candidate:
    match:
      model: openai/example-candidate
    require_uniform:
      - attributes.task_version
      - attributes.model_generate_config
```

Every selected evaluation must match at most one cohort.

## `groups`

Groups use exact manifest filters and one weight strategy. A filter value matches the
manifest's text as written, so quote one that YAML would read as a number or a boolean, as in
`filter: {tier: '01'}`: an unquoted `01`, `1`, `1.0` or `true` is an error, because YAML has
already lost how it was written. A filter on `mandatory` or `zero_tolerance`, which are read as
booleans, takes `true` or `false` unquoted.

The weight strategies are:

- `equal_prompt`;
- `nested_equal`;
- `explicit`, with a manifest `column`.

```yaml
groups:
  - id: cyber
    filter: {risk_domain: cyber}
    weights:
      strategy: nested_equal
      hierarchy: [task_id, question_id, prompt_id]
```

## `outputs`

`prompt_bounds` declares the simultaneous-bound family and alpha allocation. Prompt bounds are on by default and take their alpha from `family`, simultaneous or not, and `family` has no default: name a declared multiplicity family, or set `enabled: false`. `risk_measures` contains named measures. A plan without one reports no group risk, and the analysis warns (`PLAN_NO_RISK_MEASURES`).

`judge_claims: true` (experimental, off by default) also decides every claim on each declared
judge's verdicts, for comparing judges; it requires `inputs.judges`. See
[Claims by judge](#claims-by-judge-experimental).

```yaml
outputs:
  prompt_bounds:
    enabled: true
    simultaneous: true
    family: displayed
    coverage_mode: two_sided
    alpha_allocation: equal_bonferroni
  risk_measures:
    - {id: mean, type: weighted_mean}
    - {id: tail, type: cvar, beta: 0.90}
    - {id: entropic, type: entropic, lambda: 10}
    - {id: worst, type: maximum}
    - {id: worst_of_4, type: worst_of_k, k: 4}
```

`worst_of_k` reports the weighted probability that at least one of `k` independent attempts at
each fixed prompt fails, `sum_j w_j (1 - (1 - p_j)^k)`. It is the repeated-attempt analogue of
the weighted mean (`k: 1` is exactly the mean) and targets deployment or attack settings where an
item is retried `k` times. `k` is part of the estimand and must be chosen before results are
inspected. The per-item transform is coordinatewise monotone, so the measure inherits the
propagated simultaneous prompt envelope. Judge-misclassification correction does not apply to it.
The point estimate is the unbiased U-statistic `sum_j w_j (1 - C(n_j - f_j, k) / C(n_j, k))`,
from `f_j` failures in `n_j` classified rollouts. Plugging the failure rate into the transform
would understate the risk, because the transform is concave. With fewer than `k` classified
rollouts on some prompt, no estimate is reported, but the bounds still are.

`direct_interval: true` on a `worst_of_k` measure (experimental, off by default) adds a direct
interval to its group rows and makes it the row's primary interval. The propagated envelope stays
as a reference. Each prompt's unbiased estimate enters the KL/Chernoff bound as `floor(n / k)`
independent blocks, so the interval narrows with prompts and with rollouts per prompt. See
[Worst-of-k interval](statistical-methods.md#worst-of-k-interval-experimental).

```yaml
    - {id: worst_of_4, type: worst_of_k, k: 4, direct_interval: true}
```

The interval shares the alpha of `outputs.direct_mean_intervals`, so that section must be
enabled. Each opted-in measure is one more target in the family on every selected group. The
plan is refused (`PLAN_WORST_OF_K_INTERVAL_UNSUPPORTED`) when the design is sampled, when
`hierarchy_calibration` is `frontier_sidak`, or when the primary independence profile does not
block by rollout. `analyze` then gives the rows no direct interval: they keep the envelope as
primary and warn why. Other measure types reject the field. If a prompt has fewer than `k`
attempted rollouts, the row keeps the envelope and warns. Leaving the field out, or setting it
to `false`, keeps the plan hash unchanged.

`coverage_mode: two_sided` is the default. It interprets the multiplicity-family alpha as the
total error budget for the displayed lower-to-upper range and divides it equally between tails.
Thus family alpha `0.05` produces a simultaneous two-sided coverage floor of 95%, with 97.5%
coverage in each direction. `coverage_mode: per_direction` instead spends the full family alpha
on each direction; at alpha `0.05`, each direction has 95% coverage but the combined range has
only a 90% floor.

`hierarchy_region: linear_program` keeps the simultaneous floor but not the per-direction level.
A projected endpoint depends on the other direction's raw endpoints too, so projected intervals
report the simultaneous floor in each direction (see
[hierarchy-simultaneous-inference.md](hierarchy-simultaneous-inference.md)).

For `explicit` allocation, provide one prospective total alpha budget per manifest prompt:

```yaml
prompt_bounds:
  enabled: true
  simultaneous: true
  family: displayed
  coverage_mode: two_sided
  alpha_allocation: explicit
  explicit_alpha:
    prompt-001: 0.01
    prompt-002: 0.04
```

The entries must cover the manifest exactly and sum to no more than the family alpha. In
`two_sided` mode each entry is split equally between tails. When there are multiple cohorts, each
tail budget is also divided equally across its cohort-specific endpoints, so the declared family
budget still covers every reported prompt range.

### Direct fixed-design mean intervals

`direct_mean_intervals` adds direct concentration intervals for selected `weighted_mean` outputs
and, optionally, every non-prompt node in the declared hierarchy:

```yaml
outputs:
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    reference_methods: [weighted_empirical_bernstein, weighted_hoeffding]
    family: displayed_direct_means
    groups: [all, cyber]
    simultaneous: true
    coverage_mode: two_sided
    independence:
      primary: rollout
      sensitivities: [prompt]
    hierarchy_nodes: true
    hierarchy_calibration: frontier_sidak
    hierarchy_region: linear_program
```

If `groups` is omitted, the method applies to every declared group. With `simultaneous: true`,
the primary intervals are simultaneous across all selected fixed-design estimands. With
`simultaneous: false`, each interval is pointwise.

`hierarchy_calibration` controls how the directional family error budget is spent:

- `flat_bonferroni` (default) divides alpha over every group root, hierarchy node, and cohort;
- `frontier_sidak` divides alpha by Bonferroni over potentially dependent hierarchy frontiers,
  then uses exact Šidák product calibration among pairwise-disjoint nodes within each frontier.

`frontier_sidak` requires `hierarchy_nodes: true`. It assumes independence across blocks in the
selected named profile. It does not assume that failure probabilities are identical. The compiler
checks actual block overlap: siblings receive Šidák calibration only when their block supports are
disjoint. Ancestors, descendants, levels, cohorts, and overlapping groups are conservatively
treated as dependent.
Repeated identical targets—for example a root and a hierarchy level containing one node with the
same weights—are aliases and do not spend alpha twice.

`weighted_kl_chernoff` is a finite-sample Chernoff inversion optimized for sparse binary failures.
It uses the Bernoulli-KL exponent when atomic weights are equal and a rigorous weighted
Poisson-Chernoff fallback when they are not. `weighted_empirical_bernstein` is variance-adaptive;
`weighted_hoeffding` uses only boundedness. `weighted_betting` (experimental) inverts a fixed mixture of
common betting fractions; it is variance-adaptive without an estimated variance proxy, so with
equal or near-equal block weights it is usually the tightest choice when blocks are fractional
rates (a prompt-level or coarser independence profile) and is competitive with KL/Chernoff on
sparse binary rollouts. Unequal block weights enter as exponents `A_g / max A` and widen it much
as they widen the KL fallback, and more than they widen Hoeffding: with tasks of 2, 10 and 50
equally weighted prompts, every prompt observed at a failure rate of 0.05, its upper bound at
alpha 0.05 is 0.459 against Hoeffding's 0.371.
Comparisons do not accept it. See
[Betting interval](statistical-methods.md#betting-interval-from-a-fixed-mixture-of-common-bets-experimental).
`reference_methods` reports prespecified alternatives
without replacing the primary method. Each method's coverage guarantee is construction-specific;
the report is not permission to choose the shortest interval after seeing the results.

Each direct construction assumes independence across the blocks in its selected profile and
allows arbitrary dependence inside a block. Unclassified attempted outcomes are completed as safe
for the lower endpoint and as failures for the upper endpoint.

With `hierarchy_nodes: true`, node weights are inherited from the enclosing declared group and
renormalized within the node. `hierarchy_region: linear_program` (the default) treats prompt risks
as shared primitive parameters, places every raw direct group and node interval into one linear
confidence region, and reports the coordinate projection of that common region for each node.
Use `hierarchy_region: none` to retain the unprojected intervals. The `independence` selector names
one primary profile and optional prespecified sensitivity profiles. See
[Configuring hierarchy and independence](independence-profiles.md).

`single_valued_bound: true` (experimental, default `false`) adds a single number per direct
weighted-mean interval for system models such as fault trees: M, the smallest value of U_δ + δ
over levels δ, where U_δ is the interval's own one-sided upper bound at level δ for that node
alone. It needs no alpha and has P(M < R) ≤ R: its miss probability is the unknown risk R itself.
It is not simultaneous, does not enter claims, and is usually above a 95% bound. A direct
`worst_of_k` interval gets no such number. Each direct weighted-mean interval in `report.json`
then carries `single_valued_bound`, `single_valued_bound_delta` and
`single_valued_bound_guarantee`, and `report-full.md` lists them under "Single-valued bounds
(experimental)". The option is refused with `PLAN_SINGLE_VALUED_BOUND_UNSUPPORTED` under a
sampled design and with judge calibration. See
[Single-valued bound for system models](statistical-methods.md#single-valued-bound-for-system-models-experimental).

### Anytime-valid bounds for runs stopped early (experimental)

`outputs.anytime_bounds` bounds a run that a rule such as optstop's stopped once its outcomes
looked precise enough. The fixed-horizon bounds do not hold for such a run, because its
length depended on its outcomes. This bound holds at every point of the run's sample order at
once, so it holds wherever the run stopped, whatever rule chose the stop.

```yaml
outputs:
  anytime_bounds:
    independence_profile: prompt   # a profile of design.independence_profiles
    family: gate                   # the multiplicity family whose alpha the bounds share
    rollouts_per_item: 1           # optional, default 1: epochs per prompt in the stream
    groups: [all]                  # optional, default every group
    order: inspect_sample_shuffle  # optional, and the only value: the order Inspect recorded
```

What it needs from the run:

- **A random order fixed before the run, recorded in the log.** Run the evaluation with
  Inspect's `sample_shuffle` set to a nonzero integer seed. Inspect then records the seed and
  lists the sample ids in the shuffled order. Inspect does not shuffle by default, nor under
  the seed 0. A run without a shuffle, `sample_shuffle: 0` included, or shuffled without a
  recorded seed (`sample_shuffle: true`), is refused with an `ANYTIME_ORDER_UNRECORDED` error,
  and so is any source other than Inspect logs.
- **A stopping rule that only stops.** The rule may decide when to stop, but never which
  prompt runs next.
- **One run per cohort**, whose order lists every prompt of each group once. A `limit` that
  leaves prompts out of the order is refused.

`validate` checks the recorded orders as `analyze` does, so it refuses the same runs.

Each prompt's outcome is its failure share over its first `rollouts_per_item` epochs. Inspect
schedules every prompt's first epoch before any second epoch, so with several epochs a run
stopped early has few complete prompts; keep the default of 1 unless every prompt ran its
epochs. Prompts are read in the recorded order, never in the order they finished, up to the
last prompt with an observation. Before it, an epoch that is missing or unclassified counts as
a failure. This only raises the bound, so it stays valid.

`independence_profile` must block by `rollout`, which treats the epochs of one prompt as
independent, or by `prompt_id`, which treats only prompts as independent. A coarser profile is
refused: the bound needs independent prompts. The family's alpha is split equally over every
cohort and group bounded. The bound is the weighted mean failure probability of the group,
with the group's weights. Uneven weights widen it: the first step scales every outcome by the
largest weight, so with one prompt carrying much of the weight the bound stays at or near 1
for most of the run, wherever that prompt falls in the order
([Uneven weights](sequential-design.md#uneven-weights)).

`analyze` adds an `ANYTIME_UPPER_BOUND` diagnostic (`INFO`) for each cohort and group. Its
message starts with "experimental", and its `details` hold `experimental: true`, the run, its
seed, `population` (the group's prompts), `draws` (prompts read),
`complete_draws`, `rollouts_counted_as_failures`, the alpha, `estimate`, `upper`, and, for
each weighted-mean claim on that cohort and group, whether `upper` is at or below its
`acceptable_upper` (`certified`). As a claim can pass only in a confirmatory analysis whose
integrity is valid, `certified` is `null` under exploratory intent or invalid integrity, and
`certification_withheld` says which. A bound, once computed, never changes a claim's status or
the decision: claims keep the status the fixed-horizon analysis gives them, which for a run
stopped early is usually `INCONCLUSIVE`. A refusal does change them. Any other reason the
bound cannot be computed is an `ANYTIME_BOUND_INVALID` error, and a plan the bound cannot serve
(a sampled design, an unknown family, group or profile, a coarser profile, or a judge
calibration that covers a bounded group) is a `PLAN_ANYTIME_BOUNDS_INVALID` error. These
errors, like `ANYTIME_ORDER_UNRECORDED`, make the integrity status invalid, so no claim can
pass. A plan without the section keeps its hash and outputs. See
[Runs that optstop stopped](sequential-design.md#runs-that-optstop-stopped-126) for the
construction and what it costs.

The bound is on the judge-reported failure probability. A plan whose
`inputs.judge_calibration`, in any mode, covers a bounded group is refused with
`PLAN_ANYTIME_BOUNDS_INVALID`: a validated calibration decides that group's claims on the
corrected risk, and an assumed or transported one declares that the judge misclassifies, so a
bound at or below a claim's `acceptable_upper` would certify the judge's risk in place of the
model's. With a calibration scoped by `groups`, list in `outputs.anytime_bounds.groups` only
groups it leaves out.

## `comparison_sets`

Use a comparison set for two or more cohorts:

```yaml
comparison_sets:
  - id: models_vs_reference
    cohorts: [reference, candidate_a, candidate_b]
    contrasts:
      type: versus_reference
      reference: reference
    estimands: [risk_difference, risk_ratio]
    group: all
    measure: mean
    method: weighted_kl_chernoff
    reference_methods: [weighted_empirical_bernstein, weighted_hoeffding]
    family: model_comparisons
    coverage_mode: two_sided
    independence:
      primary: rollout
      sensitivities: [prompt]
    hierarchy_nodes: true
    hierarchy_calibration: frontier_sidak
    hierarchy_region: linear_program
```

At least two unique cohort identifiers are required. A two-arm set gives exactly the estimates and
bounds of the single `weighted_kl_chernoff` comparison with the set's settings, at the root and at
every node. `contrasts.type` is either
`versus_reference`, which requires a reference member, or `all_pairs`. Generated comparisons are
calibrated together and projected from common arm-risk regions. `all_pairs` plus
`risk_difference` also produces simultaneous rank sets. The remaining interval, hierarchy, and
independence fields have the same meanings as direct comparisons below.

## `comparisons`

```yaml
comparisons:
  - id: candidate_vs_baseline
    candidate: candidate
    baseline: baseline
    group: all
    measure: mean
    estimand: risk_difference
    method: weighted_kl_chernoff
    reference_methods: [weighted_empirical_bernstein, weighted_hoeffding]
    family: model_comparisons
    simultaneous: true
    coverage_mode: two_sided
    independence:
      primary: rollout
      sensitivities: [prompt]
    hierarchy_nodes: true
    hierarchy_calibration: frontier_sidak
    hierarchy_region: linear_program
    allocation_weight: 1
    pair_key: [prompt_id, repeat]
    diagnostic_methods: [exact_mcnemar, newcombe_paired, fixed_cluster_bootstrap]
    cluster_key: question_id
    bootstrap_replicates: 2000
    bootstrap_seed: 1234
```

`risk_difference` is candidate minus baseline, so negative values favor the candidate.
`risk_ratio` is candidate divided by baseline. `relative_risk_reduction` is one minus that ratio,
so positive values favor the candidate. Direct weighted KL/Chernoff, empirical-Bernstein, and
Hoeffding inference requires a `weighted_mean` measure. Every comparison names a multiplicity
family and an independence selection.

Each direct comparison names a simultaneous-interval family. With `flat_bonferroni`, error is
allocated over every requested overall and hierarchy-node result. With `frontier_sidak`, error is
Bonferroni-allocated over dependent hierarchy frontiers and exact Šidák product calibration is
used among disjoint sibling nodes. All comparison specifications sharing the family are calibrated
together. A `weighted_bonferroni` family uses each specification's positive `allocation_weight` as
its frontier allocation weight; ordinary `bonferroni` allocation is equal. Reference methods reuse
the same local levels and provide construction-by-construction coverage rather than joint coverage
across methods.

Compatible weighted-KL difference, ratio, and relative-reduction specifications over the same two
arms and hierarchy reuse one two-arm confidence region. Those raw regions are calibrated aliases,
so deterministic effect projections do not consume duplicate family alpha.

`hierarchy_nodes: true` repeats the same fixed-design comparison at every creator-declared node,
renormalizing the original group weights within that node. Named sensitivity profiles provide
alternative independent-block partitions. `comparison_coupling: joint` permits arbitrary
dependence between candidate and baseline observations sharing a block key.

`hierarchy_region: linear_program` places every raw node interval into one common region. Risk
differences use a prompt-contrast vector in `[-1, 1]`; ratios use a two-arm region and
linear-fractional projection. The displayed intervals therefore respect exact parent/child
aggregation and cannot be wider than their raw intervals. Set it to `none` to retain raw calibrated
intervals only. `frontier_sidak` requires `hierarchy_nodes: true`.

Pair and cluster keys for McNemar, Newcombe, and bootstrap diagnostics are supplied by the creator;
the package never infers them. These diagnostics remain separate from the primary direct interval.
A pair with an unscored arm is kept and bounded over every scoring of its unknown outcome; see
[Pairs with an unscored arm](model-comparisons.md#pairs-with-an-unscored-arm).

See [Fixed-design model and configuration comparisons](model-comparisons.md) for the complete
mathematical construction, assumptions, output fields, and visualization guide.

## `multiplicity`

Families specify `purpose`, `method`, and `alpha`. Supported method names are:

- `intersection_union`;
- `bonferroni`;
- `weighted_bonferroni`.

The simultaneous prompt-bound family must use a Bonferroni method in v1.

## `claims`

A group-risk claim names cohort, group, and measure. A comparison claim names only its comparison.

```yaml
claims:
  - id: tail_safe
    title: Upper-tail risk is acceptable
    cohort: candidate
    group: cyber
    measure: tail
    family: deployment_gate
    acceptable_upper: 0.01
    unacceptable_lower: 0.03
    mandatory: true
```

`acceptable_upper` must be lower than `unacceptable_lower`.

A [decision policy](decision-policy.md) (experimental) can set or revise the thresholds of
`corrected_interval` claims outside the plan, after the run if need be, without changing the
plan's hash. `hash-plan --inference` prints the hash a policy names, which leaves those
thresholds out. An overall `PASS` at thresholds a policy moves certifies only when one coverage
event covers every mandatory claim; see
[When the overall decision certifies](decision-policy.md#when-the-overall-decision-certifies).

## `adaptive_claims` (experimental)

An attacker that carries what it learned on one behaviour into the next makes its runs
dependent, so the bounds behind `claims` do not apply to it. An adaptive claim still gives a
guarantee, in one direction only: it bounds the attacker's running average of conditional
success probabilities from below, and is `FAIL` when that bound reaches `unacceptable_lower`.
It is never `PASS`.

```yaml
adaptive_claims:
  - id: adaptive_attacker_succeeds
    title: The adaptive attacker succeeds on at least 20% of behaviours
    cohort: candidate
    group: all
    order_field: attack_order
    family: attack_gate
    unacceptable_lower: 0.20
    mandatory: true
```

| field | meaning |
|---|---|
| `cohort`, `group` | the attacker's runs: the cohort's runs on the group's behaviours |
| `order_field` | the design-manifest column that gives each behaviour its place in the attack order, as an integer; each value once in the group |
| `family` | the multiplicity family whose `alpha` the bound takes, whole, as each claim under `claims` does; no two adaptive claims may share a family (`PLAN_INVALID_MULTIPLICITY`). The bound is a test of its own, not one of the family's simultaneous bounds: in a family whose `alpha` prompt bounds, direct-mean intervals or comparisons split, it spends that `alpha` again, so give it a family of its own |
| `unacceptable_lower` | the claim is `FAIL` when the lower bound is at or above it; in (0, 1] |
| `mandatory` | whether the claim enters the overall decision; default `true`. The claim is never `PASS`, and the decision is `PASS` only when every mandatory claim is, so a mandatory adaptive claim can make the decision `FAIL` and otherwise holds it at `INCONCLUSIVE`, whatever the other claims show. With `false` the decision can pass on `claims`, but the adaptive claim's `FAIL` no longer enters it; the report still shows the claim and its status |

What the claim assumes and reports:

- **Success** is a run whose score the plan's mapping classifies as a failure. With the
  [attack-confirmation example](../examples/attack_confirmation/README.md) that is the
  confirmation verdict, not the attacker's own. Judge error is not corrected. A failure that
  a plan policy fills in with no score to confirm it, under `score.errors: count_as_failure`,
  `score.missing: count_as_failure`, `score.token_limit: failure` or
  `score.content_filter_stops: failure`, counts as no success, and the claim's reasons say
  how many there were. Counting such a run as a failure is conservative for the upper bounds
  of `claims`, but for this lower bound it would be the opposite.
- **The order** of the behaviours is fixed in advance or randomised, and `order_field`
  records it. The attacker may adapt within and across runs. The stream runs in that order
  up to the last behaviour the cohort attacked, and a behaviour inside it with no run, or
  with an unclassified outcome, counts as no success. Each behaviour has at most one run.
- **The bound is anytime-valid.** With probability at least 1 − α it stays at or below the
  running average at every point of the stream at once, so the analysis may be repeated as
  runs arrive (statistical methods section 11.1).
- **The number describes this attacker at this budget**, on these behaviours in this order.
  An upper bound on one adaptive attacker is no evidence about other attackers or budgets,
  so a plan that gives an adaptive claim an `acceptable_upper` is refused.

The report states these assumptions beside each bound, and lists them among the assumptions
its results rest on as
[`adaptive_attack_stream`](diagnostics.md#assumption-adaptive_attack_stream).

An order that is missing, not an integer or repeated, or a behaviour attacked more than once,
leaves the claim `INCONCLUSIVE` and says why. The report lists adaptive claims in their own
section of `report-full.md` and `report.json` (`adaptive_claims`), and beside the cohort's claims
in `report.html`. When `report.include_decisions` is false, `report-full.md` and `report.html`
leave them out of view, as they leave out the threshold claims; `report.json` and the data
embedded in `report.html` still hold them.
Identifiers are unique across `claims` and `adaptive_claims`.
`report.claims_export` cannot be combined with adaptive claims yet: `claims.json` has no form
for a claim that can only fail. Plans without `adaptive_claims` keep their hash.

## `policy`

`zero_tolerance` selects a Boolean manifest field. Any mapped failure on a flagged prompt creates a policy failure event.

## `report`

| Field | Default | Meaning |
|---|---:|---|
| `include_decisions` | false | Show overall decision, threshold-claim cards, adaptive claims, and policy-event roll-up in human-facing reports. When false they are hidden, not withheld: `report.json` and the data embedded in `report.html` still hold the decision and every claim's status |
| `include_prompt_text` | false | Reserved; raw content remains omitted in v1 |
| `include_observations` | true | Write normalized observation CSV |
| `decimal_places` | 6 | Markdown display precision; JSON retains full precision |
| `judge_error_stress` | false | Experimental. Add to every claim the judge accuracy at which its verdict would change for true risk, and the verdict over the primary judge's declared accuracy box; diagnostic only. See [Stressing a threshold verdict](judge-misclassification.md#stressing-a-threshold-verdict-experimental) |
| `claims_export` | false | Experimental. Write `claims.json`, one record per claim; needs `include_decisions: true`. See [claims.json](#claimsjson-experimental) |
| `precision_target` | 0.05 | The widest a bound may lie from its estimate and still count as precise, in (0, 0.5). When tail or threshold measures are declared and, in a cohort and group, at the lower median of its prompts' rollouts, a measure's upper bound would lie further than this above its estimate even if no prompt failed, `TAIL_MEASURES_UNINFORMATIVE` warns for that cohort and group and names the rollouts per prompt each such measure would need. The summary in `report.md` marks every interval that lies further than this from its estimate and lists the rollouts that would narrow it. Left out of the resolved plan at its default; a value other than the default is recorded in `report.json` as `presentation.precision_target` |

### `claims.json` (experimental)

With `report.claims_export: true`, the report directory also holds `claims.json`, and
`checksums.json` covers it. It lists every claim of the report as one record, for tools that
build safety cases or risk registers from benchmark evidence. Nothing in it is computed anew:
each value is copied from `report.json`, and a saved `report.json` renders the same file.

The plan is refused when `report.include_decisions` is false, because each record carries the
claim's status.

The document holds, once for all claims, the report's schema and package versions, the
analysis id, intent, `plan_sha256` and `plan_frozen_at`, the hashed input logs and the design
manifest's hash, the integrity status and the overall decision. Each claim record holds:

| Part | Fields | Copied from |
|---|---|---|
| top level | `claim_id`, `title`, `status`, `descriptive_status`, `mandatory`, `reasons` | the claim in `report.json` |
| top level | `judge_corrected` | whether the `JUDGE_CORRECTED_CLAIMS` diagnostic names the claim |
| top level | `judge_correction_unavailable` | whether the `JUDGE_CORRECTION_UNAVAILABLE` diagnostic names the claim |
| `estimand` | `cohort`, `group`, `measure`, `measure_type`, `comparison` | the claim's record in `presentation.claim_evidence` |
| `estimand` | `acceptable_upper`, `unacceptable_lower` | the claim |
| `bound` | `estimate`, `lower`, `upper`, `alpha`, `method`, `multiplicity_family` | the claim: the bound it was decided on, or the bound it has when it was not decided on one (below) |
| `evidence` | `refs` | the claim's `evidence_refs` |
| `evidence` | `test` | the test the claim was decided by, or was to be decided by when the test cannot run, as a `test:` evidence ref names it; null for a claim decided on an interval |
| `evidence` | `eligibility` | the group risk or comparison the claim is about |
| `evidence` | `interval_id`, `interval_method`, `coverage_mode`, `coverage_scope`, `family_alpha`, `independence_profile`, `assumptions` | the interval the bound comes from: for a claim decided by a test, that result's primary interval, which the test was applied to; otherwise the one an `interval:` evidence ref names, else that result's primary interval |
| `sources` | `claim`, `estimand`, `evidence`, `evidence_interval` | JSON pointers (RFC 6901) into `report.json` to each of the above |

For a claim decided on a validated judge correction, `bound` holds the corrected bound,
`alpha` is the claim family's alpha plus the correction's `evidence.family_alpha`, and the
evidence interval is the correction's.

A `method: noisy_valid_test` claim in a declared family has `evidence.test` set to
`noisy_valid_certification`, and its evidence interval is the judge-reported primary interval
the test is applied to, even when the claim's refs also name the validated correction. When
the test runs, `bound` holds the test's set of true risks, and `alpha` is the claim family's
alpha plus the judge family's whole alpha. When the test cannot run, `bound.lower` and
`bound.upper` are null, `alpha` is the claim family's alone, and `evidence.test` still names
the test, as the one the claim was to be decided by. The record's `reasons` describe the test
and its calibration when it runs, and say why it did not when it cannot. Either way the
evidence `assumptions` are the interval's own, and `claims.json` does not list the test's
assumptions; see
[the noisy-but-valid test](judge-misclassification.md#certifying-a-threshold-directly-the-noisy-but-valid-test).

In an exploratory analysis, `status` stays `INCONCLUSIVE` and `descriptive_status` gives what a
confirmatory analysis would decide; it certifies nothing. Fields of `evidence` are null when
the report holds no such result or interval.

`evidence.family_alpha` is the interval's own: the budget of the multiplicity family it was
computed in, not the claim's error budget, which is `bound.alpha`. For a judge correction it is
the share of the judge family's alpha that the correction spends, plus a quarter of that share
for the bound on the row's
[unclassified share](judge-misclassification.md#unclassified-rollouts) when the correction
needs one, which is spent on top of the judge family's alpha. The calibration divides that alpha equally over its scopes, one
per declared cohort, or per cohort and group when it is scoped to groups, so with several
scopes a correction's `family_alpha` is a share, not the whole budget. A comparison's
correction spends the shares of the two scopes it spans, and a quarter of each for the share
bound of each arm that needs one. A judge-corrected claim's `bound.alpha` is the claim family's alpha plus
its `family_alpha`.

A claim can be `INCONCLUSIVE` without a decision on its bound, and its record still carries the
bound `report.json` gives it. Two such cases, among others:

- Its multiplicity family is undeclared, and `alpha` is null.
- The plan declares a validated judge calibration, but the claim's evidence has no certifying
  correction. Then `judge_correction_unavailable` is true and the bound is the uncorrected
  judge-reported one: it is not a bound on the true risk the claim is about, so do not read it
  as one.

A claim whose family is undeclared also carries the uncorrected judge-reported bound under a
validated calibration, but `judge_correction_unavailable` stays false; its null `alpha` shows
that the claim was not decided. `reasons` says why each claim has its status.

The file has its own `schema_version` (now `"1.0"`) and `"experimental": true`.
`inspect-safety-stats schema claims` prints its JSON schema.

## Generate JSON schemas

```bash
inspect-safety-stats schema plan > plan.schema.json
inspect-safety-stats schema report > report.schema.json
inspect-safety-stats schema claims > claims.schema.json
inspect-safety-stats schema decision-policy > decision-policy.schema.json
```
