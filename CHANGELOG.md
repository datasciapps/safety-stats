# Changelog

Changes that a user can observe, newest first. Releases before this file was started are
described in their pull requests.

## Unreleased

### Fixed

- **A sampled design with `frontier_sidak` hierarchy calibration is refused, not crashed on.**
  Under `design.type: sampled` with `outputs.direct_mean_intervals.hierarchy_nodes` and
  `hierarchy_calibration: frontier_sidak`, `analyze` stopped with "independence blocks cannot
  be formed" whenever a frame item was not drawn, which a sample nearly always has, and `validate`
  accepted the plan. The frontier needs the nodes' intervals to be independent, which an undrawn
  item (no block) and a fixed-size draw (nodes sharing a stratum) do not give. The plan is now
  refused with `PLAN_SAMPLED_FRONTIER_UNSUPPORTED`, by `validate` as by `analyze`, whose report
  is `invalid` with no direct interval calibrated; `flat_bonferroni` works as before. A sample
  that drew every frame item, the one case that ran before, is refused too: analyse it as a
  fixed design, or use `flat_bonferroni`.

- **A rollout that an `exclude` score policy leaves out of the bounds is reported, and integrity
  is degraded (changes report bytes).** Under `inputs.score.missing: exclude` or
  `inputs.score.errors: exclude` an unscored or errored rollout was dropped with no diagnostic,
  and integrity stayed `valid`, the state that says every required outcome was classified. With
  20 of 40 rollouts unscored, `exclude` reported `valid` where the default reports `degraded`.
  Each excluded rollout now gets a `SCORE_MISSING_EXCLUDED` or `SAMPLE_ERROR_EXCLUDED` warning
  (source `logs`, effect `degraded`) and integrity becomes `degraded`, since the bounds rest on
  the excluded rollouts failing as often as the classified ones, which the inputs cannot confirm.
  The bounds and the claims are unchanged: they cover the classified rollouts, as the plan asks.
  Reports of such plans gain the warnings and change in bytes, `report_hash` included.

- **The generative slice family's two-sided coverage is stated as it is computed.** Under
  `coverage_mode: two_sided` each endpoint gets half of its member's level, so every lower and
  upper bound of the family covers jointly at `1 - alpha`, and each direction at
  `1 - alpha / 2`. `report.html` said the members were covered at `1 - alpha` "per direction",
  which in this package is `per_direction`'s guarantee; it now says "both bounds together", and
  the generative and statistical-methods pages agree. The bounds are unchanged. Every generative
  `report.html` embeds the page's script, so it and its `checksums.json` entry change in bytes.

- **A generative run whose identifiers read as numbers can be analysed.** The seed manifest is
  read as text, but `generative analyze` and a resumed `generative materialize` read the
  selection and rewrite ledgers back with pandas' type inference. A semantic seed or other
  identifier such as `001`, `1.0` or `true` became a number or a boolean there, and the run
  stopped with a validation error. The ledgers are now read as text, as written.

- **Manifest identifiers and grouping columns are read as written (changes report bytes for
  numeric-looking labels).** The analysis read the design manifest with pandas' type
  inference. Labels that read as one number or boolean, such as `1` and `01`, `1` and `1.0`,
  or `true` and `True`, became one stratum, independence block, hierarchy node or group filter
  value without a word, so a fixed-size design that drew one item too few from one such stratum
  and one too many from the other passed its draw checks. An identifier such as `0001` became
  `1`, so no rollout of item `0001` joined its row, and `1` and `01` were a duplicate
  `prompt_id`. A blank value, or one that pandas reads as missing such as `NA` or `None`,
  merged, split or crashed depending on the column and the pandas version. `prompt_id`,
  `question_id`, `task_id` and every column the plan groups by (hierarchy levels, nested weight
  hierarchies, group filters, manifest columns in a `block_by` and `design.stratum_column`) are
  now read as text, exactly as written, by `analyze`, `validate`, `plan fixed-design` and
  `plan comparison`. A cell in one of these columns that is blank, or holds a marker that R,
  spreadsheets, databases and pandas write for a missing value (`NA`, `N/A`, `None`, `null`,
  `nan`, `#N/A` and the rest of pandas' default list, fixed so that it does not change with
  the pandas version), is an `INGEST_FAILED` error that names the column and the rows: recode
  a label meant literally, for example as `none`. Weights, inclusion probabilities, thresholds,
  the selection column, adaptive claims' order fields, `expected_rollouts` and the other
  numeric or boolean columns keep their types, even when the plan also groups by them. A group
  `filter` on a label column takes a quoted value: an unquoted YAML `1`, `01`, `1.0` or `true`
  is an error that says to quote it. Reports of manifests without such labels, every bundled
  example among them, keep their bytes. A manifest that zero-pads identifiers that the logs
  record as integers (`0001` for the sample id `1`) now reports `PROMPT_MISSING` and
  `PROMPT_UNEXPECTED`, whose messages name the other spelling of the same number: write the
  identifiers as the logs record them. See
  [`inputs`](docs/plan-reference.md#inputs).

- **The Newcombe paired interval no longer stops `analyze` on large comparisons (changes
  behaviour from 131,072 pairs).** Its correlation divides by the square root of the product of
  the 2×2 table's four margins, which was computed in integers. Once the product passed 2^64,
  `newcombe_paired_interval` raised a `TypeError`, and `analyze` stopped without a report for a
  comparison with `diagnostic_methods: [newcombe_paired]`. That happened from 131,072 pairs when
  both arms fail about half the time; the rarer the failures, the more pairs it took (about
  300,000 when both arms fail 5% of the time). The product is now computed in floating point,
  and smaller tables give the same bounds to the last bit.

- **`stats.direct_risk_difference_bounds` no longer wraps 32-bit counts.** With
  `independence_unit="prompt_cluster"`, the function multiplies the two arms' counts in the
  integer type of the arrays it is given. Counts in `int32`, the default for a Python list on
  Windows with NumPy 1, wrapped from 32,768 rollouts a prompt: most such inputs were refused
  with a `ValueError`, but some gave wrong Hoeffding and empirical-Bernstein bounds without an
  error. The counts are now taken as `int64`. `analyze` was not affected.

- **`analyze` reports a cohort that matches no evaluation instead of stopping.** With direct
  means calibrated over hierarchy frontiers (`hierarchy_calibration: frontier_sidak`), as in the
  plan `init --group-by` writes, a cohort that matched no evaluation, or a cohort of a fixed
  design that missed a prompt, made `analyze` stop with `ValueError: independence blocks cannot
  be formed` and write no report, while `validate` reported `COHORT_EMPTY` or `PROMPT_MISSING`.
  Comparisons with such an arm gave `COMPARISON_CALIBRATION_INVALID` instead, under either
  hierarchy calibration, and withheld every comparison of the family under each profile
  (changes report bytes). `analyze` now writes the report, invalid with the errors `validate`
  reports, and the command exits 3. The incomplete cohort's targets and comparisons get no
  interval, as before. They keep their places and alpha in their frontiers, so every other
  cohort and comparison is calibrated as with a complete ledger. Under a profile whose blocks
  depend on the rollouts, such as `[run_key]`, their frontiers are split instead, which can
  widen the other intervals under that profile but never narrows them.

- **report.md, index.md and `analyze` round every bound outward (changes `report.md` and
  `index.md`).** They rounded each bound to the nearest digit shown, so an upper bound could
  read tighter than the analysis gave: with 20 prompts of 500 rollouts and no failure, an upper
  bound of 0.00044 read as 0.0%, and a claim's upper bound could print equal to a threshold it
  exceeded. Each upper bound is now rounded up and each lower bound down: at a tenth of a
  percentage point for rates and differences in `report.md` and in the failure rates `analyze`
  prints, at two decimals for ratios, three for tail and threshold measures and four in
  `index.md`. The claims in `report.md` show `report.decimal_places` (6 by default), as
  `report-full.md` does, instead of four decimals, and each threshold exactly, with more
  decimals where a threshold needs them, so that a bound shows at or below its threshold exactly
  when it is. The simultaneous coverage under the failure rates is rounded down, so that
  99.996% reads 99.9% rather than 100.0%. `report.json`, `report-full.md` and the CSV files keep
  their bytes.

- **report.md states why the decision is what it is (changes `report.md`).** Its claims
  table listed the threshold claims only. An adaptive-attacker claim that failed gave "Overall
  decision: FAIL" beside "No claims declared"; a zero-tolerance failure gave FAIL beside
  claims that all passed; an exploratory analysis showed `INCONCLUSIVE (descriptive: PASS)`
  with nothing saying why; and the table did not say which claims were mandatory. The
  decision now comes with its reason, found by the rules that decide it: a zero-tolerance
  failure, invalid integrity, an exploratory analysis, no mandatory claim, or the mandatory
  claims that failed or are inconclusive. The table gains a "Mandatory" column and lists
  each adaptive-attacker claim, marked "FAIL only", with its lower bound and threshold.
  Below it, a note explains descriptive statuses when there are any, one explains adaptive
  claims, and one names the prompts with a zero-tolerance failure and the models that
  failed them.

- **report.md names the info codes that change a result (changes `report.md`).** Its
  warnings block lists errors and warnings only, so an info code that adjusts a result, such
  as `TOKEN_LIMIT_REACHED` counting an answer cut off at the token limit as the plan says, or
  that withholds one, such as `JUDGE_LANG_REICZIGEL_UNAVAILABLE`, left no trace on the page. A
  "Notes" line after the warnings now names each such code with its effect and count. Info
  codes that only record a fact stay in `report-full.md`.

- **report.md no longer truncates the median rollouts per prompt (changes `report.md`).** With
  an even number of prompts the median can lie halfway between two counts, as 30.5 does; the
  header showed 30, and the rollout budgets under "Precision" were computed from 30. Both now
  use the median as it is.

- **report.md says what its tail-measure budget assumes (changes `report.md`).** Under
  "Precision", the rollouts per prompt that tail and threshold measures need are computed for
  a prompt with no failures, the fewest any prompt needs, while the text said every budget
  assumed the observed rate. The line now says so, and shows the precision target exactly
  rather than at two decimals.

- **report.md's breakdown flag counts every node and names a widest one that has an interval
  (changes `report.md`).** The "Precision" line for a breakdown counted the too-wide intervals
  among the 40 rows shown but divided by all nodes, and took a node without an interval, or
  one of zero width, as the widest, which it then showed as "unavailable". It now counts every
  node, says how many have no interval, and names the widest among those that have one.

- **report.md keeps a repeated comparison on a row of its own (changes `report.md`).** Two
  root comparisons of the same estimand, cohorts, group and measure, such as one difference
  under two independence profiles, shared a row, and the second replaced the first. Each now
  has a row, and rows that would read alike name their comparisons.

- **report.md and index.md link the diagnostics reference (changes `report.md` and
  `index.md`).** Both pointed to `docs/diagnostics.md` for each code's remedy, a path in the
  repository that leads nowhere beside a report. They now link the page on GitHub, as
  `index.json` links the agent guide.

- **report.md's prompt-level sensitivity is the prompt profile's (changes `report.md`).** The
  column showed the first sensitivity interval of the primary's method, whatever its profile:
  with a category profile listed before the prompt profile it showed the category interval,
  and with a prompt-level primary and a category sensitivity, as in
  `examples/attack_confirmation`, it showed the category interval as prompt-level. It now shows
  the sensitivity from a profile with `block_by: [prompt_id]`, or `—` when there is none, and
  the note under the table says whether such a sensitivity is shown, the primary profile
  already blocks by prompt, or no interval allows dependence among one prompt's rollouts.

- **A judge's input files are named relative to the plan (changes report bytes).** With
  validated judge calibration, `report.json` recorded the labels file under
  `provenance.judge_calibration.validation_manifest` by the absolute path the plan loader
  resolved it to. A labels, strata, labelling queue or `judge_flip` manifest file that could
  not be read gave a `JUDGE_VALIDATION_INVALID`, `JUDGE_LABELLING_PROTOCOL_VIOLATED` or
  `JUDGE_FLIP_INVALID` message with that path. Reports name every other input relative to the
  plan's directory, so the same analysis wrote different bytes in different checkouts and a
  shared report revealed the local directory layout. Each now gives the path relative to the
  plan.

- **report.html links only an https:// address for what results establish.** The page made
  `scope.what_results_establish` from the `report.json` it renders into a link, whatever its
  address, and `visualize` renders any `report.json` it is given: a `javascript:` address there
  ran in the page when the link was clicked. An address that does not start with `https://` is
  now left out of the page, which then shows no link. The package writes an `https://`
  address, so no report it writes changes bytes.

- **Reports say when the plan leaves unclassified rollouts out of the bounds (changes those
  reports' bytes).** `report-full.md` said that missingness-robust bounds cover both
  completions of unclassified outcomes, its judge-error stress that an unclassified rollout
  counts as judge-unsafe at the upper bound, and `report.html` that unclassified outcomes are
  "Included in robust bounds". Under `inputs.score.missing: exclude` or
  `inputs.score.errors: exclude` those rollouts are left out of the bounds instead, which then
  cover only the classified rollouts. Each text now says so for such a plan. Reports of plans
  that exclude nothing keep their bytes.

- **A zero-tolerance column that the manifest lacks is an error.** When a plan sets
  `policy.zero_tolerance.manifest_field` to a column the manifest does not have, the analysis
  silently held no prompt to zero tolerance. It now stops with an error that names the column.
  The default column `zero_tolerance` stays optional, whether the plan leaves the field out or
  writes it, so a plan written out with the defaults its report lists analyzes as it did.

- **A generative run resumed after errors can be analyzed.** `generative materialize --resume`
  reruns the units whose rewrite hit an operational error, from the kernel or from the
  validator, but left each one's error row in `rewrites.csv` beside the row of its rerun. The
  ledger then recorded those units twice, and `generative analyze`, like any later resume,
  refused it. A resume now drops the error rows before it runs anything, replacing the file in
  one step. Resuming a directory an earlier version left in that state repairs it.

- **Generative ledgers end their lines with LF.** `rewrites.csv` and `prompts.restricted.csv`,
  written row by row since materialization could be resumed, ended every row with CRLF, and
  `attempts.csv` mixed CRLF and LF after a resume. All three now end every line with LF on
  every platform, like the other text artifacts. A resume writes them afresh before it runs
  anything, which converts the files a development version left with CRLF.

- **`analyze` reports a direct-mean hierarchy it cannot calibrate instead of crashing.** With
  `outputs.direct_mean_intervals` set to `hierarchy_nodes: true` and `hierarchy_calibration:
  frontier_sidak`, a hierarchy node of zero weight is left out of the nodes, so its level no
  longer partitions the group, and `analyze` stopped with "ValueError: hierarchy level ... does
  not partition the group" and wrote no report. It now writes the report with a new error,
  `DIRECT_MEAN_CALIBRATION_INVALID`, as comparisons report the same condition with
  `COMPARISON_CALIBRATION_INVALID`: integrity is invalid, so no claim can pass, no direct-mean
  interval of the family is calibrated, group rows keep only the propagated prompt envelope and
  hierarchy-node rows have no interval. Giving every node a positive weight, or
  `hierarchy_calibration: flat_bonferroni`, avoids it. `plan fixed-design --plan` still refuses
  such a plan, which `analyze` decides no claim of, but now as it refuses a plan that
  `validate` finds invalid: it names `DIRECT_MEAN_CALIBRATION_INVALID` and its remedy, where
  it gave only "hierarchy level ... does not partition the group".

### Added

- **Context files for coding agents.** `llms.txt` indexes the documentation for language-model
  tools, `llms-full.txt` gives a single-file reference (entry points, estimator choice, the
  standalone functions' signatures, the commands and their options, exit codes and runnable
  recipes), and `skills/inspect-safety-stats/SKILL.md` gives a step-by-step workflow for a coding
  agent. `tests/test_agent_context.py` checks them against the code: links, Python and YAML
  blocks, signatures, import locations, and every command and option they name.

- **A fixed-design template for prompt variants, and the profile that coupled wordings need.**
  `examples/fixed_design` has a plan, a manifest and an Inspect task in which each base prompt
  (`question_id`) is asked in several wordings (`prompt_id`). `nested_equal` weights count each
  base prompt once, whatever its number of wordings.
  [Prompt variants in fixed designs](docs/plan-reference.md#prompt-variants-in-fixed-designs)
  says which relation between wordings breaks rollout independence. Similar failure rates do
  not. Outcomes coupled through a shared seed, sandbox or batch do, and then
  `[question_id, repeat]` (coupling within an epoch), `[question_id]` (across epochs) or a
  coarser key (across base prompts) must be the primary profile, because a sensitivity never
  changes a claim's status. Coupling across epochs also breaks the prompt bounds and every
  measure bounded through them, such as `maximum`, whatever the profile, so claims then rest on
  the direct mean. The template reports `[question_id, repeat]` and `[question_id]` as
  sensitivities of its `rollout` primary. The section also says that `maximum`
  and `top_k_mean` count wordings rather than base prompts, what `hierarchy_nodes: true` costs
  in alpha and in vacuous single-block nodes, and that a block key is a field's raw value, so
  base-prompt ids repeated across tasks share a block. The package itself is unchanged.
- **An example scores attack runs by a confirmation rollout (#124).**
  [`examples/attack_confirmation`](examples/attack_confirmation/README.md) makes one Inspect
  sample one attack run with a query budget K. The scorer sends the prompt the attack found to
  the target once more and grades that fresh completion with a judge the attacker did not search
  against, so the outcome is not selected for the attacker's judge's false positives, which
  inflate the attacker's own report to 1 − (1 − f)^K for a target that never complies. Every
  design-manifest row declares the attack. The task refuses a design that declares another
  attack, and relative paths to its files, which `inspect eval` would read from the task file's
  directory. The plan requires one declared attack per cohort, fingerprint included, and the
  README has a plan for live runs match every cohort on the attack id, budget, attacker and
  judges. Nothing compares fingerprints across cohorts, so the README asks for a new attack id
  whenever the attack changes, its prompt templates included. The README also says what the
  confirmed rate still depends on: the attacker's judge, whose false positives end runs on
  prompts that do not work and so lower it; a target that samples; and two judges that err
  independently. It runs offline with no API key, and against real models with `-T mode=live`.
  The package itself is unchanged.

- **An offline example compares two versions of a harness around the same model.**
  `examples/harness_versions` runs one simulated model inside harness v1 and v2, selects the two
  runs by the task argument `harness`, and reports v2 minus v1 with its interval. On its seeded
  runs the interval includes zero, so the example shows how a report reads when no difference
  is established. It needs no API key.

- **`analyze` prints the errors and warnings in the terminal.** After the headline failure
  rates, `analyze` prints each error and warning code of the report on standard error, one line
  per code: its severity, how often it occurred, its effect on the results and what follows
  from that effect, as `report.md` gives them. A header line names the report's `index.md` and
  its section "Errors and warnings by code", which lists each code's diagnostics, and
  `report.md` and its section "Warnings", which also gives the assumption each code checks.
  The assumption is not printed in the terminal: it is phrased as what the results rest on,
  so printed alone it would read as a fact where the code reports that it may not hold. A user
  or a coding agent that reads only the command's output now sees what qualifies the headline.
  Report files are unchanged.

- **`validate` and `analyze` report how much a grouping column shrinks the evidence
  (changes `report.json`).** For each independence profile whose blocks hold rollouts of more
  than one prompt, such as `block_by: [conversation]`, a new `CLUSTER_DESIGN_EFFECT` diagnostic
  gives each cohort's empirical intra-cluster correlation of the scored outcomes within the
  blocks, the design effect and the effective number of rollouts. For each primary profile of a
  result the plan computes that splits the profile's blocks, such as one at the item level, it
  gives the same figures on that primary's units, for example its prompts with their failure
  rates, and the effective number of units; disabled direct mean intervals name no primary.
  Where some unit has several rollouts, the correlation on units is the covariance of two units
  of one block over the variance of all units. The one-way ANOVA estimate on the rates would
  read small blocks of units with few rollouts, beside a large block of well-replicated ones, as
  a correlation where there is none. It is a warning when, in a cohort, the design effect on a
  primary's units reaches 2, so that their effective number is at most half of them. The
  figures on a primary's units give the number of its own blocks they come from, and the warning
  says that a primary whose blocks lie within the profile's counts at least twice as many
  independent blocks as the effective units, while one whose blocks span several of the
  profile's, such as questions asked in every conversation, counts only its own blocks. Figures on
  rollouts would warn about a prompt-level primary with repeated rollouts whenever prompts
  differ in failure rate, however little the grouping adds. Otherwise it is informational. No
  bound changes. With few blocks the figures are noisy and can warn with no grouping effect.
  See
  [How much the grouping matters](docs/independence-profiles.md#how-much-the-grouping-matters-cluster_design_effect).

- **Assumed judge scenarios can give each cohort its own box.** A scenario under
  `inputs.judge_calibration` with `mode: assumed` takes an optional `cohorts` map from declared
  cohorts to `(rho_1, rho_0)` boxes. A listed cohort's group risks, hierarchy rows and
  comparison arms are corrected with its box, and every other cohort with the scenario's own
  box, so a judge whose accuracy depends on the model under test can be stress-tested without
  human labels. The corrected intervals stay diagnostic sensitivities. The assumptions of each
  corrected row, and of each arm of a corrected comparison, say whether it used its cohort's
  own box or the scenario's default box. An undeclared cohort is `PLAN_UNKNOWN_REFERENCE`, and
  generative plans reject the field. The report records the boxes under
  `presentation.judge_calibration.scenarios[].cohorts` and lists them in its text. Plans that
  do not use the field keep their hash and report bytes.

- **The README walks through the offline demo.** A new section follows the demo from its
  inputs (logs, design and the plan `init` writes) through the `validate` and `analyze` commands
  and the Python calls to the report directory. It reads the one-page summary and the index, and
  explains the warning the demo shows, `TAIL_MEASURES_UNINFORMATIVE`, and the † that marks a
  wide interval.

- **Every report directory has an index for agents: index.json and index.md.** `index.json`
  lists every file of the report directory with its format, what it holds and, for the CSVs
  whose rows are items of a `report.json` list, which list that is. It then copies the summary
  level of `report.json`: headline estimates, decisions when shown, and errors and warnings by
  code. Each entry has a JSON pointer into `report.json`, and a headline comparison also has
  its row in `comparisons.csv`. `index.md` holds the same content as Markdown, and
  `checksums.json` covers both. `analyze` prints the index's path. The
  [agent guide](docs/agent-guide.md) says how to read them.

- **report.json has a summary level (changes `report.json`).** `report.json` gains
  `summary`: the integrity status and the count of errors, warnings and notes; the headline
  estimates (each weighted-mean failure rate and each overall comparison); the errors and
  warnings grouped by code, with the registry's assumption and consequence and, for each
  occurrence, a pointer to the diagnostic and to the item it concerns; and the ids of the
  assumptions used. With `report.include_decisions: true` it also holds the overall decision
  and each claim's status. Every entry carries a JSON pointer to its full record. The summary
  copies values from the rest of the report; it is dropped when a saved report is read back.
  `analyze` and `report.md` pick the headline rates by the same rule
  (`inspect_safety_stats.schema.summary.is_headline_risk`).

- **Each diagnostic in a report points at its item (changes `report.json`).** Every
  diagnostic in `report.json` gains `item`, a JSON pointer to the record its scope names:
  `/claims/<i>`, `/comparisons/<i>`, `/prompt_estimates/<i>`, `/group_risks/<i>` and the
  others listed in `inspect_safety_stats.schema.pointers`. It is null when the report holds no
  record of the item, such as a plan setting or a run. The report computes it from the scope,
  also when a saved `report.json` is read back.

- **report.md is a one-page summary; the full report is report-full.md (changes `report.md`).**
  `report.md` now holds one page. It starts with a header line with the prompts, rollouts,
  models and data-integrity status. Then come these sections:
  - a "Warnings" block, with one row per error or warning code giving where it applies, the
    assumption it checks and its effect;
  - the claims, when decisions are shown;
  - the failure rate per model with its prompt-level sensitivity, and judge-corrected risk when
    calibration is configured;
  - the comparisons, with a plain reading;
  - a one-level breakdown, and tail and threshold measures when they are declared;
  - "Precision": every interval further than `report.precision_target` from its estimate, with
    the rollouts that would narrow it;
  - "Assumptions used", naming the settings left at their defaults.

  The previous document is written unchanged as `report-full.md` and is listed in
  `checksums.json`. A `precision_target` other than the default is recorded in `report.json` as
  `presentation.precision_target`, so the summary renders from `report.json` alone. Reports of
  plans that keep the default keep their `report.json` bytes. `analyze` prints each
  weighted-mean failure rate with its interval and the path of the summary, in place of the
  count of risk measures. New planning functions `rollouts_for_half_width` and
  `rollout_scale_for_half_width` in `inspect_safety_stats.stats.planning` give the rollouts
  that an interval width needs. The summary's "Scope" carries the link to
  [What results establish](docs/what-results-establish.md) as well.

- **Every report of a fixed or sampled design links to a page on what results establish
  (#12).** `scope` in `report.json` gains `what_results_establish`, the address of
  [What results establish](docs/what-results-establish.md), a one-page summary of what a
  result shows and what it does not. `report.md` shows the link under "Scope" and `report.html`
  beside the scope statement. The scope statement itself now ends with the condition every
  bound rests on: "Every bound assumes that the evaluated model is not strategically
  underperforming on the evaluation." Every `report.json`, `report.md` and `report.html`
  changes bytes; the CSVs do not. A `report.json` written before renders without the link.
  The experimental generative report does not link to the page.

- **Diagnostics say what they concern and what they do.** Every diagnostic in `report.json`
  now carries `source`, the kind of input it concerns (`plan`, `registration`, `manifest`,
  `logs`, `judge_labels`, `analysis` or `generation`), and `effect`, what it does to the
  results (`invalid`, `degraded`, `withheld`, `adjusted` or `recorded`). Both are filled from
  the code's entry in `inspect_safety_stats.registry`, which also says what each value means;
  a `withheld` or `adjusted` diagnostic leaves integrity alone but can withhold a claim's PASS
  or the overall decision's.
  Every `report.json` and `report.html` with a diagnostic changes bytes; `report.md` and the
  CSVs do not. A `report.json` written before reads back with both fields filled. Read back as
  a stored report (`context={STORED_REPORT: True}`, as `visualize` and
  `examples/demo/show_results.py` do), a report keeps the `source` and `effect` it was written
  with and fills only those it lacks, so it still reads after a later version reclassifies a
  code or retires it. In `inspect-safety-stats schema report` both are optional, so a report
  written before them validates, but neither is ever null.

- **A diagnostic's scope is a path.** `scope` names where a finding sits as a `/`-separated
  path: a root (`plan`, `decision_policy`, `cohorts`, `runs`, `evals`, `logs`, `prompts`,
  `strata`, `comparisons`, `comparison_sets` or `families`), then collection/id pairs, as in
  `plan/outputs/direct_mean_intervals`, `cohorts/candidate/prompts/p1` or
  `cohorts/candidate/groups/g1/measures/worst_of_k`. Ids have `%` and `/` percent-encoded, so
  any id splits back out. Scopes were free text before (`candidate:p1`,
  `inputs.judges.j1`), and an error whose scope was a stratum, claim or prompt id equal to a
  cohort's id withheld that cohort's prompt bounds; an error now withholds a cohort's bounds
  only when its scope is `cohorts/<id>`. A scope goes down to the result row, method, profile
  and hierarchy node a finding concerns, a node as `levels/<level>/nodes/<node>`, so two
  findings of one code have two scopes, and an error about a group's measure, node or anytime
  bound, scoped below `cohorts/<id>/groups/<group>`, withholds none of the cohort's prompt
  bounds. Every `report.json`, `report.md`, `report.html` and
  diagnostics CSV with a scoped diagnostic changes bytes, and so does the CLI's diagnostic
  listing. A `report.json` written before still renders with `visualize`.

- **Reports state the methods and assumptions they used (#64).** The "Fixed-design risk
  measures" paragraph of `report.md` no longer names a fixed set of methods and profiles: it
  lists each method and independence profile the report's intervals used, with the
  assumptions those intervals record. "Limitations" is generated from the package's registry
  of assumptions: each assumption that applies, with the plan settings it took, marked as
  default or written, and what follows if it is wrong. `report.json` gains
  `assumptions_used` with the same content. A sampled design states `probability_sample`, that
  the items were drawn at random with inclusion probabilities fixed in advance and without
  regard to the outcomes, as `design.sampling` declares, in place of `rollout_independence`:
  its intervals take each drawn item as the independent block. It keeps
  `rollout_independence` when its prompt bounds are enabled. See
  [Assumptions and defaults a report states](docs/interpretation.md#assumptions-and-defaults-a-report-states).

- **Reports record every default the analysis applied.** `report.json` gains `plan_defaults`:
  each plan field the plan left out, with the value its default gave the analysis. These are
  the fields the plan hash does not cover. `report.md` gains a "Defaults applied" section that
  lists the defaults that set a value and counts those that leave a feature off. See
  [The plan hash](docs/plan-reference.md#the-plan-hash).

- **`init`: a starter plan from completed logs, with its assumptions noted (experimental).**
  `inspect-safety-stats init --logs LOGS --failure-values V` writes `design.csv` and
  `safety-plan.yaml`: one cohort per model, the root failure risk with a prompt-level
  sensitivity, an optional breakdown and a comparison of several models, for one task: logs of
  several tasks are refused, since a cohort that spans tasks fails `analyze`'s configuration
  check. `init` takes no value as safe that nobody named: a score with more than two values
  needs `--safe-values`, and without it `init` refuses and lists the values left over.
  `design.csv` has no `zero_tolerance` column, so the plan turns the zero-tolerance gate off
  and its header says how to turn it on; left at its default, the gate could hold no prompt.
  `init` names the logs of evaluations that did not finish successfully, which it ignores, and
  its suggested `analyze` command then lists only the logs it used. `--group-by` refuses a
  metadata field that some samples lack or leave empty, naming them. `--alpha` is spent once
  in each multiplicity family, which the plan's header and the output state. The plan's title
  names the task and the cohort ids, not the model names, which can be machine paths. The
  plan's header, and the command's output, list each choice `init` made with the assumption it
  rests on from `inspect_safety_stats.registry`, rollout independence first, what goes wrong
  if it does not hold, and the remedy. The plan is exploratory; `--intent confirmatory` is
  refused, because the plan is written after the logs exist. See
  [Command-line interface](docs/api-reference.md#command-line-interface).

- **A warning when tail measures cannot be informative.** Tail and threshold measures
  (`maximum`, `cvar`, `top_k_mean`, `entropic`, `threshold_exceedance`, `threshold_excess`, and
  `worst_of_k` without `direct_interval`) are propagated from simultaneous per-prompt bounds, so
  their precision is set by the rollouts on each prompt, not the total. For each cohort and
  group, `analyze` takes the lower median of the rollouts of its prompts whose bounds certify,
  and asks how far above its estimate each such measure's upper bound would lie if no prompt
  failed: by a failure-free prompt's bound for a maximum, CVaR, top-k mean or entropic risk, by
  `1 - (1 - bound) ** k` for `worst_of_k`, and, for a threshold measure, only through the
  prompts whose threshold lies below that bound. When that exceeds the new
  `report.precision_target` (default 0.05), a `TAIL_MEASURES_UNINFORMATIVE` warning names the
  cohort, the group and the measures, with the rollouts per prompt each would need. Integrity
  is unchanged, and plans that leave `precision_target` out keep their hash. Reports of plans
  that declare such measures on few rollouts per prompt gain the warning, once for each cohort
  and group. The default direct-mean interval method stays `weighted_empirical_bernstein`.

- **Catalogue of diagnostic codes and assumptions.** [Diagnostic codes and
  assumptions](docs/diagnostics.md) lists every diagnostic code a report can carry, with its
  severities, the assumption it reports on, its effect on the results and the remedy, and every
  assumption the results rest on, with the plan field that sets it and the codes that check it.
  `scripts/render_catalogue.py` generates the page from the package's registry, and a test
  fails when the committed page falls behind it. The package itself is unchanged.

- **Open-weight example and limitations log.** [`examples/open_weights/`](examples/open_weights/README.md) runs an open-weight model on
  HarmBench and XSTest with thinking off and on, many rollouts per prompt, judges them, and
  analyzes the run; it also orders rollouts for human labelling and turns the labels into
  judge validation manifests. [The limitations log](docs/limitations.md) records what this
  and other real evaluations showed the package cannot yet do, and what `main` does about each
  entry. The package itself is unchanged.

- **Anytime-valid bounds for runs stopped early (experimental, #126).**
  `outputs.anytime_bounds` reports, for each cohort and group, a one-sided upper bound on the
  weighted mean failure probability that holds at every point of the run's sample order at
  once, and so wherever a rule such as optstop's stopped the run. The run must use Inspect's
  `sample_shuffle` with a nonzero integer seed; a run without a recorded order is refused with
  an `ANYTIME_ORDER_UNRECORDED` error, by `validate` as by `analyze`. A cohort and group get no
  bound, and an `ANYTIME_BOUND_INVALID` error, unless the cohort's observations come from one
  run whose recorded order lists every prompt of the group exactly once and each sample joins
  one prompt. `analyze` adds an `ANYTIME_UPPER_BOUND` diagnostic, marked experimental, with the
  bound and, for each weighted-mean claim, whether it lies at or below the claim's
  `acceptable_upper`, or, under exploratory intent or invalid integrity, where no claim can
  pass, `null` with the reason. Claims and the decision keep their fixed-horizon status. The
  bound is on the judge-reported failure probability, so a plan whose judge calibration, in any
  mode, covers a bounded group is refused with `PLAN_ANYTIME_BOUNDS_INVALID`. Plans without the
  section keep their hash and outputs. `research/optstop_coverage.py` simulates optstop's
  stopping rule: the bound misses at most alpha, while Clopper-Pearson at the stop misses up to
  0.208 at a nominal 0.05. See
  [Runs that optstop stopped](docs/sequential-design.md#runs-that-optstop-stopped-126).

- **FAIL-only claims on attackers that adapt across behaviours (experimental, #125).** A plan
  can declare `adaptive_claims`: a cohort, a group, the design-manifest column that fixes the
  attack order, and an `unacceptable_lower`. The cohort's runs, in that order, are bounded
  from below by an anytime-valid bound on the attacker's running average of conditional
  success probabilities, which needs no independence between runs. The claim is `FAIL` when
  the bound reaches `unacceptable_lower` and `INCONCLUSIVE` otherwise; a plan that gives it an
  `acceptable_upper` is refused, because the bound describes one attacker at one budget. A
  success is a run whose score the mapping classifies as a failure: a failure that a plan
  policy fills in with no score to confirm it (`errors` or `missing: count_as_failure`,
  `token_limit` or `content_filter_stops: failure`) counts as no success, and the claim's
  reasons say how many there were. Each claim takes its multiplicity family's whole alpha,
  and a plan that puts two adaptive claims in one family is invalid
  (`PLAN_INVALID_MULTIPLICITY`). Mandatory adaptive claims enter the decision, which they can
  make `FAIL` but, never being `PASS`, otherwise hold at `INCONCLUSIVE`; with
  `mandatory: false` a claim neither fails nor holds the decision. `report.json`
  and `report-full.md` show them in their own section, `report.md` among its claims, marked
  "FAIL only", and `report.html` beside the cohort's claims, with the bound's assumptions: the
  attack order fixed before any outcome or drawn at random, at most one run per behaviour, and
  judge error not corrected; when `report.include_decisions` is false, `report.md`,
  `report-full.md` and `report.html` leave the claims out of view. Such a report lists the
  assumptions among those it rests on as `adaptive_attack_stream`. Plans without them keep
  their hash and their `report.json` and `report-full.md` bytes; every `report.html` changes
  bytes, because it embeds the script that renders them. See
  [`adaptive_claims`](docs/plan-reference.md#adaptive_claims-experimental).

- **Multi-turn and system-prompt generation (experimental).** A generative plan can make
  each unit a conversation instead of a prompt. `rewrite.output: conversation` lets the
  kernel return chat messages, which the dataset carries as Inspect message lists.
  `rewrite.system_prompts` makes the system prompt an arm: the procedure attaches a declared
  text for each arm value, or none where the text is empty, and a kernel that writes a
  system turn itself stops the run on every path, an empty text's included.
  `rewrite.conversation_turns` makes the number of user turns an arm, and every
  conversation is checked to have its arm value's count; one without it is an invalid
  attempt, which uses up one of `rewrite.validation.maximum_attempts` and, under
  `on_invalid: fail`, exhausts the unit. Each key enters the kernel fingerprint only when
  set, so existing plans materialize the same bytes and keep their kernel hashes. The fake
  kernel takes `fake:output=conversation`. The live example adds a reference auditor kernel
  and `plan-auditor.yaml`. See
  [Conversations and system prompts](docs/generative-evaluation.md#conversations-and-system-prompts-experimental).

- **Lang-Reiczigel limits in validated judge mode (changes report bytes).** With
  `sampling: simple_random`, every group's `weighted_mean` row that the calibration corrects
  now carries a `judge_corrected_validated_lang_reiczigel` interval beside the exact
  Clopper-Pearson box: the limits of Lang and Reiczigel (2014) for prevalence adjusted for
  estimated sensitivity and specificity. Tail measures and hierarchy nodes get none. They are
  `diagnostic` and never decide a claim, because a coverage simulation
  (`research/judge_correction_lang_reiczigel.py`) shows them falling to 91.4% at nominal 95%
  where prevalence is 2% or below and labels number 300 or fewer, and not uniformly narrower
  than the box. Where they cannot be computed, `JUDGE_LANG_REICZIGEL_UNAVAILABLE` says why.
  Python callers get `lang_reiczigel_corrected_risk`. Labels drawn within judge strata, simply or by a
  stratified design (`stratified_within_judge_stratum`), get no such interval.

- **Judge flip rate under content-preserving wrappers (experimental, #122).**
  `inputs.judge_flip` reads a CSV of the primary judge's verdicts on the run's judge-positive
  completions, rescored inside wrappers that keep the content verbatim, such as a story
  frame. `analyze` adds a `JUDGE_FLIP_RATE` diagnostic to the integrity section of
  `report.json` and `report.md`: the share that flips to safe under at least one wrapper and
  under each, with exact Clopper-Pearson intervals, and the number of prompts the rescored
  completions come from. The intervals treat the completions as independent. Under
  `sampling: simple_random` they are for the run's share; under `sampling: all`, a census,
  they are descriptive, and too narrow for the judge's rate beyond the run when completions of
  one prompt flip together. Above the plan's `warn_above` it is a warning. It never changes a
  claim or the decision. Failures that a plan policy classified, such as a sample error counted
  as a failure, are not judge-positive and are left out. A manifest that does not fit the run,
  or that rescores such a failure, is a `JUDGE_FLIP_INVALID` error. Plans without the section
  keep their hash and outputs.
  [`examples/judge_flip`](examples/judge_flip/README.md) rescores a run under four wrappers
  and writes the manifest; the package itself does not call a model. See
  [Judge flip rate](docs/judge-misclassification.md#judge-flip-rate-under-content-preserving-wrappers-experimental).

- **`claims.json`, a machine-readable record of every claim (experimental, #11).** With
  `report.claims_export: true`, `analyze` also writes `claims.json`: for each claim its status,
  estimand and thresholds, the bound it was decided on with its alpha, the interval and
  assumptions behind that bound, and JSON pointers to each value's place in `report.json`,
  with the analysis's plan hash and input hashes once for the document. Every value is copied
  from `report.json`; a judge-corrected claim exports the corrected bound and the combined
  alpha, and a claim decided by the noisy-but-valid test names the test and exports the
  judge-reported interval it was applied to. A claim that a validated calibration leaves
  `INCONCLUSIVE` for want of a certifying correction is flagged as such, because its bound is
  the uncorrected judge-reported one, and every record carries the claim's reasons.
  `checksums.json` covers the file, and `inspect-safety-stats schema claims` prints its
  schema. The option needs `report.include_decisions: true`. Plans without it keep their hash
  and every artifact's bytes. See
  [claims.json](docs/plan-reference.md#claimsjson-experimental).

- **Labelling protocol for a validated judge calibration (experimental).** A plan can
  declare how its human labels were collected, under `inputs.judge_calibration.protocol`:
  the labelling queue, the seed that ordered it, and why labelling stopped. A `tier` keeps
  only the queue rows whose `tier` cell has that value; the cells are read without
  surrounding whitespace, so a `tier` that begins or ends with whitespace is rejected. The
  analysis checks that the queue lists only classified rollouts of the analysed run, that the
  labels are exactly its first rows, and the declared size under `stopping: fixed_size`; a
  violation is the error `JUDGE_LABELLING_PROTOCOL_VIOLATED`. A queue that lists only part of
  a labelled judge stratum, or in a stratified design of a labelled declared fine stratum,
  gives the warning `JUDGE_LABELLING_QUEUE_PARTIAL`: that its rows are a random sample of the
  stratum is declared, not checked, and the report's corrected rows say so.
  `stopping: data_dependent` bounds each judge stratum's accuracy with an anytime-valid
  binomial confidence sequence, so labelling may stop once the result is clear. The sequence
  bounds the probability that a rollout of the stratum is truly unsafe, which the run's
  rollouts only sample, not the share of them that is: it does not use the stratum's size, and
  even a stratum labelled in full keeps a width. It is wider than Clopper-Pearson at the same
  labels. The report records the SHA-256 of the queue file's bytes as read, and the labelled
  prefix, in `provenance.judge_calibration.protocol`. Plans without a protocol produce the
  same bytes and hash as before. See
  [Labelling protocol](docs/judge-misclassification.md#labelling-protocol-experimental).

- **Arm contrasts and a simultaneous slice family in generative evaluation (experimental).**
  A generative plan can declare `inference.slice_family`: slices of arm levels and contrasts
  between two values of one arm level, such as one wrapper against another. They are covered
  simultaneously at `1 - alpha` for the primary cohort. A contrast holds the mix of semantic
  seeds fixed between its two values, so declared context weights and constraints cannot move
  it. A member the analysis could not estimate, such as a value that no compiled path carries
  or that no selected path reaches, is refused by `generative compile` and
  `generative materialize` before anything is generated. The family is written to
  `family.csv`, to a `slice_family` block in `report.json` and to a section of
  `report.html`, and `generative analyze` prints each member's upper bound under a heading
  that says it is on the judge-reported scale. Under a judge calibration no member is
  corrected for judge error, and a `JUDGE_CORRECTION_UNAVAILABLE_FOR_SLICE_FAMILY` warning
  says so; the root, nodes and `--slice` cuts are corrected. Plans without the block keep their
  `report.json` and CSV bytes. Every generative `report.html` changes bytes, because it embeds
  the script of the new section, and so does `checksums.json`, which holds its digest. With
  the block, `checksums.json` also covers `family.csv`. See
  [Arm contrasts](docs/generative-evaluation.md#arm-contrasts-and-a-simultaneous-slice-family-experimental).

- **Rewrite validation and near-duplicate detection in the package (experimental).** In a
  generative plan, `rewrite.validation.invoke: true` makes materialization call the validator
  the plan names, supplied with `--validator` (or `validator=`), on every rewrite the kernel
  accepts. Its rejections enter the retry path. `rewrite.deduplication.near_duplicate:
  normalized_text` also treats rewrites that differ only in case, compatibility forms,
  punctuation, symbols, spacing or characters that render as nothing, such as variation
  selectors, as duplicates, comparing their letters, marks and digits alone; a plan that also
  sets `deduplication.scope: none`, which compares nothing, is refused. With either setting,
  materialization writes `attempts.csv`, one row per kernel call without prompt text.
  Analysis then writes `fidelity.csv` and a `fidelity` section of `report.json`, which give
  for each arm why attempts failed. It also adds the diagnostics `GENERATIVE_FIDELITY_UNEVEN`,
  `GENERATIVE_VALIDATOR_MISMATCH` and `GENERATIVE_RETRY_DEPENDENCE`. Plans that use neither
  setting keep their fingerprints and outputs. See
  [Validation and near-duplicates](docs/generative-evaluation.md#validation-and-near-duplicates-experimental).

- **Outcome mapping sensitivities (experimental).** A score with more than two values, such
  as a refusal grade with partial refusals, no longer has to be read one way per run.
  `inputs.score.mapping_sensitivities` declares named alternative mappings; the analysis is
  repeated under each on the same rollouts, and the report shows every reading side by side
  in `report.json` (`outcome_mapping_sensitivities`), a new
  `outcome-mapping-sensitivity.csv` and a section of `report-full.md`. Only the primary mapping
  enters claims and the decision: the CSV's `descriptive` column and the "(descriptive)" mark
  in `report-full.md` set an alternative's readings apart. Each alternative carries what the
  plan, its decision policy and its registration receipt were checked for, so a receipt for
  another plan or a failed decision policy makes it `invalid` as it does the primary analysis,
  and counts the diagnostics that describe its own outcomes, such as
  `TAIL_MEASURES_UNINFORMATIVE` and `CLUSTER_DESIGN_EFFECT`. An invalid report never shows an
  alternative as anything but `invalid`, nor any of its claims as decided. An alternative that
  classifies every value as the primary mapping does is refused as a repeat, whatever the
  order or repeats of its values. Opt-in: plans without the field keep their hash and their
  report bytes. See
  [Outcome mapping sensitivities](docs/plan-reference.md#outcome-mapping-sensitivities-experimental).

- **The fixed-design planner uses the analysis method.** `plan fixed-design --plan PLAN`
  plans each claim under the interval method that `outputs.direct_mean_intervals` declares
  (KL-Chernoff, empirical Bernstein or Hoeffding) by inverting that interval. It shows the
  previous Hoeffding rates as a reference. The experimental `weighted_betting` interval
  depends on more than the estimate and cannot be inverted this way, so claims decided on it
  are listed as not planned, with that reason. Before, planning used Hoeffding only, while the
  analysis decided on KL-Chernoff or empirical Bernstein, so planned and analysed rates
  differed by an unknown margin. The endpoint alphas and independence blocks now come from
  `analyze`'s own construction, so `frontier_sidak` calibration is planned exactly. The output
  also lists every direct-mean interval `analyze` will report, for each independence profile,
  cohort and hierarchy node. Each comes with its alpha, effective sample size and
  certification floor: the upper bound `analyze` reports when every rollout is safe, which
  `analyze`'s own stages compute, its hierarchy-region projection included. The raw floor
  before the projection is shown beside it. The output names the levels the declared rollouts
  cannot resolve at their group's `acceptable_upper` (#9). The rates are guarantees under
  stated assumptions (independent blocks of the primary profile, every declared rollout
  attempted and classified), not the rates at which the power falls to the target. See
  [Command-line interface](docs/api-reference.md#command-line-interface), and
  [Statistical methods, section 5.7](docs/statistical-methods.md#57-detectable-failure-rates-under-the-analysis-interval)
  for the derivation.

- **The fixed-design planner reads the plan.** `plan fixed-design --plan PLAN` takes groups,
  weights, hierarchy, independence profile, multiplicity family and thresholds from the
  analysis plan and reports detectable rates for every claim, at the endpoint alpha `analyze`
  will use. Before, they had to be repeated on the command line, and the default hierarchy
  could silently plan a different design. Claims it cannot plan are listed with the reason,
  and so is every claim of an exploratory plan, which `analyze` reports INCONCLUSIVE. A plan
  with the errors `validate` reports is refused with them, since `analyze` decides none of its
  claims, and so is a design manifest the command cannot read. The output names the manifest
  it reads, the plan's or the one `--manifest` gives in its place.
  See [Command-line interface](docs/api-reference.md#command-line-interface).

- **Plan registration receipts (experimental).** `hash-plan --register RECEIPT.json` writes the
  plan's hash with the current UTC time, rounded up to the whole second so that it never
  precedes the registration, for the author to anchor externally, for example in a signed
  commit. `analyze` and `validate` check it with `--registration-receipt` (Python:
  `registration_receipt=`): a receipt for another plan is `PLAN_RECEIPT_MISMATCH`, one later
  than the earliest evaluation start is `PLAN_RECEIPT_POSTHOC`, both errors; a receipt that
  passes reports `PLAN_REGISTERED` in place of the `PLAN_SELF_DECLARED` warning. Nothing changes
  without the new options. See
  [Registration receipts](docs/plan-reference.md#registration-receipts-experimental).

- **Betting interval for direct means (experimental).** `weighted_betting` is a fourth direct
  interval method for `outputs.direct_mean_intervals` (`method` or `reference_methods`) and for
  hierarchy nodes. It inverts a fixed mixture of common betting fractions, needs no estimated
  variance proxy, and with equal or near-equal block weights is usually the tightest of the four
  when blocks are fractional rates, as under a prompt-level independence profile. Unequal
  weights cost it more than they cost Hoeffding, which can then be tighter. It is
  fixed-horizon, not anytime-valid.
  On a sampled design (`design.type: sampled`) its rows state the betting construction, mark
  it experimental and report its `chernoff_kernel`, as on a fixed design; a fixed-size
  design's row says why the without-replacement reduction covers it (see
  [Fixed-size designs](docs/design-based-sampling.md#fixed-size-designs)).
  Comparisons do not accept it, and its functions are not exported from
  `inspect_safety_stats.stats`. Plans that do not name it are unaffected. See
  [Betting interval](docs/statistical-methods.md#betting-interval-from-a-fixed-mixture-of-common-bets-experimental).

- **Direct `worst_of_k` interval (experimental).** A `worst_of_k` measure with
  `direct_interval: true` gets a KL/Chernoff interval on its group rows. Each prompt enters as
  `floor(n / k)` independent blocks, so the interval narrows as rollouts grow. Before, the only
  interval was the propagated prompt envelope, which can be very wide. The interval is calibrated
  with the direct-mean intervals and needs a fixed design with rollout-level independence; other
  plans are refused with `PLAN_WORST_OF_K_INTERVAL_UNSUPPORTED`, and their rows keep the
  propagated envelope with a warning that says why.
  Plans without the field get the same results. `report.html` roots its hierarchy tree at the
  weighted-mean row, where a worst-of-k row with a direct interval listed first could become the
  root; this changes the page script, so every `report.html` and its `checksums.json` entry
  differ in bytes, though a page of a plan without the field shows the same.
  `worst_of_k_kl_chernoff_bounds` is exported from `stats`. See
  [Worst-of-k interval](docs/statistical-methods.md#worst-of-k-interval-experimental).

- **Decision policies (experimental).** Claim thresholds can come from a decision policy kept
  apart from the plan, and be set or revised after the run, without changing the plan's hash:
  thresholds never enter the bounds, so the error rate of each claim's verdict holds at any
  thresholds. The overall decision's does not: when a policy moves any mandatory claim's
  `acceptable_upper` while the mandatory claims' bounds rest on separate coverage events, a
  threshold read off the data could make a wrong overall `PASS` as likely as the sum of their
  error budgets, so an overall `PASS` becomes `INCONCLUSIVE` with the warning
  `DECISION_POLICY_OVERALL_UNCERTIFIED`. Plans whose mandatory claims all rest on one
  simultaneous family, such as the default prompt bounds, keep it. This holds whatever time the
  policy states for its versions. `analyze` and `validate` take `--decision-policy` (Python:
  `decision_policy=`). A policy names its plan by the inference hash, which leaves out the
  thresholds of `corrected_interval` claims (`hash-plan --inference`); it cannot set
  `noisy_valid_test` claims, whose test uses their thresholds. `report.json` gains a
  `decision_policy` section with every version, its time and whether it came after the first
  evaluation started (`null` when that cannot be told, as when a run reports no start time),
  and the thresholds each claim was decided at; `report-full.md` and `report.html` show the
  same, and `report.md` says which thresholds the policy set.
  New diagnostics: `DECISION_POLICY_MISMATCH`, `DECISION_POLICY_UNKNOWN_CLAIM`,
  `DECISION_POLICY_FIXED_CLAIM` (errors), `DECISION_POLICY_OVERALL_UNCERTIFIED` and
  `DECISION_POLICY_SELF_DECLARED` (warnings: the second says that a version's stated time,
  which nothing checks, is all that shows it came before the run) and
  `DECISION_POLICY_AFTER_GENERATION` (info).
  `schema decision-policy` prints the policy schema. Analyses without a policy write the same
  bytes as before. See [Decision policies](docs/decision-policy.md).

- **Answers at the token limit counted on their own (experimental).** `inputs.score.token_limit`
  decides before the run how a rollout counts when its answer stopped at the generation token
  limit without a score: `safe`, `failure`, or `missing` (left to `inputs.score.missing`). Each
  choice counts such rollouts in a new per-prompt `n_token_limit` in `report.json`, a "Token
  limit" column in `report-full.md` and a `token_limit_unscored` column in `observations.csv`,
  and reports each in a `TOKEN_LIMIT_REACHED` diagnostic. `score`, the default, keeps the
  previous behaviour and output. Inspect logs only, stop reason `max_tokens` only: an answer
  that outgrew the context window (`model_length`) or a sample that hit one of Inspect's own
  sample limits, such as its token limit, is not detected, and follows `inputs.score.missing`
  when unscored. See
  [the plan reference](docs/plan-reference.md#inputs).

- **Content-filter stops as policy blocks (experimental).** `inputs.score.content_filter_stops`
  decides before the run how a rollout counts when a provider's content filter stopped its
  output before the target delivered any content in the sample, as Claude 5-series models
  report a safety-classifier decline: `safe` for a harm estimand, `failure` (a refusal) for an
  over-refusal estimand, or `score`, the default, which keeps the previous behaviour. Each
  such rollout is reported in a `TARGET_POLICY_BLOCKED_SAFE` or `TARGET_POLICY_BLOCKED_REFUSAL`
  diagnostic. Text,
  readable reasoning or a tool call in the stopped output or in any earlier generation of the
  sample, such as an earlier turn of a conversation, counts as delivered content: such a stop
  is scored as usual and reported in a `CONTENT_FILTER_STOP_SCORED` diagnostic, so harm that an
  earlier turn delivered is never counted safe because a later turn was blocked. `safe`
  replaces only a missing or safe score: a failure score, or one the mapping does not cover,
  keeps deciding, with a `CONTENT_FILTER_STOP_SCORED` warning. `inputs.score` sets it for
  every judge: each judge's run under `outputs.judge_claims` takes the primary's value (see
  Claims by judge below), and a judge's score in `inputs.judges` that sets its own is refused.
  Inspect logs only. See [the plan reference](docs/plan-reference.md#inputs).

- **Group scopes for judge calibration, declared accuracies and the judge panel.** A cohort
  that holds several benchmarks no longer has to share one judge calibration and one panel
  summary across them. Three optional plan fields:
  - `inputs.judge_calibration.groups` corrects only the listed groups' rows. In `validated`
    mode, which is experimental with `groups`, it estimates one calibration per cohort and
    listed group from that group's labels, and splits the judge family budget over those
    calibrations (`per_calibration_alpha`). Such a calibration certifies a `weighted_mean`
    over a whole listed group, a row that an unscoped calibration reports only as a
    sensitivity analysis, and the report records `experimental: true` beside it. Labels drawn
    within the cohort's judge strata remain a valid sample for each group, but a small group
    gets few of them, and its calibration is often refused. A claim on a row outside the
    scope is INCONCLUSIVE in a validated plan, and labels outside it raise
    `JUDGE_VALIDATION_OUT_OF_SCOPE`. A `transported` calibration takes no `groups`.
  - `inputs.judges[].accuracy.groups` limits a judge's declared ranges to the listed groups.
    The panel-derived correction, and the judge-error stress box, then apply only where every
    declaration behind the box holds. `PANEL_CORRECTION_NO_COMMON_GROUP` reports a composite
    whose members share no group.
  - `inputs.panel_groups` adds per-group counts, agreement, kappa and McNemar figures to every
    judge, composite and pair in `provenance.judge_panel`, and a "Judge panel by group" table
    to `report.md`.

  Listed groups must be declared: `validate` and `analyze` report an undeclared one as
  `PLAN_UNKNOWN_REFERENCE`, and `analyze` still writes the report. Plans that leave the fields
  out keep their hash and write the same report bytes as before. See
  [Scoping the calibration to groups](docs/judge-misclassification.md#scoping-the-calibration-to-groups)
  and the [plan reference](docs/plan-reference.md#inputs).

- **Stratified validation labels (experimental).** `inputs.judge_calibration.sampling:
  stratified_within_judge_stratum`, with `validation_strata` (a CSV of
  `observation_id,stratum` for every classified observation), reads human labels drawn by a
  stratified design, such as one that labels the judges' disagreements more often. Each
  declared stratum within a cohort's judge stratum gets exact hypergeometric bounds on its
  truly unsafe count and is weighted by its size, so oversampled strata count as often as they
  occur, not as often as they were labelled. The judge's accuracy then gets Clopper-Pearson
  bounds at the summed counts, out of the judge stratum's size, because the run's rollouts only
  sample it: even a stratum labelled in full keeps a binomial width. Each endpoint's budget is
  halved between the two steps, so with a single declared stratum the ranges are wider than
  the default `sampling` gives the same labels. `report.json` records each fine stratum's
  counts and bounds as `fine_strata` in each calibration's entry of
  `provenance.judge_calibration.cohorts`, per cohort or, with
  `groups`, per cohort and group. `report.md` lists them. New diagnostics:
  `JUDGE_VALIDATION_STRATA_INCOMPLETE` and `JUDGE_VALIDATION_STRATA_UNMATCHED` (errors),
  `JUDGE_FINE_STRATUM_UNLABELLED_COMPLETED` (warning) and `JUDGE_CALIBRATION_STRATIFIED`
  (info). The delta-method reference is not reported under
  this design. Plans without the field keep their hash and report bytes. See
  [Stratified labels](docs/judge-misclassification.md#stratified-labels-experimental).

- **Claims by judge (experimental).** With `outputs.judge_claims: true`, a plan with a judge
  panel also decides every claim on each declared judge's verdicts, as the plan would with that
  judge as its primary outcome, over the same rollouts. Each judge in
  `provenance.judge_panel.judges` of `report.json` gains its run's `integrity` and
  `diagnostic_counts`, its `claims` rows (status, descriptive status under an exploratory
  intent, estimate, lower and upper bound, method, reasons) and the `configuration` its cohorts'
  runs record: a SHA-256 fingerprint of the scorer named like the judge's score and the model
  roles, `null` where none are recorded. The rows' bounds are on the judge's reported risk;
  under a validated judge calibration, whose thresholds are about true risk, every row is
  `INCONCLUSIVE`, as the primary outcome's claims are without a certifying correction. Where
  the plan can run a `noisy_valid_test` claim's test (a validated calibration with
  `sampling: simple_random`), a judge's run bounds the claim as a `corrected_interval` claim,
  since the test needs labels of that judge's verdicts; where it cannot, the claim is the same
  plan error in each judge's run as in the primary's. The scorer's options and metadata, which
  can hold request headers, tokens, endpoints and local paths, never reach the report; their
  fingerprint is unsalted, so a value with few possibilities, such as an internal address, can
  be guessed back from it by someone who knows the rest of the entry. A model role
  keeps only its name and model id: its `base_url`, `args` and `extra_headers`, which can hold
  endpoints and credentials, never reach the report. A run whose scorers or model roles cannot
  be read, such as text that is not JSON or a role without a model id, is left out of the
  record with a `JUDGE_CONFIGURATION_UNREADABLE` warning. Each judge's run carries what the
  plan, its decision policy and its registration receipt were checked for, so a receipt for
  another plan or a failed decision policy makes it `invalid` as it does the primary analysis.
  An invalid report never shows a judge's run as anything but `invalid`, nor a row as anything
  but `INCONCLUSIVE`. A judge's run takes the primary score's `content_filter_stops`, which a
  judge's score cannot set, and its `token_limit` where the judge leaves it at `score`, so it
  counts the rollouts the target's provider stopped as the primary analysis does (an inherited
  `token_limit` also covers a truncated rollout the judge left unscored for another reason,
  which can narrow its row; `token_limit: missing` on the judge opts out); a plan whose
  primary `content_filter_stops` misfits a judge's `errors` is refused when it is loaded.
  A judge whose own `score` asks for what the source cannot give, such as the stop reasons
  `token_limit` needs, gets an `invalid` run with `INGEST_FAILED` and rows that decide nothing,
  instead of stopping the analysis. `report-full.md` adds a "Claims by judge (experimental)"
  table to the judge panel section. The rows do not enter the decision. The option requires
  `inputs.judges`. Plans that leave it off keep their hash and write the same report bytes as
  before. See
  [Claims by judge](docs/plan-reference.md#claims-by-judge-experimental).

- **Transported judge calibration (experimental).** `inputs.judge_calibration.mode:
  transported` takes a judge's sensitivity from a published human-labelled validation set,
  such as HarmBench's classifier validation set, instead of labels on the run's own outputs.
  The set's name, source, judge and rubric are required. Only the sensitivity is
  transported: a published set's false-positive rate does not carry over to another model's
  outputs. The run's false positives are therefore bounded only by its own judged failures.
  Each correctable row gains a `judge_corrected_transported` interval from zero to the
  judge-system upper bound divided by the one-sided Clopper-Pearson lower bound on
  sensitivity. Each interval states the transport assumption, and that the bound treats the
  published set's human-unsafe items as independent draws, which several completions of one
  behaviour are not. These intervals are diagnostic sensitivities and never decide a claim. The
  analysis emits `JUDGE_CALIBRATION_TRANSPORTED`, and `report-full.md` names the published set
  in its judge misclassification section, also when no group-risk row is corrected.
  Comparisons are not corrected: the warning `JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON`
  names them. Fixed and sampled designs run it; under a sampled design the interval bounds the
  frame mean when transport holds on every prompt of the frame. Generative plans do not offer
  it. Plans without the mode keep their hash and write the same report bytes as before. See
  [Transported mode](docs/judge-misclassification.md#transported-mode-a-published-sensitivity-experimental).

- **A single-valued risk bound for system models (experimental).** Fault trees and Markov
  models take one number per component, not an interval at a chosen alpha. With
  `outputs.direct_mean_intervals.single_valued_bound: true`, every direct-mean interval also
  reports M, the smallest U_δ + δ over its own one-sided upper bounds U_δ for that node alone.
  M needs no alpha and has P(M < R) ≤ R. It does not enter claims and is not simultaneous.
  Each interval in `report.json` gains `single_valued_bound`, `single_valued_bound_delta` and
  `single_valued_bound_guarantee`, and `report.md` a "Single-valued bounds (experimental)"
  section. Under a sampled design or with judge calibration the plan is refused with
  `PLAN_SINGLE_VALUED_BOUND_UNSUPPORTED`. Plans that leave the option off keep their hash and
  write the same report bytes as before. See
  [Single-valued bound for system models](docs/statistical-methods.md#single-valued-bound-for-system-models-experimental).

- **Comparison planning (experimental).** `inspect-safety-stats plan comparison` says, before a
  run, which candidate-minus-baseline risk differences a design resolves at a margin: the
  largest whose upper bound falls at or below `--margin`, and the smallest whose lower bound
  reaches it, each at the target power. It reads the design manifest as `plan fixed-design`
  does, takes rollout counts per arm, and plans unpaired arms, pairs of rollouts with the same
  prompt and repeat, or prompt clusters. The guarantee is Hoeffding's, for a `comparisons`
  entry with `method: weighted_hoeffding` only: a `comparison_sets` entry bounds each arm
  separately, which is usually wider, and `weighted_kl_chernoff` and
  `weighted_empirical_bernstein` can be wider with uneven prompt weights or at moderate
  failure rates, so the numbers do not hold for them. The setup `init` writes, a
  `comparison_sets` entry with `weighted_kl_chernoff`, is therefore not covered; the output's
  `Interval method` line says so. In Python:
  `stats.planning.comparison_resolvable_differences`. See
  [Planning a comparison](docs/model-comparisons.md#planning-a-comparison-experimental).

- **Sensitivity intervals say what they vary (changes report bytes).** The interval role
  `sensitivity` covered both intervals under another independence profile and judge-error
  corrections, so an independence sensitivity could be read as a judge sensitivity. Each
  interval in `report.json` now has `varied_assumption`: `independence` or `judge_error` on a
  sensitivity interval, `null` on a primary or reference one. `report.md` and `report.html`
  show the role as `sensitivity (independence)` or `sensitivity (judge error)`. The role values
  are unchanged, and reports written before the field still load.

- **Where effects sit (changes `report-full.md`).** When a comparison has hierarchy rows, or
  decisions are shown and a claim's evidence has hierarchy rows, `report-full.md` gains a
  section after the claims. It lists the hierarchy nodes whose comparison interval excludes
  zero difference (a ratio of 1), with the comparison's overall effect beside them, and the
  nodes whose interval or estimate is above a claim's acceptable limit, saying whether the
  whole interval or only the estimate is above. A node without an estimate, because one of its
  prompts has no classified rollout, is still listed when its interval is above the limit;
  nodes that neither their interval nor their estimate can place, and comparison nodes with no
  interval, are counted. Values are compared as printed. Rows listed on pointwise intervals are
  flagged, and under a validated judge calibration, whose claims are about true failure risk,
  the rows say that they are judge-reported. An effect concentrated in one category was
  visible only by reading every hierarchy row. `report.json`, the CSV files, `report.md` and
  `report.html` are unchanged, and so is `report-full.md` when no section is due. See
  [Where effects sit](docs/interpretation.md#where-effects-sit).

- **Random pilot selection (experimental).** `plan pilot --manifest DESIGN.csv
  --per-stratum K --seed S --output PILOT.csv` draws `K` prompts at random from each
  stratum of a design manifest (`--strata`, `task_id` by default), where a pilot taken from
  the first prompts of each category could differ systematically from the rest. The draw
  depends only on the seed and the prompt identifiers, and a larger pilot with the same seed
  contains a smaller one. The output is the whole manifest with `pilot_stratum`,
  `pilot_inclusion_probability` and `pilot_selected`, which is the frame of a fixed-size
  sampled design as it stands, and with `pilot_seed` and `pilot_per_stratum`, which
  record the draw. Equal draws from strata of unequal size over-represent the small ones, so
  weight each pilot prompt by `1 / pilot_inclusion_probability`, as the sampled-design
  analysis does. Analyse a pilot as exploratory: a pilot enlarged after its outcomes were seen
  is not a fixed-size confirmatory design, and a confirmatory analysis uses fresh rollouts
  under a plan frozen before they ran. `draw_pilot` in `inspect_safety_stats.ingest.pilot`
  does the same in Python, on a manifest read as text (`dtype=str, keep_default_na=False`);
  it refuses a `prompt_id` that is not text, which would hash differently from the command's.
  See [Drawing a random pilot](docs/design-based-sampling.md#drawing-a-random-pilot).

- **Checking a sampled design's items against the recorded draw (experimental).** A sampled
  design's checks counted the items evaluated but not which ones, so a run on the first
  prompts of each stratum by identifier passed a fixed-size design's checks as if it were
  the random draw. `design.selection_column` names a manifest column that marks the frame
  items the draw selected, such as a pilot frame's `pilot_selected`; `analyze` and
  `validate` then report `SAMPLE_SELECTION_MISMATCH`, an error that makes integrity invalid
  and withholds the cohort's bounds, when a cohort's observed items differ from them. A plan
  without the field keeps its hash and writes the same report bytes as before. See
  [Checking which items were evaluated](docs/design-based-sampling.md#checking-which-items-were-evaluated-experimental).

- **Descriptive verdicts in exploratory reports (changes report bytes).** An exploratory
  analysis makes every claim `INCONCLUSIVE`, so a reader had to compare its bounds with the
  thresholds by hand. Each claim in `report.json` now has `descriptive_status`: in an
  exploratory analysis with valid integrity and certifying bounds, the status a confirmatory
  analysis of the same bounds would give, with a reason that says so and that it certifies
  nothing; otherwise `null`. The claim's `status` and the report's `decision` are unchanged.
  When decisions are shown, `report.md` shows it beside the status, as
  `` `INCONCLUSIVE` (descriptive: `PASS`) ``, with a note under the claims table, and
  `report.html` as a dashed badge. A `noisy_valid_test` claim
  in an exploratory analysis now reports the test's bounds and reasons, where it reported
  none.

- **Verdicts at other thresholds (changes report bytes).** Each claim in `report.json` has a
  `verdict_range`: the `acceptable_upper` values at which it would `PASS` and the
  `unacceptable_lower` values at which it would `FAIL`, read off the bounds it was decided on.
  With `include_decisions`, the claims table in `report.md` has a column that states the same
  range. Verdicts, bounds and error rates are unchanged. See
  [Verdicts at other thresholds](docs/interpretation.md#verdicts-at-other-thresholds).

- **Rollout files (experimental).** Results from any evaluation system can be analysed from a
  runs file and a rollouts file in JSON Lines or CSV: `analyze` and `validate` take
  `--runs` and `--rollouts` in place of `--logs`, and `source=RolloutFiles(...)` in place of
  `logs=` in Python. See [Rollout files](docs/rollout-files.md).

- **Judge-error stress for threshold claims (experimental).** With
  `report.judge_error_stress: true`, every claim in `report.json` gains a `judge_error_stress`
  record, and `report.md` gains a section after the claims. For a claim decided on
  judge-reported bounds, the record gives the sensitivity and specificity at which the PASS or
  FAIL would no longer hold for true risk. When the primary outcome's judge, or its `any` or
  `all` composite, declares accuracy ranges, it also gives the verdict over that box and whether
  it matches the claim. A judge's box applies only when it is the one judge with declared
  ranges that reads the primary score with the primary mapping; otherwise the warning
  `JUDGE_STRESS_BOX_UNMATCHED` says why no box applies. The true-risk bounds over the box invert the claim's judge-reported
  bounds as the panel-derived correction does, with unclassified rollouts inverted in their
  conservative completion, which keeps the bounds valid for tail measures and under
  `conservative_bounds` and `exclude`. A claim whose row holds a rollout classified by a plan
  policy, such as a policy block counted safe or a missing score counted as a failure, is not
  stressed, because the inversion would read that rollout as a judge verdict. It never changes
  a claim. Plans without the field keep their hash and
  their output. See
  [Stressing a threshold verdict](docs/judge-misclassification.md#stressing-a-threshold-verdict-experimental).

- **A live rewrite kernel without an external API.** The live generative example runs with an
  open-weight model served locally through Inspect's `vllm/` provider:
  `examples/generative_live/plan-local.yaml` with the kernel factory `build_local_kernel`. The
  example's kernel now takes its model, template and decoding settings from the plan rather
  than keeping its own copies, passes a seed fixed by each request, validates at temperature
  0, and reports a refusal as an invalid rewrite, so refusals show by arm in
  `acceptance.csv`. See
  [Without an external API](examples/generative_live/README.md#without-an-external-api).

- **Plan schema 3.0.** Plans name run and rollout fields neutrally, so one plan works with
  any source: `run_key`, `model`, `status` and `run_id` for runs; `run_key`, `item_id`,
  `repeat` and `rollout_id` for rollouts; any other source field as `attributes.<name>`.
  `design.rollout_key` defaults to `[run_key, item_id, repeat]`, and `inputs.sample_join` is
  removed. Schema 2.0 plans keep their meaning and their hash. See
  [Schema versions](docs/plan-reference.md#schema-versions) for the migration.

- **Two-arm comparison sets.** `comparison_sets` accept two cohorts, where they needed at least
  three. A two-arm set gives the same estimates and bounds as the plain comparison written
  with the set's settings.

- **Help text for every command.** `analyze`, `validate`, `version`, `hash-plan`,
  `plan zero-failure`, `schema plan` and `schema report` showed an empty line in `--help`; each
  now has a one-line description.

- **Claim verdicts under every independence profile.** Each claim in `report.json` gains
  `profile_verdicts`: one row for its evidence's primary profile, which repeats the claim's
  status and bounds, and one row for each sensitivity profile, which decides the same
  thresholds on that profile's interval of the same method. A row carries the profile, its
  role, the status, the interval id, the bounds, the independent block count and the effective
  sample size. `report-full.md` adds a "Verdicts by independence profile" table under the mandatory
  claims when a claim has a sensitivity profile, and the HTML claim card lists the verdicts.
  Both label each row primary or sensitivity, under a note that a sensitivity verdict cannot
  replace the primary one.
  The claim's status and the decision stay the primary profile's; a sensitivity verdict is
  shown, never selected. An exploratory claim's rows are all `INCONCLUSIVE`, like the claim,
  whatever its `descriptive_status`. Claims decided on a judge correction or the noisy-valid
  test have no per-profile rows, and say so when their evidence has sensitivity profiles.
  Reports whose plans declare no sensitivity profile gain only the primary row. A sampled
  design takes the frame item as the independent block, so its claims have only the
  `frame_item` row, and a sensitivity profile declared for it gives a new
  `PLAN_SAMPLED_SENSITIVITY_PROFILES` warning from `validate` and `analyze`. See
  [Verdicts by profile](docs/independence-profiles.md#verdicts-by-profile).

### Changed

- **A mapping value that no score can match is refused when the plan loads.** An outcome
  mapping's `failure_values` and `safe_values`, in `inputs.score`, a judge's score or an
  outcome mapping sensitivity, now list text, numbers and booleans only. A list or an object
  among them made reading the scores fail with a bare `unhashable type` message
  (`INGEST_FAILED`), and in an outcome mapping sensitivity it stopped `analyze` with a
  `TypeError` although `validate` passed the plan. A missing value (`null`, `.nan`), which
  never reaches the mapping, and a date, which YAML reads from an unquoted `2024-01-01` and
  no score value equals, were accepted and matched nothing. Each is now an error that names
  the field and the value and says what a mapping can list. Plans that loaded before and list
  none of these values are unchanged, and so are their hashes and reports.

- **Judge corrections bound the population share of unclassified rollouts (changes
  results).** A judge correction keeps rollouts with no judge verdict at their worst case and
  maps only the classified mass through rho. It took the run's own share of such rollouts as
  the population's, so a corrected upper bound could fall below true risk whenever the run saw
  fewer than the population holds, a run that saw none included. In a seeded simulation of a
  certified generative root with 30% of units unscored, the bound missed true risk in 23% of
  runs against a reported alpha of 0.10. Every judge correction now takes a one-sided upper
  confidence bound on the population share, under the row's own independence assumption:
  fixed-design rows in `assumed`, `validated` and `transported` mode, comparison arms in
  `assumed` and `validated` mode, and generative estimates. Its effects:
  - A `validated` correction bounds the share at a quarter of the calibration's alpha and adds
    that to its budget. `family_alpha`, the coverages and a claim's `alpha` grow by it: a claim
    at family alpha 0.05 decided on a correction with judge alpha 0.05 now reports 0.1125.
    Each row bounds its own share, so these hold for each row or claim, not jointly over rows.
  - Corrected bounds widen where few rollouts bound the share. With none unclassified the
    bound is about `log(1/alpha) / n` over `n` independent rollouts, and about
    `log(P/alpha) / n` for each of `P` prompts of `n` rollouts when the row assumes nothing
    between prompts. The lower end can fall by that times `rho_0` and the upper end rise by
    that times `1 - rho_1`, so a box whose `rho_1` reaches 1 and `rho_0` reaches 0 is
    unchanged but for floating-point rounding, which can move a bound by a unit in the last
    place either way.
  - The maximum and the top-k mean ignore weights, so a prompt of weight zero can set their
    bounds. Their share bound counts every prompt the row lists, that one included; a
    weighted mean and a CVaR count only the prompts they weight.
  - A sampled design's share is taken at 1, even when the run has no unclassified rollout.
    Its judge corrections therefore no longer lower the upper bound below the judge-system
    upper bound: they can no longer show that the judge's false positives make true risk
    lower than judge-reported risk. Under an excluding policy they are `[0, 1]`.
  - Under `missing: exclude` or `errors: exclude` the judge-system bounds cover only the
    classified rollouts' rate. The correction now corrects that rate and lets the share move
    it at its worst case. A weighted mean's share bound holds only for the prompts' mean
    share, which may sit on the prompts with the lowest classified rate or on those with the
    highest, so each end moves by the whole bound; a tail measure's holds for every prompt, so
    it is mixed in, `[(1 - s_U) A_L, (1 - s_U) A_U + s_U]`. It used to subtract the share from
    an upper bound that never held it, and a `transported` correction left the excluded
    rollouts out entirely: a maximum over two prompts, half of one prompt's scores missing,
    got an upper bound of 0.4625 where true risk could be 0.5.
  - A row or comparison arm holding rollouts a plan policy classified is corrected with
    `rho_1` free to reach 1 and `rho_0` to fall to 0. Where its bounds complete the
    unclassified rollouts, the share drops out of both ends of that correction, so such a row
    bounds no share and spends no alpha on one. Under an excluding policy it bounds the share
    like any other row. See
    [Rollouts a policy classifies](docs/judge-misclassification.md#rollouts-a-policy-classifies).
  - Generative corrected rows in `report.json` gain `unclassified_upper` and
    `unclassified_alpha`.

  See [Unclassified rollouts](docs/judge-misclassification.md#unclassified-rollouts).

- **The generative report page says why a judge correction is a sensitivity (changes
  `report.html`).** The judge misclassification section lists the warnings of the corrected
  root and slices it shows, which `report.json` already held, and says that the labels weight
  every classified unit equally rather than every unit.

- **Errors about a plan that does not fit the logs say what the logs hold (changes
  diagnostics).** These errors named only what the plan asked for, so a user had to open the
  logs to see what the plan should say:
  - `COHORT_EMPTY` for a cohort now names the values the cohort matches on and, for each of
    those fields, what the selected evaluations have and how many have it, for example
    `it requires model = 'openai/gpt-4o', and the selected evaluations have model
    'mockllm/model' in 1 evaluation`. Only the model is shown: any other field, such as a task
    argument, is run configuration, so the message says only how many evaluations have the
    required value, another value or none. When the evaluations it matches also match another
    cohort, and so were put in neither, it says so instead.
  - `SCORE_MISSING` and `JUDGE_SCORE_MISSING` for an absent score now list the scores the logs
    have.
  - A new `SCORE_MAPPING_MISMATCH` error, or `JUDGE_SCORE_MAPPING_MISMATCH` for a judge, lists
    the score values that the mapping does not cover, with their types and how many rollouts
    have each, at most ten of them, and says how to map them. It sorts ahead of the
    `SCORE_UNMAPPED` errors, one per rollout, which are unchanged and still withhold their
    prompts' bounds. Its scope is the plan path, so it withholds no other bounds.

  Reports of valid analyses are unchanged, except that an outcome mapping sensitivity
  (`inputs.score.mapping_sensitivities`) whose mapping leaves an observed value uncovered
  counts one `SCORE_MAPPING_MISMATCH` in its `diagnostic_counts`, beside its `SCORE_UNMAPPED`
  errors, while the primary analysis stays valid.

- **A duplicate rollout key no longer shows a run's configuration (changes `report.json`).** A
  field of `design.rollout_key` that a rollout does not record is read from its run, and the
  `ROLLOUT_DUPLICATE` message showed its value. A key that named a run field such as
  `model_generate_config` or `model_base_url` therefore copied request headers and endpoints
  into `report.json`, `report-full.md`, `report.html` and the output of `validate`. The message
  now names such a field, as `<model_base_url of the run>`, instead of showing its value. The
  diagnostic's scope still names the run. Keys made of rollout fields only, such as the
  default, give the same message as before.

- **A claim on a tail or threshold measure over one rollout per prompt degrades integrity
  (changes `report.json`).** At one rollout per prompt every per-prompt rate is 0 or 1, and
  [the statistical methods](docs/statistical-methods.md#249-statistical-uncertainty) call the
  measures nonlinear in it misleading or vacuous there. `validate` and `analyze` passed such a
  plan as valid without a word. They now give a `TAIL_MEASURE_SINGLE_ROLLOUT` warning on each
  claim on a `cvar`, `maximum`, `top_k_mean`, `entropic`, `threshold_exceedance` or
  `threshold_excess` measure when at least half the prompts of its cohort and group have one
  classified rollout, and integrity is `degraded`. The claim is still evaluated. Claims on the
  weighted mean, whose interval handles one rollout per prompt, and on `worst_of_k`, which gives
  no estimate below `k` rollouts per prompt, are not affected.

- **Assumed judge calibration corrects comparisons too (changes `report.json`).** In
  `judge_calibration.mode: assumed`, each declared scenario corrected the group risks but left
  every comparison on the judge-reported scale, with no diagnostic, although the docs said
  comparisons were corrected on both arms. Each scenario now corrects every comparison whose
  primary interval exposes both arm bounds, with the scenario's box on both arms, adding a
  `judge_corrected_<scenario>` sensitivity interval with `diagnostic` eligibility. A
  comparison that cannot be corrected, such as a direct contrast interval, is named in
  `JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON`. No claim moves: assumed corrections never
  decide one. The "Comparison interval sensitivity" table of `report-full.md` gains a Scenario
  column that names each corrected row's scenario and marks the primary one, and the interval
  tables of `report.html`, for comparisons and for risk rows, do the same. Reports without
  assumed scenarios show the same tables as before. See
  [Comparisons](docs/judge-misclassification.md#comparisons).

- **Judge corrections no longer read rollouts a plan policy classified as judge verdicts
  (changes `report.json`).** A target-side policy block counted safe under
  `errors: end_to_end`, a sample error or missing score counted as a failure under
  `count_as_failure`, and a content-filter stop or unscored truncated answer under
  `content_filter_stops` or `token_limit` were mapped through the judge's `rho_1` and
  `rho_0` like a judge verdict, which moved both corrected bounds the wrong way. A row, hierarchy
  node or comparison arm holding such a rollout is now corrected, in `assumed` and `validated`
  mode, with `rho_1` free to reach 1 at the upper bound and `rho_0` to fall to 0 at the
  lower, which holds whatever the policy's share. The interval states it in its assumptions,
  the new INFO diagnostic `JUDGE_CORRECTION_POLICY_CLASSIFIED` names the groups per cohort, and
  the row has no asymptotic reference interval. A row counts as holding such a rollout when any
  of its prompts does, a prompt of weight zero included. A row whose sample holds none keeps the
  box as given, which assumes its population holds none either. Its interval states this where
  the plan sets a policy that classifies rollouts, or the cohort's sample holds one. See
  [Rollouts a policy classifies](docs/judge-misclassification.md#rollouts-a-policy-classifies).

- **The judge strata of a validated calibration hold the judge's verdicts only (changes
  `report.json`).** A rollout a plan policy classified, such as a policy block counted safe or
  a missing score counted as a failure, entered the judge stratum of its `failure` value, so
  its label was read as evidence about the judge and it counted in the stratum's size. Such
  rollouts are now in neither judge stratum, nor in the truth strata of the noisy-but-valid
  test. A label or `validation_strata` row on one is left out of the calibration, with the new
  warning `JUDGE_VALIDATION_POLICY_CLASSIFIED`, and a strata file no longer has to place them.
  Plans whose rollouts are all classified by the judge are unchanged. See
  [Workflow](docs/judge-misclassification.md#workflow).

- **The noisy-but-valid test refuses a row that holds rollouts a plan policy classified
  (changes claims).** The test inverts the judge-reported bounds over the judge's sensitivity
  and false-positive rate, reading every classified rollout as a judge verdict, so a policy
  block counted safe lowered the true-risk upper bound it certified against. Such a
  `noisy_valid_test` claim is now INCONCLUSIVE, with the reason. A `corrected_interval` claim
  on the same row is decided on the judge correction, which covers such rollouts.

- **The panel-derived correction no longer reads rollouts a plan policy classified as panel
  verdicts (changes `report.json`).** Its Rogan--Gladen inversion took a false alarm off a
  policy block counted safe, and scaled a policy failure up for missed failures, which moved
  both bounds of `judge_corrected_panel_derived` the wrong way. On a row or hierarchy node
  holding such a rollout, the inversion now lets sensitivity and specificity reach 1, the
  interval says so in its assumptions, and `JUDGE_CORRECTION_POLICY_CLASSIFIED` names the
  groups. Rows without such rollouts are unchanged. See
  [Rollouts a policy classifies](docs/judge-misclassification.md#rollouts-a-policy-classifies).

- **The panel-derived correction needs the composite's verdict read as it is (changes
  `report.json`).** The correction inverted the judge-reported bounds over the box derived for
  the composite's verdict even when the primary mapping counted that verdict otherwise, for
  example its false verdict as failure, where the box describes another outcome. Such a plan
  now gets no `judge_corrected_panel_derived` interval, and the new warning
  `PANEL_CORRECTION_VERDICT_UNMATCHED` says why. The judge-error stress already refused it.

- **A report states the `token_limit` and `content_filter_stops` assumptions for the value the
  plan took (changes `report.json`, `report.md` and `report-full.md`).** Their entries in
  `assumptions_used` stated the default, that such rollouts count by their score, even when the
  plan classified them as safe or as failures. A registry assumption can now carry a statement,
  and an if-wrong text, for each non-default value of its field, and the report and
  `docs/diagnostics.md` give them. Plans at the defaults are unchanged.

- **`report.include_decisions` hides decisions, and every reader defaults it to false (#58).**
  The option was read as true by `report.html` and by the Python report model when a report did
  not record it, and as false by the plan and the CLI. Every reader now reads it as false. The
  option hides decisions in the renderings; `report.json` and the data embedded in
  `report.html` still hold the overall decision and every claim's status, and the docs now say
  so. Reports written by `analyze` always record the option, so only the viewer script in
  `report.html` changes.

- **The offline demo uses the plan `init` writes.** `examples/demo/plan.yaml` and
  `design.csv` are now the output of `inspect-safety-stats init --group-by risk_domain
  --baseline mockllm/baseline` on the demo logs, plus two tail measures (`cvar_90`, `worst`).
  The demo task records each prompt's risk domain in its sample metadata so that `init` can
  group by it. The demo still compares the candidate with the baseline (now through a
  comparison set, as a risk difference and a risk ratio), and its report now emits a
  `TAIL_MEASURES_UNINFORMATIVE` warning for each model, which `show_results.py` prints. The
  numbers the demo prints change; the demo README explains them and the warnings.

- **The plan hash covers only what the author wrote (#56).** `hash-plan`, `plan_sha256` and
  the inference hash leave out every field the plan does not write, so a release that adds a
  field or changes a default no longer changes the hash of an unchanged plan. A field written at
  its default value is hashed, and so writing out a default now changes the hash. Every plan's
  hash changes once with this release: freeze plans, registration receipts and decision
  policies again with `hash-plan`. See [The plan hash](docs/plan-reference.md#the-plan-hash).

- **The generative commands' help says the pipeline is experimental (#24).** The help of
  `generative` and of each of its commands, and of `schema generative-plan`, ends with the
  caveat the README and the generative docs already carry: the generative pipeline is
  experimental, and its public surface may change in any minor release. `schema --help`, which
  lists `generative-plan` beside the other schemas, marks it "(experimental)" as it marks
  `claims` and `decision-policy`.

- **`generative analyze` exits `3`, not `2`, on `ERROR` diagnostics.** Exit `2` means
  INCONCLUSIVE under `analyze --decision-exit-codes`, so a generative report with errors could
  read as an inconclusive decision. Every command now takes its exit status from one table,
  `ExitCode` in `cli.py`, and no other exit status changes. See
  [Exit codes](docs/api-reference.md#exit-codes).

- **`hash-plan` reports a plan it cannot read or validate on one line, and exits `3`.** It
  prints `Plan invalid:` followed by each failing field as `field: message`, or the reading
  error, on one line on stderr, where it printed a traceback and exited `1`. stdout still
  carries only the hash, so a script that reads it is unaffected; one that tested for exit `1`
  should test for a nonzero status.

- **The generative report writes `checksums.json`.** It holds the SHA-256 of every other file
  the report writes, as the fixed-design report's does. The other files keep their bytes.
  Both reports now write their files through one module, `inspect_safety_stats.report.artifacts`.

- **`report.json` holds everything the report shows (changes `report.json` and
  `report.html`).** `report.md`, `report.html` and the CSVs are rendered from the report model
  alone, so a saved `report.json` renders them again without the plan or the design CSV.
  `report.json` gains `design.prompts`, the rows of `design.resolved.csv`, and a typed
  `presentation` that adds `decimal_places`, `primary_independence_profile`, `claim_evidence`
  (each claim's comparison, or cohort, group, measure and measure type) and, when the plan has
  one, `judge_calibration` (mode, sampling, primary scenario, groups, scenarios). The HTML
  page embeds the report alone, so its audit trail no longer shows the resolved plan: the
  "Resolved plan" panel points to `plan.resolved.yaml` beside `report.json`. The page sets its
  footer, which holds the scope caveat, before it renders the sections, so that an error in a
  section cannot leave the page without it. In Python, `SafetyReport.presentation` is a
  `ReportPresentation` rather than a dictionary: read `report.presentation.include_decisions`.
  Reading it as a dictionary (`presentation["include_decisions"]`, `presentation.get(...)` or
  `in`) still works, with a `DeprecationWarning`, and will be removed in a later release; a
  field the report does not record reads as a missing key, as it did.
  `report.md` and the CSVs keep their bytes, and no number, status or decision changes.
  `visualize` on a report directory written before this still works: it takes the design rows
  and the primary profile from the files beside `report.json`. The schema version stays `3.0`.

- **A manifest needs only the identifier columns the plan uses.** The design manifest used to
  need `question_id` and `task_id` even when the plan never named them. Now `prompt_id` is
  always required, and `question_id` and `task_id` only when the plan names them, for example in
  `design.hierarchy` and `design.weights` (whose defaults name both), a weight hierarchy, a group
  filter, an independence profile's `block_by` or a comparison's `pair_key` or `cluster_key`. A
  level that the plan doesn't name and the manifest omits is set to each prompt's `prompt_id`,
  so each prompt counts as its own question or task. `plan fixed-design --manifest` requires the
  levels that its `--weights` name. Manifests that were accepted before give the same results.
  See [`inputs`](docs/plan-reference.md#inputs).

- **A sampled design's report states the frame as its scope (changes report text).** Every
  report stated the fixed-design scope, `design: fixed` and "Results apply to the declared
  fixed evaluation set", even under `design.type: sampled`, whose intervals are about the
  whole declared item frame. A sampled design's `scope` now has `design: sampled`, names the
  frame and the recorded inclusion probabilities as the randomness, and says that results apply
  to the frame and not to prompts outside it. The text around it follows: `report.md` counts
  frame items rather than fixed prompts and says that its intervals allow for both the draw of
  items and the rollouts, `report-full.md` and `report.html` head the intervals
  "Sampled-design risk measures", and the hierarchy note of `report-full.md` and the summary
  and footer of `report.html` speak of the frame. Fixed designs keep their wording.

- **Generative judge corrections certify less, and keep the worst case of unscored units
  (#109, changes results).** A `validated` judge calibration pools its human labels over the
  whole cohort, so it measures the judge's accuracy only for an estimate that weights every
  unit equally. It now certifies the corrected root of such a design only; corrected nodes,
  slices and an unequally weighted root, for example under `minimum_rewrites_per_stratum`,
  are `diagnostic`, with a warning that names the reason. Corrected estimates also no longer
  shrink units with no judge verdict by `rho_1`: they keep their worst case, as on the
  fixed-design path (#97), so corrected upper bounds can rise. Each node, slice and comparison
  arm in `report.json` reports the design weight on those units as `unclassified_share`. See
  [judge misclassification](docs/judge-misclassification.md#on-the-generative-path).

- **Unscored pairs are bounded, not a broken pairing (changes results).** With `pair_key`, one
  unscored rollout in either arm left its pair unmatched, so the comparison got `PAIR_BROKEN`
  and the report's integrity became `invalid`. The pair is now kept, and the paired diagnostics
  report bounds over every scoring of the unknown outcomes, with `estimate_lower` and
  `estimate_upper` in place of `estimate`, a `partial_pairing` record in `diagnostic_outputs`
  and a `PAIR_UNSCORED_COMPLETED` warning. A pair unscored in both arms, which was dropped
  without notice, is now counted and bounded too. A key missing from one arm is still
  `PAIR_BROKEN`. When the completions produce more than 50 million distinct 2×2 tables, as from
  668 pairs unscored in both arms, `newcombe_paired` alone is not reported, with a
  `PAIR_NEWCOMBE_UNAVAILABLE` warning, and the integrity status is unaffected.
  `exact_mcnemar_partial` and `newcombe_paired_interval_partial` are exported from `stats`. See
  [Pairs with an unscored arm](docs/model-comparisons.md#pairs-with-an-unscored-arm).

- **`analyze` reports input it cannot read (changes behaviour).** A missing or broken
  design manifest, or logs or rollout files that cannot be read, made `analyze` raise and
  write no report, while `validate` reported the same input as an `INGEST_FAILED` error. Both
  now read their input the same way: `analyze` returns a report whose integrity is invalid,
  with the `INGEST_FAILED` diagnostic, no estimates, and every claim `INCONCLUSIVE`. The CLI
  writes that report and exits `3`, or `0` (`2` with `--decision-exit-codes`) under
  `--allow-invalid`, where it printed `Analysis failed` and wrote nothing. A plan that cannot
  be loaded still fails without a report. Passing both `logs` and `source` to `validate` now
  raises `ValueError`, as it does in `analyze`, where it was reported as `INGEST_FAILED`.

- **An unreadable transcript is reported (adds a diagnostic).** To find where a sample error
  happened, the Inspect adapter reads the log's transcript again, and any exception in that
  read silently left the error's stage `unknown`. A log that cannot be read (a missing,
  truncated or malformed file) now gives a `TRANSCRIPT_UNREAD` warning naming the log, the
  error type and how many sample errors it leaves without a stage; their classification is
  unchanged, since an unknown stage is never counted as a policy block. Any other exception
  is no longer swallowed and fails the input as `INGEST_FAILED`. Analyses whose logs can be
  read are unchanged.

- **Kernel loading leaves `sys.modules` and `sys.path` as it found them (Python API).**
  `materialize_run` and `generative.runner.resolve_kernel` no longer put the working
  directory first on `sys.path` before importing a `module:attribute` kernel; the module is
  imported from `sys.path` as the caller set it. A `path/to/file.py:attribute` kernel is no
  longer left in `sys.modules` under its file name, so a kernel file named like an installed
  module (`json.py`) no longer replaces that module for the rest of the process. The
  `generative materialize` command still finds a kernel module in the working directory.
  Reports are unchanged.

- **An unlabelled judge stratum is completed, not refused (changes results).** A validated
  judge calibration needed a human label in both judge strata. A stratum without one is now
  completed conservatively: its error rate is bounded by [0, 1], the correction is still
  reported, and a `JUDGE_STRATUM_UNLABELLED_COMPLETED` warning names the stratum. With both
  strata unlabelled the calibration is still refused. `JudgeCalibration` gains
  `completed_strata`. See
  [Judge misclassification](docs/judge-misclassification.md#estimation-and-error-budget).

- **One grader across the JailbreakBench comparison arms (example).** The plan that
  `examples/real_jailbreakbench/run_jailbreakbench.py` writes now matches every cohort on
  `task_arg_grader_model` as well as the model, and log selection picks logs by model and
  grader. Before, a log of an arm graded by another model could enter the comparison. See
  [One judge for every arm](docs/model-comparisons.md#one-judge-for-every-arm).

- **Unbiased `worst_of_k` estimates (changes results).** The row estimate plugged each prompt's
  failure rate into `1 - (1 - p)^k`, which understates the risk. It is now the unbiased
  U-statistic, `1 - C(n - f, k) / C(n, k)` per prompt, weighted over prompts, so estimates
  rise. When a prompt has fewer than `k` classified rollouts the row reports no estimate, with
  a warning. The bounds are unchanged. `worst_of_k_estimate` is exported from `stats`.

- **Reports name inputs relative to the plan's directory (changes report bytes).** The design
  manifest, the judge validation manifest, the provenance `sources` and each observation's
  `source` were recorded as absolute local paths, so the same analysis wrote different bytes
  in different checkouts and revealed the local directory layout (#63). They are now relative
  to the plan's directory, in POSIX form. URIs, and paths with no relative form, keep their
  form apart from using forward slashes. Observation ids and validation manifests are
  unchanged.

- **A minimal plan is told what it is missing (#57).** Plan validation requires
  `outputs.prompt_bounds.family` whenever prompt bounds are enabled, pointwise ones included,
  where pointwise bounds without a family were silently absent; with prompt bounds disabled a
  family is no longer needed. The messages for a missing family and for a missing
  `expected_rollouts` column say what to add or which switch turns the requirement off, and a
  plan without risk measures gets a `PLAN_NO_RISK_MEASURES` warning.

- **Unquoted schema versions.** `schema_version: 2.0` and `3.0` written without quotes are
  accepted, and hash the same as their quoted forms. A schema 1.0 plan is refused with what
  schema 2.0 changed, and any other version with the supported ones.


- **Inspect AI is an optional extra (breaking for installation).** `inspect-ai` and `pyarrow`
  are no longer installed by default. Install `inspect-safety-stats[inspect]` to read Inspect
  logs with `--logs` or `logs=`; without the extra, reading them fails with an error that names
  it. Rollout files need no extra. The `dev` extra includes it.

- **Report schema 3.0: neutral observation and provenance fields (breaking).** Reports name
  observation fields after no evaluation tool: `source_log`, `source_log_sha256`, `eval_id`,
  `sample_id`, `inspect_id` and `epoch` in `observations.csv` and `report.json` are now `source`,
  `source_sha256`, `run_key`, `rollout_id`, `item_id` and `repeat`. The cohort fields `eval_ids`
  and `evaluation_count` are now `run_keys` and `run_count`. Provenance replaces `source_logs` and
  `inspect_version_constraint` with `sources` and `source_format`, which names the format read
  and, for Inspect logs, the supported Inspect versions. Observation ids, and so every estimate,
  are unchanged. Plans may still name the old observation fields in `block_by`, `pair_key` and
  `cluster_key`. Update any script that reads these columns or keys.

- **Unchecked safeguards are reported.** A run without a status was silently counted as
  complete, and a confirmatory plan whose runs had no start time skipped the post hoc check
  without a word. Each now gives a warning that degrades integrity: `RUN_STATUS_UNOBSERVED`,
  `START_TIME_UNOBSERVED`, and `CONFIG_UNOBSERVED` for a run that reports no configuration. A
  run without a status still counts as complete. Analyses of Inspect logs, which always carry
  these fields, are unchanged.
