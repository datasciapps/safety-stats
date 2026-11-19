# Changelog

Changes that a user can observe, newest first. Releases before this file was started are
described in their pull requests.

## Unreleased

### Added

- **Direct `worst_of_k` interval (experimental).** A `worst_of_k` measure with
  `direct_interval: true` gets a KL/Chernoff interval on its group rows. Each prompt enters as
  `floor(n / k)` independent blocks, so the interval narrows as rollouts grow. Before, the only
  interval was the propagated prompt envelope, which can be very wide. The interval is calibrated
  with the direct-mean intervals and needs a fixed design with rollout-level independence; other
  plans are refused. Plans without the field are unchanged. `worst_of_k_kl_chernoff_bounds` is
  exported from `stats`. See
  [Worst-of-k interval](docs/statistical-methods.md#worst-of-k-interval-experimental).

- **Decision policies (experimental).** Claim thresholds can come from a decision policy kept
  apart from the plan, and be set or revised after the run, without changing the plan's hash
  or voiding certification: thresholds never enter the bounds, so the error rate of a verdict
  holds at any thresholds. `analyze` and `validate` take `--decision-policy` (Python:
  `decision_policy=`). A policy names its plan by the inference hash, which leaves out the
  thresholds of `corrected_interval` claims (`hash-plan --inference`); it cannot set
  `noisy_valid_test` claims, whose test uses their thresholds. `report.json` gains a
  `decision_policy` section with every version, its time and whether it came after the first
  evaluation started, and the thresholds each claim was decided at; `report.md` shows the same.
  New diagnostics: `DECISION_POLICY_MISMATCH`, `DECISION_POLICY_UNKNOWN_CLAIM`,
  `DECISION_POLICY_FIXED_CLAIM` (errors) and `DECISION_POLICY_AFTER_GENERATION` (info).
  `schema decision-policy` prints the policy schema. Analyses without a policy write the same
  bytes as before. See [Decision policies](docs/decision-policy.md).

- **Content-filter stops as policy blocks (experimental).** `inputs.score.content_filter_stops`
  decides before the run how a rollout counts when a provider's content filter stopped its
  output before any content, as Claude 5-series models report a safety-classifier decline:
  `safe` for a harm estimand, `failure` (a refusal) for an over-refusal estimand, or `score`,
  the default, which keeps the previous behaviour. Inspect logs only. See
  [the plan reference](docs/plan-reference.md#inputs).

- **Group scopes for judge calibration, declared accuracies and the judge panel.** A cohort
  that holds several benchmarks no longer has to share one judge calibration and one panel
  summary across them. Three optional plan fields:
  - `inputs.judge_calibration.groups` corrects only the listed groups' rows. In `validated`
    mode it estimates one calibration per cohort and listed group from that group's labels,
    and splits the judge family budget over those calibrations (`per_calibration_alpha`).
    Such a calibration certifies a `weighted_mean` over a whole listed group. A claim on a
    row outside the scope is INCONCLUSIVE in a validated plan, and labels outside it raise
    `JUDGE_VALIDATION_OUT_OF_SCOPE`. A `transported` calibration takes no `groups`.
  - `inputs.judges[].accuracy.groups` limits a judge's declared ranges to the listed groups.
    The panel-derived correction, and the judge-error stress box, then apply only where every
    declaration behind the box holds. `PANEL_CORRECTION_NO_COMMON_GROUP` reports a composite
    whose members share no group.
  - `inputs.panel_groups` adds per-group counts, agreement, kappa and McNemar figures to every
    judge, composite and pair in `provenance.judge_panel`, and a "Judge panel by group" table
    to `report.md`.

  Listed groups must be declared. Plans that leave the fields out keep their hash and write the
  same report bytes as before. See
  [Scoping the calibration to groups](docs/judge-misclassification.md#scoping-the-calibration-to-groups)
  and the [plan reference](docs/plan-reference.md#inputs).

- **Stratified validation labels (experimental).** `inputs.judge_calibration.sampling:
  stratified_without_replacement`, with `validation_strata` (a CSV of
  `observation_id,stratum` for every classified observation), reads human labels drawn by a
  stratified design, such as one that labels the judges' disagreements more often. Each
  declared stratum within a cohort's judge stratum gets exact hypergeometric bounds on its
  truly unsafe count and is weighted by its size. A stratum labelled in full is therefore
  exact, and oversampled strata count as often as they occur, not as often as they were
  labelled. `report.json` records each fine stratum's counts and bounds as `fine_strata` in
  each calibration's entry of `provenance.judge_calibration.cohorts`, per cohort or, with
  `groups`, per cohort and group. `report.md` lists them. New diagnostics:
  `JUDGE_VALIDATION_STRATA_INCOMPLETE` and `JUDGE_VALIDATION_STRATA_UNMATCHED` (errors),
  `JUDGE_FINE_STRATUM_UNLABELLED_COMPLETED` (warning) and `JUDGE_CALIBRATION_STRATIFIED`
  (info). The delta-method reference is not reported under
  this design. Plans without the field keep their hash and report bytes. See
  [Stratified labels](docs/judge-misclassification.md#stratified-labels-experimental).

- **Claims by judge (experimental).** With `outputs.judge_claims: true`, a plan with a judge
  panel also decides every claim on each declared judge's verdicts, as the plan would with that
  judge as its primary outcome, over the same rollouts. Each judge in
  `provenance.judge_panel.judges` of `report.json` gains its run's `integrity`, its `claims`
  rows (status, estimate, lower and upper bound, method) and the `configuration` its cohorts'
  runs record: the options of the scorer named like the judge's score and the model roles,
  `null` where none are recorded. `report.md` adds a "Claims by judge (experimental)" table to
  the judge panel section. The rows do not enter the decision. The option requires
  `inputs.judges`. Plans that leave it off keep their hash and write the same report bytes as
  before. See [Claims by judge](docs/plan-reference.md#claims-by-judge-experimental).

- **Transported judge calibration (experimental).** `inputs.judge_calibration.mode:
  transported` takes a judge's sensitivity from a published human-labelled validation set,
  such as HarmBench's classifier validation set, instead of labels on the run's own outputs.
  The set's name, source, judge and rubric are required. Only the sensitivity is
  transported: a published set's false-positive rate does not carry over to another model's
  outputs. The run's false positives are therefore bounded only by its own judged failures.
  Each correctable row gains a `judge_corrected_transported` interval from zero to the
  judge-system upper bound divided by the one-sided Clopper-Pearson lower bound on
  sensitivity. These intervals are diagnostic sensitivities and never decide a claim. The
  analysis emits `JUDGE_CALIBRATION_TRANSPORTED`, and `report.md` names the published set.
  Fixed-design analyses only. Plans without the mode keep their hash and write the same
  report bytes as before. See
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

- **Sensitivity intervals say what they vary (changes report bytes).** The interval role
  `sensitivity` covered both intervals under another independence profile and judge-error
  corrections, so an independence sensitivity could be read as a judge sensitivity. Each
  interval in `report.json` now has `varied_assumption`: `independence` or `judge_error` on a
  sensitivity interval, `null` on a primary or reference one. `report.md` and `report.html`
  show the role as `sensitivity (independence)` or `sensitivity (judge error)`. The role values
  are unchanged, and reports written before the field still load.

- **Where effects sit (changes `report.md`).** When a report has hierarchy rows, `report.md`
  gains a section after the claims that lists the hierarchy nodes whose comparison interval
  excludes no difference, with the comparison's overall effect beside them, and the nodes
  whose estimate is above a claim's acceptable limit, saying whether the whole interval or only
  the estimate is above. An effect concentrated in one category was visible only by reading
  every hierarchy row. `report.json`, the CSV files and `report.html` are unchanged, and so
  is `report.md` when there are no hierarchy comparisons and no shown claim reaches
  hierarchy rows. See
  [Where effects sit](docs/interpretation.md#where-effects-sit).

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
  it matches the claim. It never changes a claim. Plans without the field keep their hash and
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
  sample size. `report.md` adds a "Verdicts by independence profile" table under the mandatory
  claims when a claim has a sensitivity profile, and the HTML claim card lists the verdicts.
  The claim's status and the decision stay the primary profile's; a sensitivity verdict is
  shown, never selected. Claims decided on a judge correction or the noisy-valid test have no
  per-profile rows, and say so when their evidence has sensitivity profiles. Reports whose
  plans declare no sensitivity profile gain only the primary row. See
  [Verdicts by profile](docs/independence-profiles.md#verdicts-by-profile).

### Changed

- **A sampled design's report states the frame as its scope (changes report text).** Every
  report stated the fixed-design scope, `design: fixed` and "Results apply to the declared
  fixed evaluation set", even under `design.type: sampled`, whose intervals are about the
  whole declared item frame. A sampled design's `scope` now has `design: sampled`, names the
  frame and the recorded inclusion probabilities as the randomness, and says that results apply
  to the frame and not to prompts outside it. Fixed designs keep their statement byte for byte.

- **Generative judge corrections certify less, and keep the worst case of unscored units
  (#109, changes results).** A `validated` judge calibration pools its human labels over the
  whole cohort, so it measures the judge's accuracy only for an estimate that weights every
  unit equally. It now certifies the corrected root of such a design only; corrected nodes,
  slices and an unequally weighted root, for example under `minimum_rewrites_per_stratum`,
  are `diagnostic`, with a warning that names the reason. Corrected estimates also no longer
  shrink units with no judge verdict by `rho_1`: they keep their worst case, as on the
  fixed-design path (#97), so corrected upper bounds can rise. Each node and slice reports the
  design weight on those units as `unclassified_share`. See
  [judge misclassification](docs/judge-misclassification.md#on-the-generative-path).

- **`analyze` reports input it cannot read (changes behaviour).** A missing or broken
  design manifest, or logs or rollout files that cannot be read, made `analyze` raise and
  write no report, while `validate` reported the same input as an `INGEST_FAILED` error. Both
  now read their input the same way: `analyze` returns a report whose integrity is invalid,
  with the `INGEST_FAILED` diagnostic, no estimates, and every claim `INCONCLUSIVE`. The CLI
  writes that report and exits `3`, or `0` (`2` with `--decision-exit-codes`) under
  `--allow-invalid`, where it printed `Analysis failed` and wrote nothing. A plan that cannot
  be loaded still fails without a report. The diagnostic's message names the input paths
  relative to the plan's directory, as the report does. Passing both `logs` and `source` to
  `validate` now raises `ValueError`, as it does in `analyze`, where it was reported as
  `INGEST_FAILED`.

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
  `completed_strata`, and `correct_judge_risk` gains `tail`: a completed box lets the judge
  be worse than chance, so a `maximum`, `cvar` or `top_k_mean` row's corrected upper end
  also allows a prompt the judge never flags. See
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
  `cluster_key`. An independence profile that blocks by one of the six new names, where the
  design manifest has a column of that name, is refused: that name used to mean the column.
  Update any script that reads these columns or keys. `visualize` reads schema 3.0 reports only;
  to visualize an earlier report, run its analysis again.

- **Unchecked safeguards are reported.** A run without a status was silently counted as
  complete, and a confirmatory plan whose runs had no start time skipped the post hoc check
  without a word. Each now gives a warning that degrades integrity: `RUN_STATUS_UNOBSERVED`,
  `START_TIME_UNOBSERVED`, and `CONFIG_UNOBSERVED` for a run that reports no configuration. A
  run without a status still counts as complete. Analyses of Inspect logs, which always carry
  these fields, are unchanged.
