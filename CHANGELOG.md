# Changelog

Changes that a user can observe, newest first. Releases before this file was started are
described in their pull requests.

## Unreleased

### Added

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

### Changed

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
