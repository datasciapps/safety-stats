# Unsent patch for Inspect AI

This directory holds a fix for [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)
that has not been sent upstream. It is kept here so that it is not lost before it is. Nothing in
this repository applies it or depends on it.

## The problem

`inspect eval-retry` (and `eval_retry()`) given several logs runs them one after another, even
when they came from one eval that ran its tasks concurrently under `max_tasks`. A two-arm run of
this package's generative workflow (the same task with two task-arg values, such as `thinking=`
on and off, run under `max_tasks=2`) therefore resumes at half the concurrency it started with,
and loses the batching its arms shared on the model server.

The cause is in `src/inspect_ai/_eval/eval.py`: `eval_retry_async` calls `eval_async` once per
log and awaits each in turn, and `eval_async` refuses concurrent calls.

## The fix

`eval_async` already accepts several `PreviousTask`s, each with its own task args, as
`eval-set` uses it. The patch groups the logs to retry by their original `run_id` and their
eval-level settings (model as the log records it, generate config, limits, `max_tasks`, and so
on) and runs one `eval_async` call per group. Logs of separate evals, or with different
settings, still run one after another, as before. Results come back in the order the logs were
given. A paragraph in Inspect's `docs/_errors_and_retries.md` describes the behaviour.

## Evidence

Taken on the patch applied to upstream commit `59c134074` (Inspect AI 0.3.290 plus seven
commits):

- The new test `test_eval_retry_runs_several_logs_concurrently_under_max_tasks` in
  `tests/test_retry.py` fails without the change (the retried tasks do not overlap in time) and
  passes with it.
- `test_eval_retry_runs_logs_with_different_models_one_after_another` checks that the logs of a
  multi-model eval keep their own models.
- Every Inspect test file that mentions `eval_retry` passes (2127 tests), including the trio
  variants. `ruff check`, `ruff format --check` and `mypy` are clean on the changed files.

## Applying it

Inspect's repository requires a DCO `Signed-off-by` on every commit. The patch carries a
placeholder author and no sign-off, so whoever applies it signs it as themselves:

```bash
git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
cd inspect_ai
git switch -c eval-retry-max-tasks 59c134074
git am --signoff <this-directory>/0001-eval-retry-runs-logs-concurrently-under-max-tasks.patch
git commit --amend --reset-author --no-edit
```

On a newer upstream commit, `git am -3` resolves small drifts. Rerun the tests above after any
conflict.

## Not fixed here

The same loop in `eval_retry_async` reassigns its own parameters
(`max_tasks = max_tasks or <log>.eval.config.max_tasks`, and the same for `max_samples`,
`log_format` and others), so the second log inherits the first log's value instead of its own.
That is a separate defect and would be a separate pull request.

## Sending it

Opening the issue and the pull request upstream is the maintainer's step.
