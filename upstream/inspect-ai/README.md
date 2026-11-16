# Unsent patches for Inspect AI

This directory holds fixes for [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)
that have not been sent upstream. They are kept here so that they are not lost before they are.
Nothing in this repository applies them or depends on them.

| Patch | Fixes |
|---|---|
| [0001-eval-retry-runs-logs-concurrently-under-max-tasks.patch](0001-eval-retry-runs-logs-concurrently-under-max-tasks.patch) | A retry of several logs from one eval runs them one after another |
| [0001-eval-retry-reruns-a-sample-whose-input-changed.patch](0001-eval-retry-reruns-a-sample-whose-input-changed.patch) | A retry reuses completed samples whose input has changed |

The two are independent fixes, so each would be its own pull request. Each is a single patch
against upstream commit `59c134074` (Inspect AI 0.3.290 plus seven commits).

## eval-retry runs the logs of one eval one after another

### The problem

`inspect eval-retry` (and `eval_retry()`) given several logs runs them one after another, even
when they came from one eval that ran its tasks concurrently under `max_tasks`. A two-arm run of
this package's generative workflow (the same task with two task-arg values, such as `thinking=`
on and off, run under `max_tasks=2`) therefore resumes at half the concurrency it started with,
and loses the batching its arms shared on the model server.

The cause is in `src/inspect_ai/_eval/eval.py`: `eval_retry_async` calls `eval_async` once per
log and awaits each in turn, and `eval_async` refuses concurrent calls.

### The fix

`eval_async` already accepts several `PreviousTask`s, each with its own task args, as
`eval-set` uses it. The patch groups the logs to retry by their original `run_id` and their
eval-level settings (model as the log records it, generate config, limits, `max_tasks`, and so
on) and runs one `eval_async` call per group. Logs of separate evals, or with different
settings, still run one after another, as before. Results come back in the order the logs were
given. A paragraph in Inspect's `docs/_errors_and_retries.md` describes the behaviour.

### Evidence

- The new test `test_eval_retry_runs_several_logs_concurrently_under_max_tasks` in
  `tests/test_retry.py` fails without the change (the retried tasks do not overlap in time) and
  passes with it.
- `test_eval_retry_runs_logs_with_different_models_one_after_another` checks that the logs of a
  multi-model eval keep their own models.
- Every Inspect test file that mentions `eval_retry` passes (2127 tests), including the trio
  variants. `ruff check`, `ruff format --check` and `mypy` are clean on the changed files.

### Not fixed here

The same loop in `eval_retry_async` reassigns its own parameters
(`max_tasks = max_tasks or <log>.eval.config.max_tasks`, and the same for `max_samples`,
`log_format` and others), so the second log inherits the first log's value instead of its own.
That is a separate defect and would be a separate pull request.

## eval-retry reuses samples whose input changed

### The problem

A task can take an input from outside its arguments. The open-weights example on this
repository's `demo/open-weights` branch does: it reads the path of its XSTest CSV from the
environment variable `OPEN_WEIGHTS_XSTEST_CSV`. Inspect records a task's arguments in the log, but not such an
input. `inspect eval-retry` rebuilds the task from the arguments, so it reads whatever file the
variable names at retry time. It then reuses each completed sample of the earlier log by sample
id, and checks only that the dataset has the same number of samples. If the file changed, the
retried log mixes results from two different inputs, and no warning is given.

### The fix

The change is in `src/inspect_ai/_eval/task/run.py`. Before a retry reuses a completed sample,
it compares that sample's input, target and choices in the earlier log with the sample of the
same id in the rebuilt dataset. If they differ, the sample is run again instead of being reused
(its checkpoints are dropped), and one warning names the first such sample. Chat inputs are
compared by role and text only, because message ids are assigned at run time and images may be
condensed or left out of the log. The log schema is unchanged.

### Evidence

- The new test `test_eval_retry_reruns_a_sample_whose_input_changed` in `tests/test_retry.py`,
  run with explicit and with positional sample ids, builds a task from a file named by an
  environment variable, runs it so that one sample fails, points the variable at a file with a
  different first prompt, and retries. Without the change both runs fail: the retried log keeps
  the first prompt from the old file next to the second prompt from the new one. With it, the
  first sample runs again on its new prompt and the warning is logged.
- With the change, `tests/test_retry.py` passes (25 tests), and so do the retry, eval-set,
  checkpoint and shard test files (1893 tests). `ruff check`, `ruff format --check` and `mypy`
  are clean on the changed files.

### Limits

- Only samples that a retry would reuse are compared. A sample that had not finished runs on
  the rebuilt input, which the retried log records with it.
- Samples without an explicit id are compared under the ids Inspect assigns by position.
- Images and other non-text content of chat inputs are not compared.
- Sample metadata, sandbox and setup are not compared.

## Applying them

Inspect's repository requires a DCO `Signed-off-by` on every commit. Each patch carries a
placeholder author and no sign-off, so whoever applies it signs it as themselves. One branch per
patch:

```bash
git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
cd inspect_ai
git switch -c <branch> 59c134074
git am --signoff <this-directory>/<patch>
git commit --amend --reset-author --no-edit
```

On a newer upstream commit, `git am -3` resolves small drifts. Rerun the tests named above after
any conflict.

Both patches add their tests at the end of `tests/test_retry.py`, so applying the second on top
of the first conflicts there. Keeping both sides resolves it. With both applied,
`tests/test_retry.py` passes (27 tests, one slow test skipped).

## Sending them

Inspect takes a pull request only for an issue that a maintainer has accepted. Opening the
issues and the pull requests upstream is the maintainer's step.
