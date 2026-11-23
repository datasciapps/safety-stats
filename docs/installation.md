# Installation

## Requirements

- Python 3.11 or newer.
- Linux, macOS, or Windows. CI runs the test suite on all three.
- For Inspect AI logs: Inspect AI 0.3.289 or a later 0.3 release, installed by the `inspect`
  extra, and completed Inspect evaluation logs. Logs written by Inspect AI 0.3.259 or later are
  read.
- For results from another evaluation system: [rollout files](rollout-files.md), which need no
  extra.

The analyzer is offline. It does not call model providers or make network requests.

## Install from PyPI

```bash
python -m pip install 'safety-stats[inspect]'
```

The `inspect` extra installs Inspect AI and pyarrow, which reading Inspect logs needs. Leave it
out to analyse only rollout files: `python -m pip install safety-stats`. The examples
and the offline demo are not part of the package: run them from a checkout.

## Install from a checkout

```bash
git clone https://github.com/aduncan001/safety_stats.git
cd safety_stats
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[inspect]'
```

The `inspect` extra installs Inspect AI and pyarrow, which reading Inspect logs needs. Leave it
out to analyse only rollout files: `python -m pip install -e .`. Reading Inspect logs without it
fails with an error that names the extra. The `dev` extra includes it.

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Development installation

```bash
python -m pip install -e '.[dev]'
pytest
```

Install the local machine-translation dependencies used by the multilingual JailbreakBench example
with:

```bash
python -m pip install -e '.[translation]'
```

This extra installs PyTorch, Transformers, and SentencePiece. Translation is a separate preprocessing
step that may download the selected checkpoint; the statistical analyzer remains offline.

Run static checks when the optional tools are installed:

```bash
ruff check src tests examples research scripts
mypy src/safety_stats
pytest --cov=safety_stats
```

## Confirm installation

```bash
safety-stats version
safety-stats plan zero-failure --threshold 0.01 --alpha 0.05
```

The planner should report 299 rollouts.

## Supported input storage

Any log location supported by Inspect's dataframe readers can be passed through the Python API. The CLI accepts local files and directories in v1. Local source logs receive SHA-256 checksums; remote sources retain their URI and report that a local content checksum was unavailable.

## Dependency policy

Runtime dependencies are deliberately small:

- Inspect AI for supported log ingestion;
- Pandas and NumPy for normalized tables;
- SciPy for exact distributions;
- Pydantic and PyYAML for strict plans;
- Typer for the CLI.

No model SDK is called by the analyzer itself.

The optional `translation` extra is used only by the JailbreakBench translation script. It is not
imported by the package or required to analyze an already completed multilingual run.

## Troubleshooting

### `analyze` returns exit code 3

Either the analysis could not run, in which case the error is printed, or it completed but its
integrity is invalid. In the second case the report is still written, and its diagnostics (which
`validate` also prints) say what invalidated it. Pass `--allow-invalid` only when a script needs
the earlier behaviour of exiting `0`.

### The CLI returns exit code 2

Decision-sensitive exit codes were explicitly enabled, and the optional threshold analysis was
inconclusive. Omit `--decision-exit-codes` to make every successfully completed analysis return
zero.

### On Windows, logs cannot be found

If `analyze` or `validate` fails with `FileNotFoundError` on a path that has no drive letter,
for example `\Users\...\logs\....eval`, the logs are on a different drive from the
directory the command runs in. inspect-ai drops the drive letter when it reads local logs on
Windows, so it looks for them on the current drive. Run the command from the drive that holds
the logs, or keep the logs on the same drive. This is tracked in
[issue #35](https://github.com/aduncan001/safety_stats/issues/35).

### The selected score is absent

Check the scorer name in the Inspect log. The plan field `inputs.score.name: harmful_behavior` expects a sample dataframe column named `score_harmful_behavior`.

### The model cohort is empty

The cohort's `match` fields are exact matches against evaluation-level Inspect dataframe fields. Inspect the model identifier in the log rather than assuming a display name.

### Every prompt reports a rollout shortfall

The manifest's `expected_rollouts` must match the number of terminal sample epochs per prompt and cohort. Retries do not count as additional rollouts.

### The plan is reported as post hoc

For a confirmatory analysis, `analysis.plan_frozen_at` must precede the earliest selected evaluation. A registration hash is recommended for stronger provenance. `PLAN_RECEIPT_POSTHOC` means the same for a [registration receipt](plan-reference.md#registration-receipts-experimental): it was written after generation started.
