#!/usr/bin/env bash
# Install a built wheel into a clean virtual environment and check what a user gets.
#
# The build job in .github/workflows/ci.yml runs this on every pull request, and
# .github/workflows/release.yml runs it on the distributions it publishes, so a release is
# checked exactly as every change was. Runs with the `python` on PATH.
#
# Usage: scripts/check_wheel.sh <wheel> [venv directory]
set -euo pipefail

WHEEL="${1:?usage: scripts/check_wheel.sh <wheel> [venv directory]}"
VENV="${2:-$(mktemp -d)/check}"

python -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet "$WHEEL"
# The wheel must carry both HTML report templates, not just the modules.
"$VENV/bin/python" -c "
import inspect_safety_stats
from inspect_safety_stats.report import visualization
from inspect_safety_stats.generative import visualization as generative
print('version', inspect_safety_stats.__version__)
assert visualization._TEMPLATE.is_file(), 'report template missing from wheel'
assert generative._TEMPLATE.is_file(), 'generative template missing from wheel'
"
# Without the inspect extra the core installs no Inspect AI, and reading Inspect logs
# fails with a message that names the extra.
"$VENV/bin/python" -c "
import importlib.util
assert importlib.util.find_spec('inspect_ai') is None, 'inspect-ai is a core dependency'
try:
    import inspect_safety_stats.ingest.inspect_logs
except ImportError as error:
    assert 'inspect-safety-stats[inspect]' in str(error), error
else:
    raise AssertionError('Inspect logs were readable without the inspect extra')
import inspect_safety_stats.ingest.rollout_files
"
# Reading Inspect logs must work from a clean install with the inspect extra, not only
# from a dev environment that happens to have inspect-ai's optional extras.
"$VENV/bin/python" -m pip install --quiet "${WHEEL}[inspect]"
"$VENV/bin/python" -c "
from inspect_ai.analysis import evals_df, samples_df
import inspect_safety_stats.ingest.inspect_logs
import inspect_safety_stats.generative.logs
import pyarrow
print('log ingestion importable, pyarrow', pyarrow.__version__)
"
"$VENV/bin/inspect-safety-stats" version
"$VENV/bin/inspect-safety-stats" plan zero-failure --threshold 0.01 --alpha 0.05
"$VENV/bin/inspect-safety-stats" generative --help > /dev/null
