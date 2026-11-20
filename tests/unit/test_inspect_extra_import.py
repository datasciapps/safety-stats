"""What reading Inspect logs says when an installed inspect extra fails to import.

tests/unit/test_inspect_extra.py covers an installation without the extra. Each case runs
in a fresh interpreter, since the import happens once per process.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_IMPORT = """
import sys

{setup}
try:
    import inspect_safety_stats.ingest.inspect_logs
except ImportError as error:
    print(type(error).__name__, "|", error)
else:
    print("imported")
"""

# A finder that makes importing one module fail as a broken installation does.
_BROKEN = """
class Broken:
    def find_spec(self, name, path=None, target=None):
        if name == {module!r}:
            raise {error}
        return None


sys.meta_path.insert(0, Broken())
"""


def _import_with(setup: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT.format(setup=setup)],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    ("module", "error", "message"),
    [
        # An installed Inspect whose own dependency lacks a module it imports.
        (
            "inspect_ai.analysis",
            "ModuleNotFoundError(\"No module named 'acp.helpers'\", name='acp.helpers')",
            "ModuleNotFoundError | No module named 'acp.helpers'",
        ),
        # A pyarrow built against another NumPy.
        (
            "inspect_ai.analysis",
            "ImportError('_ARRAY_API not found')",
            "ImportError | _ARRAY_API not found",
        ),
    ],
)
def test_a_broken_installation_keeps_its_own_message(
    module: str, error: str, message: str
) -> None:
    printed = _import_with(_BROKEN.format(module=module, error=error))
    assert printed == message
