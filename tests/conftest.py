"""Configuration shared by every test."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True, scope="session")
def _render_cli_output_as_ci_does() -> Iterator[None]:
    """Make typer render errors and help the way it does on CI, wherever the suite runs.

    On GitHub Actions typer always forces terminal rendering, so its output is coloured, and
    colour inserts escape codes inside words. Under pytest elsewhere it is not, so a test
    asserting on the raw text passed locally and failed on CI. With CI's rendering forced
    here it fails locally too; assert on ``tests.cli_output.plain(result.output)`` instead.

    PY_COLORS forces typer's rendering and nothing else. FORCE_COLOR would also force every
    other rich console, which CI does not. COLUMNS is the width CI wraps to; under
    ``pytest -s`` the width of the developer's terminal would apply instead.

    ORDER MATTERS: typer reads PY_COLORS once, the first time anything renders an error or
    help, so this has to apply before the first test runs. Moving it into a narrower fixture
    or into a test has no effect once something has rendered.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PY_COLORS", "1")
        patch.setenv("COLUMNS", "80")
        patch.delenv("NO_COLOR", raising=False)
        yield
