"""Helpers for asserting on what the command line prints."""

from __future__ import annotations

import re

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_BOX_DRAWING = re.compile("[─-╿]")


def plain(output: str) -> str:
    """CLI output as plain words.

    Typer prints errors and help through rich, which colours them, frames errors in a box and
    wraps them to the terminal width. Colour inserts escape codes inside words: ``--weights``
    renders as ``-`` and ``-weights``. Assert on ``plain(result.output)``, not on how one
    terminal laid the words out. tests/conftest.py makes every run render the way CI does,
    so an assertion that skips this fails locally, not only on CI.
    """
    return " ".join(_BOX_DRAWING.sub(" ", _ANSI.sub("", output)).split())
