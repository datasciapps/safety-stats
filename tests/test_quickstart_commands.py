"""The quickstart's offline commands run verbatim from the repository root (#23).

Every ```bash block of docs/quickstart.md is run unless it calls ``inspect eval``, which needs
provider credentials, or carries ``<!-- quickstart-test: needs live logs -->``, because it reads
the logs such a run writes. ``<!-- quickstart-test: exit N -->`` on the line before a block
declares the exit code the reader is told to expect; it is 0 otherwise. Each ``--output`` is
redirected into a temporary directory, so the page's commands run unchanged apart from where the
report lands.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from tests.cli_output import plain

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "docs" / "quickstart.md"
SIMPLEQA = ROOT / "examples" / "real_simpleqa"

_BLOCK = re.compile(
    r"(?:<!-- quickstart-test: (?P<marker>[^>]*?) -->\n)?```bash\n(?P<body>.*?)```", re.S
)
_EXIT = re.compile(r"exit (\d+)")


@dataclass(frozen=True)
class Block:
    line: int
    marker: str | None
    body: str

    @property
    def offline(self) -> bool:
        return "inspect eval" not in self.body and self.marker != "needs live logs"

    @property
    def expected_exit(self) -> int:
        if self.marker is None:
            return 0
        match = _EXIT.fullmatch(self.marker)
        if match is None:
            raise AssertionError(f"quickstart.md:{self.line}: unknown marker {self.marker!r}")
        return int(match.group(1))


def _blocks() -> list[Block]:
    text = QUICKSTART.read_text(encoding="utf-8")
    return [
        Block(text.count("\n", 0, match.start("body")) + 1, match["marker"], match["body"])
        for match in _BLOCK.finditer(text)
    ]


def _offline_blocks() -> list[Block]:
    return [block for block in _blocks() if block.offline]


def test_the_quickstart_has_offline_valid_and_invalid_runs() -> None:
    """Guards the parametrized test below against passing because it found nothing to run."""
    blocks = _blocks()
    assert any(block.marker == "needs live logs" for block in blocks)
    assert any("inspect eval" in block.body for block in blocks)
    for block in blocks:
        if block.marker not in (None, "needs live logs"):
            block.expected_exit  # noqa: B018 - raises on an unknown marker
    offline = [(block.body.split()[1], block.expected_exit) for block in _offline_blocks()]
    for command in ("validate", "analyze"):
        assert (command, 0) in offline
        assert (command, 3) in offline


@pytest.mark.parametrize("block", _offline_blocks(), ids=lambda block: f"line{block.line}")
def test_each_offline_quickstart_command_runs_verbatim(
    block: Block, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = shlex.split(block.body.replace("\\\n", " "))
    assert argv[0] == "inspect-safety-stats", (
        f"quickstart.md:{block.line} runs {argv[0]!r}; mark it or teach this test to run it"
    )
    args = argv[1:]
    output: Path | None = None
    if "--output" in args:
        index = args.index("--output") + 1
        output = tmp_path / args[index]
        args[index] = str(output)
    monkeypatch.chdir(ROOT)

    result = CliRunner().invoke(app, args)

    text = plain(result.output)
    assert result.exit_code == block.expected_exit, f"quickstart.md:{block.line}: {text}"
    integrity = "valid" if block.expected_exit == 0 else "invalid"
    assert f"Integrity: {integrity}" in text
    if output is not None:
        assert (output / "report.md").is_file()


def test_the_wrong_scorer_plan_differs_from_the_valid_plan_only_in_the_score_name() -> None:
    """Keeps the quickstart's invalid run about the scorer, not about drift between two copies."""

    def lines(name: str) -> list[str]:
        text = (SIMPLEQA / name).read_text(encoding="utf-8")
        return [line for line in text.splitlines() if not line.startswith("#")]

    valid, wrong = lines("safety-plan.yaml"), lines("wrong-scorer-plan.yaml")
    assert wrong[0] == "", "the header comment should be followed by one blank line"
    differing = [(a, b) for a, b in zip(valid, wrong[1:], strict=True) if a != b]
    assert differing == [("    name: model_graded_qa", "    name: match")]
