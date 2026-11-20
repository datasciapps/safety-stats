"""The interval assumption sentences: one entry per sentence, each used, each worded once."""

from __future__ import annotations

import re
import string
from pathlib import Path

import pytest

import inspect_safety_stats
from inspect_safety_stats.analysis import interval_assumptions
from inspect_safety_stats.analysis.interval_assumptions import REGISTRY, Assumption
from inspect_safety_stats.registry import ASSUMPTIONS

PACKAGE = Path(inspect_safety_stats.__file__).resolve().parent
MODULE = Path(interval_assumptions.__file__).resolve()

# Every Assumption the module defines, by the name code refers to it with.
DEFINED = {
    name: value
    for name, value in vars(interval_assumptions).items()
    if isinstance(value, Assumption)
}


def _fields(statement: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(statement) if field is not None}


def test_every_defined_sentence_is_registered_once() -> None:
    assert sorted(entry.id for entry in DEFINED.values()) == sorted(
        entry.id for entry in REGISTRY
    )
    assert len(REGISTRY) == len(set(REGISTRY))


def test_identifiers_are_unique_and_dotted() -> None:
    identifiers = [entry.id for entry in REGISTRY]
    assert len(identifiers) == len(set(identifiers))
    for identifier in identifiers:
        assert re.fullmatch(r"[a-z]+(\.[a-z0-9_]+)+", identifier), identifier


def test_every_sentence_states_an_assumption_of_the_registry() -> None:
    assert [entry.id for entry in REGISTRY if entry.assumption not in ASSUMPTIONS] == []


def test_every_sentence_is_used_outside_its_module() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PACKAGE.rglob("*.py")
        if path.resolve() != MODULE
    )
    unused = [name for name in DEFINED if not re.search(rf"\b{name}\b", sources)]
    assert unused == []


@pytest.mark.parametrize("entry", REGISTRY, ids=lambda entry: entry.id)
def test_statements_fill_exactly_their_fields(entry: Assumption) -> None:
    fields = _fields(entry.statement)
    rendered = entry.text(**{field: "value" for field in fields})
    assert "{" not in rendered and "}" not in rendered
    assert rendered.endswith(".")
    if fields:
        with pytest.raises(KeyError):
            entry.text()


def test_a_template_renders_as_the_report_has_always_worded_it() -> None:
    assert interval_assumptions.BLOCK_INDEPENDENCE.text(block_by=["prompt_id"]) == (
        "Blocks defined by ['prompt_id'] are mutually independent."
    )
    assert interval_assumptions.WITHIN_BLOCK_DEPENDENCE_COUPLED.text(coupling="joint") == (
        "Arbitrary dependence is allowed within each declared block, with cross-arm coupling "
        "set to 'joint'."
    )
