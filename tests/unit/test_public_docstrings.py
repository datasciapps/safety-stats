"""Every public name has a docstring of its own (#22), so ``help()`` never comes back empty.

Two kinds of docstring look present to ``inspect.getdoc`` without anyone having written
them: ``@dataclass`` sets a missing class docstring to the class signature, and a pydantic
model without one inherits ``BaseModel``'s. Neither counts here.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest
from pydantic import BaseModel

PUBLIC_MODULES = (
    "inspect_safety_stats",
    "inspect_safety_stats.stats",
    "inspect_safety_stats.generative",
)

# A str instance cannot carry a docstring of its own: help() on it describes str.
NOT_DOCUMENTABLE = {("inspect_safety_stats", "__version__")}


def _docstring_problem(name: str, value: object) -> str | None:
    """Why the value has no usable docstring of its own, or None when it has one."""

    if inspect.isclass(value):
        # A class's own docstring, never one inherited from a base class.
        doc = value.__dict__.get("__doc__")
        if doc and dataclasses.is_dataclass(value) and doc.startswith(f"{name}("):
            return "has only the signature @dataclass generates"
    else:
        doc = getattr(value, "__doc__", None)
    doc = inspect.cleandoc(doc) if doc else ""
    if not doc:
        return "has no docstring"
    summary = doc.splitlines()[0].rstrip()
    if not summary.endswith("."):
        return f"has a first line that does not end with a period: {summary!r}"
    return None


def _public_names() -> list[tuple[str, str]]:
    return [
        (module_name, name)
        for module_name in PUBLIC_MODULES
        for name in importlib.import_module(module_name).__all__
    ]


@pytest.mark.parametrize(("module_name", "name"), _public_names())
def test_every_public_name_resolves_and_has_a_docstring(module_name: str, name: str) -> None:
    module = importlib.import_module(module_name)
    value = getattr(module, name)  # an unresolvable name in __all__ fails here
    if (module_name, name) in NOT_DOCUMENTABLE:
        assert isinstance(value, str)
        return
    problem = _docstring_problem(name, value)
    assert problem is None, f"{module_name}.{name} {problem}"


def test_the_exemptions_name_only_public_names() -> None:
    assert NOT_DOCUMENTABLE <= set(_public_names())


def test_generated_and_inherited_docstrings_do_not_count() -> None:
    @dataclasses.dataclass
    class Undocumented:
        value: int

    class UndocumentedModel(BaseModel):
        value: int

    def no_period() -> None:
        """A summary that wraps
        onto a second line.
        """

    @dataclasses.dataclass
    class Documented:
        """A summary."""

        value: int

    assert inspect.getdoc(Undocumented)
    assert inspect.getdoc(UndocumentedModel)
    assert _docstring_problem("Undocumented", Undocumented) is not None
    assert _docstring_problem("UndocumentedModel", UndocumentedModel) is not None
    assert _docstring_problem("no_period", no_period) is not None
    assert _docstring_problem("Documented", Documented) is None
