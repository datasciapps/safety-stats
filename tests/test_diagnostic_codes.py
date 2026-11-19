"""Every diagnostic code is triggered by a case that checks its assumption, source and effect.

The cases live in ``tests/diagnostic_codes``, one module per kind of source; what each check
means is in ``tests/diagnostic_codes/harness.py``. A registry entry without a case fails
``test_every_registry_code_has_a_case``.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import tests.diagnostic_codes
from inspect_safety_stats.registry import CODES
from tests.diagnostic_codes.harness import Case

# Every module of tests/diagnostic_codes whose name ends in "_cases".
MODULES = tuple(
    importlib.import_module(f"tests.diagnostic_codes.{info.name}")
    for info in pkgutil.iter_modules(tests.diagnostic_codes.__path__)
    if info.name.endswith("_cases")
)


def _cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for module in MODULES:
        for case in module.CASES:
            assert case.code not in cases, f"{case.code} has two cases"
            cases[case.code] = case
    return cases


CASES = _cases()


def test_every_registry_code_has_a_case() -> None:
    # A new code needs a case before it lands; a case for a code the registry no longer
    # has must go with it.
    assert sorted(set(CODES) - set(CASES)) == [], "registry codes no case triggers"
    assert sorted(set(CASES) - set(CODES)) == [], "cases for codes the registry lacks"


@pytest.mark.parametrize("code", sorted(CASES))
def test_a_case_triggers_the_code_and_shows_its_source_and_effect(
    code: str, tmp_path: Path
) -> None:
    case, entry = CASES[code], CODES[code]
    triggered = case.trigger(tmp_path / "trigger")
    control = case.control(tmp_path / "control")

    # The code reports on what the trigger and the control differ in.
    found = triggered.with_code(code)
    emitted = [item.code for item in triggered.diagnostics]
    assert found, f"the trigger does not emit {code}: {emitted}"
    assert control.with_code(code) == [], f"the control emits {code} too"

    for diagnostic in found:
        assert diagnostic.severity in entry.severities
        written = diagnostic.model_dump(mode="json")
        assert (written["source"], written["effect"]) == (entry.source, entry.effect)

    # What the case shows the code does, not what the registry says it does: the two must
    # agree, so an entry that reclassifies a code without changing what it does fails here.
    assert case.shows == entry.effect, (
        f"the registry gives {code} the effect {entry.effect!r}; its case shows {case.shows!r}"
    )
    if case.shows == "invalid":
        assert (triggered.status, control.status != "invalid") == ("invalid", True)
    elif case.shows == "degraded":
        assert (triggered.status, control.status) == ("degraded", "valid")
    else:
        assert triggered.status == control.status

    if entry.assumptions and triggered.assumptions is not None:
        assert set(entry.assumptions) <= triggered.assumptions

    if case.effect is None:
        assert case.shows in ("invalid", "degraded"), (
            f"{code} is {case.shows}: its case must check what that does to the results"
        )
    else:
        case.effect(triggered, control)
