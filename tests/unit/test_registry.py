"""The registry of diagnostic codes and assumptions, checked against the package source.

The check reads the source rather than running the analysis, so that a code reached only on a
rare path is held to the registry as surely as a common one. It walks every ``Diagnostic(...)``
call in the package and resolves the codes and severities it can emit:

- ``code=`` is a string literal, or an expression listed in ``TRACED_CODES``, which says where
  its values are written; the trace is followed in the source, and each value it reaches must
  be a literal too. An f-string, an unlisted name or a positional argument fails.
- ``severity=`` is a literal, or a conditional whose branches resolve, each counted.

A code is written only where a ``Diagnostic`` call can emit it: a code-like literal elsewhere,
such as a comparison with a diagnostic's code, must name a registered code but does not count
as written.
"""

from __future__ import annotations

import ast
import re
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

import inspect_safety_stats
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.registry import (
    ASSUMPTIONS,
    CODES,
    EFFECTS,
    SOURCE_KINDS,
    checked_by,
)

PACKAGE = Path(inspect_safety_stats.__file__).resolve().parent
REGISTRY = PACKAGE / "registry.py"
CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
SEVERITIES = ("ERROR", "WARNING", "INFO")


@dataclass(frozen=True)
class Trace:
    """Where the values of a ``code=`` that is not a literal are written, in its own file.

    With ``calls``, the expression is the parameter ``code`` of the function or class
    ``calls``: as a name, the ``Diagnostic`` call is inside that function; as ``<name>.code``,
    it is inside ``except <calls> as <name>``. The codes are argument ``position`` of every
    call to ``calls``, and the parameter there must be named ``code``. Without ``calls``, the
    expression is the loop variable at ``position`` of the enclosing ``for`` statement, and the
    codes are the elements at ``position`` of the tuples it iterates over.
    """

    position: int
    calls: str | None = None


# Every Diagnostic call whose code= is not a literal, by file and expression.
TRACED_CODES = {
    ("analysis/anytime.py", "refusal.code"): Trace(position=0, calls="_Refused"),
    ("analysis/judge_calibration.py", "code"): Trace(position=0),
    ("ingest/normalize.py", "code"): Trace(position=0, calls="_unmapped_summary"),
}


@dataclass(frozen=True)
class Site:
    """One ``Diagnostic(...)`` call; ``codes`` or ``severities`` is None where unresolved."""

    where: str
    codes: frozenset[str] | None
    severities: frozenset[str] | None
    traced: tuple[str, str] | None
    node: ast.Call


def _literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _severities(node: ast.expr | None) -> frozenset[str] | None:
    """The severities a ``severity=`` can take: a literal, or both branches of a conditional."""

    value = _literal(node)
    if value is not None:
        return frozenset({value})
    if isinstance(node, ast.IfExp):
        body, orelse = _severities(node.body), _severities(node.orelse)
        return None if body is None or orelse is None else body | orelse
    return None


def _enclosing(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> typing.Iterator[ast.AST]:
    while node in parents:
        node = parents[node]
        yield node


def _parameters(tree: ast.Module, name: str) -> list[str] | None:
    """The parameters of the function ``name``, or of the class ``name``'s ``__init__``."""

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [item.arg for item in node.args.args]
        if isinstance(node, ast.ClassDef) and node.name == name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return [argument.arg for argument in item.args.args[1:]]
    return None


def _call_codes(tree: ast.Module, trace: Trace) -> frozenset[str] | None:
    """Argument ``position`` of every call to ``trace.calls``, each a literal, or None."""

    assert trace.calls is not None
    parameters = _parameters(tree, trace.calls)
    if parameters is None or parameters[trace.position : trace.position + 1] != ["code"]:
        return None
    codes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == trace.calls:
            keywords = {item.arg: item.value for item in node.keywords}
            argument = (
                node.args[trace.position]
                if len(node.args) > trace.position
                else keywords.get("code")
            )
            value = _literal(argument)
            if value is None or any(isinstance(item, ast.Starred) for item in node.args):
                return None
            codes.add(value)
    return frozenset(codes) or None


def _loop_codes(
    call: ast.Call, expression: ast.expr, trace: Trace, parents: dict[ast.AST, ast.AST]
) -> frozenset[str] | None:
    """The literal elements at ``position`` of the tuples the enclosing loop goes over."""

    if not isinstance(expression, ast.Name):
        return None
    for node in _enclosing(call, parents):
        if not isinstance(node, ast.For):
            continue
        target = node.target
        if not (
            isinstance(target, ast.Tuple)
            and len(target.elts) > trace.position
            and isinstance(target.elts[trace.position], ast.Name)
            and typing.cast(ast.Name, target.elts[trace.position]).id == expression.id
        ):
            continue
        if not isinstance(node.iter, (ast.Tuple, ast.List)):
            return None
        codes: set[str] = set()
        for item in node.iter.elts:
            if not (isinstance(item, ast.Tuple) and len(item.elts) > trace.position):
                return None
            value = _literal(item.elts[trace.position])
            if value is None:
                return None
            codes.add(value)
        return frozenset(codes) or None
    return None


def _traced_codes(
    tree: ast.Module, call: ast.Call, expression: ast.expr, trace: Trace,
    parents: dict[ast.AST, ast.AST],
) -> frozenset[str] | None:
    if trace.calls is None:
        return _loop_codes(call, expression, trace, parents)
    if isinstance(expression, ast.Name) and expression.id == "code":
        inside = any(
            isinstance(node, ast.FunctionDef) and node.name == trace.calls
            for node in _enclosing(call, parents)
        )
    elif (
        isinstance(expression, ast.Attribute)
        and expression.attr == "code"
        and isinstance(expression.value, ast.Name)
    ):
        bound = expression.value.id
        inside = any(
            isinstance(node, ast.ExceptHandler)
            and node.name == bound
            and node.type is not None
            and ast.unparse(node.type) == trace.calls
            for node in _enclosing(call, parents)
        )
    else:
        inside = False
    return _call_codes(tree, trace) if inside else None


def _is_diagnostic(function: ast.expr) -> bool:
    return (isinstance(function, ast.Name) and function.id == "Diagnostic") or (
        isinstance(function, ast.Attribute) and function.attr == "Diagnostic"
    )


def _sites_in(
    tree: ast.Module, name: str, traced: typing.Mapping[tuple[str, str], Trace] = TRACED_CODES
) -> list[Site]:
    """Every ``Diagnostic(...)`` call in one module, ``name`` relative to the package."""

    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    sites: list[Site] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_diagnostic(node.func)):
            continue
        keywords = {item.arg: item.value for item in node.keywords}
        code, severity = keywords.get("code"), keywords.get("severity")
        key = None if code is None else (name, ast.unparse(code))
        codes: frozenset[str] | None = None
        if node.args or None in keywords:
            # A positional or unpacked argument hides the code and the severity.
            severity = None
        elif _literal(code) is not None:
            codes = frozenset({typing.cast(str, _literal(code))})
            key = None
        elif code is not None and key in traced:
            codes = _traced_codes(tree, node, code, traced[key], parents)
        sites.append(
            Site(
                where=f"{name}:{node.lineno}",
                codes=codes,
                severities=_severities(severity),
                traced=key if key in traced else None,
                node=node,
            )
        )
    return sites


def _modules() -> list[tuple[ast.Module, str]]:
    return [
        (
            ast.parse(path.read_text(encoding="utf-8"), str(path)),
            path.relative_to(PACKAGE).as_posix(),
        )
        for path in sorted(PACKAGE.rglob("*.py"))
    ]


def _diagnostic_sites() -> list[Site]:
    return [site for tree, name in _modules() for site in _sites_in(tree, name)]


def _all_lists(tree: ast.Module) -> set[int]:
    """The ids of the string nodes that name exports in ``__all__``."""

    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            ids.update(id(item) for item in ast.walk(node.value))
    return ids


def _source_codes() -> dict[str, list[str]]:
    """Every code-like string literal in the package, with where it is written."""

    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if path == REGISTRY:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        exports = _all_lists(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and CODE_PATTERN.fullmatch(node.value)
                and node.value not in SEVERITIES
                and id(node) not in exports
            ):
                where = f"{path.relative_to(PACKAGE)}:{node.lineno}"
                found.setdefault(node.value, []).append(where)
    return found


DEGRADED_DETAIL = "integrity_degraded"


def _degraded_details() -> tuple[dict[str, list[bool]], list[str]]:
    """Which ``Diagnostic(code=...)`` sites write ``integrity_degraded: True`` in ``details``.

    The first value maps each code a site emits to one flag per site: whether the site's
    ``details`` is a dict literal with that key set to True. The second lists every other
    place in the package that writes the key, which the flags would miss.
    """

    marks: dict[str, list[bool]] = {}
    stray: list[str] = []
    for tree, name in _modules():
        counted: set[int] = set()
        for site in _sites_in(tree, name):
            keywords = {item.arg: item.value for item in site.node.keywords}
            details = keywords.get("details")
            marked = False
            if isinstance(details, ast.Dict):
                for key, value in zip(details.keys, details.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == DEGRADED_DETAIL:
                        counted.add(id(key))
                        marked = isinstance(value, ast.Constant) and value.value is True
            if site.codes is not None:
                for code in site.codes:
                    marks.setdefault(code, []).append(marked)
            elif marked:
                stray.append(site.where)
        stray.extend(
            f"{name}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and node.value == DEGRADED_DETAIL
            and id(node) not in counted
        )
    return marks, stray


def test_the_codes_whose_sites_mark_integrity_degraded_are_the_degraded_ones() -> None:
    # Integrity reads the registry's effect (analysis.inputs._integrity_status); the detail is
    # what the emitting site says of its own finding. An entry whose effect disagrees with its
    # sites, or a site that marks a code the registry does not degrade, fails here.
    marks, stray = _degraded_details()
    degraded = sorted(code for code, entry in CODES.items() if entry.effect == "degraded")
    assert stray == [], "integrity_degraded written outside a Diagnostic(code=...) literal"
    assert sorted(code for code, sites in marks.items() if any(sites)) == degraded
    assert sorted(code for code in degraded if not all(marks.get(code, [False]))) == []


def test_every_diagnostic_site_resolves_its_codes_and_severities() -> None:
    # Write code= as a literal, or list the expression in TRACED_CODES; write severity= as a
    # literal or a conditional of literals.
    unresolved = [
        site.where for site in _diagnostic_sites() if site.codes is None or not site.severities
    ]
    assert unresolved == []


def test_every_traced_code_is_a_diagnostic_site_s() -> None:
    used = {site.traced for site in _diagnostic_sites() if site.traced is not None}
    assert sorted(set(TRACED_CODES) - used) == []


def test_diagnostic_is_not_imported_under_another_name() -> None:
    renamed = [
        f"{name}:{node.lineno}"
        for tree, name in _modules()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name == "Diagnostic" and alias.asname not in (None, "Diagnostic")
    ]
    assert renamed == []


def test_every_code_a_diagnostic_site_emits_has_a_registry_entry() -> None:
    missing = sorted(
        f"{code} at {site.where}"
        for site in _diagnostic_sites()
        for code in site.codes or ()
        if code not in CODES
    )
    assert missing == []


def test_every_code_like_literal_names_a_registered_code() -> None:
    # Catches a misspelt code compared with a diagnostic's, which no site emits.
    missing = {code: sites for code, sites in _source_codes().items() if code not in CODES}
    assert missing == {}


def test_every_registry_entry_names_a_code_a_diagnostic_site_emits() -> None:
    emitted = {code for site in _diagnostic_sites() for code in site.codes or ()}
    assert sorted(set(CODES) - emitted) == []


def test_entries_list_every_severity_their_code_is_emitted_with() -> None:
    wrong = sorted(
        f"{code} {severity} at {site.where}"
        for site in _diagnostic_sites()
        for code in site.codes or ()
        for severity in site.severities or ()
        if code in CODES and severity not in CODES[code].severities
    )
    assert wrong == []


def _sites(source: str, traced: typing.Mapping[tuple[str, str], Trace] | None = None) -> list[Site]:
    return _sites_in(ast.parse(source), "module.py", traced or {})


@pytest.mark.parametrize(
    ("source", "codes", "severities"),
    [
        # A code of one word, which a pattern of underscored words would miss.
        ('Diagnostic(code="TIMEOUT", severity="ERROR", message="m")', {"TIMEOUT"}, {"ERROR"}),
        # Both branches of a conditional severity, and of a nested one.
        (
            'Diagnostic(code="A_B", severity="WARNING" if x else "INFO", message="m")',
            {"A_B"},
            {"WARNING", "INFO"},
        ),
        (
            'Diagnostic(code="A_B", severity="INFO" if x else "ERROR" if y else "WARNING")',
            {"A_B"},
            {"INFO", "ERROR", "WARNING"},
        ),
        ('schema.Diagnostic(code="A_B", severity="INFO")', {"A_B"}, {"INFO"}),
    ],
)
def test_the_check_resolves_literal_codes_and_conditional_severities(
    source: str, codes: set[str], severities: set[str]
) -> None:
    (site,) = _sites(source)
    assert (site.codes, site.severities) == (codes, severities)


@pytest.mark.parametrize(
    "source",
    [
        'Diagnostic(code=f"JUDGE_{kind}", severity="ERROR", message="m")',
        'Diagnostic(code=name, severity="ERROR", message="m")',
        'Diagnostic(code=CODES["A_B"].code, severity="ERROR", message="m")',
        'Diagnostic(**fields)',
        'Diagnostic("A_B", "ERROR", "m")',
        'Diagnostic(code="A_B", severity=level, message="m")',
        'Diagnostic(code="A_B", severity="ERROR" if x else level, message="m")',
    ],
)
def test_the_check_refuses_a_site_it_cannot_resolve(source: str) -> None:
    (site,) = _sites(source)
    assert site.codes is None or not site.severities


TRACED_FUNCTION = """
def _summary(code, scope):
    return Diagnostic(code=code, severity="ERROR", message="m", scope=scope)

_summary("A_B", "s")
_summary(code="C_D", scope="s")
"""

TRACED_REFUSAL = """
class _Refused(ValueError):
    def __init__(self, code, message):
        self.code = code

def bound():
    raise _Refused("A_B", "m")

try:
    bound()
except _Refused as refusal:
    Diagnostic(code=refusal.code, severity="ERROR", message="m")
"""

TRACED_LOOP = """
for code, what in (("A_B", "x"), ("C_D", "y")):
    Diagnostic(code=code, severity="ERROR", message=what)
"""


@pytest.mark.parametrize(
    ("source", "expression", "trace", "codes"),
    [
        (TRACED_FUNCTION, "code", Trace(position=0, calls="_summary"), {"A_B", "C_D"}),
        (TRACED_REFUSAL, "refusal.code", Trace(position=0, calls="_Refused"), {"A_B"}),
        (TRACED_LOOP, "code", Trace(position=0), {"A_B", "C_D"}),
    ],
)
def test_a_traced_code_reaches_the_literals_it_is_written_from(
    source: str, expression: str, trace: Trace, codes: set[str]
) -> None:
    (site,) = _sites(source, {("module.py", expression): trace})
    assert site.codes == codes


@pytest.mark.parametrize(
    ("source", "expression", "trace"),
    [
        # The trace reaches a value that is not a literal.
        (TRACED_FUNCTION + "_summary(other, 's')\n", "code", Trace(position=0, calls="_summary")),
        (
            TRACED_REFUSAL.replace('_Refused("A_B", "m")', '_Refused(code_of(x), "m")'),
            "refusal.code",
            Trace(position=0, calls="_Refused"),
        ),
        (TRACED_LOOP.replace('("C_D", "y")', '(name, "y")'), "code", Trace(position=0)),
        # The trace names the wrong argument, or a function the site is not in.
        (TRACED_FUNCTION, "code", Trace(position=1, calls="_summary")),
        (TRACED_REFUSAL, "refusal.code", Trace(position=0, calls="_summary")),
        (TRACED_LOOP, "code", Trace(position=1)),
    ],
)
def test_a_trace_that_reaches_no_literal_resolves_nothing(
    source: str, expression: str, trace: Trace
) -> None:
    (site,) = _sites(source, {("module.py", expression): trace})
    assert site.codes is None


@pytest.mark.parametrize("code", sorted(CODES))
def test_code_entries_are_complete_and_consistent(code: str) -> None:
    entry = CODES[code]
    assert entry.code == code
    assert entry.severities, "an entry lists at least one severity"
    assert set(entry.severities) <= set(SEVERITIES)
    assert list(entry.severities) == sorted(set(entry.severities), key=SEVERITIES.index)
    assert entry.source in SOURCE_KINDS
    assert entry.effect in EFFECTS
    # Any ERROR makes the integrity status invalid (analysis.inputs._integrity_status).
    assert (entry.effect == "invalid") == ("ERROR" in entry.severities)
    for text in (entry.assumption, entry.consequence, entry.remedy):
        assert text.strip() == text and text.endswith("."), text
    assert set(entry.assumptions) <= set(ASSUMPTIONS)


def test_the_registry_lists_its_codes_in_order_once() -> None:
    from inspect_safety_stats import registry

    codes = [entry.code for entry in registry._CODES]
    assert codes == sorted(set(codes))


def _field(model: type[BaseModel], path: str) -> typing.Any:
    """The pydantic field a dotted plan path names; ``[]`` and ``<id>`` step into items."""

    field = None
    current: typing.Any = model
    for part in path.split("."):
        if part == "<id>":
            current = _item_type(current, dict)
            continue
        name = part.removesuffix("[]")
        assert isinstance(current, type) and issubclass(current, BaseModel), path
        field = current.model_fields[name]
        current = field.annotation
        if part.endswith("[]"):
            current = _item_type(current, list)
        else:
            current = _model_in(current)
    return field


def _model_in(annotation: typing.Any) -> typing.Any:
    """The model of an optional or plain annotation, or the annotation itself."""

    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        models = [
            item
            for item in typing.get_args(annotation)
            if isinstance(item, type) and issubclass(item, BaseModel)
        ]
        return models[0] if len(models) == 1 else annotation
    return annotation


def _item_type(annotation: typing.Any, container: type) -> typing.Any:
    assert typing.get_origin(annotation) is container, annotation
    return _model_in(typing.get_args(annotation)[-1])


def _as_plan_text(value: typing.Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


@pytest.mark.parametrize("assumption", sorted(ASSUMPTIONS))
def test_assumption_entries_name_real_plan_fields_and_their_defaults(assumption: str) -> None:
    entry = ASSUMPTIONS[assumption]
    assert entry.id == assumption
    assert entry.source in SOURCE_KINDS
    for text in (entry.statement, entry.if_wrong, entry.remedy):
        assert text.strip() == text and text.endswith("."), text
    if entry.plan_field is None:
        assert entry.default is None
        return
    field = _field(AnalysisPlan, entry.plan_field)
    if field.is_required():
        assert entry.default is None
    else:
        assert entry.default == _as_plan_text(field.get_default(call_default_factory=True))


def test_checked_by_reads_the_code_entries() -> None:
    for assumption in ASSUMPTIONS:
        codes = checked_by(assumption)
        assert list(codes) == sorted(codes)
        assert all(assumption in CODES[code].assumptions for code in codes)
    with pytest.raises(KeyError):
        checked_by("no_such_assumption")
