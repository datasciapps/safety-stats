"""The report template's JavaScript functions, cut out so that tests can run them under Node."""

from __future__ import annotations


def function_source(template: str, name: str) -> str:
    start = template.index(f"function {name}(")
    depth = 0
    for index in range(template.index("{", start), len(template)):
        if template[index] == "{":
            depth += 1
        elif template[index] == "}":
            depth -= 1
            if depth == 0:
                return template[start : index + 1]
    raise AssertionError(f"unbalanced braces in {name}")
