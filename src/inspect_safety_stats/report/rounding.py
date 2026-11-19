"""Numbers rounded for display so that what a page says about a bound still holds.

A bound rounded to the nearest digit can read tighter than the analysis gives: an upper bound
of 0.000438 reads as 0.0% at one decimal of a percentage point. Every page therefore rounds an
upper bound up and a lower bound down, and shows a threshold that a bound is compared with
exactly, so that the shown bound and threshold compare as the analysis compared them.

Each value is taken as the shortest decimal that reads back as the same float (its ``repr``),
so that a value a plan states, such as 0.05, is exact, and so that two floats keep their order.
"""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Literal

Direction = Literal["up", "down"]

_ROUNDING = {"up": ROUND_CEILING, "down": ROUND_FLOOR}


def exact(value: float) -> Decimal:
    """``value`` as the shortest decimal that reads back as the same float."""

    return Decimal(repr(value))


def rounded(value: float, places: int, direction: Direction, *, percent: bool = False) -> Decimal:
    """``value``, times 100 with ``percent``, rounded to ``places`` decimals in ``direction``."""

    number = exact(value)
    if percent:
        number = number.scaleb(2)
    return number.quantize(Decimal(1).scaleb(-places), rounding=_ROUNDING[direction])


def bound_text(
    value: float | None,
    places: int,
    *,
    upper: bool,
    percent: bool = False,
    signed: bool = False,
) -> str:
    """A bound in fixed point: an upper bound rounded up, a lower bound down; "—" for None."""

    if value is None or math.isnan(value):
        return "—"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    number = rounded(value, places, "up" if upper else "down", percent=percent)
    sign = "+" if signed else ""
    return f"{number:{sign}.{places}f}"


def decimals(value: float) -> int:
    """The decimals that ``value`` needs to be shown exactly."""

    exponent = exact(value).normalize().as_tuple().exponent
    return max(0, -int(exponent))


def threshold_text(value: float, places: int) -> str:
    """A threshold in fixed point at ``places`` decimals, which must show it exactly."""

    if decimals(value) > places:
        raise ValueError(f"{value!r} needs more than {places} decimals to be shown exactly")
    return f"{exact(value):.{places}f}"
