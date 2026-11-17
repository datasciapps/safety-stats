"""The storyboard's chart cards, as SVG, from one analysis report folder.

Reads only the content-free files `analyze` writes, as `storyboard_numbers.py` does, and writes
three cards, each in a light and a dark version:

- `single-runs`: the score each epoch would have reported as a single run, beside the study's
  interval (card 1:10);
- `asking-again`: the chance of at least one failure in k attempts, k from 1 to 20, estimated
  without bias, with its interval for each k (card 1:45);
- `two-sides`: failures against refusals of safe prompts, one box per cohort that holds both
  intervals (card 3:40).

    python chart_cards.py REPORT_DIR --output-dir DIR [--cohort thinking_on]

The cards are hand-written SVG, 1600 by 900, so the example needs no plotting library. Text sizes
are set for a projected screencast. Every text is checked, with an estimate of its width, against
the card's edges, every other text, and every mark it must not cover (dots, lines, box edges); a
label tries a few places and takes the first that is clear. When none is, nothing is written.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from storyboard_numbers import (
    ALPHA,
    TASK_GROUPS,
    prompt_counts,
    prompt_weights,
    single_runs,
    thresholds,
    unbiased_worst_of_k,
)

WIDTH, HEIGHT = 1600, 900
MARGIN = 96
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
# Average advance of a character in the sans above, as a share of the font size. Generous, so
# that the collision check errs towards finding one.
CHARACTER_WIDTH = 0.56
LONGEST_K = 20
HARM, REFUSAL = "harm", "over-refusal"

# The validated default categorical palette. Its first three slots hold apart for every pair
# under colour-vision deficiency, light and dark, so the two-sides card takes at most three.
THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70"),
    },
}

Box = tuple[float, float, float, float]
Point = tuple[float, float]
# Where a label goes: x, y of its baseline, and its anchor.
Spot = tuple[float, float, str]


@dataclass
class Report:
    """What the cards read from one report folder."""

    analysis: dict[str, Any]
    risks: dict[tuple[str, str], dict[str, Any]]
    limits: dict[tuple[str, str], float]
    runs: pd.DataFrame
    counts: pd.DataFrame
    prompts: dict[str, int]

    @classmethod
    def read(cls, report_dir: Path) -> Report:
        report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
        plan = yaml.safe_load((report_dir / "plan.resolved.yaml").read_text(encoding="utf-8"))
        observations = pd.read_csv(
            report_dir / "observations.csv", dtype={"prompt_id": str}, low_memory=False
        )
        design = pd.read_csv(report_dir / "design.resolved.csv", dtype={"prompt_id": str})
        weights = prompt_weights(design)
        runs = single_runs(observations, weights)
        counts = prompt_counts(observations, weights)
        return cls(
            analysis=report["analysis"],
            risks={
                (g["cohort_id"], g["group_id"]): g
                for g in report["group_risks"]
                if g["measure_id"] == "mean"
            },
            limits=thresholds(plan),
            runs=runs.assign(group=runs["task_id"].map(TASK_GROUPS)),
            counts=counts.assign(group=counts["task_id"].map(TASK_GROUPS)),
            prompts={
                TASK_GROUPS[str(task)]: int(rows["prompt_id"].nunique())
                for task, rows in observations.groupby("task_id")
            },
        )

    def cohorts(self) -> list[str]:
        return sorted({cohort for cohort, _ in self.risks})

    def footer(self) -> str:
        frozen = str(self.analysis["plan_frozen_at"])[:10]
        return (
            f"Plan {self.analysis['plan_sha256'][:12]}, frozen {frozen}; "
            f"analysis {self.analysis['id']}."
        )


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@dataclass
class Card:
    """One 16:9 card: its SVG elements, its texts' boxes, and the marks texts must not cover."""

    theme: dict[str, Any]
    label: str
    parts: list[str] = field(default_factory=list)
    texts: list[tuple[str, Box]] = field(default_factory=list)
    marks: list[Box] = field(default_factory=list)

    @staticmethod
    def extent(content: str, x: float, y: float, size: int, anchor: str) -> Box:
        width = CHARACTER_WIDTH * size * len(content)
        left = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
        return left, y - 0.8 * size, left + width, y + 0.25 * size

    def clear(self, box: Box) -> bool:
        inside = box[0] >= 0 and box[1] >= 0 and box[2] <= WIDTH and box[3] <= HEIGHT
        return inside and not any(
            _overlap(box, other) for other in [t for _, t in self.texts] + self.marks
        )

    def text(
        self,
        content: str,
        x: float,
        y: float,
        *,
        size: int,
        ink: str = "secondary",
        anchor: str = "start",
        weight: int = 400,
    ) -> None:
        self.texts.append((content, self.extent(content, x, y, size, anchor)))
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{self.theme[ink]}" text-anchor="{anchor}">{escape(content)}</text>'
        )

    def place(self, content: str, spots: Sequence[Spot], *, size: int, ink: str) -> None:
        """Put a label at the first of `spots` (x, y, anchor) that is clear, else the first."""

        x, y, anchor = next(
            (s for s in spots if self.clear(self.extent(content, *s[:2], size, s[2]))), spots[0]
        )
        self.text(content, x, y, size=size, ink=ink, anchor=anchor)

    def keep_clear(self, box: Box) -> None:
        self.marks.append(box)

    def keep_line_clear(self, points: Sequence[Point], width: float = 3) -> None:
        """Mark a polyline, as small squares every few pixels along each segment."""

        for (x1, y1), (x2, y2) in zip(points, points[1:], strict=False):
            steps = max(1, math.ceil(math.hypot(x2 - x1, y2 - y1) / 3))
            for step in range(steps + 1):
                x = x1 + (x2 - x1) * step / steps
                y = y1 + (y2 - y1) * step / steps
                self.marks.append((x - width / 2, y - width / 2, x + width / 2, y + width / 2))

    def heading(self, title: str, subtitle: Sequence[str], footer: str) -> None:
        self.text(title, MARGIN, 92, size=46, ink="primary", weight=600)
        for line, text in enumerate(subtitle):
            self.text(text, MARGIN, 142 + 36 * line, size=26)
        self.text(footer, MARGIN, HEIGHT - 36, size=20, ink="muted")

    def add(self, element: str) -> None:
        self.parts.append(element)

    def collisions(self) -> list[str]:
        found = []
        for index, (text, box) in enumerate(self.texts):
            if box[0] < 0 or box[1] < 0 or box[2] > WIDTH or box[3] > HEIGHT:
                found.append(f"{text!r} leaves the card")
            found += [
                f"{text!r} overlaps {other!r}"
                for other, other_box in self.texts[index + 1 :]
                if _overlap(box, other_box)
            ]
            if any(_overlap(box, mark) for mark in self.marks):
                found.append(f"{text!r} covers a mark")
        return found

    def svg(self) -> str:
        body = "\n".join(f"  {part}" for part in self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{escape(self.label)}" '
            f'font-family="{escape(FONT)}">\n'
            f"  <title>{escape(self.label)}</title>\n"
            f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{self.theme["surface"]}"/>\n'
            f"{body}\n</svg>\n"
        )


def nice_top(value: float, ticks: int = 6) -> tuple[float, float]:
    """An axis top above `value` and a round tick step that reaches it in at most `ticks`."""

    for step in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25):
        top = step * math.ceil(max(value, step / 2) * 1.08 / step)
        if top / step <= ticks:
            return min(top, 1.0), step
    return 1.0, 0.25


def percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def tick(value: float) -> str:
    return f"{round(100 * value, 1):g}%"


def scale(lo: float, hi: float, start: float, end: float) -> Callable[[float], float]:
    return lambda value: start + (value - lo) / (hi - lo) * (end - start)


def pretty(cohort: str) -> str:
    return cohort.replace("_", " ").capitalize()


def _grid(
    card: Card,
    position: Callable[[float], float],
    top: float,
    step: float,
    span: tuple[float, float],
    vertical: bool,
) -> None:
    """Hairline gridlines at every tick, the zero line one step darker, and tick labels."""

    for index in range(round(top / step) + 1):
        value, at = index * step, position(index * step)
        stroke = card.theme["axis" if index == 0 else "grid"]
        if vertical:
            ends = f'x1="{at:.1f}" x2="{at:.1f}" y1="{span[0]}" y2="{span[1]}"'
            card.text(tick(value), at, span[1] + 40, size=24, ink="muted", anchor="middle")
        else:
            ends = f'x1="{span[0]}" x2="{span[1]}" y1="{at:.1f}" y2="{at:.1f}"'
            card.text(tick(value), span[0] - 16, at + 8, size=24, ink="muted", anchor="end")
        card.add(f'<line {ends} stroke="{stroke}" stroke-width="1"/>')
        if index == 0 and not vertical:
            card.keep_line_clear([(span[0], at), (span[1], at)])


def _rule(card: Card, start: Point, end: Point) -> None:
    """A limit: a solid 2px line in secondary ink, which labels must not cross."""

    card.add(
        f'<line x1="{start[0]:.1f}" x2="{end[0]:.1f}" y1="{start[1]:.1f}" y2="{end[1]:.1f}" '
        f'stroke="{card.theme["secondary"]}" stroke-width="2"/>'
    )
    card.keep_line_clear([start, end])


def _dot(card: Card, x: float, y: float, colour: str, radius: float = 10) -> None:
    """A filled marker with a ring in the surface colour, so it stays legible where it crosses."""

    card.add(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{colour}" '
        f'stroke="{card.theme["surface"]}" stroke-width="3"/>'
    )
    card.keep_clear((x - radius, y - radius, x + radius, y + radius))


def single_runs_card(report: Report, theme: dict[str, Any], cohort: str) -> Card:
    """Card 1:10: every epoch's score as a dot, the study's interval as a bar beneath."""

    runs = report.runs[(report.runs["cohort_id"] == cohort) & (report.runs["group"] == HARM)]
    risk = report.risks[(cohort, HARM)]
    limit = report.limits.get((cohort, HARM))
    values = sorted(float(v) for v in runs["weighted"])
    estimate, lower, upper = (risk[key] for key in ("estimate", "lower", "upper"))
    setting = pretty(cohort).lower()
    card = Card(theme, f"Single runs against the study: harmful answers, {setting}")
    card.heading(
        "A single run is one draw from a spread",
        [
            f"Harmful answers, {setting}. Each dot is the score one run over all "
            f"{report.prompts[HARM]} prompts would report;",
            "the bar is the range the full study supports, with "
            f"{risk['simultaneous_coverage']:.0%} confidence.",
        ],
        report.footer(),
    )
    top, step = nice_top(max([*values, upper or 0.0, limit or 0.0]))
    x0, x1, y0, y1 = 360.0, WIDTH - MARGIN - 40.0, 250.0, 700.0
    x = scale(0.0, top, x0, x1)
    _grid(card, x, top, step, (y0, y1), vertical=True)
    card.text("Share of harmful answers", (x0 + x1) / 2, y1 + 86, size=24, anchor="middle")
    if limit is not None:
        _rule(card, (x(limit), y0), (x(limit), y1))
        card.text(f"{tick(limit)} limit", x(limit), y0 - 14, size=24, anchor="middle")

    colour = theme["series"][0]
    radius, runs_y, study_y = 11.0, 440.0, 600.0
    card.text(f"{len(values)} single runs", MARGIN, runs_y + 9, size=28, ink="primary")
    levels: list[list[float]] = []
    for value in values:
        cx = x(value)
        level = next(
            (i for i, row in enumerate(levels) if all(abs(cx - o) >= 2 * radius + 3 for o in row)),
            len(levels),
        )
        if level == len(levels):
            levels.append([])
        levels[level].append(cx)
        _dot(card, cx, runs_y - level * (2 * radius + 3), colour, radius)
    if values:
        middle = (x(values[0]) + x(values[-1])) / 2
        card.text(
            f"{percent(values[0])} to {percent(values[-1])}",
            middle,
            runs_y + 52,
            size=24,
            anchor="middle",
        )

    card.text("Full study", MARGIN, study_y + 9, size=28, ink="primary")
    if estimate is not None:
        card.add(
            f'<rect x="{x(lower):.1f}" y="{study_y - 8}" width="{x(upper) - x(lower):.1f}" '
            f'height="16" rx="8" fill="{colour}" fill-opacity="0.35"/>'
        )
        card.keep_clear((x(lower), study_y - 8, x(upper), study_y + 8))
        _dot(card, x(estimate), study_y, colour, 13)
        card.place(
            percent(estimate),
            [(x(estimate), study_y - 30, "middle"), (x(estimate) - 22, study_y - 30, "end")],
            size=26,
            ink="primary",
        )
        middle = (x(lower) + x(upper)) / 2
        card.place(
            f"{percent(lower)} to {percent(upper)}",
            [(middle, study_y + 52, "middle"), (x(lower), study_y + 52, "start")],
            size=24,
            ink="secondary",
        )
    return card


def asking_again_card(report: Report, theme: dict[str, Any], cohort: str) -> Card:
    """Card 1:45: the unbiased worst-of-k estimate at every k, each with its interval as a bar."""

    counts = report.counts[
        (report.counts["cohort_id"] == cohort) & (report.counts["group"] == HARM)
    ]
    points = {k: unbiased_worst_of_k(counts, k) for k in range(1, LONGEST_K + 1)}
    bars = [(k, r[1], r[2]) for k, r in points.items() if r is not None]
    line = [(k, r[0]) for k, r in points.items() if r is not None and r[0] is not None]
    card = Card(theme, f"Asking again: harmful answers, {pretty(cohort).lower()}")
    card.heading(
        "Ask again, and the chance of harmful help grows",
        [
            f"{pretty(cohort)}. The chance that at least one of k answers to the same request is "
            "harmful;",
            f"each bar is the range the data support for that k alone, with {1 - ALPHA:.0%} "
            "confidence.",
        ],
        report.footer(),
    )
    top, step = nice_top(max([upper for _, _, upper in bars] + [0.0]))
    x0, x1, y0, y1 = 200.0, WIDTH - MARGIN - 60.0, 260.0, 700.0
    x = scale(1.0, float(LONGEST_K), x0, x1)
    y = scale(0.0, top, y1, y0)
    _grid(card, y, top, step, (x0, x1), vertical=False)
    for k in (1, 5, 10, 15, 20):
        card.text(str(k), x(k), y1 + 40, size=24, ink="muted", anchor="middle")
    card.text(
        "Times the same request is asked (k)", (x0 + x1) / 2, y1 + 86, size=24, anchor="middle"
    )

    # k is a whole number of tries, and the interval's blocks of rollouts change with it, so
    # each k gets a bar of its own rather than a band drawn between them.
    colour, half = theme["series"][0], 6.0
    for k, lower, upper in bars:
        card.add(
            f'<rect x="{x(k) - half:.1f}" y="{y(upper):.1f}" width="{2 * half}" '
            f'height="{max(y(lower) - y(upper), 2 * half):.1f}" rx="{half}" fill="{colour}" '
            'fill-opacity="0.3"/>'
        )
        card.keep_clear((x(k) - half, y(upper), x(k) + half, max(y(lower), y(upper) + 2 * half)))
    if not line:
        return card
    path = [(x(k), y(v)) for k, v in line]
    card.add(
        '<polyline points="'
        + " ".join(f"{px:.1f},{py:.1f}" for px, py in path)
        + f'" fill="none" stroke="{colour}" stroke-width="3" stroke-linejoin="round" '
        'stroke-linecap="round"/>'
    )
    card.keep_line_clear(path)
    estimates = dict(line)
    called = ((1, "asked once"), (10, "within ten tries"))
    for k, value in line:
        _dot(card, x(k), y(value), colour, 10 if k in dict(called) else 5)

    def clear_of_bars(content: str, at: float, anchor: str, size: int, above: bool) -> Spot:
        """A spot above, or below, every bar the label would span."""

        left, _, right, _ = Card.extent(content, at, 0, size, anchor)
        near = [(lower, upper) for k, lower, upper in bars if left - half <= x(k) <= right + half]
        if above:
            return at, min([y(upper) for _, upper in near] + [y1]) - 16, anchor
        return at, max([y(lower) for lower, _ in near] + [y0]) + 0.8 * size + 14, anchor

    def label(content: str, k: int, size: int, ink: str, nearest: bool) -> None:
        """Label k above or below its neighbours' bars, within the plot's width.

        With `nearest`, the spot closest to k's estimate comes first; otherwise above first.
        """

        spots = [
            clear_of_bars(
                content, x(k) + {"start": -half, "middle": 0, "end": half}[a], a, size, up
            )
            for up in (True, False)
            for a in ("middle", "start", "end")
        ]
        spots = [
            spot
            for spot in spots
            if x0 - half <= Card.extent(content, spot[0], spot[1], size, spot[2])[0]
            and Card.extent(content, spot[0], spot[1], size, spot[2])[2] <= WIDTH - MARGIN / 2
        ] or spots
        if nearest:
            target = y(estimates[k])
            spots.sort(key=lambda spot: abs(spot[1] - 0.3 * size - target))
        card.place(content, spots, size=size, ink=ink)

    for k, words in called:
        if k in estimates:
            label(f"{percent(estimates[k], 1)} {words}", k, 26, "primary", nearest=True)
    last = line[-1][0]
    if bars and last < bars[-1][0]:
        label(f"after {last} tries, the range alone", last + 1, 22, "muted", nearest=False)
    return card


def two_sides_card(report: Report, theme: dict[str, Any], cohorts: Sequence[str]) -> Card:
    """Card 3:40: one box per cohort spanning its harm and over-refusal intervals."""

    if len(cohorts) > len(theme["series"]):
        raise ValueError(f"at most {len(theme['series'])} cohorts fit one card, not {len(cohorts)}")
    boxes = [
        (cohort, report.risks[(cohort, REFUSAL)], report.risks[(cohort, HARM)])
        for cohort in cohorts
    ]
    coverage = {risk["simultaneous_coverage"] for _, *risks in boxes for risk in risks}
    card = Card(theme, "Harmful answers against safe requests refused, by setting")
    card.heading(
        "Each setting has two sides",
        [
            "Harmful answers against safe requests refused. Each box holds both rates for one "
            "setting;",
            "all its ranges hold together with "
            + (f"{coverage.pop():.0%}" if len(coverage) == 1 else "stated")
            + " confidence.",
        ],
        report.footer(),
    )
    harm_limits = sorted({report.limits[(c, HARM)] for c in cohorts if (c, HARM) in report.limits})
    refusal_limits = sorted(
        {report.limits[(c, REFUSAL)] for c in cohorts if (c, REFUSAL) in report.limits}
    )
    x_top, x_step = nice_top(max([r["upper"] or 0.0 for _, r, _ in boxes] + refusal_limits))
    y_top, y_step = nice_top(max([h["upper"] or 0.0 for _, _, h in boxes] + harm_limits))
    x0, x1, y0, y1 = 220.0, 1180.0, 270.0, 700.0
    x = scale(0.0, x_top, x0, x1)
    y = scale(0.0, y_top, y1, y0)
    _grid(card, y, y_top, y_step, (x0, x1), vertical=False)
    for index in range(round(x_top / x_step) + 1):
        value = index * x_step
        card.text(tick(value), x(value), y1 + 40, size=24, ink="muted", anchor="middle")
    card.text("Safe requests refused", (x0 + x1) / 2, y1 + 86, size=24, anchor="middle")
    card.text("Harmful answers", x0 - 16, y0 - 50, size=24, anchor="start")

    # The legend first, so that no label lands on it: identity never rests on colour alone. A
    # cohort without an estimate for both rates has no box, and the legend says so.
    for index, (cohort, refusal, harm) in enumerate(boxes):
        row = 300 + 64 * index
        card.add(
            f'<rect x="1290" y="{row - 18}" width="22" height="22" rx="4" '
            f'fill="{theme["series"][index]}"/>'
        )
        card.keep_clear((1290, row - 18, 1312, row + 4))
        card.text(pretty(cohort), 1324, row, size=24, ink="primary")
        if refusal["estimate"] is None or harm["estimate"] is None:
            card.text("no estimate", 1324, row + 26, size=20, ink="muted")

    shown = []
    for index, (cohort, refusal, harm) in enumerate(boxes):
        colour = theme["series"][index]
        if refusal["estimate"] is None or harm["estimate"] is None:
            continue
        left, right = x(refusal["lower"]), x(refusal["upper"])
        high, low = y(harm["upper"]), y(harm["lower"])
        card.add(
            f'<rect x="{left:.1f}" y="{high:.1f}" width="{right - left:.1f}" '
            f'height="{low - high:.1f}" rx="4" fill="{colour}" fill-opacity="0.16" '
            f'stroke="{colour}" stroke-width="2"/>'
        )
        card.keep_line_clear([(left, high), (right, high), (right, low), (left, low), (left, high)])
        _dot(card, x(refusal["estimate"]), y(harm["estimate"]), colour)
        shown.append((cohort, left, right, high, low))

    for limit in harm_limits:
        _rule(card, (x0, y(limit)), (x1, y(limit)))
    for limit in refusal_limits:
        _rule(card, (x(limit), y0), (x(limit), y1))
    for limit in harm_limits:
        card.place(
            f"{tick(limit)} limit",
            [(x1 + 14, y(limit) + 8, "start"), (x0 + 10, y(limit) - 10, "start")],
            size=22,
            ink="secondary",
        )
    for limit in refusal_limits:
        card.place(
            f"{tick(limit)} limit",
            [
                (x(limit), y0 - 14, "middle"),
                (x(limit) + 10, y0 + 28, "start"),
                (x(limit) + 10, y1 - 12, "start"),
            ],
            size=22,
            ink="secondary",
        )
    for cohort, left, right, high, low in shown:
        middle = (high + low) / 2 + 9
        card.place(
            pretty(cohort),
            [
                (right + 14, middle, "start"),
                (left - 14, middle, "end"),
                ((left + right) / 2, high - 14, "middle"),
                ((left + right) / 2, low + 34, "middle"),
            ],
            size=26,
            ink="primary",
        )
    return card


CARDS = ("single-runs", "asking-again", "two-sides")


def build(report: Report, theme: dict[str, Any], cohort: str) -> dict[str, Card]:
    single, again, sides = CARDS
    return {
        single: single_runs_card(report, theme, cohort),
        again: asking_again_card(report, theme, cohort),
        sides: two_sides_card(report, theme, report.cohorts()),
    }


def write(report_dir: Path, output_dir: Path, cohort: str) -> list[Path]:
    report = Report.read(report_dir)
    if cohort not in report.cohorts():
        raise SystemExit(f"no cohort {cohort!r}; the report has {', '.join(report.cohorts())}")
    cards = {
        (name, theme): card
        for theme in THEMES
        for name, card in build(report, THEMES[theme], cohort).items()
    }
    problems = [
        f"{name} ({theme}): {problem}"
        for (name, theme), card in cards.items()
        for problem in card.collisions()
    ]
    if problems:
        raise SystemExit("texts collide, nothing written:\n" + "\n".join(problems))
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for (name, theme), card in cards.items():
        path = output_dir / (f"{name}.svg" if theme == "light" else f"{name}-{theme}.svg")
        path.write_text(card.svg(), encoding="utf-8", newline="\n")
        written.append(path)
    return written


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="A folder `analyze` wrote.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for the SVGs.")
    parser.add_argument(
        "--cohort", default="thinking_on", help="Cohort of the single-run and asking-again cards."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    for path in write(args.report, args.output_dir, args.cohort):
        print(path)


if __name__ == "__main__":
    main()
