"""Tests for the Lang-Reiczigel limits for true risk with estimated judge accuracy."""

from __future__ import annotations

import pytest

from inspect_safety_stats.stats.misclassification import (
    TruthStratumCounts,
    lang_reiczigel_corrected_risk,
)

# The worked examples of Lang and Reiczigel (2014), with the 95% limits the paper reports
# (as quoted in the documentation of the R package asht, function prevSeSp). Each row is
# (apparent rate, its sample size, sensitivity counts, specificity counts, limits, decimals).
PAPER_EXAMPLES = [
    (4060 / 11284, 11284, (179, 178), (359, 358), (0.349, 0.372), 3),
    (51 / 2971, 2971, (33, 32), (20, 20), (0.0, 0.053), 3),
    (0.06, 11862, (10, 8), (12, 12), (0.0, 0.147), 3),
    (259 / 509, 509, (127, 84), (109, 96), (0.58, 0.87), 2),
    (51 / 509, 509, (41, 23), (195, 187), (0.037, 0.195), 3),
]


@pytest.mark.parametrize(
    ("rate", "size", "sensitivity", "specificity", "limits", "decimals"), PAPER_EXAMPLES
)
def test_lang_reiczigel_reproduces_the_papers_worked_examples(
    rate: float,
    size: int,
    sensitivity: tuple[int, int],
    specificity: tuple[int, int],
    limits: tuple[float, float],
    decimals: int,
) -> None:
    se_labelled, se_positive = sensitivity
    sp_labelled, sp_negative = specificity
    result = lang_reiczigel_corrected_risk(
        apparent_rate=rate,
        apparent_size=size,
        truly_unsafe=TruthStratumCounts(labelled=se_labelled, judge_positive=se_positive),
        truly_safe=TruthStratumCounts(
            labelled=sp_labelled, judge_positive=sp_labelled - sp_negative
        ),
        alpha=0.05,
    )
    assert round(result.lower, decimals) == pytest.approx(limits[0], abs=0)
    assert round(result.upper, decimals) == pytest.approx(limits[1], abs=0)
    assert result.sensitivity_estimate == pytest.approx(se_positive / se_labelled, abs=0)
    assert result.specificity_estimate == pytest.approx(sp_negative / sp_labelled, abs=0)


def test_lang_reiczigel_estimate_is_the_rogan_gladen_inversion() -> None:
    result = lang_reiczigel_corrected_risk(
        apparent_rate=0.12,
        apparent_size=1000,
        truly_unsafe=TruthStratumCounts(labelled=50, judge_positive=45),
        truly_safe=TruthStratumCounts(labelled=200, judge_positive=6),
        alpha=0.05,
    )
    assert result.estimate == pytest.approx((0.12 + 0.97 - 1) / (0.90 + 0.97 - 1), abs=1e-15)
    assert result.lower < result.estimate < result.upper
    assert result.alpha == 0.05


def test_lang_reiczigel_clips_a_negative_estimate_and_limit_to_zero() -> None:
    # The judge reports fewer failures than its false-positive rate alone would produce.
    result = lang_reiczigel_corrected_risk(
        apparent_rate=0.04,
        apparent_size=500,
        truly_unsafe=TruthStratumCounts(labelled=20, judge_positive=18),
        truly_safe=TruthStratumCounts(labelled=100, judge_positive=5),
        alpha=0.05,
    )
    assert result.estimate == 0.0
    assert result.lower == 0.0
    assert 0.0 < result.upper < 1.0


def test_lang_reiczigel_collapses_to_zero_when_the_data_contradict_the_accuracies() -> None:
    # Far fewer judge-reported failures than the estimated false-positive rate implies:
    # the whole Wald interval lies below zero, and clipping leaves the single point 0.
    result = lang_reiczigel_corrected_risk(
        apparent_rate=0.01,
        apparent_size=500,
        truly_unsafe=TruthStratumCounts(labelled=20, judge_positive=18),
        truly_safe=TruthStratumCounts(labelled=100, judge_positive=5),
        alpha=0.05,
    )
    assert (result.lower, result.upper) == (0.0, 0.0)


def test_lang_reiczigel_narrows_as_the_confidence_level_falls() -> None:
    kwargs = {
        "apparent_rate": 0.2,
        "apparent_size": 800,
        "truly_unsafe": TruthStratumCounts(labelled=60, judge_positive=50),
        "truly_safe": TruthStratumCounts(labelled=150, judge_positive=9),
    }
    wide = lang_reiczigel_corrected_risk(alpha=0.01, **kwargs)  # type: ignore[arg-type]
    narrow = lang_reiczigel_corrected_risk(alpha=0.2, **kwargs)  # type: ignore[arg-type]
    assert narrow.upper - narrow.lower < wide.upper - wide.lower


def test_lang_reiczigel_refuses_an_uninformative_judge() -> None:
    with pytest.raises(ValueError, match="informative judge"):
        lang_reiczigel_corrected_risk(
            apparent_rate=0.3,
            apparent_size=100,
            truly_unsafe=TruthStratumCounts(labelled=10, judge_positive=5),
            truly_safe=TruthStratumCounts(labelled=10, judge_positive=5),
            alpha=0.05,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"apparent_rate": -0.1}, "apparent rate"),
        ({"apparent_rate": 1.1}, "apparent rate"),
        ({"apparent_size": 0}, "apparent size"),
        ({"truly_unsafe": TruthStratumCounts(labelled=0, judge_positive=0)}, "truth strata"),
        ({"truly_safe": TruthStratumCounts(labelled=0, judge_positive=0)}, "truth strata"),
    ],
)
def test_lang_reiczigel_rejects_invalid_inputs(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "apparent_rate": 0.1,
        "apparent_size": 100,
        "truly_unsafe": TruthStratumCounts(labelled=10, judge_positive=9),
        "truly_safe": TruthStratumCounts(labelled=50, judge_positive=1),
        "alpha": 0.05,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        lang_reiczigel_corrected_risk(**arguments)  # type: ignore[arg-type]
