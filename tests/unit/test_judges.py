"""Tests for multi-judge bookkeeping: composites, effective votes, agreement."""

from __future__ import annotations

import pytest

from inspect_safety_stats.config.models import InputsSpec
from inspect_safety_stats.stats.judges import (
    compose_verdicts,
    effective_votes,
    pairwise_agreement,
)


def test_any_all_majority_rules_handle_missing_verdicts_conservatively() -> None:
    assert compose_verdicts({"a": True, "b": None}, "any") is True
    assert compose_verdicts({"a": False, "b": None}, "any") is None
    assert compose_verdicts({"a": False, "b": False}, "any") is False
    assert compose_verdicts({"a": True, "b": None}, "all") is None
    assert compose_verdicts({"a": False, "b": None}, "all") is False
    assert compose_verdicts({"a": True, "b": True}, "all") is True
    assert compose_verdicts({"a": True, "b": True, "c": False}, "majority") is True
    assert compose_verdicts({"a": False, "b": False, "c": True}, "majority") is False
    # A tie is not a verdict, and neither is a majority that depends on a missing vote.
    assert compose_verdicts({"a": True, "b": False}, "majority") is None
    assert compose_verdicts({"a": True, "b": None, "c": None}, "majority") is None
    # A majority already reached does not depend on the missing vote.
    assert compose_verdicts({"a": True, "b": True, "c": None}, "majority") is True
    with pytest.raises(ValueError, match="rule"):
        compose_verdicts({"a": True}, "mode")
    with pytest.raises(ValueError, match="at least"):
        compose_verdicts({}, "any")


def test_effective_votes_is_kish_over_declared_blocks() -> None:
    assert effective_votes(["x", "y", "z"]) == pytest.approx(3.0)
    assert effective_votes(["x", "x", "x"]) == pytest.approx(1.0)
    # Two judges in one block and one alone: (3)^2 / (2^2 + 1^2).
    assert effective_votes(["x", "x", "y"]) == pytest.approx(9 / 5)
    with pytest.raises(ValueError):
        effective_votes([])


def test_pairwise_agreement_and_cohens_kappa_by_hand() -> None:
    a = [True, True, False, False, True, None]
    b = [True, False, False, False, True, True]
    summary = pairwise_agreement(a, b)
    # The pair with a missing verdict is not compared.
    assert summary.compared == 5
    assert summary.disagreements == 1
    assert summary.agreement == pytest.approx(4 / 5)
    # p_o = 0.8; marginals 3/5 and 2/5 give p_e = 0.6*0.4 + 0.4*0.6 = 0.48.
    assert summary.kappa == pytest.approx((0.8 - 0.48) / (1 - 0.48))


def test_kappa_is_undefined_without_variation() -> None:
    summary = pairwise_agreement([True, True, True], [True, True, True])
    assert summary.agreement == pytest.approx(1.0)
    assert summary.kappa is None
    empty = pairwise_agreement([None, None], [True, False])
    assert empty.compared == 0
    assert empty.agreement is None


def _inputs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "design_manifest": "design.csv",
        "score": {
            "name": "harmful",
            "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
        },
        "judges": [
            {
                "id": "a",
                "score": {
                    "name": "harmful_a",
                    "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                },
            },
            {
                "id": "b",
                "score": {
                    "name": "harmful_b",
                    "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                },
                "block": "openai",
                "kind": "monitor",
            },
        ],
        "composites": [{"id": "any_judge", "rule": "any", "judges": ["a", "b"]}],
    }
    base.update(overrides)
    return base


def test_inputs_accept_judges_and_composites() -> None:
    spec = InputsSpec.model_validate(_inputs())
    assert [judge.id for judge in spec.judges] == ["a", "b"]
    # A judge's block defaults to its own identifier: independent unless declared otherwise.
    assert spec.judges[0].block == "a"
    assert spec.judges[1].block == "openai"
    assert spec.judges[1].kind == "monitor"
    assert spec.composites[0].rule == "any"


def test_judge_and_composite_identifiers_are_cross_checked() -> None:
    with pytest.raises(ValueError, match="unique"):
        InputsSpec.model_validate(
            _inputs(
                judges=[
                    {
                        "id": "a",
                        "score": {
                            "name": "x",
                            "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                        },
                    },
                    {
                        "id": "a",
                        "score": {
                            "name": "y",
                            "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                        },
                    },
                ]
            )
        )
    with pytest.raises(ValueError, match="unknown judge"):
        InputsSpec.model_validate(
            _inputs(composites=[{"id": "c", "rule": "any", "judges": ["a", "zzz"]}])
        )
    with pytest.raises(ValueError):
        InputsSpec.model_validate(_inputs(composites=[{"id": "c", "rule": "any", "judges": ["a"]}]))
    with pytest.raises(ValueError, match="unique"):
        InputsSpec.model_validate(
            _inputs(composites=[{"id": "a", "rule": "any", "judges": ["a", "b"]}])
        )
    with pytest.raises(ValueError, match="block"):
        InputsSpec.model_validate(
            _inputs(
                judges=[
                    {
                        "id": "a",
                        "score": {
                            "name": "x",
                            "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                        },
                        "block": "",
                    },
                    {
                        "id": "b",
                        "score": {
                            "name": "y",
                            "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                        },
                    },
                ]
            )
        )
