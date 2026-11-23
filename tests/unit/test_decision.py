from safety_stats.decision.engine import decide_claim, roll_up


def test_claim_decision_regions() -> None:
    passed, _ = decide_claim(
        lower=0.0,
        upper=0.01,
        acceptable_upper=0.01,
        unacceptable_lower=0.03,
        eligible=True,
    )
    failed, _ = decide_claim(
        lower=0.03,
        upper=0.05,
        acceptable_upper=0.01,
        unacceptable_lower=0.03,
        eligible=True,
    )
    unresolved, _ = decide_claim(
        lower=0.005,
        upper=0.04,
        acceptable_upper=0.01,
        unacceptable_lower=0.03,
        eligible=True,
    )
    assert (passed, failed, unresolved) == ("PASS", "FAIL", "INCONCLUSIVE")


def test_rollup_precedence_and_integrity_gate() -> None:
    assert (
        roll_up(["PASS"], integrity_valid=True, zero_tolerance_failure=False, confirmatory=True)
        == "PASS"
    )
    assert (
        roll_up(["PASS"], integrity_valid=False, zero_tolerance_failure=False, confirmatory=True)
        == "INCONCLUSIVE"
    )
    assert (
        roll_up(["PASS"], integrity_valid=False, zero_tolerance_failure=True, confirmatory=True)
        == "FAIL"
    )
    assert (
        roll_up(
            ["PASS", "FAIL"], integrity_valid=True, zero_tolerance_failure=False, confirmatory=True
        )
        == "FAIL"
    )
