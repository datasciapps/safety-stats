"""Projecting what a budget buys, before anything is generated or evaluated."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from safety_stats.generative.planning import forecast, rewrites_for_bound
from safety_stats.generative.schema import load_generative_plan
from safety_stats.generative.taxonomy import compile_taxonomy
from safety_stats.stats.binomial import binomial_upper_bound
from tests.generative_fixtures import SEEDS, base_plan, materialize


def prepared(tmp_path: Path, **sampling: Any) -> tuple[Any, Any]:
    plan = base_plan()
    if sampling:
        plan["sampling"] = sampling
    parsed = load_generative_plan(materialize(tmp_path, plan, SEEDS))
    return compile_taxonomy(parsed, base_path=tmp_path), parsed


def test_the_root_bound_is_the_exact_binomial_one(tmp_path: Path) -> None:
    taxonomy, plan = prepared(tmp_path)
    item = forecast(taxonomy, plan, total_rewrites=1200, assumed_rate=0.03)
    assert item.root_upper == binomial_upper_bound(36, 1200, alpha=0.05)


def test_more_budget_tightens_the_bound_and_fills_the_tree(tmp_path: Path) -> None:
    taxonomy, plan = prepared(tmp_path)
    small = forecast(taxonomy, plan, total_rewrites=100, assumed_rate=0.03)
    large = forecast(taxonomy, plan, total_rewrites=10_000, assumed_rate=0.03)
    assert large.root_upper < small.root_upper
    assert large.paths_with_a_draw > small.paths_with_a_draw
    assert large.expected_unsampled_nodes < small.expected_unsampled_nodes
    assert large.paths_with_a_draw == pytest.approx(1.0, abs=1e-6)
    assert large.expected_unsampled_nodes == pytest.approx(0.0, abs=1e-6)


def test_levels_are_reported_in_hierarchy_order_and_get_thinner(tmp_path: Path) -> None:
    taxonomy, plan = prepared(tmp_path)
    item = forecast(taxonomy, plan, total_rewrites=1200, assumed_rate=0.03)
    assert [level.level for level in item.levels] == list(taxonomy.levels)
    draws = [level.median_draws for level in item.levels]
    assert draws == sorted(draws, reverse=True)
    bounds = [level.median_upper for level in item.levels]
    assert bounds == sorted(bounds)
    assert item.levels[0].median_upper < item.levels[-1].median_upper


def test_a_deep_level_is_visibly_unresolvable(tmp_path: Path) -> None:
    """The point of the tool: this is knowable before spending anything."""

    taxonomy, plan = prepared(tmp_path)
    item = forecast(taxonomy, plan, total_rewrites=100, assumed_rate=0.03)
    deepest = item.levels[-1]
    assert deepest.median_draws < 10
    assert deepest.median_upper > 4 * item.root_upper

    # And that the same level becomes resolvable once the budget is large enough.
    ample = forecast(taxonomy, plan, total_rewrites=20_000, assumed_rate=0.03)
    assert ample.levels[-1].median_upper < 2 * ample.root_upper


def test_retry_cost_and_lost_units_are_projected(tmp_path: Path) -> None:
    taxonomy, plan = prepared(tmp_path)
    clean = forecast(taxonomy, plan, total_rewrites=1000, assumed_rate=0.03)
    assert clean.rewrite_calls == pytest.approx(1000)
    assert clean.exhausted_units == pytest.approx(0.0)

    lossy = forecast(
        taxonomy, plan, total_rewrites=1000, assumed_rate=0.03, acceptance_rate=0.5
    )
    assert lossy.rewrite_calls > 1900          # about two attempts per accepted unit
    assert lossy.exhausted_units == pytest.approx(1000 * 0.5**5)
    assert lossy.target_rollouts < 1000        # exhausted units are never evaluated


def test_direct_sampling_keeps_equal_coefficients(tmp_path: Path) -> None:
    taxonomy, plan = prepared(tmp_path, method="direct", total_rewrites=600)
    item = forecast(taxonomy, plan, total_rewrites=600, assumed_rate=0.03)
    assert item.coefficient_ratio == 1.0
    assert item.starved_strata == 0


def test_a_stratum_floor_shows_up_as_unequal_coefficients(tmp_path: Path) -> None:
    taxonomy, plan = prepared(
        tmp_path,
        method="proportional_stratified",
        total_rewrites=600,
        minimum_rewrites_per_stratum=150,
    )
    item = forecast(taxonomy, plan, total_rewrites=600, assumed_rate=0.03)
    assert item.coefficient_ratio > 1.0


def test_an_unaffordable_floor_is_reported_not_raised(tmp_path: Path) -> None:
    taxonomy, plan = prepared(
        tmp_path,
        method="proportional_stratified",
        total_rewrites=600,
        minimum_rewrites_per_stratum=150,
    )
    item = forecast(taxonomy, plan, total_rewrites=100, assumed_rate=0.03)
    assert item.starved_strata > 0


def test_certification_power_rises_with_budget(tmp_path: Path) -> None:
    taxonomy, plan = prepared(tmp_path)
    weak = forecast(taxonomy, plan, total_rewrites=200, assumed_rate=0.02, threshold=0.05)
    strong = forecast(taxonomy, plan, total_rewrites=5000, assumed_rate=0.02, threshold=0.05)
    assert weak.certification_power is not None and strong.certification_power is not None
    assert 0.0 <= weak.certification_power < strong.certification_power <= 1.0
    assert strong.certification_power > 0.99


def test_a_threshold_at_or_below_the_assumed_rate_is_rejected(tmp_path: Path) -> None:
    taxonomy, plan = prepared(tmp_path)
    with pytest.raises(ValueError, match="threshold must exceed"):
        forecast(taxonomy, plan, total_rewrites=500, assumed_rate=0.05, threshold=0.05)


def test_rewrites_for_bound_inverts_the_forecast(tmp_path: Path) -> None:
    taxonomy, plan = prepared(tmp_path)
    needed = rewrites_for_bound(assumed_rate=0.03, target_upper=0.04, alpha=0.05)
    assert needed is not None
    assert forecast(taxonomy, plan, total_rewrites=needed, assumed_rate=0.03).root_upper <= 0.04
    tighter = forecast(taxonomy, plan, total_rewrites=needed - 1, assumed_rate=0.03)
    assert tighter.root_upper > 0.04


def test_a_level_s_worst_bound_is_never_below_its_median_bound(tmp_path: Path) -> None:
    """Rounded failure counts once let a smaller node forecast a tighter bound.

    At 3%, 16 draws expected no failure and 20 expected one, so at a budget of 34 the
    ``problem_class`` level's worst node forecast 0.193 against its median node's 0.250.
    """

    taxonomy, plan = prepared(tmp_path)
    inverted = [
        (budget, level.level)
        for budget in range(20, 401)
        for level in forecast(taxonomy, plan, total_rewrites=budget, assumed_rate=0.03).levels
        if level.worst_upper < level.median_upper
    ]
    assert inverted == []


def test_every_budget_above_the_one_found_reaches_the_target(tmp_path: Path) -> None:
    """The search bisects the forecast bound, which must fall as the budget grows."""

    taxonomy, plan = prepared(tmp_path)
    needed = rewrites_for_bound(assumed_rate=0.03, target_upper=0.06, alpha=0.05)
    assert needed is not None
    roots = [
        forecast(taxonomy, plan, total_rewrites=budget, assumed_rate=0.03).root_upper
        for budget in range(needed - 50, needed + 200)
    ]
    assert all(later < earlier for earlier, later in zip(roots, roots[1:], strict=False))
    assert roots[49] > 0.06 >= roots[50]


def test_rewrites_for_bound_gives_up_rather_than_hanging() -> None:
    assert rewrites_for_bound(
        assumed_rate=0.03, target_upper=0.0300001, alpha=0.05, maximum=50_000
    ) is None
    with pytest.raises(ValueError, match="must exceed the assumed rate"):
        rewrites_for_bound(assumed_rate=0.05, target_upper=0.04, alpha=0.05)


def test_forecast_validates_its_inputs(tmp_path: Path) -> None:
    taxonomy, plan = prepared(tmp_path)
    with pytest.raises(ValueError, match="total_rewrites must be positive"):
        forecast(taxonomy, plan, total_rewrites=0, assumed_rate=0.03)
    with pytest.raises(ValueError, match="assumed_rate must be"):
        forecast(taxonomy, plan, total_rewrites=100, assumed_rate=1.0)
    with pytest.raises(ValueError, match="acceptance_rate must be"):
        forecast(taxonomy, plan, total_rewrites=100, assumed_rate=0.03, acceptance_rate=0.0)


def test_the_forecast_does_not_disturb_the_plan(tmp_path: Path) -> None:
    """Projecting other budgets must not leave the plan describing one of them."""

    taxonomy, plan = prepared(
        tmp_path, method="proportional_stratified", total_rewrites=600
    )
    forecast(taxonomy, plan, total_rewrites=9999, assumed_rate=0.03)
    assert plan.sampling.total_rewrites == 600
