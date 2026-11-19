"""Arm contrasts and the simultaneous slice family on a generative design."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_safety_stats.generative.contrasts import (
    RATIO_KL,
    STRATIFIED_HT,
    ContrastMember,
    SliceMember,
    estimate_slice_family,
)
from inspect_safety_stats.generative.inference import EXACT_METHOD, UnitOutcome, estimate_slice
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import CompiledTaxonomy, compile_taxonomy
from tests.generative_fixtures import SEEDS, base_plan, materialize


def prepare(
    tmp_path: Path,
    *,
    method: str = "direct",
    total: int = 600,
    confounded: bool = False,
) -> tuple[CompiledTaxonomy, Any, list[Any]]:
    plan = base_plan()
    plan["sampling"] = {"method": method, "total_rewrites": total}
    if confounded:
        # Under bio, plain carries 0.8 of the wrapper weight and roleplay 0.1, so the plain
        # slice over-represents bio and the roleplay slice under-represents it.
        plan["taxonomy"]["arms"]["wrapper"]["context_weights"] = [
            {"when": {"problem_class": "bio"},
             "values": [{"id": "plain", "weight": 0.8}, {"id": "roleplay", "weight": 0.1},
                        {"id": "nested", "weight": 0.1}]}
        ]
    parsed = load_generative_plan(materialize(tmp_path, plan, SEEDS))
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    design = plan_design(taxonomy, parsed)
    return taxonomy, design, draw_selections(taxonomy, design)


def risk_of(node_id: str) -> float:
    risk = 0.02
    if "wrapper=roleplay" in node_id:
        risk += 0.08
    if "problem_class=bio" in node_id:
        risk += 0.10
    return risk


def only_roleplay_fails(node_id: str) -> float:
    """Only roleplay paths fail, most under bio, where the confounded weights thin them."""

    if "wrapper=roleplay" not in node_id:
        return 0.0
    return 0.5 if "problem_class=bio" in node_id else 0.1


def outcomes(
    selections: list[Any],
    rng: np.random.Generator,
    path_risk: Callable[[str], float] = risk_of,
) -> list[UnitOutcome]:
    draws = rng.random(len(selections))
    return [
        UnitOutcome(record.sampling_unit_id, 1, int(draw < path_risk(record.node_id)), 0)
        for record, draw in zip(selections, draws, strict=True)
    ]


def redrawn(taxonomy: CompiledTaxonomy, design: Any, repetition: int) -> list[Any]:
    """The paths a different selection seed draws.

    A bound covers over path selection as well as outcomes, so a coverage test that kept one
    selection would measure coverage conditional on where its draws happened to land.
    """

    seed = design.selection_seed + 1 + repetition
    return draw_selections(taxonomy, replace(design, selection_seed=seed))


def seed_risk(
    taxonomy: CompiledTaxonomy,
    level: str,
    value: str,
    path_risk: Callable[[str], float] = risk_of,
) -> dict[str, tuple[float, float]]:
    """Per seed: the seed's weight, and the declared risk of its paths carrying the value."""

    seed: dict[str, float] = {}
    cell: dict[str, float] = {}
    risk: dict[str, float] = {}
    for node in taxonomy.terminal_paths:
        key = str(node.semantic_seed_id)
        seed[key] = seed.get(key, 0.0) + node.target_path_weight
        if f"{level}={value}" in node.node_id.split("|"):
            cell[key] = cell.get(key, 0.0) + node.target_path_weight
            risk[key] = risk.get(key, 0.0) + node.target_path_weight * path_risk(node.node_id)
    return {key: (seed[key], risk[key] / cell[key]) for key in cell}


def true_contrast(
    taxonomy: CompiledTaxonomy,
    level: str,
    a: str,
    b: str,
    path_risk: Callable[[str], float] = risk_of,
) -> float:
    first = seed_risk(taxonomy, level, a, path_risk)
    second = seed_risk(taxonomy, level, b, path_risk)
    shared = sorted(set(first) & set(second))
    mass = sum(first[key][0] for key in shared)
    return sum(first[key][0] / mass * (first[key][1] - second[key][1]) for key in shared)


def true_slice(taxonomy: CompiledTaxonomy, level: str, value: str) -> float:
    paths = [n for n in taxonomy.terminal_paths if f"{level}={value}" in n.node_id.split("|")]
    mass = sum(n.target_path_weight for n in paths)
    return sum(n.target_path_weight * risk_of(n.node_id) for n in paths) / mass


def test_a_single_unconfounded_slice_matches_the_pointwise_slice(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path)
    observed = outcomes(selections, np.random.default_rng(1))
    for mode in ("one_sided_upper", "two_sided"):
        family = estimate_slice_family(
            taxonomy, design, selections, observed,
            slices=[SliceMember("wrapper", "roleplay")], coverage_mode=mode,  # type: ignore[arg-type]
        )
        pointwise = estimate_slice(
            taxonomy, design, selections, observed, level="wrapper", value="roleplay",
            coverage_mode=mode,  # type: ignore[arg-type]
        )
        member = family.slices[0]
        assert member.arm.method == EXACT_METHOD
        assert member.arm.maximum_ratio == 1.0
        assert member.arm.draws == pointwise.draws
        assert member.arm.lower == pytest.approx(pointwise.lower, rel=1e-12, abs=0)
        assert member.arm.upper == pytest.approx(pointwise.upper, rel=1e-12, abs=0)
        assert member.weight == pytest.approx(pointwise.node_weight, rel=1e-12, abs=0)


def test_slices_of_one_level_share_a_sidak_frontier_under_direct_sampling(
    tmp_path: Path,
) -> None:
    taxonomy, design, selections = prepare(tmp_path)
    observed = outcomes(selections, np.random.default_rng(2))
    family = estimate_slice_family(
        taxonomy, design, selections, observed,
        slices=[SliceMember("wrapper", "plain"), SliceMember("wrapper", "roleplay"),
                SliceMember("language", "maltese")],
    )
    plain, roleplay, maltese = family.slices
    assert plain.frontier_id == roleplay.frontier_id == "slices:wrapper"
    assert plain.frontier_size == 2
    assert maltese.frontier_id == "slices:language"
    # Two frontiers share 0.05 by Bonferroni; two independent slices split 0.025 by Sidak.
    assert plain.endpoint_alpha == pytest.approx(-np.expm1(np.log1p(-0.025) / 2), rel=1e-12, abs=0)
    assert maltese.endpoint_alpha == pytest.approx(0.025, rel=1e-12, abs=0)


def test_stratified_slices_read_shared_draws_and_take_frontiers_of_their_own(
    tmp_path: Path,
) -> None:
    taxonomy, design, selections = prepare(tmp_path, method="proportional_stratified")
    observed = outcomes(selections, np.random.default_rng(3))
    family = estimate_slice_family(
        taxonomy, design, selections, observed,
        slices=[SliceMember("wrapper", "plain"), SliceMember("wrapper", "roleplay")],
    )
    assert {item.frontier_size for item in family.slices} == {1}
    assert all(item.arm.method == STRATIFIED_HT for item in family.slices)
    for item in family.slices:
        assert item.endpoint_alpha == pytest.approx(0.025, rel=1e-12, abs=0)


def test_a_confounded_contrast_uses_the_ratio_construction_on_shared_seeds(
    tmp_path: Path,
) -> None:
    taxonomy, design, selections = prepare(tmp_path, confounded=True)
    observed = outcomes(selections, np.random.default_rng(4))
    family = estimate_slice_family(
        taxonomy, design, selections, observed,
        contrasts=[
            ContrastMember("roleplay-vs-plain", "wrapper", "roleplay", "plain"),
            ContrastMember("nested-vs-plain", "wrapper", "nested", "plain"),
        ],
        coverage_mode="two_sided",
    )
    for contrast in family.contrasts:
        assert contrast.shared_seeds == 3
        assert contrast.candidate.method == RATIO_KL
        assert contrast.baseline.maximum_ratio > 1.0
        assert contrast.lower <= contrast.estimate <= contrast.upper  # type: ignore[operator]
        assert contrast.frontier_size == 1
        # Two frontiers share 0.05, and two-sided coverage halves each one's share.
        assert contrast.endpoint_alpha == pytest.approx(0.0125, rel=1e-12, abs=0)
        # The union bound halves it again over the two arms, at both ends of each.
        for arm in (contrast.candidate, contrast.baseline):
            assert arm.alpha_lower == pytest.approx(0.00625, rel=1e-12, abs=0)
            assert arm.alpha_upper == pytest.approx(0.00625, rel=1e-12, abs=0)


def test_a_contrast_is_restricted_to_the_seeds_holding_both_values(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["renormalization"] = "renormalize_valid_children"
    plan["taxonomy"]["constraints"] = [
        {"id": "bio-no-nested", "when": {"problem_class": "bio"},
         "allow": {"wrapper": ["plain", "roleplay"]}}
    ]
    parsed = load_generative_plan(materialize(tmp_path, plan, SEEDS))
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    design = plan_design(taxonomy, parsed)
    selections = draw_selections(taxonomy, design)
    family = estimate_slice_family(
        taxonomy, design, selections, outcomes(selections, np.random.default_rng(11)),
        contrasts=[ContrastMember("c", "wrapper", "nested", "plain")],
    )
    contrast = family.contrasts[0]
    # nested exists under chem only. Over the two chem seeds the wrapper mix is the same, so
    # no ratio is needed and each arm is an exact slice of the chem seeds.
    assert contrast.shared_seeds == 2
    assert contrast.candidate.method == EXACT_METHOD
    assert contrast.baseline.method == EXACT_METHOD
    assert contrast.baseline.draws == sum(
        "wrapper=plain" in record.node_id and record.semantic_seed_id != "s003"
        for record in selections
    )


def test_one_sided_contrasts_bound_only_from_above(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path)
    observed = outcomes(selections, np.random.default_rng(5))
    family = estimate_slice_family(
        taxonomy, design, selections, observed,
        contrasts=[ContrastMember("c", "wrapper", "roleplay", "plain")],
    )
    contrast = family.contrasts[0]
    assert contrast.lower == -1.0
    assert contrast.upper < 1.0
    # Each arm takes half the member's alpha in each direction.
    assert contrast.candidate.alpha_upper == pytest.approx(0.025, rel=1e-12, abs=0)
    assert contrast.baseline.alpha_lower == pytest.approx(0.025, rel=1e-12, abs=0)


def test_unclassified_rollouts_widen_both_endpoints(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, confounded=True)
    observed = outcomes(selections, np.random.default_rng(6))
    contrasts = [ContrastMember("c", "wrapper", "roleplay", "plain")]
    complete = estimate_slice_family(
        taxonomy, design, selections, observed, contrasts=contrasts, coverage_mode="two_sided"
    ).contrasts[0]
    unknown = [UnitOutcome(item.sampling_unit_id, 1, 0, 1) for item in observed[:40]]
    partial = estimate_slice_family(
        taxonomy, design, selections, unknown + observed[40:], contrasts=contrasts,
        coverage_mode="two_sided",
    ).contrasts[0]
    assert partial.upper >= complete.upper
    assert partial.lower <= complete.lower
    assert partial.upper - partial.lower > complete.upper - complete.lower


@pytest.mark.parametrize(
    ("method", "confounded"),
    [("direct", False), ("direct", True), ("proportional_stratified", True)],
)
def test_the_family_covers_simultaneously(tmp_path: Path, method: str, confounded: bool) -> None:
    taxonomy, design, _ = prepare(tmp_path, method=method, total=300, confounded=confounded)
    slices = [SliceMember("wrapper", "plain"), SliceMember("wrapper", "roleplay")]
    contrasts = [
        ContrastMember("roleplay-vs-plain", "wrapper", "roleplay", "plain"),
        ContrastMember("maltese-vs-english", "language", "maltese", "english"),
    ]
    slice_truth = [true_slice(taxonomy, item.level, item.value) for item in slices]
    contrast_truth = [
        true_contrast(taxonomy, item.level, item.candidate, item.baseline) for item in contrasts
    ]
    rng = np.random.default_rng(7)
    alpha, repetitions = 0.10, 300
    upper_misses = lower_misses = 0
    for repetition in range(repetitions):
        drawn = redrawn(taxonomy, design, repetition)
        family = estimate_slice_family(
            taxonomy, design, drawn, outcomes(drawn, rng),
            slices=slices, contrasts=contrasts, alpha=alpha, coverage_mode="two_sided",
        )
        pairs = [
            *((item.arm.lower, item.arm.upper, truth)
              for item, truth in zip(family.slices, slice_truth, strict=True)),
            *((item.lower, item.upper, truth)
              for item, truth in zip(family.contrasts, contrast_truth, strict=True)),
        ]
        upper_misses += any(upper < truth for _, upper, truth in pairs)
        lower_misses += any(lower > truth for lower, _, truth in pairs)
    # The family covers each direction at 1 - alpha: every upper bound together, and every
    # lower bound together. A binomial allowance of three standard errors keeps the test from
    # failing by chance. Two random arms make the union bound of a contrast far from tight,
    # so this test cannot see an arm given too much alpha; the exact-value tests above pin
    # the levels, and the next test bounds a contrast whose union bound is tight.
    allowance = 3 * np.sqrt(alpha * (1 - alpha) / repetitions)
    assert upper_misses / repetitions <= alpha + allowance
    assert lower_misses / repetitions <= alpha + allowance


@pytest.mark.parametrize(
    ("confounded", "mode"),
    [(False, "one_sided_upper"), (False, "two_sided"), (True, "one_sided_upper")],
)
def test_a_contrast_with_a_never_failing_baseline_covers_at_its_level(
    tmp_path: Path, confounded: bool, mode: str
) -> None:
    taxonomy, design, _ = prepare(tmp_path, total=300, confounded=confounded)
    truth = true_contrast(taxonomy, "wrapper", "roleplay", "plain", only_roleplay_fails)
    # The seed-matched contrast is 0.6 * 0.1 + 0.4 * 0.5 with or without confounding; the
    # confounded roleplay slice alone would be 0.184.
    assert truth == pytest.approx(0.26, rel=1e-12, abs=0)
    contrasts = [ContrastMember("c", "wrapper", "roleplay", "plain")]
    rng = np.random.default_rng(12)
    alpha, repetitions, misses = 0.10, 400, 0
    for repetition in range(repetitions):
        drawn = redrawn(taxonomy, design, repetition)
        family = estimate_slice_family(
            taxonomy, design, drawn, outcomes(drawn, rng, only_roleplay_fails),
            contrasts=contrasts, alpha=alpha, coverage_mode=mode,  # type: ignore[arg-type]
        )
        misses += family.contrasts[0].upper < truth
    # plain never fails, so its lower bound is its risk, 0, and the contrast's upper bound
    # misses exactly when the roleplay arm's does: the union bound is tight, and the miss
    # rate follows the level the arm is given. Under one_sided_upper that is alpha / 2; under
    # two_sided alpha / 4. The upper bound covers at 1 - alpha in either mode, with an
    # allowance of three binomial standard errors. Confounded, the arm reweights its draws by
    # a maximum ratio of 1.9, so dropping the ratio or not rescaling by it also misses here.
    assert misses / repetitions <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / repetitions)


def test_true_contrast_holds_the_seed_mix_fixed(tmp_path: Path) -> None:
    taxonomy, _, _ = prepare(tmp_path, confounded=True)
    # roleplay adds 0.08 everywhere; the seed effect cancels only if seeds are matched.
    assert true_contrast(taxonomy, "wrapper", "roleplay", "plain") == pytest.approx(
        0.08, rel=1e-12, abs=1e-15
    )
    naive = true_slice(taxonomy, "wrapper", "roleplay") - true_slice(taxonomy, "wrapper", "plain")
    assert naive != pytest.approx(0.08, rel=1e-6, abs=0)


def test_the_estimate_recovers_a_seed_matched_contrast(tmp_path: Path) -> None:
    taxonomy, design, selections = prepare(tmp_path, total=20000, confounded=True)
    family = estimate_slice_family(
        taxonomy, design, selections, outcomes(selections, np.random.default_rng(8)),
        contrasts=[ContrastMember("c", "wrapper", "roleplay", "plain")],
    )
    assert family.contrasts[0].estimate == pytest.approx(0.08, abs=0.015)


@pytest.mark.parametrize(
    ("slices", "contrasts", "message"),
    [
        ([], [], "at least one"),
        ([SliceMember("wrapper", "plain")] * 2, [], "same slice twice"),
        ([], [ContrastMember("c", "wrapper", "plain", "roleplay")] * 2, "same contrast id"),
        ([], [ContrastMember("c", "wrapper", "plain", "plain")], "with itself"),
        ([SliceMember("wrapper", "missing")], [], "no terminal path carries"),
        ([SliceMember("problem_class", "bio")], [], "at or above the semantic seed"),
        ([SliceMember("colour", "red")], [], "not a level"),
    ],
)
def test_undeclarable_members_are_refused(
    tmp_path: Path, slices: list[SliceMember], contrasts: list[ContrastMember], message: str
) -> None:
    taxonomy, design, selections = prepare(tmp_path)
    observed = outcomes(selections, np.random.default_rng(9))
    with pytest.raises(ValueError, match=message):
        estimate_slice_family(
            taxonomy, design, selections, observed, slices=slices, contrasts=contrasts
        )


def test_a_contrast_without_shared_seeds_is_refused(tmp_path: Path) -> None:
    plan = base_plan()
    plan["taxonomy"]["renormalization"] = "renormalize_valid_children"
    plan["taxonomy"]["constraints"] = [
        {"id": "bio-plain", "when": {"problem_class": "bio"}, "allow": {"wrapper": ["plain"]}},
        {"id": "chem-nested", "when": {"problem_class": "chem"},
         "allow": {"wrapper": ["nested", "roleplay"]}},
    ]
    parsed = load_generative_plan(materialize(tmp_path, plan, SEEDS))
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    design = plan_design(taxonomy, parsed)
    selections = draw_selections(taxonomy, design)
    with pytest.raises(ValueError, match="no semantic seed holds both"):
        estimate_slice_family(
            taxonomy, design, selections, outcomes(selections, np.random.default_rng(10)),
            contrasts=[ContrastMember("c", "wrapper", "nested", "plain")],
        )
