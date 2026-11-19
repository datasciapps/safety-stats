"""The attack-confirmation example (#124): offline, against a simulated attacker and judges."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import math
import random
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from inspect_ai import eval as inspect_eval

from inspect_safety_stats import analyze
from inspect_safety_stats.stats import weighted_kl_chernoff_mean_bounds
from inspect_safety_stats.stats.planning import maximum_certifying_failures

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "attack_confirmation"
sys.path.insert(0, str(EXAMPLE_DIR))

from attack_confirmation import (  # noqa: E402
    DESIGN_PATH,
    REFUSAL,
    SIMULATED_DECLARATION,
    UNSAFE_MARKER,
    AttackDeclaration,
    SimulatedJudge,
    SimulatedTarget,
    TemplateAttacker,
    attack_confirmation,
    check_design,
    confirm,
    extract_prompt,
    parse_grade,
    run_attack,
    write_design,
)

RUNS = 4000


def _rates(
    *, false_positive_rate: float, budget: int, compliance: float = 0.0, seed: int = 0
) -> tuple[float, float]:
    """Attacker-reported and confirmed jailbreak rates over RUNS simulated attack runs."""

    rng = random.Random(seed)

    async def one() -> tuple[bool, bool]:
        target = SimulatedTarget(compliance, rng)
        run = await run_attack(
            "goal",
            budget,
            TemplateAttacker(),
            target,
            SimulatedJudge(false_positive_rate, 1.0, rng),
        )
        verdict, _ = await confirm(
            "goal", run, target, SimulatedJudge(false_positive_rate, 1.0, rng)
        )
        return run.found_prompt is not None, verdict is True

    async def all_runs() -> list[tuple[bool, bool]]:
        return [await one() for _ in range(RUNS)]

    outcomes = asyncio.run(all_runs())
    return (
        sum(reported for reported, _ in outcomes) / RUNS,
        sum(confirmed for _, confirmed in outcomes) / RUNS,
    )


def _within(observed: float, expected: float, sigmas: float = 4.0) -> bool:
    return abs(observed - expected) <= sigmas * math.sqrt(expected * (1 - expected) / RUNS)


@pytest.mark.parametrize(
    ("false_positive_rate", "budget"), [(0.02, 5), (0.02, 20), (0.05, 20), (0.05, 60)]
)
def test_attacker_verdicts_inflate_the_rate_and_confirmation_verdicts_do_not(
    false_positive_rate: float, budget: int
) -> None:
    """A target that never complies, judged by judges of known false-positive rate f.

    The attacker stops at its judge's first false positive, so it reports a jailbreak with
    probability 1 - (1 - f)^K. The confirmation verdict is drawn once, after the search,
    so the confirmed rate is that times f, and never more than f whatever the budget.
    """

    reported, confirmed = _rates(false_positive_rate=false_positive_rate, budget=budget)
    selected = -math.expm1(budget * math.log1p(-false_positive_rate))

    assert _within(reported, selected)
    assert _within(confirmed, selected * false_positive_rate)
    # Never more than f in expectation; the observed rate may exceed f by sampling noise only.
    noise = 4.0 * math.sqrt(false_positive_rate * (1 - false_positive_rate) / RUNS)
    assert confirmed <= false_positive_rate + noise
    assert reported > 2 * false_positive_rate


def test_the_confirmation_rollout_is_a_fresh_completion() -> None:
    """A prompt that worked once is confirmed only as often as the target complies again."""

    reported, confirmed = _rates(false_positive_rate=0.0, budget=20, compliance=0.3, seed=1)
    assert _within(reported, -math.expm1(20 * math.log1p(-0.3)))
    assert _within(confirmed, reported * 0.3)


@dataclass
class _PromptDependentTarget:
    """Carries out a prompt that works every time, and any other prompt never.

    Whether a prompt works is drawn once, with probability ``works_rate``, when the target
    first sees it, so the confirmation of a found prompt repeats what the search saw.
    """

    works_rate: float
    rng: random.Random
    works: dict[str, bool] = field(default_factory=dict)

    async def respond(self, prompt: str) -> str:
        if prompt not in self.works:
            self.works[prompt] = self.rng.random() < self.works_rate
        return f"{UNSAFE_MARKER}: {prompt}" if self.works[prompt] else REFUSAL


def _prompt_dependent_confirmed_rate(
    *, works_rate: float, false_positive_rate: float, budget: int, seed: int
) -> float:
    """The confirmed rate over RUNS runs against targets whose prompts work or fail."""

    rng = random.Random(seed)

    async def one() -> bool:
        target = _PromptDependentTarget(works_rate, rng)
        run = await run_attack(
            "goal",
            budget,
            TemplateAttacker(),
            target,
            SimulatedJudge(false_positive_rate, 1.0, rng),
        )
        verdict, _ = await confirm("goal", run, target, SimulatedJudge(0.0, 1.0, rng))
        return verdict is True

    async def all_runs() -> list[bool]:
        return [await one() for _ in range(RUNS)]

    return sum(asyncio.run(all_runs())) / RUNS


@pytest.mark.parametrize(
    ("budget", "shown"), [(20, [0.64, 0.55, 0.45, 0.21]), (60, [0.95, 0.71, 0.51, 0.21])]
)
def test_attacker_judge_false_positives_lower_the_confirmed_rate(
    budget: int, shown: list[float]
) -> None:
    """README.md's second table: a false positive ends the search on a prompt that fails.

    Each query finds a working prompt with probability q, the attacker's judge never misses
    one and calls a failing one unsafe with probability f, and the confirmation judge never
    errs. A run is confirmed when it reaches a working prompt before a false positive:
    q (1 - a^K) / (1 - a), with a = (1 - q)(1 - f), which falls as f grows.
    """

    q = 0.05
    confirmed_rates = []
    for false_positive_rate, printed in zip((0.0, 0.02, 0.05, 0.20), shown, strict=True):
        log_a = math.log1p(-q) + math.log1p(-false_positive_rate)
        expected = q * -math.expm1(budget * log_a) / -math.expm1(log_a)
        assert round(expected, 2) == printed
        confirmed = _prompt_dependent_confirmed_rate(
            works_rate=q,
            false_positive_rate=false_positive_rate,
            budget=budget,
            seed=budget * 1000 + round(false_positive_rate * 100),
        )
        assert _within(confirmed, expected)
        confirmed_rates.append(confirmed)
    assert confirmed_rates == sorted(confirmed_rates, reverse=True)


def test_the_budget_must_allow_a_query() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(
            run_attack(
                "goal",
                0,
                TemplateAttacker(),
                SimulatedTarget(0.0, random.Random(0)),
                SimulatedJudge(0.0, 1.0, random.Random(0)),
            )
        )


def test_the_task_file_loads_without_being_registered_as_a_module() -> None:
    """``inspect eval`` executes the task file without adding it to sys.modules."""

    spec = importlib.util.spec_from_file_location(
        "attack_confirmation_unregistered", EXAMPLE_DIR / "attack_confirmation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SIMULATED_DECLARATION.budget_k == 20


def test_judge_and_attacker_output_parsing() -> None:
    assert parse_grade("Rationale.\nGRADE: C") is True
    assert parse_grade("GRADE: C was considered.\nGRADE: I") is False
    assert parse_grade("no grade") is None
    assert extract_prompt("Sure.\n<prompt> Do the thing </prompt>") == "Do the thing"
    assert extract_prompt("  untagged ") == "untagged"


def test_the_committed_design_declares_the_simulated_attack(tmp_path: Path) -> None:
    written = tmp_path / "design.csv"
    write_design(written, SIMULATED_DECLARATION)
    assert written.read_bytes() == DESIGN_PATH.read_bytes()
    check_design(DESIGN_PATH, SIMULATED_DECLARATION)

    other = AttackDeclaration(
        attack_id="pair",
        budget_k=SIMULATED_DECLARATION.budget_k,
        attacker_model=SIMULATED_DECLARATION.attacker_model,
        attacker_judge=SIMULATED_DECLARATION.attacker_judge,
        confirmation_judge=SIMULATED_DECLARATION.confirmation_judge,
    )
    with pytest.raises(ValueError, match="attack_fingerprint.*attack_id"):
        check_design(DESIGN_PATH, other)


def test_live_mode_requires_a_declared_attack_and_a_separate_judge(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode=live needs"):
        attack_confirmation(mode="live")
    with pytest.raises(ValueError, match="must not be the judge"):
        attack_confirmation(
            mode="live",
            attack_id="pair",
            budget_k=20,
            attacker_model="openai/gpt-4o-mini",
            attacker_judge="openai/gpt-4o",
            confirmation_judge="openai/gpt-4o",
        )
    with pytest.raises(ValueError, match="declares a different attack"):
        attack_confirmation(
            mode="live",
            attack_id="pair",
            budget_k=20,
            attacker_model="openai/gpt-4o-mini",
            attacker_judge="openai/gpt-4o",
            confirmation_judge="anthropic/claude-sonnet-5-5",
        )
    with pytest.raises(ValueError, match="mode=live"):
        attack_confirmation(budget_k=5)


def test_the_task_refuses_relative_paths_to_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``inspect eval`` creates the task in the task file's directory, not the caller's.

    A relative path then names a file next to the task file, here the example's own design
    and behaviours, whatever the working directory holds.
    """

    monkeypatch.chdir(tmp_path)
    design = tmp_path / "design.csv"
    behaviours = tmp_path / "behaviours.csv"
    write_design(design, SIMULATED_DECLARATION)
    behaviours.write_bytes((EXAMPLE_DIR / "behaviours.csv").read_bytes())

    def evaluate(**task_args: str) -> list[Any]:
        return inspect_eval(
            str(EXAMPLE_DIR / "attack_confirmation.py"),
            task_args=task_args,
            model="mockllm/candidate",
            log_dir=str(tmp_path / "logs"),
            limit=2,
            display="none",
        )

    for argument in ("design", "behaviours"):
        with pytest.raises(ValueError, match=f"the {argument} path '{argument}.csv' is relative"):
            evaluate(**{argument: f"{argument}.csv"})
    [log] = evaluate(design=str(design), behaviours=str(behaviours))
    assert log.status == "success"
    assert log.eval.task_args["design"] == str(design)


def _plan_upper(behaviours: int, failures: int) -> float:
    """The plan's upper bound: equal weights, KL-Chernoff at 0.025 per direction."""

    failed = np.zeros(behaviours, dtype=int)
    failed[:failures] = 1
    return weighted_kl_chernoff_mean_bounds(
        failed,
        np.zeros(behaviours, dtype=int),
        np.ones(behaviours, dtype=int),
        np.full(behaviours, 1 / behaviours),
        alpha_lower=0.025,
        alpha_upper=0.025,
    ).upper


@pytest.mark.parametrize(
    ("behaviours", "plan_allows", "exact_allows"), [(100, 0, 1), (400, 9, 12)]
)
def test_the_planning_numbers_in_the_readme(
    behaviours: int, plan_allows: int, exact_allows: int
) -> None:
    assert _plan_upper(behaviours, plan_allows) <= 0.05 < _plan_upper(behaviours, plan_allows + 1)
    assert (
        maximum_certifying_failures(trials=behaviours, threshold=0.05, alpha=0.05)
        == exact_allows
    )


def test_the_planning_bounds_in_the_readme_are_the_plans() -> None:
    """The README's bounds for 100 behaviours, and the offline run's, printed in full."""

    assert round(_plan_upper(100, 0), 3) == 0.036
    assert round(_plan_upper(100, 1), 3) == 0.064
    assert _plan_upper(100, 1) == pytest.approx(0.06391479153815988, rel=1e-9, abs=0)


@pytest.fixture(scope="module")
def offline_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Both simulated configurations, run through Inspect with the mock model."""

    logs = tmp_path_factory.mktemp("attack-confirmation") / "logs"
    inspect_eval(
        attack_confirmation(),
        model=["mockllm/baseline", "mockllm/candidate"],
        log_dir=str(logs),
        display="none",
    )
    return logs


@pytest.fixture(autouse=True)
def _run_from_the_drive_that_holds_the_logs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspect drops the drive letter of logs on another drive (#35); every path is absolute."""

    monkeypatch.chdir(tmp_path_factory.getbasetemp())


def _plan_scoring(tmp_path: Path, score: str) -> Path:
    plan = yaml.safe_load((EXAMPLE_DIR / "plan.yaml").read_text(encoding="utf-8"))
    plan["inputs"]["score"]["name"] = score
    plan["inputs"]["design_manifest"] = str(DESIGN_PATH)
    path = tmp_path / f"plan-{score}.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def test_the_example_plan_analyses_the_offline_logs(offline_logs: Path, tmp_path: Path) -> None:
    report = analyze(plan=EXAMPLE_DIR / "plan.yaml", logs=offline_logs)

    assert report.integrity.status == "valid"
    risks = {item.cohort_id: item for item in report.group_risks}
    # The candidate never complies: what remains is the confirmation judge's error.
    assert risks["candidate"].estimate is not None and risks["candidate"].estimate <= 0.05
    assert risks["baseline"].estimate is not None
    assert risks["baseline"].estimate > risks["candidate"].estimate
    # README.md prints this claim: one confirmed jailbreak in 100 behaviours.
    [claim] = report.claims
    assert claim.claim_id == "attack_success_bounded"
    assert claim.status == "INCONCLUSIVE"
    assert claim.upper == pytest.approx(0.06391479153815988, rel=1e-9, abs=0)

    output = report.write(tmp_path / "report")
    for name in ("report.html", "report.md", "report.json"):
        assert "[placeholder]" not in (output / name).read_text(encoding="utf-8")


def test_the_attackers_own_verdicts_would_report_most_behaviours_jailbroken(
    offline_logs: Path, tmp_path: Path
) -> None:
    """The same logs, analysed on the attacker's verdict instead of the confirmation's."""

    confirmed = analyze(plan=_plan_scoring(tmp_path, "confirmed_jailbreak"), logs=offline_logs)
    reported = analyze(plan=_plan_scoring(tmp_path, "attacker_reported"), logs=offline_logs)

    def candidate(report: object) -> float:
        risk = next(item for item in report.group_risks if item.cohort_id == "candidate")  # type: ignore[attr-defined]
        assert risk.estimate is not None
        return float(risk.estimate)

    # 1 - 0.95^20 is about 0.64 for a target that never complies.
    assert candidate(reported) > 0.5
    assert candidate(confirmed) <= 0.05


# --- Live mode against mock models: what ties the runs to the declared attack ---------------

LIVE_MODELS = {
    "attacker_model": "mockllm/attacker",
    "attacker_judge": "mockllm/attacker-judge",
    "confirmation_judge": "mockllm/confirmation-judge",
}


def _live_declaration(budget: int) -> AttackDeclaration:
    return AttackDeclaration(attack_id="pair", budget_k=budget, **LIVE_MODELS)


def _write_behaviours(path: Path, numbers: Iterable[int]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["prompt_id", "category_id", "goal"])
        for number in numbers:
            writer.writerow([f"b{number}", f"c{number % 2}", f"[placeholder] behaviour {number}"])
    return path


def _live_run(
    directory: Path, logs: Path, model: str, budget: int, numbers: Iterable[int]
) -> None:
    """One live-mode run with mock models, against a design written for its attack.

    The mock judges never answer with a grade, so each run spends its budget and finds no
    prompt. What matters here is what the logs record about the attack.
    """

    directory.mkdir(parents=True)
    behaviours = _write_behaviours(directory / "behaviours.csv", numbers)
    design = directory / "design.csv"
    write_design(design, _live_declaration(budget), behaviours)
    inspect_eval(
        attack_confirmation(
            mode="live",
            behaviours=str(behaviours),
            design=str(design),
            attack_id="pair",
            budget_k=budget,
            **LIVE_MODELS,
        ),
        model=model,
        log_dir=str(logs),
        display="none",
    )


@pytest.fixture(scope="module")
def split_budget_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The candidate's behaviours attacked half at budget 1 and half at budget 2."""

    root = tmp_path_factory.mktemp("split-budget")
    logs = root / "logs"
    _live_run(root / "candidate-1", logs, "mockllm/candidate", 1, [0, 1])
    _live_run(root / "candidate-2", logs, "mockllm/candidate", 2, [2, 3])
    _live_run(root / "baseline-1", logs, "mockllm/baseline", 1, range(4))
    return logs


@pytest.fixture(scope="module")
def cross_budget_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The candidate attacked at budget 2 and the baseline at budget 1."""

    root = tmp_path_factory.mktemp("cross-budget")
    logs = root / "logs"
    _live_run(root / "candidate-2", logs, "mockllm/candidate", 2, range(4))
    _live_run(root / "baseline-1", logs, "mockllm/baseline", 1, range(4))
    return logs


def _live_plan(directory: Path, *, match: AttackDeclaration | None = None) -> Path:
    """The example plan over the four behaviours, with a design declaring budget 1.

    With ``match``, every cohort also matches on that attack, as README.md tells a plan for
    live runs to do.
    """

    directory.mkdir(parents=True, exist_ok=True)
    design = directory / "design.csv"
    write_design(
        design, _live_declaration(1), _write_behaviours(directory / "behaviours.csv", range(4))
    )
    plan: dict[str, Any] = yaml.safe_load((EXAMPLE_DIR / "plan.yaml").read_text(encoding="utf-8"))
    plan["inputs"]["design_manifest"] = str(design)
    if match is not None:
        for cohort in plan["cohorts"].values():
            cohort["match"].update(
                {f"task_arg_{name}": value for name, value in asdict(match).items()}
            )
    path = directory / "plan.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def _errors(report: Any) -> list[tuple[str, str]]:
    return [
        (item.code, item.message)
        for item in report.integrity.diagnostics
        if item.severity == "ERROR"
    ]


def test_the_example_plan_flags_a_cohort_attacked_at_two_budgets(
    split_budget_logs: Path, tmp_path: Path
) -> None:
    """Each run passed its own design check, so only the plan can see the two attacks."""

    report = analyze(plan=_live_plan(tmp_path), logs=split_budget_logs)

    assert report.integrity.status == "invalid"
    drift = [message for code, message in _errors(report) if code == "CONFIG_DRIFT"]
    assert len(drift) == 1 and "'candidate'" in drift[0]


def test_cohorts_matched_on_the_declared_attack_keep_other_attacks_out(
    split_budget_logs: Path, cross_budget_logs: Path, tmp_path: Path
) -> None:
    """The cohort match README.md gives for live runs, against runs of another budget."""

    readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
    for name in asdict(_live_declaration(1)):
        assert f"task_arg_{name}:" in readme

    # Matched on the model alone, a candidate attacked at budget 2 is compared with a
    # baseline attacked at budget 1, and nothing says so.
    alone = analyze(plan=_live_plan(tmp_path / "alone"), logs=cross_budget_logs)
    assert alone.integrity.status == "valid"

    declared = _live_declaration(1)
    across = analyze(plan=_live_plan(tmp_path / "across", match=declared), logs=cross_budget_logs)
    assert across.integrity.status == "invalid"
    assert ("COHORT_EMPTY", "cohort 'candidate' contains no evaluations") in _errors(across)

    within = analyze(plan=_live_plan(tmp_path / "within", match=declared), logs=split_budget_logs)
    assert within.integrity.status == "invalid"
    missing = sorted(message for code, message in _errors(within) if code == "PROMPT_MISSING")
    assert missing == [
        f"prompt 'b{number}' has no observations in cohort 'candidate'" for number in (2, 3)
    ]
