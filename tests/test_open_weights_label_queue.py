import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

from inspect_safety_stats.ingest.validation import load_validation_manifest

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "open_weights"
sys.path.insert(0, str(EXAMPLE_DIR))

import label_queue  # noqa: E402

STATUS = {True: "classified_failure", False: "classified_safe"}


def _reports(tmp_path: Path, seed: int = 7) -> tuple[Path, Path, pd.Series]:
    """A primary and a second judge's observations, and a hidden truth per rollout.

    Misses concentrate where the judges disagree, so a design that favoured those rollouts
    within a judge stratum would overestimate the stratum's error rate.
    """

    rng = np.random.default_rng(seed)
    rows, other, truth = [], [], {}
    for cohort in ("a", "b"):
        for task in ("harm", "refusal"):
            for prompt in range(30):
                rate = rng.choice([0.0, 0.0, 0.0, 0.05, 0.4, 0.8])
                for epoch in range(1, 11):
                    key = f"{cohort}-{task}-{prompt}-{epoch}"
                    primary = bool(rng.random() < rate)
                    second = primary if rng.random() < 0.9 else not primary
                    if primary == second:
                        flip = 0.1 if primary else 0.02
                    else:
                        flip = 0.4
                    identity = {
                        "observation_id": f"id-{key}",
                        "eval_id": f"eval-{cohort}",
                        "sample_id": f"sample-{key}",
                        "epoch": str(epoch),
                        "inspect_id": f"p{prompt}",
                        "prompt_id": f"{task}-p{prompt}",
                        "cohort_id": cohort,
                        "task_id": task,
                        "source_log": f"/logs/{cohort}.eval",
                    }
                    rows.append({**identity, "outcome_status": STATUS[primary]})
                    other.append({**identity, "outcome_status": STATUS[second]})
                    truth[identity["observation_id"]] = primary != bool(rng.random() < flip)
    rows[0]["outcome_status"] = "unclassified"
    paths = []
    for name, frame in (("primary", rows), ("second", other)):
        folder = tmp_path / name
        folder.mkdir()
        pd.DataFrame(frame).to_csv(folder / "observations.csv", index=False)
        (folder / "plan.resolved.yaml").write_text("groups: []\nclaims: []\n", encoding="utf-8")
        paths.append(folder)
    return paths[0], paths[1], pd.Series(truth)


def _frame(primary: Path, second: Path) -> pd.DataFrame:
    judges = {"second": (label_queue.verdicts(second), ())}
    return label_queue.rollouts(label_queue.verdicts(primary), judges)


def _build(frame: pd.DataFrame, seed: int, size: int = 240) -> tuple[pd.DataFrame, dict]:
    return label_queue.build_queue(
        frame,
        seed=seed,
        size=size,
        oversample=0.5,
        floor=2,
        calibration_floor=3,
        thresholds={("a", "harm"): 0.05},
    )


def test_strata_come_from_verdicts_and_the_prompts_other_rollouts(tmp_path: Path) -> None:
    primary, second, _ = _reports(tmp_path)
    frame = _frame(primary, second)

    assert len(frame) == 1199  # the unclassified rollout is left out
    row = frame.iloc[0]
    assert row["judge_stratum"] == f"{row['cohort_id']}/{row['task_id']}/{row['verdict']}"
    mixed = "mixed" if row["prompt_mixed"] else "steady"
    assert row["fine_stratum"] == f"{row['judge_stratum']}/second={row['by_second']}/{mixed}"
    assert frame["flagged"].equals((frame["by_second"] != frame["verdict"]) | frame["prompt_mixed"])
    # Leave-one-out: a rollout's own verdict never counts towards its prompt's rate.
    for _, rollouts in frame.groupby(["cohort_id", "task_id", "prompt_id"]):
        failures = (rollouts["verdict"] == "unsafe").sum()
        own = (rollouts["verdict"] == "unsafe").astype(int)
        expected = (failures - own) / (len(rollouts) - 1)
        assert rollouts["prompt_rate"].to_numpy() == pytest.approx(expected.to_numpy(), abs=0)

    limited = label_queue.rollouts(
        label_queue.verdicts(primary), {"second": (label_queue.verdicts(second), ("harm",))}
    )
    assert set(limited.loc[limited["task_id"] == "refusal", "by_second"]) == {"-"}
    assert set(limited.loc[limited["task_id"] == "harm", "by_second"]) == {"safe", "unsafe"}


def test_every_prefix_holds_a_fixed_random_sample_of_each_judge_stratum(tmp_path: Path) -> None:
    primary, second, _ = _reports(tmp_path)
    frame = _frame(primary, second)
    first, design = _build(frame, seed=1)
    other, _ = _build(frame, seed=2)

    assert len(first) == 240 and not first["observation_id"].duplicated().any()
    assert set(first["tier"]) == {label_queue.CALIBRATION, label_queue.OVERSAMPLE}
    # Which judge stratum each calibration slot serves is fixed before any permutation.
    for queue in (first, other):
        assert queue["tier"].tolist() == first["tier"].tolist()
    calibration = first[first["tier"] == label_queue.CALIBRATION]
    assert (
        calibration["judge_stratum"].tolist()
        == other.loc[other["tier"] == label_queue.CALIBRATION, "judge_stratum"].tolist()
    )
    assert calibration["observation_id"].tolist() != other.loc[
        other["tier"] == label_queue.CALIBRATION, "observation_id"
    ].tolist()
    # Calibration rows take each stratum's permutation from the front, oversampling rows only
    # from beyond what the calibration tier reaches.
    for key, rows in first.groupby("judge_stratum"):
        reach = design["judge_strata"][key]["calibration"]
        positions = rows["order_position"].astype(int)
        taken = positions[rows["tier"] == label_queue.CALIBRATION].tolist()
        assert taken == list(range(1, reach + 1))
        assert (positions[rows["tier"] == label_queue.OVERSAMPLE] > reach).all()
    # Flagged fine strata get their floor before Neyman allocation takes over.
    fine = pd.DataFrame.from_dict(design["fine_strata"], orient="index")
    floor = np.minimum(2, fine["population"] - fine["calibration_reach"])
    assert (fine.loc[fine["flagged"], "oversampled"] >= floor[fine["flagged"]]).all()


def test_every_judge_stratum_gets_its_calibration_floor_first() -> None:
    strata = pd.DataFrame(
        {
            "prior": [0.02, 0.4, 0.3],
            "weight": [0.97, 0.025, 0.005],
            "leverage": [2.0, 1.0, 1.0],
            "population": [970, 25, 2],
        },
        index=["big/safe", "big/unsafe", "tiny/unsafe"],
    )

    without = label_queue.allocate_calibration(strata, 40, minimum=0)
    order = label_queue.allocate_calibration(strata, 40, minimum=3)

    # Left to the widths alone, the small strata get nothing the package can use.
    assert without.count("tiny/unsafe") == 0
    # With the floor, the first slots go round the strata, a stratum smaller than the floor
    # is labelled in full, and the rest follows the widths.
    assert order[:8] == ["big/safe", "big/unsafe", "tiny/unsafe"] * 2 + ["big/safe", "big/unsafe"]
    assert order.count("tiny/unsafe") == 2 and order.count("big/unsafe") >= 3
    assert label_queue.allocate_calibration(strata, 5, minimum=3) == order[:5]
    assert label_queue.allocate_calibration(strata, 1000, minimum=3)[-1] is None


def test_labels_estimate_without_bias_although_disagreement_is_oversampled(
    tmp_path: Path,
) -> None:
    primary, second, truth = _reports(tmp_path)
    frame = _frame(primary, second)
    frame["truth"] = frame["observation_id"].map(truth)
    frame["disagree"] = frame["by_second"] != frame["verdict"]
    disagree_share, calibration_means, estimates = [], [], []
    for seed in range(60):
        queue, design = _build(frame, seed=seed, size=160)
        rows = queue.merge(frame[["observation_id", "truth", "disagree"]], on="observation_id")
        disagree_share.append(rows["disagree"].mean())
        safe = rows[(rows["judge_stratum"] == "a/harm/safe") & (rows["tier"] == "calibration")]
        calibration_means.append(safe["truth"].mean())
        prefix = queue.assign(human=np.where(rows["truth"], "unsafe", "safe"))
        cell = label_queue.summarize(prefix, design, alpha=0.05)["cells"][0]
        assert (cell["cohort_id"], cell["task_id"]) == ("a", "harm")
        estimates.append(cell["corrected_estimate"])
    population = frame[frame["judge_stratum"] == "a/harm/safe"]["truth"].mean()
    cell_truth = frame[(frame["cohort_id"] == "a") & (frame["task_id"] == "harm")]["truth"].mean()

    assert np.mean(disagree_share) > 2 * frame["disagree"].mean()
    standard_error = np.std(calibration_means) / np.sqrt(len(calibration_means))
    assert abs(np.mean(calibration_means) - population) < 4 * standard_error
    standard_error = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - cell_truth) < 4 * standard_error


def test_hypergeometric_bounds_are_exact_for_a_census_and_cover_otherwise() -> None:
    assert label_queue.hypergeometric_bounds(50, 0, 0, 0.05) == (0.0, 1.0)
    assert label_queue.hypergeometric_bounds(50, 50, 7, 0.05) == (7 / 50, 7 / 50)
    population, sample, alpha = 30, 9, 0.1
    for unsafe in range(population + 1):
        draws = np.arange(sample + 1)
        bounds = [label_queue.hypergeometric_bounds(population, sample, x, alpha) for x in draws]
        chance = hypergeom.pmf(draws, population, unsafe, sample)
        share = unsafe / population
        above = sum(p for p, (_, high) in zip(chance, bounds, strict=True) if high < share)
        below = sum(p for p, (low, _) in zip(chance, bounds, strict=True) if low > share)
        assert above <= alpha + 1e-12 and below <= alpha + 1e-12


def test_a_census_of_a_cell_gives_its_exact_share_and_each_judges_errors(tmp_path: Path) -> None:
    primary, second, truth = _reports(tmp_path)
    frame = _frame(primary, second)
    cell = frame[(frame["cohort_id"] == "b") & (frame["task_id"] == "refusal")]
    queue, design = _build(cell.reset_index(drop=True), seed=3, size=len(cell))
    prefix = queue.assign(human=np.where(queue["observation_id"].map(truth), "unsafe", "safe"))
    summary = label_queue.summarize(prefix, design, alpha=0.05)

    result = summary["cells"][0]
    share = cell["observation_id"].map(truth).mean()
    assert result["corrected_lower"] == pytest.approx(share, abs=1e-12)
    assert result["corrected_upper"] == pytest.approx(share, abs=1e-12)
    assert result["corrected_estimate"] == pytest.approx(share, abs=1e-12)
    said = cell["by_second"] == "unsafe"
    real = cell["observation_id"].map(truth).to_numpy()
    assert result["judges"]["second"]["false_unsafe_rate"] == pytest.approx(
        (said & ~real).sum() / (~real).sum(), abs=1e-12
    )
    assert result["judges"]["second"]["false_safe_rate"] == pytest.approx(
        (~said & real).sum() / real.sum(), abs=1e-12
    )
    assert "| b | refusal |" in label_queue.summary_markdown(summary)


def _answers(queue: pd.DataFrame, answered: int) -> pd.DataFrame:
    answers = ["yes" if i % 3 == 0 else "no" for i in range(answered)]
    return pd.DataFrame(
        {
            "observation_id": queue["observation_id"],
            "answer": answers + [""] * (len(queue) - answered),
        }
    )


def test_manifests_hold_the_labelled_calibration_rows_and_refuse_gaps(tmp_path: Path) -> None:
    primary, second, _ = _reports(tmp_path)
    output = tmp_path / "queue"
    label_queue.main(
        [
            "queue",
            "--primary",
            str(primary),
            "--judge",
            f"second={second}",
            "--seed",
            "11",
            "--size",
            "120",
            "--output",
            str(output),
        ]
    )
    again = ["queue", "--primary", str(primary), "--seed", "1", "--size", "9"]
    with pytest.raises(SystemExit, match="must not change"):
        label_queue.main([*again, "--output", str(output)])
    queue = pd.read_csv(output / "queue.csv", dtype=str, keep_default_na=False)
    design = json.loads((output / "queue.json").read_text(encoding="utf-8"))
    assert design["seed"] == 11 and design["rows"] == 120
    assert design["inputs"]["judges"][0]["name"] == "second"

    sheet = tmp_path / "labels.csv"
    _answers(queue, 40).to_csv(sheet, index=False)
    manifests = tmp_path / "manifests"
    label_queue.main(
        ["manifest", "--queue", str(output), "--labels", str(sheet), "--output", str(manifests)]
    )
    expected = queue.iloc[:40]
    expected = expected[expected["tier"] == label_queue.CALIBRATION]
    written = {}
    for task in ("harm", "refusal"):
        written.update(load_validation_manifest(manifests / f"judge-labels-{task}.csv"))
    assert sorted(written) == sorted(expected["observation_id"])
    index = {key: i for i, key in enumerate(queue["observation_id"])}
    assert all(unsafe == (index[key] % 3 == 0) for key, unsafe in written.items())
    summary = json.loads((manifests / "summary.json").read_text(encoding="utf-8"))
    assert summary["rows_labelled"] == 40

    gap = _answers(queue, 40)
    gap.loc[5, "answer"] = ""
    gap.to_csv(sheet, index=False)
    with pytest.raises(SystemExit, match="row 6 is blank"):
        label_queue.main(
            ["manifest", "--queue", str(output), "--labels", str(sheet), "--output", str(manifests)]
        )
    unsure = _answers(queue, 40)
    unsure.loc[40, "answer"] = "unsure"
    unsure.to_csv(sheet, index=False)
    with pytest.raises(SystemExit, match="row 41 is unsure"):
        label_queue.main(
            ["manifest", "--queue", str(output), "--labels", str(sheet), "--output", str(manifests)]
        )
    unknown = _answers(queue, 40)
    unknown.loc[3, "answer"] = "maybe"
    unknown.to_csv(sheet, index=False)
    with pytest.raises(SystemExit, match="'maybe'"):
        label_queue.main(
            ["manifest", "--queue", str(output), "--labels", str(sheet), "--output", str(manifests)]
        )


@pytest.mark.parametrize("text", ["second", "=report", "bad name=report", "x="])
def test_judge_specs_are_checked(text: str) -> None:
    with pytest.raises(SystemExit):
        label_queue.arguments(
            ["queue", "--primary", "p", "--judge", text, "--seed", "1", "--size", "1"]
            + ["--output", "o"]
        )
