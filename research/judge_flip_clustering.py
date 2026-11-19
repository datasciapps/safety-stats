"""Coverage of the judge flip interval for the process rate when flips cluster by prompt.

Under ``sampling: all`` the flip rate's Clopper-Pearson interval treats the rescored
completions as independent. As a range for the flip rate of the process that produces prompts
and their judge-positive completions, it holds only if they are. Here each replicate draws
``prompts`` prompts; a prompt's flip probability is Beta with mean 0.5 and intra-prompt
correlation ``rho``, and each of its completions flips independently with that probability.
Each row is the share of replicates whose 0.95 interval covers 0.5. Nominal coverage needs
``rho`` near 0 or one completion per prompt.

    python research/judge_flip_clustering.py
"""

from __future__ import annotations

import numpy as np

from inspect_safety_stats.stats.judge_flip import judge_flip_rate

MEAN, ALPHA, REPLICATIONS = 0.5, 0.05, 4000
DESIGNS = ((5, 7), (20, 2), (36, 1))
CORRELATIONS = (0.0, 0.1, 0.3, 0.6)


def coverage(prompts: int, per_prompt: int, rho: float, rng: np.random.Generator) -> float:
    ids = [f"{prompt}-{index}" for prompt in range(prompts) for index in range(per_prompt)]
    covered = 0
    for _ in range(REPLICATIONS):
        if rho == 0:
            probability = np.full(prompts, MEAN)
        else:
            scale = (1 - rho) / rho
            probability = rng.beta(MEAN * scale, (1 - MEAN) * scale, size=prompts)
        flips = rng.random((prompts, per_prompt)) < probability[:, None]
        verdicts = {
            f"{prompt}-{index}": {"w": not bool(flips[prompt, index])}
            for prompt in range(prompts)
            for index in range(per_prompt)
        }
        rate = judge_flip_rate(ids, verdicts, ("w",), sampling="all", alpha=ALPHA).any_wrapper
        if rate is None:
            raise ValueError("every replicate rescores at least one completion")
        covered += rate.lower <= MEAN <= rate.upper
    return covered / REPLICATIONS


def main() -> None:
    rng = np.random.default_rng(20261119)
    for prompts, per_prompt in DESIGNS:
        for rho in CORRELATIONS:
            found = coverage(prompts, per_prompt, rho, rng)
            print(f"{prompts:>2} prompts x {per_prompt} completions, rho {rho}: {found:.3f}")


if __name__ == "__main__":
    main()
