"""Small, offline example of the public statistical API."""

from inspect_safety_stats.stats import (
    binomial_upper_bound,
    entropic_risk,
    weighted_cvar,
    zero_failure_sample_size,
)


def main() -> None:
    rollouts = zero_failure_sample_size(threshold=0.01, alpha=0.05)
    upper = binomial_upper_bound(0, rollouts, alpha=0.05)

    values = [0.001, 0.02, 0.20]
    weights = [0.5, 0.3, 0.2]
    cvar = weighted_cvar(values, weights, beta=0.8)
    entropic = entropic_risk(values, weights, lambda_=10)

    print(f"Zero-failure plan: n={rollouts}, upper={upper:.6f}")
    print(f"Fixed-design CVaR: {cvar:.6f}")
    print(f"Fixed-design entropic risk: {entropic:.6f}")


if __name__ == "__main__":
    main()
