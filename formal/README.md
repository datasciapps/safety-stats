# Formal proofs

Lean 4 proofs, using Mathlib, of statements that the package's statistical guarantees rest on.
They are optional. Nothing in `src/` or the Python test suite needs them, and no Python
contributor needs Lean installed.

## What is proved

`SafetyStatsFormal/RiskMeasures.lean` covers the risk measures in
`src/safety_stats/stats/risk.py`. The package bounds a group's risk by pushing per-prompt
bounds through a measure, which is valid only because every measure is coordinatewise
nondecreasing.

| Python | Lean definition | Monotonicity theorem |
|---|---|---|
| `weighted_mean` | `weightedMean` | `weightedMean_mono` |
| `maximum_risk` | `maximumRisk` | `maximumRisk_mono` |
| `threshold_exceedance` | `thresholdExceedance` | `thresholdExceedance_mono` |
| `threshold_excess` | `thresholdExcess` | `thresholdExcess_mono` |
| `entropic_risk` | `entropicRisk` | `entropicRisk_mono` |
| `worst_of_k_risk` | `worstOfK` | `worstOfK_mono` |
| `weighted_cvar` | `cvar` (Rockafellar–Uryasev form) | `cvar_mono` |
| `top_k_mean` | `topKMean` (CVaR with equal weights) | `topKMean_mono` |

The file also proves two supporting facts:
- CVaR is at least the weighted mean: `weightedMean_le_cvarObjective`.
- One item of `worst_of_k` lies between a single attempt and the union bound: `worstOfK_item_sandwich`, via Bernoulli's inequality `one_sub_mul_le_pow`.

## How the proofs connect to the code

A proof covers the mathematics. A property test in `tests/property/test_risk_properties.py`
checks that the implementation computes it. Each theorem's docstring names that test, and
`tests/test_repository_consistency.py` fails if a named test disappears.

Two measures are proved in a different but equal form, and a property test checks each equality:

| Python computes | Lean proves | Property test for the equality |
|---|---|---|
| greedy CVaR, the worst `1 − β` of the weight | Rockafellar–Uryasev form | `test_weighted_cvar_equals_its_rockafellar_uryasev_form` |
| top-`k` mean by partition | CVaR at `β = 1 − k/n` with equal weights | `test_top_k_mean_is_cvar_with_equal_weights` |

The proofs are about real numbers. Floating-point rounding is outside their scope. For example,
`worst_of_k_risk` evaluates `1 − (1 − p)^k` as `-expm1(k·log1p(-p))`.

## Building

CI builds this on every change to `formal/`, weekly and on demand
(`.github/workflows/formal.yml`). The build fails if any theorem depends on `sorry`. To build
locally, install [elan](https://github.com/leanprover/elan), then:

```bash
cd formal
lake exe cache get   # Mathlib's prebuilt files: about 5 GB
lake build
```

`lean-toolchain` and `lakefile.toml` pin Lean and Mathlib to the same release. Update both
together, in a pull request of its own.
