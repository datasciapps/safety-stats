# Changelog

## 0.16.0

The first release candidate. It narrows the package to fixed-design analysis of finished
Inspect logs and makes the common path short: `init`, `analyze`, read one page.

### Added

- `inspect-safety-stats init` writes a starter design manifest and plan from completed logs:
  one cohort per model, the weighted-mean failure risk with a KL/Chernoff interval and a
  prompt-level sensitivity, an optional one-level breakdown (`--group-by`) and a comparison
  set. The score mapping is never inferred, and the plan is exploratory by construction.
- `report.md` is now a one-page summary: failure risk per model, comparisons with a plain
  reading, a one-level breakdown, judge-corrected true risk when configured, claims when
  decisions are enabled, and a **Precision** section. The previous document is
  `report-full.md`.
- Imprecise intervals are marked against `report.precision_target` (default 0.05) and listed
  with the rollouts that would resolve them.
- `inspect-safety-stats judge sample` draws the human-labelling sample for validated judge
  calibration: a seeded simple random sample within each judge stratum of each cohort, written
  as a blinded, shuffled `labels.csv` with a `labels.sampling.json` audit record.
- `inspect-safety-stats plan judge-labels` sizes that sample for a target true-risk bound.
- `inspect-safety-stats migrate-plan` upgrades a schema `1.0` plan to `2.0` in a new file,
  listing every change.
- `TAIL_MEASURES_UNINFORMATIVE` warns when tail or threshold measures are requested with too
  few rollouts per prompt to be informative.
- Planning functions `rollouts_for_half_width`, `rollout_scale_for_half_width`,
  `exact_half_width` and `judge_label_plan`; `attempt_risk_estimate` and
  `attempt_risk_from_rate` in `inspect_safety_stats.stats`.
- Graded scores: `score.mapping.scale` maps a rubric grade or probability onto harm in [0, 1];
  every bound becomes a bound on mean harm, prompt-level bounds use Bernoulli-KL inversion, and
  `init --harm-scale MIN MAX [--lower-is-harmful]` writes the mapping. Judge calibration is
  refused for graded scales.
- `k_attempt_risk`: the probability that at least one of `k` attempts at a prompt fails (the
  expected worst outcome for graded scores), estimated by an unbiased U-statistic and bounded
  through Hoeffding's U-statistic decomposition with the KL/Chernoff and Hoeffding engines, in a
  dedicated Bonferroni family.
- `inspect-safety-stats plan comparison`: rollouts per arm for the two-arm rectangle to resolve
  a difference or establish non-inferiority at a margin, from exact power; also
  `comparison_power`, `comparison_rollouts` and `kl_rate_bounds`.
- `analyze` prints the headline rate for each model.
- [Confirmatory mode](docs/confirmatory-mode.md) documents claims, decisions, plan freezing
  and exit codes in one place.

### Changed

- `comparison_sets` accept two cohorts, so a single candidate-versus-baseline comparison no
  longer needs a hand-written `comparisons` block.
- `outputs.direct_mean_intervals.method` defaults to `weighted_kl_chernoff`, the recommended
  primary for sparse binary failures and already the default for comparisons. Plans that
  relied on the empirical-Bernstein default should name it.
- A plan declaring `schema_version: "1.0"` is refused with a pointer to `migrate-plan`; an
  unknown version is refused outright. An unquoted `schema_version: 2.0` is accepted.
- Canonical plan hashes change in this release, because plan defaults changed and fields were
  removed. Re-hash and re-register frozen plans.

### Removed

These remain on `main` and may return in a later release.

- The generative evaluation path (`inspect_safety_stats.generative`, the `generative` command
  group, its examples and research scripts).
- Sampled-frame designs (`design.type: sampled`, Horvitz–Thompson and fixed-size estimation).
- Paired and cluster diagnostics on comparisons (`pair_key`, `diagnostic_methods`,
  `cluster_key`, `bootstrap_*`; exact McNemar, Newcombe, fixed-cluster bootstrap). Given the
  prompt, the two arms' rollouts are independent, so epoch matching paired observations that
  share nothing; declare real cross-arm dependence with a `joint` independence profile.
  Comparison results no longer carry `pair_count`, `discordant_counts`, `p_value`,
  `cluster_count` or `diagnostic_outputs`.
