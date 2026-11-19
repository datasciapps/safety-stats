# Index: Eight real SimpleQA items as a fixed-design failure analysis

`report.md` is the one-page summary for people. Pointers such as `/group_risks/0` are JSON pointers into `report.json`. How to read this index: <https://github.com/aduncan001/safety_stats/blob/main/docs/agent-guide.md>.

Analysis `real-simpleqa-eight` (confirmatory), plan SHA-256 `fd9ee06715157246e1e6d90d86f2cada7d7cffe96ee9fd7e280c8d026a0f654c`, inspect-safety-stats 0.15.0, report schema 3.0, index version 1.

Data integrity: **valid** (0 errors, 2 warnings, 0 informational).

Decisions are hidden (`report.include_decisions` is false); `report.json` still holds them at `/decision` and `/claims`.

## Headline estimates

| Kind | Model | Baseline | Group | Measure | Estimand | Estimate | Lower | Upper | Eligibility | Pointer | CSV row |
|---|---|---|---|---|---|---:|---:|---:|---|---|---|
| group_risk | candidate | — | all-eight | mean | — | 0.5000 | 0.0920 | 0.9080 | certifying | `/group_risks/2` | — |

## Errors and warnings by code

| Code | Severity | Count | Effect | Diagnostics | Items |
|---|---|---:|---|---|---|
| `PLAN_SELF_DECLARED` | warning | 1 | recorded | `/integrity/diagnostics/0` | — |
| `TAIL_MEASURES_UNINFORMATIVE` | warning | 1 | recorded | `/integrity/diagnostics/1` | — |

What each code assumes and what follows is in `index.json` and in `report.md`; [the diagnostics reference](https://github.com/aduncan001/safety_stats/blob/main/docs/diagnostics.md) gives the remedy.

## Assumptions used

`fixed_design`, `run_stability`, `rollout_independence`, `scorer_validity`, `judge_accuracy`, `missing_scores`, `sample_errors`, `token_limit`, `content_filter_stops`, `complete_design`, `plan_fixed_in_advance`, in order at `/assumptions_used/<i>`.

## Files

| File | Format | Holds | Data row n is |
|---|---|---|---|
| `checksums.json` | json | SHA-256 of every other file. | — |
| `comparison-sets.csv` | csv | One row per comparison set. | `/comparison_sets/<n-1>` |
| `comparisons.csv` | csv | One row per comparison result. | `/comparisons/<n-1>` |
| `design.resolved.csv` | csv | One row per design prompt. | `/design/prompts/<n-1>` |
| `hierarchy-risks.csv` | csv | One row per hierarchy node risk. | `/hierarchy_risks/<n-1>` |
| `index.json` | json | This index. | — |
| `index.md` | markdown | This index as Markdown. | — |
| `observations.csv` | csv | One row per normalised observation (analysis input). | — |
| `plan.resolved.yaml` | yaml | The plan as analysed, defaults filled in. | — |
| `report-full.md` | markdown | The full report: every interval, sensitivity and message. | — |
| `report.html` | html | Interactive report; embeds report.json. | — |
| `report.json` | json | Every result, diagnostic and setting; the source of all others. | — |
| `report.md` | markdown | One-page summary for people: read first. | — |
