# Index: Published GPT-4 behavior on four XSTest contrast pairs

`report.md` is the one-page summary for people. Pointers such as `/group_risks/0` are JSON pointers into `report.json`. How to read this index: <https://github.com/aduncan001/safety_stats/blob/main/docs/agent-guide.md>.

Analysis `real-xstest-published-gpt4-replay` (exploratory), plan SHA-256 `6aa8818abe40e749c05e70868542753f738401ec49f49317fb8bcba8e5e8081c`, inspect-safety-stats 0.15.0, report schema 3.0, index version 1.

Data integrity: **valid** (0 errors, 1 warnings, 0 informational).

Decisions are hidden (`report.include_decisions` is false); `report.json` still holds them at `/decision` and `/claims`.

## Headline estimates

| Kind | Model | Baseline | Group | Measure | Estimand | Estimate | Lower | Upper | Eligibility | Pointer | CSV row |
|---|---|---|---|---|---|---:|---:|---:|---|---|---|
| group_risk | published_gpt4_replay | — | all-pairs | mean | — | 0.1250 | 0.0002 | 0.6865 | certifying | `/group_risks/2` | — |
| group_risk | published_gpt4_replay | — | safe-prompts | mean | — | 0.0000 | 0.0000 | 0.7460 | certifying | `/group_risks/6` | — |
| group_risk | published_gpt4_replay | — | unsafe-prompts | mean | — | 0.2500 | 0.0004 | 0.9219 | certifying | `/group_risks/10` | — |

## Errors and warnings by code

| Code | Severity | Count | Effect | Diagnostics | Items |
|---|---|---:|---|---|---|
| `TAIL_MEASURES_UNINFORMATIVE` | warning | 1 | recorded | `/integrity/diagnostics/0` | — |

What each code assumes and what follows is in `index.json` and in `report.md`; [the diagnostics reference](https://github.com/aduncan001/safety_stats/blob/main/docs/diagnostics.md) gives the remedy.

## Assumptions used

`fixed_design`, `run_stability`, `rollout_independence`, `scorer_validity`, `judge_accuracy`, `missing_scores`, `sample_errors`, `token_limit`, `content_filter_stops`, `complete_design`, in order at `/assumptions_used/<i>`.

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
