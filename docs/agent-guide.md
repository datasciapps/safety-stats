# Reading a report as an agent

This page is for agents and other tools that read the report directory `analyze` writes. It
says which file to open first, how pointers work, and what not to conclude. People can start
with `report.md`, the one-page summary.

## Open `index.json` first

Every report directory has `index.json` and `index.md`, the same content in JSON and Markdown.
The index holds:

- `artifacts`: every file in the directory, with its format and what it holds. A CSV whose
  rows are the items of a `report.json` list names that list in `rows_of`: data row *n*
  (counting from 1, after the header) is the item at `rows_of/<n-1>`. For example, row 2 of
  `comparisons.csv` is `/comparisons/1`.
- `analysis`: the analysis id, title, intent and `plan_sha256`; then `package_version` and
  `report_schema_version`.
- `summary`: a copy of the `summary` object in `report.json`, plus a `csv` row for each
  headline that a CSV holds.
- `guide`: the address of this page.

`checksums.json` gives the SHA-256 of every other file, the index included. Check it when the
directory has been copied or edited.

## Pointers

Every `item` and `diagnostic` value is a JSON pointer ([RFC 6901](https://www.rfc-editor.org/rfc/rfc6901))
into `report.json`: `/group_risks/2` is the third entry of `group_risks`. Split the pointer
on `/` and index lists by number. The pointers in the index point into the `report.json` in the
same directory, and only that one.

## The summary

`summary`, in both `report.json` and the index, holds:

| Field | What it holds |
|---|---|
| `integrity` | `valid`, `degraded` or `invalid`. Read it before any number. |
| `error_count`, `warning_count`, `info_count` | Diagnostics by severity. |
| `headline` | Each weighted-mean failure rate (`kind: group_risk`) and each overall comparison (`kind: comparison`), with estimate, bounds, `eligibility` and `item`. |
| `warnings` | The errors, then the warnings, one entry per code. |
| `assumptions` | The ids of the assumptions the results rest on; each is at `/assumptions_used/<i>` with what follows if it is wrong. |
| `decisions_shown` | Whether the plan shows decisions (`report.include_decisions`). |
| `decision`, `claims` | The overall decision and each claim's status, present only when `decisions_shown` is true. |

The headline lists the weighted-mean rates and the overall comparisons only. Tail measures,
hierarchy nodes and comparisons per hierarchy level are in `group_risks`, `hierarchy_risks`
and `comparisons` in `report.json`.

Only `certifying` results support a claim. A `diagnostic` result is a sensitivity or a
description, and an `invalid` one has no usable bound. Bounds are null when none could be
computed.

## Warnings

Each entry of `warnings` gives a code, its severity and count, and what the code means in every
analysis: the input kind it concerns (`source`), its `effect`, the `assumption` it checks and
the `consequence` when that assumption fails. `occurrences` lists each diagnostic of the code:

- `diagnostic`: the pointer to the diagnostic, whose `message` says what was found in this
  analysis and whose `details` hold the values;
- `scope`: where the finding sits, as a path such as `plan/claims/<id>` or
  `cohorts/<cohort>/prompts/<prompt>`;
- `item`: the pointer to the record the scope names, such as `/claims/0` or
  `/prompt_estimates/12`. It is null when `report.json` holds no record of it, as for a plan
  setting such as `plan/outputs/risk_measures`, a run or a log.

Every diagnostic in `report.json` has the same `item` field. Effects:

| Effect | What it does |
|---|---|
| `invalid` | The integrity status is invalid. No claim can pass. |
| `degraded` | The results are reported, resting on an assumption the inputs could not confirm. |
| `withheld` | One result is not reported; the rest is unaffected. |
| `adjusted` | A result was computed differently, for example with a wider bound. |
| `recorded` | Nothing changed; the diagnostic records a fact needed to read the results. |

[Diagnostic codes and assumptions](diagnostics.md) gives each code's remedy.

## Decisions

With `report.include_decisions: false`, the default, the summary, the index and every rendering
leave out the overall decision and the claims' statuses. The option hides them: `report.json`
still holds them at `/decision` and `/claims`. Do not report a hidden decision as the
analysis's verdict. The plan's author chose not to present one.

## What not to conclude

- The bounds cover the fixed prompts of the design, not prompts in general.
- A degraded or invalid integrity status qualifies every number. Report it with them.
- An exploratory analysis certifies nothing. Every claim is `INCONCLUSIVE`, whatever its
  bounds.

[What results establish](what-results-establish.md) sets out what a result shows and what it
does not, and [Interpreting results](interpretation.md) explains the statuses.
