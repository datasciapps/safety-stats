# Security

## Reporting a vulnerability

Report a suspected vulnerability privately, through GitHub's private vulnerability reporting:
on the repository's **Security** tab, choose **Report a vulnerability**. Do not open an issue or
a pull request for it, and do not attach logs or reports from private evaluations: reduce the problem
to a minimal reproduction, with local paths and usernames redacted
([CONTRIBUTING section 9](CONTRIBUTING.md#9-data-secrets-and-sensitive-content)).

A report helps most when it says which version (`safety-stats version`) and which
command or function it concerns, what an attacker controls (a plan, a manifest, a log, a
rollout file, a kernel), and what they gain.

## What the package does and does not do

These are the guarantees a report can hold the package to.

### Analysis is offline

`analyze`, `validate`, `visualize`, `hash-plan`, the `plan` and `schema` commands, and the
matching Python functions make no network requests of their own and call no model provider.
They read the files you name and write the report directory you name. `report.html` embeds its
data and needs no server or external JavaScript.

The tests never call a model provider or the network
([CONTRIBUTING section 7](CONTRIBUTING.md#7-tests)).

### Plans are data, not code

Analysis plans (YAML or JSON) and generative plans (YAML) are parsed with `yaml.safe_load` or
`json.loads` and validated against a strict schema that rejects unknown fields. Nothing in a plan is imported or
executed.

The one place the package runs code you supply is the `--kernel` option of
`generative materialize`:

- `path/to/file.py:attribute` executes that file.
- `module:attribute` imports that module, after putting the current working directory first on
  `sys.path`, so a module of that name in the working directory wins over an installed one.

Name only a kernel you trust, and run the command from a directory you trust. A kernel may call
a rewriter model over the network; that is the kernel's doing, not the package's. The built-in
`fake` kernel makes no network requests.

### Prompt text stays out of reports

Analysis reports (`report.json`, `report.md`, `report.html` and the CSV files) identify prompts
by their IDs and contain no prompt text, model responses, or scorer explanations. The plan field
`report.include_prompt_text` is reserved and does not change this.

In the generative workflow, `generative materialize` writes two artifacts that do carry prompt
text, because Inspect has to evaluate it: `datasets/dataset.jsonl` and
`prompts.restricted.csv`. The ledgers (`selections.csv`, `rewrites.csv`) and everything
`generative analyze` writes carry hashes and IDs only. Handle the two restricted artifacts under
the access rules that apply to the source benchmark.

### Secrets

The package reads no API keys itself. The live examples take provider keys from environment
variables or an ignored `.env` file.
