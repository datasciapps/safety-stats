# JailbreakBench-100 English/French/German/Maltese comparison

The four-language plan uses one joint max-T family for the English, French, German, and Maltese
summary trees and every declared language risk difference. Shared bootstrap replicates preserve
the parent/child overlap, and all raw bands are projected through one multi-arm hierarchy region.
The visualiser shows these approximate intervals in purple beside the finite-sample bands and marks
the separately calibrated one-sided upper endpoints in orange.

This workflow evaluates one target model on four frozen language conditions of the same complete
JailbreakBench harmful design. It machine-translates the 100 English goals into French and German,
and Maltese, preserves the same semantic `prompt_id` in every arm, runs repeated rollouts through Inspect AI,
and produces one report containing absolute risks, category-level intervals, all pairwise language
effects, and simultaneous rank sets.

The report contains no prompt text, target response, or grader rationale. The frozen translation
CSV and Inspect `.eval` logs do contain sensitive evaluation content and should be access-controlled.

## Why translation is a separate frozen step

The default translator is `facebook/nllb-200-distilled-600M`, a sequence-to-sequence machine
translation checkpoint rather than an instruction-following chat model. Sampling is disabled and
beam search is used. This avoids relying on a safety-tuned assistant that may refuse to process a
harmful source sentence. It does not guarantee a faithful translation: unusual terminology,
ambiguity, and safety-relevant details still require review.

Every generated artifact has a JSON sidecar recording:

- the pinned JBB source dataset checksum;
- the requested translation checkpoint and resolved Hugging Face commit when available;
- source and target language codes;
- generation settings;
- the SHA-256 of the complete translation CSV; and
- the exact expected 100-by-3 prompt/language cross-product.

The loader validates the sidecar, whole-file checksum, per-row source and translation hashes,
language labels, prompt IDs, coverage, duplicates, and the identity English arm before an eval can
start. The run then copies the verified artifact into its own `inputs/` directory. This makes the
actual strings immutable inputs to the analysis rather than a service that can change between
language arms.

NLLB-200 is published under CC-BY-NC-4.0 and its model card describes it as a research model. Check
that this is appropriate for your use. The translator model and revision are command-line options,
so a differently licensed local or Hugging Face-compatible sequence-to-sequence checkpoint can be
substituted.

## Installation

From the repository root:

```bash
cd /path/to/safety_stats
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[openai,translation]'
export OPENAI_API_KEY="your-api-key"
```

The `translation` extra installs PyTorch, Transformers, and SentencePiece. The first translation
run downloads the selected checkpoint unless it is already cached. Use `--device cpu`, `--device
mps`, or `--device cuda`; `auto` selects CUDA, then Apple MPS, then CPU.

## 1. Freeze translations

```bash
python examples/real_jailbreakbench/translate_jailbreakbench.py \
  --output examples/real_jailbreakbench/translations/jbb100-en-fr-de-mt.csv \
  --model facebook/nllb-200-distilled-600M \
  --revision f8d333a098d19b4fd9a8b18f94170487ad3f821d \
  --device auto \
  --batch-size 8
```

This produces:

```text
translations/
  jbb100-en-fr-de-mt.csv                 sensitive translated inputs
  jbb100-en-fr-de-mt.csv.metadata.json   content-free provenance and checksums
```

The default command pins the current checkpoint commit rather than following `main`; the resolved
revision is also recorded when the Hugging Face loader supplies it. Re-running against an existing
output fails unless `--overwrite` is supplied.

Before a substantive evaluation, review translation quality without editing the frozen CSV in
place. Corrections should create a new versioned artifact and updated provenance. Useful review
checks include blind bilingual assessment, terminology consistency, preservation of requested
specificity, and absence of accidental softening or intensification. The statistical intervals
cannot correct semantic mistranslation.

## 2. Smoke-test the Inspect workflow

Run one rollout per prompt first:

```bash
python examples/real_jailbreakbench/run_multilingual_jailbreakbench.py \
  --translations examples/real_jailbreakbench/translations/jbb100-en-fr-de-mt.csv \
  --target-model TARGET_MODEL \
  --grader-model GRADER_MODEL \
  --rollouts 1 \
  --max-samples 2
```

The target and grader accept provider-qualified Inspect names. Names without `/` are interpreted as
OpenAI model names. Use the same multilingual grader for all four arms and keep its configuration
fixed. A deterministic grader removes sampling variation in the judge, but it does not establish
measurement equivalence across languages; validation against blinded multilingual human labels is
still important.

## 3. Run repeated rollouts and analyze

```bash
python examples/real_jailbreakbench/run_multilingual_jailbreakbench.py \
  --translations examples/real_jailbreakbench/translations/jbb100-en-fr-de-mt.csv \
  --target-model TARGET_MODEL \
  --grader-model GRADER_MODEL \
  --rollouts 8 \
  --temperature 1.0 \
  --max-tokens 512 \
  --max-samples 4 \
  --max-retries 6 \
  --sample-retries 2
```

With four languages and eight rollouts, this requests 3,200 target generations and normally 3,200
grader calls. The language arms run sequentially. `--max-samples` controls concurrent samples, not
how many benchmark items are used.

The runner creates:

```text
multilingual-live-runs/RUN_ID/
  analysis-plan.yaml
  design.csv
  inputs/
    jbb100-en-fr-de-mt.csv
    jbb100-en-fr-de-mt.csv.metadata.json
  logs/
  report/
    report.html
    report.md
    report.json
    hierarchy-risks.csv
    comparisons.csv
    observations.csv
    checksums.json
```

If a language arm is interrupted, use Inspect's `eval-retry` on that `.eval` log. Once all four
tasks have a successful log, rerun only the numerical analysis with:

```bash
python examples/real_jailbreakbench/analyze_multilingual_run.py \
  examples/real_jailbreakbench/multilingual-live-runs/RUN_ID \
  --overwrite
```

## Add Maltese to an existing three-language run

An already completed English/French/German run does not need to be repeated. The incremental
runner appends only the 100 Maltese translations, runs only `jailbreakbench_100_maltese`, adds the
new cohort to the common all-pairs comparison family, and writes a separate four-language report:

```bash
python examples/real_jailbreakbench/add_multilingual_language.py \
  examples/real_jailbreakbench/multilingual-live-runs/RUN_ID \
  --language maltese \
  --translation-device auto \
  --max-samples 4 \
  --report-name report-4-language
```

The command infers the target model, grader, rollout count, temperature, output-token limit, and
API retry setting from the existing successful logs. It refuses configurations it cannot reproduce
exactly. The original translation rows and report are retained, the extended artifact records the
parent artifact SHA-256, and the original analysis plan is backed up as
`analysis-plan.3-language.yaml`. Re-running the command reuses a valid extended translation artifact
and any successful Maltese log.

## Statistical design

The fixed hierarchy is:

```text
all 100 behaviors
  -> policy category (10)
     -> semantic behavior/prompt (10 per category)
        -> stochastic rollout
```

Each language cohort uses the same target model but a distinct task name:

```text
english -> jailbreakbench_100_english
french  -> jailbreakbench_100_french
german  -> jailbreakbench_100_german
maltese -> jailbreakbench_100_maltese
```

The design uses nested-equal category/prompt weights, so each category has weight 0.1 and each
behavior weight 0.01. The report estimates harmful-compliance risk separately for English, French,
German, and Maltese at the root, every category, and every prompt. Weighted KL/Chernoff intervals are primary;
empirical-Bernstein and Hoeffding intervals are prespecified references.

The `language_conditions` comparison set requests all pairs:

- French minus English;
- German minus English; and
- German minus French;
- Maltese minus English;
- Maltese minus French; and
- Maltese minus German.

For each pair the report includes risk difference, risk ratio, and relative risk reduction at the
root and all categories. All arms and requested contrasts are projected from one jointly calibrated
hierarchy region. All-pairs risk differences also yield simultaneous rank sets. In the visualizer,
use **Configuration risk and comparison hierarchy** to switch between the four within-language
trees and the pairwise language-effect trees.

The primary rollout profile treats stochastic target generations as independent conditional on the
fixed prompt and recorded configuration. Prompt- and category-block profiles are reported as
sensitivity analyses allowing arbitrary dependence within a semantic behavior or policy category.
The rollout index is not interpreted as a paired random seed across languages; at the rollout level,
language-arm generations are independent. At prompt and category sensitivity levels, matching
semantic nodes are coupled jointly.

## Interpretation and limitations

The estimands are conditional on exactly these 100 English prompts, these two frozen translations,
the selected target model, generation configuration, and grader. They answer questions such as
“What is the category-balanced harmful-compliance risk on the frozen German condition?” and “How
different are the French and English conditions on this design?” They do not estimate performance
over all French or German safety prompts, all possible translations, or a deployment population.

A language difference can combine several mechanisms: target-model multilingual behavior,
translation quality, cultural or lexical choices, tokenizer behavior, and judge performance. The
report quantifies repeated-rollout uncertainty, not uncertainty from translation or judge validity.
Those sources should be addressed through artifact review and scorer validation rather than folded
into the rollout confidence interval.
