"""Two rewrite validators that need no provider: entailment, and embedding similarity.

The model-graded validator in [rewriter.py](rewriter.py) asks a second model whether a
candidate makes the same request. These two answer the same question without a provider, so
a run can be validated offline and reproducibly. Both declare `mode: deterministic` in the
plan: given fixed weights they return the same verdict for the same pair, which the
model-graded check cannot promise.

**Entailment** accepts when the seed and the candidate entail *each other* under an NLI
model. Mutual entailment is the closer match to "makes the same request": one-directional
entailment accepts a candidate that adds a condition the seed never carried -- "how do I
synthesize X" entails "how do I synthesize X in a school lab", but not the reverse.

**Cosine similarity** accepts when the two embeddings are close enough. It is cheaper and
simpler, and it measures whether two texts *look alike* rather than whether they ask the
same thing. Measured on six pairs against the defaults below (LaBSE, 14 September 2026),
that difference is fatal rather than theoretical:

    candidate                                     cosine   entailment (fwd/back)
    What is the procedure for making X?            0.858     1.00 / 0.98
    Comment synthetiser le compose X ?             0.968     0.97 / 0.94
    ... in a school laboratory?                    0.854     0.00 / 0.97
    How do I safely dispose of compound X?         0.872     0.03 / 0.01
    What is the capital of Malta?                  0.424     0.02 / 0.06

The candidate asking the *opposite* question scores higher (0.872) than the faithful
paraphrase (0.858), so no threshold separates them: this is not a tuning problem. Entailment
got all six right. Use this validator to find near-duplicates, which exact-hash
deduplication cannot see, rather than as the rule that decides what enters the design.

Both default to cross-lingual checkpoints, because `language` is a declared arm here and a
faithful Maltese rewrite of an English seed must be accepted. Declare the model and the
threshold in the plan so that they enter the kernel fingerprint:

```yaml
  validation:
    validator_id: entailment-bidirectional-v1
    mode: deterministic
    model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
    threshold: 0.5
```

Install what they need with `uv pip install -e '.[equivalence]'` (or `python -m pip`).
Verified against torch 2.14 and transformers 5.17 on CPU. After the checkpoint loads --
about a minute the first time, then cached -- entailment costs roughly 1.4 s per direction
and so 3 s per candidate, and an embedding pair about 0.1 s.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from inspect_safety_stats.generative import RewriteRequest, ValidationVerdict

DEFAULT_ENTAILMENT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEFAULT_SIMILARITY_MODEL = "sentence-transformers/LaBSE"

# P(premise entails hypothesis), for one ordered pair.
EntailmentScorer = Callable[[str, str], float]
# One vector per text, in the order given.
Embedder = Callable[[Sequence[str]], list[list[float]]]

MISSING_DEPENDENCIES = (
    "equivalence dependencies are unavailable; install with "
    "python -m pip install -e '.[equivalence]'"
)


@dataclass
class EntailmentValidator:
    """Accept a candidate that entails its seed and is entailed by it.

    ``scorer`` exists so tests can supply their own probabilities; leave it unset and the
    checkpoint is loaded on first use. Loading failures are raised rather than reported as
    a verdict: a missing dependency is a misconfigured run, not a transient fault.
    """

    model_id: str = DEFAULT_ENTAILMENT_MODEL
    threshold: float = 0.5
    revision: str | None = None
    local_files_only: bool = False
    scorer: EntailmentScorer | None = None
    validator_id: str = "entailment-bidirectional-v1"

    def _entails(self) -> EntailmentScorer:
        if self.scorer is None:
            self.scorer = load_entailment_scorer(
                self.model_id,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
        return self.scorer

    def validate(self, request: RewriteRequest, candidate: str) -> ValidationVerdict:
        entails = self._entails()
        try:
            forward = entails(request.seed_text, candidate)
            backward = entails(candidate, request.seed_text)
        except Exception as exc:  # noqa: BLE001 - a scoring failure is operational
            return ValidationVerdict(status="error", detail=f"{type(exc).__name__}: {exc}")

        reading = f"entailment {forward:.2f} forward, {backward:.2f} back"
        if min(forward, backward) >= self.threshold:
            return ValidationVerdict(status="equivalent", detail=reading)
        return ValidationVerdict(
            status="not_equivalent", detail=f"{reading}, below {self.threshold:.2f}"
        )


@dataclass
class SimilarityValidator:
    """Accept a candidate whose embedding is within ``threshold`` cosine of its seed's.

    The threshold is the whole method, and it is arbitrary until you calibrate it. Read
    what it accepts with ``review_prompts.py`` before trusting a number built on it.
    """

    model_id: str = DEFAULT_SIMILARITY_MODEL
    threshold: float = 0.85
    revision: str | None = None
    local_files_only: bool = False
    embedder: Embedder | None = None
    validator_id: str = "similarity-cosine-v1"

    def _embed(self) -> Embedder:
        if self.embedder is None:
            self.embedder = load_embedder(
                self.model_id,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
        return self.embedder

    def validate(self, request: RewriteRequest, candidate: str) -> ValidationVerdict:
        embed = self._embed()
        try:
            seed_vector, candidate_vector = embed([request.seed_text, candidate])
            score = cosine(seed_vector, candidate_vector)
        except Exception as exc:  # noqa: BLE001 - an embedding failure is operational
            return ValidationVerdict(status="error", detail=f"{type(exc).__name__}: {exc}")

        if score >= self.threshold:
            return ValidationVerdict(status="equivalent", detail=f"cosine {score:.3f}")
        return ValidationVerdict(
            status="not_equivalent", detail=f"cosine {score:.3f}, below {self.threshold:.3f}"
        )


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(math.fsum(a * a for a in left)) * math.sqrt(math.fsum(b * b for b in right))
    if norms == 0.0:
        raise ValueError("an embedding has zero norm, so cosine similarity is undefined")
    return dot / norms


def entailment_index(config: Any) -> int:
    """Find the output column that means *entailment*.

    NLI checkpoints do not agree on label order, and reading the wrong column silently
    turns the check into its opposite, so this fails loudly rather than assuming.
    """

    labels = dict(getattr(config, "id2label", {}) or {})
    for index, name in labels.items():
        if str(name).lower().startswith("entail"):
            return int(index)
    raise RuntimeError(
        f"{getattr(config, 'name_or_path', 'the checkpoint')!r} does not label an entailment "
        f"class; it labels {sorted(str(name) for name in labels.values())}"
    )


def load_entailment_scorer(
    model_id: str, *, revision: str | None = None, local_files_only: bool = False
) -> EntailmentScorer:
    """Load an NLI checkpoint and return P(entailment) for an ordered pair."""

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(MISSING_DEPENDENCIES) from error

    options: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **options)
    model.eval()
    column = entailment_index(model.config)

    def score(premise: str, hypothesis: str) -> float:
        encoded = tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        return float(torch.softmax(logits, dim=-1)[column])

    return score


def load_embedder(
    model_id: str, *, revision: str | None = None, local_files_only: bool = False
) -> Embedder:
    """Load an embedding checkpoint, pooling token states by the attention mask.

    Mean pooling is the generic recipe. Some checkpoints publish a pooled output their
    authors prefer, so check the model card before swapping the model out.
    """

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(MISSING_DEPENDENCIES) from error

    options: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
    model = AutoModel.from_pretrained(model_id, **options)
    model.eval()

    def embed(texts: Sequence[str]) -> list[list[float]]:
        batch = tokenizer(
            list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return [row.tolist() for row in pooled]

    return embed


def build_entailment_validator() -> EntailmentValidator:
    """Factory for ``offline_validators.py:build_entailment_validator``."""

    return EntailmentValidator()


def build_similarity_validator() -> SimilarityValidator:
    """Factory for ``offline_validators.py:build_similarity_validator``."""

    return SimilarityValidator()
