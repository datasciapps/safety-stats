"""Freeze English, French, German, and Maltese JailbreakBench-100 inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jailbreakbench_multilingual import (
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TRANSLATION_REVISION,
    write_translation_artifact,
)

HERE = Path(__file__).resolve().parent


class NllbTranslator:
    """Deterministic sequence-to-sequence translation using an NLLB checkpoint."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        device: str,
        max_source_tokens: int,
        max_new_tokens: int,
        num_beams: int,
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "translation dependencies are unavailable; install with "
                "python -m pip install -e '.[translation]'"
            ) from error

        self._torch = torch
        self.model_id = model_id
        self.requested_revision = revision
        self.max_source_tokens = max_source_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.device = self._resolve_device(device)
        load_options: dict[str, Any] = {
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": False,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            src_lang="eng_Latn",
            **load_options,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_options)
        self.model.to(self.device)
        self.model.eval()
        self.resolved_revision = (
            getattr(self.model.config, "_commit_hash", None)
            or self.tokenizer.init_kwargs.get("_commit_hash")
            or revision
        )

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self._torch.cuda.is_available():
            return "cuda"
        if hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def generation_config(self) -> dict[str, Any]:
        return {
            "do_sample": False,
            "num_beams": self.num_beams,
            "max_source_tokens": self.max_source_tokens,
            "max_new_tokens": self.max_new_tokens,
            "source_language_code": "eng_Latn",
            "device": self.device,
        }

    def translate(self, texts: Sequence[str], target_language_code: str) -> list[str]:
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_language_code)
        if forced_bos_token_id is None or forced_bos_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"NLLB does not recognize language code {target_language_code!r}")
        inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_source_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                do_sample=False,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
            )
        return list(self.tokenizer.batch_decode(tokens, skip_special_tokens=True))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "translations" / "jbb100-en-fr-de-mt.csv",
        help="Destination CSV; a content-free .metadata.json sidecar is written beside it.",
    )
    parser.add_argument("--model", default=DEFAULT_TRANSLATION_MODEL)
    parser.add_argument(
        "--revision",
        default=DEFAULT_TRANSLATION_REVISION,
        help="Hugging Face model revision. The resolved commit is recorded in the sidecar.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download model files; require an existing local cache.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing translation artifact and sidecar.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if min(
        args.batch_size,
        args.max_source_tokens,
        args.max_new_tokens,
        args.num_beams,
    ) < 1:
        raise SystemExit("batch size, token limits, and beam count must all be positive")

    print(f"Loading translation model: {args.model} (revision {args.revision})")
    translator = NllbTranslator(
        model_id=args.model,
        revision=args.revision,
        device=args.device,
        max_source_tokens=args.max_source_tokens,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        local_files_only=args.local_files_only,
    )
    destination = write_translation_artifact(
        args.output.resolve(),
        translate=translator.translate,
        translation_model=translator.model_id,
        requested_revision=translator.requested_revision,
        resolved_revision=translator.resolved_revision,
        generation_config=translator.generation_config,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print("Frozen 400 inputs: 100 semantic behaviors x 4 language conditions")
    print(f"Translation artifact: {destination}")
    provenance = destination.with_suffix(destination.suffix + ".metadata.json")
    print(f"Content-free provenance: {provenance}")


if __name__ == "__main__":
    main()
