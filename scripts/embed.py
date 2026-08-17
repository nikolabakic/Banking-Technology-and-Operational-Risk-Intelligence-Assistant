"""Generate normalized Qwen embeddings for prepared BankScope chunks."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bankscope.io import read_jsonl, sha256_file
from bankscope.parsing.corpus import MAX_EMBEDDING_TOKENS

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT / "data/processed/chunks.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "data/processed/embeddings.npz"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_SEQ_LENGTH = MAX_EMBEDDING_TOKENS


def _required_text(record: dict[str, Any], field: str, record_number: int) -> str:
    value = record.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record {record_number} has no non-empty string '{field}'")

    return value.strip()


def load_embedding_inputs(path: Path) -> tuple[list[str], list[str]]:
    records = read_jsonl(path)

    if not records:
        raise ValueError(f"No chunk records found: {path}")

    record_ids: list[str] = []
    embedding_texts: list[str] = []

    for record_number, record in enumerate(records, start=1):
        record_ids.append(_required_text(record, "record_id", record_number))
        embedding_texts.append(_required_text(record, "embedding_text", record_number))

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"Chunk record_id values must be unique: {path}")

    return record_ids, embedding_texts


def resolve_device(requested: str, *, cuda_available: bool) -> str:
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested


def load_model(max_seq_length: int, *, device: str, model_revision: str) -> Any:
    """Load SentenceTransformer lazily so CLI import and pytest collection stay fast."""

    import torch
    from sentence_transformers import SentenceTransformer

    device = resolve_device(device, cuda_available=torch.cuda.is_available())
    model_kwargs: dict[str, Any] = {}

    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = SentenceTransformer(
        MODEL_NAME,
        revision=model_revision,
        device=device,
        model_kwargs=model_kwargs,
    )
    model.max_seq_length = max_seq_length
    print(f"Model: {MODEL_NAME} ({device}, max_seq_length={max_seq_length})")
    return model


def get_model_revision(model: Any) -> str:
    first_module = model[0]
    auto_model = getattr(first_module, "auto_model", None)
    config = getattr(auto_model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return str(revision or "unknown")


def encode(model: Any, texts: list[str], batch_size: int) -> Any:
    import numpy as np

    encode_method = getattr(model, "encode_document", model.encode)
    raw_embeddings = encode_method(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Model returned shape {embeddings.shape}; expected {len(texts)} embedding rows"
        )

    if embeddings.shape[1] == 0:
        raise ValueError("Model returned zero-dimensional embeddings")

    if not np.isfinite(embeddings).all():
        raise ValueError("Model returned NaN or infinite embedding values")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Model returned a zero embedding vector")

    embeddings = np.asarray(embeddings / norms, dtype=np.float32)
    normalized_norms = np.linalg.norm(embeddings, axis=1)

    if embeddings.dtype != np.float32 or not np.allclose(
        normalized_norms, 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError("Could not produce normalized float32 embeddings")

    return embeddings


def validate_input_lengths(model: Any, texts: Sequence[str], max_seq_length: int) -> None:
    """Fail before encoding instead of silently truncating an oversized chunk."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise ValueError("The embedding model does not expose a tokenizer for length validation")

    for record_number, text in enumerate(texts, start=1):
        token_length = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        if token_length > max_seq_length:
            raise ValueError(
                f"Embedding input {record_number} has {token_length} tokens; "
                f"maximum is {max_seq_length}"
            )


def save_archive(
    path: Path,
    *,
    embeddings: Any,
    record_ids: list[str],
    model_revision: str,
    input_sha256: str,
) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            np.savez(
                output_file,
                embeddings=embeddings,
                record_ids=np.asarray(record_ids, dtype=np.str_),
                model_name=np.asarray(MODEL_NAME),
                model_revision=np.asarray(model_revision),
                input_sha256=np.asarray(input_sha256),
            )

        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, help="Smoke test first N records without writing.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be greater than zero")

    if not args.model_revision.strip():
        raise ValueError("--model-revision cannot be empty")

    smoke_test = args.limit is not None

    if not smoke_test and args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Use --overwrite.")

    record_ids, embedding_texts = load_embedding_inputs(args.input)
    input_sha256 = sha256_file(args.input)

    if smoke_test:
        record_ids = record_ids[: args.limit]
        embedding_texts = embedding_texts[: args.limit]

    model = load_model(
        args.max_seq_length,
        device=args.device,
        model_revision=args.model_revision,
    )
    validate_input_lengths(model, embedding_texts, args.max_seq_length)
    embeddings = encode(model, embedding_texts, args.batch_size)

    if smoke_test:
        print(f"Smoke test passed: {embeddings.shape}; nothing was written.")
        return

    revision = get_model_revision(model)
    save_archive(
        args.output,
        embeddings=embeddings,
        record_ids=record_ids,
        model_revision=revision,
        input_sha256=input_sha256,
    )
    print(f"Saved {embeddings.shape[0]} embeddings to {args.output}")


if __name__ == "__main__":
    main()
