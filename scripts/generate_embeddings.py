import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024
DEFAULT_BATCH_SIZE = 8

INPUT_PATH = Path("data/processed/embedding_records/sec_10k_embedding_records.jsonl")
OUTPUT_PATH = Path("data/processed/embeddings/qwen3_embedding_0_6b_records.npz")
CHECKPOINT_DIR = Path("data/processed/embeddings/qwen3_embedding_0_6b_record_checkpoints")

Record = dict[str, Any]


def load_records(path: Path) -> list[Record]:
    records: list[Record] = []

    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {line_number}: {path}")

            records.append(record)

    return records


def required_text(
    record: Record,
    field: str,
    record_name: str,
) -> str:
    value = str(record.get(field) or "").strip()

    if not value:
        raise ValueError(f"Missing {field} for {record_name}")

    return value


def get_ticker(record: Record) -> str:
    record_id = required_text(record, "record_id", "embedding record")
    metadata = record.get("metadata")

    if not isinstance(metadata, dict):
        raise ValueError(f"Missing metadata for {record_id}")

    return required_text(metadata, "ticker", record_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Qwen embeddings for prepared records.")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Embed only the first N records without saving them.",
    )
    return parser.parse_args()


def load_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs: dict[str, Any] = {}

    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs=model_kwargs,
    )

    embedding_dimension = model.get_embedding_dimension()

    if embedding_dimension != EMBEDDING_DIMENSION:
        raise ValueError(f"Unexpected embedding dimension: {embedding_dimension}")

    model_dtype = next(model.parameters()).dtype

    print(f"Device: {model.device}")
    print(f"Model dtype: {model_dtype}")
    print(f"Maximum sequence length: {model.max_seq_length}")

    return model


def normalize_float32(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains NaN or infinite values")

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError("Embedding matrix contains zero vectors")

    return embeddings / norms


def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
) -> None:
    expected_shape = (
        expected_rows,
        EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    if embeddings.dtype != np.float32:
        raise ValueError(f"Unexpected embedding dtype: {embeddings.dtype}")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains NaN or infinite values")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
        rtol=0.0,
    ):
        maximum_deviation = np.abs(norms - 1.0).max()

        raise ValueError(
            f"Embeddings are not properly normalized: maximum deviation={maximum_deviation:.8f}"
        )


def encode_records(
    model: SentenceTransformer,
    records: list[Record],
    batch_size: int,
) -> np.ndarray:
    embedding_texts = [
        required_text(
            record,
            "embedding_text",
            required_text(
                record,
                "record_id",
                "embedding record",
            ),
        )
        for record in records
    ]

    embeddings = model.encode_document(
        embedding_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Qwen may normalize in BF16 or FP16. Normalize once more
    # after conversion so stored vectors have precise unit norms.
    embeddings = normalize_float32(embeddings)

    validate_embeddings(
        embeddings,
        len(records),
    )

    return embeddings


def load_checkpoint(
    path: Path,
    record_ids: list[str],
) -> np.ndarray:
    with np.load(path) as checkpoint:
        saved_record_ids = checkpoint["record_ids"].tolist()
        embeddings = np.asarray(
            checkpoint["embeddings"],
            dtype=np.float32,
        )
        model_name = str(checkpoint["model_name"])

    if saved_record_ids != record_ids:
        raise ValueError(f"Checkpoint record IDs do not match: {path}")

    if model_name != MODEL_NAME:
        raise ValueError(f"Checkpoint model does not match: {path}")

    validate_embeddings(
        embeddings,
        len(record_ids),
    )

    return embeddings


def create_checkpoint(
    model: SentenceTransformer,
    records: list[Record],
    path: Path,
    batch_size: int,
) -> np.ndarray:
    record_ids = [
        required_text(
            record,
            "record_id",
            "embedding record",
        )
        for record in records
    ]

    embeddings = encode_records(
        model,
        records,
        batch_size,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = path.with_suffix(".tmp.npz")

    np.savez(
        temporary_path,
        embeddings=embeddings,
        record_ids=np.asarray(record_ids),
        model_name=np.asarray(MODEL_NAME),
    )
    temporary_path.replace(path)

    return embeddings


def run_smoke_test(
    model: SentenceTransformer,
    records: list[Record],
    batch_size: int,
) -> None:
    started_at = perf_counter()

    embeddings = encode_records(
        model,
        records,
        batch_size,
    )

    norms = np.linalg.norm(embeddings, axis=1)
    maximum_deviation = np.abs(norms - 1.0).max()

    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
    print(f"Norms: min={norms.min():.8f}, mean={norms.mean():.8f}, max={norms.max():.8f}")
    print(f"Maximum norm deviation: {maximum_deviation:.8f}")
    print(f"Time: {perf_counter() - started_at:.1f} s")
    print("Smoke test passed. Embeddings were not saved.")


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    records = load_records(args.input)

    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        raise ValueError("No embedding records found")

    record_ids = [
        required_text(
            record,
            "record_id",
            "embedding record",
        )
        for record in records
    ]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate embedding record IDs found")

    print(f"Model: {MODEL_NAME}")
    print(f"Embedding records: {len(records)}")

    if args.limit is not None:
        model = load_model()
        run_smoke_test(
            model,
            records,
            args.batch_size,
        )
        return

    records_by_ticker: dict[str, list[Record]] = defaultdict(list)

    for record in records:
        records_by_ticker[get_ticker(record)].append(record)

    missing_tickers = [
        ticker
        for ticker in records_by_ticker
        if not (args.checkpoint_dir / f"{ticker}.npz").exists()
    ]

    model = load_model() if missing_tickers else None

    all_embeddings: list[np.ndarray] = []
    output_record_ids: list[str] = []
    started_at = perf_counter()

    for ticker, ticker_records in records_by_ticker.items():
        checkpoint_path = args.checkpoint_dir / f"{ticker}.npz"
        ticker_record_ids = [
            required_text(
                record,
                "record_id",
                "embedding record",
            )
            for record in ticker_records
        ]

        if checkpoint_path.exists():
            embeddings = load_checkpoint(
                checkpoint_path,
                ticker_record_ids,
            )
            print(f"{ticker}: loaded checkpoint ({len(ticker_records)})")
        else:
            if model is None:
                raise RuntimeError("Embedding model is not loaded")

            print(f"{ticker}: generating {len(ticker_records)} embeddings")

            embeddings = create_checkpoint(
                model,
                ticker_records,
                checkpoint_path,
                args.batch_size,
            )
            print(f"{ticker}: checkpoint saved")

        all_embeddings.append(embeddings)
        output_record_ids.extend(ticker_record_ids)

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    validate_embeddings(
        embeddings,
        len(records),
    )

    norms = np.linalg.norm(embeddings, axis=1)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        args.output,
        embeddings=embeddings,
        record_ids=np.asarray(output_record_ids),
        model_name=np.asarray(MODEL_NAME),
    )

    print(f"Shape: {embeddings.shape}")
    print(f"Norms: min={norms.min():.8f}, mean={norms.mean():.8f}, max={norms.max():.8f}")
    print(f"Time for this run: {perf_counter() - started_at:.1f} s")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
