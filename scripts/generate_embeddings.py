import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024
DEFAULT_BATCH_SIZE = 8

INPUT_PATH = Path("data/processed/chunks/sec_10k_chunks.jsonl")
OUTPUT_PATH = Path("data/processed/embeddings/qwen3_embedding_0_6b.npz")
CHECKPOINT_DIR = Path("data/processed/embeddings/qwen3_embedding_0_6b_checkpoints")

PAGE_NUMBER_PATTERN = re.compile(r"\d{1,3}")

Record = dict[str, Any]


def load_chunks(path: Path) -> list[Record]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def is_standalone_page_number(chunk: Record) -> bool:
    text = str(chunk["text"]).strip()

    return chunk["element_type"] == "text" and PAGE_NUMBER_PATTERN.fullmatch(text) is not None


def build_embedding_text(chunk: Record) -> str:
    report_year = str(chunk["report_date"])[:4]
    metadata = [
        str(chunk["ticker"]),
        f"{report_year} 10-K",
    ]

    if chunk.get("sec_item"):
        metadata.append(str(chunk["sec_item"]))

    metadata.append(str(chunk["element_type"]))

    return f"{' | '.join(metadata)}\n\n{str(chunk['text']).strip()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
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
        help="Embeduje samo prvih N chunkova i ne čuva rezultat.",
    )
    return parser.parse_args()


def load_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME)

    if model.device.type == "cuda":
        model.half()

    print(f"Device: {model.device}")
    return model


def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
) -> None:
    expected_shape = (expected_rows, EMBEDDING_DIMENSION)

    if embeddings.shape != expected_shape:
        raise ValueError(f"Neočekivan oblik embeddinga: {embeddings.shape}")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrica sadrži NaN ili beskonačne vrednosti")


def load_checkpoint(
    path: Path,
    chunk_ids: list[str],
) -> np.ndarray:
    with np.load(path) as checkpoint:
        saved_chunk_ids = checkpoint["chunk_ids"].tolist()
        embeddings = np.asarray(
            checkpoint["embeddings"],
            dtype=np.float32,
        )
        model_name = str(checkpoint["model_name"])

    if saved_chunk_ids != chunk_ids or model_name != MODEL_NAME:
        raise ValueError(f"Checkpoint ne odgovara trenutnom korpusu: {path}")

    validate_embeddings(embeddings, len(chunk_ids))
    return embeddings


def create_checkpoint(
    model: SentenceTransformer,
    chunks: list[Record],
    path: Path,
    batch_size: int,
) -> np.ndarray:
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    embedding_texts = [build_embedding_text(chunk) for chunk in chunks]

    embeddings = model.encode(
        embedding_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    validate_embeddings(embeddings, len(chunks))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.npz")

    np.savez(
        temporary_path,
        embeddings=embeddings,
        chunk_ids=np.asarray(chunk_ids),
        model_name=np.asarray(MODEL_NAME),
    )
    temporary_path.replace(path)

    return embeddings


def run_smoke_test(
    model: SentenceTransformer,
    chunks: list[Record],
    batch_size: int,
) -> None:
    started_at = perf_counter()

    embeddings = model.encode(
        [build_embedding_text(chunk) for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    validate_embeddings(embeddings, len(chunks))

    print(f"Oblik: {embeddings.shape}")
    print(f"Vreme: {perf_counter() - started_at:.1f} s")
    print("Smoke test je završen. Rezultat nije sačuvan.")


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit mora biti veći od nule")

    if args.batch_size <= 0:
        raise ValueError("--batch-size mora biti veći od nule")

    all_chunks = load_chunks(args.input)
    chunks = [chunk for chunk in all_chunks if not is_standalone_page_number(chunk)]
    removed_count = len(all_chunks) - len(chunks)

    if args.limit is not None:
        chunks = chunks[: args.limit]

    if not chunks:
        raise ValueError("Nema chunkova za embedding")

    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Pronađeni su duplirani chunk_id identifikatori")

    print(f"Model: {MODEL_NAME}")
    print(f"Chunkovi: {len(chunks)}")
    print(f"Filtrirani samostalni brojevi stranica: {removed_count}")

    if args.limit is not None:
        model = load_model()
        run_smoke_test(model, chunks, args.batch_size)
        return

    chunks_by_ticker: dict[str, list[Record]] = defaultdict(list)

    for chunk in chunks:
        chunks_by_ticker[str(chunk["ticker"])].append(chunk)

    missing_tickers = [
        ticker
        for ticker in chunks_by_ticker
        if not (args.checkpoint_dir / f"{ticker}.npz").exists()
    ]

    model = load_model() if missing_tickers else None
    all_embeddings: list[np.ndarray] = []
    output_chunk_ids: list[str] = []
    started_at = perf_counter()

    for ticker, ticker_chunks in chunks_by_ticker.items():
        checkpoint_path = args.checkpoint_dir / f"{ticker}.npz"
        ticker_chunk_ids = [str(chunk["chunk_id"]) for chunk in ticker_chunks]

        if checkpoint_path.exists():
            embeddings = load_checkpoint(
                checkpoint_path,
                ticker_chunk_ids,
            )
            print(f"{ticker}: učitan checkpoint ({len(ticker_chunks)})")
        else:
            if model is None:
                raise RuntimeError("Embedding model nije učitan")

            print(f"{ticker}: generisanje {len(ticker_chunks)} embeddinga")

            embeddings = create_checkpoint(
                model,
                ticker_chunks,
                checkpoint_path,
                args.batch_size,
            )
            print(f"{ticker}: checkpoint sačuvan")

        all_embeddings.append(embeddings)
        output_chunk_ids.extend(ticker_chunk_ids)

    embeddings = np.concatenate(all_embeddings)
    validate_embeddings(embeddings, len(chunks))

    norms = np.linalg.norm(embeddings, axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        embeddings=embeddings,
        chunk_ids=np.asarray(output_chunk_ids),
        model_name=np.asarray(MODEL_NAME),
    )

    print(f"Oblik: {embeddings.shape}")
    print(f"Norme: min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")
    print(f"Vreme ovog pokretanja: {perf_counter() - started_at:.1f} s")
    print(f"Sačuvano: {args.output}")


if __name__ == "__main__":
    main()
