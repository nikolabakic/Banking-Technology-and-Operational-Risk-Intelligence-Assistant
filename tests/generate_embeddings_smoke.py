import argparse
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024
BATCH_SIZE = 8

INPUT_PATH = Path("data/processed/chunks/sec_10k_chunks.jsonl")
OUTPUT_PATH = Path("data/processed/embeddings/qwen3_embedding_0_6b.npz")

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
    parser.add_argument(
        "--limit",
        type=int,
        help="Embeduje samo prvih N chunkova i ne čuva rezultat.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit mora biti veći od nule")

    all_chunks = load_chunks(INPUT_PATH)
    chunks = [chunk for chunk in all_chunks if not is_standalone_page_number(chunk)]
    removed_count = len(all_chunks) - len(chunks)

    if args.limit is not None:
        chunks = chunks[: args.limit]

    if not chunks:
        raise ValueError("Nema chunkova za embedding")

    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Pronađeni su duplirani chunk_id identifikatori")

    embedding_texts = [build_embedding_text(chunk) for chunk in chunks]

    model = SentenceTransformer(MODEL_NAME)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {model.device}")
    print(f"Chunkovi: {len(chunks)}")
    print(f"Filtrirani samostalni brojevi stranica: {removed_count}")

    started_at = perf_counter()

    embeddings = model.encode(
        embedding_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    elapsed_seconds = perf_counter() - started_at

    if embeddings.shape != (len(chunks), EMBEDDING_DIMENSION):
        raise ValueError(f"Neočekivan oblik embeddinga: {embeddings.shape}")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrica sadrži NaN ili beskonačne vrednosti")

    norms = np.linalg.norm(embeddings, axis=1)

    print(f"Oblik: {embeddings.shape}")
    print(f"Norme: min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")
    print(f"Vreme: {elapsed_seconds:.1f} s")

    if args.limit is not None:
        print("Smoke test je završen. Rezultat nije sačuvan.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        embeddings=embeddings,
        chunk_ids=np.asarray(chunk_ids),
        model_name=np.asarray(MODEL_NAME),
    )

    print(f"Sačuvano: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
