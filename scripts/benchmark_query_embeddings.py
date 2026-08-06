"""Benchmark local query-embedding latency with the active dense model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from bankscope.io import load_embedding_archive, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "data/evaluation/queries.jsonl"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/embeddings.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure CPU latency for embedding the existing evaluation queries. "
            "No corpus embeddings or project data are modified."
        )
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device to benchmark (default: cpu).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N questions; by default all questions are used.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for the final all-query benchmark (default: 8).",
    )
    return parser.parse_args()


def load_query_texts(path: Path, limit: int | None = None) -> list[str]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")

    rows = read_jsonl(path)
    texts: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Query row {line_number} has no non-empty 'query' field: {path}")
        texts.append(query.strip())

    if not texts:
        raise ValueError(f"No queries found: {path}")
    return texts[:limit]


def synchronize(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def timed_encode(
    model: Any,
    texts: str | list[str],
    *,
    device: str,
    batch_size: int | None = None,
) -> tuple[np.ndarray, float]:
    options: dict[str, Any] = {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    if batch_size is not None:
        options["batch_size"] = batch_size

    synchronize(device)
    started = time.perf_counter()
    vectors = model.encode_query(texts, **options)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return np.asarray(vectors, dtype=np.float32), elapsed


def validate_vectors(vectors: np.ndarray, expected_rows: int, expected_dimensions: int) -> None:
    matrix = vectors.reshape(1, -1) if vectors.ndim == 1 else vectors
    expected_shape = (expected_rows, expected_dimensions)
    if matrix.shape != expected_shape:
        raise ValueError(f"Model returned shape {matrix.shape}; expected {expected_shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Model returned NaN or infinite query embeddings")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    queries = load_query_texts(args.queries, args.limit)
    archive = load_embedding_archive(args.embeddings)
    model_name = str(archive["model_name"])
    revision = str(archive["model_revision"])
    dimensions = int(archive["embeddings"].shape[1])

    print(f"Model: {model_name}")
    print(f"Revision: {revision}")
    print(f"Device: {args.device}")
    print(f"Questions: {len(queries)}")
    print(f"Expected dimensions: {dimensions}")

    # Include imports and model initialization in the cold-start measurement.
    load_started = time.perf_counter()
    from sentence_transformers import SentenceTransformer

    model_options = {} if revision.strip().lower() == "unknown" else {"revision": revision}
    model = SentenceTransformer(model_name, device=args.device, **model_options)
    model_load_seconds = time.perf_counter() - load_started

    first_vector, first_seconds = timed_encode(model, queries[0], device=args.device)
    validate_vectors(first_vector, 1, dimensions)

    individual_seconds: list[float] = []
    for query in queries:
        vector, elapsed = timed_encode(model, query, device=args.device)
        validate_vectors(vector, 1, dimensions)
        individual_seconds.append(elapsed)

    batch_vectors, batch_seconds = timed_encode(
        model,
        queries,
        device=args.device,
        batch_size=args.batch_size,
    )
    validate_vectors(batch_vectors, len(queries), dimensions)

    result = {
        "model": model_name,
        "revision": revision,
        "device": args.device,
        "query_count": len(queries),
        "dimensions": dimensions,
        "model_load_seconds": model_load_seconds,
        "first_query_seconds": first_seconds,
        "individual_mean_seconds": fmean(individual_seconds),
        "individual_median_seconds": median(individual_seconds),
        "individual_min_seconds": min(individual_seconds),
        "individual_max_seconds": max(individual_seconds),
        "batch_total_seconds": batch_seconds,
        "batch_seconds_per_query": batch_seconds / len(queries),
        "batch_size": args.batch_size,
    }

    print("\nResults:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
