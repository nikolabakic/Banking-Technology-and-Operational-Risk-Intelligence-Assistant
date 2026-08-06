"""Search the active BankScope corpus with dense, BM25, or hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from bankscope.io import load_embedding_archive, read_jsonl, sha256_file
from bankscope.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/chunks.jsonl"
DEFAULT_TABLES = ROOT / "data/processed/tables.jsonl"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/embeddings.npz"
UNKNOWN_REVISION_WARNING = (
    "Embedding archive has model_revision='unknown'; exact model reproducibility is reduced."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--ticker", help="Optional bank ticker filter, for example JPM.")
    parser.add_argument("--record-type", choices=("text", "table"))
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    return parser.parse_args()


def encode_query(text: str, model_name: str, model_revision: str) -> np.ndarray:
    # Keep torch and SentenceTransformer out of imports used by unit tests and BM25-only runs.
    from sentence_transformers import SentenceTransformer

    model_options = (
        {} if model_revision.strip().lower() == "unknown" else {"revision": model_revision}
    )
    model = SentenceTransformer(model_name, **model_options)
    vector = model.encode_query(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vector[0], dtype=np.float32)


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep one copy of retrieval text and evidence in CLI JSON."""
    omitted = {"record_index", "embedding_text", "evidence"}
    return {key: value for key, value in result.items() if key not in omitted}


def retrieve(
    retriever: HybridRetriever,
    *,
    mode: str,
    query: str,
    query_vector: np.ndarray | None,
    limit: int,
    candidate_k: int,
    rrf_k: int,
    ticker: str | None,
    record_type: str | None,
) -> list[dict[str, Any]]:
    filters = {"limit": limit, "ticker": ticker, "record_type": record_type}
    if mode == "bm25":
        return retriever.search_bm25(query, **filters)
    if query_vector is None:
        raise ValueError(f"A query vector is required for {mode} retrieval.")
    if mode == "dense":
        return retriever.search_dense(query_vector, **filters)
    return retriever.search_hybrid(
        query, query_vector, candidate_k=candidate_k, rrf_k=rrf_k, **filters
    )


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive.")
    if args.mode == "hybrid":
        if args.rrf_k <= 0:
            raise ValueError("rrf-k must be positive.")
        if args.candidate_k < args.limit:
            raise ValueError("candidate-k must be at least limit.")

    records = read_jsonl(args.chunks)
    tables = read_jsonl(args.tables)
    archive: dict[str, Any] | None = None
    query_vector: np.ndarray | None = None
    if args.mode != "bm25":
        record_ids = [str(record.get("record_id") or "") for record in records]
        archive = load_embedding_archive(args.embeddings, expected_record_ids=record_ids)
        chunks_sha256 = sha256_file(args.chunks)
        if archive["input_sha256"] != chunks_sha256:
            raise ValueError("Embedding archive input hash does not match chunks.jsonl.")
        revision = str(archive["model_revision"])
        if revision.strip().lower() == "unknown":
            warnings.warn(UNKNOWN_REVISION_WARNING, RuntimeWarning, stacklevel=2)
        query_vector = encode_query(args.query, str(archive["model_name"]), revision)

    retriever = HybridRetriever(records, None if archive is None else archive["embeddings"], tables)
    results = retrieve(
        retriever,
        mode=args.mode,
        query=args.query,
        query_vector=query_vector,
        limit=args.limit,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        ticker=args.ticker,
        record_type=args.record_type,
    )
    print(
        json.dumps(
            {
                "query": args.query,
                "mode": args.mode,
                "result_count": len(results),
                "embedding_model": (
                    None
                    if archive is None
                    else {
                        "name": archive["model_name"],
                        "revision": archive["model_revision"],
                        "warning": (
                            UNKNOWN_REVISION_WARNING
                            if str(archive["model_revision"]).strip().lower() == "unknown"
                            else None
                        ),
                    }
                ),
                "results": [compact_result(result) for result in results],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
