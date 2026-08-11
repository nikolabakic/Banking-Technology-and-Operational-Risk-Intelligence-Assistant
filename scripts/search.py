"""Search the active BankScope corpus with dense, BM25, or hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from bankscope.io import load_embedding_archive, read_jsonl, sha256_file
from bankscope.retrieval.glossary_locators import validate_glossary_locators
from bankscope.retrieval.hybrid_retriever import HybridRetriever
from bankscope.retrieval.mixed_retriever import MixedRetriever
from bankscope.retrieval.qdrant_retriever import (
    DEFAULT_COLLECTION_NAME,
    QdrantRetriever,
    load_qdrant_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/chunks.jsonl"
DEFAULT_TABLES = ROOT / "data/processed/tables.jsonl"
DEFAULT_GLOSSARY_LOCATORS = ROOT / "data/processed/lexical_glossary_locators_v1.jsonl"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/embeddings.npz"
DEFAULT_QDRANT_PATH = ROOT / "data/processed/qdrant"
DEFAULT_QDRANT_MANIFEST = ROOT / "data/processed/qdrant_manifest.json"
UNKNOWN_REVISION_WARNING = (
    "Embedding archive has model_revision='unknown'; exact model reproducibility is reduced."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument("--backend", choices=("baseline", "qdrant", "mixed"), default="mixed")
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--ticker", help="Optional bank ticker filter, for example JPM.")
    parser.add_argument("--record-type", choices=("text", "table"))
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--glossary-locators", type=Path, default=DEFAULT_GLOSSARY_LOCATORS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-manifest", type=Path, default=DEFAULT_QDRANT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
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
    retriever: HybridRetriever | QdrantRetriever | MixedRetriever,
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
    backend = getattr(args, "backend", "mixed")
    if args.limit <= 0:
        raise ValueError("limit must be positive.")
    if args.mode == "hybrid":
        if args.rrf_k <= 0:
            raise ValueError("rrf-k must be positive.")
        if args.candidate_k < args.limit:
            raise ValueError("candidate-k must be at least limit.")

    tables = read_jsonl(args.tables)
    archive: dict[str, Any] | None = None
    embedding_model: dict[str, str] | None = None
    query_vector: np.ndarray | None = None
    if backend in {"baseline", "mixed"}:
        records = read_jsonl(args.chunks)
    else:
        records = []
    glossary_locators = (
        read_jsonl(getattr(args, "glossary_locators", DEFAULT_GLOSSARY_LOCATORS))
        if backend == "mixed"
        else []
    )
    if backend == "mixed":
        validate_glossary_locators(glossary_locators, records, tables)

    if args.mode != "bm25" and backend == "baseline":
        record_ids = [str(record.get("record_id") or "") for record in records]
        archive = load_embedding_archive(args.embeddings, expected_record_ids=record_ids)
        chunks_sha256 = sha256_file(args.chunks)
        if archive["input_sha256"] != chunks_sha256:
            raise ValueError("Embedding archive input hash does not match chunks.jsonl.")
        revision = str(archive["model_revision"])
        if revision.strip().lower() == "unknown":
            warnings.warn(UNKNOWN_REVISION_WARNING, RuntimeWarning, stacklevel=2)
        embedding_model = {"name": str(archive["model_name"]), "revision": revision}
    elif args.mode != "bm25":
        manifest = load_qdrant_manifest(args.qdrant_manifest)
        dense_model = manifest.get("dense_model")
        if not isinstance(dense_model, dict):
            raise ValueError("Qdrant manifest has no valid dense_model.")
        embedding_model = {
            "name": str(dense_model.get("name") or ""),
            "revision": str(dense_model.get("revision") or ""),
        }
        if not all(embedding_model.values()):
            raise ValueError("Qdrant manifest has incomplete dense model metadata.")
        if backend == "mixed":
            expected_chunks_hash = str(
                manifest.get("sources", {}).get("chunks", {}).get("sha256") or ""
            )
            if expected_chunks_hash != sha256_file(args.chunks):
                raise ValueError("chunks.jsonl does not match the Qdrant manifest.")

    if embedding_model is not None:
        query_vector = encode_query(
            args.query, embedding_model["name"], embedding_model["revision"]
        )

    retriever: HybridRetriever | QdrantRetriever | MixedRetriever
    qdrant_retriever: QdrantRetriever | None = None
    if backend == "baseline":
        retriever = HybridRetriever(
            records, None if archive is None else archive["embeddings"], tables
        )
    elif backend == "qdrant":
        qdrant_retriever = QdrantRetriever(
            args.qdrant_path,
            tables,
            manifest_path=args.qdrant_manifest,
            collection_name=args.collection,
            tables_path=args.tables,
        )
        retriever = qdrant_retriever
    elif args.mode == "bm25":
        retriever = HybridRetriever(
            records,
            tables=tables,
            lexical_records=glossary_locators,
        )
    else:
        qdrant_retriever = QdrantRetriever(
            args.qdrant_path,
            tables,
            manifest_path=args.qdrant_manifest,
            collection_name=args.collection,
            tables_path=args.tables,
        )
        retriever = MixedRetriever(
            qdrant_retriever,
            HybridRetriever(records, tables=tables, lexical_records=glossary_locators),
        )
    try:
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
    finally:
        if qdrant_retriever is not None:
            qdrant_retriever.close()
    print(
        json.dumps(
            {
                "query": args.query,
                "backend": backend,
                "mode": args.mode,
                "result_count": len(results),
                "embedding_model": (
                    None
                    if embedding_model is None
                    else {
                        "name": embedding_model["name"],
                        "revision": embedding_model["revision"],
                        "warning": (
                            UNKNOWN_REVISION_WARNING
                            if embedding_model["revision"].strip().lower() == "unknown"
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
