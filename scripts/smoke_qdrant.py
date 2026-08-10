"""Run one quick random query against the local Qdrant collection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np

from bankscope.io import read_jsonl
from bankscope.retrieval.qdrant_retriever import QdrantRetriever, load_qdrant_manifest

ROOT = Path(__file__).resolve().parents[1]


def encode_query(text: str, model_name: str, revision: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision)
    vector = model.encode_query(
        [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return np.asarray(vector[0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--seed", type=int, help="Optional reproducible random seed.")
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = [
        query
        for query in read_jsonl(ROOT / "data/evaluation/queries.jsonl")
        if query["status"] == "answerable"
    ]
    question = random.Random(args.seed).choice(queries)
    tables_path = ROOT / "data/processed/tables.jsonl"
    manifest_path = ROOT / "data/processed/qdrant_manifest.json"
    manifest = load_qdrant_manifest(manifest_path)

    query_vector = None
    encoding_ms = 0.0
    if args.mode != "bm25":
        model = manifest["dense_model"]
        started = perf_counter()
        query_vector = encode_query(question["query"], model["name"], model["revision"])
        encoding_ms = (perf_counter() - started) * 1000

    started = perf_counter()
    retriever = QdrantRetriever(
        ROOT / "data/processed/qdrant",
        read_jsonl(tables_path),
        manifest_path=manifest_path,
        tables_path=tables_path,
    )
    open_ms = (perf_counter() - started) * 1000
    try:
        started = perf_counter()
        filters = {
            "limit": args.limit,
            "ticker": question.get("ticker"),
            "record_type": question.get("record_type"),
        }
        if args.mode == "bm25":
            results = retriever.search_bm25(question["query"], **filters)
        elif args.mode == "dense":
            results = retriever.search_dense(query_vector, **filters)
        else:
            results = retriever.search_hybrid(
                question["query"], query_vector, candidate_k=30, rrf_k=60, **filters
            )
        query_ms = (perf_counter() - started) * 1000
    finally:
        retriever.close()

    print(
        json.dumps(
            {
                "question": question["query"],
                "mode": args.mode,
                "filters": {
                    "ticker": question.get("ticker"),
                    "record_type": question.get("record_type"),
                },
                "timing_ms": {
                    "query_encoding": round(encoding_ms, 2),
                    "open_database": round(open_ms, 2),
                    "qdrant_query": round(query_ms, 2),
                    "total": round(encoding_ms + open_ms + query_ms, 2),
                },
                "results": [
                    {
                        "rank": result["rank"],
                        "score": round(result["score"], 4),
                        "ticker": result["ticker"],
                        "record_type": result["record_type"],
                        "target_chunk_id": result["target_chunk_id"],
                        "preview": result["document"][:250].replace("\n", " "),
                    }
                    for result in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
