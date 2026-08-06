from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from bankscope.retrieval.hybrid_retriever import (
    HybridRetriever,
)

ROOT = Path(__file__).resolve().parents[1]

RECORDS_PATH = ROOT / "data/processed/embedding_records/sec_10k_embedding_records.jsonl"
EMBEDDINGS_PATH = ROOT / "data/processed/embeddings/qwen3_embedding_0_6b_records.npz"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the BankScope 10-K corpus.")
    parser.add_argument("query")
    parser.add_argument(
        "--mode",
        choices=("dense", "bm25", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--ticker")
    parser.add_argument(
        "--record-type",
        choices=("text", "table"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )
    return parser.parse_args()


def print_results(
    results: list[dict[str, Any]],
) -> None:
    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        preview = " ".join(result["document"].split())[:500]

        print(
            f"\n===== RESULT {rank} =====\n"
            f"Ticker: {result['ticker']}\n"
            f"Type: {result['record_type']}\n"
            f"Record ID: {result['record_id']}\n"
            f"Target chunk ID: {result['target_chunk_id']}"
        )

        if result["retrieval_method"] == "hybrid":
            print(
                f"RRF score: {result['rrf_score']:.8f}\n"
                f"Dense rank: {result['dense_rank']}\n"
                f"Dense score: {result['dense_score']}\n"
                f"BM25 rank: {result['bm25_rank']}\n"
                f"BM25 score: {result['bm25_score']}"
            )
        else:
            print(f"Method: {result['retrieval_method']}\nScore: {result['score']:.8f}")

        sec_item = metadata.get("sec_item")
        section_title = metadata.get("section_title")

        if sec_item or section_title:
            print(f"Parser metadata: {sec_item or '<missing>'} | {section_title or '<missing>'}")

        print(f"Evidence preview: {preview}")


def main() -> None:
    args = parse_args()
    records = load_jsonl(RECORDS_PATH)

    with np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False,
    ) as archive:
        embeddings = archive["embeddings"]
        npz_record_ids = archive["record_ids"].astype(str).tolist()
        model_name = str(archive["model_name"].item())

    jsonl_record_ids = [str(record["record_id"]) for record in records]

    # Vectors and records are joined positionally.
    if npz_record_ids != jsonl_record_ids:
        raise ValueError("NPZ vectors and embedding records are not in the same order.")

    retriever = HybridRetriever(
        records,
        embeddings,
    )

    if args.mode == "bm25":
        results = retriever.search_bm25(
            args.query,
            limit=args.limit,
            ticker=args.ticker,
            record_type=args.record_type,
        )
    else:
        model = SentenceTransformer(model_name)

        query_vector = model.encode_query(
            args.query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if args.mode == "dense":
            results = retriever.search_dense(
                query_vector,
                limit=args.limit,
                ticker=args.ticker,
                record_type=args.record_type,
            )
        else:
            results = retriever.search_hybrid(
                args.query,
                query_vector,
                limit=args.limit,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
                ticker=args.ticker,
                record_type=args.record_type,
            )

    print_results(results)


if __name__ == "__main__":
    main()
