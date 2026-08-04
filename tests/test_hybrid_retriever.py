from typing import Any

from bankscope.retrieval.hybrid_retriever import (
    build_hybrid_candidate_pool,
)


def make_result(
    target_chunk_id: str,
    method: str,
    rank: int,
) -> dict[str, Any]:
    return {
        "record_index": rank,
        "record_id": f"record::{target_chunk_id}",
        "target_chunk_id": target_chunk_id,
        "record_type": "text",
        "ticker": "JPM",
        "document": target_chunk_id,
        "metadata": {},
        "retrieval_method": method,
        "rank": rank,
        "score": 1.0 / rank,
    }


def test_candidate_pool_preserves_method_specific_results() -> None:
    dense_results = [
        make_result("dense_only", "dense", 1),
        make_result("shared", "dense", 2),
    ]
    bm25_results = [
        make_result("bm25_only", "bm25", 1),
        make_result("shared", "bm25", 2),
    ]

    candidates = build_hybrid_candidate_pool(
        dense_results,
        bm25_results,
        rrf_pool_size=1,
        per_method_limit=1,
        max_candidates=3,
    )

    candidate_ids = [result["target_chunk_id"] for result in candidates]

    assert candidate_ids == [
        "shared",
        "bm25_only",
        "dense_only",
    ]
