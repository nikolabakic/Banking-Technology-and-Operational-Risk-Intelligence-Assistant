from typing import Any

import numpy as np

from bankscope.retrieval.hybrid_retriever import (
    HybridRetriever,
    build_hybrid_candidate_pool,
)


def make_result(
    target_chunk_id: str,
    method: str,
    rank: int,
    parent_id: str | None = None,
) -> dict[str, Any]:
    return {
        "record_index": rank,
        "record_id": f"record::{target_chunk_id}",
        "target_chunk_id": target_chunk_id,
        "record_type": "text",
        "ticker": "JPM",
        "document": target_chunk_id,
        "metadata": {"parent_id": parent_id} if parent_id else {},
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


def test_candidate_pool_fills_requested_size_after_overlaps() -> None:
    dense_results = [make_result(f"shared_{index}", "dense", index) for index in range(1, 31)]
    bm25_results = [make_result(f"shared_{index}", "bm25", index) for index in range(1, 31)]

    candidates = build_hybrid_candidate_pool(
        dense_results,
        bm25_results,
        rrf_pool_size=20,
        per_method_limit=5,
        max_candidates=30,
    )

    assert len(candidates) == 30
    assert len({result["target_chunk_id"] for result in candidates}) == 30


def test_candidate_pool_prioritizes_distinct_parent_tables() -> None:
    dense_results = [
        make_result("table_a_1", "dense", 1, "table_a"),
        make_result("table_a_2", "dense", 2, "table_a"),
        make_result("table_a_3", "dense", 3, "table_a"),
        make_result("table_b_1", "dense", 4, "table_b"),
    ]
    bm25_results = [
        make_result("table_a_1", "bm25", 1, "table_a"),
        make_result("table_a_2", "bm25", 2, "table_a"),
        make_result("table_a_3", "bm25", 3, "table_a"),
        make_result("table_b_1", "bm25", 4, "table_b"),
    ]

    candidates = build_hybrid_candidate_pool(
        dense_results,
        bm25_results,
        rrf_pool_size=4,
        per_method_limit=0,
        max_candidates=3,
        max_per_parent=2,
    )

    assert [result["target_chunk_id"] for result in candidates] == [
        "table_a_1",
        "table_a_2",
        "table_b_1",
    ]


def test_bm25_indexes_embedding_text_but_returns_original_document() -> None:
    records = [
        {
            "record_id": "text::jpm",
            "target_chunk_id": "jpm",
            "record_type": "text",
            "embedding_text": "Bank: JPM\nReport: 2025 10-K\n\nOperational risk definition",
            "document": "Operational risk definition",
            "metadata": {"ticker": "JPM"},
        },
        {
            "record_id": "text::wfc",
            "target_chunk_id": "wfc",
            "record_type": "text",
            "embedding_text": "Bank: WFC\nReport: 2024 10-K\n\nOperational risk definition",
            "document": "Operational risk definition",
            "metadata": {"ticker": "WFC"},
        },
    ]
    retriever = HybridRetriever(records, np.eye(2, dtype=np.float32))

    results = retriever.search_bm25("JPM 2025 operational risk", limit=2)

    assert results[0]["target_chunk_id"] == "jpm"
    assert results[0]["embedding_text"].startswith("Bank: JPM")
    assert results[0]["document"] == "Operational risk definition"
