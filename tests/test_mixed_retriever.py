from typing import Any

import numpy as np
import pytest

from bankscope.retrieval.mixed_retriever import MixedRetriever


def ranked(target_id: str, method: str, rank: int) -> dict[str, Any]:
    return {
        "record_id": f"record::{target_id}",
        "target_chunk_id": target_id,
        "record_type": "text",
        "ticker": "JPM",
        "embedding_text": target_id,
        "retrieval_text": target_id,
        "document": f"evidence::{target_id}",
        "evidence": f"evidence::{target_id}",
        "metadata": {},
        "retrieval_method": method,
        "rank": rank,
        "score": 1.0 / rank,
    }


class RecordingDenseRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search_dense(self, query_vector: np.ndarray, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"query_vector": query_vector, **kwargs})
        return [ranked("a", "dense", 1), ranked("b", "dense", 2)]


class RecordingLexicalRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search_bm25(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"query": query, **kwargs})
        return [ranked("b", "bm25", 1), ranked("c", "bm25", 2)]


def test_mixed_retriever_delegates_dense_and_bm25() -> None:
    dense = RecordingDenseRetriever()
    lexical = RecordingLexicalRetriever()
    retriever = MixedRetriever(dense, lexical)  # type: ignore[arg-type]
    vector = np.asarray([1.0, 0.0], dtype=np.float32)

    dense_results = retriever.search_dense(vector, limit=2, ticker="jpm", record_type="text")
    bm25_results = retriever.search_bm25("risk capital", limit=2, ticker="jpm")

    assert [result["target_chunk_id"] for result in dense_results] == ["a", "b"]
    np.testing.assert_array_equal(dense.calls[0].pop("query_vector"), vector)
    assert dense.calls == [{"limit": 2, "ticker": "jpm", "record_type": "text"}]
    assert [result["target_chunk_id"] for result in bm25_results] == ["b", "c"]
    assert lexical.calls == [
        {"query": "risk capital", "limit": 2, "ticker": "jpm", "record_type": None}
    ]


def test_mixed_hybrid_uses_candidate_window_filters_and_application_rrf() -> None:
    dense = RecordingDenseRetriever()
    lexical = RecordingLexicalRetriever()
    retriever = MixedRetriever(dense, lexical)  # type: ignore[arg-type]
    vector = np.asarray([1.0, 0.0], dtype=np.float32)

    results = retriever.search_hybrid(
        "operational risk",
        vector,
        limit=3,
        candidate_k=5,
        rrf_k=10,
        ticker="JPM",
        record_type="TABLE",
    )

    assert [result["target_chunk_id"] for result in results] == ["b", "a", "c"]
    assert results[0]["dense_rank"] == 2
    assert results[0]["bm25_rank"] == 1
    np.testing.assert_array_equal(dense.calls[0].pop("query_vector"), vector)
    assert dense.calls[0] == {
        "limit": 5,
        "ticker": "JPM",
        "record_type": "TABLE",
    }
    assert lexical.calls[0] == {
        "query": "operational risk",
        "limit": 5,
        "ticker": "JPM",
        "record_type": "TABLE",
    }


def test_mixed_hybrid_validates_parameters() -> None:
    retriever = MixedRetriever(  # type: ignore[arg-type]
        RecordingDenseRetriever(), RecordingLexicalRetriever()
    )
    vector = np.asarray([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="candidate_k"):
        retriever.search_hybrid("risk", vector, limit=2, candidate_k=1)
    with pytest.raises(ValueError, match="rrf_k"):
        retriever.search_hybrid("risk", vector, rrf_k=0)
    with pytest.raises(ValueError, match="empty"):
        retriever.search_hybrid(" ", vector)
