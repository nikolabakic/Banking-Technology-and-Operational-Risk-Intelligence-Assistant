from typing import Any

import numpy as np
import pytest

from bankscope.retrieval.mixed_retriever import (
    BankSearchResult,
    MixedRetriever,
    interleave_bank_results,
)


def ranked(target_id: str, method: str, rank: int, *, ticker: str = "JPM") -> dict[str, Any]:
    return {
        "record_id": f"record::{target_id}",
        "target_chunk_id": target_id,
        "record_type": "text",
        "ticker": ticker,
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
        ticker = str(kwargs.get("ticker") or "JPM").upper()
        return [ranked("a", "dense", 1, ticker=ticker), ranked("b", "dense", 2, ticker=ticker)]


class RecordingLexicalRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search_bm25(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"query": query, **kwargs})
        ticker = str(kwargs.get("ticker") or "JPM").upper()
        return [ranked("b", "bm25", 1, ticker=ticker), ranked("c", "bm25", 2, ticker=ticker)]

    def search_exact(self, terms, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"terms": list(terms), **kwargs})
        ticker = str(kwargs.get("ticker") or "JPM").upper()
        return [ranked("exact", "exact", 1, ticker=ticker)]


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


def test_mixed_retriever_delegates_bounded_exact_search() -> None:
    lexical = RecordingLexicalRetriever()
    retriever = MixedRetriever(RecordingDenseRetriever(), lexical)  # type: ignore[arg-type]

    results = retriever.search_exact(
        ["Common Equity Tier 1"], limit=4, ticker="JPM", record_type="text"
    )

    assert [result["target_chunk_id"] for result in results] == ["exact"]
    assert lexical.calls == [
        {
            "terms": ["Common Equity Tier 1"],
            "limit": 4,
            "ticker": "JPM",
            "record_type": "text",
        }
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


def test_multi_bank_search_preserves_order_filters_and_per_bank_limits() -> None:
    dense = RecordingDenseRetriever()
    lexical = RecordingLexicalRetriever()
    retriever = MixedRetriever(dense, lexical)  # type: ignore[arg-type]
    vector = np.asarray([1.0, 0.0], dtype=np.float32)

    searches = retriever.search_hybrid_by_ticker(
        "compare capital",
        vector,
        tickers=["bac", "C"],
        limit_per_ticker=2,
        candidate_k=4,
        rrf_k=10,
        record_type="TABLE",
    )

    assert [search.ticker for search in searches] == ["BAC", "C"]
    assert all(len(search.results) == 2 for search in searches)
    assert [call["ticker"] for call in dense.calls] == ["BAC", "C"]
    assert [call["ticker"] for call in lexical.calls] == ["BAC", "C"]
    assert all(call["limit"] == 4 for call in dense.calls + lexical.calls)
    assert all(call["record_type"] == "TABLE" for call in dense.calls + lexical.calls)


@pytest.mark.parametrize(
    ("tickers", "message"),
    [
        (["JPM"], "two to four"),
        (["JPM", "BAC", "C", "PNC", "TFC"], "two to four"),
        (["JPM", "jpm"], "unique"),
        (["JPM", ""], "empty"),
    ],
)
def test_multi_bank_search_validates_tickers(tickers: list[str], message: str) -> None:
    retriever = MixedRetriever(  # type: ignore[arg-type]
        RecordingDenseRetriever(), RecordingLexicalRetriever()
    )
    with pytest.raises(ValueError, match=message):
        retriever.search_hybrid_by_ticker(
            "risk", np.asarray([1.0, 0.0], dtype=np.float32), tickers=tickers
        )


def test_interleave_bank_results_is_deterministic_and_deduplicated() -> None:
    searches = [
        BankSearchResult(
            ticker="BAC",
            results=[
                ranked("bac-1", "hybrid", 1, ticker="BAC"),
                ranked("shared", "hybrid", 2, ticker="BAC"),
            ],
            latency_ms=1.0,
        ),
        BankSearchResult(
            ticker="C",
            results=[
                ranked("c-1", "hybrid", 1, ticker="C"),
                ranked("shared", "hybrid", 2, ticker="C"),
                ranked("c-3", "hybrid", 3, ticker="C"),
            ],
            latency_ms=2.0,
        ),
    ]

    results = interleave_bank_results(searches, limit=4)

    assert [result["target_chunk_id"] for result in results] == [
        "bac-1",
        "c-1",
        "shared",
        "c-3",
    ]
    assert [result["rank"] for result in results] == [1, 2, 3, 4]
    assert [result["bank_rank"] for result in results] == [1, 1, 2, 3]
