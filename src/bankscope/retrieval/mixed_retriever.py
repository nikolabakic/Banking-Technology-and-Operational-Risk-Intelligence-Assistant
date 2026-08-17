from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from bankscope.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from bankscope.retrieval.qdrant_retriever import QdrantRetriever


@dataclass(frozen=True)
class BankSearchResult:
    ticker: str
    results: list[dict[str, Any]]
    latency_ms: float


def interleave_bank_results(
    bank_results: list[BankSearchResult], *, limit: int
) -> list[dict[str, Any]]:
    """Round-robin bank-owned rankings into one deterministic, deduplicated list."""
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if not bank_results:
        raise ValueError("At least one bank result is required.")

    ranked: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    max_bank_results = max(len(bank.results) for bank in bank_results)
    for bank_rank in range(max_bank_results):
        for bank in bank_results:
            if bank_rank >= len(bank.results):
                continue
            raw = bank.results[bank_rank]
            target_id = str(raw.get("target_chunk_id") or "").strip()
            if not target_id or target_id in seen_targets:
                continue
            ticker = str(raw.get("ticker") or "").upper()
            if ticker != bank.ticker:
                raise ValueError(
                    f"Bank result ticker mismatch: expected {bank.ticker}, received {ticker}."
                )
            seen_targets.add(target_id)
            result = dict(raw)
            result["bank_rank"] = int(raw.get("rank") or bank_rank + 1)
            result["rank"] = len(ranked) + 1
            ranked.append(result)
            if len(ranked) == limit:
                return ranked
    return ranked


class MixedRetriever:
    """Combine Qdrant dense search with local BM25S and application RRF."""

    def __init__(
        self,
        dense_retriever: QdrantRetriever,
        lexical_retriever: HybridRetriever,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever

    def search_dense(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.dense_retriever.search_dense(
            query_vector,
            limit=limit,
            ticker=ticker,
            record_type=record_type,
        )

    def search_bm25(
        self,
        query: str,
        *,
        limit: int = 10,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.lexical_retriever.search_bm25(
            query,
            limit=limit,
            ticker=ticker,
            record_type=record_type,
        )

    def search_exact(
        self,
        terms: list[str] | tuple[str, ...],
        *,
        limit: int = 20,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.lexical_retriever.search_exact(
            terms,
            limit=limit,
            ticker=ticker,
            record_type=record_type,
        )

    def search_hybrid(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        candidate_k: int = 30,
        rrf_k: int = 60,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if candidate_k < limit:
            raise ValueError("candidate_k must be at least limit.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        filters = {
            "limit": candidate_k,
            "ticker": ticker,
            "record_type": record_type,
        }
        dense = self.dense_retriever.search_dense(query_vector, **filters)
        lexical = self.lexical_retriever.search_bm25(query, **filters)
        return reciprocal_rank_fusion(dense, lexical, limit=limit, rrf_k=rrf_k)

    def search_hybrid_by_ticker(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        tickers: list[str] | tuple[str, ...],
        limit_per_ticker: int = 5,
        candidate_k: int = 30,
        rrf_k: int = 60,
        record_type: str | None = None,
    ) -> list[BankSearchResult]:
        """Run the accepted hybrid search independently for two to four banks."""
        normalized = [ticker.strip().upper() for ticker in tickers]
        if not 2 <= len(normalized) <= 4:
            raise ValueError("Multi-bank retrieval requires two to four tickers.")
        if any(not ticker for ticker in normalized):
            raise ValueError("Multi-bank tickers cannot be empty.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Multi-bank tickers must be unique.")
        if limit_per_ticker <= 0:
            raise ValueError("limit_per_ticker must be positive.")

        results: list[BankSearchResult] = []
        for ticker in normalized:
            started = perf_counter()
            evidence = self.search_hybrid(
                query,
                query_vector,
                limit=limit_per_ticker,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                ticker=ticker,
                record_type=record_type,
            )
            results.append(
                BankSearchResult(
                    ticker=ticker,
                    results=evidence,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        return results
