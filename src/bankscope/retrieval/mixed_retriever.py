from __future__ import annotations

from typing import Any

import numpy as np

from bankscope.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from bankscope.retrieval.qdrant_retriever import QdrantRetriever


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
