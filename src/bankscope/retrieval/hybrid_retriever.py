from __future__ import annotations

import re
import unicodedata
from typing import Any

import bm25s
import numpy as np
from bm25s.tokenization import Tokenizer

FINANCIAL_TOKEN_PATTERN = r"(?iu)(?<!\w)[a-z0-9]+(?:[._-][a-z0-9]+)*%?(?!\w)"


def get_field(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record[field]

    metadata = record.get("metadata", {})

    if isinstance(metadata, dict):
        return metadata.get(field)

    return None


def normalize_lexical_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("–", "-").replace("—", "-")

    # Treat 12,345 and 12345 as the same lexical value.
    return re.sub(r"(?<=\d),(?=\d)", "", text)


class HybridRetriever:
    def __init__(
        self,
        records: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        self.records = records
        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got {self.embeddings.shape}.")

        if len(records) != self.embeddings.shape[0]:
            raise ValueError("Embedding and record counts do not match.")

        self.tickers = np.asarray(
            [str(get_field(record, "ticker") or "").upper() for record in records]
        )
        self.record_types = np.asarray(
            [str(get_field(record, "record_type") or "") for record in records]
        )

        self.tokenizer = Tokenizer(
            lower=True,
            splitter=FINANCIAL_TOKEN_PATTERN,
            stopwords=[],
            stemmer=None,
        )

        documents = [normalize_lexical_text(str(record["document"])) for record in records]

        corpus_tokens = self.tokenizer.tokenize(
            documents,
            update_vocab=True,
            show_progress=True,
        )

        self.bm25 = bm25s.BM25(method="lucene")
        self.bm25.index(
            corpus_tokens,
            show_progress=True,
        )

    def _allowed_indices(
        self,
        *,
        ticker: str | None,
        record_type: str | None,
    ) -> np.ndarray:
        mask = np.ones(len(self.records), dtype=bool)

        if ticker:
            mask &= self.tickers == ticker.upper()

        if record_type:
            mask &= self.record_types == record_type

        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            raise ValueError(
                "No records match the requested filters: "
                f"ticker={ticker}, record_type={record_type}."
            )

        return indices

    def _make_result(
        self,
        index: int,
        *,
        method: str,
        rank: int,
        score: float,
    ) -> dict[str, Any]:
        record = self.records[index]
        metadata = record.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        return {
            "record_index": index,
            "record_id": str(record["record_id"]),
            "target_chunk_id": str(record["target_chunk_id"]),
            "record_type": str(record["record_type"]),
            "ticker": str(get_field(record, "ticker") or ""),
            "document": str(record["document"]),
            "metadata": metadata,
            "retrieval_method": method,
            "rank": rank,
            "score": score,
        }

    def search_dense(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 30,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if query_vector.ndim != 1:
            raise ValueError(f"Expected one query vector, got {query_vector.shape}.")

        if query_vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query and document embedding dimensions do not match.")

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            raise ValueError("Query embedding has zero norm.")

        query_vector = query_vector / query_norm

        allowed_indices = self._allowed_indices(
            ticker=ticker,
            record_type=record_type,
        )

        scores = self.embeddings @ query_vector
        allowed_scores = scores[allowed_indices]

        order = np.argsort(
            -allowed_scores,
            kind="stable",
        )[:limit]

        ranked_indices = allowed_indices[order]

        return [
            self._make_result(
                int(index),
                method="dense",
                rank=rank,
                score=float(scores[index]),
            )
            for rank, index in enumerate(
                ranked_indices,
                start=1,
            )
        ]

    def search_bm25(
        self,
        query: str,
        *,
        limit: int = 30,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_query = normalize_lexical_text(query)

        query_tokens = self.tokenizer.tokenize(
            [normalized_query],
            update_vocab=False,
            show_progress=False,
        )

        has_filter = ticker is not None or record_type is not None
        retrieval_limit = len(self.records) if has_filter else min(limit, len(self.records))

        document_ids, scores = self.bm25.retrieve(
            query_tokens,
            k=retrieval_limit,
            show_progress=False,
        )

        allowed_indices = set(
            self._allowed_indices(
                ticker=ticker,
                record_type=record_type,
            ).tolist()
        )

        results: list[dict[str, Any]] = []

        for index, score in zip(
            document_ids[0],
            scores[0],
            strict=True,
        ):
            index = int(index)
            score = float(score)

            if score <= 0:
                continue

            if index not in allowed_indices:
                continue

            results.append(
                self._make_result(
                    index,
                    method="bm25",
                    rank=len(results) + 1,
                    score=score,
                )
            )

            if len(results) == limit:
                break

        return results

    def search_hybrid(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        limit: int = 5,
        candidate_k: int = 30,
        rrf_k: int = 60,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        dense_results = self.search_dense(
            query_vector,
            limit=candidate_k,
            ticker=ticker,
            record_type=record_type,
        )
        bm25_results = self.search_bm25(
            query,
            limit=candidate_k,
            ticker=ticker,
            record_type=record_type,
        )

        return reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            limit=limit,
            rrf_k=rrf_k,
        )

    def get_hybrid_candidates(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        candidate_k: int = 30,
        rrf_pool_size: int = 20,
        per_method_limit: int = 5,
        max_candidates: int = 30,
        rrf_k: int = 60,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        dense_results = self.search_dense(
            query_vector,
            limit=candidate_k,
            ticker=ticker,
            record_type=record_type,
        )
        bm25_results = self.search_bm25(
            query,
            limit=candidate_k,
            ticker=ticker,
            record_type=record_type,
        )

        return build_hybrid_candidate_pool(
            dense_results,
            bm25_results,
            rrf_pool_size=rrf_pool_size,
            per_method_limit=per_method_limit,
            max_candidates=max_candidates,
            rrf_k=rrf_k,
        )


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    *,
    limit: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for ranking in (dense_results, bm25_results):
        for result in ranking:
            target_chunk_id = result["target_chunk_id"]
            method = result["retrieval_method"]

            entry = fused.setdefault(
                target_chunk_id,
                {
                    "record_index": result["record_index"],
                    "record_id": result["record_id"],
                    "target_chunk_id": target_chunk_id,
                    "record_type": result["record_type"],
                    "ticker": result["ticker"],
                    "document": result["document"],
                    "metadata": result["metadata"],
                    "retrieval_method": "hybrid",
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                },
            )

            rank_field = f"{method}_rank"

            if entry[rank_field] is not None:
                continue

            entry["rrf_score"] += 1.0 / (rrf_k + result["rank"])
            entry[rank_field] = result["rank"]
            entry[f"{method}_score"] = result["score"]

    def sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
        ranks = [
            rank
            for rank in (
                result["dense_rank"],
                result["bm25_rank"],
            )
            if rank is not None
        ]

        return (
            -result["rrf_score"],
            min(ranks),
            result["target_chunk_id"],
        )

    return sorted(
        fused.values(),
        key=sort_key,
    )[:limit]


def build_hybrid_candidate_pool(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    *,
    rrf_pool_size: int = 20,
    per_method_limit: int = 5,
    max_candidates: int = 30,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    if rrf_pool_size <= 0:
        raise ValueError("rrf_pool_size must be positive.")

    if per_method_limit < 0:
        raise ValueError("per_method_limit cannot be negative.")

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")

    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        limit=len(dense_results) + len(bm25_results),
        rrf_k=rrf_k,
    )

    fused_by_target_id = {str(result["target_chunk_id"]): result for result in fused_results}

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_result(result: dict[str, Any]) -> None:
        target_chunk_id = str(result["target_chunk_id"])

        if target_chunk_id in selected_ids:
            return

        if len(selected) >= max_candidates:
            return

        selected.append(result)
        selected_ids.add(target_chunk_id)

    for result in fused_results[:rrf_pool_size]:
        add_result(result)

    # Preserve strong lexical candidates before dense-only candidates.
    for ranking in (bm25_results, dense_results):
        for result in ranking[:per_method_limit]:
            target_chunk_id = str(result["target_chunk_id"])
            add_result(fused_by_target_id[target_chunk_id])

    return selected
