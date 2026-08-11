from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
from typing import Any

import numpy as np

FINANCIAL_TOKEN_PATTERN = r"(?iu)(?<!\w)[a-z0-9]+(?:[._-][a-z0-9]+)*%?(?!\w)"


def get_field(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record[field]

    metadata = record.get("metadata")
    return metadata.get(field) if isinstance(metadata, dict) else None


def normalize_lexical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"(?<=\d),(?=\d)", "", normalized)


def get_retrieval_text(record: dict[str, Any]) -> str:
    """Return the text used by both dense and lexical retrieval."""
    for field in ("embedding_text", "retrieval_text", "document"):
        value = str(record.get(field) or "").strip()
        if value:
            return value

    raise ValueError("Record has no embedding_text, retrieval_text, or document.")


class HybridRetriever:
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        embeddings: np.ndarray | None = None,
        tables: Sequence[dict[str, Any]] | None = None,
        lexical_records: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("At least one retrieval record is required.")
        self.lexical_records = [*self.records, *(lexical_records or [])]
        self.embeddings: np.ndarray | None = None
        if embeddings is not None:
            matrix = np.asarray(embeddings, dtype=np.float32)
            if matrix.ndim != 2:
                raise ValueError(f"Expected a 2D embedding matrix, got {matrix.shape}.")
            if len(self.records) != matrix.shape[0]:
                raise ValueError("Embedding and record counts do not match.")
            if matrix.shape[1] == 0:
                raise ValueError("Embedding vectors must have at least one dimension.")
            if not np.isfinite(matrix).all():
                raise ValueError("Embeddings contain NaN or Inf values.")
            norms = np.linalg.norm(matrix, axis=1)
            if np.any(norms == 0):
                raise ValueError("Document embeddings must have non-zero norm.")
            self.embeddings = matrix / norms[:, None]
        self._validate_record_order(self.records)
        self._validate_record_order(self.lexical_records)
        self.tables_by_id = self._index_tables(tables or [])
        table_ids = {
            str(get_field(record, "table_id") or record["target_chunk_id"])
            for record in self.lexical_records
            if str(get_field(record, "record_type") or "").lower() == "table"
        }
        if table_ids and tables is None:
            raise ValueError("A table store is required when retrieval records include tables.")
        missing_tables = table_ids - self.tables_by_id.keys()
        if missing_tables:
            raise ValueError(
                f"Table records reference unknown table IDs: {sorted(missing_tables)}."
            )
        self.tickers = np.asarray(
            [str(get_field(record, "ticker") or "").upper() for record in self.records]
        )
        self.record_types = np.asarray(
            [str(get_field(record, "record_type") or "").lower() for record in self.records]
        )
        self.lexical_tickers = np.asarray(
            [str(get_field(record, "ticker") or "").upper() for record in self.lexical_records]
        )
        self.lexical_record_types = np.asarray(
            [str(get_field(record, "record_type") or "").lower() for record in self.lexical_records]
        )
        # bm25s prints a Windows-only import notice to stdout; contain only that import.
        with redirect_stdout(StringIO()):
            bm25s = import_module("bm25s")
            tokenizer_class = import_module("bm25s.tokenization").Tokenizer
        self.tokenizer = tokenizer_class(
            lower=True, splitter=FINANCIAL_TOKEN_PATTERN, stopwords=[], stemmer=None
        )
        corpus = [
            normalize_lexical_text(get_retrieval_text(record)) for record in self.lexical_records
        ]
        corpus_tokens = self.tokenizer.tokenize(corpus, update_vocab=True, show_progress=False)
        self.bm25 = bm25s.BM25(method="lucene")
        self.bm25.index(corpus_tokens, show_progress=False)

    @staticmethod
    def _validate_record_order(records: Sequence[dict[str, Any]]) -> None:
        record_ids = [str(record.get("record_id") or "").strip() for record in records]
        if any(not record_id for record_id in record_ids):
            raise ValueError("Every retrieval record must have a non-empty record_id.")
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Retrieval record IDs must be unique.")
        for record in records:
            if not str(record.get("target_chunk_id") or "").strip():
                raise ValueError(f"Record {record['record_id']} has no target_chunk_id.")
            get_retrieval_text(record)

    @staticmethod
    def _index_tables(tables: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for table in tables:
            table_id = str(get_field(table, "table_id") or "").strip()
            if not table_id:
                raise ValueError("Every table must have a non-empty table_id.")
            if table_id in indexed:
                raise ValueError(f"Duplicate table_id: {table_id}.")
            if not str(table.get("document") or "").strip():
                raise ValueError(f"Table {table_id} has no document.")
            indexed[table_id] = table
        return indexed

    def _allowed_indices(self, *, ticker: str | None, record_type: str | None) -> np.ndarray:
        mask = np.ones(len(self.records), dtype=bool)
        if ticker:
            mask &= self.tickers == ticker.upper()
        if record_type:
            mask &= self.record_types == record_type.lower()
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            raise ValueError(
                "No records match the requested filters: "
                f"ticker={ticker}, record_type={record_type}."
            )
        return indices

    def _allowed_lexical_indices(
        self, *, ticker: str | None, record_type: str | None
    ) -> np.ndarray:
        mask = np.ones(len(self.lexical_records), dtype=bool)
        if ticker:
            mask &= self.lexical_tickers == ticker.upper()
        if record_type:
            mask &= self.lexical_record_types == record_type.lower()
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
        records: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        source_records = self.records if records is None else records
        record = source_records[index]
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        retrieval_text = get_retrieval_text(record)
        target_chunk_id = str(record["target_chunk_id"])
        record_type = str(get_field(record, "record_type") or "")
        document = str(record.get("document") or retrieval_text)
        if record_type.lower() == "table" and self.tables_by_id:
            table_id = str(get_field(record, "table_id") or target_chunk_id)
            table = self.tables_by_id.get(table_id)
            if table is None:
                raise ValueError(f"Table retrieval record references unknown table_id: {table_id}.")
            document = str(table["document"])
        return {
            "record_index": index,
            "record_id": str(record["record_id"]),
            "target_chunk_id": target_chunk_id,
            "record_type": record_type,
            "ticker": str(get_field(record, "ticker") or ""),
            "embedding_text": retrieval_text,
            "retrieval_text": retrieval_text,
            "document": document,
            "evidence": document,
            "metadata": metadata,
            "retrieval_method": method,
            "rank": rank,
            "score": score,
        }

    def search_dense(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        _validate_positive("limit", limit)
        if self.embeddings is None:
            raise ValueError("Dense retrieval requires document embeddings.")
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"Expected one query vector, got {vector.shape}.")
        if vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query and document embedding dimensions do not match.")
        if not np.isfinite(vector).all():
            raise ValueError("Query embedding contains NaN or Inf values.")
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Query embedding has zero norm.")
        scores = self.embeddings @ (vector / norm)
        allowed = self._allowed_indices(ticker=ticker, record_type=record_type)
        order = np.argsort(-scores[allowed], kind="stable")[:limit]
        return [
            self._make_result(int(index), method="dense", rank=rank, score=float(scores[index]))
            for rank, index in enumerate(allowed[order], start=1)
        ]

    def search_bm25(
        self,
        query: str,
        *,
        limit: int = 10,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        _validate_positive("limit", limit)
        allowed = set(
            self._allowed_lexical_indices(ticker=ticker, record_type=record_type).tolist()
        )
        query_tokens = self.tokenizer.tokenize(
            [normalize_lexical_text(query)], update_vocab=False, show_progress=False
        )
        document_ids, scores = self.bm25.retrieve(
            query_tokens, k=len(self.lexical_records), show_progress=False
        )
        results: list[dict[str, Any]] = []
        seen_targets: set[str] = set()
        for raw_index, raw_score in zip(document_ids[0], scores[0], strict=True):
            index, score = int(raw_index), float(raw_score)
            if index not in allowed or score <= 0:
                continue
            target_id = str(self.lexical_records[index]["target_chunk_id"])
            if target_id in seen_targets:
                continue
            seen_targets.add(target_id)
            results.append(
                self._make_result(
                    index,
                    method="bm25",
                    rank=len(results) + 1,
                    score=score,
                    records=self.lexical_records,
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
        limit: int = 10,
        candidate_k: int = 30,
        rrf_k: int = 60,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        _validate_positive("limit", limit)
        if candidate_k < limit:
            raise ValueError("candidate_k must be at least limit.")
        _validate_positive("rrf_k", rrf_k)
        dense = self.search_dense(
            query_vector, limit=candidate_k, ticker=ticker, record_type=record_type
        )
        lexical = self.search_bm25(query, limit=candidate_k, ticker=ticker, record_type=record_type)
        return reciprocal_rank_fusion(dense, lexical, limit=limit, rrf_k=rrf_k)


def reciprocal_rank_fusion(
    dense_results: Sequence[dict[str, Any]],
    bm25_results: Sequence[dict[str, Any]],
    *,
    limit: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse dense and BM25 rankings, deduplicating by target evidence ID."""
    _validate_positive("limit", limit)
    _validate_positive("rrf_k", rrf_k)
    fused: dict[str, dict[str, Any]] = {}
    for method, ranking in (("dense", dense_results), ("bm25", bm25_results)):
        for fallback_rank, result in enumerate(ranking, start=1):
            target_id = str(result["target_chunk_id"])
            entry = fused.setdefault(
                target_id,
                {
                    **result,
                    "retrieval_method": "hybrid",
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                },
            )
            if entry[f"{method}_rank"] is not None:
                continue
            rank = int(result.get("rank", fallback_rank))
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry[f"{method}_rank"] = rank
            entry[f"{method}_score"] = float(result["score"])
    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            min(rank for rank in (item["dense_rank"], item["bm25_rank"]) if rank is not None),
            str(item["target_chunk_id"]),
        ),
    )[:limit]
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
        result["score"] = result["rrf_score"]
    return ranked


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
