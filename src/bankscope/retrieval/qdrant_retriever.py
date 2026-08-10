from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from bankscope.io import sha256_file

DEFAULT_COLLECTION_NAME = "bankscope_retrieval"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL_NAME = "Qdrant/bm25"


def load_qdrant_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load Qdrant manifest: {manifest_path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Qdrant manifest must be a JSON object: {manifest_path}")

    required = {
        "format_version",
        "collection_name",
        "point_count",
        "dense_dimension",
        "dense_model",
        "sparse_model",
        "sources",
    }
    if missing := required - value.keys():
        raise ValueError(f"Qdrant manifest is missing fields: {sorted(missing)}")
    if value["format_version"] != 1:
        raise ValueError(f"Unsupported Qdrant manifest version: {value['format_version']}")
    return value


class QdrantRetriever:
    def __init__(
        self,
        path: str | Path,
        tables: Sequence[dict[str, Any]],
        *,
        manifest_path: str | Path | None = None,
        collection_name: str | None = None,
        tables_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.path.parent / "qdrant_manifest.json"
        )
        self.manifest = load_qdrant_manifest(self.manifest_path)
        manifest_collection = str(self.manifest["collection_name"])
        self.collection_name = collection_name or manifest_collection
        if self.collection_name != manifest_collection:
            raise ValueError(
                "Requested Qdrant collection does not match the manifest: "
                f"{self.collection_name} != {manifest_collection}."
            )

        self.dense_dimension = int(self.manifest["dense_dimension"])
        self.tables_by_id = self._index_tables(tables)
        table_source = self.manifest.get("sources", {}).get("tables", {})
        expected_table_hash = str(table_source.get("sha256") or "")
        table_path = tables_path or table_source.get("path")
        if expected_table_hash and table_path and Path(table_path).exists():
            if sha256_file(table_path) != expected_table_hash:
                raise ValueError("tables.jsonl does not match the Qdrant manifest.")

        self.client = QdrantClient(path=str(self.path))
        try:
            info = self.client.get_collection(self.collection_name)
            if info.points_count != int(self.manifest["point_count"]):
                raise ValueError("Qdrant point count does not match the manifest.")
            vectors = info.config.params.vectors
            dense = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
            if dense is None or dense.size != self.dense_dimension:
                raise ValueError("Qdrant dense vector schema does not match the manifest.")
            sparse = info.config.params.sparse_vectors or {}
            if SPARSE_VECTOR_NAME not in sparse:
                raise ValueError("Qdrant collection has no sparse vector named 'sparse'.")
        except Exception:
            self.client.close()
            raise

    @staticmethod
    def _index_tables(tables: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for table in tables:
            metadata = table.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            table_id = str(table.get("table_id") or metadata.get("table_id") or "").strip()
            if not table_id or not str(table.get("document") or "").strip():
                raise ValueError("Every table must have a table_id and document.")
            if table_id in indexed:
                raise ValueError(f"Duplicate table_id: {table_id}.")
            indexed[table_id] = table
        return indexed

    def close(self) -> None:
        self.client.close()

    def _filter(self, ticker: str | None, record_type: str | None) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []
        if ticker:
            conditions.append(
                models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))
            )
        if record_type:
            conditions.append(
                models.FieldCondition(
                    key="record_type", match=models.MatchValue(value=record_type.lower())
                )
            )
        return models.Filter(must=conditions) if conditions else None

    def _dense_vector(self, query_vector: np.ndarray) -> list[float]:
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.dense_dimension:
            raise ValueError(
                f"Expected one {self.dense_dimension}-dimensional query vector, got {vector.shape}."
            )
        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Query embedding must contain finite values and have non-zero norm.")
        return vector.tolist()

    def _results(self, points: Sequence[Any], *, method: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for rank, point in enumerate(points, start=1):
            payload = point.payload or {}
            required = {"record_id", "target_chunk_id", "record_type", "embedding_text"}
            if missing := required - payload.keys():
                raise ValueError(
                    f"Qdrant point {point.id} is missing payload fields: {sorted(missing)}"
                )
            record_type = str(payload["record_type"])
            retrieval_text = str(payload["embedding_text"])
            document = str(payload.get("document") or retrieval_text)
            if record_type.lower() == "table":
                table_id = str(payload.get("table_id") or payload["target_chunk_id"])
                table = self.tables_by_id.get(table_id)
                if table is None:
                    raise ValueError(f"Qdrant point references unknown table_id: {table_id}.")
                document = str(table["document"])
            metadata = payload.get("metadata")
            results.append(
                {
                    "record_id": str(payload["record_id"]),
                    "target_chunk_id": str(payload["target_chunk_id"]),
                    "record_type": record_type,
                    "ticker": str(payload.get("ticker") or ""),
                    "bank_name": str(payload.get("bank_name") or ""),
                    "embedding_text": retrieval_text,
                    "retrieval_text": retrieval_text,
                    "document": document,
                    "evidence": document,
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "retrieval_method": method,
                    "rank": rank,
                    "score": float(point.score),
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                }
            )
        return results

    def search_dense(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        _validate_positive("limit", limit)
        response = self.client.query_points(
            self.collection_name,
            query=self._dense_vector(query_vector),
            using=DENSE_VECTOR_NAME,
            query_filter=self._filter(ticker, record_type),
            limit=limit,
            with_payload=True,
        )
        return self._results(response.points, method="dense")

    def search_bm25(
        self,
        query: str,
        *,
        limit: int = 10,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        _validate_positive("limit", limit)
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        response = self.client.query_points(
            self.collection_name,
            query=models.Document(text=query, model=SPARSE_MODEL_NAME),
            using=SPARSE_VECTOR_NAME,
            query_filter=self._filter(ticker, record_type),
            limit=limit,
            with_payload=True,
        )
        return self._results(response.points, method="bm25")

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
        _validate_positive("rrf_k", rrf_k)
        if candidate_k < limit:
            raise ValueError("candidate_k must be at least limit.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        query_filter = self._filter(ticker, record_type)
        response = self.client.query_points(
            self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=self._dense_vector(query_vector),
                    using=DENSE_VECTOR_NAME,
                    limit=candidate_k,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=SPARSE_MODEL_NAME),
                    using=SPARSE_VECTOR_NAME,
                    limit=candidate_k,
                    filter=query_filter,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
            limit=limit,
            with_payload=True,
        )
        return self._results(response.points, method="hybrid")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
