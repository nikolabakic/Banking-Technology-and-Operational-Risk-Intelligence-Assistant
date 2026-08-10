"""Benchmark isolated Supabase/pgvector dense and hybrid retrieval quality."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import numpy as np
import psycopg

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bankscope.evaluation.retrieval_metrics import (  # noqa: E402
    DEFAULT_K_VALUES,
    evaluate_evidence_groups,
    evaluate_ranking,
)
from bankscope.io import load_embedding_archive, read_jsonl, sha256_file  # noqa: E402
from bankscope.retrieval.hybrid_retriever import (  # noqa: E402
    HybridRetriever,
    reciprocal_rank_fusion,
)

HERE = Path(__file__).resolve().parent
SCHEMA = "bankscope_supabase_experiment"
TABLE = f"{SCHEMA}.documents"
DEFAULT_CHUNKS = ROOT / "data/processed/chunks.jsonl"
DEFAULT_TABLES = ROOT / "data/processed/tables.jsonl"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/embeddings.npz"
DEFAULT_QRELS = ROOT / "data/evaluation/queries.jsonl"
DEFAULT_OUTPUT = HERE / "results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", default=os.getenv("SUPABASE_DB_URL"))
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--hnsw-ef-search", type=int, default=100)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.db_url:
        raise ValueError("Set SUPABASE_DB_URL or pass --db-url.")
    if args.candidate_k < max(DEFAULT_K_VALUES):
        raise ValueError(f"candidate-k must be at least {max(DEFAULT_K_VALUES)}.")
    if args.rrf_k <= 0 or args.hnsw_ef_search <= 0:
        raise ValueError("rrf-k and hnsw-ef-search must be positive.")


def vector_literal(vector: np.ndarray) -> str:
    values = np.asarray(vector, dtype=np.float32)
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def prepare_database(
    connection: psycopg.Connection[Any],
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    recreate: bool,
) -> None:
    with connection.cursor() as cursor:
        if recreate:
            cursor.execute(f"drop schema if exists {SCHEMA} cascade")
        cursor.execute((HERE / "schema.sql").read_text(encoding="utf-8"))
        cursor.execute(f"select count(*) from {TABLE}")
        current_count = int(cursor.fetchone()[0])
        if current_count == len(records) and not recreate:
            return
        if current_count and not recreate:
            raise ValueError(
                f"Experimental table has {current_count} rows, expected {len(records)}; "
                "rerun with --recreate."
            )
        cursor.execute(f"truncate table {TABLE}")
        copy_sql = (
            f"copy {TABLE} "
            "(ordinal, record_id, target_chunk_id, record_type, ticker, bank_name, "
            "embedding_text, metadata, embedding) from stdin"
        )
        with cursor.copy(copy_sql) as copy:
            for ordinal, (record, embedding) in enumerate(zip(records, embeddings, strict=True)):
                metadata = record.get("metadata")
                metadata = metadata if isinstance(metadata, dict) else {}
                copy.write_row(
                    (
                        ordinal,
                        str(record["record_id"]),
                        str(record["target_chunk_id"]),
                        str(record["record_type"]).lower(),
                        str(record.get("ticker") or metadata.get("ticker") or "").upper(),
                        str(record.get("bank_name") or metadata.get("bank_name") or ""),
                        str(record["embedding_text"]),
                        json.dumps(metadata, ensure_ascii=False),
                        vector_literal(embedding),
                    )
                )
        cursor.execute(f"analyze {TABLE}")
    connection.commit()


class SupabaseRetriever:
    def __init__(
        self,
        connection: psycopg.Connection[Any],
        records: list[dict[str, Any]],
        tables: list[dict[str, Any]],
        *,
        hnsw_ef_search: int,
    ) -> None:
        self.connection = connection
        self.records = {str(record["record_id"]): record for record in records}
        self.tables = {
            str(table.get("table_id") or table.get("metadata", {}).get("table_id")): table
            for table in tables
        }
        self.hnsw_ef_search = hnsw_ef_search

    @staticmethod
    def _where(ticker: str | None, record_type: str | None) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if ticker:
            clauses.append("ticker = %s")
            values.append(ticker.upper())
        if record_type:
            clauses.append("record_type = %s")
            values.append(record_type.lower())
        return (" and ".join(clauses) if clauses else "true"), values

    def _result(self, row: tuple[Any, ...], *, method: str, rank: int) -> dict[str, Any]:
        record_id, target_id, record_type, ticker, score = row
        record = self.records[str(record_id)]
        document = str(record.get("document") or record["embedding_text"])
        if str(record_type) == "table":
            table = self.tables[str(target_id)]
            document = str(table["document"])
        return {
            "record_id": str(record_id),
            "target_chunk_id": str(target_id),
            "record_type": str(record_type),
            "ticker": str(ticker),
            "document": document,
            "evidence": document,
            "retrieval_method": method,
            "rank": rank,
            "score": float(score),
        }

    def search_dense(
        self,
        query_vector: np.ndarray,
        *,
        limit: int,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        where, values = self._where(ticker, record_type)
        query = f"""
            select record_id, target_chunk_id, record_type, ticker,
                   1 - (embedding <=> %s::vector) as score
            from {TABLE}
            where {where}
            order by embedding <=> %s::vector, ordinal
            limit %s
        """
        literal = vector_literal(query_vector)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "select set_config('hnsw.ef_search', %s, true)",
                (str(self.hnsw_ef_search),),
            )
            cursor.execute(query, [literal, *values, literal, limit])
            rows = cursor.fetchall()
        self.connection.rollback()
        return [
            self._result(row, method="dense", rank=rank) for rank, row in enumerate(rows, start=1)
        ]

    def search_fts(
        self,
        query_text: str,
        *,
        limit: int,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        where, values = self._where(ticker, record_type)
        query = f"""
            select record_id, target_chunk_id, record_type, ticker,
                   ts_rank_cd(fts, websearch_to_tsquery('english', %s)) as score
            from {TABLE}
            where {where}
              and fts @@ websearch_to_tsquery('english', %s)
            order by score desc, ordinal
            limit %s
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, [query_text, *values, query_text, limit])
            rows = cursor.fetchall()
        self.connection.rollback()
        return [
            self._result(row, method="fts", rank=rank) for rank, row in enumerate(rows, start=1)
        ]

    def search_hybrid(
        self,
        query_text: str,
        query_vector: np.ndarray,
        *,
        limit: int,
        candidate_k: int,
        rrf_k: int,
        ticker: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        dense = self.search_dense(
            query_vector,
            limit=candidate_k,
            ticker=ticker,
            record_type=record_type,
        )
        lexical = self.search_fts(
            query_text,
            limit=candidate_k,
            ticker=ticker,
            record_type=record_type,
        )
        return reciprocal_rank_fusion(dense, lexical, limit=limit, rrf_k=rrf_k)


def encode_queries(texts: list[str], model_name: str, revision: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision)
    return np.asarray(
        model.encode_query(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for method in ("baseline.dense", "baseline.hybrid", "supabase.dense", "supabase.hybrid"):
        selected = [row for row in rows if row["method"] == method]
        values: dict[str, float | int] = {"query_count": len(selected)}
        for k in DEFAULT_K_VALUES:
            values[f"hit_count_at_{k}"] = sum(
                int(row["metrics"][f"hit_at_{k}"]) for row in selected
            )
            values[f"mean_recall_at_{k}"] = fmean(
                float(row["metrics"][f"recall_at_{k}"]) for row in selected
            )
        values["mrr_at_10"] = fmean(
            float(row["metrics"]["reciprocal_rank_at_10"]) for row in selected
        )
        values["mean_retrieval_latency_ms"] = fmean(
            float(row["retrieval_latency_ms"]) for row in selected
        )
        output[method] = values
    return output


def main() -> None:
    args = parse_args()
    validate_args(args)
    records = read_jsonl(args.chunks)
    tables = read_jsonl(args.tables)
    queries = [query for query in read_jsonl(args.qrels) if query["status"] == "answerable"]
    record_ids = [str(record["record_id"]) for record in records]
    archive = load_embedding_archive(args.embeddings, expected_record_ids=record_ids)
    if archive["input_sha256"] != sha256_file(args.chunks):
        raise ValueError("Embedding archive does not match chunks.jsonl.")
    baseline = HybridRetriever(records, archive["embeddings"], tables)

    rows: list[dict[str, Any]] = []
    with psycopg.connect(args.db_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("select 1")
        query_vectors = encode_queries(
            [str(query["query"]) for query in queries],
            str(archive["model_name"]),
            str(archive["model_revision"]),
        )
        prepare_database(connection, records, archive["embeddings"], recreate=args.recreate)
        supabase = SupabaseRetriever(
            connection, records, tables, hnsw_ef_search=args.hnsw_ef_search
        )
        for query, query_vector in zip(queries, query_vectors, strict=True):
            text = str(query["query"])
            relevant = [str(value) for value in query["relevant_target_chunk_ids"]]
            ticker = str(query["ticker"]) if query.get("ticker") else None
            record_type = str(query["record_type"]) if query.get("record_type") else None
            filters = {"limit": 10, "ticker": ticker, "record_type": record_type}
            for method in (
                "baseline.dense",
                "baseline.hybrid",
                "supabase.dense",
                "supabase.hybrid",
            ):
                started = perf_counter()
                if method == "baseline.dense":
                    results = baseline.search_dense(query_vector, **filters)
                elif method == "baseline.hybrid":
                    results = baseline.search_hybrid(
                        text,
                        query_vector,
                        candidate_k=args.candidate_k,
                        rrf_k=args.rrf_k,
                        **filters,
                    )
                elif method == "supabase.dense":
                    results = supabase.search_dense(query_vector, **filters)
                else:
                    results = supabase.search_hybrid(
                        text,
                        query_vector,
                        candidate_k=args.candidate_k,
                        rrf_k=args.rrf_k,
                        **filters,
                    )
                latency_ms = (perf_counter() - started) * 1000
                retrieved = [str(result["target_chunk_id"]) for result in results]
                metrics = evaluate_ranking(retrieved, relevant)
                groups = [
                    [str(value) for value in group["target_chunk_ids"]]
                    for group in query.get("required_evidence_groups", [])
                ]
                metrics.update(evaluate_evidence_groups(retrieved, groups))
                rows.append(
                    {
                        "query_id": query["query_id"],
                        "question_type": query.get("question_type"),
                        "method": method,
                        "retrieval_latency_ms": latency_ms,
                        "metrics": metrics,
                        "retrieved_target_chunk_ids": retrieved,
                    }
                )

    summary = summarize(rows)
    dense_parity = all(
        row["retrieved_target_chunk_ids"]
        == next(
            other["retrieved_target_chunk_ids"]
            for other in rows
            if other["query_id"] == row["query_id"] and other["method"] == "baseline.dense"
        )
        for row in rows
        if row["method"] == "supabase.dense"
    )
    hybrid = summary["supabase.hybrid"]
    quality_gate = {
        "hit_count_at_5": int(hybrid["hit_count_at_5"]) >= 24,
        "hit_count_at_10": int(hybrid["hit_count_at_10"]) >= 25,
        "mrr_at_10": float(hybrid["mrr_at_10"]) >= 0.584,
        "mean_recall_at_10": float(hybrid["mean_recall_at_10"]) >= 0.825,
    }
    output = {
        "experiment": "Supabase-compatible PostgreSQL pgvector + FTS",
        "corpus": {
            "record_count": len(records),
            "chunks_sha256": sha256_file(args.chunks),
            "embeddings_sha256": sha256_file(args.embeddings),
            "qrels_sha256": sha256_file(args.qrels),
        },
        "settings": {
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "hnsw_ef_search": args.hnsw_ef_search,
        },
        "summary": summary,
        "dense_top_10_exact_parity": dense_parity,
        "supabase_hybrid_quality_gate": {
            "passed": all(quality_gate.values()),
            "checks": quality_gate,
        },
        "per_query": rows,
    }
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["summary"], indent=2))
    print(json.dumps(output["supabase_hybrid_quality_gate"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
