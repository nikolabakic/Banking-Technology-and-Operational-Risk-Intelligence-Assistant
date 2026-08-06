"""Evaluate the frozen retrieval qrels against the active corpus."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from bankscope.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    evaluate_evidence_groups,
    evaluate_ranking,
)
from bankscope.io import load_embedding_archive, read_jsonl, sha256_file
from bankscope.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/chunks.jsonl"
DEFAULT_TABLES = ROOT / "data/processed/tables.jsonl"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/embeddings.npz"
DEFAULT_QRELS = ROOT / "data/evaluation/queries.jsonl"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/retrieval.json"
VALID_STATUSES = {"answerable", "ambiguous", "unsupported"}
METHODS = ("dense", "bm25", "hybrid")
PREVIEW_FIELDS = (
    "rank",
    "record_id",
    "target_chunk_id",
    "record_type",
    "ticker",
    "score",
    "dense_rank",
    "bm25_rank",
)
UNKNOWN_REVISION_WARNING = (
    "Embedding archive has model_revision='unknown'; exact model reproducibility is reduced."
)


def validate_evidence_groups(
    query: dict[str, Any], relevant: list[str], corpus_ids: set[str]
) -> None:
    query_id = str(query["query_id"])
    groups = query.get("required_evidence_groups", [])
    if not isinstance(groups, list):
        raise ValueError(f"required_evidence_groups for {query_id} must be a list.")
    if groups and query["status"] != "answerable":
        raise ValueError(f"Only answerable query {query_id} may require evidence groups.")
    for group_index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            raise ValueError(f"Evidence group {group_index} for {query_id} must be an object.")
        raw_ids = group.get("target_chunk_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"Evidence group {group_index} for {query_id} must have target IDs.")
        group_ids = [str(value) for value in raw_ids]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(f"Evidence group {group_index} for {query_id} has duplicate IDs.")
        if unknown := set(group_ids) - corpus_ids:
            raise ValueError(
                f"Evidence group {group_index} has unknown target IDs: {sorted(unknown)}."
            )
        if not set(group_ids).issubset(relevant):
            raise ValueError(f"Evidence group {group_index} for {query_id} is outside its qrels.")


def validate_qrels(
    queries: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not queries:
        raise ValueError("The qrel file is empty.")
    corpus_ids = {str(record.get("target_chunk_id") or "") for record in records}
    seen_query_ids: set[str] = set()
    answerable: list[dict[str, Any]] = []
    for line_number, query in enumerate(queries, start=1):
        required = {"query_id", "query", "status", "relevant_target_chunk_ids"}
        missing = required - query.keys()
        if missing:
            raise ValueError(f"Qrel line {line_number} is missing fields: {sorted(missing)}.")
        query_id = str(query["query_id"]).strip()
        status = str(query["status"])
        raw_relevant = query["relevant_target_chunk_ids"]
        if not query_id or not str(query["query"]).strip():
            raise ValueError(f"Qrel line {line_number} has an empty query ID or query.")
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate query_id: {query_id}.")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status for {query_id}: {status}.")
        if not isinstance(raw_relevant, list):
            raise ValueError(f"Qrels for {query_id} must be a list.")
        relevant = [str(value) for value in raw_relevant]
        if len(relevant) != len(set(relevant)):
            raise ValueError(f"Duplicate relevant target IDs in {query_id}.")
        if status == "answerable" and not relevant:
            raise ValueError(f"Answerable query {query_id} has no qrels.")
        if status == "unsupported" and relevant:
            raise ValueError(f"Unsupported query {query_id} must not contain qrels.")
        if unknown := set(relevant) - corpus_ids:
            raise ValueError(f"Qrel {query_id} references unknown target IDs: {sorted(unknown)}.")
        primary = query.get("primary_target_chunk_id")
        if primary is not None and str(primary) not in relevant:
            raise ValueError(f"Primary target ID for {query_id} is not in its qrels.")
        validate_evidence_groups(query, relevant, corpus_ids)
        seen_query_ids.add(query_id)
        if status == "answerable":
            answerable.append(query)
    if not answerable:
        raise ValueError("The qrel file contains no answerable queries.")
    return answerable


def encode_queries(texts: list[str], model_name: str, model_revision: str) -> np.ndarray:
    # SentenceTransformer imports torch, so keep it behind the evaluator entry point.
    from sentence_transformers import SentenceTransformer

    model_options = (
        {} if model_revision.strip().lower() == "unknown" else {"revision": model_revision}
    )
    model = SentenceTransformer(model_name, **model_options)
    vectors = model.encode_query(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    )
    return np.asarray(vectors, dtype=np.float32)


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue

        values: dict[str, float | int] = {"query_count": len(method_rows)}
        for k in DEFAULT_K_VALUES:
            hits = [int(row["metrics"][f"hit_at_{k}"]) for row in method_rows]
            recalls = [float(row["metrics"][f"recall_at_{k}"]) for row in method_rows]
            values[f"hit_rate_at_{k}"] = fmean(hits)
            values[f"mean_recall_at_{k}"] = fmean(recalls)
        values["mrr_at_10"] = fmean(
            float(row["metrics"]["reciprocal_rank_at_10"]) for row in method_rows
        )
        grouped_rows = [
            row for row in method_rows if "required_evidence_group_count" in row["metrics"]
        ]
        if grouped_rows:
            values["grouped_query_count"] = len(grouped_rows)
            for k in DEFAULT_K_VALUES:
                values[f"mean_group_recall_at_{k}"] = fmean(
                    float(row["metrics"][f"group_recall_at_{k}"]) for row in grouped_rows
                )
                values[f"complete_group_hit_rate_at_{k}"] = fmean(
                    int(row["metrics"][f"complete_group_hit_at_{k}"]) for row in grouped_rows
                )
        summary[method] = values
    return summary


def result_preview(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in PREVIEW_FIELDS}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("methods must not contain duplicates.")
    if "hybrid" in args.methods:
        if args.rrf_k <= 0:
            raise ValueError("rrf-k must be positive.")
        if args.candidate_k < max(DEFAULT_K_VALUES):
            raise ValueError(f"candidate-k must be at least {max(DEFAULT_K_VALUES)}.")

    records = read_jsonl(args.chunks)
    tables = read_jsonl(args.tables)
    queries = read_jsonl(args.qrels)
    chunks_sha256 = sha256_file(args.chunks)
    answerable = validate_qrels(queries, records)
    needs_vectors = any(method != "bm25" for method in args.methods)
    archive: dict[str, Any] | None = None
    revision_warning: str | None = None
    if needs_vectors:
        record_ids = [str(record.get("record_id") or "") for record in records]
        archive = load_embedding_archive(args.embeddings, expected_record_ids=record_ids)
        if archive["input_sha256"] != chunks_sha256:
            raise ValueError("Embedding archive input hash does not match chunks.jsonl.")
        if str(archive["model_revision"]).strip().lower() == "unknown":
            revision_warning = UNKNOWN_REVISION_WARNING
            warnings.warn(revision_warning, RuntimeWarning, stacklevel=2)

    retriever = HybridRetriever(records, None if archive is None else archive["embeddings"], tables)
    vectors = (
        encode_queries(
            [str(query["query"]) for query in answerable],
            str(archive["model_name"]),
            str(archive["model_revision"]),
        )
        if archive is not None
        else None
    )

    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(answerable):
        query_text = str(query["query"])
        query_vector = None if vectors is None else vectors[query_index]
        relevant = [str(value) for value in query["relevant_target_chunk_ids"]]
        ticker = str(query["ticker"]) if query.get("ticker") else None
        record_type = str(query["record_type"]) if query.get("record_type") else None

        for method in args.methods:
            filters = {"limit": max(DEFAULT_K_VALUES), "ticker": ticker, "record_type": record_type}
            if method == "bm25":
                results = retriever.search_bm25(query_text, **filters)
            elif method == "dense":
                results = retriever.search_dense(query_vector, **filters)
            else:
                results = retriever.search_hybrid(
                    query_text,
                    query_vector,
                    candidate_k=args.candidate_k,
                    rrf_k=args.rrf_k,
                    **filters,
                )
            retrieved_ids = [str(result["target_chunk_id"]) for result in results]
            metrics = evaluate_ranking(retrieved_ids, relevant)
            groups = [
                [str(value) for value in group["target_chunk_ids"]]
                for group in query.get("required_evidence_groups", [])
            ]
            metrics.update(evaluate_evidence_groups(retrieved_ids, groups))
            rows.append(
                {
                    "query_id": query["query_id"],
                    "question_type": query.get("question_type"),
                    "method": method,
                    "relevant_target_chunk_ids": relevant,
                    "required_evidence_groups": query.get("required_evidence_groups", []),
                    "metrics": metrics,
                    "retrieved": [result_preview(result) for result in results],
                }
            )

    skipped = [query for query in queries if query["status"] != "answerable"]
    output = {
        "corpus": {
            "chunks": str(args.chunks),
            "tables": str(args.tables),
            "embeddings": str(args.embeddings),
            "record_count": len(records),
            "chunks_sha256": chunks_sha256,
            "embedding_input_sha256": None if archive is None else archive["input_sha256"],
            "embedding_model": None if archive is None else archive["model_name"],
            "embedding_model_revision": (None if archive is None else archive["model_revision"]),
            "embedding_revision_warning": revision_warning,
            "record_order_validated": archive is not None,
        },
        "qrels": {
            "path": str(args.qrels),
            "sha256": sha256_file(args.qrels),
            "answerable_count": len(answerable),
            "skipped_by_status": dict(Counter(str(query["status"]) for query in skipped)),
            "skipped": [
                {"query_id": query["query_id"], "status": query["status"]} for query in skipped
            ],
        },
        "methods": args.methods,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "summary": summarize(rows),
        "per_query": rows,
    }
    write_json(args.output, output)
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
