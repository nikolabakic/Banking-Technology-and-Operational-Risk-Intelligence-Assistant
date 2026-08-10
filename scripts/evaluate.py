"""Evaluate the frozen retrieval qrels against the active corpus."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import numpy as np

from bankscope.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    evaluate_evidence_groups,
    evaluate_ranking,
)
from bankscope.io import load_embedding_archive, read_jsonl, sha256_file
from bankscope.retrieval.hybrid_retriever import HybridRetriever
from bankscope.retrieval.mixed_retriever import MixedRetriever
from bankscope.retrieval.qdrant_retriever import (
    DEFAULT_COLLECTION_NAME,
    QdrantRetriever,
    load_qdrant_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data/processed/chunks.jsonl"
DEFAULT_TABLES = ROOT / "data/processed/tables.jsonl"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/embeddings.npz"
DEFAULT_QRELS = ROOT / "data/evaluation/queries.jsonl"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/retrieval.json"
DEFAULT_QDRANT_PATH = ROOT / "data/processed/qdrant"
DEFAULT_QDRANT_MANIFEST = ROOT / "data/processed/qdrant_manifest.json"
VALID_STATUSES = {"answerable", "ambiguous", "unsupported"}
METHODS = ("dense", "bm25", "hybrid")
BACKENDS = ("baseline", "qdrant", "mixed")
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
    labels = list(
        dict.fromkeys(
            str(
                row.get("method_key")
                or (f"{row['backend']}.{row['method']}" if row.get("backend") else row["method"])
            )
            for row in rows
        )
    )
    for label in labels:
        method_rows = [
            row
            for row in rows
            if str(
                row.get("method_key")
                or (f"{row['backend']}.{row['method']}" if row.get("backend") else row["method"])
            )
            == label
        ]
        if not method_rows:
            continue

        values: dict[str, float | int] = {"query_count": len(method_rows)}
        for k in DEFAULT_K_VALUES:
            hits = [int(row["metrics"][f"hit_at_{k}"]) for row in method_rows]
            recalls = [float(row["metrics"][f"recall_at_{k}"]) for row in method_rows]
            values[f"hit_count_at_{k}"] = sum(hits)
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
        timed_rows = [row for row in method_rows if "retrieval_latency_ms" in row]
        if timed_rows:
            timings = [float(row["retrieval_latency_ms"]) for row in timed_rows]
            values["mean_retrieval_latency_ms"] = fmean(timings)
            values["total_retrieval_latency_ms"] = sum(timings)
        summary[label] = values
    return summary


def assess_quality_gate(
    rows: list[dict[str, Any]], summary: dict[str, dict[str, float | int]]
) -> dict[str, Any] | None:
    baseline = summary.get("baseline.hybrid")
    qdrant = summary.get("qdrant.hybrid")
    if baseline is None or qdrant is None:
        return None

    baseline_rows = {
        str(row["query_id"]): row for row in rows if row.get("method_key") == "baseline.hybrid"
    }
    qdrant_rows = {
        str(row["query_id"]): row for row in rows if row.get("method_key") == "qdrant.hybrid"
    }
    category_counts: dict[str, list[int]] = {}
    lost_grouped_queries: list[str] = []
    for query_id, baseline_row in baseline_rows.items():
        qdrant_row = qdrant_rows[query_id]
        category = str(baseline_row.get("question_type") or "unknown")
        counts = category_counts.setdefault(category, [0, 0])
        counts[0] += int(baseline_row["metrics"]["hit_at_10"])
        counts[1] += int(qdrant_row["metrics"]["hit_at_10"])
        baseline_complete = baseline_row["metrics"].get("complete_group_hit_at_10")
        qdrant_complete = qdrant_row["metrics"].get("complete_group_hit_at_10")
        if baseline_complete == 1 and qdrant_complete == 0:
            lost_grouped_queries.append(query_id)

    lost_categories = [
        category
        for category, (baseline_hits, qdrant_hits) in category_counts.items()
        if baseline_hits > 0 and qdrant_hits == 0
    ]
    checks = {
        "hit_count_at_5": int(qdrant["hit_count_at_5"]) >= 24,
        "hit_count_at_10": int(qdrant["hit_count_at_10"]) >= 25,
        "mrr_at_10": float(qdrant["mrr_at_10"]) >= 0.584,
        "mean_recall_at_10": float(qdrant["mean_recall_at_10"]) >= 0.825,
        "no_lost_question_category": not lost_categories,
        "no_lost_complete_evidence_group": not lost_grouped_queries,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "lost_question_categories": lost_categories,
        "lost_complete_evidence_group_queries": lost_grouped_queries,
    }


def compare_backend_hits(
    rows: list[dict[str, Any]], candidate_backend: str
) -> dict[str, list[dict[str, Any]]]:
    comparison: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        baseline = {
            str(row["query_id"]): row
            for row in rows
            if row.get("method_key") == f"baseline.{method}"
        }
        candidate = {
            str(row["query_id"]): row
            for row in rows
            if row.get("method_key") == f"{candidate_backend}.{method}"
        }
        differences: list[dict[str, Any]] = []
        for query_id in baseline.keys() & candidate.keys():
            changes = {
                f"hit_at_{k}": {
                    "baseline": int(baseline[query_id]["metrics"][f"hit_at_{k}"]),
                    candidate_backend: int(candidate[query_id]["metrics"][f"hit_at_{k}"]),
                }
                for k in (1, 5, 10)
                if baseline[query_id]["metrics"][f"hit_at_{k}"]
                != candidate[query_id]["metrics"][f"hit_at_{k}"]
            }
            if changes:
                differences.append(
                    {
                        "query_id": query_id,
                        "question_type": baseline[query_id].get("question_type"),
                        "changes": changes,
                    }
                )
        comparison[method] = sorted(differences, key=lambda item: str(item["query_id"]))
    return comparison


def assess_mixed_parity(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    differences: dict[str, list[dict[str, Any]]] = {}
    compared = False
    for method in METHODS:
        baseline = {
            str(row["query_id"]): row
            for row in rows
            if row.get("method_key") == f"baseline.{method}"
        }
        mixed = {
            str(row["query_id"]): row for row in rows if row.get("method_key") == f"mixed.{method}"
        }
        method_differences: list[dict[str, Any]] = []
        for query_id in sorted(baseline.keys() & mixed.keys()):
            compared = True
            baseline_ids = [
                str(item["target_chunk_id"]) for item in baseline[query_id]["retrieved"]
            ]
            mixed_ids = [str(item["target_chunk_id"]) for item in mixed[query_id]["retrieved"]]
            if baseline_ids != mixed_ids:
                method_differences.append(
                    {
                        "query_id": query_id,
                        "baseline_target_chunk_ids": baseline_ids,
                        "mixed_target_chunk_ids": mixed_ids,
                    }
                )
        differences[method] = method_differences
    if not compared:
        return None
    return {
        "passed": not any(differences.values()),
        "ranking_differences": differences,
    }


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
    parser.add_argument("--backend", choices=(*BACKENDS, "all"), default="mixed")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-manifest", type=Path, default=DEFAULT_QDRANT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    return parser.parse_args()


def main() -> None:
    evaluation_started = perf_counter()
    args = parse_args()
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("methods must not contain duplicates.")
    backend_arg = getattr(args, "backend", "mixed")
    backends = list(BACKENDS) if backend_arg == "all" else [backend_arg]
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
    qdrant_manifest: dict[str, Any] | None = None
    embedding_model: dict[str, str] | None = None
    revision_warning: str | None = None
    requires_qdrant = "qdrant" in backends or (needs_vectors and "mixed" in backends)
    if requires_qdrant:
        qdrant_manifest = load_qdrant_manifest(args.qdrant_manifest)
        qdrant_chunks_hash = str(
            qdrant_manifest.get("sources", {}).get("chunks", {}).get("sha256") or ""
        )
        if qdrant_chunks_hash != chunks_sha256:
            raise ValueError("chunks.jsonl does not match the Qdrant manifest.")
    if needs_vectors and "baseline" in backends:
        record_ids = [str(record.get("record_id") or "") for record in records]
        archive = load_embedding_archive(args.embeddings, expected_record_ids=record_ids)
        if archive["input_sha256"] != chunks_sha256:
            raise ValueError("Embedding archive input hash does not match chunks.jsonl.")
        if str(archive["model_revision"]).strip().lower() == "unknown":
            revision_warning = UNKNOWN_REVISION_WARNING
            warnings.warn(revision_warning, RuntimeWarning, stacklevel=2)
        embedding_model = {
            "name": str(archive["model_name"]),
            "revision": str(archive["model_revision"]),
        }
    elif needs_vectors and qdrant_manifest is not None:
        dense_model = qdrant_manifest.get("dense_model")
        if not isinstance(dense_model, dict):
            raise ValueError("Qdrant manifest has no valid dense_model.")
        embedding_model = {
            "name": str(dense_model.get("name") or ""),
            "revision": str(dense_model.get("revision") or ""),
        }
    if needs_vectors and (embedding_model is None or not all(embedding_model.values())):
        raise ValueError("Dense retrieval requires complete model metadata.")

    vectors = (
        encode_queries(
            [str(query["query"]) for query in answerable],
            embedding_model["name"],
            embedding_model["revision"],
        )
        if embedding_model is not None
        else None
    )

    retrievers: dict[str, HybridRetriever | QdrantRetriever | MixedRetriever] = {}
    baseline_retriever: HybridRetriever | None = None
    if "baseline" in backends:
        baseline_retriever = HybridRetriever(
            records, None if archive is None else archive["embeddings"], tables
        )
        retrievers["baseline"] = baseline_retriever

    qdrant_retriever: QdrantRetriever | None = None
    if requires_qdrant:
        qdrant_retriever = QdrantRetriever(
            args.qdrant_path,
            tables,
            manifest_path=args.qdrant_manifest,
            collection_name=args.collection,
            tables_path=args.tables,
        )
    if "qdrant" in backends:
        if qdrant_retriever is None:
            raise RuntimeError("Qdrant backend was not initialized.")
        retrievers["qdrant"] = qdrant_retriever
    if "mixed" in backends:
        lexical_retriever = baseline_retriever or HybridRetriever(records, tables=tables)
        if needs_vectors:
            if qdrant_retriever is None:
                raise RuntimeError("Mixed dense backend was not initialized.")
            retrievers["mixed"] = MixedRetriever(qdrant_retriever, lexical_retriever)
        else:
            retrievers["mixed"] = lexical_retriever

    rows: list[dict[str, Any]] = []
    try:
        for query_index, query in enumerate(answerable):
            query_text = str(query["query"])
            query_vector = None if vectors is None else vectors[query_index]
            relevant = [str(value) for value in query["relevant_target_chunk_ids"]]
            ticker = str(query["ticker"]) if query.get("ticker") else None
            record_type = str(query["record_type"]) if query.get("record_type") else None

            for backend in backends:
                retriever = retrievers[backend]
                for method in args.methods:
                    filters = {
                        "limit": max(DEFAULT_K_VALUES),
                        "ticker": ticker,
                        "record_type": record_type,
                    }
                    retrieval_started = perf_counter()
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
                    retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000
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
                            "backend": backend,
                            "method": method,
                            "method_key": f"{backend}.{method}",
                            "retrieval_latency_ms": retrieval_latency_ms,
                            "relevant_target_chunk_ids": relevant,
                            "required_evidence_groups": query.get("required_evidence_groups", []),
                            "metrics": metrics,
                            "retrieved": [result_preview(result) for result in results],
                        }
                    )
    finally:
        if qdrant_retriever is not None:
            qdrant_retriever.close()

    skipped = [query for query in queries if query["status"] != "answerable"]
    summary = summarize(rows)
    quality_gate = assess_quality_gate(rows, summary)
    backend_comparison = {
        f"baseline_vs_{candidate}": compare_backend_hits(rows, candidate)
        for candidate in ("qdrant", "mixed")
        if "baseline" in backends and candidate in backends
    }
    mixed_parity = assess_mixed_parity(rows)
    output = {
        "corpus": {
            "chunks": str(args.chunks),
            "tables": str(args.tables),
            "embeddings": str(args.embeddings),
            "record_count": len(records),
            "chunks_sha256": chunks_sha256,
            "embedding_input_sha256": None if archive is None else archive["input_sha256"],
            "embedding_model": None if embedding_model is None else embedding_model["name"],
            "embedding_model_revision": (
                None if embedding_model is None else embedding_model["revision"]
            ),
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
        "backend": backend_arg,
        "backends": backends,
        "methods": args.methods,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "evaluation_elapsed_seconds": perf_counter() - evaluation_started,
        "summary": summary,
        "quality_gate": quality_gate,
        "backend_comparison": backend_comparison or None,
        "mixed_parity": mixed_parity,
        "per_query": rows,
    }
    write_json(args.output, output)
    print(json.dumps(output["summary"], indent=2))
    if quality_gate is not None:
        print(json.dumps({"quality_gate": quality_gate}, indent=2))
    if mixed_parity is not None:
        print(json.dumps({"mixed_parity": mixed_parity}, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
