from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from bankscope.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    evaluate_ranking,
)
from bankscope.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from bankscope.retrieval.reranker import (
    RERANKER_MODEL_NAME,
    load_reranker,
    rerank_candidates,
)

ROOT = Path(__file__).resolve().parents[1]

RECORDS_PATH = ROOT / "data/processed/embedding_records/sec_10k_embedding_records.jsonl"
EMBEDDINGS_PATH = ROOT / "data/processed/embeddings/qwen3_embedding_0_6b_records.npz"
QUERIES_PATH = ROOT / "data/evaluation/retrieval_queries_dev.jsonl"
RESULTS_PATH = ROOT / "data/evaluation/results/retrieval_eval_dev.json"
TOP1_MISSES_PATH = ROOT / "data/evaluation/results/retrieval_top1_misses_dev.jsonl"

VALID_STATUSES = {
    "answerable",
    "ambiguous",
    "unsupported",
}
METHODS = ("dense", "bm25", "hybrid", "reranked")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_queries(
    queries: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not queries:
        raise ValueError("The retrieval query file is empty.")

    corpus_target_ids = {str(record["target_chunk_id"]) for record in records}

    seen_query_ids: set[str] = set()
    answerable_queries: list[dict[str, Any]] = []

    for line_number, query in enumerate(
        queries,
        start=1,
    ):
        for required_field in (
            "query_id",
            "query",
            "question_type",
            "status",
            "relevant_target_chunk_ids",
        ):
            if required_field not in query:
                raise ValueError(f"Query line {line_number} is missing '{required_field}'.")

        query_id = str(query["query_id"])
        status = str(query["status"])
        relevant_ids = [
            str(target_chunk_id) for target_chunk_id in query["relevant_target_chunk_ids"]
        ]

        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate query_id: {query_id}.")

        seen_query_ids.add(query_id)

        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status for {query_id}: {status}.")

        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError(f"Duplicate relevant target IDs in {query_id}.")

        if status == "answerable" and not relevant_ids:
            raise ValueError(f"Answerable query {query_id} has no qrels.")

        if status == "unsupported" and relevant_ids:
            raise ValueError(f"Unsupported query {query_id} must not contain relevant target IDs.")

        unknown_ids = set(relevant_ids) - corpus_target_ids

        if unknown_ids:
            raise ValueError(
                f"Query {query_id} references target IDs "
                f"that are not in the embedding records: "
                f"{sorted(unknown_ids)}."
            )

        primary_target_id = query.get("primary_target_chunk_id")

        if (
            status == "answerable"
            and primary_target_id is not None
            and str(primary_target_id) not in relevant_ids
        ):
            raise ValueError(f"Primary target ID for {query_id} is not present in its qrels.")

        evidence_groups = query.get("required_evidence_groups", [])

        if evidence_groups:
            if status != "answerable":
                raise ValueError(f"Only answerable query {query_id} may define evidence groups.")

            for group_index, group in enumerate(evidence_groups, start=1):
                group_ids = [
                    str(target_chunk_id) for target_chunk_id in group.get("target_chunk_ids", [])
                ]

                if not group_ids:
                    raise ValueError(f"Evidence group {group_index} for {query_id} is empty.")

                if not set(group_ids).issubset(relevant_ids):
                    raise ValueError(
                        f"Evidence group {group_index} for {query_id} "
                        "contains IDs outside its qrels."
                    )

        if status == "answerable":
            answerable_queries.append(query)

    if not answerable_queries:
        raise ValueError("There are no answerable queries to evaluate.")

    return answerable_queries


def classify_question_family(question_type: str) -> str:
    if question_type == "narrative_risk":
        return "narrative"

    if question_type == "cross_bank_coverage":
        return "cross_bank"

    return "table"


def evidence_group_coverage_at_k(
    retrieved_ids: Sequence[str],
    evidence_groups: Sequence[dict[str, Any]],
    *,
    k: int,
) -> float | None:
    if not evidence_groups:
        return None

    top_ids = set(retrieved_ids[:k])
    covered_groups = 0

    for group in evidence_groups:
        group_ids = {str(value) for value in group["target_chunk_ids"]}

        if top_ids & group_ids:
            covered_groups += 1

    return covered_groups / len(evidence_groups)


def make_result_preview(
    result: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    document = " ".join(str(result["document"]).split())

    score = result.get(
        "reranker_score",
        result.get(
            "rrf_score",
            result.get("score"),
        ),
    )

    return {
        "rank": rank,
        "target_chunk_id": result["target_chunk_id"],
        "record_id": result["record_id"],
        "ticker": result["ticker"],
        "record_type": result["record_type"],
        "reranker_score": result.get("reranker_score"),
        "rrf_score": result.get("rrf_score"),
        "score": score,
        "dense_rank": result.get("dense_rank"),
        "bm25_rank": result.get("bm25_rank"),
        "parent_id": result.get("metadata", {}).get("parent_id"),
        "preview": document[:500],
    }


def parent_table_diagnostics(
    results: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, int]:
    parent_counts = Counter(
        str(parent_id)
        for result in results[:limit]
        if (parent_id := result.get("metadata", {}).get("parent_id"))
    )
    return {
        "unique_parent_tables": len(parent_counts),
        "max_candidates_from_same_table": max(parent_counts.values(), default=0),
    }


def retrieve_with_method(
    retriever: HybridRetriever,
    *,
    method: str,
    query: str,
    query_vector: np.ndarray,
    ticker: str | None,
    relevant_ids: Sequence[str],
    limit: int,
    candidate_k: int,
    rrf_k: int,
    reranker: Any | None,
    reranker_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retrieval_started = perf_counter()

    if method == "dense":
        results = retriever.search_dense(
            query_vector,
            limit=limit,
            ticker=ticker,
        )
    elif method == "bm25":
        results = retriever.search_bm25(
            query,
            limit=limit,
            ticker=ticker,
        )
    elif method == "hybrid":
        results = retriever.search_hybrid(
            query,
            query_vector,
            limit=limit,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            ticker=ticker,
        )
    elif method == "reranked":
        if reranker is None:
            raise ValueError("Reranker must be loaded for the reranked method.")

        candidates = retriever.get_hybrid_candidates(
            query,
            query_vector,
            candidate_k=candidate_k,
            rrf_pool_size=20,
            per_method_limit=5,
            max_candidates=30,
            rrf_k=rrf_k,
            ticker=ticker,
        )
        retrieval_seconds = perf_counter() - retrieval_started
        candidate_ids = [str(candidate["target_chunk_id"]) for candidate in candidates]
        candidate_metrics = evaluate_ranking(
            candidate_ids,
            relevant_ids,
            k_values=(30,),
            reciprocal_rank_limit=30,
        )
        candidate_parent_diagnostics = parent_table_diagnostics(candidates, limit=30)

        reranking_started = perf_counter()
        results = rerank_candidates(
            reranker,
            query,
            candidates,
            limit=limit,
            batch_size=reranker_batch_size,
        )
        reranking_seconds = perf_counter() - reranking_started

        return results, {
            "retrieval_seconds": retrieval_seconds,
            "reranking_seconds": reranking_seconds,
            "candidate_count": len(candidates),
            "candidate_recall_at_30": candidate_metrics["recall_at_30"],
            "candidate_hit_at_30": candidate_metrics["hit_at_30"],
            "candidate_unique_parent_tables": candidate_parent_diagnostics[
                "unique_parent_tables"
            ],
            "candidate_max_from_same_table": candidate_parent_diagnostics[
                "max_candidates_from_same_table"
            ],
        }
    else:
        raise ValueError(f"Unknown retrieval method: {method}.")

    return results, {
        "retrieval_seconds": perf_counter() - retrieval_started,
        "reranking_seconds": 0.0,
        "candidate_count": None,
        "candidate_recall_at_30": None,
        "candidate_hit_at_30": None,
        "candidate_unique_parent_tables": None,
        "candidate_max_from_same_table": None,
    }


def summarize_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}

    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]

        if not method_rows:
            continue

        method_summary: dict[str, float | int] = {
            "query_count": len(method_rows),
        }

        for k in DEFAULT_K_VALUES:
            hit_values = [int(row["metrics"][f"hit_at_{k}"]) for row in method_rows]
            recall_values = [float(row["metrics"][f"recall_at_{k}"]) for row in method_rows]

            method_summary[f"hits_at_{k}"] = sum(hit_values)
            method_summary[f"hit_rate_at_{k}"] = fmean(hit_values)
            method_summary[f"mean_recall_at_{k}"] = fmean(recall_values)

        method_summary["mrr_at_10"] = fmean(
            float(row["metrics"]["reciprocal_rank_at_10"]) for row in method_rows
        )
        method_summary["mean_retrieval_seconds"] = fmean(
            float(row["diagnostics"]["retrieval_seconds"]) for row in method_rows
        )
        method_summary["mean_reranking_seconds"] = fmean(
            float(row["diagnostics"]["reranking_seconds"]) for row in method_rows
        )
        method_summary["mean_unique_parent_tables_at_10"] = fmean(
            int(row["diagnostics"]["unique_parent_tables"]) for row in method_rows
        )
        method_summary["max_candidates_from_same_table_at_10"] = max(
            int(row["diagnostics"]["max_candidates_from_same_table"])
            for row in method_rows
        )

        ticker_rows = [
            row for row in method_rows if row["diagnostics"]["ticker_filter_accuracy"] is not None
        ]

        if ticker_rows:
            method_summary["ticker_filter_accuracy"] = fmean(
                float(row["diagnostics"]["ticker_filter_accuracy"]) for row in ticker_rows
            )

        candidate_rows = [
            row for row in method_rows if row["diagnostics"]["candidate_recall_at_30"] is not None
        ]

        if candidate_rows:
            method_summary["candidate_hits_at_30"] = sum(
                int(row["diagnostics"]["candidate_hit_at_30"]) for row in candidate_rows
            )
            method_summary["candidate_hit_rate_at_30"] = fmean(
                int(row["diagnostics"]["candidate_hit_at_30"]) for row in candidate_rows
            )
            method_summary["mean_candidate_recall_at_30"] = fmean(
                float(row["diagnostics"]["candidate_recall_at_30"]) for row in candidate_rows
            )
            method_summary["mean_candidate_unique_parent_tables"] = fmean(
                int(row["diagnostics"]["candidate_unique_parent_tables"])
                for row in candidate_rows
            )
            method_summary["max_candidate_siblings_from_same_table"] = max(
                int(row["diagnostics"]["candidate_max_from_same_table"])
                for row in candidate_rows
            )

        group_rows = [
            row
            for row in method_rows
            if row["metrics"].get("evidence_group_coverage_at_10") is not None
        ]

        if group_rows:
            method_summary["cross_bank_query_count"] = len(group_rows)

            for k in DEFAULT_K_VALUES:
                coverage_values = [
                    float(row["metrics"][f"evidence_group_coverage_at_{k}"]) for row in group_rows
                ]
                method_summary[f"mean_evidence_group_coverage_at_{k}"] = fmean(coverage_values)
                method_summary[f"complete_evidence_group_hits_at_{k}"] = sum(
                    value == 1.0 for value in coverage_values
                )

        summary[method] = method_summary

    return summary


def print_summary(
    summary: dict[str, dict[str, float | int]],
) -> None:
    print("\nMethod   Queries   Hit@1   Hit@3   Hit@5   Hit@10   MRR@10")

    for method in METHODS:
        values = summary.get(method)

        if values is None:
            continue

        print(
            f"{method:<8} "
            f"{int(values['query_count']):>7} "
            f"{float(values['hit_rate_at_1']):>8.3f} "
            f"{float(values['hit_rate_at_3']):>7.3f} "
            f"{float(values['hit_rate_at_5']):>7.3f} "
            f"{float(values['hit_rate_at_10']):>8.3f} "
            f"{float(values['mrr_at_10']):>8.3f}"
        )


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate dense, BM25, hybrid and reranked retrieval against BankScope qrels.")
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=QUERIES_PATH,
    )
    parser.add_argument("--records", type=Path, default=RECORDS_PATH)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--top1-misses", type=Path, default=TOP1_MISSES_PATH)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument("--reranker-device", help="Device such as cuda, cuda:0, cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_limit = max(DEFAULT_K_VALUES)

    if args.candidate_k < retrieval_limit:
        raise ValueError(f"candidate-k must be at least {retrieval_limit}.")

    if args.rrf_k <= 0:
        raise ValueError("rrf-k must be positive.")

    if args.reranker_batch_size <= 0:
        raise ValueError("reranker-batch-size must be positive.")

    records = load_jsonl(args.records)
    queries = load_jsonl(args.queries)

    with np.load(
        args.embeddings,
        allow_pickle=False,
    ) as archive:
        embeddings = archive["embeddings"]
        npz_record_ids = archive["record_ids"].astype(str).tolist()
        model_name = str(archive["model_name"].item())
        embedded_input_sha256 = (
            str(archive["input_sha256"].item())
            if "input_sha256" in archive.files
            else None
        )

    jsonl_record_ids = [str(record["record_id"]) for record in records]

    if npz_record_ids != jsonl_record_ids:
        raise ValueError("NPZ vectors and embedding records are not in the same order.")

    records_sha256 = sha256_file(args.records)

    if embedded_input_sha256 is not None and embedded_input_sha256 != records_sha256:
        raise ValueError(
            "NPZ input SHA-256 does not match embedding records: "
            f"{embedded_input_sha256} != {records_sha256}."
        )

    answerable_queries = validate_queries(
        queries,
        records,
    )

    retriever = HybridRetriever(
        records,
        embeddings,
    )

    model = SentenceTransformer(model_name)

    query_vectors = model.encode_query(
        [str(query["query"]) for query in queries],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_vectors = np.asarray(
        query_vectors,
        dtype=np.float32,
    )

    del model

    print("\nLoading reranker...")

    reranker = load_reranker(
        device=args.reranker_device,
    )

    evaluation_rows: list[dict[str, Any]] = []

    query_vectors_by_id = {
        str(query["query_id"]): vector
        for query, vector in zip(queries, query_vectors, strict=True)
    }

    for query_index, query_record in enumerate(answerable_queries, start=1):
        print(
            f"Evaluating {query_index}/{len(answerable_queries)}: "
            f"{query_record['query_id']}"
        )
        query_vector = query_vectors_by_id[str(query_record["query_id"])]
        query_text = str(query_record["query"])
        ticker_value = query_record.get("ticker")
        ticker = str(ticker_value) if ticker_value else None
        relevant_ids = [
            str(target_chunk_id) for target_chunk_id in query_record["relevant_target_chunk_ids"]
        ]

        for method in METHODS:
            results, diagnostics = retrieve_with_method(
                retriever,
                method=method,
                query=query_text,
                query_vector=query_vector,
                ticker=ticker,
                relevant_ids=relevant_ids,
                limit=retrieval_limit,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
                reranker=reranker,
                reranker_batch_size=args.reranker_batch_size,
            )

            retrieved_ids = [str(result["target_chunk_id"]) for result in results]

            metrics = evaluate_ranking(
                retrieved_ids,
                relevant_ids,
            )

            evidence_groups = query_record.get("required_evidence_groups", [])

            for k in DEFAULT_K_VALUES:
                metrics[f"evidence_group_coverage_at_{k}"] = evidence_group_coverage_at_k(
                    retrieved_ids,
                    evidence_groups,
                    k=k,
                )

            diagnostics["duplicate_target_count"] = len(retrieved_ids) - len(set(retrieved_ids))
            diagnostics.update(parent_table_diagnostics(results, limit=10))
            diagnostics["ticker_filter_accuracy"] = (
                None
                if ticker is None or not results
                else float(
                    all(str(result["ticker"]).upper() == ticker.upper() for result in results)
                )
            )

            evaluation_rows.append(
                {
                    "query_id": query_record["query_id"],
                    "query": query_text,
                    "ticker": ticker,
                    "question_type": query_record["question_type"],
                    "question_family": classify_question_family(str(query_record["question_type"])),
                    "method": method,
                    "relevant_target_chunk_ids": (relevant_ids),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "retrieved": [
                        make_result_preview(
                            result,
                            rank,
                        )
                        for rank, result in enumerate(
                            results,
                            start=1,
                        )
                    ],
                }
            )

    overall_summary = summarize_rows(evaluation_rows)

    question_types = sorted({str(row["question_type"]) for row in evaluation_rows})
    by_question_type = {
        question_type: summarize_rows(
            [row for row in evaluation_rows if row["question_type"] == question_type]
        )
        for question_type in question_types
    }

    question_families = sorted({str(row["question_family"]) for row in evaluation_rows})
    by_question_family = {
        question_family: summarize_rows(
            [row for row in evaluation_rows if row["question_family"] == question_family]
        )
        for question_family in question_families
    }

    skipped_statuses = Counter(
        str(query["status"]) for query in queries if query["status"] != "answerable"
    )
    non_answerable_diagnostics: list[dict[str, Any]] = []

    for query_record in queries:
        if query_record["status"] == "answerable":
            continue

        ticker_value = query_record.get("ticker")
        ticker = str(ticker_value) if ticker_value else None
        query_text = str(query_record["query"])
        candidates = retriever.get_hybrid_candidates(
            query_text,
            query_vectors_by_id[str(query_record["query_id"])],
            candidate_k=args.candidate_k,
            rrf_pool_size=20,
            per_method_limit=5,
            max_candidates=30,
            rrf_k=args.rrf_k,
            ticker=ticker,
        )
        results = rerank_candidates(
            reranker,
            query_text,
            candidates,
            limit=retrieval_limit,
            batch_size=args.reranker_batch_size,
        )
        non_answerable_diagnostics.append(
            {
                "query_id": query_record["query_id"],
                "status": query_record["status"],
                "query": query_text,
                "annotation_notes": query_record.get("annotation_notes"),
                "candidate_count": len(candidates),
                "candidate_parent_diagnostics": parent_table_diagnostics(
                    candidates,
                    limit=30,
                ),
                "top10_parent_diagnostics": parent_table_diagnostics(results, limit=10),
                "retrieved": [
                    make_result_preview(result, rank)
                    for rank, result in enumerate(results, start=1)
                ],
            }
        )

    output = {
        "model_name": model_name,
        "record_count": len(records),
        "query_file": str(args.queries),
        "records_file": str(args.records),
        "embeddings_file": str(args.embeddings),
        "records_sha256": records_sha256,
        "embedded_input_sha256": embedded_input_sha256,
        "answerable_query_count": len(answerable_queries),
        "skipped_queries_by_status": dict(skipped_statuses),
        "skipped_queries": [
            {
                "query_id": query["query_id"],
                "status": query["status"],
                "query": query["query"],
                "annotation_notes": query.get("annotation_notes"),
            }
            for query in queries
            if query["status"] != "answerable"
        ],
        "non_answerable_diagnostics": non_answerable_diagnostics,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "overall": overall_summary,
        "by_question_type": by_question_type,
        "by_question_family": by_question_family,
        "per_query": evaluation_rows,
        "reranker_batch_size": args.reranker_batch_size,
        "reranker_model_name": RERANKER_MODEL_NAME,
    }

    top1_misses = [row for row in evaluation_rows if int(row["metrics"]["hit_at_1"]) == 0]

    write_json(args.results, output)
    write_jsonl(
        args.top1_misses,
        top1_misses,
    )

    print(f"\nAnswerable queries evaluated: {len(answerable_queries)}")
    print(f"Skipped by status: {dict(skipped_statuses)}")
    print_summary(overall_summary)
    print(f"\nFull results: {args.results}")
    print(f"Top-1 misses: {args.top1_misses}")


if __name__ == "__main__":
    main()
