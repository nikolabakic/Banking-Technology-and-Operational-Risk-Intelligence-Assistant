from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
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
METHODS = ("dense", "bm25", "hybrid")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


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

        if status == "answerable":
            answerable_queries.append(query)

    if not answerable_queries:
        raise ValueError("There are no answerable queries to evaluate.")

    return answerable_queries


def make_result_preview(
    result: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    document = " ".join(str(result["document"]).split())

    score = result.get(
        "rrf_score",
        result.get("score"),
    )

    return {
        "rank": rank,
        "target_chunk_id": result["target_chunk_id"],
        "record_id": result["record_id"],
        "ticker": result["ticker"],
        "record_type": result["record_type"],
        "score": score,
        "dense_rank": result.get("dense_rank"),
        "bm25_rank": result.get("bm25_rank"),
        "preview": document[:500],
    }


def retrieve_with_method(
    retriever: HybridRetriever,
    *,
    method: str,
    query: str,
    query_vector: np.ndarray,
    ticker: str | None,
    limit: int,
    candidate_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    if method == "dense":
        return retriever.search_dense(
            query_vector,
            limit=limit,
            ticker=ticker,
        )

    if method == "bm25":
        return retriever.search_bm25(
            query,
            limit=limit,
            ticker=ticker,
        )

    if method == "hybrid":
        return retriever.search_hybrid(
            query,
            query_vector,
            limit=limit,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            ticker=ticker,
        )

    raise ValueError(f"Unknown retrieval method: {method}.")


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
        description=("Evaluate dense, BM25 and hybrid retrieval against BankScope qrels.")
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=QUERIES_PATH,
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_limit = max(DEFAULT_K_VALUES)

    if args.candidate_k < retrieval_limit:
        raise ValueError(f"candidate-k must be at least {retrieval_limit}.")

    if args.rrf_k <= 0:
        raise ValueError("rrf-k must be positive.")

    records = load_jsonl(RECORDS_PATH)
    queries = load_jsonl(args.queries)

    with np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False,
    ) as archive:
        embeddings = archive["embeddings"]
        npz_record_ids = archive["record_ids"].astype(str).tolist()
        model_name = str(archive["model_name"].item())

    jsonl_record_ids = [str(record["record_id"]) for record in records]

    if npz_record_ids != jsonl_record_ids:
        raise ValueError("NPZ vectors and embedding records are not in the same order.")

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
        [str(query["query"]) for query in answerable_queries],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_vectors = np.asarray(
        query_vectors,
        dtype=np.float32,
    )

    evaluation_rows: list[dict[str, Any]] = []

    for query_record, query_vector in zip(
        answerable_queries,
        query_vectors,
        strict=True,
    ):
        query_text = str(query_record["query"])
        ticker_value = query_record.get("ticker")
        ticker = str(ticker_value) if ticker_value else None
        relevant_ids = [
            str(target_chunk_id) for target_chunk_id in query_record["relevant_target_chunk_ids"]
        ]

        for method in METHODS:
            results = retrieve_with_method(
                retriever,
                method=method,
                query=query_text,
                query_vector=query_vector,
                ticker=ticker,
                limit=retrieval_limit,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
            )

            retrieved_ids = [str(result["target_chunk_id"]) for result in results]

            metrics = evaluate_ranking(
                retrieved_ids,
                relevant_ids,
            )

            evaluation_rows.append(
                {
                    "query_id": query_record["query_id"],
                    "query": query_text,
                    "ticker": ticker,
                    "question_type": query_record["question_type"],
                    "method": method,
                    "relevant_target_chunk_ids": (relevant_ids),
                    "metrics": metrics,
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

    skipped_statuses = Counter(
        str(query["status"]) for query in queries if query["status"] != "answerable"
    )

    output = {
        "model_name": model_name,
        "record_count": len(records),
        "query_file": str(args.queries),
        "answerable_query_count": len(answerable_queries),
        "skipped_queries_by_status": dict(skipped_statuses),
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "overall": overall_summary,
        "by_question_type": by_question_type,
        "per_query": evaluation_rows,
    }

    top1_misses = [row for row in evaluation_rows if int(row["metrics"]["hit_at_1"]) == 0]

    write_json(RESULTS_PATH, output)
    write_jsonl(
        TOP1_MISSES_PATH,
        top1_misses,
    )

    print(f"\nAnswerable queries evaluated: {len(answerable_queries)}")
    print_summary(overall_summary)
    print(f"\nFull results: {RESULTS_PATH}")
    print(f"Top-1 misses: {TOP1_MISSES_PATH}")


if __name__ == "__main__":
    main()
