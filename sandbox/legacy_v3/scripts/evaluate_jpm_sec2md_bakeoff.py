from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from bankscope.evaluation.retrieval_metrics import evaluate_ranking
from bankscope.parsing.sec2md_adapter import looks_like_page_furniture
from bankscope.retrieval.hybrid_retriever import HybridRetriever
from bankscope.retrieval.reranker import (
    RERANKER_MODEL_NAME,
    load_reranker,
    rerank_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "data/experiments/jpm_sec2md"
OUTPUT_PATH = EXPERIMENT_DIR / "retrieval_bakeoff.json"

VARIANT_PATHS = {
    "sec2md_builtin": {
        "records": EXPERIMENT_DIR / "sec2md_builtin/embedding_records.jsonl",
        "embeddings": EXPERIMENT_DIR / "sec2md_builtin/embeddings.npz",
        "queries": EXPERIMENT_DIR / "sec2md_builtin/queries.jsonl",
    },
    "structure_aware": {
        "records": EXPERIMENT_DIR / "structure_aware/embedding_records.jsonl",
        "embeddings": EXPERIMENT_DIR / "structure_aware/embeddings.npz",
        "queries": EXPERIMENT_DIR / "structure_aware/queries.jsonl",
    },
}

METHODS = ("dense", "bm25", "hybrid", "reranked")
ANSWERABLE_QUERY_IDS = {
    "dev_jpm_cybersecurity_risk_definition_2025",
    "dev_jpm_standardized_cet1_ratio_2025",
    "dev_jpm_standardized_cet1_requirement_2025",
    "dev_jpm_bank_advanced_cet1_ratio_2025_metadata",
}
CET1_SELF_CONTAINMENT_ANCHORS = {
    "dev_jpm_standardized_cet1_ratio_2025": (
        "December 31, 2025",
        "Standardized",
        "CET1",
        "14.6",
    ),
    "dev_jpm_bank_advanced_cet1_ratio_2025_metadata": (
        "December 31, 2025",
        "Advanced",
        "JPMorgan Chase Bank, N.A.",
        "CET1",
        "15.8",
    ),
}

Record = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the JPM sec2md built-in and BankScope structure-aware chunk variants."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--reranker-device", help="Device such as cuda, cuda:0, or cpu.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_jsonl(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_variant_paths(experiment_dir: Path, variant: str) -> dict[str, Path]:
    return {
        "records": experiment_dir / variant / "embedding_records.jsonl",
        "embeddings": experiment_dir / variant / "embeddings.npz",
        "queries": experiment_dir / variant / "queries.jsonl",
    }


def validate_queries(queries: list[Record], records: list[Record], variant: str) -> None:
    if {str(query["query_id"]) for query in queries if query["status"] == "answerable"} != (
        ANSWERABLE_QUERY_IDS
    ):
        raise ValueError(f"Unexpected answerable JPM query set for {variant}.")

    corpus_ids = {str(record["target_chunk_id"]) for record in records}

    for query in queries:
        query_id = str(query["query_id"])
        relevant_ids = [str(value) for value in query["relevant_target_chunk_ids"]]

        if query["status"] == "answerable" and not relevant_ids:
            raise ValueError(f"Answerable query has no qrels: {query_id}.")

        if query["status"] == "unsupported" and relevant_ids:
            raise ValueError(f"Unsupported query has qrels: {query_id}.")

        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError(f"Duplicate qrel IDs: {query_id}.")

        unknown_ids = set(relevant_ids) - corpus_ids

        if unknown_ids:
            raise ValueError(f"Unknown qrel IDs for {query_id}: {sorted(unknown_ids)}.")

        primary_id = query.get("primary_target_chunk_id")

        if primary_id is not None and str(primary_id) not in relevant_ids:
            raise ValueError(f"Primary qrel is not relevant: {query_id}.")


def load_variant(
    experiment_dir: Path,
    variant: str,
) -> tuple[list[Record], np.ndarray, list[Record], str, Record]:
    paths = get_variant_paths(experiment_dir, variant)
    records = load_jsonl(paths["records"])
    queries = load_jsonl(paths["queries"])

    with np.load(paths["embeddings"], allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        record_ids = archive["record_ids"].astype(str).tolist()
        model_name = str(archive["model_name"].item())

    jsonl_record_ids = [str(record["record_id"]) for record in records]

    if record_ids != jsonl_record_ids:
        raise ValueError(f"NPZ and JSONL record order differs for {variant}.")

    if embeddings.shape[0] != len(records):
        raise ValueError(f"Embedding count differs for {variant}.")

    validate_queries(queries, records, variant)
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    return records, embeddings, queries, model_name, hashes


def result_preview(result: Record, rank: int) -> Record:
    return {
        "rank": rank,
        "target_chunk_id": result["target_chunk_id"],
        "record_id": result["record_id"],
        "record_type": result["record_type"],
        "dense_rank": result.get("dense_rank"),
        "bm25_rank": result.get("bm25_rank"),
        "rrf_score": result.get("rrf_score"),
        "reranker_score": result.get("reranker_score"),
        "page_start": result["metadata"].get("page_start"),
        "page_end": result["metadata"].get("page_end"),
        "parent_id": result["metadata"].get("parent_id"),
        "preview": " ".join(str(result["document"]).split())[:700],
    }


def minimum_relevant_rank(results: list[Record], relevant_ids: list[str]) -> int | None:
    relevant = set(relevant_ids)

    for rank, result in enumerate(results, start=1):
        if str(result["target_chunk_id"]) in relevant:
            return rank

    return None


def retrieve_methods(
    retriever: HybridRetriever,
    *,
    query: str,
    query_vector: np.ndarray,
    candidate_k: int,
    rrf_k: int,
    reranker: Any,
    reranker_batch_size: int,
) -> tuple[dict[str, list[Record]], list[Record], Record]:
    timings: Record = {}
    method_results: dict[str, list[Record]] = {}

    started = perf_counter()
    method_results["dense"] = retriever.search_dense(
        query_vector,
        limit=10,
        ticker="JPM",
    )
    timings["dense_seconds"] = perf_counter() - started

    started = perf_counter()
    method_results["bm25"] = retriever.search_bm25(
        query,
        limit=10,
        ticker="JPM",
    )
    timings["bm25_seconds"] = perf_counter() - started

    started = perf_counter()
    method_results["hybrid"] = retriever.search_hybrid(
        query,
        query_vector,
        limit=10,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        ticker="JPM",
    )
    timings["hybrid_seconds"] = perf_counter() - started

    started = perf_counter()
    candidates = retriever.get_hybrid_candidates(
        query,
        query_vector,
        candidate_k=candidate_k,
        rrf_pool_size=20,
        per_method_limit=5,
        max_candidates=30,
        rrf_k=rrf_k,
        ticker="JPM",
    )
    timings["candidate_retrieval_seconds"] = perf_counter() - started

    if len(candidates) != 30:
        raise ValueError(
            f"The reranker candidate pool must contain exactly 30 records, got {len(candidates)}."
        )

    started = perf_counter()
    method_results["reranked"] = rerank_candidates(
        reranker,
        query,
        candidates,
        limit=10,
        batch_size=reranker_batch_size,
    )
    timings["reranking_seconds"] = perf_counter() - started
    return method_results, candidates, timings


def summarize_answerable(rows: list[Record]) -> Record:
    method_rows: dict[str, list[Record]] = defaultdict(list)

    for row in rows:
        if row["status"] != "answerable":
            continue

        for method, result in row["methods"].items():
            method_rows[method].append(result)

    summary: Record = {}

    for method in METHODS:
        results = method_rows[method]
        summary[method] = {
            "hit_rate_at_1": fmean(float(result["metrics"]["hit_at_1"]) for result in results),
            "hit_rate_at_5": fmean(float(result["metrics"]["hit_at_5"]) for result in results),
            "hit_rate_at_10": fmean(float(result["metrics"]["hit_at_10"]) for result in results),
            "mrr_at_10": fmean(
                float(result["metrics"]["reciprocal_rank_at_10"]) for result in results
            ),
        }

    summary["candidate_hit_rate_at_30"] = fmean(
        float(row["candidate_hit_at_30"]) for row in rows if row["status"] == "answerable"
    )
    return summary


def normalize_for_match(value: str) -> str:
    return " ".join(value.split()).casefold()


def self_containment_pass(
    query_id: str,
    query: Record,
    records_by_target: dict[str, Record],
) -> bool:
    anchors = CET1_SELF_CONTAINMENT_ANCHORS[query_id]

    for target_id in query["relevant_target_chunk_ids"]:
        record = records_by_target[str(target_id)]
        document = normalize_for_match(str(record["document"]))
        metadata = record["metadata"]

        if (
            all(normalize_for_match(anchor) in document for anchor in anchors)
            and metadata.get("page_start")
            and metadata.get("page_end")
        ):
            return True

    return False


def build_gates(
    rows: list[Record],
    queries: list[Record],
    records: list[Record],
) -> Record:
    answerable_rows = {str(row["query_id"]): row for row in rows if row["status"] == "answerable"}
    records_by_target = {str(record["target_chunk_id"]): record for record in records}
    queries_by_id = {str(query["query_id"]): query for query in queries}
    cyber_row = answerable_rows["dev_jpm_cybersecurity_risk_definition_2025"]
    unsupported_rows = [row for row in rows if row["status"] == "unsupported"]

    gates = {
        "candidate_pool_is_exactly_30": all(
            int(row["candidate_count"]) == 30 for row in answerable_rows.values()
        ),
        "all_four_proofs_enter_pool_30": all(
            bool(row["candidate_hit_at_30"]) for row in answerable_rows.values()
        ),
        "cybersecurity_dense_rank_not_worse_than_3": (
            cyber_row["methods"]["dense"]["minimum_relevant_rank"] is not None
            and int(cyber_row["methods"]["dense"]["minimum_relevant_rank"]) <= 3
        ),
        "standardized_cet1_is_self_contained": self_containment_pass(
            "dev_jpm_standardized_cet1_ratio_2025",
            queries_by_id["dev_jpm_standardized_cet1_ratio_2025"],
            records_by_target,
        ),
        "advanced_bank_cet1_is_self_contained": self_containment_pass(
            "dev_jpm_bank_advanced_cet1_ratio_2025_metadata",
            queries_by_id["dev_jpm_bank_advanced_cet1_ratio_2025_metadata"],
            records_by_target,
        ),
        "unsupported_2026_has_no_matching_report_period": all(
            int(row["matching_report_period_count_at_10"]) == 0 for row in unsupported_rows
        ),
        "no_page_furniture_in_candidate_pools": all(
            int(row["page_furniture_candidate_count"]) == 0 for row in rows
        ),
        "all_qrels_have_page_provenance": all(
            record["metadata"].get("page_start") and record["metadata"].get("page_end")
            for query in queries
            for target_id in query["relevant_target_chunk_ids"]
            for record in [records_by_target[str(target_id)]]
        ),
    }
    gates["pass"] = all(gates.values())
    return gates


def evaluate_variant(
    variant: str,
    records: list[Record],
    embeddings: np.ndarray,
    queries: list[Record],
    query_vectors: dict[str, np.ndarray],
    *,
    candidate_k: int,
    rrf_k: int,
    reranker: Any,
    reranker_batch_size: int,
) -> Record:
    retriever = HybridRetriever(records, embeddings)
    rows: list[Record] = []

    for query in queries:
        query_id = str(query["query_id"])
        relevant_ids = [str(value) for value in query["relevant_target_chunk_ids"]]
        method_results, candidates, timings = retrieve_methods(
            retriever,
            query=str(query["query"]),
            query_vector=query_vectors[query_id],
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            reranker=reranker,
            reranker_batch_size=reranker_batch_size,
        )
        candidate_ids = [str(result["target_chunk_id"]) for result in candidates]
        row: Record = {
            "query_id": query_id,
            "query": query["query"],
            "status": query["status"],
            "candidate_count": len(candidates),
            "page_furniture_candidate_count": sum(
                looks_like_page_furniture(str(result["document"])) for result in candidates
            ),
            "timings": timings,
            "candidates": [
                result_preview(result, rank) for rank, result in enumerate(candidates, start=1)
            ],
            "methods": {},
        }

        if query["status"] == "answerable":
            candidate_metrics = evaluate_ranking(
                candidate_ids,
                relevant_ids,
                k_values=(30,),
                reciprocal_rank_limit=30,
            )
            row["candidate_hit_at_30"] = candidate_metrics["hit_at_30"]
            row["candidate_recall_at_30"] = candidate_metrics["recall_at_30"]

            for method, results in method_results.items():
                retrieved_ids = [str(result["target_chunk_id"]) for result in results]
                row["methods"][method] = {
                    "minimum_relevant_rank": minimum_relevant_rank(results, relevant_ids),
                    "metrics": evaluate_ranking(retrieved_ids, relevant_ids),
                    "retrieved": [
                        result_preview(result, rank) for rank, result in enumerate(results, start=1)
                    ],
                }
        else:
            expected_period = str(query["expected_period"])
            row["candidate_hit_at_30"] = None
            row["candidate_recall_at_30"] = None
            row["matching_report_period_count_at_10"] = sum(
                str(result["metadata"].get("report_date")) == expected_period
                for result in method_results["reranked"]
            )

            for method, results in method_results.items():
                row["methods"][method] = {
                    "retrieved": [
                        result_preview(result, rank) for rank, result in enumerate(results, start=1)
                    ]
                }

        rows.append(row)

    return {
        "variant": variant,
        "record_count": len(records),
        "summary": summarize_answerable(rows),
        "gates": build_gates(rows, queries, records),
        "per_query": rows,
    }


def main() -> None:
    args = parse_args()

    if args.candidate_k != 30:
        raise ValueError("This locked bake-off requires --candidate-k 30.")

    loaded: dict[str, tuple[list[Record], np.ndarray, list[Record], str, Record]] = {}

    for variant in VARIANT_PATHS:
        loaded[variant] = load_variant(args.experiment_dir, variant)

    model_names = {values[3] for values in loaded.values()}

    if len(model_names) != 1:
        raise ValueError(f"Both variants must use the same embedding model: {model_names}.")

    model_name = model_names.pop()
    reference_queries = loaded["sec2md_builtin"][2]
    query_text_by_id = {str(query["query_id"]): str(query["query"]) for query in reference_queries}

    for variant, values in loaded.items():
        variant_query_text = {str(query["query_id"]): str(query["query"]) for query in values[2]}

        if variant_query_text != query_text_by_id:
            raise ValueError(f"Query text differs between variants: {variant}.")

    model = SentenceTransformer(model_name)
    query_ids = list(query_text_by_id)
    encoded_queries = model.encode_query(
        [query_text_by_id[query_id] for query_id in query_ids],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_vectors = {
        query_id: np.asarray(vector, dtype=np.float32)
        for query_id, vector in zip(query_ids, encoded_queries, strict=True)
    }
    del model

    reranker = load_reranker(device=args.reranker_device)
    variant_results: dict[str, Record] = {}
    input_hashes: dict[str, Record] = {}

    for variant, (records, embeddings, queries, _, hashes) in loaded.items():
        variant_results[variant] = evaluate_variant(
            variant,
            records,
            embeddings,
            queries,
            query_vectors,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
            reranker=reranker,
            reranker_batch_size=args.reranker_batch_size,
        )
        input_hashes[variant] = hashes

    output = {
        "experiment": "jpm_sec2md_chunk_bakeoff_v1",
        "embedding_model_name": model_name,
        "reranker_model_name": RERANKER_MODEL_NAME,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "reranker_batch_size": args.reranker_batch_size,
        "input_sha256": input_hashes,
        "variants": variant_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for variant, result in variant_results.items():
        summary = result["summary"]["reranked"]
        print(
            f"{variant}: candidate_hit@30={result['summary']['candidate_hit_rate_at_30']:.3f}, "
            f"Hit@5={summary['hit_rate_at_5']:.3f}, "
            f"MRR@10={summary['mrr_at_10']:.3f}, "
            f"gates={result['gates']['pass']}"
        )

    print(f"Full comparison: {args.output}")


if __name__ == "__main__":
    main()
