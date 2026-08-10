"""Evaluate grounded single-bank answers against the frozen BankScope questions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from bankscope.config.settings import get_settings
from bankscope.evaluation import (
    evaluate_answer,
    expected_answer_status,
    judge_semantic_answer,
    summarize_answer_metrics,
)
from bankscope.evaluation.semantic_judge import SEMANTIC_JUDGE_PROMPT_VERSION
from bankscope.generation import SingleBankAnswerPipeline
from bankscope.generation.pipeline import (
    DEFAULT_CHUNKS,
    DEFAULT_QDRANT_MANIFEST,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TABLES,
)
from bankscope.io import read_jsonl, sha256_file
from bankscope.llm import create_openai_client
from bankscope.retrieval.qdrant_retriever import DEFAULT_COLLECTION_NAME

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QRELS = ROOT / "data/evaluation/queries.jsonl"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/generation.json"
SUPPORTED_QUERY_STATUSES = {"answerable", "unsupported"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument(
        "--query-id", action="append", help="Evaluate only this query ID; repeatable."
    )
    parser.add_argument("--model", help="Generation model override (defaults to OPENAI_MODEL).")
    parser.add_argument(
        "--judge-model", help="Semantic judge override (defaults to generation model)."
    )
    parser.add_argument("--skip-judge", action="store_true", help="Skip advisory semantic judging.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-manifest", type=Path, default=DEFAULT_QDRANT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_generation_queries(queries: Sequence[Mapping[str, Any]]) -> None:
    if not queries:
        raise ValueError("The evaluation query file is empty.")
    seen: set[str] = set()
    for line_number, query in enumerate(queries, start=1):
        required = {"query_id", "query", "ticker", "status", "relevant_target_chunk_ids"}
        if missing := required - query.keys():
            raise ValueError(f"Query line {line_number} is missing fields: {sorted(missing)}.")
        query_id = str(query["query_id"]).strip()
        if not query_id or not str(query["query"]).strip():
            raise ValueError(f"Query line {line_number} has an empty ID or question.")
        if query_id in seen:
            raise ValueError(f"Duplicate query_id: {query_id}.")
        expected_answer_status(str(query["status"]))
        if query["status"] == "answerable" and not str(query.get("gold_answer") or "").strip():
            raise ValueError(f"Answerable query {query_id} has no gold_answer.")
        seen.add(query_id)


def select_generation_queries(
    queries: Sequence[Mapping[str, Any]], query_ids: Sequence[str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    selected_ids = set(query_ids or [])
    available_ids = {str(query["query_id"]) for query in queries}
    if unknown := selected_ids - available_ids:
        raise ValueError(f"Unknown query ID(s): {sorted(unknown)}.")

    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for raw_query in queries:
        query = dict(raw_query)
        query_id = str(query["query_id"])
        if selected_ids and query_id not in selected_ids:
            continue
        ticker = str(query.get("ticker") or "").strip()
        question_type = str(query.get("question_type") or "")
        status = str(query.get("status") or "")
        reason = None
        if not ticker and question_type == "cross_bank_coverage":
            reason = "cross_bank_out_of_single_bank_scope"
        elif not ticker:
            reason = "missing_ticker_out_of_single_bank_scope"
        elif status not in SUPPORTED_QUERY_STATUSES:
            reason = "status_out_of_single_bank_scope"
        if reason:
            skipped.append({"query_id": query_id, "status": status, "reason": reason})
        else:
            eligible.append(query)
    return eligible, skipped


def _answer_payload(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: output.get(key) for key in ("status", "answer", "reason", "citations", "generation")
    }


def _retrieval_payload(run: Any) -> dict[str, Any]:
    return {
        **run.output["retrieval"],
        "target_chunk_ids": [str(item.get("target_chunk_id") or "") for item in run.evidence],
        "embedding_latency_ms": run.embedding_latency_ms,
        "retrieval_latency_ms": run.retrieval_latency_ms,
        "generation_latency_ms": run.generation_latency_ms,
    }


def evaluate_queries(
    queries: Sequence[Mapping[str, Any]],
    *,
    pipeline: SingleBankAnswerPipeline,
    judge_client: Any,
    judge_model: str,
    skip_judge: bool,
    limit: int,
    candidate_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_id"])
        row: dict[str, Any] = {
            "query_id": query_id,
            "question_type": query.get("question_type"),
            "expected_status": expected_answer_status(str(query["status"])),
        }
        started = perf_counter()
        try:
            run = pipeline.answer(
                str(query["query"]),
                ticker=str(query["ticker"]),
                record_type=(str(query["record_type"]) if query.get("record_type") else None),
                limit=limit,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
            )
            answer = _answer_payload(run.output)
            row["retrieval"] = _retrieval_payload(run)
            row["answer"] = answer
            row["metrics"] = evaluate_answer(query, answer)
            should_judge = (
                not skip_judge
                and query["status"] == "answerable"
                and query.get("expected_value") is None
                and answer["status"] == "supported"
            )
            if should_judge:
                judge_started = perf_counter()
                try:
                    row["judge"] = judge_semantic_answer(
                        question=str(query["query"]),
                        gold_answer=str(query["gold_answer"]),
                        generated_answer=str(answer["answer"]),
                        evidence=run.evidence,
                        client=judge_client,
                        model=judge_model,
                    )
                except Exception as error:
                    row["judge_error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                row["judge_latency_ms"] = (perf_counter() - judge_started) * 1000
        except Exception as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
        row["elapsed_ms"] = (perf_counter() - started) * 1000
        rows.append(row)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive.")
    if args.candidate_k < args.limit:
        raise ValueError("candidate-k must be at least limit.")
    if args.rrf_k <= 0:
        raise ValueError("rrf-k must be positive.")
    if args.query_id and len(args.query_id) != len(set(args.query_id)):
        raise ValueError("query-id must not contain duplicates.")

    queries = read_jsonl(args.qrels)
    validate_generation_queries(queries)
    eligible, skipped = select_generation_queries(queries, args.query_id)
    settings = get_settings()
    generation_model = args.model or settings.openai_model
    judge_model = args.judge_model or generation_model
    rows: list[dict[str, Any]] = []
    dense_model: dict[str, str] = {}
    if eligible:
        client = create_openai_client(settings)
        with SingleBankAnswerPipeline.from_paths(
            client=client,
            generation_model=generation_model,
            temperature=settings.llm_temperature,
            chunks_path=args.chunks,
            tables_path=args.tables,
            qdrant_path=args.qdrant_path,
            qdrant_manifest_path=args.qdrant_manifest,
            collection_name=args.collection,
        ) as pipeline:
            dense_model = pipeline.dense_model
            rows = evaluate_queries(
                eligible,
                pipeline=pipeline,
                judge_client=client,
                judge_model=judge_model,
                skip_judge=args.skip_judge,
                limit=args.limit,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
            )

    output = {
        "format_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": {
            "description": "single-bank questions with an explicit ticker",
            "eligible_count": len(eligible),
            "skipped_count": len(skipped),
            "eligible_by_status": dict(Counter(str(query["status"]) for query in eligible)),
            "skipped": skipped,
        },
        "sources": {
            "qrels": {"path": str(args.qrels), "sha256": sha256_file(args.qrels)},
            "chunks": {"path": str(args.chunks), "sha256": sha256_file(args.chunks)},
            "tables": {"path": str(args.tables), "sha256": sha256_file(args.tables)},
            "qdrant_manifest": {
                "path": str(args.qdrant_manifest),
                "sha256": sha256_file(args.qdrant_manifest),
            },
        },
        "retrieval": {
            "backend": "mixed",
            "mode": "hybrid",
            "limit": args.limit,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "collection": args.collection,
            "dense_model": dense_model,
        },
        "generation": {
            "model": generation_model,
            "temperature": settings.llm_temperature,
        },
        "semantic_judge": {
            "enabled": not args.skip_judge,
            "model": judge_model,
            "prompt_version": SEMANTIC_JUDGE_PROMPT_VERSION,
            "advisory": True,
        },
        "summary": summarize_answer_metrics(rows),
        "per_query": rows,
    }
    write_json(args.output, output)
    print(json.dumps(output["summary"], indent=2))
    print(json.dumps(output["scope"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
