"""Run the separate BankScope evidence-audit challenge without changing frozen baselines."""

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
from bankscope.evaluation import evaluate_answer, expected_answer_status, summarize_answer_metrics
from bankscope.generation import BankAnswerPipeline
from bankscope.generation.pipeline import (
    DEFAULT_CHUNKS,
    DEFAULT_GLOSSARY_LOCATORS,
    DEFAULT_QDRANT_MANIFEST,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TABLES,
)
from bankscope.io import read_jsonl, sha256_file
from bankscope.llm import create_openai_client
from bankscope.retrieval.qdrant_retriever import DEFAULT_COLLECTION_NAME

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHALLENGE = ROOT / "data/evaluation/evidence_audit_challenge_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/evidence-audit-challenge-v1.json"
EXPECTED_CATEGORIES = {
    "unsupported_or_missing_period": 2,
    "ambiguous": 2,
    "numeric_entity_period_variant": 2,
    "narrative_multi_claim": 2,
    "citation_evidence_trap": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", help="Generation and runtime-audit model override.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--glossary-locators", type=Path, default=DEFAULT_GLOSSARY_LOCATORS)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-manifest", type=Path, default=DEFAULT_QDRANT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    return parser.parse_args()


def _string_ids(value: Any, *, field: str, query_id: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Challenge {query_id} must define {field} as a list.")
    ids = [str(item).strip() for item in value]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Challenge {query_id} has invalid {field}.")
    return ids


def validate_challenge(queries: Sequence[Mapping[str, Any]], corpus_ids: set[str]) -> None:
    if len(queries) != sum(EXPECTED_CATEGORIES.values()):
        raise ValueError("Evidence-audit challenge v1 must contain exactly 10 cases.")
    seen: set[str] = set()
    categories: Counter[str] = Counter()
    for line_number, query in enumerate(queries, start=1):
        required = {
            "query_id",
            "query",
            "challenge_category",
            "status",
            "relevant_target_chunk_ids",
            "annotation_notes",
        }
        if missing := required - query.keys():
            raise ValueError(f"Challenge line {line_number} is missing: {sorted(missing)}.")
        query_id = str(query["query_id"]).strip()
        question = str(query["query"]).strip()
        category = str(query["challenge_category"]).strip()
        status = str(query["status"]).strip()
        if not query_id or not question or not str(query["annotation_notes"]).strip():
            raise ValueError(f"Challenge line {line_number} has an empty required string.")
        if query_id in seen:
            raise ValueError(f"Duplicate challenge query_id: {query_id}.")
        expected_answer_status(status)
        relevant = _string_ids(
            query["relevant_target_chunk_ids"],
            field="relevant_target_chunk_ids",
            query_id=query_id,
        )
        traps = _string_ids(
            query.get("trap_target_chunk_ids", []),
            field="trap_target_chunk_ids",
            query_id=query_id,
        )
        if status == "answerable" and (not relevant or not str(query.get("gold_answer") or "")):
            raise ValueError(
                f"Answerable challenge {query_id} needs qrels and a manual gold answer."
            )
        if status != "answerable" and relevant:
            raise ValueError(f"Non-answerable challenge {query_id} cannot define relevant IDs.")
        if set(relevant) & set(traps):
            raise ValueError(f"Challenge {query_id} overlaps relevant and trap IDs.")
        if unknown := (set(relevant) | set(traps)) - corpus_ids:
            raise ValueError(
                f"Challenge {query_id} references unknown corpus IDs: {sorted(unknown)}."
            )
        seen.add(query_id)
        categories[category] += 1
    if dict(categories) != EXPECTED_CATEGORIES:
        raise ValueError(
            "Evidence-audit challenge categories must be "
            f"{EXPECTED_CATEGORIES}; got {dict(categories)}."
        )


def _answer_payload(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: output.get(key)
        for key in (
            "status",
            "answer_type",
            "answer",
            "facts",
            "reason",
            "citations",
            "generation",
            "evidence_audit",
        )
    }


def evaluate_challenge(
    queries: Sequence[Mapping[str, Any]],
    *,
    pipeline: BankAnswerPipeline,
    limit: int,
    candidate_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        started = perf_counter()
        row: dict[str, Any] = {
            "query_id": query["query_id"],
            "challenge_category": query["challenge_category"],
            "required_claims": list(query.get("required_claims") or []),
        }
        try:
            ticker = str(query.get("ticker") or "").strip() or None
            run = pipeline.answer(
                str(query["query"]),
                ticker=ticker,
                record_type=str(query["record_type"]) if query.get("record_type") else None,
                limit=limit,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
            )
            answer = _answer_payload(run.output)
            cited_ids = {
                str(citation.get("target_chunk_id") or "")
                for citation in answer.get("citations") or []
                if isinstance(citation, Mapping)
            }
            trap_ids = {str(value) for value in query.get("trap_target_chunk_ids") or []}
            row["answer"] = answer
            row["metrics"] = evaluate_answer(query, answer)
            row["trap_citation_ids"] = sorted(cited_ids & trap_ids)
            row["trap_avoidance"] = not row["trap_citation_ids"]
            row["retrieved_target_chunk_ids"] = [
                str(item.get("target_chunk_id") or "") for item in run.evidence
            ]
        except Exception as error:
            row["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "code": str(getattr(error, "code", "") or "") or None,
            }
        row["elapsed_ms"] = (perf_counter() - started) * 1000
        rows.append(row)
    return rows


def summarize_challenge(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = summarize_answer_metrics(rows)
    summary["category_counts"] = dict(Counter(str(row["challenge_category"]) for row in rows))
    trap_rows = [row for row in rows if row["challenge_category"] == "citation_evidence_trap"]
    summary["trap_avoidance_count"] = sum(row.get("trap_avoidance") is True for row in trap_rows)
    summary["trap_case_count"] = len(trap_rows)
    audits = [
        row["answer"]["evidence_audit"]
        for row in rows
        if isinstance(row.get("answer"), Mapping)
        and isinstance(row["answer"].get("evidence_audit"), Mapping)
    ]
    summary["evidence_audit_status_counts"] = dict(
        Counter(str(audit.get("status") or "missing") for audit in audits)
    )
    summary["manual_claim_review_case_count"] = sum(
        bool(row.get("required_claims")) for row in rows
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.candidate_k < args.limit or args.rrf_k <= 0:
        raise ValueError("Require limit > 0, candidate-k >= limit, and rrf-k > 0.")
    queries = read_jsonl(args.challenge)
    records = read_jsonl(args.chunks)
    validate_challenge(queries, {str(record.get("target_chunk_id") or "") for record in records})
    settings = get_settings()
    model = args.model or settings.openai_model
    client = create_openai_client(settings)
    with BankAnswerPipeline.from_paths(
        client=client,
        generation_model=model,
        temperature=settings.llm_temperature,
        chunks_path=args.chunks,
        tables_path=args.tables,
        glossary_locators_path=args.glossary_locators,
        qdrant_path=args.qdrant_path,
        qdrant_manifest_path=args.qdrant_manifest,
        collection_name=args.collection,
        bank_registry_path=settings.bank_registry_path,
    ) as pipeline:
        rows = evaluate_challenge(
            queries,
            pipeline=pipeline,
            limit=args.limit,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )
    output = {
        "format_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "scope": "descriptive challenge; does not change or gate the frozen 34-query baseline",
        "sources": {
            "challenge": {"path": str(args.challenge), "sha256": sha256_file(args.challenge)},
            "chunks": {"path": str(args.chunks), "sha256": sha256_file(args.chunks)},
            "tables": {"path": str(args.tables), "sha256": sha256_file(args.tables)},
            "qdrant_manifest": {
                "path": str(args.qdrant_manifest),
                "sha256": sha256_file(args.qdrant_manifest),
            },
        },
        "summary": summarize_challenge(rows),
        "per_query": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
