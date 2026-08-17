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
from bankscope.generation import GPT51_CANDIDATE_MODEL, SingleBankAnswerPipeline
from bankscope.generation.answer_generator import (
    ANSWER_PROMPT_VERSION,
    ANSWER_RESPONSE_FORMAT,
    ANSWER_SCHEMA_VERSION,
)
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
DEFAULT_QRELS = ROOT / "data/evaluation/queries.jsonl"
DEFAULT_CITATION_AUDIT = ROOT / "data/evaluation/generation_citation_audit_v2.jsonl"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/generation-gpt51-json-v2.json"
DEFAULT_JUDGE_MODEL = "AZURE_GPT_4o_2024_1120"
CITATION_AUDIT_VERSION = 2
SUPPORTED_QUERY_STATUSES = {"answerable", "unsupported"}
AUDIT_CLASSIFICATIONS = {
    "accepted_additional_evidence",
    "relevant_but_insufficient_precision",
    "does_not_support_claim",
}
EXPECTED_GATE_COUNTS = {
    "queries": 30,
    "supported": 29,
    "numeric": 17,
    "narrative": 12,
    "variant": 11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--citation-audit", type=Path, default=DEFAULT_CITATION_AUDIT)
    parser.add_argument(
        "--query-id", action="append", help="Evaluate only this query ID; repeatable."
    )
    parser.add_argument("--model", help="Generation model override (defaults to OPENAI_MODEL).")
    parser.add_argument(
        "--judge-model",
        help=f"Semantic judge override (defaults to {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument("--skip-judge", action="store_true", help="Skip advisory semantic judging.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--glossary-locators", type=Path, default=DEFAULT_GLOSSARY_LOCATORS)
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


def validate_output_model(
    output: Path, generation_model: str, query_ids: Sequence[str] | None = None
) -> None:
    if output.resolve() == DEFAULT_OUTPUT.resolve() and generation_model != GPT51_CANDIDATE_MODEL:
        raise ValueError(
            f"The default output is reserved for {GPT51_CANDIDATE_MODEL}; pass that model with "
            "--model or choose an explicit --output path."
        )
    if output.resolve() == DEFAULT_OUTPUT.resolve() and query_ids:
        raise ValueError(
            "A filtered run cannot use the full candidate output path; choose an explicit "
            "--output path."
        )


def validate_citation_audit(
    records: Sequence[Mapping[str, Any]], queries: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    query_ids = {str(query["query_id"]) for query in queries}
    by_query: dict[str, dict[str, Any]] = {}
    for line_number, raw_record in enumerate(records, start=1):
        record = dict(raw_record)
        if record.get("audit_version") != CITATION_AUDIT_VERSION:
            raise ValueError(
                f"Citation audit line {line_number} must use version {CITATION_AUDIT_VERSION}."
            )
        query_id = str(record.get("query_id") or "").strip()
        if not query_id:
            raise ValueError(f"Citation audit line {line_number} has no query_id.")
        if query_id not in query_ids:
            raise ValueError(f"Citation audit references unknown query_id: {query_id}.")
        if query_id in by_query:
            raise ValueError(f"Duplicate citation audit query_id: {query_id}.")
        accepted = record.get("accepted_additional_target_chunk_ids")
        reviewed = record.get("reviewed_targets")
        if not isinstance(accepted, list) or not isinstance(reviewed, list):
            raise ValueError(
                f"Citation audit {query_id} needs accepted_additional_target_chunk_ids and "
                "reviewed_targets lists."
            )
        accepted_ids = [str(value).strip() for value in accepted]
        if any(not value for value in accepted_ids) or len(accepted_ids) != len(set(accepted_ids)):
            raise ValueError(f"Citation audit {query_id} has invalid accepted target IDs.")
        reviewed_ids: set[str] = set()
        accepted_reviewed_ids: set[str] = set()
        for item in reviewed:
            if not isinstance(item, Mapping):
                raise ValueError(f"Citation audit {query_id} has an invalid reviewed target.")
            target_id = str(item.get("target_chunk_id") or "").strip()
            classification = str(item.get("classification") or "").strip()
            note = str(item.get("note") or "").strip()
            if not target_id or not note or classification not in AUDIT_CLASSIFICATIONS:
                raise ValueError(f"Citation audit {query_id} has an invalid review entry.")
            if target_id in reviewed_ids:
                raise ValueError(f"Citation audit {query_id} repeats reviewed target {target_id}.")
            reviewed_ids.add(target_id)
            if classification == "accepted_additional_evidence":
                accepted_reviewed_ids.add(target_id)
        if set(accepted_ids) != accepted_reviewed_ids:
            raise ValueError(
                f"Citation audit {query_id} accepted IDs do not match accepted review entries."
            )
        by_query[query_id] = record
    return by_query


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
            "bank_resolution",
        )
    }


def _retrieval_payload(run: Any) -> dict[str, Any]:
    return {
        **run.output["retrieval"],
        "target_chunk_ids": [str(item.get("target_chunk_id") or "") for item in run.evidence],
        "embedding_latency_ms": run.embedding_latency_ms,
        "retrieval_latency_ms": run.retrieval_latency_ms,
        "generation_latency_ms": run.generation_latency_ms,
    }


def _error_payload(error: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": type(error).__name__, "message": str(error)}
    code = getattr(error, "code", None)
    if code:
        payload["code"] = str(code)
    generation = getattr(error, "generation", None)
    if isinstance(generation, Mapping) and generation:
        payload["generation"] = dict(generation)
    return payload


def _cited_evidence(
    answer: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    raw_citations = answer.get("citations")
    citations = (
        raw_citations
        if isinstance(raw_citations, Sequence) and not isinstance(raw_citations, str)
        else []
    )
    cited_ids = {
        str(citation.get("target_chunk_id") or "")
        for citation in citations
        if isinstance(citation, Mapping)
    }
    return [dict(item) for item in evidence if str(item.get("target_chunk_id") or "") in cited_ids]


def assess_generation_quality_gate(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> dict[str, Any]:
    supported_rows = [
        row
        for row in rows
        if isinstance(row.get("metrics"), Mapping)
        and row["metrics"].get("expected_status") == "supported"
    ]
    numeric_rows = [row for row in supported_rows if row.get("expected_numeric") is True]
    narrative_rows = [row for row in supported_rows if row.get("expected_numeric") is False]
    exact_numeric_support = sum(
        isinstance(row.get("answer"), Mapping)
        and row["answer"].get("answer_type") == "numeric"
        and isinstance(row["answer"].get("generation"), Mapping)
        and "numeric_value_in_cited_evidence"
        in row["answer"]["generation"].get("validation_checks", [])
        for row in numeric_rows
    )
    grounded_narratives = sum(
        isinstance(row.get("judge"), Mapping)
        and row["judge"].get("groundedness") is True
        and not row.get("judge_error")
        for row in narrative_rows
    )
    citations_inside_contract = sum(
        row["metrics"]["citations"].get("citation_support_precision") == 1.0
        for row in supported_rows
    )

    checks = {
        "all_queries_error_free": {
            "actual": summary.get("evaluated_count"),
            "expected": EXPECTED_GATE_COUNTS["queries"],
            "passed": (
                len(rows) == EXPECTED_GATE_COUNTS["queries"]
                and summary.get("evaluated_count") == EXPECTED_GATE_COUNTS["queries"]
                and summary.get("error_count") == 0
            ),
        },
        "status": {
            "actual": summary.get("status_accuracy"),
            "expected": 1.0,
            "passed": summary.get("status_accuracy") == 1.0,
        },
    }
    for name, expected_count in (
        ("value_match", EXPECTED_GATE_COUNTS["numeric"]),
        ("unit_match", EXPECTED_GATE_COUNTS["numeric"]),
        ("period_match", EXPECTED_GATE_COUNTS["numeric"]),
        ("entity_match", EXPECTED_GATE_COUNTS["numeric"]),
        ("variant_match", EXPECTED_GATE_COUNTS["variant"]),
    ):
        count = summary.get(f"{name}_count")
        rate = summary.get(f"{name}_rate")
        checks[name] = {
            "actual": {"count": count, "rate": rate},
            "expected": {"count": expected_count, "rate": 1.0},
            "passed": count == expected_count and rate == 1.0,
        }
    checks["exact_numeric_cited_support"] = {
        "actual": exact_numeric_support,
        "expected": EXPECTED_GATE_COUNTS["numeric"],
        "passed": exact_numeric_support == EXPECTED_GATE_COUNTS["numeric"],
    }
    checks["narrative_groundedness"] = {
        "actual": grounded_narratives,
        "expected": EXPECTED_GATE_COUNTS["narrative"],
        "passed": (
            len(narrative_rows) == EXPECTED_GATE_COUNTS["narrative"]
            and grounded_narratives == EXPECTED_GATE_COUNTS["narrative"]
        ),
    }
    checks["citations_inside_qrel_or_audit_contract"] = {
        "actual": citations_inside_contract,
        "expected": EXPECTED_GATE_COUNTS["supported"],
        "passed": (
            len(supported_rows) == EXPECTED_GATE_COUNTS["supported"]
            and citations_inside_contract == EXPECTED_GATE_COUNTS["supported"]
        ),
    }
    request_count = summary.get("generation_request_count")
    checks["generation_request_budget"] = {
        "actual": request_count,
        "expected": f"at most {EXPECTED_GATE_COUNTS['supported']}",
        "passed": isinstance(request_count, int)
        and request_count <= EXPECTED_GATE_COUNTS["supported"],
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def evaluate_queries(
    queries: Sequence[Mapping[str, Any]],
    *,
    pipeline: SingleBankAnswerPipeline,
    judge_client: Any,
    judge_model: str,
    skip_judge: bool,
    citation_audit: Mapping[str, Mapping[str, Any]],
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
            "expected_numeric": query.get("expected_value") is not None,
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
            row["metrics"] = evaluate_answer(
                query, answer, citation_audit=citation_audit.get(query_id)
            )
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
                        evidence=_cited_evidence(answer, run.evidence),
                        client=judge_client,
                        model=judge_model,
                    )
                except Exception as error:
                    row["judge_error"] = _error_payload(error)
                row["judge_latency_ms"] = (perf_counter() - judge_started) * 1000
        except Exception as error:
            row["error"] = _error_payload(error)
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
    citation_audit_records = read_jsonl(args.citation_audit)
    citation_audit = validate_citation_audit(citation_audit_records, queries)
    eligible, skipped = select_generation_queries(queries, args.query_id)
    settings = get_settings()
    generation_model = args.model or settings.openai_model
    validate_output_model(args.output, generation_model, args.query_id)
    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL
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
            glossary_locators_path=args.glossary_locators,
            qdrant_path=args.qdrant_path,
            qdrant_manifest_path=args.qdrant_manifest,
            collection_name=args.collection,
            bank_registry_path=settings.bank_registry_path,
        ) as pipeline:
            dense_model = pipeline.dense_model
            rows = evaluate_queries(
                eligible,
                pipeline=pipeline,
                judge_client=client,
                judge_model=judge_model,
                skip_judge=args.skip_judge,
                citation_audit=citation_audit,
                limit=args.limit,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
            )

    summary = summarize_answer_metrics(rows)
    output = {
        "format_version": 2,
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
            "citation_audit": {
                "path": str(args.citation_audit),
                "sha256": sha256_file(args.citation_audit),
                "version": CITATION_AUDIT_VERSION,
            },
            "chunks": {"path": str(args.chunks), "sha256": sha256_file(args.chunks)},
            "tables": {"path": str(args.tables), "sha256": sha256_file(args.tables)},
            "glossary_locators": {
                "path": str(args.glossary_locators),
                "sha256": sha256_file(args.glossary_locators),
            },
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
            "prompt_version": ANSWER_PROMPT_VERSION,
            "schema_version": ANSWER_SCHEMA_VERSION,
            "response_format": ANSWER_RESPONSE_FORMAT,
            "one_call_per_question": True,
        },
        "semantic_judge": {
            "enabled": not args.skip_judge,
            "model": judge_model,
            "prompt_version": SEMANTIC_JUDGE_PROMPT_VERSION,
            "advisory": True,
        },
        "summary": summary,
        "quality_gate": assess_generation_quality_gate(rows, summary),
        "per_query": rows,
    }
    write_json(args.output, output)
    print(json.dumps(output["summary"], indent=2))
    print(json.dumps(output["scope"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
