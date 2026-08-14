"""Evaluate the three frozen cross-bank BankScope questions."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bankscope.config.settings import get_settings  # noqa: E402
from bankscope.evaluation import judge_semantic_answer  # noqa: E402
from bankscope.generation import GPT51_CANDIDATE_MODEL, BankAnswerPipeline  # noqa: E402
from bankscope.generation.pipeline import (  # noqa: E402
    DEFAULT_CHUNKS,
    DEFAULT_GLOSSARY_LOCATORS,
    DEFAULT_QDRANT_MANIFEST,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TABLES,
)
from bankscope.io import read_jsonl  # noqa: E402
from bankscope.llm import create_openai_client  # noqa: E402
from bankscope.retrieval.qdrant_retriever import DEFAULT_COLLECTION_NAME  # noqa: E402
from bankscope.sec.company_registry import load_bank_registry  # noqa: E402

DEFAULT_QRELS = ROOT / "data/evaluation/queries.jsonl"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/multi-bank-v1.json"
DEFAULT_JUDGE_MODEL = "AZURE_GPT_4o_2024_1120"
EXPECTED_QUERY_COUNT = 3
EXPECTED_EVIDENCE_GROUP_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--model", default=GPT51_CANDIDATE_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--skip-judge", action="store_true")
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


def select_comparison_queries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if row.get("question_type") == "cross_bank_coverage"]
    if len(selected) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUERY_COUNT} frozen cross-bank questions, found {len(selected)}."
        )
    groups = sum(len(row.get("required_evidence_groups") or []) for row in selected)
    if groups != EXPECTED_EVIDENCE_GROUP_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_EVIDENCE_GROUP_COUNT} evidence groups, found {groups}."
        )
    return selected


def evaluate_comparison_run(
    query: Mapping[str, Any],
    output: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    *,
    entity_tickers: Mapping[str, str],
) -> dict[str, Any]:
    retrieved_ids = {str(item.get("target_chunk_id") or "") for item in evidence}
    group_results = []
    for group in query.get("required_evidence_groups") or []:
        entity = str(group.get("entity") or "")
        targets = {str(value) for value in group.get("target_chunk_ids") or []}
        hits = sorted(targets & retrieved_ids)
        group_results.append(
            {
                "entity": entity,
                "ticker": entity_tickers.get(entity),
                "hit": bool(hits),
                "hits": hits,
            }
        )

    ownership_violations = []
    for result in output.get("bank_results") or []:
        ticker = str(result.get("ticker") or "").upper()
        for citation in result.get("citations") or []:
            citation_ticker = str(citation.get("ticker") or "").upper()
            if citation_ticker != ticker:
                ownership_violations.append(
                    {
                        "bank_ticker": ticker,
                        "citation_label": citation.get("label"),
                        "citation_ticker": citation_ticker,
                    }
                )
    supported_tickers = {
        str(result.get("ticker") or "").upper()
        for result in output.get("bank_results") or []
        if result.get("status") == "supported"
    }
    return {
        "status_supported": output.get("status") == "supported",
        "expected_tickers_supported": supported_tickers == set(output.get("tickers") or []),
        "evidence_groups": group_results,
        "retrieval_complete": all(group["hit"] for group in group_results),
        "citation_ownership_violations": ownership_violations,
    }


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.candidate_k < args.limit or args.rrf_k <= 0:
        raise ValueError("Invalid retrieval limits.")
    queries = select_comparison_queries(read_jsonl(args.qrels))
    settings = get_settings()
    client = create_openai_client(settings)
    registry = load_bank_registry(settings.bank_registry_path)
    entity_tickers = {bank.legal_name: bank.ticker for bank in registry.banks if bank.enabled}
    records = []
    with BankAnswerPipeline.from_paths(
        client=client,
        generation_model=args.model,
        temperature=settings.llm_temperature,
        chunks_path=args.chunks,
        tables_path=args.tables,
        glossary_locators_path=args.glossary_locators,
        qdrant_path=args.qdrant_path,
        qdrant_manifest_path=args.qdrant_manifest,
        collection_name=args.collection,
        bank_registry_path=settings.bank_registry_path,
    ) as pipeline:
        for query in queries:
            run = pipeline.answer(
                str(query["query"]),
                limit=args.limit,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
            )
            checks = evaluate_comparison_run(
                query, run.output, run.evidence, entity_tickers=entity_tickers
            )
            judgement = None
            if not args.skip_judge:
                cited_ids = {
                    str(citation.get("target_chunk_id") or "")
                    for citation in run.output.get("citations") or []
                }
                cited_evidence = [
                    item
                    for item in run.evidence
                    if str(item.get("target_chunk_id") or "") in cited_ids
                ]
                judgement = judge_semantic_answer(
                    question=str(query["query"]),
                    gold_answer=str(query["gold_answer"]),
                    generated_answer=str(run.output["answer"]),
                    evidence=cited_evidence,
                    client=client,
                    model=args.judge_model,
                )
            records.append(
                {
                    "query_id": query["query_id"],
                    "question": query["query"],
                    "gold_answer": query["gold_answer"],
                    "output": run.output,
                    "checks": checks,
                    "semantic_judgement": judgement,
                }
            )

    group_hits = sum(
        group["hit"] for record in records for group in record["checks"]["evidence_groups"]
    )
    ownership_violations = sum(
        len(record["checks"]["citation_ownership_violations"]) for record in records
    )
    deterministic_pass = (
        all(
            record["checks"]["status_supported"]
            and record["checks"]["expected_tickers_supported"]
            and record["checks"]["retrieval_complete"]
            for record in records
        )
        and ownership_violations == 0
    )
    semantic_pass = not args.skip_judge and all(
        judgement
        and judgement["correctness"]
        and judgement["completeness"]
        and judgement["groundedness"]
        for judgement in (record["semantic_judgement"] for record in records)
    )
    report = {
        "evaluation": "multi-bank-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "generation_model": args.model,
        "judge_model": None if args.skip_judge else args.judge_model,
        "summary": {
            "queries": len(records),
            "evidence_group_hits": group_hits,
            "evidence_groups": EXPECTED_EVIDENCE_GROUP_COUNT,
            "citation_ownership_violations": ownership_violations,
            "deterministic_pass": deterministic_pass,
            "semantic_pass": semantic_pass,
            "overall_pass": deterministic_pass and semantic_pass,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
