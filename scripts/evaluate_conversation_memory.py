"""Compare stateless and contextualized retrieval on conversational follow-ups."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bankscope.config.settings import get_settings
from bankscope.evaluation.retrieval_metrics import evaluate_ranking
from bankscope.generation.contextualizer import (
    CONTEXTUALIZATION_PROMPT_VERSION,
    contextualize_question,
)
from bankscope.generation.pipeline import SingleBankAnswerPipeline
from bankscope.io import read_jsonl
from bankscope.llm import create_openai_client
from bankscope.sec.bank_resolver import resolve_bank

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data/evaluation/conversation_memory.jsonl"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/conversation-memory-v1.json"
ISOLATION_CATEGORIES = {"topic_switch", "bank_switch", "isolation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", help="Contextualization model override.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


def validate_cases(cases: Sequence[Mapping[str, Any]]) -> None:
    if not cases:
        raise ValueError("The conversation-memory evaluation set is empty.")
    seen: set[str] = set()
    required = {
        "case_id",
        "category",
        "history",
        "current_question",
        "session_ticker",
        "expected_terms",
        "forbidden_terms",
        "relevant_target_chunk_ids",
    }
    for line_number, case in enumerate(cases, start=1):
        if missing := required - case.keys():
            raise ValueError(f"Case line {line_number} is missing fields: {sorted(missing)}.")
        case_id = str(case["case_id"]).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"Invalid or duplicate case_id: {case_id!r}.")
        history = case["history"]
        if not isinstance(history, list) or len(history) % 2:
            raise ValueError(f"Case {case_id} must have complete history turn pairs.")
        for index, message in enumerate(history):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if (
                not isinstance(message, Mapping)
                or message.get("role") != expected_role
                or not str(message.get("content") or "").strip()
            ):
                raise ValueError(f"Case {case_id} has invalid history at index {index}.")
        if not str(case["current_question"]).strip():
            raise ValueError(f"Case {case_id} has an empty current question.")
        for field in ("expected_terms", "forbidden_terms", "relevant_target_chunk_ids"):
            values = case[field]
            if not isinstance(values, list) or any(not str(value).strip() for value in values):
                raise ValueError(f"Case {case_id} has an invalid {field} list.")
        if not case["expected_terms"] or not case["relevant_target_chunk_ids"]:
            raise ValueError(f"Case {case_id} needs expected terms and retrieval judgments.")
        if case["category"] == "isolation" and history:
            raise ValueError(f"Isolation case {case_id} must not contain history.")
        seen.add(case_id)


def assess_rewrite(case: Mapping[str, Any], standalone_question: str) -> dict[str, Any]:
    normalized = standalone_question.casefold()
    missing = [term for term in case["expected_terms"] if str(term).casefold() not in normalized]
    forbidden = [term for term in case["forbidden_terms"] if str(term).casefold() in normalized]
    return {
        "passed": not missing and not forbidden,
        "missing_terms": missing,
        "forbidden_terms": forbidden,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_hits = sum(bool(row["baseline"]["hit_at_5"]) for row in rows)
    candidate_hits = sum(bool(row["contextualized"]["hit_at_5"]) for row in rows)
    rewrite_passes = sum(bool(row["rewrite_contract"]["passed"]) for row in rows)
    isolation_rows = [row for row in rows if row["category"] in ISOLATION_CATEGORIES]
    isolation_pass = all(
        row["rewrite_contract"]["passed"] and row["contextualized"]["hit_at_5"]
        for row in isolation_rows
    )
    gate_passed = (
        rewrite_passes == len(rows)
        and candidate_hits == len(rows)
        and candidate_hits >= baseline_hits
        and isolation_pass
    )
    return {
        "case_count": len(rows),
        "baseline_hit_at_5": baseline_hits,
        "contextualized_hit_at_5": candidate_hits,
        "rewrite_contract_passes": rewrite_passes,
        "isolation_case_count": len(isolation_rows),
        "isolation_passed": isolation_pass,
        "gate_passed": gate_passed,
    }


def _retrieve(
    pipeline: SingleBankAnswerPipeline, question: str, ticker: str, args: argparse.Namespace
):
    vector = pipeline.query_encoder.encode(question)
    results = pipeline.retriever.search_hybrid(
        question,
        vector,
        ticker=ticker,
        limit=args.limit,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )
    return [str(result["target_chunk_id"]) for result in results]


def main() -> None:
    args = parse_args()
    if args.limit != 5:
        raise ValueError("The v1 acceptance gate is fixed at limit=5.")
    cases = read_jsonl(args.cases)
    validate_cases(cases)
    settings = get_settings()
    model = args.model or settings.openai_model
    client = create_openai_client(settings)
    rows: list[dict[str, Any]] = []
    with SingleBankAnswerPipeline.from_paths(
        client=client,
        generation_model=model,
        temperature=settings.llm_temperature,
        bank_registry_path=settings.bank_registry_path,
    ) as pipeline:
        for case in cases:
            current_question = str(case["current_question"])
            history = case["history"]
            standalone = current_question
            if history:
                standalone = contextualize_question(
                    current_question,
                    history,
                    client=client,
                    model=model,
                    session_ticker=case["session_ticker"],
                ).standalone_question
            baseline_resolution = resolve_bank(
                current_question,
                bank_names=pipeline.bank_names,
                bank_aliases=pipeline.bank_aliases,
                session_ticker=case["session_ticker"],
            )
            candidate_resolution = resolve_bank(
                standalone,
                bank_names=pipeline.bank_names,
                bank_aliases=pipeline.bank_aliases,
                session_ticker=case["session_ticker"],
            )
            if (
                baseline_resolution.status != "resolved"
                or candidate_resolution.status != "resolved"
            ):
                raise ValueError(f"Case {case['case_id']} did not resolve to one bank.")
            relevant = [str(value) for value in case["relevant_target_chunk_ids"]]
            baseline_ids = _retrieve(
                pipeline, current_question, str(baseline_resolution.ticker), args
            )
            candidate_ids = _retrieve(pipeline, standalone, str(candidate_resolution.ticker), args)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "current_question": current_question,
                    "standalone_question": standalone,
                    "rewrite_contract": assess_rewrite(case, standalone),
                    "baseline": {
                        **evaluate_ranking(baseline_ids, relevant, k_values=(5,)),
                        "retrieved_target_chunk_ids": baseline_ids,
                    },
                    "contextualized": {
                        **evaluate_ranking(candidate_ids, relevant, k_values=(5,)),
                        "retrieved_target_chunk_ids": candidate_ids,
                    },
                }
            )
    result = {
        "evaluation": "short-term-conversation-memory-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model": model,
        "prompt_version": CONTEXTUALIZATION_PROMPT_VERSION,
        "summary": summarize(rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))
    if not result["summary"]["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
