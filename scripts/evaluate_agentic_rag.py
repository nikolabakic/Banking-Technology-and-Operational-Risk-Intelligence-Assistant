"""Compare baseline and bounded agentic RAG on the frozen challenge set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bankscope.config.settings import get_settings  # noqa: E402
from bankscope.generation import GPT51_CANDIDATE_MODEL, BankAnswerPipeline  # noqa: E402
from bankscope.generation.answer_generator import GenerationValidationError  # noqa: E402
from bankscope.io import read_jsonl  # noqa: E402
from bankscope.llm import create_openai_client  # noqa: E402
from bankscope.retrieval.mixed_retriever import (  # noqa: E402
    BankSearchResult,
    interleave_bank_results,
)

DEFAULT_CHALLENGE = ROOT / "data/evaluation/agentic_rag_challenge_v1.jsonl"
DEFAULT_QRELS = ROOT / "data/evaluation/queries.jsonl"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/agentic-rag-v1.json"
EXPECTED_COUNTS = {
    "rewrite_search": 4,
    "expand_context": 3,
    "multi_bank": 3,
    "sufficient_initial": 1,
    "unsupported": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        default=GPT51_CANDIDATE_MODEL,
        help=f"Gateway model (default: {GPT51_CANDIDATE_MODEL}).",
    )
    parser.add_argument(
        "--prerequisite-gates-passed",
        action="store_true",
        help="Confirm the frozen retrieval, generation, memory, and comparison gates passed.",
    )
    return parser.parse_args()


def validate_challenge(rows: list[dict[str, Any]], qrels: dict[str, dict[str, Any]]) -> None:
    if len(rows) != 12:
        raise ValueError("Agentic challenge set must contain exactly 12 questions.")
    counts = Counter(str(row.get("category")) for row in rows)
    if counts != Counter(EXPECTED_COUNTS):
        raise ValueError(f"Invalid challenge category distribution: {dict(counts)}")
    seen: set[str] = set()
    for row in rows:
        query_id = str(row.get("query_id") or "")
        tickers = [str(value).upper() for value in row.get("tickers") or []]
        if not query_id or query_id in seen or not str(row.get("query") or "").strip():
            raise ValueError(f"Invalid or duplicate challenge query: {query_id!r}")
        if not tickers or len(tickers) != len(set(tickers)):
            raise ValueError(f"Invalid ticker scope for {query_id}.")
        for source_id in row.get("source_query_ids") or []:
            source = qrels.get(str(source_id))
            if source is None:
                raise ValueError(f"Unknown source qrel {source_id} for {query_id}.")
            if str(source.get("ticker") or "").upper() not in tickers:
                raise ValueError(f"Source qrel {source_id} crosses the bank scope for {query_id}.")
        seen.add(query_id)


def relevant_ids(row: dict[str, Any], qrels: dict[str, dict[str, Any]]) -> set[str]:
    return {
        str(target_id)
        for source_id in row.get("source_query_ids") or []
        for target_id in qrels[str(source_id)].get("relevant_target_chunk_ids") or []
    }


def run_mode(
    pipeline: BankAnswerPipeline,
    row: dict[str, Any],
    qrels: dict[str, dict[str, Any]],
    *,
    enabled: bool,
) -> dict[str, Any]:
    pipeline.agentic_rag_enabled = enabled
    tickers = list(row["tickers"])
    mode = "agentic" if enabled else "baseline"
    print(f"[{row['query_id']}] starting {mode}", flush=True)
    try:
        retrieval_runs = []
        searches = []
        for ticker in tickers:
            retrieval = pipeline.retrieve_evidence(str(row["query"]), ticker=ticker)
            retrieval_runs.append(retrieval)
            searches.append(
                BankSearchResult(
                    ticker=ticker,
                    results=retrieval.evidence,
                    latency_ms=retrieval.retrieval_latency_ms,
                )
            )
        evidence = (
            searches[0].results
            if len(searches) == 1
            else interleave_bank_results(searches, limit=10)
        )
        plans = [
            dict(plan)
            for retrieval in retrieval_runs
            for plan in retrieval.agentic_plans
        ]
    except Exception as error:
        print(
            f"[{row['query_id']}] {mode} retrieval failed: {type(error).__name__}",
            flush=True,
        )
        return {
            "executed": False,
            "failed_stage": "retrieval",
            "error_code": "retrieval_failed",
            "exception_type": type(error).__name__,
            "route": None,
            "plans": [],
            "actions": [],
            "hit_at_5": None,
            "hit_at_10": None,
            "status": "error",
            "request_count": None,
            "latency_ms": None,
            "quality_gate": None,
            "retrieved_target_chunk_ids": [],
            "end_to_end": None,
        }

    # Capture end-to-end behavior separately; generation failures never erase retrieval metrics.
    try:
        answer_run = pipeline.answer(
            str(row["query"]),
            ticker=tickers[0] if len(tickers) == 1 else None,
            tickers=tickers if len(tickers) > 1 else (),
        )
        end_to_end = {
            "executed": True,
            "status": answer_run.output.get("status"),
            "citation_count": len(answer_run.output.get("citations") or []),
            "error_code": None,
        }
    except GenerationValidationError as error:
        end_to_end = {
            "executed": False,
            "status": "error",
            "citation_count": 0,
            "error_code": error.code,
        }
    except Exception as error:
        end_to_end = {
            "executed": False,
            "status": "error",
            "citation_count": 0,
            "error_code": "pipeline_failed",
            "exception_type": type(error).__name__,
        }

    retrieved = [str(item.get("target_chunk_id") or "") for item in evidence]
    relevant = relevant_ids(row, qrels)
    diagnostics = {
        "route": "domain_rag",
        "bank_plans": plans,
        "model_request_count": sum(int(plan.get("model_request_count") or 0) for plan in plans),
        "quality_gate": {
            "passed": all(
                bool((retrieval.diagnostics.get("quality_gate") or {}).get("passed"))
                for retrieval in retrieval_runs
            )
        },
    }
    return {
        "executed": True,
        "failed_stage": None,
        "error_code": None,
        "route": diagnostics.get("route"),
        "plans": diagnostics.get("bank_plans") or [],
        "actions": [
            step.get("action")
            for plan in diagnostics.get("bank_plans") or []
            for step in plan.get("steps") or []
            if step.get("action") not in {"verify_evidence", "invalid_step", "invalid_verdict"}
        ],
        "hit_at_5": bool(relevant.intersection(retrieved[:5])) if relevant else None,
        "hit_at_10": bool(relevant.intersection(retrieved[:10])) if relevant else None,
        "status": (
            "unsupported"
            if retrieval_runs and all(run.status == "unsupported" for run in retrieval_runs)
            else "sufficient"
        ),
        "request_count": diagnostics.get("model_request_count"),
        "latency_ms": {
            "embedding": sum(run.embedding_latency_ms for run in retrieval_runs),
            "retrieval": sum(run.retrieval_latency_ms for run in retrieval_runs),
            "orchestration": sum(run.orchestration_latency_ms for run in retrieval_runs),
        },
        "quality_gate": diagnostics.get("quality_gate"),
        "retrieved_target_chunk_ids": retrieved,
        "end_to_end": end_to_end,
    }


def main() -> None:
    args = parse_args()
    challenge = read_jsonl(args.challenge)
    qrels = {str(row["query_id"]): row for row in read_jsonl(args.qrels)}
    validate_challenge(challenge, qrels)
    settings = get_settings()
    pipeline = BankAnswerPipeline.from_paths(
        client=create_openai_client(settings),
        generation_model=args.model,
        temperature=settings.llm_temperature,
        agentic_rag_enabled=False,
    )
    rows = []
    try:
        for challenge_row in challenge:
            rows.append(
                {
                    "query_id": challenge_row["query_id"],
                    "category": challenge_row["category"],
                    "expected_action": challenge_row["expected_action"],
                    "baseline": run_mode(pipeline, challenge_row, qrels, enabled=False),
                    "agentic": run_mode(pipeline, challenge_row, qrels, enabled=True),
                }
            )
    finally:
        pipeline.close()

    answerable = [row for row in rows if row["category"] != "unsupported"]
    baseline_misses = [row for row in answerable if row["baseline"]["hit_at_5"] is False]
    recovered = [row for row in baseline_misses if row["agentic"]["hit_at_5"] is True]
    sufficient = next(row for row in rows if row["category"] == "sufficient_initial")
    unsupported = next(row for row in rows if row["category"] == "unsupported")
    checks = {
        "prerequisite_frozen_gates": args.prerequisite_gates_passed,
        "all_runs_executed": all(
            row[mode]["executed"] for row in rows for mode in ("baseline", "agentic")
        ),
        "no_baseline_hit_at_5_lost": all(
            not row["baseline"]["hit_at_5"] or row["agentic"]["hit_at_5"] for row in answerable
        ),
        "at_least_three_baseline_misses_recovered": len(recovered) >= 3,
        "all_runtime_contracts_pass": all(
            bool((row["agentic"].get("quality_gate") or {}).get("passed")) for row in rows
        ),
        "sufficient_retrieval_did_not_expand": all(
            int(plan.get("tool_action_count") or 0) == 0
            for plan in sufficient["agentic"]["plans"]
        ),
        "unsupported_abstained_without_citations": (
            unsupported["agentic"]["status"] == "unsupported"
            and int(
                (unsupported["agentic"].get("end_to_end") or {}).get("citation_count") or 0
            )
            == 0
        ),
    }
    report = {
        "evaluation": "agentic-rag-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "challenge_path": str(args.challenge),
        "qrels_path": str(args.qrels),
        "feature_default": False,
        "quality_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "recovered_baseline_misses": [row["query_id"] for row in recovered],
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["quality_gate"], ensure_ascii=False, indent=2))
    if not report["quality_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
