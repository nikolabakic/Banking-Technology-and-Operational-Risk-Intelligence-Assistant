"""Evaluate semantic conversation routing without running retrieval or answer generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bankscope.config.settings import get_settings
from bankscope.generation.conversation import CONVERSATION_PROMPT_VERSION, ConversationGraph
from bankscope.generation.query_planner import validate_contextualized_rewrite
from bankscope.io import read_jsonl
from bankscope.llm import create_langchain_chat_model
from bankscope.sec.bank_resolver import resolve_bank
from bankscope.sec.company_registry import load_bank_registry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data/evaluation/conversation_routing_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/conversation-routing-v1.json"
VALID_ACTIONS = {
    "filing_research",
    "direct_response",
    "clarification",
    "out_of_scope",
    "web_research",
    "calculator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", help="Conversation-router model override.")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional prior result; the candidate fails if any acceptance metric regresses.",
    )
    return parser.parse_args()


def validate_cases(cases: Sequence[Mapping[str, Any]]) -> None:
    if len(cases) < 40:
        raise ValueError("The routing evaluation requires at least 40 cases.")
    required = {
        "case_id",
        "question",
        "expected_action",
        "expected_tickers",
        "history",
        "session_tickers",
    }
    seen: set[str] = set()
    for line_number, case in enumerate(cases, start=1):
        if missing := required - case.keys():
            raise ValueError(f"Case line {line_number} is missing fields: {sorted(missing)}.")
        case_id = str(case["case_id"]).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"Invalid or duplicate case_id: {case_id!r}.")
        if not str(case["question"]).strip():
            raise ValueError(f"Case {case_id} has an empty question.")
        if case["expected_action"] not in VALID_ACTIONS:
            raise ValueError(f"Case {case_id} has an invalid expected action.")
        if not isinstance(case["expected_tickers"], list):
            raise ValueError(f"Case {case_id} expected_tickers must be a list.")
        if not isinstance(case["session_tickers"], list):
            raise ValueError(f"Case {case_id} session_tickers must be a list.")
        history = case["history"]
        if not isinstance(history, list) or len(history) % 2:
            raise ValueError(f"Case {case_id} must contain complete history pairs.")
        seen.add(case_id)


def _scope_preserved(
    case: Mapping[str, Any],
    search_question: str | None,
    *,
    bank_names: Mapping[str, str],
    bank_aliases: Mapping[str, Sequence[str]],
) -> tuple[bool, str | None]:
    if not search_question:
        return True, None
    user_history = [
        str(message.get("content") or "")
        for message in case["history"]
        if message.get("role") == "user"
    ]
    try:
        validate_contextualized_rewrite(
            str(case["question"]),
            search_question,
            allowed_user_context=user_history,
        )
    except Exception as error:
        return False, getattr(error, "code", type(error).__name__)
    planned = set(
        resolve_bank(
            search_question,
            bank_names=bank_names,
            bank_aliases=bank_aliases,
            session_tickers=case["session_tickers"],
        ).tickers
    )
    expected = {str(ticker) for ticker in case["expected_tickers"]}
    if expected and planned != expected:
        return False, "bank_scope_mismatch"
    return True, None


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["action_correct"]) for row in rows)
    bank_cases = [
        row
        for row in rows
        if row["expected_action"] == "filing_research" and row["expected_tickers"]
    ]
    no_retrieval_expected = [
        row
        for row in rows
        if row["expected_action"] in {"direct_response", "calculator", "out_of_scope"}
    ]
    out_of_scope_cases = [
        row for row in rows if row["expected_action"] == "out_of_scope"
    ]
    scope_ok = sum(bool(row["scope_preserved"]) for row in rows)
    accuracy = correct / len(rows)
    bank_recall = sum(bool(row["action_correct"]) for row in bank_cases) / len(bank_cases)
    unrelated_no_retrieval = (
        sum(
            row["actual_action"] not in {"filing_research", "web_research"}
            for row in no_retrieval_expected
        )
        / len(no_retrieval_expected)
        if no_retrieval_expected
        else 1.0
    )
    out_of_scope_recall = (
        sum(row["actual_action"] == "out_of_scope" for row in out_of_scope_cases)
        / len(out_of_scope_cases)
        if out_of_scope_cases
        else 1.0
    )
    gate_passed = (
        accuracy >= 0.95
        and bank_recall == 1.0
        and unrelated_no_retrieval == 1.0
        and out_of_scope_recall == 1.0
        and scope_ok == len(rows)
    )
    return {
        "case_count": len(rows),
        "correct_count": correct,
        "route_accuracy": accuracy,
        "supported_bank_filing_recall": bank_recall,
        "unrelated_no_retrieval_rate": unrelated_no_retrieval,
        "general_chat_no_retrieval_rate": unrelated_no_retrieval,
        "out_of_scope_recall": out_of_scope_recall,
        "scope_preservation_passes": scope_ok,
        "gate_passed": gate_passed,
    }


def compare_with_baseline(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    metric_names = (
        "route_accuracy",
        "supported_bank_filing_recall",
        "unrelated_no_retrieval_rate",
        "out_of_scope_recall",
        "scope_preservation_passes",
    )
    regressions = {
        name: {"baseline": baseline.get(name), "candidate": candidate.get(name)}
        for name in metric_names
        if candidate.get(name) is not None
        and baseline.get(name) is not None
        and float(candidate[name]) < float(baseline[name])
    }
    return {"passed": not regressions, "regressions": regressions}


def main() -> None:
    args = parse_args()
    cases = read_jsonl(args.cases)
    validate_cases(cases)
    settings = get_settings()
    registry = load_bank_registry(settings.bank_registry_path)
    enabled = [bank for bank in registry.banks if bank.enabled]
    bank_names = {bank.ticker: bank.legal_name for bank in enabled}
    bank_aliases = {bank.ticker: bank.aliases for bank in enabled}
    model_name = args.model or settings.openai_model
    graph = ConversationGraph(
        client=None,
        model=model_name,
        bank_names=bank_names,
        bank_aliases=bank_aliases,
        chat_model=create_langchain_chat_model(settings, model=model_name),
        backend="langgraph",
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        decision = graph.route(
            str(case["question"]),
            case["history"],
            session_tickers=case["session_tickers"],
        )
        search_question = getattr(decision.action, "search_question", None)
        scope_preserved, scope_error = _scope_preserved(
            case,
            search_question,
            bank_names=bank_names,
            bank_aliases=bank_aliases,
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "expected_action": case["expected_action"],
                "actual_action": decision.route_action,
                "expected_tickers": case["expected_tickers"],
                "action_correct": decision.route_action == case["expected_action"],
                "confidence": decision.confidence,
                "reason": decision.reason,
                "fallback": decision.fallback,
                "error_code": decision.error_code,
                "search_question": search_question,
                "scope_preserved": scope_preserved,
                "scope_error": scope_error,
            }
        )
    summary = summarize(rows)
    baseline_comparison = None
    if args.baseline:
        baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_summary = baseline_payload.get("summary") or baseline_payload
        baseline_comparison = compare_with_baseline(summary, baseline_summary)
        summary["gate_passed"] = bool(
            summary["gate_passed"] and baseline_comparison["passed"]
        )
    result = {
        "evaluation": "conversation-routing-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model": model_name,
        "prompt_version": CONVERSATION_PROMPT_VERSION,
        "baseline": str(args.baseline) if args.baseline else None,
        "baseline_comparison": baseline_comparison,
        "summary": summary,
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
