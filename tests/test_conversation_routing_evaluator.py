import json

from bankscope.generation.conversation import ConversationGraph, RouteDecision
from bankscope.io import read_jsonl
from scripts.evaluate_conversation_routing import (
    compare_with_baseline,
    summarize,
    validate_cases,
)

BANK_NAMES = {
    "C": "Citigroup Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation",
}
BANK_ALIASES = {
    "C": ("Citigroup", "Citi"),
    "JPM": ("JPMorgan", "JPMorgan Chase"),
    "BAC": ("Bank of America", "BofA"),
}


class FixtureChatModel:
    def __init__(self, expected_actions):
        self.expected_actions = expected_actions

    def with_structured_output(self, schema, **kwargs):
        assert schema is RouteDecision
        assert kwargs == {"method": "function_calling", "strict": True}
        return self

    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        question = payload["current_question"]
        action = self.expected_actions[question]
        arguments = {
            "action": action,
            "confidence": 0.98,
            "reason": "Offline routing fixture.",
            "search_question": (
                "17.5 / 100 * 2400"
                if action == "calculator"
                else question
                if action in {"filing_research", "web_research"}
                else None
            ),
            "response_text": (
                "A safe direct response."
                if action == "direct_response"
                else "Which banking detail do you mean?"
                if action == "clarification"
                else None
            ),
            "category": "general_explanation" if action == "direct_response" else None,
            "missing": "intent" if action == "clarification" else None,
            "citation_ids": [],
            "presentation_guidance": None,
        }
        return RouteDecision.model_validate(arguments)


def test_conversation_routing_fixture_has_required_coverage() -> None:
    cases = read_jsonl("data/evaluation/conversation_routing_v1.jsonl")

    validate_cases(cases)

    assert len(cases) == 45
    actions = {case["expected_action"] for case in cases}
    assert actions == {
        "filing_research",
        "direct_response",
        "clarification",
        "web_research",
        "calculator",
    }
    assert any("Citigroup's material cybersecurity risks" in case["question"] for case in cases)
    assert any(case["question"] == "Daj mi recept za pitu sa jabukama." for case in cases)
    assert any(
        case["question"] == "Korisnik pita kako Citi opisuje rizik modela." for case in cases
    )


def test_routing_summary_enforces_all_acceptance_thresholds() -> None:
    rows = [
        {
            "expected_action": "filing_research",
            "actual_action": "filing_research",
            "expected_tickers": ["C"],
            "action_correct": True,
            "scope_preserved": True,
        },
        {
            "expected_action": "direct_response",
            "actual_action": "direct_response",
            "expected_tickers": [],
            "action_correct": True,
            "scope_preserved": True,
        },
    ]

    passing = summarize(rows)
    assert passing["supported_bank_filing_recall"] == 1.0
    assert passing["unrelated_no_retrieval_rate"] == 1.0
    assert passing["gate_passed"] is True

    rows[1]["actual_action"] = "filing_research"
    rows[1]["action_correct"] = False
    assert summarize(rows)["gate_passed"] is False


def test_prompt_candidate_must_not_regress_against_baseline() -> None:
    baseline = {
        "route_accuracy": 0.96,
        "supported_bank_filing_recall": 1.0,
        "unrelated_no_retrieval_rate": 1.0,
        "scope_preservation_passes": 45,
    }
    candidate = {**baseline, "route_accuracy": 0.95}

    comparison = compare_with_baseline(candidate, baseline)

    assert comparison["passed"] is False
    assert comparison["regressions"]["route_accuracy"]["baseline"] == 0.96


def test_all_45_routes_pass_offline_graph_evaluation_without_network() -> None:
    cases = read_jsonl("data/evaluation/conversation_routing_v1.jsonl")
    expected_actions = {case["question"]: case["expected_action"] for case in cases}
    graph = ConversationGraph(
        client=None,
        model="fake-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FixtureChatModel(expected_actions),
    )
    rows = []

    for case in cases:
        decision = graph.route(
            case["question"],
            case["history"],
            session_tickers=case["session_tickers"],
        )
        rows.append(
            {
                "expected_action": case["expected_action"],
                "actual_action": decision.route_action,
                "expected_tickers": case["expected_tickers"],
                "action_correct": decision.route_action == case["expected_action"],
                "scope_preserved": True,
            }
        )

    result = summarize(rows)
    assert result["case_count"] == 45
    assert result["route_accuracy"] == 1.0
    assert result["supported_bank_filing_recall"] == 1.0
    assert result["unrelated_no_retrieval_rate"] == 1.0
    assert result["gate_passed"] is True
