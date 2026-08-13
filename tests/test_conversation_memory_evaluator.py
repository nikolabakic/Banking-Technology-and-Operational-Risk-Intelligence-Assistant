import pytest

from bankscope.io import read_jsonl
from scripts.evaluate_conversation_memory import assess_rewrite, summarize, validate_cases


def test_conversation_memory_fixture_is_valid() -> None:
    cases = read_jsonl("data/evaluation/conversation_memory.jsonl")
    qrels = read_jsonl("data/evaluation/queries.jsonl")

    validate_cases(cases)

    known_target_ids = {
        target_id for query in qrels for target_id in query["relevant_target_chunk_ids"]
    }
    assert all(set(case["relevant_target_chunk_ids"]).issubset(known_target_ids) for case in cases)
    assert {case["category"] for case in cases} >= {
        "entity_carryover",
        "metric_carryover",
        "year_carryover",
        "qualifier_carryover",
        "topic_switch",
        "bank_switch",
        "isolation",
    }


def test_rewrite_contract_checks_required_and_stale_terms() -> None:
    case = {"expected_terms": ["PNC", "deposits"], "forbidden_terms": ["JPMorgan"]}

    assert assess_rewrite(case, "What were PNC's deposits?")["passed"] is True
    failed = assess_rewrite(case, "What were JPMorgan's deposits?")
    assert failed["passed"] is False
    assert failed["missing_terms"] == ["PNC"]
    assert failed["forbidden_terms"] == ["JPMorgan"]


def test_summary_requires_all_candidate_and_isolation_cases_to_pass() -> None:
    rows = [
        {
            "category": "year_carryover",
            "rewrite_contract": {"passed": True},
            "baseline": {"hit_at_5": False},
            "contextualized": {"hit_at_5": True},
        },
        {
            "category": "topic_switch",
            "rewrite_contract": {"passed": True},
            "baseline": {"hit_at_5": True},
            "contextualized": {"hit_at_5": True},
        },
    ]

    result = summarize(rows)

    assert result["baseline_hit_at_5"] == 1
    assert result["contextualized_hit_at_5"] == 2
    assert result["gate_passed"] is True

    rows[1]["contextualized"]["hit_at_5"] = False
    assert summarize(rows)["gate_passed"] is False


def test_fixture_validation_rejects_partial_history() -> None:
    case = {
        "case_id": "partial",
        "category": "entity_carryover",
        "history": [{"role": "user", "content": "Question"}],
        "current_question": "Follow-up?",
        "session_ticker": "JPM",
        "expected_terms": ["JPM"],
        "forbidden_terms": [],
        "relevant_target_chunk_ids": ["chunk"],
    }

    with pytest.raises(ValueError, match="complete history"):
        validate_cases([case])
