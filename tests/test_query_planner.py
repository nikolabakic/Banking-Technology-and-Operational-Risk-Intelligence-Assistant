import pytest

from bankscope.generation.answer_generator import GenerationValidationError
from bankscope.generation.query_planner import (
    build_bank_subquestion,
    build_retrieval_queries,
    needs_contextualization,
    recent_conversation_history,
    round_robin_evidence,
    validate_contextualized_rewrite,
)


def test_only_referential_questions_use_conversation_rewriting() -> None:
    assert needs_contextualization("What about 2024?") is True
    assert needs_contextualization("How does it define operational risk?") is True
    assert needs_contextualization("A kako Citi definiše operativni rizik?") is True
    assert needs_contextualization("Summarize the JP Morgans 10-K doc") is False
    assert needs_contextualization("What is operational risk?") is False
    assert needs_contextualization("Report on JPM operational risk.") is False
    assert needs_contextualization("To compare JPM and Citi, use CET1.") is False


@pytest.mark.parametrize(
    "question",
    ["Tell me more", "More details", "Reci mi više", "Detaljnije", "Nastavi"],
)
def test_natural_continuations_use_conversation_rewriting(question: str) -> None:
    assert needs_contextualization(question) is True


def test_recent_history_keeps_only_two_newest_complete_turns() -> None:
    history = [
        {"role": "user", "content": f"question-{index}"}
        if role == "user"
        else {"role": "assistant", "content": f"answer-{index}"}
        for index in range(4)
        for role in ("user", "assistant")
    ]

    selected = recent_conversation_history(history)

    assert [message["content"] for message in selected if message["role"] == "user"] == [
        "question-2",
        "question-3",
    ]
    assert all(
        "answer-" not in message["content"]
        for message in selected
        if message["role"] == "assistant"
    )


def test_serbian_follow_up_variants_are_contextualized() -> None:
    for question in (
        "A šta je sa Citi?",
        "Šta je sa 2024?",
        "Reci mi više.",
        "Takođe me zanima kapital.",
        "Procenat",
        "Iznos",
    ):
        assert needs_contextualization(question) is True


def test_contextualization_rejects_stale_periods_and_new_numeric_facts() -> None:
    validate_contextualized_rewrite("What about 2024?", "What was JPMorgan's Tier 1 ratio in 2024?")
    validate_contextualized_rewrite(
        "And the CET1 ratio?",
        "What was JPMorgan's CET1 ratio in 2025?",
        allowed_user_context=("What did JPMorgan report in 2025?",),
    )
    with pytest.raises(GenerationValidationError, match="explicit period"):
        validate_contextualized_rewrite(
            "What about 2024?", "What was JPMorgan's ratio in 2024 and 2025?"
        )
    with pytest.raises(GenerationValidationError, match="numeric fact"):
        validate_contextualized_rewrite(
            "And on December 31, 2025?", "What was the ratio of 14.6 on December 31, 2025?"
        )
    with pytest.raises(GenerationValidationError, match="numeric fact"):
        validate_contextualized_rewrite(
            "And the CET1 ratio?",
            "What was JPMorgan's 14.6% CET1 ratio in 2025?",
            allowed_user_context=("What did JPMorgan report in 2025?",),
        )
    with pytest.raises(GenerationValidationError, match="dropped a numeric fact"):
        validate_contextualized_rewrite(
            "And on December 31, 2025?",
            "What was the ratio in 2025?",
        )


def test_comparison_is_decomposed_into_peer_free_bank_queries() -> None:
    bank_names = {"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"}
    aliases = {"JPM": ("JPMorgan", "JP Morgan"), "BAC": ("Bank of America", "BofA")}
    question = "Compare JP Morgans and Bank of Americas CET1 ratios in 2025."

    jpm = build_bank_subquestion(
        question,
        ticker="JPM",
        selected_tickers=("JPM", "BAC"),
        bank_names=bank_names,
        bank_aliases=aliases,
    )
    bac = build_bank_subquestion(
        question,
        ticker="BAC",
        selected_tickers=("JPM", "BAC"),
        bank_names=bank_names,
        bank_aliases=aliases,
    )

    assert jpm.startswith("JPMorgan Chase & Co. (JPM) Form 10-K:")
    assert bac.startswith("Bank of America Corporation (BAC) Form 10-K:")
    assert "bank of america" not in jpm.casefold()
    assert "jp morgan" not in bac.casefold()
    assert "cet1 ratios in 2025" in jpm.casefold()
    assert "cet1 ratios in 2025" in bac.casefold()

    compact = build_bank_subquestion(
        "Compare JPMorgans CET1 ratio with Bank of Americas for 2025.",
        ticker="BAC",
        selected_tickers=("JPM", "BAC"),
        bank_names=bank_names,
        bank_aliases=aliases,
    )
    assert compact == "Bank of America Corporation (BAC) Form 10-K: cet1 ratio for 2025"


def test_whole_filing_summary_uses_diverse_queries_and_balanced_merge() -> None:
    queries = build_retrieval_queries(
        "Summarize the JPMorgan 2025 10-K doc",
        ticker="JPM",
        bank_name="JPMorgan Chase & Co.",
    )

    assert len(queries) == 5
    assert all("JPMorgan Chase & Co. (JPM) Form 10-K 2025" in query for query in queries)
    focused = build_retrieval_queries(
        "Summarize JPMorgan's 2025 10-K filing cybersecurity disclosures",
        ticker="JPM",
        bank_name="JPMorgan Chase & Co.",
    )
    assert focused == (
        "Summarize JPMorgan's 2025 10-K filing cybersecurity disclosures",
        "JPMorgan Chase & Co. (JPM) Form 10-K: cybersecurity risk information security "
        "cyber attack",
    )
    groups = [
        [{"target_chunk_id": "a1"}, {"target_chunk_id": "a2"}],
        [{"target_chunk_id": "b1"}, {"target_chunk_id": "b2"}],
    ]
    assert [item["target_chunk_id"] for item in round_robin_evidence(groups, limit=3)] == [
        "a1",
        "b1",
        "a2",
    ]


def test_focused_retrieval_preserves_original_and_adds_bank_scoped_concept_queries() -> None:
    queries = build_retrieval_queries(
        "How does JPMorgan Chase describe its operational risk framework?",
        ticker="JPM",
        bank_name="JPMorgan Chase & Co.",
        original_question="And what about its operational-risk framework?",
    )

    assert queries[0] == "How does JPMorgan Chase describe its operational risk framework?"
    assert queries[1] == "And what about its operational-risk framework?"
    assert queries[2].endswith("operational risk framework operational risk management")

    cyber = build_retrieval_queries(
        "Sta Citigroup navodi o sajber riziku?",
        ticker="C",
        bank_name="Citigroup Inc.",
    )
    third_party = build_retrieval_queries(
        "A sta kazu o third-party riziku?",
        ticker="C",
        bank_name="Citigroup Inc.",
    )
    assert cyber[-1].endswith("cybersecurity risk information security cyber attack")
    assert third_party[-1].endswith(
        "third-party risk management vendor service provider outsourcing"
    )


def test_unicode_form_10k_dash_is_not_treated_as_an_added_number() -> None:
    validate_contextualized_rewrite(
        "Kako Citigroup definise sajber rizik?",
        "Kako Citigroup definise sajber rizik u Form 10‑K?",
    )
