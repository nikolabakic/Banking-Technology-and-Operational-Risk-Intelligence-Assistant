import sqlite3

import pytest

from bankscope.chat import ChatStore


def answer(ticker: str = "JPM") -> dict:
    return {
        "question": "Question",
        "ticker": ticker,
        "status": "supported",
        "answer_type": "narrative",
        "answer": "Answer [E1]",
        "reason": "Supported",
        "citations": [{"label": "E1", "target_chunk_id": "chunk-1"}],
    }


def retryable_answer() -> dict:
    output = answer()
    output.update(
        {
            "dialog_act": "retryable_error",
            "answer": "Research paused safely.",
            "citations": [],
            "status": "unsupported",
        }
    )
    return output


def test_chat_store_persists_threads_turns_and_citations(tmp_path) -> None:
    path = tmp_path / "chat.db"
    store = ChatStore(path)
    store.initialize()
    thread = store.create_thread()
    turn = store.append_answer_turn(
        thread["id"], "  A   persistent question?  ", answer(), corpus_hash="hash-1"
    )

    reopened = ChatStore(path)
    reopened.initialize()
    persisted = reopened.get_thread(thread["id"])
    assert persisted["title"] == "A persistent question?"
    assert persisted["session_ticker"] == "JPM"
    assert persisted["session_tickers"] == ["JPM"]
    assert reopened.list_turns(thread["id"])[0]["response"] == turn["response"]

    citation_id = turn["response"]["citations"][0]["citation_id"]
    citation = reopened.get_citation(citation_id)
    assert citation["target_chunk_id"] == "chunk-1"
    assert citation["corpus_hash"] == "hash-1"


def test_chat_store_persists_web_citation_metadata_without_a_corpus_target(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    output = answer("")
    output["citations"] = [
        {
            "kind": "web",
            "label": "E1",
            "title": "Official risk update",
            "snippet": "The regulator published an updated risk notice.",
            "source_url": "https://example.com/risk-update",
        }
    ]

    turn = store.append_answer_turn(
        thread["id"], "What changed?", output, corpus_hash="corpus-at-answer-time"
    )
    citation_id = turn["response"]["citations"][0]["citation_id"]
    citation = store.get_citation(citation_id)

    assert citation["target_chunk_id"] == ""
    assert citation["corpus_hash"] == "corpus-at-answer-time"
    assert citation["metadata"] == {
        "kind": "web",
        "label": "E1",
        "title": "Official risk update",
        "snippet": "The regulator published an updated risk notice.",
        "source_url": "https://example.com/risk-update",
        "citation_id": citation_id,
    }


def test_chat_store_errors_do_not_replace_session_bank(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    store.append_answer_turn(thread["id"], "JPM question", answer(), corpus_hash="hash-1")
    store.append_error_turn(
        thread["id"],
        "Broken follow-up",
        "Stable public error",
        code="pipeline_failed",
        diagnostics={"failed_stage": "retrieving", "error_code": "pipeline_failed"},
    )
    assert store.get_thread(thread["id"])["session_ticker"] == "JPM"
    assert store.list_turns(thread["id"])[1]["error_code"] == "pipeline_failed"
    assert store.list_turns(thread["id"])[1]["diagnostics"]["failed_stage"] == "retrieving"
    history = store.conversation_history(thread["id"])
    assert history[0] == {"role": "user", "content": "JPM question"}
    assert history[1] == {"role": "assistant", "content": "Answer [E1]"}


def test_conversation_history_is_thread_scoped_and_preserves_raw_pairs(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    first = store.create_thread()
    second = store.create_thread()
    for index in range(6):
        output = answer()
        output["answer"] = f"Answer {index} [E1]"
        store.append_answer_turn(first["id"], f"Question {index}", output, corpus_hash="hash-1")
    store.append_answer_turn(second["id"], "Other thread", answer("BAC"), corpus_hash="hash-1")

    history = store.conversation_history(first["id"])

    assert [item["content"] for item in history if item["role"] == "user"] == [
        "Question 0",
        "Question 1",
        "Question 2",
        "Question 3",
        "Question 4",
        "Question 5",
    ]
    assert [item["content"] for item in history if item["role"] == "assistant"] == [
        f"Answer {index} [E1]" for index in range(6)
    ]
    assert all(item["content"] != "Other thread" for item in history)

    context = store.conversation_context(first["id"], max_tokens=1)
    assert context["needs_compaction"] is False
    assert len(context["messages"]) == 12


def test_retryable_answer_turns_remain_visible_but_do_not_poison_model_context(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    store.append_answer_turn(thread["id"], "JPM topic", answer(), corpus_hash="hash-1")
    recovery = retryable_answer()
    store.append_answer_turn(thread["id"], "Tell me more", recovery, corpus_hash="hash-1")

    history = store.conversation_history(thread["id"])
    assert [item["content"] for item in history if item["role"] == "user"] == ["JPM topic"]
    assert all(item["content"] != "Research paused safely." for item in history)
    context = store.conversation_context(thread["id"])
    assert context["unresolved_requests"] == ["Tell me more"]
    turns = store.list_turns(thread["id"])
    assert turns[-1]["response"]["answer"] == "Research paused safely."

    store.append_answer_turn(thread["id"], "Retry it", answer(), corpus_hash="hash-1")
    assert store.conversation_context(thread["id"])["unresolved_requests"] == []


def test_two_failed_natural_retries_keep_the_original_unresolved_request(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    original = "How does Ally Financial define operational risk in its 2025 Form 10-K?"

    for question in (original, "Try again.", "Retry it."):
        store.append_answer_turn(
            thread["id"], question, retryable_answer(), corpus_hash="hash-1"
        )

    assert store.conversation_context(thread["id"])["unresolved_requests"] == [original]


def test_identical_retry_button_requests_are_deduplicated_and_bounded(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()

    for question in ("request 0", "request 1", "request 2", "request 3", "request 3"):
        store.append_answer_turn(
            thread["id"], question, retryable_answer(), corpus_hash="hash-1"
        )
    assert store.conversation_context(thread["id"])["unresolved_requests"] == [
        "request 1",
        "request 2",
        "request 3",
    ]

    for question in ("long-a-" + "x" * 10_000, "long-b-" + "x" * 10_000):
        store.append_answer_turn(
            thread["id"], question, retryable_answer(), corpus_hash="hash-1"
        )
    requests = store.conversation_context(thread["id"])["unresolved_requests"]
    assert len(requests) <= 3
    assert all(len(request) <= 4_000 for request in requests)
    assert sum(len(request) for request in requests) <= 8_000
    assert requests[0].startswith("long-a-")
    assert requests[1].startswith("long-b-")


@pytest.mark.parametrize("dialog_act", ["acknowledgement", "clarification"])
def test_non_resolving_turn_between_failure_and_retry_preserves_target(
    tmp_path, dialog_act
) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    original = "How does Ally Financial define operational risk in its 2025 Form 10-K?"
    store.append_answer_turn(
        thread["id"], original, retryable_answer(), corpus_hash="hash-1"
    )
    interlude = answer("")
    interlude.update(
        {
            "dialog_act": dialog_act,
            "answer": "Which detail do you mean?" if dialog_act == "clarification" else "Okay.",
            "citations": [],
        }
    )
    store.append_answer_turn(thread["id"], "Okay", interlude, corpus_hash="hash-1")
    store.append_answer_turn(
        thread["id"], "Try again.", retryable_answer(), corpus_hash="hash-1"
    )

    assert store.conversation_context(thread["id"])["unresolved_requests"] == [original]


def test_out_of_scope_turn_remains_in_thread_scoped_memory(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    store.append_answer_turn(thread["id"], "JPM topic", answer(), corpus_hash="hash-1")
    declined = answer("")
    declined.update(
        {
            "dialog_act": "out_of_scope",
            "ticker": None,
            "tickers": [],
            "status": "unsupported",
            "answer": "Scope boundary.",
            "citations": [],
        }
    )
    store.append_answer_turn(thread["id"], "Apple pie recipe", declined, corpus_hash="hash-1")

    assert [item["content"] for item in store.conversation_history(thread["id"])] == [
        "JPM topic",
        "Answer [E1]",
        "Apple pie recipe",
        "Scope boundary.",
    ]


def test_acknowledgement_remains_available_to_the_model(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    store.append_answer_turn(thread["id"], "JPM topic", answer(), corpus_hash="hash-1")
    acknowledgement = answer("")
    acknowledgement.update(
        {
            "dialog_act": "acknowledgement",
            "ticker": None,
            "answer": "You're welcome.",
            "citations": [],
        }
    )
    store.append_answer_turn(thread["id"], "Thanks", acknowledgement, corpus_hash="hash-1")

    history = store.conversation_history(thread["id"])
    assert [item["content"] for item in history if item["role"] == "user"] == [
        "JPM topic",
        "Thanks",
    ]

    context = store.conversation_context(thread["id"])
    assert context["previous_answer"]["question"] == "JPM topic"
    assert context["previous_answer"]["answer"] == "Answer [E1]"


@pytest.mark.parametrize("dialog_act", ["general_explanation", "calculation", "web_answer"])
def test_latest_substantive_non_filing_answer_can_be_transformed(tmp_path, dialog_act) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    store.append_answer_turn(thread["id"], "JPM topic", answer(), corpus_hash="hash-1")
    latest = answer("")
    latest.update(
        {
            "dialog_act": dialog_act,
            "answer": "Latest substantive answer.",
            "citations": [],
        }
    )
    store.append_answer_turn(thread["id"], "A newer request", latest, corpus_hash="hash-1")

    previous = store.conversation_context(thread["id"])["previous_answer"]

    assert previous["question"] == "A newer request"
    assert previous["answer"] == "Latest substantive answer."


def test_conversation_context_compacts_old_pairs_and_keeps_six_raw_turns(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    for index in range(8):
        output = answer()
        output["answer"] = f"Grounded answer {index} [E1]"
        store.append_answer_turn(
            thread["id"], f"Question {index}", output, corpus_hash="hash-1"
        )

    pending = store.conversation_context(thread["id"], max_tokens=1)
    assert pending["needs_compaction"] is True
    assert [item["content"] for item in pending["compaction_messages"]] == [
        "Question 0",
        "Grounded answer 0 [E1]",
        "Question 1",
        "Grounded answer 1 [E1]",
    ]
    assert len(pending["messages"]) == 12

    store.save_conversation_summary(
        thread["id"],
        "The user prefers concise answers and is discussing JPMorgan.",
        through_sequence=pending["compaction_through_sequence"],
        prompt_version="conversation-summary-tool-v1",
    )
    compacted = store.conversation_context(thread["id"], max_tokens=12_000)
    assert compacted["summary"].startswith("The user prefers concise answers")
    assert compacted["summary_through_sequence"] == 4
    assert len(compacted["messages"]) == 12
    assert compacted["messages"][0]["content"] == "Question 2"


def test_thread_payload_does_not_expose_internal_conversation_summary(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()

    assert "conversation_summary" not in thread
    assert "summary_through_sequence" not in thread


def test_chat_store_cascade_deletes_messages_and_citations(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    store.append_answer_turn(thread["id"], "Question", answer(), corpus_hash="hash-1")
    store.delete_thread(thread["id"])
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM message_citations").fetchone()[0] == 0


def test_chat_store_rejects_newer_schema(tmp_path) -> None:
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="newer"):
        ChatStore(path).initialize()


def test_chat_store_migrates_v1_session_and_persists_comparison_scope(tmp_path) -> None:
    path = tmp_path / "v1.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE chat_threads (
               id TEXT PRIMARY KEY, title TEXT NOT NULL, session_ticker TEXT,
               created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
        )
        connection.execute("INSERT INTO chat_threads VALUES ('old', 'Old', 'JPM', 'now', 'now')")
        connection.execute("PRAGMA user_version = 1")
    store = ChatStore(path)
    store.initialize()
    assert store.get_thread("old")["session_tickers"] == ["JPM"]

    store = ChatStore(tmp_path / "comparison.db")
    store.initialize()
    thread = store.create_thread()
    comparison = answer()
    comparison.update(
        {
            "ticker": None,
            "tickers": ["BAC", "C"],
            "mode": "comparison",
            "bank_results": [
                {"ticker": "BAC", "citations": [comparison["citations"][0]]},
                {"ticker": "C", "citations": []},
            ],
        }
    )
    turn = store.append_answer_turn(thread["id"], "Compare", comparison, corpus_hash="hash-1")
    persisted = store.get_thread(thread["id"])
    assert persisted["session_ticker"] is None
    assert persisted["session_tickers"] == ["BAC", "C"]
    assert turn["response"]["bank_results"][0]["citations"][0]["citation_id"]
