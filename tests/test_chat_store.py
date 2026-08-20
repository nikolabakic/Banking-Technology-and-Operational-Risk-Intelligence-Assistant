import json
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
    assert json.loads(history[1]["content"])["tickers"] == ["JPM"]
    assert "Answer [E1]" not in history[1]["content"]


def test_conversation_history_is_thread_scoped_and_bounded_by_complete_pairs(tmp_path) -> None:
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
        "Question 2",
        "Question 3",
        "Question 4",
        "Question 5",
    ]
    assert all("Answer" not in item["content"] for item in history if item["role"] == "assistant")
    assert all(item["content"] != "Other thread" for item in history)
    one_pair = store.conversation_history(first["id"], max_turns=1)
    assert one_pair[0] == {"role": "user", "content": "Question 5"}
    assert store.conversation_history(first["id"], max_chars=1) == []
    assert store.conversation_history(first["id"], max_tokens=1) == []


def test_retryable_answer_turns_do_not_pollute_conversation_history(tmp_path) -> None:
    store = ChatStore(tmp_path / "chat.db")
    store.initialize()
    thread = store.create_thread()
    store.append_answer_turn(thread["id"], "JPM topic", answer(), corpus_hash="hash-1")
    recovery = answer()
    recovery.update(
        {
            "dialog_act": "retryable_error",
            "answer": "Research paused safely.",
            "citations": [],
            "status": "unsupported",
        }
    )
    store.append_answer_turn(thread["id"], "Tell me more", recovery, corpus_hash="hash-1")

    history = store.conversation_history(thread["id"])
    assert [item["content"] for item in history if item["role"] == "user"] == ["JPM topic"]
    assert "Answer [E1]" not in history[1]["content"]


def test_out_of_scope_turn_resets_research_memory_without_entering_it(tmp_path) -> None:
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

    assert store.conversation_history(thread["id"]) == []


def test_acknowledgement_does_not_displace_latest_research_topic(tmp_path) -> None:
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
    assert [item["content"] for item in history if item["role"] == "user"] == ["JPM topic"]


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
