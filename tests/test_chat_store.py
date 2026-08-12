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
        thread["id"], "Broken follow-up", "Stable public error", code="pipeline_failed"
    )
    assert store.get_thread(thread["id"])["session_ticker"] == "JPM"
    assert store.list_turns(thread["id"])[1]["error_code"] == "pipeline_failed"


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
